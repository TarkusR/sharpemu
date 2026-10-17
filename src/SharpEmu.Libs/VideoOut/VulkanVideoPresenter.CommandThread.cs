// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.VideoOut;

using System.Diagnostics;
using SharpEmu.HLE;
using SharpEmu.Libs.Agc;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Gpu.Scheduling;

internal static partial class VulkanVideoPresenter
{
    // The command stream is interpreted on its own thread, which hands its work to the render thread
    // in stream order. Draws, dispatches, barriers, completions and the interpreter's own stores are
    // queued; a callback that returns a result waits for the render thread to run everything before it.
    private sealed class CommandThread : AgcExports.IDeferredWorkSink
    {
        // The thread runs this far ahead of the render thread at most: enough to keep it busy, and
        // no more, since a GPU round trip (a readback, a label wait) waits for everything queued.
        // Measured in Astro Bot: 4096 items halved the frame rate of a crowd scene, 16 starved the
        // render thread, 32 to 64 kept it busy.
        private const int MaxOutstanding = 128;
        private const int ShadowSweepCount = 64;

        private enum Kind : byte
        {
            Deferred,
            Call,
            Write,
        }

        private sealed class Item
        {
            public Kind Kind;
            public bool IsDraw;
            public AgcExports.DeferredWork? Work;
            public Action? Action;
            public ulong Address;
            public byte[]? Data;
            // The guest range a queued fill or copy writes; zero for other items.
            public ulong RangeSize;
            public long Sequence;
            public bool Sync;
            public volatile bool Done;
            public Exception? Failure;
        }

        private sealed class Shadow
        {
            public ulong Address;
            public int Length;
            public long Sequence;
            public KnownGpuWrite Known = null!;
        }

        private readonly CommandStreamQueue _queue;
        private readonly ICpuMemory _memory;
        private readonly Action<AgcExports.DeferredWork> _runDeferred;
        private readonly Func<SliceResult> _runSlice;
        private readonly Thread _thread;
        private readonly object _lock = new();
        private List<Item> _incoming = new();
        private List<Item> _draining = new();
        private int _drainIndex;
        private int _outstanding;
        private long _enqueued;
        private long _executed;
        private long _waitingForSequence = long.MaxValue;
        // What the producer sleeps on: room in the queue, or its synchronous item.
        private volatile bool _waitingForSpace;
        private volatile bool _waitingForSync;
        private volatile bool _consumerIdle;
        private volatile bool _stopping;
        private volatile Exception? _failure;

        // The interpreter's stores the render thread has not applied yet, oldest first; reads on
        // this thread see them over guest memory.
        private readonly object _writeLock = new();
        private readonly Queue<Item> _pendingWrites = new();
        private volatile int _pendingWriteCount;

        // The destinations of queued fills and copies; a read on this thread that overlaps one waits
        // for the render thread to run it.
        private readonly List<Item> _pendingRanges = new();
        private volatile int _pendingRangeCount;

        private readonly List<Shadow> _shadows = new();
        private readonly Stack<Shadow> _spareShadows = new();
        private long _shadowGeneration;

        private static readonly long RetryTicks = CommandStreamQueue.AllBlockedRetryMilliseconds * Stopwatch.Frequency / 1000L;

        public CommandThread(CommandStreamQueue queue, ICpuMemory memory, Action<AgcExports.DeferredWork> runDeferred, Func<SliceResult> runSlice)
        {
            _queue = queue;
            _memory = memory;
            _runDeferred = runDeferred;
            _runSlice = runSlice;
            _thread = new Thread(Run) { IsBackground = true, Name = "SharpEmu command stream" };
        }

        public void Start() => _thread.Start();

        public bool IsProducerThread => Environment.CurrentManagedThreadId == _thread.ManagedThreadId;

        public bool HasWork => Volatile.Read(ref _outstanding) != 0 || _failure is not null;

        public long NextSequence => Volatile.Read(ref _enqueued) + 1;

        public long Executed => Volatile.Read(ref _executed);

        public long ShadowGeneration => Volatile.Read(ref _shadowGeneration);

        public bool HasPendingStores => _pendingWriteCount != 0;

        // Read by the render loop before it waits, so a queued item wakes it.
        public bool ConsumerIdle
        {
            set => _consumerIdle = value;
        }

        public bool IsAlive => _thread.IsAlive;

        // The producer side, on the command stream thread.

        public void EnqueueDeferred(AgcExports.DeferredWork work, bool isDraw) =>
            Enqueue(new Item { Kind = Kind.Deferred, Work = work, IsDraw = isDraw });

        public void Post(Action action) => Enqueue(new Item { Kind = Kind.Call, Action = action });

        public void Run(Action action)
        {
            var item = new Item { Kind = Kind.Call, Action = action, Sync = true };
            Enqueue(item);
            WaitFor(item);
        }

        public T Run<T>(Func<T> function)
        {
            T result = default!;
            Run(() => { result = function(); });
            return result;
        }

        public void PostWrite(ulong address, ReadOnlySpan<byte> data)
        {
            var item = new Item { Kind = Kind.Write, Address = address, Data = data.ToArray() };
            lock (_writeLock)
            {
                _pendingWrites.Enqueue(item);
                _pendingWriteCount = _pendingWrites.Count;
            }

            Enqueue(item);
        }

        // A fill or copy the render thread runs in order; reads of its destination wait for it.
        public void PostRange(Action action, ulong address, ulong size)
        {
            var item = new Item { Kind = Kind.Call, Action = action, Address = address, RangeSize = size };
            if (size != 0)
            {
                lock (_writeLock)
                {
                    _pendingRanges.Add(item);
                    _pendingRangeCount = _pendingRanges.Count;
                }
            }

            Enqueue(item);
        }

        // Runs every queued item when one of them is a fill or copy into this range.
        public void WaitForPendingRange(ulong address, ulong size)
        {
            if (_pendingRangeCount != 0 && OverlapsPendingRange(address, size))
            {
                Run(static () => { });
            }
        }

        private bool OverlapsPendingRange(ulong address, ulong size)
        {
            lock (_writeLock)
            {
                foreach (var pending in _pendingRanges)
                {
                    if (address < pending.Address + pending.RangeSize && pending.Address < address + size)
                    {
                        return true;
                    }
                }

                return false;
            }
        }

        public void NoteKnownWrite(KnownGpuWrite known, long sequence)
        {
            if (_shadows.Count >= ShadowSweepCount)
            {
                SweepShadows();
            }

            var shadow = _spareShadows.TryPop(out var spare) ? spare : new Shadow();
            shadow.Address = known.Address;
            shadow.Length = known.ByteCount;
            shadow.Sequence = sequence;
            shadow.Known = known;
            _shadows.Add(shadow);
            Interlocked.Increment(ref _shadowGeneration);
        }

        private void SweepShadows()
        {
            var executed = Volatile.Read(ref _executed);
            var kept = 0;
            for (var index = 0; index < _shadows.Count; index++)
            {
                var shadow = _shadows[index];
                if (shadow.Sequence <= executed)
                {
                    shadow.Known = null!;
                    _spareShadows.Push(shadow);
                    continue;
                }

                _shadows[kept++] = shadow;
            }

            _shadows.RemoveRange(kept, _shadows.Count - kept);
        }

        public bool HasShadow(ulong address, ulong size)
        {
            var executed = Volatile.Read(ref _executed);
            var end = address + size;
            foreach (var shadow in _shadows)
            {
                if (shadow.Sequence > executed && address < shadow.Address + (ulong)shadow.Length && shadow.Address < end)
                {
                    return true;
                }
            }

            return false;
        }

        public bool TryReadShadow(ulong address, Span<byte> destination, out long sequence)
        {
            sequence = 0;
            if (_shadows.Count == 0)
            {
                return false;
            }

            var end = address + (ulong)destination.Length;
            lock (_writeLock)
            {
                var executed = Volatile.Read(ref _executed);
                for (var index = _shadows.Count - 1; index >= 0; index--)
                {
                    var shadow = _shadows[index];
                    var shadowEnd = shadow.Address + (ulong)shadow.Length;
                    if (shadow.Sequence <= executed || address >= shadowEnd || end <= shadow.Address)
                    {
                        continue;
                    }

                    if (address < shadow.Address || end > shadowEnd || !shadow.Known.TryCopyWords((int)(address - shadow.Address), destination))
                    {
                        return false;
                    }

                    sequence = shadow.Sequence;
                    ApplyPendingStoresLocked(address, destination, shadow.Sequence);
                    return true;
                }
            }

            return false;
        }

        // Guest memory as the stream sees it: with its own stores applied, queued or not.
        public bool ReadOverlay(ulong address, Span<byte> destination)
        {
            WaitForPendingRange(address, (ulong)destination.Length);
            if (_pendingWriteCount == 0)
            {
                return _memory.TryRead(address, destination);
            }

            lock (_writeLock)
            {
                if (!_memory.TryRead(address, destination))
                {
                    return false;
                }

                ApplyPendingStoresLocked(address, destination, 0);
                return true;
            }
        }

        private void ApplyPendingStoresLocked(ulong address, Span<byte> destination, long afterSequence)
        {
            var end = address + (ulong)destination.Length;
            foreach (var write in _pendingWrites)
            {
                var writeEnd = write.Address + (ulong)write.Data!.Length;
                if (write.Sequence <= afterSequence || write.Address >= end || writeEnd <= address)
                {
                    continue;
                }

                var start = Math.Max(address, write.Address);
                var stop = Math.Min(end, writeEnd);
                write.Data.AsSpan((int)(start - write.Address), (int)(stop - start))
                    .CopyTo(destination[(int)(start - address)..]);
            }
        }

        private void Enqueue(Item item)
        {
            ThrowIfFailed();
            if (Volatile.Read(ref _outstanding) >= MaxOutstanding)
            {
                WaitForSpace();
            }

            lock (_lock)
            {
                item.Sequence = ++_enqueued;
                _incoming.Add(item);
            }

            Interlocked.Increment(ref _outstanding);
            if (_consumerIdle || item.Sync)
            {
                Presenter.WakeRenderThread();
            }
        }

        private void WaitForSpace()
        {
            var started = Diagnostics.CommandThreadTally.Enabled ? Stopwatch.GetTimestamp() : 0;
            try
            {
                WaitForSpaceCore();
            }
            finally
            {
                if (started != 0)
                {
                    Diagnostics.CommandThreadTally.NoteSpaceWait(Stopwatch.GetTimestamp() - started);
                }
            }
        }

        private void WaitForSpaceCore()
        {
            lock (_lock)
            {
                // The flag is set before the check: the render thread reads it after it changes what is checked.
                for (;;)
                {
                    _waitingForSpace = true;
                    Interlocked.MemoryBarrier();
                    if (Volatile.Read(ref _outstanding) < MaxOutstanding / 2 || _failure is not null)
                    {
                        break;
                    }

                    Monitor.Wait(_lock, 10);
                }

                _waitingForSpace = false;
            }

            ThrowIfFailed();
        }

        public void WaitPending(long sequence)
        {
            if (Volatile.Read(ref _executed) >= sequence)
            {
                return;
            }

            var started = Diagnostics.CommandThreadTally.Enabled ? Stopwatch.GetTimestamp() : 0;
            var spinner = new SpinWait();
            for (var spin = 0; spin < 100 && Volatile.Read(ref _executed) < sequence; spin++)
            {
                spinner.SpinOnce(sleep1Threshold: -1);
            }

            if (Volatile.Read(ref _executed) < sequence)
            {
                lock (_lock)
                {
                    for (;;)
                    {
                        Volatile.Write(ref _waitingForSequence, sequence);
                        Interlocked.MemoryBarrier();
                        if (Volatile.Read(ref _executed) >= sequence || _failure is not null)
                        {
                            break;
                        }

                        Monitor.Wait(_lock, 10);
                    }

                    Volatile.Write(ref _waitingForSequence, long.MaxValue);
                }
            }

            if (started != 0)
            {
                Diagnostics.CommandThreadTally.NotePendingWait(Stopwatch.GetTimestamp() - started);
            }

            ThrowIfFailed();
        }

        private void WaitFor(Item item)
        {
            var spinner = new SpinWait();
            for (var spin = 0; spin < 100 && !item.Done; spin++)
            {
                spinner.SpinOnce(sleep1Threshold: -1);
            }

            if (!item.Done)
            {
                lock (_lock)
                {
                    for (;;)
                    {
                        _waitingForSync = true;
                        Interlocked.MemoryBarrier();
                        if (item.Done || _failure is not null)
                        {
                            break;
                        }

                        Monitor.Wait(_lock, 10);
                    }

                    _waitingForSync = false;
                }
            }

            if (item.Failure is { } failure)
            {
                throw new InvalidOperationException("A command stream callback failed on the render thread.", failure);
            }

            if (!item.Done)
            {
                ThrowIfFailed();
            }
        }

        // The consumer side, on the render thread. Runs queued items until the deadline or until
        // none is left; returns how many ran.
        public int RunBatch(long deadline, int maxItems, Action betweenItems)
        {
            if (_drainIndex == _draining.Count)
            {
                _draining.Clear();
                _drainIndex = 0;
                lock (_lock)
                {
                    (_incoming, _draining) = (_draining, _incoming);
                }

                if (_draining.Count == 0)
                {
                    ThrowIfFailed();
                    return 0;
                }
            }

            var ran = 0;
            while (_drainIndex < _draining.Count)
            {
                var item = _draining[_drainIndex];
                _draining[_drainIndex++] = null!;
                Execute(item);
                ran++;
                Volatile.Write(ref _executed, item.Sequence);
                var outstanding = Interlocked.Decrement(ref _outstanding);
                // One pulse per wait: the producer sets its flag again if it keeps waiting.
                var wakeForSync = item.Sync && _waitingForSync;
                var wakeForSpace = _waitingForSpace && outstanding < MaxOutstanding / 2;
                var wakeForSequence = item.Sequence >= Volatile.Read(ref _waitingForSequence);
                if (wakeForSync || wakeForSpace || wakeForSequence)
                {
                    if (wakeForSync)
                    {
                        _waitingForSync = false;
                    }

                    if (wakeForSpace)
                    {
                        _waitingForSpace = false;
                    }

                    if (wakeForSequence)
                    {
                        Volatile.Write(ref _waitingForSequence, long.MaxValue);
                    }

                    lock (_lock)
                    {
                        Monitor.PulseAll(_lock);
                    }
                }

                betweenItems();
                if (ran >= maxItems || ((ran & 31) == 0 && Stopwatch.GetTimestamp() >= deadline))
                {
                    break;
                }
            }

            return ran;
        }

        private void Execute(Item item)
        {
            if (item.IsDraw)
            {
                _runDeferred(item.Work!);
                return;
            }

            try
            {
                switch (item.Kind)
                {
                    case Kind.Deferred:
                        _runDeferred(item.Work!);
                        break;
                    case Kind.Call:
                        item.Action!();
                        break;
                    case Kind.Write:
                        ApplyWrite(item);
                        break;
                }
            }
            catch (Exception exception) when (item.Sync)
            {
                item.Failure = exception;
            }
            catch (Exception exception)
            {
                _failure ??= exception;
                throw;
            }
            finally
            {
                if (item.RangeSize != 0)
                {
                    lock (_writeLock)
                    {
                        _pendingRanges.Remove(item);
                        _pendingRangeCount = _pendingRanges.Count;
                    }
                }

                if (item.Sync)
                {
                    item.Done = true;
                }
            }
        }

        private void ApplyWrite(Item item)
        {
            lock (_writeLock)
            {
                var written = _memory.TryWrite(item.Address, item.Data);
                if (!ReferenceEquals(_pendingWrites.Peek(), item))
                {
                    throw SubmissionScheduler.Fatal("The command stream stores ran out of order.");
                }

                _pendingWrites.Dequeue();
                _pendingWriteCount = _pendingWrites.Count;
                if (!written)
                {
                    throw SubmissionScheduler.Fatal(
                        $"The command stream cannot write guest memory: address=0x{item.Address:X16} size={item.Data!.Length}.");
                }
            }
        }

        public void ThrowIfFailed()
        {
            if (_failure is { } failure)
            {
                throw new InvalidOperationException("The command stream thread failed.", failure);
            }
        }

        // Ends the thread after its current slice; the render thread keeps running its items
        // meanwhile, since the slice may wait for one.
        public void Stop(Action runItems)
        {
            _stopping = true;
            _queue.Wake();
            var deadline = Stopwatch.GetTimestamp() + 5 * Stopwatch.Frequency;
            while (_thread.IsAlive && Stopwatch.GetTimestamp() < deadline)
            {
                runItems();
                _thread.Join(1);
            }

            runItems();
        }

        private void Run()
        {
            SharpEmu.HLE.Host.HostLaneReservation.ApplyToFrontEndThread();
            RenderPhaseProfile.AttachResolveWorker();
            try
            {
                var lastProgress = Stopwatch.GetTimestamp();
                var tally = Diagnostics.CommandThreadTally.Enabled;
                while (!_stopping)
                {
                    var started = tally ? Stopwatch.GetTimestamp() : 0;
                    var result = _runSlice();
                    if (tally)
                    {
                        Diagnostics.CommandThreadTally.NoteSlice(Stopwatch.GetTimestamp() - started);
                        started = Stopwatch.GetTimestamp();
                    }

                    switch (result)
                    {
                        case SliceResult.NoWork:
                            _ = _queue.WaitForWork(8);
                            if (tally)
                            {
                                Diagnostics.CommandThreadTally.NoteIdle(Stopwatch.GetTimestamp() - started);
                            }

                            break;
                        case SliceResult.AllBlocked:
                            WaitBlocked(ref lastProgress);
                            if (tally)
                            {
                                Diagnostics.CommandThreadTally.NoteBlocked(Stopwatch.GetTimestamp() - started);
                            }

                            break;
                        case SliceResult.BlockedWithoutProgress:
                            if (!_queue.HasUnblockedPending)
                            {
                                WaitBlocked(ref lastProgress);
                                if (tally)
                                {
                                    Diagnostics.CommandThreadTally.NoteBlocked(Stopwatch.GetTimestamp() - started);
                                }
                            }

                            break;
                        default:
                            lastProgress = Stopwatch.GetTimestamp();
                            break;
                    }
                }
            }
            catch (Exception exception)
            {
                _failure ??= exception;
                Presenter.WakeRenderThread();
            }
        }

        // Blocked heads retry when a GPU completion or a presentation unblocks them, or every
        // retry interval without progress.
        private void WaitBlocked(ref long lastProgress)
        {
            var remaining = RetryTicks - (Stopwatch.GetTimestamp() - lastProgress);
            if (remaining <= 0)
            {
                _queue.RetryBlocked();
                lastProgress = Stopwatch.GetTimestamp();
                return;
            }

            _ = _queue.WaitForRetryInterval((int)Math.Max(1, remaining * 1000 / Stopwatch.Frequency));
        }
    }
}
