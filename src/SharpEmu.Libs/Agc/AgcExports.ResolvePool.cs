// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Concurrent;
using System.Diagnostics;
using SharpEmu.Libs.Gpu.GpuCommands;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.Libs.VideoOut;

namespace SharpEmu.Libs.Agc;

public static partial class AgcExports
{
    // A draw a worker resolves ahead of the stream from a snapshot of the banks it was issued under;
    // the stream position checks what the worker read before it records the result.
    internal sealed record DeferredDraw : DeferredWork
    {
        public RegisterBanks Banks = null!;
        public ulong SubmitId;
        public bool Indexed;
        public DrawIndexedArguments IndexedArguments;
        public DrawAutoArguments AutoArguments;
        public long Sequence;
        public ResolvedDraw? Record;
        public string? SerialReason;
        public Exception? Failure;
        public readonly ResolveReadLog Reads = new();
        public readonly ManualResetEventSlim Done = new(false);

        public void Reset()
        {
            Banks = null!;
            Record = null;
            SerialReason = null;
            Failure = null;
            Reads.Clear();
            Done.Reset();
        }
    }

    private static long _deferredDraws;
    private static long _deferredValid;
    private static long _deferredInvalid;
    private static long _deferredJoinWaitTicks;
    private static long _deferredSerialTicks;
    private static long _workerShadowReads;
    private static long _deferredHeld;
    private static readonly ConcurrentDictionary<string, long> _deferredSerialReasons = new();

    internal static void NoteWorkerShadowRead() => Interlocked.Increment(ref _workerShadowReads);

    public static string TakeResolvePoolReport()
    {
        var reasons = _deferredSerialReasons.ToArray();
        foreach (var reason in reasons)
        {
            _deferredSerialReasons.TryRemove(reason.Key, out _);
        }

        return FormattableString.Invariant($"[PERF][RESOLVE_POOL] deferred={Interlocked.Exchange(ref _deferredDraws, 0)} held={Interlocked.Exchange(ref _deferredHeld, 0)} valid={Interlocked.Exchange(ref _deferredValid, 0)} invalid={Interlocked.Exchange(ref _deferredInvalid, 0)} ") +
            FormattableString.Invariant($"join_wait_ms={Interlocked.Exchange(ref _deferredJoinWaitTicks, 0) * 1000.0 / Stopwatch.Frequency:F1} serial_ms={Interlocked.Exchange(ref _deferredSerialTicks, 0) * 1000.0 / Stopwatch.Frequency:F1} ") +
            FormattableString.Invariant($"shadow_reads={Interlocked.Exchange(ref _workerShadowReads, 0)} ") +
            "serial=" + string.Join(",", reasons.OrderByDescending(reason => reason.Value)
                .Select(reason => reason.Key.Replace(' ', '_') + ":" + reason.Value.ToString(System.Globalization.CultureInfo.InvariantCulture)));
    }

    internal sealed partial class CommandStreamTranslation
    {
        private ResolvePool? _resolvePool;
        private RenderExecutor? _joinExecutor;
        private CapturingSink? _joinSink;
        private readonly ConcurrentQueue<DeferredDraw> _spareDeferred = new();

        private readonly Dictionary<(ulong Vertex, ulong Pixel), (bool Writes, long Registrations)> _drawWrites = new();
        private (ulong Vertex, ulong Pixel) _learningKey;

        // Only a draw whose shaders are known not to write memory resolves ahead of the stream: its
        // pending-write marks are then render targets only, which the stream never reads back.
        private bool DrawKnownNotToWrite(RegisterBanks banks)
        {
            if (_drawWrites.TryGetValue((banks.Shader.Vertex.ExportAddress, banks.Shader.Pixel.Address), out var known) &&
                known.Registrations == ShaderHeaderRegistry.Registrations && !known.Writes)
            {
                return true;
            }

            Interlocked.Increment(ref _deferredHeld);
            return false;
        }

        private void LearnDrawWrites(ResolvedDraw record)
        {
            if (record.Kind == ResolvedDrawKind.Draw && _learningKey != default)
            {
                _drawWrites[_learningKey] = (record.WritesMemory, ShaderHeaderRegistry.Registrations);
            }
        }

        private void DeferDraw(ResolvePool pool, ulong submitId, RegisterBanks banks, bool indexed, in DrawIndexedArguments indexedArguments, in DrawAutoArguments autoArguments)
        {
            var work = _spareDeferred.TryDequeue(out var spare) ? spare : new DeferredDraw();
            work.Banks = banks.Snapshot();
            work.SubmitId = submitId;
            work.Indexed = indexed;
            work.IndexedArguments = indexedArguments;
            work.AutoArguments = autoArguments;
            work.Sequence = Sink!.EnqueueDeferred(work, isDraw: true);
            Interlocked.Increment(ref _deferredDraws);
            pool.Dispatch(work);
        }

        private void JoinDeferredDraw(DeferredDraw deferred)
        {
            var waitStarted = Stopwatch.GetTimestamp();
            var spinner = new SpinWait();
            while (!deferred.Done.IsSet && spinner.Count < 200)
            {
                spinner.SpinOnce(sleep1Threshold: -1);
            }

            deferred.Done.Wait();
            Interlocked.Add(ref _deferredJoinWaitTicks, Stopwatch.GetTimestamp() - waitStarted);
            if (deferred.Failure is { } failure)
            {
                throw new InvalidOperationException("A resolve worker failed.", failure);
            }

            var record = deferred.Record;
            if (deferred.SerialReason is { } reason)
            {
                _deferredSerialReasons.AddOrUpdate(reason, 1, static (_, count) => count + 1);
                record = ResolveOnThisThread(deferred);
            }
            else if (record is null)
            {
                throw _host.Fatal("A deferred draw resolved to nothing.");
            }
            else if (!_host.ValidateWorkerReads(deferred.Reads))
            {
                Interlocked.Increment(ref _deferredInvalid);
                _pool!.Return(record);
                record = ResolveOnThisThread(deferred);
            }
            else
            {
                Interlocked.Increment(ref _deferredValid);
            }

            _recorder!.Record(record);
            deferred.Reset();
            _spareDeferred.Enqueue(deferred);
        }

        private ResolvedDraw ResolveOnThisThread(DeferredDraw deferred)
        {
            var started = Stopwatch.GetTimestamp();
            var sink = _joinSink ??= new CapturingSink();
            var executor = _joinExecutor ??= new RenderExecutor((IRenderHost)_host, _pipelines!, sink, _executor!.PendingWrites, _pool);
            sink.Sequence = deferred.Sequence;
            sink.Draw = null;
            if (deferred.Indexed)
            {
                executor.DrawIndexed(deferred.SubmitId, deferred.Banks, in deferred.IndexedArguments);
            }
            else
            {
                executor.DrawAuto(deferred.SubmitId, deferred.Banks, in deferred.AutoArguments);
            }

            Interlocked.Add(ref _deferredSerialTicks, Stopwatch.GetTimestamp() - started);
            return sink.Draw ?? throw _host.Fatal("The serial resolve of a deferred draw produced no draw.");
        }

        // Takes the one record an executor produces per draw, with the draw's stream position as its sequence.
        private sealed class CapturingSink : IResolvedCommandSink
        {
            public long Sequence;
            public ResolvedDraw? Draw;

            public long NextSequence() => Sequence;

            public void Record(ResolvedDraw record) => Draw = record;

            public void Record(ResolvedDispatch record) => throw SubmissionScheduler.Fatal("A dispatch reached a draw resolve sink.");
        }

        // Resolves deferred draws on worker threads. One shader pair always goes to the same worker, so
        // a worker's program and materialization memos stay warm and its draws stay in stream order.
        private sealed class ResolvePool : IDisposable
        {
            private readonly CommandStreamTranslation _owner;
            private readonly Worker[] _workers;
            private readonly object _drained = new();
            private int _pending;

            private sealed class Worker
            {
                public readonly BlockingCollection<DeferredDraw> Queue = new();
                public readonly CapturingSink Sink = new();
                public Thread Thread = null!;
                public RenderExecutor Executor = null!;
            }

            public ResolvePool(CommandStreamTranslation owner, IRenderHost host, int count)
            {
                _owner = owner;
                _workers = new Worker[count];
                for (var index = 0; index < count; index++)
                {
                    var worker = new Worker();
                    worker.Executor = new RenderExecutor(host, owner._pipelines!, worker.Sink, owner._executor!.PendingWrites, owner._pool);
                    worker.Thread = new Thread(() => Run(worker)) { IsBackground = true, Name = $"SharpEmu resolve worker {index}" };
                    _workers[index] = worker;
                    worker.Thread.Start();
                }
            }

            public void Dispatch(DeferredDraw work)
            {
                var shader = work.Banks.Shader;
                var route = (uint)HashCode.Combine(shader.Vertex.ExportAddress, shader.Pixel.Address) % (uint)_workers.Length;
                Interlocked.Increment(ref _pending);
                _workers[route].Queue.Add(work);
            }

            public void WaitAllResolved()
            {
                var spinner = new SpinWait();
                while (Volatile.Read(ref _pending) != 0 && spinner.Count < 200)
                {
                    spinner.SpinOnce(sleep1Threshold: -1);
                }

                if (Volatile.Read(ref _pending) == 0)
                {
                    return;
                }

                lock (_drained)
                {
                    while (Volatile.Read(ref _pending) != 0)
                    {
                        Monitor.Wait(_drained, 1);
                    }
                }
            }

            private void Run(Worker worker)
            {
                RenderPhaseProfile.AttachResolveWorker();
                foreach (var work in worker.Queue.GetConsumingEnumerable())
                {
                    Resolve(worker, work);
                    if (Interlocked.Decrement(ref _pending) == 0)
                    {
                        lock (_drained)
                        {
                            Monitor.PulseAll(_drained);
                        }
                    }

                    work.Done.Set();
                }
            }

            private void Resolve(Worker worker, DeferredDraw work)
            {
                var host = _owner._host;
                host.BeginWorkerResolve(work.Sequence, work.Reads);
                try
                {
                    worker.Sink.Sequence = work.Sequence;
                    worker.Sink.Draw = null;
                    if (work.Indexed)
                    {
                        worker.Executor.DrawIndexed(work.SubmitId, work.Banks, in work.IndexedArguments);
                    }
                    else
                    {
                        worker.Executor.DrawAuto(work.SubmitId, work.Banks, in work.AutoArguments);
                    }

                    work.Record = worker.Sink.Draw;
                }
                catch (SerialResolveRequiredException serial)
                {
                    work.SerialReason = serial.Message;
                }
                catch (Exception exception)
                {
                    work.Failure = exception;
                }
                finally
                {
                    host.EndWorkerResolve();
                }
            }

            public void Dispose()
            {
                foreach (var worker in _workers)
                {
                    worker.Queue.CompleteAdding();
                }

                foreach (var worker in _workers)
                {
                    worker.Thread.Join();
                }
            }
        }
    }
}
