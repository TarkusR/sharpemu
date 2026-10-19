// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace SharpEmu.Libs.VideoOut;

using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.Libs.Gpu.Vulkan;
using Silk.NET.Vulkan;
using VkBuffer = Silk.NET.Vulkan.Buffer;

// Thrown by a recording thread when an item needs a cache mutation only the render thread may make in
// stream order; the slice ends before the item and the render thread records it itself.
internal sealed class SerialRequiredException(string reason, bool resumable = false) : Exception(reason)
{
    public bool Resumable => resumable;
}

internal static unsafe partial class VulkanVideoPresenter
{
    private static long _slicesRecorded;
    private static long _segmentsRecorded;
    private static long _segmentsEmpty;
    private static long _slicesSerial;
    private static long _slicesInvalid;
    private static long _sliceItems;
    private static long _serialItems;
    private static long _sliceWaitTicks;
    private static long _slicesStolen;
    private static long _stolenItems;
    private static long _restChunks;
    private static long _restItems;
    private static long _streamCopies;
    private static long _streamFallbacks;
    private static readonly ConcurrentDictionary<string, long> _serialReasons = new();
    private static int _gc0;
    private static int _gc1;
    private static int _gc2;
    private static long _allocated;
    private static TimeSpan _gcPaused;
    private static long _renderAllocated;
    private static long _workerAllocated;

    internal static string TakeSliceReport() =>
        FormattableString.Invariant($"[PERF][SLICES] slices={Interlocked.Exchange(ref _slicesRecorded, 0)} segments={Interlocked.Exchange(ref _segmentsRecorded, 0)} empty={Interlocked.Exchange(ref _segmentsEmpty, 0)} ") +
        FormattableString.Invariant($"serial_cuts={Interlocked.Exchange(ref _slicesSerial, 0)} invalid={Interlocked.Exchange(ref _slicesInvalid, 0)} ") +
        FormattableString.Invariant($"slice_items={Interlocked.Exchange(ref _sliceItems, 0)} serial_items={Interlocked.Exchange(ref _serialItems, 0)} join_wait_ms={Interlocked.Exchange(ref _sliceWaitTicks, 0) * 1000.0 / System.Diagnostics.Stopwatch.Frequency:F1} ") +
        FormattableString.Invariant($"stream_copies={Interlocked.Exchange(ref _streamCopies, 0)} stream_fallbacks={Interlocked.Exchange(ref _streamFallbacks, 0)} ") +
        FormattableString.Invariant($"stolen={Interlocked.Exchange(ref _slicesStolen, 0)} stolen_items={Interlocked.Exchange(ref _stolenItems, 0)} rest_chunks={Interlocked.Exchange(ref _restChunks, 0)} rest_items={Interlocked.Exchange(ref _restItems, 0)} ") +
        "serial_reasons=" + TakeSerialReasons() + " " + GuestImageCache.TakeSliceLookupReport() + " " + TakeGcReport();

    private static string TakeGcReport()
    {
        var gc0 = GC.CollectionCount(0);
        var gc1 = GC.CollectionCount(1);
        var gc2 = GC.CollectionCount(2);
        var allocated = GC.GetTotalAllocatedBytes();
        var paused = GC.GetTotalPauseDuration();
        var renderAllocated = GC.GetAllocatedBytesForCurrentThread();
        var report = FormattableString.Invariant($"gc={gc0 - _gc0}/{gc1 - _gc1}/{gc2 - _gc2} alloc_mb={(allocated - _allocated) / 1048576.0:F0} gc_pause_ms={(paused - _gcPaused).TotalMilliseconds:F0} ") +
            FormattableString.Invariant($"render_alloc_mb={(renderAllocated - _renderAllocated) / 1048576.0:F0} worker_alloc_mb={Interlocked.Exchange(ref _workerAllocated, 0) / 1048576.0:F0}");
        _renderAllocated = renderAllocated;
        _gc0 = gc0;
        _gc1 = gc1;
        _gc2 = gc2;
        _allocated = allocated;
        _gcPaused = paused;
        return report;
    }

    private static string TakeSerialReasons()
    {
        var reasons = _serialReasons.ToArray();
        foreach (var reason in reasons)
        {
            _serialReasons.TryRemove(reason.Key, out _);
        }

        return string.Join(",", reasons.OrderByDescending(reason => reason.Value).Select(reason => reason.Key.Replace(' ', '_') + ":" + reason.Value.ToString(System.Globalization.CultureInfo.InvariantCulture)));
    }

    private sealed class SliceRequests
    {
        public readonly List<BufferSliceUse> Buffers = new();
        public readonly List<(ulong Address, ulong Size)> SyncRanges = new();
        public readonly List<ImageSliceUse> Images = new();
        public readonly List<(ulong Address, uint[] Words)> KnownWrites = new();
        public readonly List<SubmissionUploadResources> Resources = new();
        public bool NeedsBdaSweep;
        public ulong LookupGeneration;
        public long MetadataGeneration;
        public IReadOnlyList<GuestSpan>? BdaSpans;
        public ulong BdaMapping;
    }

    // Items recorded into one command buffer; the item after them, if any, needs the render thread.
    private sealed class SliceSegment
    {
        public readonly SliceRequests Requests = new();
        public int First;
        public int Count;
        public string? SerialReason;
        public RecordingBuffer? Buffer;
        public int DrawCount;
    }

    private sealed class RenderSlice
    {
        public readonly List<CommandThread.Item> Items = new();
        public readonly ConcurrentQueue<SliceSegment> Segments = new();
        public readonly ManualResetEventSlim Progress = new(false);
        public readonly ManualResetEventSlim Done = new(false);
        public long Generation;
        public int Resume;
        public RecordingHost? Host;
        public List<SliceArenaChunk> Chunks = [];
        public bool Submitted;
        public bool Dispatched;
        public Exception? Failure;
        private int _claimed;

        public bool TryClaim() => Interlocked.CompareExchange(ref _claimed, 1, 0) == 0;
    }


    private sealed class RenderWorkerPool : IDisposable
    {
        private readonly ConcurrentQueue<RenderSlice> _queue = new();
        private readonly ConcurrentQueue<RenderSlice> _urgent = new();
        private readonly SemaphoreSlim _available = new(0);
        private volatile bool _stopping;
        private readonly Thread[] _threads;
        private readonly RecordingHost[] _hosts;

        public RenderWorkerPool(Presenter presenter, int count)
        {
            _hosts = new RecordingHost[count];
            _threads = new Thread[count];
            for (var index = 0; index < count; index++)
            {
                var host = new RecordingHost(presenter, worker: true);
                _hosts[index] = host;
                _threads[index] = new Thread(() => Run(host)) { IsBackground = true, Name = $"SharpEmu render worker {index}" };
                _threads[index].Start();
            }
        }

        public int Count => _threads.Length;

        public void Dispatch(RenderSlice slice, bool urgent)
        {
            (urgent ? _urgent : _queue).Enqueue(slice);
            _available.Release();
        }

        private void Run(RecordingHost host)
        {
            RenderPhaseProfile.AttachResolveWorker();
            SubmissionScheduler.MarkReadOnlyThread(static operation => new SerialRequiredException("unguarded " + operation));
            while (true)
            {
                _available.Wait();
                if (!_urgent.TryDequeue(out var slice) && !_queue.TryDequeue(out slice))
                {
                    if (_stopping)
                    {
                        break;
                    }

                    continue;
                }

                if (!slice.TryClaim())
                {
                    continue;
                }

                var allocated = RenderPhaseProfile.Enabled ? GC.GetAllocatedBytesForCurrentThread() : 0;
                try
                {
                    host.RecordSlice(slice);
                }
                catch (Exception exception)
                {
                    slice.Failure = exception;
                }
                finally
                {
                    if (allocated != 0)
                    {
                        Interlocked.Add(ref _workerAllocated, GC.GetAllocatedBytesForCurrentThread() - allocated);
                    }

                    slice.Done.Set();
                    slice.Progress.Set();
                }
            }
        }

        public void Dispose()
        {
            _stopping = true;
            _available.Release(_threads.Length);
            foreach (var thread in _threads)
            {
                thread.Join();
            }

            foreach (var host in _hosts)
            {
                host.DisposeWorker();
            }
        }
    }

    private sealed partial class Presenter
    {
        internal static readonly int RenderWorkerCount =
            int.TryParse(Environment.GetEnvironmentVariable("SHARPEMU_RENDER_WORKERS"), out var workers) && workers >= 0
                ? workers
                : Math.Clamp(Environment.ProcessorCount / 4, 1, 6);

        private RenderWorkerPool? _workers;
        internal SliceArenaPool? _arenaPool;

        private void CreateRenderWorkers()
        {
            if (RenderWorkerCount == 0)
            {
                return;
            }

            _scheduler.MemoryBarrierRecorder ??= RecordRequestedMemoryBarrier;
            _arenaPool = new SliceArenaPool(_deviceInfo, _scheduler);
            _workers = new RenderWorkerPool(this, RenderWorkerCount);
            Console.Error.WriteLine($"[LOADER][INFO] Render workers: {RenderWorkerCount}");
        }

        private void StopRenderWorkers()
        {
            _workers?.Dispose();
            _workers = null;
            _arenaPool?.Dispose();
            _arenaPool = null;
        }

        internal bool SlicesEnabled => _workers is not null && !_deviceLost;

        internal int WorkerCount => _workers?.Count ?? 0;

        internal void DispatchSlice(RenderSlice slice, bool urgent = false)
        {
            slice.Dispatched = true;
            slice.Generation = _scheduler.BeginSliceCut();
            _bufferCache.RefreshSnapshot();
            _workers!.Dispatch(slice, urgent);
        }

        internal static void NoteStolen(int items)
        {
            Interlocked.Increment(ref _slicesStolen);
            Interlocked.Add(ref _stolenItems, items);
        }

        internal static void NoteRestChunk(int items)
        {
            Interlocked.Increment(ref _restChunks);
            Interlocked.Add(ref _restItems, items);
        }

        // The next finished segment of the slice in order, or null once the recording thread is done with it.
        internal SliceSegment? NextSegment(RenderSlice slice)
        {
            var waitStarted = System.Diagnostics.Stopwatch.GetTimestamp();
            var spinner = new SpinWait();
            SliceSegment? segment;
            while (!slice.Segments.TryDequeue(out segment))
            {
                if (slice.Done.IsSet)
                {
                    if (slice.Segments.TryDequeue(out segment))
                    {
                        break;
                    }

                    Interlocked.Add(ref _sliceWaitTicks, System.Diagnostics.Stopwatch.GetTimestamp() - waitStarted);
                    if (slice.Failure is { } failure)
                    {
                        throw new InvalidOperationException("A render worker failed.", failure);
                    }

                    return null;
                }

                if (spinner.Count < 200)
                {
                    spinner.SpinOnce(sleep1Threshold: -1);
                    continue;
                }

                slice.Progress.Reset();
                if (slice.Segments.TryDequeue(out segment))
                {
                    break;
                }

                if (!slice.Done.IsSet)
                {
                    slice.Progress.Wait();
                }
            }

            Interlocked.Add(ref _sliceWaitTicks, System.Diagnostics.Stopwatch.GetTimestamp() - waitStarted);
            return segment;
        }

        // Joins one segment in stream order: validates what the recording thread resolved, records the
        // prologue, applies the mutations it skipped and submits the prologue and the segment together.
        // False when the segment's items must run on this thread instead.
        internal bool JoinSegment(RenderSlice slice, SliceSegment segment)
        {
            try
            {
                if (segment.Count == 0)
                {
                    Interlocked.Increment(ref _segmentsEmpty);
                    _scheduler.DiscardRecording(segment.Buffer!);
                    return false;
                }

                if (!_imageCache.ValidateSliceImages(segment.Requests.Images, segment.Requests.LookupGeneration, segment.Requests.MetadataGeneration) || !_bufferCache.ValidateSliceBuffers(segment.Requests.Buffers) ||
                    !ValidateDeferredItems(slice, segment))
                {
                    Interlocked.Increment(ref _slicesInvalid);
                    _scheduler.DiscardRecording(segment.Buffer!);
                    return false;
                }

                RecordSlicePrologue(segment.Requests);
                var uploads = _spareSubmissionResources is { Count: not 0 } spare ? spare.Pop() : new SubmissionUploadResources();
                uploads.DebugName = "SharpEmu slice";
                _batchResources.Add(uploads);
                _batchResources.AddRange(segment.Requests.Resources);
                if (segment.DrawCount != 0)
                {
                    if (_batchDrawCount == 0)
                    {
                        _batchOpenedAt = System.Diagnostics.Stopwatch.GetTimestamp();
                    }

                    _batchDrawCount += segment.DrawCount;
                }

                using (RenderPhaseProfile.Measure(RenderPhaseProfile.Phase.QueueSubmit))
                {
                    _ = _scheduler.Flush(new SubmitBundle(), [segment.Buffer!]);
                }

                EnsureBatchedGuestCommands();
                RequestGuestMemoryBarrier(RecordingHost.GlobalMemoryBarrier);

                slice.Submitted = true;
                _bufferCache.OnBatchSubmitted();
                Interlocked.Increment(ref _segmentsRecorded);
                Interlocked.Add(ref _sliceItems, segment.Count);
                if (segment.SerialReason is not null)
                {
                    Interlocked.Increment(ref _slicesSerial);
                }

                return true;
            }
            finally
            {
                if (segment.Buffer is { } buffer)
                {
                    segment.Buffer = null;
                    slice.Host!.ReturnWorkerBuffer(buffer);
                }
            }
        }

        private bool ValidateDeferredItems(RenderSlice slice, SliceSegment segment)
        {
            var end = segment.First + segment.Count;
            for (var position = segment.First; position < end; position++)
            {
                if (!_translation.ValidateDeferred(slice.Items[position].Work!))
                {
                    return false;
                }
            }

            return true;
        }

        internal void FinishSlice(RenderSlice slice)
        {
            slice.Done.Wait();
            Interlocked.Increment(ref _slicesRecorded);
            _scheduler.NoteSliceJoined(slice.Generation);
            if (slice.Chunks.Count == 0)
            {
                return;
            }

            var chunks = slice.Chunks;
            slice.Chunks = [];
            if (slice.Submitted)
            {
                _scheduler.QueueCompletionAction(() => chunks.ForEach(_arenaPool!.Return));
            }
            else
            {
                chunks.ForEach(_arenaPool!.Return);
            }
        }

        private void RecordSlicePrologue(SliceRequests requests)
        {
            Recording.EndRendering();
            EnsureBatchedGuestCommands();
            RequestGuestMemoryBarrier(RecordingHost.GlobalMemoryBarrier);
            if (requests.NeedsBdaSweep && requests.BdaSpans is { } spans)
            {
                _bufferCache.PrepareBda(spans, requests.BdaMapping);
            }

            foreach (var (address, size) in requests.SyncRanges)
            {
                _bufferCache.SynchronizeBuffersInRange(address, size);
            }

            foreach (var use in requests.Buffers)
            {
                _bufferCache.ApplySliceUse(in use);
            }

            foreach (var use in requests.Images)
            {
                var request = use.Request;
                _imageCache.ApplySliceUse(use.Image, in request);
            }

            foreach (var (address, words) in requests.KnownWrites)
            {
                _bufferCache.NoteKnownGpuContents(address, words);
            }
        }
    }

    private sealed partial class RecordingHost : IRenderingState
    {
        private readonly bool _worker;
        private RecordingBuffer? _workerBuffer;
        private long _itemSequence;
        private readonly ConcurrentBag<RecordingBuffer>? _spareBuffers;
        private readonly SliceArena? _arena;
        private readonly SliceHazards? _hazards;
        private readonly RenderRecorder? _workerRecorder;
        private readonly Dictionary<CachedImage, BindingBox> _workerBindings = new(ReferenceEqualityComparer.Instance);

        private sealed class BindingBox
        {
            public ImageBindingState Value;
        }
        private SliceSegment? _segment;
        private BufferSnapshot _buffers = BufferSnapshot.Empty;
        private int _sliceDrawCount;

        bool IRenderingState.IsRendering => _renderingActive;

        void IRenderingState.EndRendering() => EndRendering();

        bool IRenderingState.TryDeferUntilRenderingEnds(PipelineStageFlags sourceStages, PipelineStageFlags destinationStages, List<ImageMemoryBarrier2> barriers) =>
            TryDeferUntilRenderingEnds(sourceStages, destinationStages, barriers);

        private static SerialRequiredException Serial(string reason, bool resumable = false) => new(reason, resumable);

        private SliceRequests Requests => _segment?.Requests ?? throw SubmissionScheduler.Fatal("The recording host is not recording a slice.");

        internal void RecordSlice(RenderSlice slice)
        {
            slice.Host = this;
            _buffers = _bufferCache.Snapshot;
            var segment = BeginSegment(0);
            slice.Resume = slice.Items.Count;
            try
            {
                for (var index = 0; index < slice.Items.Count; index++)
                {
                    try
                    {
                        _itemSequence = slice.Items[index].Sequence;
                        _presenter._translation.RunDeferred(slice.Items[index].Work!, _workerRecorder!);
                        segment!.Count++;
                    }
                    catch (SerialRequiredException serial)
                    {
                        segment!.SerialReason = serial.Message;
                        _serialReasons.AddOrUpdate(serial.Message, 1, static (_, count) => count + 1);
                        EndSegment(slice, segment);
                        segment = null;
                        if (!serial.Resumable || index + 1 == slice.Items.Count)
                        {
                            slice.Resume = index + 1;
                            break;
                        }

                        segment = BeginSegment(index + 1);
                    }
                }

                if (segment is not null)
                {
                    EndSegment(slice, segment);
                }
            }
            finally
            {
                slice.Chunks = _arena!.TakeChunks();
                _segment = null;
            }
        }

        private SliceSegment BeginSegment(int first)
        {
            var segment = new SliceSegment { First = first };
            segment.Requests.LookupGeneration = _imageCache.LookupGeneration;
            segment.Requests.MetadataGeneration = _imageCache.MetadataGeneration;
            _segment = segment;
            _workerBuffer = _spareBuffers!.TryTake(out var buffer) ? buffer : _scheduler.CreateRecordingBuffer(this);
            _scheduler.BeginRecording(_workerBuffer);
            _workerBuffer.RequestMemoryBarrier(GlobalMemoryBarrier);
            _hazards!.Reset();
            _workerBindings.Clear();
            _hasBoundDepth = false;
            _depthClearRecordedSeparately = false;
            _boundGraphicsPipeline = null;
            _sliceDrawCount = 0;
            return segment;
        }

        private void EndSegment(RenderSlice slice, SliceSegment segment)
        {
            EndRendering();
            _scheduler.EndRecording(_workerBuffer!);
            segment.Buffer = _workerBuffer;
            segment.DrawCount = _sliceDrawCount;
            slice.Segments.Enqueue(segment);
            slice.Progress.Set();
        }

        internal void ReturnWorkerBuffer(RecordingBuffer buffer)
        {
            if (!buffer.IsInvalid)
            {
                _scheduler.DiscardRecording(buffer);
            }

            _spareBuffers!.Add(buffer);
        }

        internal void DisposeWorker()
        {
            while (_spareBuffers!.TryTake(out var buffer))
            {
                if (!buffer.IsInvalid)
                {
                    _scheduler.DiscardRecording(buffer);
                }
            }
        }

        private CachedImage Image(ResourceSlotIdentifier imageIdentifier) =>
            _worker ? _imageCache.PeekImage(imageIdentifier) ?? throw Serial("image released", resumable: true) : _imageCache.GetImage(imageIdentifier);

        private ref ImageBindingState BindingOf(CachedImage image)
        {
            if (!_worker)
            {
                return ref image.Binding;
            }

            if (!_workerBindings.TryGetValue(image, out var box))
            {
                box = new BindingBox();
                _workerBindings.Add(image, box);
            }

            return ref box.Value;
        }

        private ResourceSlotIdentifier FindBuffer(ulong address, ulong size)
        {
            if (!_worker)
            {
                return _bufferCache.FindBuffer(address, size);
            }

            if (address == 0)
            {
                return GuestBufferCache.NullBufferId;
            }

            return _buffers.TryFind(address, size, out _, out var identifier) ? identifier : throw Serial("buffer lookup", resumable: true);
        }

        private (VkBuffer Buffer, ulong Offset) ObtainBufferForSlice(ulong address, ulong size, bool isWritten, bool isTexelBuffer, bool invalidateImages = false)
        {
            var (buffer, offset, _) = ObtainBufferRangeForSlice(address, size, isWritten, isTexelBuffer, invalidateImages);
            return (buffer, offset);
        }

        private (VkBuffer Buffer, ulong Offset, ulong Available) ObtainBufferRangeForSlice(ulong address, ulong size, bool isWritten, bool isTexelBuffer,
            bool invalidateImages = false)
        {
            if (address == 0 || size == 0)
            {
                throw Serial("buffer range", resumable: true);
            }

            if (_presenter.HasOverlappingPublications(address, size, includeLabels: true))
            {
                throw Serial("pending publication", resumable: true);
            }

            var imageBytes = isTexelBuffer && _bufferCache.MayHaveImageBytes(address, size);
            if (!isWritten && !imageBytes && _bufferCache.PrefersStreamCopy(address, size, isTexelBuffer) &&
                _presenter._translation.PendingWrites is { } pending && !pending.MayHavePending(address, size))
            {
                var thread = _presenter._commandThread!;
                if (thread.HasPendingRange(address, size))
                {
                    Interlocked.Increment(ref _streamFallbacks);
                }
                else if (_arena!.TryReserve(size, _bufferCache.StreamOffsetAlignment, out var reservation))
                {
                    if (thread.ReadOverlayBefore(address, reservation.Destination, _itemSequence))
                    {
                        _arena.Commit(in reservation);
                        Interlocked.Increment(ref _streamCopies);
                        return (reservation.Buffer, reservation.Offset, (size + 3) & ~3UL);
                    }
                }
            }

            if (!_buffers.TryFind(address, size, out var buffer, out _))
            {
                throw Serial("buffer lookup", resumable: true);
            }

            Requests.Buffers.Add(new BufferSliceUse(address, size, isWritten, isTexelBuffer, invalidateImages, buffer));
            return (buffer.Handle, buffer.Offset(address), buffer.Size - buffer.Offset(address));
        }

        private void CompleteOverlappingPublications(ulong address, ulong size, bool includeLabels = true)
        {
            if (!_worker)
            {
                _presenter.CompleteOverlappingPublications(address, size, includeLabels);
            }
            else if (_presenter.HasOverlappingPublications(address, size, includeLabels))
            {
                throw Serial("pending publication", resumable: true);
            }
        }

        private void SynchronizeColorMetadata(in ImageRequest request)
        {
            if (_worker && request.Description.Metadata.NativeColorClear)
            {
                throw Serial("native color metadata", resumable: true);
            }

            _imageCache.SynchronizeColorMetadata(request);
        }

        private void ApplyNativeColorClear(ResourceSlotIdentifier imageIdentifier, in ImageRequest request)
        {
            if (_worker && request.Description.Metadata.NativeColorClear)
            {
                throw Serial("native color clear", resumable: true);
            }

            _imageCache.ApplyNativeColorClear(imageIdentifier, request);
        }

        private void TransitionImage(CachedImage image, AccessFlags access, SubresourceRange? range, CommandBuffer command)
        {
            if (!_worker)
            {
                image.Transition(access, range, command);
                return;
            }

            var stage = SliceHazards.StagesOf(access);
            var barriers = _hazards!.Transition(image, access, stage, out var sourceStages);
            if (barriers.Count == 0)
            {
                return;
            }

            var source = sourceStages == 0 ? PipelineStageFlags.TopOfPipeBit : sourceStages;
            if (CachedImage.OnlyRepeatsAttachmentAccess(barriers) && TryDeferUntilRenderingEnds(source, stage, barriers))
            {
                return;
            }

            EndRendering();
            fixed (ImageMemoryBarrier2* pointer = CollectionsMarshal.AsSpan(barriers))
            {
                VulkanSynchronization.PipelineBarrier(_vk, command, source, stage, DependencyFlags.ByRegionBit, 0, null, 0, null, (uint)barriers.Count, pointer);
            }
        }

        private void AcquireColorAttachmentForSlice(ColorAttachment target)
        {
            if (_imageCache.IsStaleForSlice(target.ImageIdentifier))
            {
                throw Serial("stale color target");
            }

            if (!_imageCache.TryPrepareColorTargetViewForSlice(target.ImageIdentifier, target.Request, out var view))
            {
                throw Serial("color target acquire", resumable: true);
            }

            var image = Image(target.ImageIdentifier);
            if (image.Backing.Samples != target.Resolution.Samples || view.Handle == 0)
            {
                throw SubmissionScheduler.Fatal(
                    $"The color target backing does not match its request: address=0x{target.Address:X16} backingSamples={image.Backing.Samples} samples={target.Resolution.Samples}.");
            }

            target.View = view;
            target.Image = image;
            Requests.Images.Add(new ImageSliceUse(target.ImageIdentifier, image, target.Request));
            var range = target.Request.View;
            TransitionImage(image, AccessFlags.ColorAttachmentReadBit | AccessFlags.ColorAttachmentWriteBit,
                new SubresourceRange(range.BaseLevel, range.LevelCount, range.BaseLayer, range.LayerCount), BeginBatchedGuestCommands());
            target.Clear = false;
            target.ClearValue = default;
            var description = target.Request.Description;
            if (description.Metadata.Kind != MetadataKind.Dcc || description.Metadata.Range.Size != 0)
            {
                return;
            }

            var metadataAddress = description.Metadata.Range.Address;
            for (uint layer = 0; layer < range.LayerCount; layer++)
            {
                if (_imageCache.IsMetadataCleared(metadataAddress, range.BaseLayer + layer))
                {
                    throw Serial("pending dcc clear");
                }

                if (description.DccSliceSize != 0 &&
                    _imageCache.TryReadGuestDccClear(metadataAddress, description.DccSliceSize, range.BaseLayer + layer, out _, out _))
                {
                    throw Serial("guest dcc clear");
                }
            }
        }

        private DepthAttachmentAcquisition AcquireDepthAttachmentForSlice(in DepthAttachmentState depth)
        {
            var resolution = depth.Target.Target;
            if (_imageCache.IsStaleForSlice(depth.Image))
            {
                throw Serial("stale depth target");
            }

            var layer = resolution.Request.View.BaseLayer;
            if (resolution.HasHtile && (resolution.DepthClearEnabled || _imageCache.IsMetadataCleared(resolution.HtileAddress, layer)))
            {
                throw Serial("depth clear");
            }

            if (!_imageCache.TryPrepareDepthTargetViewForSlice(depth.Image, resolution.Request, out var view))
            {
                throw Serial("depth target acquire", resumable: true);
            }

            var image = Image(depth.Image);
            Requests.Images.Add(new ImageSliceUse(depth.Image, image, resolution.Request));
            return new DepthAttachmentAcquisition(view, image.Backing.Samples, false);
        }

        private ImageView AcquireTextureViewForSlice(ResourceSlotIdentifier imageIdentifier, in ImageRequest request)
        {
            if (!_imageCache.TryPrepareTextureViewForSlice(imageIdentifier, in request, out var view))
            {
                throw Serial("texture acquire", resumable: true);
            }

            Requests.Images.Add(new ImageSliceUse(imageIdentifier, Image(imageIdentifier), request));
            return view;
        }
    }
}
