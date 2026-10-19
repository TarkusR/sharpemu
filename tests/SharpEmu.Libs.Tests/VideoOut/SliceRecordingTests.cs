// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.GpuMemory;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Images;
using SharpEmu.Libs.Gpu.Scheduling;
using SharpEmu.Libs.Tests.Gpu.Buffers;
using SharpEmu.Libs.Tests.Gpu.Images;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using SharpEmu.Libs.Tests.Gpu.Vulkan;
using SharpEmu.Libs.VideoOut;
using Silk.NET.Vulkan;
using Xunit;
using static SharpEmu.Libs.Tests.Gpu.Scheduling.SchedulingTestSupport;

namespace SharpEmu.Libs.Tests.VideoOut;

[Collection(SchedulingStateCollection.Name)]
public sealed class SliceRecordingTests : IClassFixture<HeadlessVulkanFixture>
{
    private readonly HeadlessVulkan? _vulkan;

    public SliceRecordingTests(HeadlessVulkanFixture fixture) => _vulkan = fixture.Vulkan;

    [Fact]
    public void SliceHazards_FirstAccessNeedsNoBarrierAndRepeatedWritesDo()
    {
        if (!GatePrerequisites.Ready(_vulkan)) return;
        using var harness = new ImageTestHarness(_vulkan);
        var image = harness.CreateImage(CachedImageTests.Color2D(16, 16));
        var hazards = new SliceHazards();

        var first = hazards.Transition(image, AccessFlags.ColorAttachmentWriteBit, PipelineStageFlags.ColorAttachmentOutputBit, out var firstStages);
        Assert.Empty(first);
        Assert.Equal(PipelineStageFlags.None, firstStages);

        var repeated = hazards.Transition(image, AccessFlags.ColorAttachmentWriteBit, PipelineStageFlags.ColorAttachmentOutputBit, out var repeatedStages);
        var barrier = Assert.Single(repeated);
        Assert.Equal((ImageLayout.General, ImageLayout.General), (barrier.OldLayout, barrier.NewLayout));
        Assert.Equal(image.Backing.Handle.Handle, barrier.Image.Handle);
        Assert.Equal(PipelineStageFlags.ColorAttachmentOutputBit, repeatedStages);

        var read = hazards.Transition(image, AccessFlags.ShaderReadBit, PipelineStageFlags.FragmentShaderBit, out _);
        Assert.Single(read);
        var sameRead = hazards.Transition(image, AccessFlags.ShaderReadBit, PipelineStageFlags.FragmentShaderBit, out _);
        Assert.Empty(sameRead);

        hazards.Reset();
        Assert.Empty(hazards.Transition(image, AccessFlags.ShaderReadBit, PipelineStageFlags.FragmentShaderBit, out _));
    }

    [Fact]
    public void SliceHazards_StagesFollowTheAccess()
    {
        Assert.Equal(PipelineStageFlags.TransferBit, SliceHazards.StagesOf(AccessFlags.TransferWriteBit));
        Assert.Equal(PipelineStageFlags.AllGraphicsBit | PipelineStageFlags.ComputeShaderBit, SliceHazards.StagesOf(AccessFlags.ShaderReadBit));
        Assert.Equal(PipelineStageFlags.TransferBit | PipelineStageFlags.AllGraphicsBit | PipelineStageFlags.ComputeShaderBit,
            SliceHazards.StagesOf(AccessFlags.TransferReadBit | AccessFlags.ShaderReadBit));
    }

    [Fact]
    public void SliceArena_PacksAlignedReservationsAndHandsItsChunksOver()
    {
        if (!GatePrerequisites.Ready(_vulkan)) return;
        using var harness = new ImageTestHarness(_vulkan);
        harness.Run(() =>
        {
            using var pool = new SliceArenaPool(harness.Device, harness.Scheduler);
            var arena = new SliceArena(pool, harness.Device.NonCoherentAtomSize);

            Assert.True(arena.TryReserve(100, 256, out var first));
            Assert.True(arena.TryReserve(100, 256, out var second));
            Assert.Same(first.Chunk, second.Chunk);
            Assert.Equal(0UL, first.Offset);
            Assert.Equal(0UL, second.Offset % 256);
            Assert.True(second.Offset >= 100);
            Assert.Equal(100, first.Destination.Length);

            Assert.True(arena.TryUpload(new byte[] { 1, 2, 3, 4 }, 16, out var buffer, out var offset));
            Assert.Equal(first.Buffer.Handle, buffer.Handle);
            Assert.Equal(0UL, offset % 16);
            Assert.Equal(new byte[] { 1, 2, 3, 4 }, first.Chunk.Buffer.Mapped.Slice((int)offset, 4).ToArray());

            Assert.True(arena.TryReserve(SliceArenaPool.ChunkBytes, 256, out var large));
            Assert.NotSame(first.Chunk, large.Chunk);
            Assert.Equal(SliceArenaPool.ChunkBytes, large.Size);

            var chunks = arena.TakeChunks();
            Assert.Equal(2, chunks.Count);
            Assert.Empty(arena.TakeChunks());
            Assert.False(arena.TryReserve(0, 1, out _));

            var allocated = pool.ChunkCount;
            foreach (var chunk in chunks)
            {
                pool.Return(chunk);
            }

            Assert.True(arena.TryReserve(8, 1, out var again));
            Assert.Equal(0UL, again.Offset);
            Assert.True(pool.ChunkCount <= allocated);
            arena.TakeChunks().ForEach(pool.Return);
        });
        harness.AssertNoValidationMessages();
    }

    [Fact]
    public void BufferSnapshot_FindsTheContainingBufferByBinarySearch()
    {
        var snapshot = new BufferSnapshot(
            [0x1000, 0x5000, 0x9000],
            [0x3000, 0x6000, 0xA000],
            [null!, null!, null!],
            [new ResourceSlotIdentifier(1, 1), new ResourceSlotIdentifier(2, 1), new ResourceSlotIdentifier(3, 1)],
            7);

        Assert.True(snapshot.TryFind(0x1000, 0x2000, out _, out var first));
        Assert.Equal(new ResourceSlotIdentifier(1, 1), first);
        Assert.True(snapshot.TryFind(0x5800, 0x100, out _, out var second));
        Assert.Equal(new ResourceSlotIdentifier(2, 1), second);
        Assert.True(snapshot.TryFind(0x9FFF, 1, out _, out var third));
        Assert.Equal(new ResourceSlotIdentifier(3, 1), third);
        Assert.False(snapshot.TryFind(0x2FFF, 2, out _, out _));
        Assert.False(snapshot.TryFind(0x4000, 0x100, out _, out _));
        Assert.False(snapshot.TryFind(0x0100, 0x100, out _, out _));
        Assert.False(BufferSnapshot.Empty.TryFind(0x1000, 1, out _, out _));
        Assert.Equal(7, snapshot.Serial);
    }

    [Fact]
    public void SpinReadWriteLock_ReadersShareWritersExcludeAndTheWriterMayReadAgain()
    {
        var gate = new SpinReadWriteLock();
        using (gate.Read())
        using (gate.Read())
        {
            Assert.False(gate.IsWriteHeldByCurrentThread);
        }

        using (gate.Hold())
        {
            Assert.True(gate.IsWriteHeldByCurrentThread);
            using (gate.Read())
            {
                Assert.True(gate.IsWriteHeldByCurrentThread);
            }

            var entered = 0;
            var reader = new Thread(() =>
            {
                using (gate.Read())
                {
                    Volatile.Write(ref entered, 1);
                }
            });
            reader.Start();
            Assert.False(SpinWait.SpinUntil(() => Volatile.Read(ref entered) != 0, 50));
        }

        using (gate.Hold())
        {
        }
    }

    [Fact]
    public void CompletionActionsParkWhileSlicesAreCutAndReleaseWhenTheyJoin()
    {
        var device = new FakeTickDevice { CompleteOnSubmit = true };
        using var scheduler = NewActiveScheduler(device);
        var ran = new List<string>();

        var generation = scheduler.BeginSliceCut();
        scheduler.QueueCompletionAction(() => ran.Add("parked"));
        scheduler.Flush();
        scheduler.RunCompletedOperations();
        Assert.Empty(ran);

        scheduler.NoteSliceJoined(generation);
        scheduler.RunCompletedOperations();
        Assert.Empty(ran);
        scheduler.Flush();
        scheduler.RunCompletedOperations();
        Assert.Equal(["parked"], ran);

        scheduler.QueueCompletionAction(() => ran.Add("direct"));
        scheduler.Flush();
        scheduler.RunCompletedOperations();
        Assert.Equal(["parked", "direct"], ran);
    }
}
