// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.HLE.GuestMemory;
using SharpEmu.Libs.Gpu.Buffers;
using SharpEmu.Libs.Gpu.Vulkan;
using SharpEmu.Libs.Tests.Gpu.Images;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using SharpEmu.Libs.Tests.Gpu.Vulkan;
using Xunit;
using static SharpEmu.Libs.Tests.Gpu.Images.ImageCacheTestSupport;

namespace SharpEmu.Libs.Tests.Gpu.Buffers;

[Collection(SchedulingStateCollection.Name)]
public sealed class GuestBufferDirectReadTests(HeadlessVulkanFixture fixture) : IClassFixture<HeadlessVulkanFixture>
{
    private const ulong BufferBytes = 0x10000;

    private readonly HeadlessVulkan? _vulkan = fixture.Vulkan;

    private static void Attach(CacheHarness harness, HeadlessVulkan vulkan) =>
        harness.Cache.AsyncReadback = new VulkanAsyncReadback(vulkan.DeviceInfo, harness.Scheduler, vulkan.Queue, vulkan.QueueFamily);

    private static void Detach(CacheHarness harness) => harness.Worker.Run(() =>
    {
        harness.Cache.AsyncReadback?.Dispose();
        harness.Cache.AsyncReadback = null;
    });

    private static void GpuWrite(CacheHarness harness, ulong address, uint value, ulong size = 0x100) => harness.Worker.Run(() =>
    {
        var (buffer, offset) = harness.Cache.ObtainBuffer(address, size, isWritten: true);
        buffer.Fill(offset, size, value);
    });

    private static void Finish(CacheHarness harness) => harness.Worker.Run(() => harness.Scheduler.Finish());

    private static void CoverWithOneBuffer(CacheHarness harness, ulong address, ulong size) =>
        harness.Worker.Run(() => _ = harness.Cache.ObtainBuffer(address, size, isWritten: false));

    private static bool IsMapped(CacheHarness harness, ulong address) =>
        harness.Worker.Run(() => harness.Cache.GetBuffer(harness.Cache.FindBuffer(address, 4)).IsMapped);

    private static uint ReadWord(CacheHarness harness, ulong address) => BitConverter.ToUInt32(harness.Read(address, 4));

    private static void ReadTwice(CacheHarness harness, ulong address)
    {
        for (var round = 0u; round < 2; round++)
        {
            GpuWrite(harness, address, 0x10101010u + round);
            Assert.True(harness.Cache.DownloadToCpu(address, 4));
            Assert.Equal(0x10101010u + round, ReadWord(harness, address + 0x10));
        }
    }

    private static void Promote(CacheHarness harness, ulong address)
    {
        ReadTwice(harness, address);
        GpuWrite(harness, address, 0x20202020);
        Assert.True(harness.Cache.DownloadToCpu(address, 4));
        Assert.Equal(0x20202020u, ReadWord(harness, address + 0x10));
        Assert.True(IsMapped(harness, address));
    }

    private CacheHarness? Start(out ulong address, ulong mappedBytes = BufferBytes)
    {
        address = 0;
        if (_vulkan is null || !GatePrerequisites.Ready(_vulkan))
        {
            return null;
        }

        var harness = new CacheHarness(_vulkan);
        if (harness.Cache.MappedBudget < 4 * BufferBytes)
        {
            harness.Dispose();
            return null;
        }

        Attach(harness, _vulkan);
        address = harness.MapBacked(mappedBytes, ReadWrite);
        CoverWithOneBuffer(harness, address, BufferBytes);
        return harness;
    }

    private static void Stop(CacheHarness harness)
    {
        Detach(harness);
        harness.Shutdown();
    }

    [Fact]
    public void ABufferReadBackTwiceMovesToMappedMemory()
    {
        using var harness = Start(out var address);
        if (harness is null) return;
        using var fatal = new FatalScope();
        ReadTwice(harness, address);
        Assert.False(IsMapped(harness, address));
        Assert.Equal(0UL, harness.Cache.PromotedBytes);

        GpuWrite(harness, address, 0x20202020);
        Assert.True(harness.Cache.DownloadToCpu(address, 4));

        Assert.True(IsMapped(harness, address));
        Assert.Equal(BufferBytes, harness.Cache.PromotedBytes);
        Assert.Equal(0x20202020u, ReadWord(harness, address + 0x10));
        Assert.Equal(0, harness.Cache.DirectReadsApplied);
        Stop(harness);
    }

    [Fact]
    public void ACompletedWriterIsReadStraightFromMappedMemory()
    {
        using var harness = Start(out var address);
        if (harness is null) return;
        using var fatal = new FatalScope();
        Promote(harness, address);
        var applied = harness.Cache.PendingReadbacksApplied;

        GpuWrite(harness, address, 0x30303030);
        Finish(harness);
        Assert.True(harness.Cache.HasGpuDirtyBytes(address, 0x100));
        Assert.True(harness.Cache.DownloadToCpu(address + 8, 4));

        Assert.Equal(1, harness.Cache.DirectReadsApplied);
        Assert.Equal(applied, harness.Cache.PendingReadbacksApplied);
        Assert.False(harness.Cache.HasGpuDirtyBytes(address, 0x100));
        Assert.False(harness.Cache.HasGpuDirtyPages(address, 0x100));
        Assert.Equal(0x30303030u, ReadWord(harness, address + 0xF0));
        Stop(harness);
    }

    [Fact]
    public void AWriterStillRecordingFallsBackToTheReadbackQueue()
    {
        using var harness = Start(out var address);
        if (harness is null) return;
        using var fatal = new FatalScope();
        Promote(harness, address);
        var applied = harness.Cache.PendingReadbacksApplied;

        GpuWrite(harness, address, 0x40404040);
        Assert.True(harness.Cache.DownloadToCpu(address, 4));

        Assert.Equal(0, harness.Cache.DirectReadsApplied);
        Assert.Equal(applied + 1, harness.Cache.PendingReadbacksApplied);
        Assert.False(harness.Cache.HasGpuDirtyBytes(address, 0x100));
        Assert.Equal(0x40404040u, ReadWord(harness, address + 0x20));
        Stop(harness);
    }

    [Fact]
    public void AWriteInFlightElsewhereInTheBufferStaysGpuDirty()
    {
        using var harness = Start(out var address);
        if (harness is null) return;
        using var fatal = new FatalScope();
        Promote(harness, address);

        GpuWrite(harness, address, 0x50505050);
        Finish(harness);
        GpuWrite(harness, address + 0x8000, 0x51515151);
        Assert.True(harness.Cache.DownloadToCpu(address, 4));

        Assert.Equal(1, harness.Cache.DirectReadsApplied);
        Assert.Equal(0x50505050u, ReadWord(harness, address + 0x40));
        Assert.False(harness.Cache.HasGpuDirtyBytes(address, 0x100));
        Assert.True(harness.Cache.HasGpuDirtyBytes(address + 0x8000, 0x100));
        Assert.True(harness.Cache.HasGpuDirtyPages(address + 0x8000, 0x100));

        Assert.True(harness.Cache.DownloadToCpu(address + 0x8000, 4));
        Assert.Equal(0x51515151u, ReadWord(harness, address + 0x8040));
        Assert.False(harness.Cache.HasGpuDirtyBytes(address + 0x8000, 0x100));
        Stop(harness);
    }

    [Fact]
    public void ACpuWriteTakesEveryGpuWriteInItsPage()
    {
        using var harness = Start(out var address);
        if (harness is null) return;
        using var fatal = new FatalScope();
        Promote(harness, address);

        GpuWrite(harness, address, 0x60606060);
        GpuWrite(harness, address + 0x800, 0x61616161);
        Finish(harness);
        Assert.True(harness.Cache.TrySynchronizeCpuRead(address, 4, GuestMemoryProfile.ReadbackSource.CpuReadSynchronization));
        Assert.Equal(1, harness.Cache.DirectReadsApplied);
        Assert.Equal(0x60606060u, ReadWord(harness, address));
        Assert.True(harness.Cache.HasGpuDirtyBytes(address + 0x800, 0x100));
        Assert.True(harness.Cache.HasGpuDirtyPages(address, 4));

        Assert.True(harness.Store.MarkCpuWrite(address, 8));

        Assert.Equal(2, harness.Cache.DirectReadsApplied);
        Assert.False(harness.Cache.HasGpuDirtyBytes(address, 0x1000));
        Assert.False(harness.Cache.HasGpuDirtyPages(address, 0x1000));
        Assert.Equal(0x61616161u, ReadWord(harness, address + 0x840));
        Stop(harness);
    }

    [Fact]
    public void ACpuWriteIntoAPageWithAPendingWriterFallsBack()
    {
        using var harness = Start(out var address);
        if (harness is null) return;
        using var fatal = new FatalScope();
        Promote(harness, address);

        GpuWrite(harness, address, 0x62626262);
        Finish(harness);
        GpuWrite(harness, address + 0x800, 0x63636363);
        var applied = harness.Cache.PendingReadbacksApplied;

        Assert.True(harness.Store.MarkCpuWrite(address, 8));

        Assert.Equal(0, harness.Cache.DirectReadsApplied);
        Assert.Equal(applied + 1, harness.Cache.PendingReadbacksApplied);
        Assert.False(harness.Cache.HasGpuDirtyPages(address, 0x1000));
        Assert.Equal(0x62626262u, ReadWord(harness, address + 0x40));
        Assert.Equal(0x63636363u, ReadWord(harness, address + 0x840));
        Stop(harness);
    }

    [Fact]
    public void AMergeKeepsThePromotionWhileTheBudgetAllows()
    {
        using var harness = Start(out var address, 4 * BufferBytes);
        if (harness is null) return;
        using var fatal = new FatalScope();
        Promote(harness, address);

        CoverWithOneBuffer(harness, address, 2 * BufferBytes);
        Assert.True(IsMapped(harness, address));
        Assert.Equal(2 * BufferBytes, harness.Cache.PromotedBytes);
        GpuWrite(harness, address + BufferBytes, 0x70707070);
        Finish(harness);
        Assert.True(harness.Cache.DownloadToCpu(address + BufferBytes, 4));
        Assert.Equal(1, harness.Cache.DirectReadsApplied);
        Assert.Equal(0x70707070u, ReadWord(harness, address + BufferBytes + 0x10));

        harness.Cache.MappedBudget = 2 * BufferBytes;
        CoverWithOneBuffer(harness, address, 3 * BufferBytes);
        Assert.False(IsMapped(harness, address));
        Assert.Equal(0UL, harness.Cache.PromotedBytes);
        GpuWrite(harness, address + 2 * BufferBytes, 0x71717171);
        Finish(harness);
        Assert.True(harness.Cache.DownloadToCpu(address + 2 * BufferBytes, 4));
        Assert.Equal(1, harness.Cache.DirectReadsApplied);
        Assert.Equal(0x71717171u, ReadWord(harness, address + 2 * BufferBytes + 0x10));
        Stop(harness);
    }

    [Fact]
    public void AFullBudgetLeavesTheBufferInDeviceMemory()
    {
        using var harness = Start(out var address);
        if (harness is null) return;
        using var fatal = new FatalScope();
        harness.Cache.MappedBudget = BufferBytes - 1;
        ReadTwice(harness, address);
        GpuWrite(harness, address, 0x80808080);
        Assert.True(harness.Cache.DownloadToCpu(address, 4));

        Assert.False(IsMapped(harness, address));
        Assert.Equal(0UL, harness.Cache.PromotedBytes);
        Assert.Equal(0x80808080u, ReadWord(harness, address + 0x10));
        Stop(harness);
    }
}
