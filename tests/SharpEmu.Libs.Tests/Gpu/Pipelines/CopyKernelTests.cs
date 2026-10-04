// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.Pipelines;

[Collection(SchedulingStateCollection.Name)]
public sealed class CopyKernelTests : IDisposable
{

    public CopyKernelTests()
    {
        ShaderPipelineCache.CpuModuloCopyEnabled = true;
        ShaderPipelineCache.PredictMaskedCopyEnabled = true;
    }

    public void Dispose()
    {
        ShaderPipelineCache.CpuModuloCopyEnabled = false;
        ShaderPipelineCache.PredictMaskedCopyEnabled = true;
    }

    private const ulong CodeAddress = PipelineTestGuest.MemoryBase + 0x1000;
    private const ulong HeaderAddress = PipelineTestGuest.MemoryBase + 0x8000;
    private const ulong SourceAddress = PipelineTestGuest.MemoryBase + 0x20000;
    private const ulong DestinationAddress = PipelineTestGuest.MemoryBase + 0x30000;
    private const ulong ControlAddress = PipelineTestGuest.MemoryBase + 0x40000;
    private const uint Untouched = 0xEEEEEEEE;

    private static readonly uint[] ModuloKernel =
    [
        0xBEFC03FF, 0xA65CD91E,
        0xBF960000, 0xBFA00003, 0xD7460002, 0x04010C0C, 0xF4201A84, 0xFA000000, 0xBF8CC07F, 0x7DA8046A, 0xBF88002A, 0xF4200304,
        0xFA000004, 0xBF8CC07F, 0x7E000C0C, 0xBF070C80, 0x8588807E, 0x7E005700, 0x100000FF, 0x4F800000, 0x7E060F00, 0xD5766A00,
        0x0202060C, 0x7D8A0280, 0x4C020080, 0x02000101, 0xD56A0001, 0x00020700, 0x4C000303, 0x4A020303, 0x02000101, 0xD56A0000,
        0x00020500, 0xD5690001, 0x0002000C, 0x4C060302, 0x7D8C02F9, 0x06068A02, 0x7D86060C, 0x87EA6A0A, 0x50000080, 0xD5286A00,
        0x002A00C1, 0xD5010000, 0x002200C1, 0xD5690000, 0x0002000C, 0x4C000102, 0xE0002000, 0x80000000, 0xBF8C3F70, 0xE0102000,
        0x80010002, 0xBF810000,
    ];

    private static uint[] DwordBuffer(ulong address, uint records) =>
        [unchecked((uint)address), 0x0004_0000 | (uint)(address >> 32), records, 0x0001_4004];

    private static ComputeStageRegisters Registers(uint sourceRecords, uint destinationRecords)
    {
        var registers = new ComputeStageRegisters
        {
            Address = CodeAddress, ThreadsX = 64, ThreadsY = 1, ThreadsZ = 1, UserScalarCount = 12, ThreadGroupIdXEnable = true,
            ThreadIdComponentCount = 1,
        };
        uint[] words =
        [
            .. DwordBuffer(SourceAddress, sourceRecords),
            .. DwordBuffer(DestinationAddress, destinationRecords),
            unchecked((uint)ControlAddress), 0x0010_0000 | (uint)(ControlAddress >> 32), 1, 0x0004_DFAC,
        ];
        for (var index = 0u; index < words.Length; index++)
        {
            registers.UserScalars.Set(index, words[index], default);
        }

        return registers;
    }

    private static uint[] Destination(PipelineTestGuest guest, int count)
    {
        var bytes = new byte[count * sizeof(uint)];
        Assert.True(guest.Memory.TryRead(DestinationAddress, bytes));
        var words = new uint[count];
        for (var index = 0; index < count; index++)
        {
            words[index] = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(index * sizeof(uint)));
        }

        return words;
    }

    private static PipelineTestGuest Guest(uint elementCount, uint period, uint[] source)
    {
        var guest = new PipelineTestGuest();
        guest.Host.ResidentReads = true;
        guest.RegisterProgram(CodeAddress, HeaderAddress, ModuloKernel);
        guest.WriteWords(SourceAddress, source);
        guest.WriteWords(ControlAddress, elementCount, period, 0, 0);
        guest.WriteWords(DestinationAddress, Enumerable.Repeat(Untouched, 200).ToArray());
        return guest;
    }

    [Theory]
    [InlineData(10u, 3u, 2u, false)]
    [InlineData(130u, 7u, 3u, false)]
    [InlineData(64u, 0u, 1u, false)]
    [InlineData(5u, 1u, 1u, false)]
    [InlineData(100u, 12u, 1u, false)]
    [InlineData(100u, 12u, 70u, true)]
    [InlineData(20u, 5u, 64u, true)]
    public void TheModuloCopyKernel_RunsOnTheCpuInStreamOrder(uint elementCount, uint period, uint dimension, bool threadDimensions)
    {
        var source = Enumerable.Range(0, 80).Select(index => 0x1000u + (uint)index).ToArray();
        var guest = Guest(elementCount, period, source);
        var cache = new ShaderPipelineCache(guest.Context, guest.Host, guest.Compiler, guest.Registry);

        var result = cache.GetComputeProgram(
            Registers(80, 160), new ShaderInterfaceRegisters(), threadDimensions ? 0x8021u : 0x8001u, dimension, 1, 1);

        Assert.True(result.Consumed, ShaderPipelineCache.TakeCopyKernelReport());
        Assert.Empty(guest.Compiler.Requests);
        var written = (int)Math.Min(elementCount, threadDimensions ? dimension : dimension * 64);
        var destination = Destination(guest, 200);
        for (var index = 0; index < 200; index++)
        {
            var sourceIndex = period == 0 ? index : index % (int)period;
            var expected = index >= written ? Untouched : sourceIndex < source.Length ? source[sourceIndex] : 0u;
            Assert.Equal(expected, destination[index]);
        }
    }

    [Theory]
    [InlineData(null)]
    [InlineData("0")]
    public void TheModuloCopyKernel_StaysOnTheGpuUnlessItIsSwitchedOn_AndItsResultIsKnown(string? value)
    {
        ShaderPipelineCache.CpuModuloCopyEnabled = value == "1";
        var guest = Guest(elementCount: 10, period: 4, [1, 2, 3, 4]);
        var cache = new ShaderPipelineCache(guest.Context, guest.Host, guest.Compiler, guest.Registry);

        var result = cache.GetComputeProgram(Registers(4, 64), new ShaderInterfaceRegisters(), 0x8001, 1, 1, 1);

        Assert.False(result.Consumed);
        Assert.True(result.Available);
        Assert.Equal(Untouched, Destination(guest, 1)[0]);
        Assert.NotNull(result.KnownWrite);
        Assert.Equal(DestinationAddress, result.KnownWrite.Address);
        Assert.Equal(new uint[] { 1, 2, 3, 4, 1, 2, 3, 4, 1, 2 }, result.KnownWrite.Words.ToArray());
    }

    private static readonly uint[] MaskedKernel =
    [
        0xBEFC03FF, 0x94E6DD39,
        0xBF960000, 0xBFA00001, 0xD7460000, 0x04010C0C, 0xF4201A84, 0xFA000000, 0xBF8CC07F, 0x7DA8006A, 0xBF880009, 0xF4201A84,
        0xFA000004, 0xBF8CC07F, 0x3602006A, 0xE0002000, 0x80000101, 0xBF8C3F70, 0xE0102000, 0x80010100, 0xBF810000,
    ];

    private static ComputeStageRegisters MaskedRegisters(uint sourceRecords, uint destinationRecords)
    {
        var registers = Registers(sourceRecords, destinationRecords);
        uint[] control = [.. DwordBuffer(ControlAddress, 2)];
        for (var index = 0u; index < control.Length; index++)
        {
            registers.UserScalars.Set(8 + index, control[index], default);
        }

        return registers;
    }

    [Fact]
    public void TheMaskedCopyKernel_InGroupCountMode_IsLeftAloneWhenSwitchedOff()
    {
        ShaderPipelineCache.PredictMaskedCopyEnabled = false;
        var guest = Guest(elementCount: 10, period: 3, [1, 2, 3, 4]);
        guest.RegisterProgram(CodeAddress, HeaderAddress, MaskedKernel);
        var cache = new ShaderPipelineCache(guest.Context, guest.Host, guest.Compiler, guest.Registry);

        var result = cache.GetComputeProgram(Registers(4, 64), new ShaderInterfaceRegisters(), 0x8001, 1, 1, 1);

        Assert.False(result.Consumed);
        Assert.Null(result.KnownWrite);
        Assert.Equal(Untouched, Destination(guest, 1)[0]);
    }

    [Fact]
    public void TheMaskedCopyKernel_InGroupCountMode_StaysOnTheGpuAndItsResultIsKnown()
    {
        var guest = Guest(elementCount: 10, period: 3, [1, 2, 3, 4]);
        guest.RegisterProgram(CodeAddress, HeaderAddress, MaskedKernel);
        var cache = new ShaderPipelineCache(guest.Context, guest.Host, guest.Compiler, guest.Registry);

        var result = cache.GetComputeProgram(Registers(4, 64), new ShaderInterfaceRegisters(), 0x8001, 1, 1, 1);

        Assert.False(result.Consumed);
        Assert.Equal(Untouched, Destination(guest, 1)[0]);
        Assert.NotNull(result.KnownWrite);
        Assert.Equal(new uint[] { 1, 2, 3, 4, 1, 2, 3, 4, 1, 2 }, result.KnownWrite.Words.ToArray());
    }

    [Fact]
    public void TheMaskedCopyKernel_InThreadDimensionMode_StillRunsOnTheCpu()
    {
        var guest = Guest(elementCount: 10, period: 3, [1, 2, 3, 4]);
        guest.RegisterProgram(CodeAddress, HeaderAddress, MaskedKernel);
        var cache = new ShaderPipelineCache(guest.Context, guest.Host, guest.Compiler, guest.Registry);

        var result = cache.GetComputeProgram(MaskedRegisters(4, 64), new ShaderInterfaceRegisters(), 0x8021, 64, 1, 1);

        Assert.True(result.Consumed, ShaderPipelineCache.TakeCopyKernelReport());
        Assert.Equal(new uint[] { 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, Untouched }, Destination(guest, 11));
    }

    [Fact]
    public void TheModuloCopyKernel_ResultIsUnknownWhileTheGpuOwnsItsSource()
    {
        ShaderPipelineCache.CpuModuloCopyEnabled = false;
        var guest = Guest(elementCount: 6, period: 4, [1, 2, 3, 4]);
        guest.Host.GpuOwned = (SourceAddress, 16);
        var cache = new ShaderPipelineCache(guest.Context, guest.Host, guest.Compiler, guest.Registry);

        Assert.Null(cache.GetComputeProgram(Registers(4, 64), new ShaderInterfaceRegisters(), 0x8001, 1, 1, 1).KnownWrite);
    }

    [Fact]
    public void TheModuloCopyKernel_ResultIsUnknownWhenItIsLargerThanATablePatch()
    {
        ShaderPipelineCache.CpuModuloCopyEnabled = false;
        var guest = Guest(elementCount: 1100000, period: 4, [1, 2, 3, 4]);
        var cache = new ShaderPipelineCache(guest.Context, guest.Host, guest.Compiler, guest.Registry);

        Assert.Null(cache.GetComputeProgram(Registers(4, 1100000), new ShaderInterfaceRegisters(), 0x8001, 17500, 1, 1).KnownWrite);
    }

    [Fact]
    public void TheModuloCopyKernel_ResultOfSeveralMegabytesIsKnown()
    {
        ShaderPipelineCache.CpuModuloCopyEnabled = false;
        var guest = Guest(elementCount: 300000, period: 4, [1, 2, 3, 4]);
        var cache = new ShaderPipelineCache(guest.Context, guest.Host, guest.Compiler, guest.Registry);

        var known = cache.GetComputeProgram(Registers(4, 300000), new ShaderInterfaceRegisters(), 0x8001, 5000, 1, 1).KnownWrite;

        Assert.NotNull(known);
        Assert.Equal(300000, known.Words.Length);
        Assert.Equal(new uint[] { 1, 2, 3, 4, 1 }, known.Words[..5].ToArray());
        Assert.Equal(4u, known.Words[299999]);
        known.Release();
    }

    [Theory]
    [InlineData(true, 0u)]
    [InlineData(true, 8u)]
    [InlineData(true, 100u)]
    [InlineData(false, 7u)]
    [InlineData(false, 0xFFFFFFFFu)]
    public void ExpandCopyKernel_CopiesTheSourceAsOneBlockWhenEveryElementMapsToItself(bool modulo, uint parameter)
    {
        uint[] source = [10, 20, 30, 40, 50, 60, 70, 80];
        var output = new uint[8];

        ShaderPipelineCache.ExpandCopyKernel(source, output, parameter, modulo);

        Assert.Equal(source, output);
    }

    [Theory]
    [InlineData(true, 3u, new uint[] { 10, 20, 30, 10, 20, 30, 10, 20 })]
    [InlineData(false, 3u, new uint[] { 10, 20, 30, 40, 10, 20, 30, 40 })]
    [InlineData(false, 5u, new uint[] { 10, 20, 10, 20, 50, 60, 50, 60 })]
    public void ExpandCopyKernel_KeepsThePatternWhenElementsRepeat(bool modulo, uint parameter, uint[] expected)
    {
        uint[] source = [10, 20, 30, 40, 50, 60, 70, 80];
        var output = new uint[8];

        ShaderPipelineCache.ExpandCopyKernel(source, output, parameter, modulo);

        Assert.Equal(expected, output);
    }

    [Fact]
    public void TheModuloCopyKernel_ResultBeyondATablePatchIsKnownAndBorrowsItsBuffer()
    {
        ShaderPipelineCache.CpuModuloCopyEnabled = false;
        var guest = Guest(elementCount: 20000, period: 4, [1, 2, 3, 4]);
        var cache = new ShaderPipelineCache(guest.Context, guest.Host, guest.Compiler, guest.Registry);

        var known = cache.GetComputeProgram(Registers(4, 20000), new ShaderInterfaceRegisters(), 0x8001, 400, 1, 1).KnownWrite;

        Assert.NotNull(known);
        Assert.Equal(20000, known.Words.Length);
        Assert.Equal(new uint[] { 1, 2, 3, 4, 1 }, known.Words[..5].ToArray());
        Assert.Equal(4u, known.Words[19999]);
        known.Release();
        known.Release();
    }

    [Fact]
    public void TheModuloCopyKernel_StopsAtTheDestinationRecords()
    {
        var guest = Guest(elementCount: 50, period: 4, [1, 2, 3, 4]);
        var cache = new ShaderPipelineCache(guest.Context, guest.Host, guest.Compiler, guest.Registry);

        Assert.True(cache.GetComputeProgram(Registers(4, 20), new ShaderInterfaceRegisters(), 0x8001, 1, 1, 1).Consumed);

        var destination = Destination(guest, 24);
        Assert.Equal(4u, destination[19]);
        Assert.Equal(Untouched, destination[20]);
    }

    [Fact]
    public void TheModuloCopyKernel_StaysOnTheGpuWhenItIsLargerThanATablePatch()
    {
        var guest = Guest(elementCount: 20000, period: 4, [1, 2, 3, 4]);
        var cache = new ShaderPipelineCache(guest.Context, guest.Host, guest.Compiler, guest.Registry);

        var result = cache.GetComputeProgram(Registers(4, 20000), new ShaderInterfaceRegisters(), 0x8001, 400, 1, 1);

        Assert.False(result.Consumed);
        Assert.Equal(Untouched, Destination(guest, 1)[0]);
    }

    [Theory]
    [InlineData(SourceAddress)]
    [InlineData(ControlAddress)]
    [InlineData(DestinationAddress + 8)]
    public void TheModuloCopyKernel_StaysOnTheGpuWhileTheGpuOwnsAnInput(ulong owned)
    {
        var guest = Guest(elementCount: 16, period: 4, [1, 2, 3, 4]);
        guest.Host.GpuOwned = (owned, sizeof(uint));
        var cache = new ShaderPipelineCache(guest.Context, guest.Host, guest.Compiler, guest.Registry);

        var result = cache.GetComputeProgram(Registers(4, 64), new ShaderInterfaceRegisters(), 0x8001, 1, 1, 1);

        Assert.False(result.Consumed);
        guest.Host.GpuOwned = default;
        Assert.Equal(Untouched, Destination(guest, 1)[0]);
    }

    [Fact]
    public void AKernelThatDiffersInOneWord_IsNotReplaced()
    {
        var guest = Guest(elementCount: 16, period: 4, [1, 2, 3, 4]);
        var changed = (uint[])ModuloKernel.Clone();
        changed[^6] = 0xE0002000 ^ 0x1000;
        guest.RegisterProgram(CodeAddress, HeaderAddress, changed);
        var cache = new ShaderPipelineCache(guest.Context, guest.Host, guest.Compiler, guest.Registry);

        var result = cache.GetComputeProgram(Registers(4, 64), new ShaderInterfaceRegisters(), 0x8001, 1, 1, 1);

        Assert.False(result.Consumed);
        Assert.Equal(Untouched, Destination(guest, 1)[0]);
    }

    [Theory]
    [InlineData(false, 0u)]
    [InlineData(false, 3u)]
    [InlineData(true, 0u)]
    [InlineData(true, 3u)]
    [InlineData(true, 200u)]
    public void ExpandCopyKernel_SelectsTheSourceElement(bool modulo, uint parameter)
    {
        uint[] source = [10, 11, 12, 13, 14];
        var output = new uint[12];

        ShaderPipelineCache.ExpandCopyKernel(source, output, parameter, modulo);

        for (var index = 0u; index < output.Length; index++)
        {
            var sourceIndex = modulo ? (parameter == 0 ? index : index % parameter) : index & parameter;
            Assert.Equal(sourceIndex < source.Length ? source[sourceIndex] : 0u, output[index]);
        }
    }
}
