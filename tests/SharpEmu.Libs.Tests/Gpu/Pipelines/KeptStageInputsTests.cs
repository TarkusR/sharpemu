// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.Libs.Diagnostics;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Pipelines;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using Xunit;

namespace SharpEmu.Libs.Tests.Gpu.Pipelines;

[Collection(SchedulingStateCollection.Name)]
public sealed class KeptStageInputsTests : IDisposable
{
    private const ulong VertexAddress = PipelineTestGuest.MemoryBase + 0x1000;
    private const ulong VertexHeader = PipelineTestGuest.MemoryBase + 0x8000;
    private const ulong UserDataBlock = PipelineTestGuest.MemoryBase + 0x9000;
    private const ulong DirectOffsets = PipelineTestGuest.MemoryBase + 0x9100;
    private const ulong Semantics = PipelineTestGuest.MemoryBase + 0x9200;
    private const ulong AttributeTable = PipelineTestGuest.MemoryBase + 0xA000;
    private const ulong BufferTable = PipelineTestGuest.MemoryBase + 0xA100;
    private const ulong MovedAttributeTable = PipelineTestGuest.MemoryBase + 0xB000;
    private const ulong MovedBufferTable = PipelineTestGuest.MemoryBase + 0xB100;
    private const ulong StreamBase = PipelineTestGuest.MemoryBase + 0x20000;
    private const ulong PixelAddress = PipelineTestGuest.MemoryBase + 0x2000;
    private const ulong PixelHeader = PipelineTestGuest.MemoryBase + 0x8100;
    private const ulong PixelUserDataBlock = PipelineTestGuest.MemoryBase + 0xC000;
    private const uint DirectResourceCount = 11;
    private const uint VertexBufferTableType = 8;
    private const uint VertexAttributeTableType = 10;
    private const uint AttributeRegister = 0;
    private const uint BufferRegister = 2;
    private const uint Format32x4Float = 77;
    private const ulong ShaderSizeOffset = 0x44;

    private readonly FatalScope _fatal = new();
    private readonly PipelineTestGuest _guest = new();
    private readonly ShaderPipelineCache _cache;
    private readonly RegisterBanks _banks = new(static message => new InvalidOperationException(message));

    public KeptStageInputsTests()
    {
        _cache = new ShaderPipelineCache(_guest.Context, _guest.Host, _guest.Compiler, _guest.Registry);
        WriteQword(UserDataBlock, DirectOffsets);
        WriteHalf(UserDataBlock + 0x2C, (ushort)DirectResourceCount);
        for (uint type = 0; type < DirectResourceCount; type++)
        {
            WriteHalf(DirectOffsets + type * 2, 0xFFFF);
        }

        WriteHalf(DirectOffsets + VertexBufferTableType * 2, (ushort)BufferRegister);
        WriteHalf(DirectOffsets + VertexAttributeTableType * 2, (ushort)AttributeRegister);
        _guest.WriteWords(Semantics, 0 | (4u << 16), 1 | (4u << 8) | (2u << 16));
        _guest.RegisterProgram(VertexAddress, VertexHeader, PipelineTestGuest.EndProgram, UserDataBlock, Semantics, 2);
        WriteTables(AttributeTable, BufferTable, stride: 24);
        _guest.RegisterProgram(PixelAddress, PixelHeader, PipelineTestGuest.EndProgram, PixelUserDataBlock);
        _banks.Shader.Vertex.ExportAddress = VertexAddress;
        _banks.Shader.Pixel.Address = PixelAddress;
        PointAtTables(AttributeTable, BufferTable);
    }

    public void Dispose() => _fatal.Dispose();

    private void WriteQword(ulong address, ulong value)
    {
        var bytes = new byte[sizeof(ulong)];
        BinaryPrimitives.WriteUInt64LittleEndian(bytes, value);
        _guest.Write(address, bytes);
    }

    private void WriteHalf(ulong address, ushort value)
    {
        var bytes = new byte[sizeof(ushort)];
        BinaryPrimitives.WriteUInt16LittleEndian(bytes, value);
        _guest.Write(address, bytes);
    }

    private void WriteTables(ulong attributeTable, ulong bufferTable, uint stride)
    {
        _guest.WriteWords(attributeTable, 0, 16u << 14);
        _guest.WriteWords(bufferTable, PipelineTestGuest.BufferDescriptor(StreamBase, stride, 3, Format32x4Float));
    }

    private void PointAtTables(ulong attributeTable, ulong bufferTable)
    {
        var registers = _banks.Shader.Vertex.GeometryUserScalars;
        registers.Set(AttributeRegister, (uint)attributeTable, UserScalarKind.Unknown);
        registers.Set(AttributeRegister + 1, (uint)(attributeTable >> 32), UserScalarKind.Unknown);
        registers.Set(BufferRegister, (uint)bufferTable, UserScalarKind.Unknown);
        registers.Set(BufferRegister + 1, (uint)(bufferTable >> 32), UserScalarKind.Unknown);
        _banks.MarkWritten(1u << (int)RegisterStamp.VertexUserData);
    }

    private VertexInputInfo Lookup()
    {
        var programs = _cache.GetGraphicsPrograms(_banks, [], pixelActive: false, depthBound: false);
        Assert.True(programs.Available);
        return programs.VertexInput;
    }

    private PixelInputInfo PixelLookup()
    {
        var programs = _cache.GetGraphicsPrograms(_banks, [], pixelActive: true, depthBound: false);
        Assert.True(programs.Available);
        return programs.PixelInput;
    }

    private void SetKillEnable(bool enabled)
    {
        var shaderInterface = _banks.Context.ShaderInterface;
        var control = shaderInterface.DepthShaderControl;
        control.KillEnable = enabled;
        shaderInterface.DepthShaderControl = control;
        _banks.MarkWritten(1u << (int)RegisterStamp.Context);
    }

    [Fact]
    public void VertexInput_IsKeptWhileItsTablesHoldTheSameWords()
    {
        var first = Lookup();
        var second = Lookup();

        Assert.Equal(2, first.Attributes.Length);
        Assert.Equal(24u, Assert.Single(first.Buffers).Stride);
        Assert.NotSame(first, second);
        Assert.Same(first.Attributes, second.Attributes);
        Assert.Same(first.Buffers, second.Buffers);
        Assert.True(second.Stage.IsValid);
    }

    [Fact]
    public void VertexInput_IsResolvedAgainWhenATableWordChanges()
    {
        var first = Lookup();
        WriteTables(AttributeTable, BufferTable, stride: 32);
        var changedBuffer = Lookup();
        _guest.WriteWords(AttributeTable + 4, 8u << 14);
        var changedAttribute = Lookup();

        Assert.NotSame(first.Buffers, changedBuffer.Buffers);
        Assert.Equal(32u, Assert.Single(changedBuffer.Buffers).Stride);
        Assert.NotSame(changedBuffer.Attributes, changedAttribute.Attributes);
        Assert.Equal(8u, changedAttribute.Attributes[1].OffsetBytes);
    }

    [Fact]
    public void VertexInput_IsKeptWhenTheSameTablesMove()
    {
        var first = Lookup();
        WriteTables(MovedAttributeTable, MovedBufferTable, stride: 24);
        PointAtTables(MovedAttributeTable, MovedBufferTable);
        var moved = Lookup();

        Assert.Same(first.Attributes, moved.Attributes);
        Assert.Same(first.Buffers, moved.Buffers);
    }

    [Fact]
    public void VertexInput_IsResolvedAgainWhenTheClipTransformChanges()
    {
        var first = Lookup();
        _banks.Context.Clip.ClipDisable = true;
        _banks.Context.ScreenViewport.Viewports[0] = new ViewportRegisters { XScale = 32, XOffset = 32, YScale = 32, YOffset = 32 };
        var clipped = Lookup();

        Assert.NotSame(first.Attributes, clipped.Attributes);
        Assert.True(clipped.ClipSpace.Enabled);
        Assert.Equal(32f, clipped.ClipSpace.ScaleX);
    }

    [Fact]
    public void UserData_IsKeptUntilARegisterHoldsAnotherWord()
    {
        Lookup();
        var kept = (KeptUserData)typeof(ShaderPipelineCache).GetField("_vertexUserData", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!.GetValue(_cache)!;
        var first = kept.Words;
        Lookup();
        Assert.Same(first, kept.Words);

        _banks.MarkWritten(1u << (int)RegisterStamp.VertexUserData);
        Lookup();
        Assert.Same(first, kept.Words);

        _banks.Shader.Vertex.GeometryUserScalars.Set(4, 0x1234, UserScalarKind.Unknown);
        _banks.MarkWritten(1u << (int)RegisterStamp.VertexUserData);
        Lookup();
        Assert.NotSame(first, kept.Words);
        Assert.Equal(5, kept.Words!.Length);
        Assert.Equal(0x1234u, kept.Words[4]);
    }

    [Fact]
    public void Header_IsReadAgainOnlyAfterARegistration()
    {
        var first = _guest.Registry.Require(VertexAddress, "vertex");
        var size = new byte[sizeof(uint)];
        BinaryPrimitives.WriteUInt32LittleEndian(size, 8);
        _guest.Write(VertexHeader + ShaderSizeOffset, size);

        Assert.Same(first, _guest.Registry.Require(VertexAddress, "vertex"));
        ShaderHeaderRegistry.NoteRegistration();
        Assert.Equal(8u, _guest.Registry.Require(VertexAddress, "vertex").CodeSizeBytes);
    }
}
