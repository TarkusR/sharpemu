// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using static SharpEmu.Libs.Gpu.GpuCommands.Registers.ContextRegisterOffset;
using static SharpEmu.Libs.Gpu.GpuCommands.Registers.ShaderRegisterOffset;

namespace SharpEmu.Libs.Gpu.GpuCommands.Registers;

public enum RegisterStamp
{
    Context,
    ContextDynamic,
    ShaderPrograms,
    VertexUserData,
    PixelUserData,
    ComputeUserData,
    UserConfig,
}

public static class RegisterStamps
{
    public const int Count = 7;
    public const uint ContextBits = (1u << (int)RegisterStamp.Context) | (1u << (int)RegisterStamp.ContextDynamic);
    public const uint UserConfigBits = 1u << (int)RegisterStamp.UserConfig;
    public const uint AllBits = (1u << Count) - 1;

    private const uint ClipRectangleRegisterCount = 8;
    private const uint GraphicsUserDataCount = 32;
    private const uint ComputeUserDataCount = 16;

    private static readonly bool[] _dynamicContext = BuildDynamicContext();

    public static bool IsDynamicContext(uint offset) => offset < _dynamicContext.Length && _dynamicContext[offset];

    public static uint ContextBitsFor(uint offset, uint count)
    {
        var bits = 0u;
        for (var index = 0u; index < count; index++)
        {
            bits |= 1u << (int)(IsDynamicContext(offset + index) ? RegisterStamp.ContextDynamic : RegisterStamp.Context);
        }

        return bits;
    }

    public static uint ShaderBitsFor(uint offset, uint count)
    {
        var bits = 0u;
        for (var index = 0u; index < count; index++)
        {
            bits |= 1u << (int)ShaderStampOf(offset + index);
        }

        return bits;
    }

    public static RegisterStamp ShaderStampOf(uint offset)
    {
        if (offset - SpiShaderUserDataPs0 < GraphicsUserDataCount)
        {
            return RegisterStamp.PixelUserData;
        }

        if (offset - SpiShaderUserDataGs0 < GraphicsUserDataCount)
        {
            return RegisterStamp.VertexUserData;
        }

        return offset - ComputeUserData0 < ComputeUserDataCount ? RegisterStamp.ComputeUserData : RegisterStamp.ShaderPrograms;
    }

    private static bool[] BuildDynamicContext()
    {
        var dynamic = new bool[RegisterBankLayout.ContextRegisterCount];
        void Range(uint first, uint last)
        {
            for (var offset = first; offset <= last; offset++)
            {
                dynamic[offset] = true;
            }
        }

        Range(PaScScreenScissorTl, PaScScreenScissorBr);
        Range(PaScWindowOffset, PaScWindowScissorBr);
        Range(PaScClipRectRule, PaScClipRect0Tl + ClipRectangleRegisterCount - 1);
        Range(PaScGenericScissorTl, PaScGenericScissorBr);
        Range(PaScViewportScissor0Tl, PaScViewportScissor15Br);
        Range(PaScViewportZMin0, PaScViewportZMax15);
        Range(PaClViewportXScale, PaClViewportZOffset15);
        Range(CbBlendRed, CbBlendAlpha);
        Range(PaSuLineCntl, PaSuLineCntl);
        Range(PaSuPolyOffsetDbFmtCntl, PaSuPolyOffsetBackOffset);
        return dynamic;
    }
}

[System.Runtime.CompilerServices.InlineArray(RegisterStamps.Count)]
public struct RegisterStampSet
{
    private ulong _first;
}
