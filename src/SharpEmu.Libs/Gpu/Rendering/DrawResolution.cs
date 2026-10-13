// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.Images;

namespace SharpEmu.Libs.Gpu.Rendering;

internal sealed class DrawResolution
{
    public ulong TargetContextStamp;
    public ulong TargetProgramsStamp;
    public uint TargetVersion;
    public int ColorCount;
    public readonly ColorTargetResolution[] Colors = new ColorTargetResolution[RenderingState.ColorAttachmentCapacity];
    public readonly uint[] ColorSlots = new uint[RenderingState.ColorAttachmentCapacity];
    public DepthTargetState? Depth;
    public bool PixelActive;
}
