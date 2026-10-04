// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Buffers.Binary;
using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.ShaderCompiler;

namespace SharpEmu.Libs.Gpu.Pipelines;

// The draw and dispatch recognisers kept beside the ported cache: solid clears, fills and one copy kernel.
internal sealed partial class ShaderPipelineCache
{
    // A float32x3 vertex position stream.
    private const uint PositionDataFormat = 13;
    private const uint PositionNumberFormat = 7;
    private const uint PositionExportTarget = 12;
    private const uint ColorExportTarget = 0;
    private const int WorkGroupRegisterOfCopyKernel = 12;
    private const uint MaxCopyKernelSourceBytes = 16 * 1024 * 1024;
    private const uint MaxModuloCopyBytes = 16 * 1024;
    private const uint MaxWideCopyBytes = 4 * 1024 * 1024;
    private const uint PooledCopyBytes = 64 * 1024;

    internal static bool PredictMaskedCopyEnabled { get; set; } = true;

    internal static bool CpuModuloCopyEnabled { get; set; }

    private enum CopyKernel { None, Masked, Modulo }

    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<Gen5ShaderProgram, System.Runtime.CompilerServices.StrongBox<CopyKernel>> _copyKernels = new();
    private static long _reportedMaskedCopies;
    private static long _reportedModuloCopies;
    private static long _reportedRefusedCopies;
    private static long _reportedPredictedCopies;

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, long> _copyRefusals = new();

    private static bool RefuseCopyKernel(string reason)
    {
        Interlocked.Increment(ref _reportedRefusedCopies);
        _copyRefusals.AddOrUpdate(reason, 1, static (_, count) => count + 1);
        return false;
    }

    private static readonly string[] MaskedCopyKernelOpcodes =
    [
        "SMovB32", "STtraceData", "SInstPrefetch", "VLshlAddU32", "SBufferLoadDword", "SWaitcnt", "VCmpxGtU32", "SCbranchExecz",
        "SBufferLoadDword", "SWaitcnt", "VAndB32", "BufferLoadFormatX", "SWaitcnt", "BufferStoreFormatX", "SEndpgm",
    ];

    private static readonly uint[] ModuloCopyKernelWords =
    [
        0xBF960000, 0xBFA00003, 0xD7460002, 0x04010C0C, 0xF4201A84, 0xFA000000, 0xBF8CC07F, 0x7DA8046A, 0xBF88002A, 0xF4200304,
        0xFA000004, 0xBF8CC07F, 0x7E000C0C, 0xBF070C80, 0x8588807E, 0x7E005700, 0x100000FF, 0x4F800000, 0x7E060F00, 0xD5766A00,
        0x0202060C, 0x7D8A0280, 0x4C020080, 0x02000101, 0xD56A0001, 0x00020700, 0x4C000303, 0x4A020303, 0x02000101, 0xD56A0000,
        0x00020500, 0xD5690001, 0x0002000C, 0x4C060302, 0x7D8C02F9, 0x06068A02, 0x7D86060C, 0x87EA6A0A, 0x50000080, 0xD5286A00,
        0x002A00C1, 0xD5010000, 0x002200C1, 0xD5690000, 0x0002000C, 0x4C000102, 0xE0002000, 0x80000000, 0xBF8C3F70, 0xE0102000,
        0x80010002, 0xBF810000,
    ];

    private static long _predictedTo4K;
    private static long _predictedTo64K;
    private static long _predictedTo1M;
    private static long _tooLargeTo4M;
    private static long _tooLargeTo16M;
    private static long _tooLargeBeyond;

    private static string TakeCopySizes() => FormattableString.Invariant(
        $" predicted_sizes=4k:{Interlocked.Exchange(ref _predictedTo4K, 0)},64k:{Interlocked.Exchange(ref _predictedTo64K, 0)},1m:{Interlocked.Exchange(ref _predictedTo1M, 0)} too_large=4m:{Interlocked.Exchange(ref _tooLargeTo4M, 0)},16m:{Interlocked.Exchange(ref _tooLargeTo16M, 0)},more:{Interlocked.Exchange(ref _tooLargeBeyond, 0)}");

    public static string TakeCopyKernelReport() => TakeCopyKernelCounters() + TakeCopySizes();

    private static string TakeCopyKernelCounters() => FormattableString.Invariant(
        $"[PERF][COPY_KERNEL] masked={Interlocked.Exchange(ref _reportedMaskedCopies, 0)} modulo={Interlocked.Exchange(ref _reportedModuloCopies, 0)} predicted={Interlocked.Exchange(ref _reportedPredictedCopies, 0)} refused={Interlocked.Exchange(ref _reportedRefusedCopies, 0)} reasons={TakeCopyRefusals()}");

    private static string TakeCopyRefusals()
    {
        var reasons = _copyRefusals.ToArray();
        _copyRefusals.Clear();
        return reasons.Length == 0
            ? "none"
            : string.Join(",", reasons.OrderByDescending(static pair => pair.Value).Select(static pair => $"{pair.Key.Replace(' ', '_')}:{pair.Value}"));
    }

    private static CopyKernel ClassifyCopyKernel(Gen5ShaderProgram program)
    {
        var instructions = program.Instructions;
        if (instructions.Count == MaskedCopyKernelOpcodes.Length)
        {
            for (var index = 0; index < instructions.Count; index++)
            {
                if (instructions[index].Opcode != MaskedCopyKernelOpcodes[index])
                {
                    return CopyKernel.None;
                }
            }

            return IsExactMaskedDwordCopyInstructionShape(instructions) ? CopyKernel.Masked : CopyKernel.None;
        }

        if (instructions.Count < 2 || instructions[0].Opcode != "SMovB32")
        {
            return CopyKernel.None;
        }

        var word = 0;
        for (var index = 1; index < instructions.Count; index++)
        {
            foreach (var encoded in instructions[index].Words)
            {
                if (word == ModuloCopyKernelWords.Length || encoded != ModuloCopyKernelWords[word++])
                {
                    return CopyKernel.None;
                }
            }
        }

        return word == ModuloCopyKernelWords.Length ? CopyKernel.Modulo : CopyKernel.None;
    }

    internal static void ExpandCopyKernel(ReadOnlySpan<uint> source, Span<uint> output, uint parameter, bool modulo)
    {
        if (!modulo && parameter == 0)
        {
            output.Fill(source[0]);
            return;
        }

        var identity = modulo
            ? parameter == 0 || parameter >= (uint)output.Length
            : (parameter & (parameter + 1)) == 0 && parameter >= (uint)(output.Length - 1);
        if (identity && source.Length >= output.Length)
        {
            source[..output.Length].CopyTo(output);
            return;
        }

        for (var index = 0; index < output.Length; index++)
        {
            var sourceIndex = modulo ? (parameter == 0 ? (uint)index : (uint)index % parameter) : (uint)index & parameter;
            output[index] = sourceIndex < (uint)source.Length ? source[(int)sourceIndex] : 0;
        }
    }

    private static readonly bool _premultipliedFillClearEnabled =
        !string.Equals(Environment.GetEnvironmentVariable("SHARPEMU_DISABLE_FILL_CLEAR"), "1", StringComparison.Ordinal);

    // A procedural fullscreen vertex program paired with a constant-color pixel program.
    private static bool IsProceduralFullscreenClearPair(
        Gen5ShaderProgram vertexProgram,
        VertexInputInfo vertexInfo,
        ShaderStageResources vertexStage,
        Gen5ShaderProgram pixelProgram,
        ShaderStageResources pixelStage)
    {
        var vertexResources = vertexStage.Program?.Resources?.Info;
        var pixelResources = pixelStage.Program?.Resources?.Info;
        if (vertexResources is null || pixelResources is null || vertexInfo.Attributes.Length != 0 ||
            vertexResources.Images.Count != 0 || pixelResources.Images.Count != 0 ||
            vertexResources.Buffers.Count != 0 || pixelResources.Buffers.Count != 0)
        {
            return false;
        }

        if (!HasExportTarget(vertexProgram, PositionExportTarget) || !HasExportTarget(pixelProgram, ColorExportTarget))
        {
            return false;
        }

        if (pixelProgram.Instructions.Count is 0 or > 8 || vertexProgram.Instructions.Count is 0 or > 48)
        {
            return false;
        }

        return pixelProgram.Instructions.All(IsBenignClearPixelInstruction) &&
               vertexProgram.Instructions.All(IsBenignProceduralVertexInstruction);
    }

    private static bool HasExportTarget(Gen5ShaderProgram program, uint target) =>
        program.Instructions.Any(instruction => instruction.Control is Gen5ExportControl export && export.Target == target);

    private static bool IsBenignClearPixelInstruction(Gen5ShaderInstruction instruction) =>
        instruction.Opcode is "SNop" or "SWaitcnt" or "SInstPrefetch" or "SEndpgm" or "VMovB32" ||
        instruction.Control is Gen5ExportControl { Target: ColorExportTarget };

    private static bool IsBenignProceduralVertexInstruction(Gen5ShaderInstruction instruction)
    {
        if (instruction.Control is Gen5BufferMemoryControl or Gen5ImageControl or Gen5GlobalMemoryControl or Gen5ScalarMemoryControl)
        {
            return false;
        }

        if (instruction.Control is Gen5ExportControl export)
        {
            // Position plus the primitive and parameter exports.
            return export.Target is PositionExportTarget or (>= 13 and < 32) or 20;
        }

        return instruction.Opcode is
            "SNop" or "SWaitcnt" or "SInstPrefetch" or "SEndpgm" or "SSendmsg" or
            "VMovB32" or "VAndB32" or "VAddI32" or "VLshlrevB32" or "VCvtF32I32" or "VCvtF32U32" ||
            instruction.Encoding is Gen5ShaderEncoding.Sop1 or Gen5ShaderEncoding.Sop2 or Gen5ShaderEncoding.Sopc or Gen5ShaderEncoding.Sopk or Gen5ShaderEncoding.Sopp;
    }

    // Opaque white unless the first user register carries a finite color in [0, 4].
    private static (float Red, float Green, float Blue, float Alpha) DecodeSolidClearColor(IReadOnlyList<uint> pixelUserData)
    {
        float red = 1f, green = 1f, blue = 1f, alpha = 1f;
        if (pixelUserData.Count > 0)
        {
            var bits = pixelUserData[0];
            if (bits != 0)
            {
                red = green = blue = alpha = BitConverter.UInt32BitsToSingle(bits);
                if (!float.IsFinite(red) || red < 0f || red > 4f)
                {
                    red = green = blue = alpha = 1f;
                }
            }
        }

        return (red, green, blue, alpha);
    }

    // An untextured transparent fill through premultiplied blending overwrites its targets.
    private static bool IsTransparentPremultipliedFill(
        ContextRegisters context,
        Gen5PixelOutputBinding[] outputs,
        VertexInputInfo vertexInfo,
        ShaderStageResources vertexStage,
        ShaderStageResources pixelStage)
    {
        var vertexResources = vertexStage.Program?.Resources?.Info;
        var pixelResources = pixelStage.Program?.Resources?.Info;
        var pixelUserData = pixelStage.Resources.UserData;
        if (!_premultipliedFillClearEnabled || vertexResources is null || pixelResources is null ||
            vertexResources.Images.Count != 0 || pixelResources.Images.Count != 0 ||
            vertexInfo.Attributes.Length != 0 || pixelUserData.Length < 4 || outputs.Length == 0)
        {
            return false;
        }

        foreach (var output in outputs)
        {
            var blend = context.BlendControls[output.GuestSlot];
            if (!blend.Enable || blend.ColorSourceFactor != 1 || blend.ColorDestinationFactor != 5 || blend.ColorFunction != 0)
            {
                return false;
            }
        }

        for (var index = 0; index < 4; index++)
        {
            if ((pixelUserData[index] & 0x7FFF_FFFFu) != 0)
            {
                return false;
            }
        }

        return true;
    }

    // Replaces only the exact masked-copy kernel shape; its writes land in command order over the guest range.
    private bool TrySubmitMaskedDwordCopyKernel(
        Gen5ShaderProgram program,
        ShaderSource source,
        Gen5ComputeSystemRegisters systemRegisters,
        ComputeInputInfo input,
        uint dimensionX,
        uint dimensionY,
        uint dimensionZ,
        out string description,
        out KnownGpuWrite? knownWrite)
    {
        description = string.Empty;
        knownWrite = null;
        var kernel = _copyKernels.GetValue(program, static decoded =>
            new System.Runtime.CompilerServices.StrongBox<CopyKernel>(ClassifyCopyKernel(decoded))).Value;
        if (kernel == CopyKernel.None)
        {
            return false;
        }

        var modulo = kernel == CopyKernel.Modulo;
        var predict = modulo ? !CpuModuloCopyEnabled : !input.DispatchThreadDimensions;
        var wide = PredictMaskedCopyEnabled;
        if (predict && !modulo && !wide)
        {
            return RefuseCopyKernel("masked outside thread dimension mode");
        }

        var guarded = modulo || predict;
        var sizeX = dimensionX;
        var sizeY = dimensionY;
        var sizeZ = dimensionZ;
        var groupsX = input.DispatchThreadDimensions ? RenderExecutor.GroupsFromThreads(sizeX, input.ThreadsX) : sizeX;
        var groupsY = input.DispatchThreadDimensions ? RenderExecutor.GroupsFromThreads(sizeY, input.ThreadsY) : sizeY;
        var groupsZ = input.DispatchThreadDimensions ? RenderExecutor.GroupsFromThreads(sizeZ, input.ThreadsZ) : sizeZ;
        if (groupsY != 1 || groupsZ != 1 ||
            input.ThreadsX != 64 || input.ThreadsY != 1 || input.ThreadsZ != 1 ||
            systemRegisters.WorkGroupXRegister != WorkGroupRegisterOfCopyKernel)
        {
            return RefuseCopyKernel("dispatch shape");
        }

        var userData = source.UserData;
        BufferDescriptorWords controlWords = default;
        if (userData.Length < 12 ||
            !IsExactMaskedDwordCopyDescriptor(userData, 0, out var sourceWords) ||
            !IsExactMaskedDwordCopyDescriptor(userData, 4, out var destinationWords) ||
            (guarded
                ? (controlWords = BufferDescriptorWords.From(userData.AsSpan(8, 4))).SwizzleEnabled || controlWords.AddThreadId
                : !IsExactMaskedDwordCopyDescriptor(userData, 8, out controlWords)))
        {
            return RefuseCopyKernel("descriptors");
        }

        var controlAddress = controlWords.Address;
        var sourceAddress = sourceWords.Address;
        var destinationAddress = destinationWords.Address;
        var destinationBytes = destinationWords.Footprint() ?? 0;
        var sourceBytes = sourceWords.Footprint() ?? 0;
        var controlBytes = controlWords.Footprint() ?? 0;
        if (controlAddress == 0 || sourceAddress == 0 || destinationAddress == 0 ||
            controlBytes < 2 * sizeof(uint) || sourceBytes < sizeof(uint) || destinationBytes < sizeof(uint))
        {
            return false;
        }

        Span<byte> control = stackalloc byte[2 * sizeof(uint)];
        if (guarded ? !TryReadKernelInput(controlAddress, control) : !_context.Memory.TryRead(controlAddress, control))
        {
            return RefuseCopyKernel("control owned by the GPU");
        }

        var elementCount = BinaryPrimitives.ReadUInt32LittleEndian(control);
        var sourceMask = BinaryPrimitives.ReadUInt32LittleEndian(control[sizeof(uint)..]);
        var dispatchedThreads = modulo && input.DispatchThreadDimensions
            ? sizeX
            : Math.Min((ulong)uint.MaxValue, (ulong)groupsX * input.ThreadsX);
        var writableDwords = (uint)Math.Min(destinationBytes / sizeof(uint), uint.MaxValue);
        var outputDwords = (uint)Math.Min(Math.Min((ulong)elementCount, dispatchedThreads), writableDwords);
        if (outputDwords == 0)
        {
            return false;
        }

        var outputBytes = (ulong)outputDwords * sizeof(uint);
        if (guarded && outputBytes > (!wide ? MaxModuloCopyBytes : predict ? MaxWideCopyBytes : PooledCopyBytes))
        {
            Interlocked.Increment(ref outputBytes <= 4 * 1024 * 1024 ? ref _tooLargeTo4M : ref outputBytes <= 16 * 1024 * 1024 ? ref _tooLargeTo16M : ref _tooLargeBeyond);
            return RefuseCopyKernel("larger than a table patch");
        }

        if (modulo && !predict && _host.CpuWriteNeedsGpuData(destinationAddress, (ulong)outputDwords * sizeof(uint)))
        {
            return RefuseCopyKernel("destination owned by the GPU");
        }

        var neededSourceDwords = modulo
            ? (sourceMask == 0 ? outputDwords : Math.Min(sourceMask, outputDwords))
            : (sourceMask == 0 ? 1UL : Math.Min((ulong)sourceMask + 1, outputDwords));
        if (modulo && neededSourceDwords * sizeof(uint) > MaxCopyKernelSourceBytes)
        {
            return RefuseCopyKernel("source too large");
        }

        var readableSourceBytes = Math.Min(Math.Min(sourceBytes, MaxCopyKernelSourceBytes), neededSourceDwords * sizeof(uint));
        var sourceLength = (int)(readableSourceBytes - readableSourceBytes % sizeof(uint));
        var pooledSource = predict && sourceLength > PooledCopyBytes;
        var sourceArray = pooledSource ? System.Buffers.ArrayPool<byte>.Shared.Rent(sourceLength) : new byte[sourceLength];
        var sourceData = sourceArray.AsSpan(0, sourceLength);
        if (sourceLength < sizeof(uint) ||
            (guarded ? !TryReadKernelInput(sourceAddress, sourceData) : !_context.Memory.TryRead(sourceAddress, sourceData)))
        {
            if (pooledSource)
            {
                System.Buffers.ArrayPool<byte>.Shared.Return(sourceArray);
            }

            return RefuseCopyKernel("source owned by the GPU");
        }

        if (predict)
        {
            var pooledWords = outputBytes > PooledCopyBytes;
            var words = pooledWords ? System.Buffers.ArrayPool<uint>.Shared.Rent((int)outputDwords) : new uint[outputDwords];
            ExpandCopyKernel(System.Runtime.InteropServices.MemoryMarshal.Cast<byte, uint>(sourceData), words.AsSpan(0, (int)outputDwords), sourceMask, modulo);
            if (pooledSource)
            {
                System.Buffers.ArrayPool<byte>.Shared.Return(sourceArray);
            }

            knownWrite = new KnownGpuWrite(destinationAddress, words, (int)outputDwords, pooledWords);
            Interlocked.Increment(ref _reportedPredictedCopies);
            Interlocked.Increment(ref outputBytes <= 4096 ? ref _predictedTo4K : ref outputBytes <= PooledCopyBytes ? ref _predictedTo64K : ref _predictedTo1M);
            return false;
        }

        var output = new byte[checked((int)outputDwords * sizeof(uint))];
        ExpandCopyKernel(
            System.Runtime.InteropServices.MemoryMarshal.Cast<byte, uint>(sourceData),
            System.Runtime.InteropServices.MemoryMarshal.Cast<byte, uint>(output.AsSpan()),
            sourceMask,
            modulo);

        // The dispatch runs in stream order on the worker, so the replacement writes at once.
        if (!_context.Memory.TryWrite(destinationAddress, output))
        {
            Console.Error.WriteLine($"[LOADER][ERROR] AGC masked-copy fast path failed dst=0x{destinationAddress:X16} bytes={output.Length}");
            return false;
        }

        Interlocked.Increment(ref modulo ? ref _reportedModuloCopies : ref _reportedMaskedCopies);
        if (RenderTrace.Enabled)
        {
            description = $"dst=0x{destinationAddress:X16} bytes={output.Length} elements={elementCount} {(modulo ? "period" : "mask")}=0x{sourceMask:X8} dispatch={groupsX}x{input.ThreadsX}";
        }

        return true;
    }

    private bool TryReadKernelInput(ulong address, Span<byte> destination) =>
        _host.TryReadResidentGuestBytes(address, destination, clean: false);

    private static bool IsExactMaskedDwordCopyInstructionShape(IReadOnlyList<Gen5ShaderInstruction> instructions)
    {
        static bool IsOperand(Gen5Operand operand, Gen5OperandKind kind, uint value) => operand.Kind == kind && operand.Value == value;

        static bool IsBufferControl(Gen5ShaderInstruction instruction, uint vectorAddress, uint vectorData, uint scalarResource) =>
            instruction.Control is Gen5BufferMemoryControl { DwordCount: 1, OffsetBytes: 0, IndexEnabled: true, OffsetEnabled: false } control &&
            control.VectorAddress == vectorAddress && control.VectorData == vectorData && control.ScalarResource == scalarResource;

        static bool IsScalarLoad(Gen5ShaderInstruction instruction, int offsetBytes) =>
            instruction.Control is Gen5ScalarMemoryControl { DestinationCount: 1, DynamicOffsetRegister: null } control &&
            control.ImmediateOffsetBytes == offsetBytes &&
            instruction.Destinations.Count == 1 && IsOperand(instruction.Destinations[0], Gen5OperandKind.ScalarRegister, 106) &&
            instruction.Sources.Count >= 1 && IsOperand(instruction.Sources[0], Gen5OperandKind.ScalarRegister, 8);

        // Require matching operands and opcodes; either change can alter written lanes or addresses.
        var globalId = instructions[3];
        var compare = instructions[6];
        var sourceIndex = instructions[10];
        var load = instructions[11];
        var store = instructions[13];
        return
            globalId.Destinations.Count == 1 &&
            IsOperand(globalId.Destinations[0], Gen5OperandKind.VectorRegister, 0) &&
            globalId.Sources.Count == 3 &&
            IsOperand(globalId.Sources[0], Gen5OperandKind.ScalarRegister, 12) &&
            IsOperand(globalId.Sources[1], Gen5OperandKind.EncodedConstant, 134) &&
            IsOperand(globalId.Sources[2], Gen5OperandKind.VectorRegister, 0) &&
            IsScalarLoad(instructions[4], offsetBytes: 0) &&
            compare.Sources.Count == 2 &&
            IsOperand(compare.Sources[0], Gen5OperandKind.ScalarRegister, 106) &&
            IsOperand(compare.Sources[1], Gen5OperandKind.VectorRegister, 0) &&
            instructions[7].Words.Count == 1 &&
            (instructions[7].Words[0] & 0xFFFFu) == 9 &&
            IsScalarLoad(instructions[8], offsetBytes: sizeof(uint)) &&
            sourceIndex.Destinations.Count == 1 &&
            IsOperand(sourceIndex.Destinations[0], Gen5OperandKind.VectorRegister, 1) &&
            sourceIndex.Sources.Count == 2 &&
            IsOperand(sourceIndex.Sources[0], Gen5OperandKind.ScalarRegister, 106) &&
            IsOperand(sourceIndex.Sources[1], Gen5OperandKind.VectorRegister, 0) &&
            IsBufferControl(load, vectorAddress: 1, vectorData: 1, scalarResource: 0) &&
            IsBufferControl(store, vectorAddress: 0, vectorData: 1, scalarResource: 4);
    }

    // Structured 32-bit records without thread-index or swizzle address changes.
    private static bool IsExactMaskedDwordCopyDescriptor(uint[] userData, int scalarBase, out BufferDescriptorWords descriptor)
    {
        descriptor = BufferDescriptorWords.From(userData.AsSpan(scalarBase, 4));
        var cacheSwizzle = (descriptor.Word1 & (1u << 30)) != 0;
        return descriptor.Stride == sizeof(uint) &&
               !cacheSwizzle &&
               !descriptor.SwizzleEnabled &&
               descriptor.Format == 20 &&
               !descriptor.AddThreadId &&
               descriptor.OutOfBounds == 0 &&
               descriptor.Type == 0 &&
               descriptor.DestinationSelectX == 4;
    }
}
