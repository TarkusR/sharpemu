// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Collections.Generic;

namespace SharpEmu.ShaderCompiler.Resources;

internal sealed class RawReadGroup
{
    public required ScalarValue Handle { get; init; }
    public required ScalarValueKind Kind { get; init; }
    public required ScalarValue[] Reads { get; init; }
    public required long[] Immediates { get; init; }
    public required uint[] Offsets { get; init; }

    public CompiledResourceEvaluator? IndexedFor;
    public int[] HandleIndices = [];
    public int[] ReadIndices = [];
}

internal sealed class RawReadPrefetchPlan
{
    public required RawReadGroup[] Table { get; init; }
    public required RawReadGroup[] CleanTable { get; init; }
    public required RawReadGroup[][] Sources { get; init; }
}

internal static class RawReadPrefetch
{
    private const int MinimumRunWords = 2;
    private const int MaximumRunBytes = 64 * 1024;

    [ThreadStatic]
    private static Scratch? _scratch;

    private static long _runs;
    private static long _words;
    private static long _refused;

    public static string TakeReport() => FormattableString.Invariant(
        $"prefetch_runs={Interlocked.Exchange(ref _runs, 0)} prefetch_words={Interlocked.Exchange(ref _words, 0)} prefetch_refused={Interlocked.Exchange(ref _refused, 0)}");

    private sealed class Scratch
    {
        public ulong[] Addresses = new ulong[256];
        public int[] Order = new int[256];
        public ScalarValue[] Values = new ScalarValue[256];
        public int[] Indices = new int[256];
        public byte[] Bytes = new byte[1024];
        public int Count;

        public void Add(ulong address, ScalarValue value, int index)
        {
            if (Count == Addresses.Length)
            {
                Array.Resize(ref Addresses, Count * 2);
                Array.Resize(ref Order, Count * 2);
                Array.Resize(ref Values, Count * 2);
                Array.Resize(ref Indices, Count * 2);
            }

            Addresses[Count] = address;
            Order[Count] = Count;
            Values[Count] = value;
            Indices[Count] = index;
            Count++;
        }
    }

    public static RawReadPrefetchPlan For(ShaderResourcePlan plan)
    {
        if (plan.RawReadPrefetch is { } existing)
        {
            return existing;
        }

        var table = new List<ScalarValue>();
        var cleanTable = new List<ScalarValue>();
        foreach (var read in plan.TableReads)
        {
            var clean = read.FlatOffset < plan.CleanFlatSlots.Count && plan.CleanFlatSlots[(int)read.FlatOffset] != 0;
            (clean ? cleanTable : table).Add(read.Value);
        }

        var sources = new RawReadGroup[plan.DescriptorSources.Count][];
        for (var index = 0; index < sources.Length; index++)
        {
            sources[index] = Group(plan, plan.DescriptorSources[index].Dwords);
        }

        var built = new RawReadPrefetchPlan
        {
            Table = Group(plan, table),
            CleanTable = Group(plan, cleanTable),
            Sources = sources,
        };
        plan.RawReadPrefetch = built;
        return built;
    }

    private static RawReadGroup[] Group(ShaderResourcePlan plan, IEnumerable<ScalarValue> values)
    {
        var groups = new Dictionary<(ScalarValue Handle, ScalarValueKind Kind), List<ScalarValue>>(HandleComparer.Instance);
        var seen = new HashSet<ScalarValue>(ReferenceEqualityComparer.Instance);
        foreach (var value in values)
        {
            if (value.Kind is not (ScalarValueKind.ScalarBufferWord or ScalarValueKind.ScalarAddressWord) ||
                value.Operands.Length < 2 || value.MemoryIndex >= plan.Memory.Count || !value.Operands[1].IsConstant)
            {
                continue;
            }

            var handle = value.Operands[0];
            if (handle.Operands.Length < 2 || (value.Kind == ScalarValueKind.ScalarBufferWord && handle.Operands.Length != 4) ||
                !seen.Add(value))
            {
                continue;
            }

            if (!groups.TryGetValue((handle, value.Kind), out var reads))
            {
                groups[(handle, value.Kind)] = reads = [];
            }

            reads.Add(value);
        }

        var result = new List<RawReadGroup>();
        foreach (var ((handle, kind), reads) in groups)
        {
            var immediates = new long[reads.Count];
            var offsets = new uint[reads.Count];
            for (var index = 0; index < reads.Count; index++)
            {
                immediates[index] = (int)plan.Memory[reads[index].MemoryIndex].Offset;
                offsets[index] = (uint)reads[index].Operands[1].Payload;
            }

            result.Add(new RawReadGroup
            {
                Handle = handle,
                Kind = kind,
                Reads = [.. reads],
                Immediates = immediates,
                Offsets = offsets,
            });
        }

        return [.. result];
    }

    public static void PrefetchSources(ShaderResourcePlan plan, IReadOnlyList<uint> sources, bool[] activeSources, RuntimeValueEvaluator evaluator)
    {
        if (evaluator.Inputs.ReadResidentMemory is null || evaluator.Inputs.ReadMemory is null)
        {
            return;
        }

        var prefetch = For(plan);
        var scratch = Begin();
        foreach (var source in sources)
        {
            if (source >= prefetch.Sources.Length || (activeSources.Length != 0 && !activeSources[source]))
            {
                continue;
            }

            Collect(prefetch.Sources[source], evaluator, scratch);
        }

        ReadRuns(evaluator, scratch);
    }

    public static void PrefetchTable(ShaderResourcePlan plan, RuntimeValueEvaluator evaluator, RuntimeValueEvaluator cleanEvaluator,
        IReadOnlyList<byte> cleanFlatSlots)
    {
        if (evaluator.Inputs.ReadResidentMemory is null || !ReferenceEquals(cleanFlatSlots, plan.CleanFlatSlots))
        {
            return;
        }

        var prefetch = For(plan);
        if (prefetch.Table.Length != 0 && evaluator.Inputs.ReadMemory is not null)
        {
            var scratch = Begin();
            Collect(prefetch.Table, evaluator, scratch);
            ReadRuns(evaluator, scratch);
        }

        if (prefetch.CleanTable.Length != 0 && cleanEvaluator.Inputs.ReadMemory is not null)
        {
            var scratch = Begin();
            Collect(prefetch.CleanTable, cleanEvaluator, scratch);
            ReadRuns(cleanEvaluator, scratch);
        }
    }

    private static Scratch Begin()
    {
        var scratch = _scratch ??= new Scratch();
        scratch.Count = 0;
        return scratch;
    }

    private static void Collect(RawReadGroup[] groups, RuntimeValueEvaluator evaluator, Scratch scratch)
    {
        var compiled = evaluator.IndexedEvaluator;
        foreach (var group in groups)
        {
            int[]? readIndices = null;
            bool based;
            ulong baseAddress;
            ulong high;
            ulong records;
            if (compiled is not null)
            {
                if (!ReferenceEquals(Volatile.Read(ref group.IndexedFor), compiled))
                {
                    Index(group, compiled);
                }

                readIndices = group.ReadIndices;
                based = evaluator.TryEvaluateRawBase(group.Handle, group.Kind, group.HandleIndices, out baseAddress, out high, out records);
            }
            else
            {
                based = evaluator.TryEvaluateRawBase(group.Handle, group.Kind, out baseAddress, out high, out records);
            }

            if (!based || (baseAddress & ~3ul) == 0)
            {
                continue;
            }

            for (var index = 0; index < group.Reads.Length; index++)
            {
                var readIndex = readIndices is null ? -1 : readIndices[index];
                var evaluated = readIndex >= 0 ? evaluator.IsEvaluatedAt(readIndex) : evaluator.IsEvaluated(group.Reads[index]);
                if (!evaluated &&
                    RuntimeValueEvaluator.ResolveRawAddress(group.Kind, baseAddress, high, records, group.Immediates[index], group.Offsets[index],
                        out var address) == RuntimeValueEvaluator.RawAddress.Read)
                {
                    scratch.Add(address, group.Reads[index], readIndex);
                }
            }
        }
    }

    private static void Index(RawReadGroup group, CompiledResourceEvaluator compiled)
    {
        static int IndexOf(CompiledResourceEvaluator compiled, ScalarValue value) =>
            !value.IsConstant && compiled.TryGetIndex(value, out var index) ? index : -1;

        var handle = new int[group.Handle.Operands.Length];
        for (var operand = 0; operand < handle.Length; operand++)
        {
            handle[operand] = IndexOf(compiled, group.Handle.Operands[operand]);
        }

        var reads = new int[group.Reads.Length];
        for (var read = 0; read < reads.Length; read++)
        {
            reads[read] = IndexOf(compiled, group.Reads[read]);
        }

        group.HandleIndices = handle;
        group.ReadIndices = reads;
        Volatile.Write(ref group.IndexedFor, compiled);
    }

    private static void ReadRuns(RuntimeValueEvaluator evaluator, Scratch scratch)
    {
        var count = scratch.Count;
        if (count < MinimumRunWords)
        {
            Array.Clear(scratch.Values, 0, count);
            return;
        }

        var reader = evaluator.Inputs.ReadResidentMemory!;
        var clean = evaluator.Inputs.ReadsClean;
        var addresses = scratch.Addresses;
        var values = scratch.Values;
        Array.Sort(addresses, scratch.Order, 0, count);
        var start = 0;
        while (start < count)
        {
            var end = start + 1;
            var words = 1;
            while (end < count && addresses[end] - addresses[end - 1] <= sizeof(uint) &&
                   addresses[end] - addresses[start] < MaximumRunBytes)
            {
                if (addresses[end] != addresses[end - 1])
                {
                    words++;
                }

                end++;
            }

            if (words >= MinimumRunWords)
            {
                ReadRun(evaluator, reader, clean, scratch, start, end);
            }

            start = end;
        }

        Array.Clear(values, 0, count);
    }

    private static void ReadRun(RuntimeValueEvaluator evaluator, ResidentGuestBytesReader reader, bool clean, Scratch scratch, int start, int end)
    {
        var addresses = scratch.Addresses;
        var first = addresses[start];
        var length = checked((int)(addresses[end - 1] - first) + sizeof(uint));
        if (scratch.Bytes.Length < length)
        {
            scratch.Bytes = new byte[Math.Max(length, scratch.Bytes.Length * 2)];
        }

        var bytes = scratch.Bytes.AsSpan(0, length);
        if (!reader(first, bytes, clean))
        {
            Interlocked.Increment(ref _refused);
            return;
        }

        Interlocked.Increment(ref _runs);
        Interlocked.Add(ref _words, end - start);
        for (var index = start; index < end; index++)
        {
            var word = System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(bytes[(int)(addresses[index] - first)..]);
            var added = scratch.Order[index];
            var compiledIndex = scratch.Indices[added];
            if (compiledIndex >= 0)
            {
                evaluator.SeedAt(compiledIndex, word);
            }
            else
            {
                evaluator.Seed(scratch.Values[added], word);
            }

            evaluator.RecordPrefetchedConsumer(scratch.Values[added], addresses[index]);
        }
    }

    private sealed class HandleComparer : IEqualityComparer<(ScalarValue Handle, ScalarValueKind Kind)>
    {
        public static readonly HandleComparer Instance = new();

        public bool Equals((ScalarValue Handle, ScalarValueKind Kind) left, (ScalarValue Handle, ScalarValueKind Kind) right) =>
            ReferenceEquals(left.Handle, right.Handle) && left.Kind == right.Kind;

        public int GetHashCode((ScalarValue Handle, ScalarValueKind Kind) value) =>
            HashCode.Combine(System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(value.Handle), value.Kind);
    }
}
