// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Runtime.InteropServices;
using SharpEmu.Libs.Gpu.Scheduling;
using Silk.NET.Vulkan;
using VkBuffer = Silk.NET.Vulkan.Buffer;

namespace SharpEmu.Libs.Gpu.Buffers;

public sealed class ContentUploadArena : IDisposable
{
    private const ulong ArenaBytes = 32UL * 1024 * 1024;
    private const int MaxContentBytes = 64 * 1024;

    private sealed class Arena(GpuBuffer buffer)
    {
        public readonly GpuBuffer Buffer = buffer;
        public readonly byte[] Mirror = GC.AllocateUninitializedArray<byte>((int)ArenaBytes);
        public readonly Dictionary<ulong, (ulong Offset, int Size)> Entries = new();
        public readonly Dictionary<uint[], (ulong Offset, int Size)> IdentityEntries = new(ReferenceEqualityComparer.Instance);
        public ulong Offset;
        public ulong LastUseTick;
    }

    private readonly SubmissionScheduler _scheduler;
    private readonly Arena[] _arenas;
    private int _active;

    private static long _hits;
    private static long _identityHits;
    private static long _stores;
    private static long _recycles;
    private static long _refused;
    private static long _collisions;

    public ContentUploadArena(GpuDeviceInfo device, SubmissionScheduler scheduler)
    {
        _scheduler = scheduler;
        _arenas =
        [
            new Arena(new GpuBuffer(device, scheduler, GpuBufferUsage.Stream, 0, GpuBuffer.AddressedFlags, ArenaBytes)),
            new Arena(new GpuBuffer(device, scheduler, GpuBufferUsage.Stream, 0, GpuBuffer.AddressedFlags, ArenaBytes)),
        ];
    }

    public bool TryUpload(uint[] words, ulong alignment, out VkBuffer buffer, out ulong offset)
    {
        buffer = default;
        offset = 0;
        if (words.Length == 0 || words.Length * sizeof(uint) > MaxContentBytes || alignment == 0)
        {
            return false;
        }

        var tick = _scheduler.CurrentTick;
        for (var index = 0; index < _arenas.Length; index++)
        {
            var arena = _arenas[(_active + index) % _arenas.Length];
            if (!arena.IdentityEntries.TryGetValue(words, out var entry))
            {
                continue;
            }

            arena.LastUseTick = tick;
            Interlocked.Increment(ref _identityHits);
            buffer = arena.Buffer.Handle;
            offset = entry.Offset;
            return true;
        }

        if (!TryStore(MemoryMarshal.AsBytes<uint>(words), alignment, tick, out var stored, out buffer, out offset))
        {
            return false;
        }

        stored.IdentityEntries[words] = (offset, words.Length * sizeof(uint));
        return true;
    }

    public bool TryUpload(ReadOnlySpan<byte> data, ulong alignment, out VkBuffer buffer, out ulong offset)
    {
        buffer = default;
        offset = 0;
        if (data.Length == 0 || data.Length > MaxContentBytes || alignment == 0)
        {
            return false;
        }

        var hash = Hash(data);
        var tick = _scheduler.CurrentTick;
        for (var index = 0; index < _arenas.Length; index++)
        {
            var arena = _arenas[(_active + index) % _arenas.Length];
            if (!arena.Entries.TryGetValue(hash, out var entry))
            {
                continue;
            }

            if (entry.Size != data.Length || !data.SequenceEqual(arena.Mirror.AsSpan((int)entry.Offset, entry.Size)))
            {
                Interlocked.Increment(ref _collisions);
                break;
            }

            arena.LastUseTick = tick;
            Interlocked.Increment(ref _hits);
            buffer = arena.Buffer.Handle;
            offset = entry.Offset;
            return true;
        }

        if (!TryStore(data, alignment, tick, out var stored, out buffer, out offset))
        {
            return false;
        }

        stored.Entries[hash] = (offset, data.Length);
        return true;
    }

    public bool TryUploadAddress(ReadOnlySpan<byte> data, ulong alignment, out ulong address)
    {
        address = 0;
        if (!TryUpload(data, alignment, out var buffer, out var offset))
        {
            return false;
        }

        foreach (var arena in _arenas)
        {
            if (arena.Buffer.Handle.Handle == buffer.Handle)
            {
                address = arena.Buffer.DeviceAddress + offset;
                return true;
            }
        }

        return false;
    }

    private bool TryStore(ReadOnlySpan<byte> data, ulong alignment, ulong tick, out Arena stored, out VkBuffer buffer, out ulong offset)
    {
        var active = _arenas[_active];
        var aligned = (active.Offset + alignment - 1) / alignment * alignment;
        if (aligned + (ulong)data.Length > ArenaBytes)
        {
            var other = _arenas[1 - _active];
            if (other.LastUseTick > _scheduler.CompletedTick)
            {
                Interlocked.Increment(ref _refused);
                stored = active;
                buffer = default;
                offset = 0;
                return false;
            }

            other.Entries.Clear();
            other.IdentityEntries.Clear();
            other.Offset = 0;
            _active = 1 - _active;
            active = other;
            aligned = 0;
            Interlocked.Increment(ref _recycles);
        }

        active.Buffer.Write(aligned, data);
        data.CopyTo(active.Mirror.AsSpan((int)aligned));
        active.Offset = aligned + (ulong)data.Length;
        active.LastUseTick = tick;
        Interlocked.Increment(ref _stores);
        stored = active;
        buffer = active.Buffer.Handle;
        offset = aligned;
        return true;
    }

    private static ulong Hash(ReadOnlySpan<byte> data)
    {
        var hash = 0xCBF29CE484222325ul ^ (ulong)data.Length;
        var words = MemoryMarshal.Cast<byte, ulong>(data[..(data.Length & ~7)]);
        foreach (var word in words)
        {
            hash = (hash ^ word) * 0x100000001B3ul;
            hash ^= hash >> 29;
        }

        foreach (var tail in data[(data.Length & ~7)..])
        {
            hash = (hash ^ tail) * 0x100000001B3ul;
        }

        return hash;
    }

    public static string TakeReport() =>
        FormattableString.Invariant($"[PERF][CONTENT_UPLOADS] hits={Interlocked.Exchange(ref _hits, 0)} identity_hits={Interlocked.Exchange(ref _identityHits, 0)} stores={Interlocked.Exchange(ref _stores, 0)} recycles={Interlocked.Exchange(ref _recycles, 0)} ") +
        FormattableString.Invariant($"refused={Interlocked.Exchange(ref _refused, 0)} collisions={Interlocked.Exchange(ref _collisions, 0)}");

    public void Dispose()
    {
        foreach (var arena in _arenas)
        {
            arena.Buffer.Dispose();
        }
    }
}
