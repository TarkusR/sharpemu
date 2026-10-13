// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;

namespace SharpEmu.Libs.Diagnostics;

internal static class ProgramPrefetchTally
{
    public enum Take
    {
        NoJob,
        OtherBanks,
        Abandoned,
        InputsDiffer,
        Stale,
        Taken,
        Count,
    }

    public static readonly bool Enabled = VideoOut.RenderPhaseProfile.Enabled;

    private static readonly long[] _takes = new long[(int)Take.Count];
    private static long _takeWaits;
    private static long _takeWaitTicks;
    private static long _finishedWaits;
    private static long _finishedWaitTicks;
    private static long _gateSleeps;
    private static long _gateSleepTicks;
    private static long _workerResolves;
    private static long _workerResolveTicks;
    private static long _renderResolves;
    private static long _renderResolveTicks;

    public static void NoteTake(Take take) => Interlocked.Increment(ref _takes[(int)take]);

    public static void NoteTakeWait(long ticks)
    {
        Interlocked.Increment(ref _takeWaits);
        Interlocked.Add(ref _takeWaitTicks, ticks);
    }

    public static void NoteFinishedWait(long ticks)
    {
        Interlocked.Increment(ref _finishedWaits);
        Interlocked.Add(ref _finishedWaitTicks, ticks);
    }

    public static void NoteGateSleep(long ticks)
    {
        Interlocked.Increment(ref _gateSleeps);
        Interlocked.Add(ref _gateSleepTicks, ticks);
    }

    public static void NoteWorkerResolve(long ticks)
    {
        Interlocked.Increment(ref _workerResolves);
        Interlocked.Add(ref _workerResolveTicks, ticks);
    }

    public static void NoteRenderResolve(long ticks)
    {
        Interlocked.Increment(ref _renderResolves);
        Interlocked.Add(ref _renderResolveTicks, ticks);
    }

    private static string Ms(ref long ticks) =>
        FormattableString.Invariant($"{Interlocked.Exchange(ref ticks, 0) * 1000.0 / Stopwatch.Frequency:F1}");

    public static string TakeReport()
    {
        var takes = string.Join(",", Enumerable.Range(0, (int)Take.Count).Select(index =>
            FormattableString.Invariant($"{(Take)index}:{Interlocked.Exchange(ref _takes[index], 0)}")));
        return FormattableString.Invariant($"[PERF][PROGRAM_PREFETCH] takes={takes} ") +
            FormattableString.Invariant($"take_waits={Interlocked.Exchange(ref _takeWaits, 0)} take_wait_ms={Ms(ref _takeWaitTicks)} ") +
            FormattableString.Invariant($"finished_waits={Interlocked.Exchange(ref _finishedWaits, 0)} finished_wait_ms={Ms(ref _finishedWaitTicks)} ") +
            FormattableString.Invariant($"gate_sleeps={Interlocked.Exchange(ref _gateSleeps, 0)} gate_sleep_ms={Ms(ref _gateSleepTicks)} ") +
            FormattableString.Invariant($"worker_resolves={Interlocked.Exchange(ref _workerResolves, 0)} worker_resolve_ms={Ms(ref _workerResolveTicks)} ") +
            FormattableString.Invariant($"render_resolves={Interlocked.Exchange(ref _renderResolves, 0)} render_resolve_ms={Ms(ref _renderResolveTicks)}");
    }
}
