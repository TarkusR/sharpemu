// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using System.Diagnostics;

namespace SharpEmu.Libs.Diagnostics;

internal static class SubmitSiteTally
{
    private const int SiteFrames = 6;

    public static readonly bool Enabled = VideoOut.RenderPhaseProfile.Enabled;

    private static readonly object _gate = new();
    private static readonly Dictionary<string, long> _sites = new(StringComparer.Ordinal);

    private static readonly HashSet<string> PlumbingTypes = new(StringComparer.Ordinal)
    {
        "SubmitSiteTally",
        "TickTimeline",
        "SubmissionScheduler",
        "VulkanTickDevice",
        "GpuWorkerRelay",
    };

    public static void Note()
    {
        var names = new List<string>(SiteFrames);
        foreach (var frame in new StackTrace(1, false).GetFrames())
        {
            if (frame.GetMethod() is not { } method)
            {
                continue;
            }

            var type = method.DeclaringType;
            while (type is { DeclaringType: not null } && type.Name.StartsWith('<'))
            {
                type = type.DeclaringType;
            }

            var typeName = type?.Name ?? "?";
            if (PlumbingTypes.Contains(typeName) || method.Name.StartsWith('<'))
            {
                continue;
            }

            names.Add($"{typeName}.{method.Name}");
            if (names.Count == SiteFrames)
            {
                break;
            }
        }

        var site = string.Join(" < ", names);
        lock (_gate)
        {
            _sites.TryGetValue(site, out var count);
            _sites[site] = count + 1;
        }
    }

    public static void Report()
    {
        List<KeyValuePair<string, long>> ranked;
        lock (_gate)
        {
            if (_sites.Count == 0)
            {
                return;
            }

            ranked = _sites.OrderByDescending(static pair => pair.Value).Take(10).ToList();
            _sites.Clear();
        }

        foreach (var (site, count) in ranked)
        {
            Console.Error.WriteLine(FormattableString.Invariant($"[PERF][SUBMIT_SITE] n={count} site={site}"));
        }
    }
}
