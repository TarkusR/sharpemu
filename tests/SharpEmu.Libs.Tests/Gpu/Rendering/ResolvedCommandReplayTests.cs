// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.Libs.Gpu.GpuCommands.Registers;
using SharpEmu.Libs.Gpu.Rendering;
using SharpEmu.Libs.Tests.Gpu.Scheduling;
using Xunit;
using static SharpEmu.Libs.Tests.Gpu.Rendering.RenderExecutorFixtures;

namespace SharpEmu.Libs.Tests.Gpu.Rendering;

[Collection(SchedulingStateCollection.Name)]
public sealed class ResolvedCommandReplayTests : IDisposable
{
    private sealed class CapturingSink : IResolvedCommandSink
    {
        private long _sequence;

        public long NextSequence() => ++_sequence;

        public List<object> Records { get; } = [];

        public void Record(ResolvedDraw record) => Records.Add(record);

        public void Record(ResolvedDispatch record) => Records.Add(record);
    }

    private readonly FatalScope _fatal = new();

    public void Dispose() => _fatal.Dispose();

    private static (RecordingRenderHost Host, FakePipelineProvider Pipelines) Arrange()
    {
        var pipelines = new FakePipelineProvider
        {
            Graphics = Programs(vertexBuffers: [new VertexInputBuffer(VertexBase, 16, 4)]),
            Compute = ComputeProgram(Stage(Program(ShaderStageKind.Compute, buffers: [new BufferResourceInfo(false, true, false, false, false, 4, 0)]),
                buffers: [BufferDescriptor(VertexBase, 4, 16)])),
        };
        return (new RecordingRenderHost(), pipelines);
    }

    private static void Run(RenderExecutor executor, RegisterBanks banks)
    {
        executor.DrawIndexed(7, banks, Indexed(6, instances: 2, baseVertex: 3));
        executor.DrawAuto(8, banks, Auto(3));
        executor.Dispatch(9, banks, 1, 1, 1, 0x41);
    }

    [Fact]
    public void RecordsReplayedAfterTheBanksWereClearedMakeTheSameHostCalls()
    {
        var (direct, directPipelines) = Arrange();
        Run(new RenderExecutor(direct, directPipelines), Banks());

        var (host, pipelines) = Arrange();
        var sink = new CapturingSink();
        var banks = Banks();
        Run(new RenderExecutor(host, pipelines, sink), banks);
        Assert.Empty(host.Calls);
        Assert.Equal(3, sink.Records.Count);
        Assert.Equal([1L, 2L, 3L], sink.Records.Select(static record => record is ResolvedDraw draw ? draw.Sequence : ((ResolvedDispatch)record).Sequence));

        banks.Reset();
        var recorder = new RenderRecorder(host, pipelines, strictDrawResources: true);
        foreach (var record in sink.Records)
        {
            if (record is ResolvedDraw draw)
            {
                recorder.Record(draw);
            }
            else
            {
                recorder.Record((ResolvedDispatch)record);
            }
        }

        Assert.NotEmpty(host.Calls);
        Assert.Equal(direct.Calls, host.Calls);
        Assert.Equal(directPipelines.Calls.Order(), pipelines.Calls.Order());
        Assert.Equal(["get_compute_program threadDimensions=False", "get_graphics_programs pixelActive=True", "get_graphics_programs pixelActive=True"],
            pipelines.Calls.Take(3).Order());
    }
}
