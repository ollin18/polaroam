"""GPU engine gives exactly the CPU results (skipped without CUDA)."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import polaroam as pr

pytestmark = pytest.mark.skipif(not pr.cuda_available(), reason="cudf-polars / CUDA not available")


def test_every_stage_matches_cpu(pings):
    out = {}
    for cuda in (False, True):
        stops, trajs = pr.detect_stops(pings, trajectories=True, use_cuda=cuda)
        clusters = pr.cluster_stops(stops, pr.ClusterParams(20, 3), use_cuda=cuda)
        local = pr.add_local_time(clusters, "America/Mexico_City", use_cuda=cuda)
        hw = pr.classify_home_work(local, home_min_days=3, work_min_days=2, use_cuda=cuda)
        out[cuda] = (stops, trajs, clusters.sort("uid", "stop_event"), hw.sort("uid", "stop_event"))
    for cpu, gpu in zip(out[False], out[True]):
        assert_frame_equal(cpu, gpu, check_exact=True)


def test_incremental_on_gpu_matches_one_shot_cpu(pings):
    ref = pr.detect_stops(pings).sort("uid", "stop_event")
    det = pr.StopDetector(use_cuda=True)
    half = pings["timestamp"].median()
    parts = [det.update(pings.filter(pl.col("timestamp") <= half)).stops,
             det.update(pings.filter(pl.col("timestamp") > half)).stops, det.flush().stops]
    assert pl.concat(parts).sort("uid", "stop_event").equals(ref)


def test_hybrid_gpu_and_cpu_shards_match_cpu(pings, tmp_path, capsys):
    """hybrid: shards on the GPU and the CPU at the same time, from one queue == CPU only."""
    cfg = pr.PipelineConfig(timezone="America/Mexico_City", n_shards=6)
    half = pings["timestamp"].median()
    batches = [("a", pings.filter(pl.col("timestamp") <= half)), ("b", pings.filter(pl.col("timestamp") > half))]
    out = {}
    for name, kw in (("cpu", {}), ("hybrid", {"use_cuda": True, "hybrid": True})):
        pipe = pr.Pipeline(tmp_path / name, cfg, trajectories=True, verbose=name == "hybrid", **kw)
        for b, df in batches:
            pipe.update(df, batch_id=b)
        pipe.finalize()
        out[name] = [pipe.stops().collect().sort("uid", "stop_event"),
                     pipe.trajectories_table().collect().sort("uid", "trajectory_id"),
                     pipe.home_work().collect().sort("uid", "stop_event"),
                     pipe.locations().collect().sort("uid")]
    for cpu, hyb in zip(out["cpu"], out["hybrid"]):
        assert_frame_equal(cpu, hyb, check_exact=True)
    shards = [line for line in capsys.readouterr().out.splitlines() if " shard " in line]
    assert any(line.endswith("(cuda)") for line in shards), "the GPU took shards"
    assert any(not line.endswith("(cuda)") for line in shards), "the CPU took shards"
