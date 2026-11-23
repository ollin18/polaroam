"""Several processes on one output directory, each with its own device shards."""

import json
import shlex
import subprocess
import sys

import polars as pl
import pytest

import polaroam as pr


def tables(d):
    pipe = pr.Pipeline(d, verbose=False)
    out = {}
    for name, key in [("stops", ["uid", "stop_event"]), ("trajectories", ["uid", "trajectory_id"]),
                      ("clusters", ["uid", "stop_event"]), ("home_work", ["uid", "stop_event"]),
                      ("locations", ["uid"])]:
        out[name] = getattr(pipe, "trajectories_table" if name == "trajectories" else name)().collect().sort(key)
    return out


def days(pings, n=4):
    """``n`` time batches of equal length (any timestamp unit)."""
    t0, t1 = pings["timestamp"].min(), pings["timestamp"].max() + 1
    cut = pl.col("timestamp") - t0
    return [(f"b{i}", pings.filter((cut * n // (t1 - t0)) == i)) for i in range(n)]


def _worker_cmd(out, files, shards, common):
    """One process of many, as a SLURM array task would run it: every batch, then finalize."""
    cli = [sys.executable, "-m", "polaroam.cli"]
    steps = [[*cli, "update", str(f), str(out), *common, "--trajectories", "--batch-id", b, "--only-shards", shards]
             for b, f in files]
    steps.append([*cli, "finalize", str(out), "--only-shards", shards])
    return " && ".join(shlex.join(c) for c in steps)


@pytest.mark.parametrize("shared_split", [False, True])
def test_processes_with_disjoint_shards_equal_one_process(tmp_path, pings, shared_split):
    cfg = pr.PipelineConfig(timezone="America/Sao_Paulo", n_shards=4)
    batches = days(pings)
    one = pr.Pipeline(tmp_path / "one", cfg, trajectories=True, verbose=False)
    for b, df in batches:
        one.update(df, batch_id=b)
    one.finalize()

    files = []
    for b, df in batches:
        df.write_parquet(tmp_path / f"{b}.parquet")
        files.append((b, tmp_path / f"{b}.parquet"))
    many = tmp_path / "many"
    common = ["--timezone", "America/Sao_Paulo", "--shards", "4"]
    if shared_split:
        prep = pr.Pipeline(many, cfg, trajectories=True, verbose=False)
        for b, f in files:
            prep.split(f, b)
    # three processes at the same time on one directory: shards {0, 3}, {1}, {2}
    procs = [subprocess.Popen(_worker_cmd(many, files, s, common), shell=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) for s in ("0,3", "1", "2")]
    for pr_ in procs:
        out, _ = pr_.communicate(timeout=300)
        assert pr_.returncode == 0, out.decode()[-2000:]

    a, b = tables(tmp_path / "one"), tables(many)
    for name in a:
        assert a[name].equals(b[name]), name
    st = pr.Pipeline(many, verbose=False).status()
    assert st["finalized"] == [0, 1, 2, 3] and all(v == [0, 1, 2, 3] for v in st["batches"].values())
    assert not any((many / "_split").glob("*")), "splits are removed once every shard is committed"
    assert "batches" not in json.loads((many / "manifest.json").read_text())


def test_shard_retry_and_validation(tmp_path, pings):
    cfg = pr.PipelineConfig(timezone="America/Sao_Paulo", n_shards=3)
    pipe = pr.Pipeline(tmp_path, cfg, verbose=False)
    first = pipe.update(pings, batch_id="all", shards=[1])
    assert first.n_stops > 0
    assert pipe.update(pings, batch_id="all", shards=[1]).n_stops == 0  # committed: a no-op
    with pytest.raises(ValueError):
        pipe.update(pings, batch_id="all", shards=[3])
    with pytest.raises(ValueError):
        pipe.update(pings, shards=[0])  # no batch_id
    assert pipe.status()["batches"] == {"all": [1]}


def test_old_manifest_progress_is_kept(tmp_path, pings):
    cfg = pr.PipelineConfig(timezone="America/Sao_Paulo", n_shards=2)
    pipe = pr.Pipeline(tmp_path, cfg, verbose=False)
    pipe.update(pings, batch_id="b0")
    m = json.loads((tmp_path / "manifest.json").read_text())  # as written before _commits/
    m.update(batches={"b0": [0, 1]}, finalized=False)
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    import shutil
    shutil.rmtree(tmp_path / "_commits")
    assert pr.Pipeline(tmp_path, verbose=False).update(pings, batch_id="b0").n_stops == 0


def test_cli_only_shards(tmp_path, pings, monkeypatch):
    from polaroam.cli import _only_shards, main

    assert _only_shards("0-2,5", 8) == [0, 1, 2, 5]
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "3")
    assert _only_shards("slurm", 8) == [1, 4, 7]
    f = tmp_path / "p.parquet"
    pings.write_parquet(f)
    out = str(tmp_path / "out")
    common = ["--timezone", "America/Sao_Paulo", "--shards", "3"]
    assert main(["update", str(f), out, *common, "--batch-id", "x", "--split-only"]) == 0
    for k in ("0", "1-2"):
        assert main(["update", str(f), out, *common, "--batch-id", "x", "--only-shards", k]) == 0
    assert pr.Pipeline(out, verbose=False).status()["batches"] == {"x": [0, 1, 2]}
