"""Coordinate sinks: ``remove_sinks`` in the pipeline, the standalone functions and the CLI."""

import warnings

import polars as pl
import pytest

import polaroam as pr
from polaroam.cli import main
from polaroam.pipeline import _config_from_dict

SINK = (19.4326, -99.1332)  # the Zócalo default geocode
MIN = 20  # the synthetic data has 25 devices


@pytest.fixture(scope="module")
def dirty(pings) -> pl.DataFrame:
    """The synthetic pings plus, for every device, a fake 10-minute stay snapped to the sink
    an hour after its last ping, and one isolated snap in the middle of its data."""
    ends = pings.group_by("uid").agg(pl.col("timestamp").max().alias("t1"), pl.col("timestamp").median().alias("tm"))
    steps = pl.DataFrame({"k": range(6)})
    block = ends.join(steps, how="cross").select("uid", (pl.col("t1") + 3_600_000 + pl.col("k") * 120_000).alias("timestamp"))
    snap = ends.select("uid", (pl.col("tm").cast(pl.Int64) + 1).alias("timestamp"))
    extra = pl.concat([block, snap]).with_columns(pl.lit(SINK[0]).alias("latitude"), pl.lit(SINK[1]).alias("longitude"))
    return pl.concat([pings, extra.select(pings.columns)]).sort("uid", "timestamp")


def at_sink(df: pl.DataFrame, lat: str, lon: str) -> int:
    return df.filter((pl.col(lat) == SINK[0]) & (pl.col(lon) == SINK[1])).height


def clean(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(~((pl.col("latitude") == SINK[0]) & (pl.col("longitude") == SINK[1])))


def test_find_sinks(dirty):
    s = pr.find_sinks(dirty, MIN)
    assert s.select("latitude", "longitude").rows() == [SINK]
    assert s["devices"][0] == 25


def test_detect_stops_remove_sinks(dirty, tmp_path):
    kept = pr.detect_stops(dirty)
    assert at_sink(kept, "stop_latitude", "stop_longitude") == 25  # gc-lbs default: one fake stay per device
    ref_stops, ref_traj = pr.detect_stops(clean(dirty), trajectories=True)
    stops, traj = pr.detect_stops(dirty, trajectories=True, remove_sinks=True, sinks_min_devices=MIN)
    assert stops.equals(ref_stops) and traj.equals(ref_traj)
    # a list computed beforehand, as a DataFrame or a parquet path, gives the same result
    sinks = pr.find_sinks(dirty, MIN)
    sinks.write_parquet(tmp_path / "s.parquet")
    assert pr.detect_stops(dirty, remove_sinks=sinks).equals(ref_stops)
    assert pr.detect_stops(dirty, remove_sinks=str(tmp_path / "s.parquet")).equals(ref_stops)
    # extract_trajectories takes the same option
    tr = pr.extract_trajectories(dirty, stops, remove_sinks=True, sinks_min_devices=MIN)
    assert tr.equals(pr.extract_trajectories(clean(dirty), ref_stops))


def test_no_sinks_found_warns(pings):
    with pytest.warns(UserWarning, match="no ping was removed"):
        out = pr.detect_stops(pings, remove_sinks=True)  # 25 devices never reach 5,000
    assert out.equals(pr.detect_stops(pings))


def test_pipeline_remove_sinks(dirty, tmp_path):
    hw = pr.HomeWorkParams(home_min_days=3, work_min_days=2)
    auto = pr.Pipeline(tmp_path / "auto", pr.PipelineConfig(home_work=hw, remove_sinks=True, sinks_min_devices=MIN),
                       trajectories=True, verbose=False)
    auto.run(dirty)
    assert (tmp_path / "auto" / "sinks.parquet").exists()
    assert auto.sinks().select("latitude", "longitude").rows() == [SINK]
    ref = pr.Pipeline(tmp_path / "ref", pr.PipelineConfig(home_work=hw), trajectories=True, verbose=False)
    ref.run(clean(dirty))
    for name, key in [("stops", ["uid", "stop_event"]), ("trajectories_table", ["uid", "trajectory_id"]),
                      ("home_work", ["uid", "stop_event"]), ("locations", ["uid"])]:
        assert getattr(auto, name)().collect().sort(key).equals(getattr(ref, name)().collect().sort(key)), name
    assert at_sink(auto.locations().collect(), "home_latitude", "home_longitude") == 0
    # a file passed explicitly is used as is and not copied into the output directory
    auto.sinks().write_parquet(tmp_path / "s.parquet")
    by_file = pr.Pipeline(tmp_path / "file", pr.PipelineConfig(home_work=hw, remove_sinks=str(tmp_path / "s.parquet")),
                          verbose=False)
    by_file.run(dirty)
    assert not (tmp_path / "file" / "sinks.parquet").exists()
    assert by_file.stops().collect().sort("uid", "stop_event").equals(ref.stops().collect().sort("uid", "stop_event"))
    auto.reset()
    assert not (tmp_path / "auto" / "sinks.parquet").exists()


def test_incremental_reuses_first_batch_sinks(dirty, tmp_path):
    """The sinks are found in the first batch and reused, so later batches never recompute them."""
    cfg = pr.PipelineConfig(remove_sinks=True, sinks_min_devices=MIN, n_shards=2)
    pipe = pr.Pipeline(tmp_path / "inc", cfg, verbose=False)
    # batch 1: each device's own data (with its mid-data snap); batch 2: the fake stays at the sink
    real = ~((pl.col("latitude") == SINK[0]) & (pl.col("longitude") == SINK[1]))
    last = pl.col("timestamp").filter(real).max().over("uid")
    first, second = dirty.filter(pl.col("timestamp") <= last), dirty.filter(pl.col("timestamp") > last)
    pipe.update(first, batch_id="1")
    f = tmp_path / "inc" / "sinks.parquet"
    found, mtime = pl.read_parquet(f), f.stat().st_mtime_ns
    assert found.select("latitude", "longitude").rows() == [SINK]
    pipe.update(second, batch_id="2")
    pipe.finalize()
    assert f.stat().st_mtime_ns == mtime and pl.read_parquet(f).equals(found)
    assert at_sink(pipe.stops().collect(), "stop_latitude", "stop_longitude") == 0


def test_config_alias_and_old_manifests():
    with pytest.warns(DeprecationWarning):
        cfg = pr.PipelineConfig(drop_coordinates="s.parquet")
    assert cfg.remove_sinks == "s.parquet" and "drop_coordinates" not in cfg.to_dict()
    old = pr.PipelineConfig().to_dict()
    del old["remove_sinks"], old["sinks_min_devices"]
    old["drop_coordinates"] = "s.parquet"
    assert _config_from_dict(old).remove_sinks == "s.parquet"
    with pytest.raises(TypeError):
        pr.PipelineConfig(remove_sinks=pl.DataFrame())


def test_cli_remove_sinks(dirty, tmp_path):
    src = tmp_path / "pings.parquet"
    dirty.write_parquet(src)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert main(["run", str(src), str(tmp_path / "out"), "--remove-sinks", "--sinks-min-devices", str(MIN)]) == 0
    assert pl.read_parquet(tmp_path / "out" / "sinks.parquet").select("latitude", "longitude").rows() == [SINK]
    assert main(["sinks", str(src), str(tmp_path / "s.parquet"), "--min-devices", str(MIN)]) == 0
    assert main(["run", str(src), str(tmp_path / "out2"), "--remove-sinks", str(tmp_path / "s.parquet")]) == 0
    a = pr.Pipeline(tmp_path / "out").stops().collect().sort("uid", "stop_event")
    b = pr.Pipeline(tmp_path / "out2").stops().collect().sort("uid", "stop_event")
    assert a.equals(b) and at_sink(a, "stop_latitude", "stop_longitude") == 0
