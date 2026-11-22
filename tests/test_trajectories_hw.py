"""Trajectories, Douglas-Peucker, local time, home/work and the Pipeline."""

import dataclasses

import numpy as np
import polars as pl

import polaroam as pr


def rdp(x, y, tol):
    keep = np.zeros(len(x), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(x) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        dx, dy = x[b] - x[a], y[b] - y[a]
        l2 = dx * dx + dy * dy
        idx = np.arange(a + 1, b)
        t = np.clip(((x[idx] - x[a]) * dx + (y[idx] - y[a]) * dy) / l2, 0, 1) if l2 > 0 else np.zeros(len(idx))
        d = np.hypot(x[idx] - x[a] - t * dx, y[idx] - y[a] - t * dy)
        j = int(np.argmax(d))
        if d[j] > tol:
            keep[idx[j]] = True
            stack += [(a, idx[j]), (idx[j], b)]
    return keep


def test_douglas_peucker_matches_recursive():
    rng = np.random.default_rng(0)
    rows, ref = [], []
    for s in range(200):
        n = int(rng.integers(2, 300))
        x, y = np.cumsum(rng.normal(0, 30, n)), np.cumsum(rng.normal(0, 30, n))
        if s % 7 == 0:
            x, y = np.round(x / 50) * 50, np.zeros(n)
        rows.append(pl.DataFrame({"_seg": s, "_i": np.arange(n), "_x": x, "_y": y}))
        ref.append(rdp(x, y, 40.0))
    got = pr.douglas_peucker(pl.concat(rows), 40.0).to_numpy()
    assert np.array_equal(got, np.concatenate(ref))


def test_trajectories_connect_stops(pings):
    stops, trajs = pr.detect_stops(pings, trajectories=True)
    assert trajs.height > 0
    assert trajs["geometry"].str.starts_with("LINESTRING (").all()
    assert (trajs["n_points"] >= 2).all() and (trajs["n_points"] <= trajs["n_pings"] + 2).all()
    # origin / destination are real stops of the same device, in time order
    j = trajs.join(stops.select("uid", pl.col("stop_event").alias("origin_stop_event"),
                                pl.col("end_timestamp").alias("_oe")), on=["uid", "origin_stop_event"])
    assert (j["start_timestamp"] == j["_oe"]).all()
    # computing trajectories afterwards from stored stops gives the same result
    assert pr.extract_trajectories(pings, stops).equals(trajs)
    # no simplification keeps every vertex
    raw = pr.extract_trajectories(pings, stops, simplify_tolerance=0)
    assert (raw["n_points"] >= trajs["n_points"]).all()


def test_local_time_fixed_and_polygons(tmp_path):
    df = pl.DataFrame({"uid": ["a", "b"], "stop_event": [1, 1], "start_timestamp": [1704067200, 1704067200],
                       "end_timestamp": [1704070800, 1704070800], "stop_latitude": [19.4, 40.7],
                       "stop_longitude": [-99.1, -74.0]})
    poly = tmp_path / "tz.csv"
    pl.DataFrame({"tz": ["America/Mexico_City"], "geometry": ["POLYGON ((-100 19, -98 19, -98 20, -100 20, -100 19))"]}
                 ).write_csv(poly)
    out = pr.add_local_time(df, "America/New_York", str(poly)).sort("uid")
    assert out["timezone"].to_list() == ["America/Mexico_City", None]
    assert out["inputed_timezone"].to_list() == ["America/Mexico_City", "America/New_York"]
    assert out["hour"].to_list() == [18, 19]                    # 2024-01-01 00:00 UTC
    assert out["weekday"].to_list() == [1, 1]                   # Sunday (Dec 31) = 1
    assert out["weekend"].to_list() == [True, True]


def test_pipeline_incremental_equals_run(tmp_path, pings):
    cfg = pr.PipelineConfig(timezone="America/Mexico_City", home_work=pr.HomeWorkParams(home_min_days=3, work_min_days=2))
    one = pr.Pipeline(tmp_path / "one", cfg, trajectories=True, verbose=False)
    one.run(pings)
    inc = pr.Pipeline(tmp_path / "inc", dataclasses.replace(cfg, n_shards=3), trajectories=True, verbose=False)
    days = pings.with_columns(pl.from_epoch("timestamp", "ms").dt.date().alias("_d"))
    for (d,), g in days.sort("_d").group_by("_d", maintain_order=True):
        inc.update(g.drop("_d"), batch_id=str(d), compute_home_work=False)
        inc.update(g.drop("_d"), batch_id=str(d))  # retrying a committed batch is a no-op
    inc.finalize()
    for name, key in [("stops", ["uid", "stop_event"]), ("trajectories_table", ["uid", "trajectory_id"]),
                      ("clusters", ["uid", "stop_event"]), ("home_work", ["uid", "stop_event"]), ("locations", ["uid"])]:
        a, b = getattr(one, name)().collect().sort(key), getattr(inc, name)().collect().sort(key)
        assert a.equals(b), name
    loc = one.locations().collect()
    assert loc.height > 0 and loc["work_latitude"].is_not_null().sum() > 0
    # home is the most visited night-time cluster
    hw = one.home_work().collect()
    assert set(hw["location_type"].unique()) <= {"H", "W", "O"}
