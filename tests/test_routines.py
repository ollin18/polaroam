"""Routines: trips between stays, routine modes, and completion along the usual route."""

import polars as pl
import pytest

shapely = pytest.importorskip("shapely")
pytest.importorskip("numba")

import polaroam as pr

from .test_roads import grid, ll, trajectory

DAY = 86_400
T0 = 1_700_000_000
HOME, WORK = ll(0, 0), ll(0, 300)
# the device's habit: east to x=100, north, back west; 500 m where the straight street is 300 m
HABIT = [(0, 0), (100, 0), (100, 300), (0, 300)]


def stays():
    rows = []
    for d in range(3):
        t = T0 + d * DAY
        rows += [(2 * d, t - 3600, t, HOME, 0), (2 * d + 1, t + 600, t + 4 * 3600, WORK, 1)]
    return pl.DataFrame([{"uid": "u", "stop_event": e, "start_timestamp": a, "end_timestamp": b,
                          "stop_longitude": p[0], "stop_latitude": p[1], "cluster_label": c,
                          "cluster_longitude": p[0], "cluster_latitude": p[1]} for e, a, b, p, c in rows])


def trajectories():
    rows = []
    for d in range(3):
        t = T0 + d * DAY + 30
        if d < 2:  # observed all the way, walking
            coords, ts = trajectory(HABIT, every=25, noise=2, seed=d, t0=t, speed=1.4)
            o, dst, mode = 2 * d, 2 * d + 1, "walk"
        else:  # day 3: three pings near home, then nothing
            coords, ts = trajectory([(0, 0), (50, 0)], every=25, noise=1, seed=d, t0=t, speed=1.4)
            o, dst, mode = 2 * d, None, "unknown"
        rows.append({"uid": "u", "trajectory_id": d, "origin_stop_event": o, "destination_stop_event": dst,
                     "start_timestamp": ts[0], "end_timestamp": ts[-1], "duration": ts[-1] - ts[0],
                     "mode_guess": mode, "coordinates": coords, "timestamps": ts})
    return pl.DataFrame(rows, schema_overrides={"coordinates": pl.List(pl.List(pl.Float64)),
                                                "timestamps": pl.List(pl.Int64),
                                                "destination_stop_event": pl.Int64})


@pytest.fixture(scope="module")
def setup():
    net = pr.RoadNetwork(grid())
    tr = trajectories()
    trips = pr.stay_trips(stays(), tr, timezone="UTC")
    trips = pr.find_routines(trips, min_days=3)
    tm = pr.routine_modes(tr, trips)
    res = pr.match_trajectories(tm, net, by="mode")
    return net, trips, tm, res


def test_trips_states_and_routine_mode(setup):
    _, trips, tm, _ = setup
    go = trips.filter(pl.col("origin_location") == 0).sort("trip_id")
    assert go["state"].to_list() == ["complete", "complete", "partial"]
    assert trips.filter(pl.col("origin_location") == 1)["state"].to_list() == ["empty", "empty"]
    assert go["routine_id"].n_unique() == 1 and go["routine_days"][0] == 3
    assert trips.filter(pl.col("origin_location") == 1)["routine_id"].is_null().all()  # only 2 days
    # day 3 had no mode of its own: it gets the routine's
    assert go["mode"].to_list() == ["walk"] * 3
    assert go["mode_source"].to_list() == ["observed", "observed", "routine"]
    assert tm.filter(pl.col("trajectory_id") == 2)["mode"][0] == "walk"


def max_x(wkt):
    import math

    from polaroam.geo import M_PER_DEG
    x0 = shapely.from_wkt(wkt).bounds[2]
    return (x0 - ll(0, 0)[0]) * M_PER_DEG * math.cos(math.radians(19.4))


def test_completion_follows_the_habit(setup):
    net, trips, _, res = setup
    out = pr.complete_trips(trips, res, net, gap_m=150)
    d3 = out.trips.filter(pl.col("trip_id") == 4).row(0, named=True)
    assert d3["n_parts"] == 1 and d3["n_gaps"] >= 1 and d3["n_filled"] == d3["n_gaps"]
    assert d3["imputed_m"] > 300 and d3["imputed_on_route"] > 0.9
    assert d3["route_trips"] == 2
    assert max_x(d3["geometry"]) > 90  # along x = 100, the habit
    assert d3["depart_estimated"] >= d3["depart_after"] and d3["arrive_estimated"] <= d3["arrive_before"]
    # observed days are not imputed (beyond the short connections to the stays)
    full = out.trips.filter(pl.col("trip_id") == 0).row(0, named=True)
    assert full["imputed_m"] == 0 and full["observed_m"] > 450
    # the routine's route: edges observed on its trips
    assert out.routes["trips"].max() == 3  # the first stretch east was observed on all three days
    e = out.edges.filter(pl.col("trip_id") == 4)
    assert e["imputed"].any() and (~e["imputed"]).any() and e["seq"].to_list() == list(range(e.height))


def test_fill_empty(setup):
    net, trips, _, res = setup
    e = pl.col("trip_id") == 4  # day 3 without its pings
    tr = trips.with_columns(pl.when(e).then(pl.lit("empty")).otherwise("state").alias("state"),
                            pl.when(~e).then(pl.col("trajectory_ids")).alias("trajectory_ids"))
    assert pr.complete_trips(tr, res, net, gap_m=150).trips.filter(pl.col("trip_id") == 4).height == 0
    out = pr.complete_trips(tr, res, net, gap_m=150, fill_empty=True)
    d3 = out.trips.filter(pl.col("trip_id") == 4).row(0, named=True)
    assert max_x(d3["geometry"]) > 90 and d3["depart_estimated"] is None
    # without the preference: the shortest way, straight north
    plain = pr.complete_trips(tr, res, net, gap_m=150, fill_empty=True, preference=0.0)
    assert max_x(plain.trips.filter(pl.col("trip_id") == 4)["geometry"][0]) < 10
