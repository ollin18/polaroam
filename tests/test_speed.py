"""Trajectory speed features, timestamps and mode guess on trips with known answers."""

import math

import polars as pl
import pytest

import polaroam as pr

M_PER_DEG = 6371008.8 * math.pi / 180
LAT0, LON0, T0 = 19.40, -99.15, 1_700_000_000


def trip(speeds_kmh, dt_s=60, uid="u", start=T0, gap_before_s=0):
    """Pings along a parallel; ``speeds_kmh[i]`` is the speed between ping i and i+1."""
    rows, x, t = [(uid, LAT0, LON0, (start + gap_before_s) * 1000)], 0.0, start + gap_before_s
    for v in speeds_kmh:
        x += v / 3.6 * dt_s
        t += dt_s
        rows.append((uid, LAT0, LON0 + x / (M_PER_DEG * math.cos(math.radians(LAT0))), t * 1000))
    return pl.DataFrame(rows, schema=["uid", "latitude", "longitude", "timestamp"], orient="row")


NO_STOPS = pl.DataFrame(schema={"uid": pl.String, "stop_event": pl.Int64, "start_timestamp": pl.Int64,
                                "end_timestamp": pl.Int64, "stop_latitude": pl.Float64, "stop_longitude": pl.Float64})


def one(pings, **kw):
    out = pr.extract_trajectories(pings, NO_STOPS, **kw)
    assert out.height == 1
    return out.row(0, named=True)


@pytest.mark.parametrize("v,mode", [(5, "walk"), (15, "bike"), (45, "motorized")])
def test_constant_speed(v, mode):
    r = one(trip([v] * 20))
    for c in ("speed_mean_kmh", "speed_median_kmh", "speed_p85_kmh", "speed_max_kmh"):
        assert r[c] == pytest.approx(v, rel=1e-3), c
    assert r["speed_cv"] == pytest.approx(0, abs=1e-3)
    assert r["frac_slow"] == (1.0 if v < 7 else 0.0)
    assert r["straightness"] == pytest.approx(1, abs=1e-6)
    assert r["ping_gap_median_s"] == 60 and r["n_jumps"] == 0
    assert r["mode_guess"] == mode


def test_stop_and_go_is_time_weighted():
    # 10 minutes at 40 km/h, 10 minutes at 4 km/h -> half the time slow, high variability
    r = one(trip([40] * 10 + [4] * 10))
    assert r["frac_slow"] == pytest.approx(0.5, abs=0.06)
    assert r["speed_mean_kmh"] == pytest.approx(22, rel=1e-3)
    assert r["speed_p85_kmh"] == pytest.approx(40, rel=0.01)
    assert r["speed_cv"] > 0.7
    assert r["mode_guess"] == "motorized"


def test_gps_jump_splits_the_trajectory():
    p = trip([5] * 20)
    p = p.with_columns(pl.when(pl.int_range(pl.len()) == 10).then(pl.col("latitude") + 0.5)  # ~55 km away
                       .otherwise(pl.col("latitude")).alias("latitude"))
    t = pr.extract_trajectories(p, NO_STOPS)
    # the walk is split around the jumped ping; no piece contains the 55 km excursion
    assert t.height == 2 and (t["length_m"] < 1000).all()
    assert (t["mode_guess"] == "walk").all() and (t["n_jumps"] == 0).all()


def test_ping_pong_is_not_a_mega_trajectory():
    # 3 minutes bouncing every 5 s between a true position and a fallback point 5 km away
    rows = [("u", LAT0 + (0.045 if i % 2 else 0.0), LON0, (T0 + 5 * i) * 1000) for i in range(36)]
    p = pl.DataFrame(rows, schema=["uid", "latitude", "longitude", "timestamp"], orient="row")
    t = pr.extract_trajectories(p, NO_STOPS, min_length_m=0)
    assert t.height == 0 or (t["length_m"] < 100).all()     # nothing but jitter-free single points
    assert pr.extract_trajectories(p, NO_STOPS).height == 0  # and the default 100 m filter drops them


def test_windows_tame_gps_noise():
    # pings every 2 s with alternating 20 m GPS error while walking: raw step speeds reach ~40 km/h
    p = trip([5] * 150, dt_s=2).with_columns(
        (pl.col("latitude") + (pl.int_range(pl.len()) % 2) * 20 / M_PER_DEG).alias("latitude"))
    r = one(p, speed_window=30)
    assert r["speed_p85_kmh"] < 8


def test_timestamps_align_with_vertices_and_wkt_m():
    r = one(trip([30, 30, 30, 5, 5, 5, 30, 30]), simplify_tolerance=0, wkt_measure=True)
    assert len(r["timestamps"]) == len(r["coordinates"]) == r["n_points"] == 9
    assert r["timestamps"] == sorted(r["timestamps"]) and r["timestamps"][0] == T0
    assert r["geometry"].startswith("LINESTRING M (") and r["geometry"].count(",") == 8
    assert r["geometry"].split(", ")[1].split()[2] == str(T0 + 60)


def test_filters(pings):
    base = pr.detect_stops(pings, trajectories=True)[1]
    filt = pr.detect_stops(pings, trajectories=True,
                           trajectory_params=pr.TrajectoryParams(min_pings=2, min_length_m=100))[1]
    assert 0 < filt.height < base.height
    assert (filt["n_pings"] >= 2).all() and (filt["length_m"] >= 100).all()
    assert filt["trajectory_id"].to_list() == list(range(1, filt.height + 1)) or filt["uid"].n_unique() > 1


def test_guess_mode_thresholds():
    t = pr.extract_trajectories(trip([15] * 20), NO_STOPS)
    assert t["mode_guess"][0] == "bike"
    assert pr.guess_mode(t, bike_p85_kmh=12)["mode_guess"][0] == "motorized"


def test_long_gaps_do_not_count_as_slow_time():
    # leave, wait 20 minutes without pings (gap), then drive 10 minutes at 40 km/h
    p = trip([40] * 10)
    first = p.head(1).with_columns(pl.col("timestamp") - 20 * 60 * 1000)
    r = one(pl.concat([first, p]))
    assert r["speed_p85_kmh"] == pytest.approx(40, rel=0.01)
    assert r["frac_slow"] == 0 and r["speed_median_kmh"] == pytest.approx(40, rel=0.01)


def test_short_fast_end_is_not_walking():
    # slow for 20 minutes then 2 minutes at 60 km/h: p85 is low, but no walker reaches 60 km/h
    r = one(trip([3] * 20 + [60] * 2))
    assert r["speed_p85_kmh"] < 8 and r["mode_guess"] == "motorized"


def test_distance_across_long_gaps_is_not_walking():
    # slow pings, then 30 km covered across a 12-minute gap (unobserved, < 250 km/h): not a walk
    slow = trip([4] * 10)
    last = slow.tail(1)
    far = last.with_columns(pl.col("longitude") + 0.28, pl.col("timestamp") + 12 * 60 * 1000)
    after = pl.concat([far, far.with_columns(pl.col("timestamp") + 60_000, pl.col("longitude") + 0.0001)])
    r = one(pl.concat([slow, after]))
    assert r["speed_p85_kmh"] < 8 and r["speed_max_kmh"] < 15
    assert r["mode_guess"] == "motorized"


def test_robust_speed_on_noisy_sparse_walk():
    """A 5 km/h walk with pings a minute apart, every other fix jumping ~150 m sideways (Wi-Fi /
    cell positioning): short windows read the hops as 15-20 km/h; the robust speed does not."""
    p = trip([5] * 24).with_columns(
        pl.when(pl.int_range(pl.len()) % 2 == 1).then(pl.col("latitude") + 150 / M_PER_DEG)
        .otherwise(pl.col("latitude")).alias("latitude"))
    r = one(p)
    assert r["speed_p85_kmh"] > 8           # the short-window cruising speed is noise
    assert r["speed_p85_robust_kmh"] < 8    # 3 pings per window average the hops out
    assert r["mode_guess"] == "walk"
    # with the earlier rule (no robust column), the same walk was a bike
    assert pr.guess_mode(pl.DataFrame([r]).drop("speed_p85_robust_kmh"))["mode_guess"][0] == "bike"


def test_robust_speed_equals_short_on_dense_pings():
    # pings every 5 s: 3 gaps are shorter than speed_window, so both use the same windows
    r = one(trip([15] * 60, dt_s=5))
    assert r["speed_p85_robust_kmh"] == r["speed_p85_kmh"]


def test_motorized_decided_by_short_windows():
    # stop-and-go bus, 1 minute at 40 km/h then 4 minutes stopped: the long robust windows
    # average in the stops, but motorized is decided by the short windows
    r = one(trip(([40] + [0] * 4) * 6))
    assert r["speed_p85_robust_kmh"] < r["speed_p85_kmh"] and r["mode_guess"] == "motorized"
