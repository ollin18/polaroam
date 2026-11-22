"""Stop detection: SQL semantics and incremental == one-shot."""

import itertools

import numpy as np
import polars as pl
import pytest

import polaroam as pr
from polaroam.config import StopParams
from tests.conftest import hav


def sql_reference(df: pl.DataFrame, D=20.0, T=3600.0, MIN=300.0) -> pl.DataFrame:
    """Row-by-row Python port of the gc-lbs BigQuery stop query (oracle).

    The helper closures are only called inside the loop iteration that defines them.
    """
    out = []
    df = df.unique().sort("uid", "timestamp", "latitude", "longitude")
    for (uid,), g in df.group_by("uid", maintain_order=True):
        lat, lon, ts = g["latitude"].to_list(), g["longitude"].to_list(), g["timestamp"].to_list()
        n = len(ts)
        dist = [hav(lat[i], lon[i], lat[i + 1], lon[i + 1]) if i < n - 1 else None for i in range(n)]
        td = [(ts[i + 1] - ts[i]) / 1000 if i < n - 1 else None for i in range(n)]

        def ok(i):  # SQL 3-valued: None = NULL
            if i < 0 or dist[i] is None:
                return None
            return dist[i] <= D and td[i] <= T

        stat = []
        for i in range(n):
            a, b = ok(i), ok(i - 1)
            stat.append(True if (a or b) else (None if (a is None or b is None) else False))
        cur, ids = 0, []
        for i in range(n):
            prev = stat[i - 1] if i > 0 else None
            start = stat[i] is True and (prev is None or prev is False or dist[i - 1] > D)
            cur += start
            ids.append(cur)
        groups = {}
        for i in range(n):
            if stat[i] is True and ids[i] > 0:
                groups.setdefault(ids[i], []).append(i)

        def agg(idx):
            t = [ts[i] for i in idx]
            la, lo = sorted(lat[i] for i in idx), sorted(lon[i] for i in idx)
            m = (len(idx) - 1) // 2
            return min(t), max(t), la[m], lo[m]

        stops = []
        for idx in groups.values():
            s, e, mla, mlo = agg(idx)
            if e - s < MIN * 1000:
                continue
            idx2 = [i for i in idx if hav(lat[i], lon[i], mla, mlo) <= D]
            if not idx2:
                continue
            s, e, mla, mlo = agg(idx2)
            if e - s >= MIN * 1000:
                stops.append((s, e, mla, mlo, len(idx2)))
        for k, (s, e, mla, mlo, npng) in enumerate(sorted(stops), 1):
            out.append((uid, k, s // 1000, e // 1000, e // 1000 - s // 1000, mla, mlo, npng))
    return pl.DataFrame(out, schema=dict(zip(pr.stops.STOP_COLUMNS, [pl.String, pl.Int64, pl.Int64, pl.Int64,
                                                                     pl.Int64, pl.Float64, pl.Float64, pl.UInt32])),
                        orient="row")


@pytest.mark.parametrize("D,T,MIN", [(20, 3600, 300), (50, 600, 120), (10, 7200, 900)])
def test_matches_sql_oracle(pings, D, T, MIN):
    sub = pings.filter(pl.col("uid") < "dev008")
    got = pr.detect_stops(sub, StopParams(D, T, MIN)).sort("uid", "stop_event")
    exp = sql_reference(sub, D, T, MIN).sort("uid", "stop_event")
    assert got.height > 50
    assert got.equals(exp)


def test_timestamp_units(pings):
    a = pr.detect_stops(pings.head(5000))
    b = pr.detect_stops(pings.head(5000).with_columns(pl.col("timestamp") // 1000))  # seconds
    c = pr.detect_stops(pings.head(5000).with_columns(pl.from_epoch("timestamp", "ms").alias("datetime"))
                        .drop("timestamp"))
    # second-resolution input loses sub-second info only
    assert a.equals(c)
    assert a.select("uid", "stop_event").equals(b.select("uid", "stop_event"))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_incremental_equals_one_shot(pings, seed):
    batch_s, batch_t = pr.detect_stops(pings, trajectories=True)
    rng = np.random.default_rng(seed)
    tmin, tmax = pings["timestamp"].min(), pings["timestamp"].max()
    cuts = np.sort(rng.integers(tmin, tmax, 15))
    edges = [tmin - 1, *cuts.tolist(), tmax]
    det = pr.StopDetector(trajectories=True)
    S, T = [], []
    for a, b in itertools.pairwise(edges):
        o = det.update(pings.filter((pl.col("timestamp") > a) & (pl.col("timestamp") <= b)))
        S.append(o.stops), T.append(o.trajectories)
    o = det.flush()
    S.append(o.stops), T.append(o.trajectories)
    assert pl.concat(S).sort("uid", "stop_event").equals(batch_s.sort("uid", "stop_event"))
    assert pl.concat(T).sort("uid", "trajectory_id").equals(batch_t.sort("uid", "trajectory_id"))


def test_state_roundtrip(tmp_path, pings):
    half = pings["timestamp"].median()
    det = pr.StopDetector()
    a = det.update(pings.filter(pl.col("timestamp") <= half)).stops
    det.state.save(tmp_path)
    det2 = pr.StopDetector(state=pr.StopState.load(tmp_path))
    b = det2.update(pings.filter(pl.col("timestamp") > half)).stops
    c = det2.flush().stops
    assert pl.concat([a, b, c]).sort("uid", "stop_event").equals(pr.detect_stops(pings).sort("uid", "stop_event"))
