"""Incremental DBSCAN vs scikit-learn, and incremental == one-shot."""

import numpy as np
import polars as pl
import pytest

import polaroam as pr
from polaroam.config import ClusterParams

R = 6371008.8


def random_stops(n_dev=15, seed=0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dev):
        centers = np.array([19.4, -99.1]) + rng.normal(0, 0.003, (6, 2))
        lat0 = rng.choice([0.0, 45.0, 70.0])  # exercise the longitude grid at high latitude
        for k in range(int(rng.integers(20, 250))):
            c = centers[rng.integers(6)] + np.array([lat0, 0])
            spread = rng.choice([5, 15, 40]) / 111_000
            p = c + rng.normal(0, spread, 2)
            rows.append((f"u{d}", k + 1, 1_600_000_000 + k * 7200, 1_600_000_000 + k * 7200 + 600, 600, p[0], p[1]))
    return pl.DataFrame(rows, schema=["uid", "stop_event", "start_timestamp", "end_timestamp", "stop_duration",
                                      "stop_latitude", "stop_longitude"], orient="row")


def sklearn_labels(df, eps, min_pts):
    from sklearn.cluster import DBSCAN

    out = {}
    for (uid,), g in df.group_by("uid"):
        X = np.radians(g.select("stop_latitude", "stop_longitude").to_numpy())
        m = DBSCAN(eps=eps / R, min_samples=min_pts, metric="haversine", algorithm="ball_tree").fit(X)
        core = np.zeros(len(X), bool)
        core[m.core_sample_indices_] = True
        for se, lab, c in zip(g["stop_event"], m.labels_, core):
            out[(uid, se)] = (lab, c)
    return out


def partition(pairs):
    groups = {}
    for key, lab in pairs:
        if lab >= 0:
            groups.setdefault((key[0], lab), set()).add(key)
    return sorted(sorted(v) for v in groups.values())


@pytest.mark.parametrize("eps", [10, 20, 50])
def test_min_points_2_equals_sklearn(eps):
    df = random_stops()
    got = pr.cluster_stops(df, ClusterParams(eps, 2))
    ref = sklearn_labels(df, eps, 2)
    mine = {(u, s): lab for u, s, lab in got.select("uid", "stop_event", "cluster_label").iter_rows()}
    assert partition(mine.items()) == partition((k, v[0]) for k, v in ref.items())
    assert {k for k, v in mine.items() if v < 0} == {k for k, v in ref.items() if v[0] < 0}


@pytest.mark.parametrize("min_pts", [3, 5])
def test_core_clusters_equal_sklearn(min_pts):
    df = random_stops(seed=1)
    got = pr.cluster_stops(df, ClusterParams(20, min_pts))
    ref = sklearn_labels(df, 20, min_pts)
    mine = {(u, s): lab for u, s, lab in got.select("uid", "stop_event", "cluster_label").iter_rows()}
    core = {k for k, v in ref.items() if v[1]}
    # Core points: identical partition. Border points: in some cluster iff sklearn puts them in one
    # (sklearn picks the first core neighbor in scan order; polaroam the nearest one).
    assert partition((k, mine[k]) for k in core) == partition((k, ref[k][0]) for k in core)
    assert {k for k, v in mine.items() if v < 0} == {k for k, v in ref.items() if v[0] < 0}


@pytest.mark.parametrize("min_pts", [2, 3, 4])
def test_incremental_equals_one_shot(min_pts):
    df = random_stops(seed=2)
    p = ClusterParams(20, min_pts)
    one = pr.cluster_stops(df, p, chunk_size=None).sort("uid", "stop_event")
    rng = np.random.default_rng(min_pts)
    df = df.with_columns(pl.Series("_b", rng.integers(0, 7, df.height)))
    db = pr.IncrementalDBSCAN(p)
    for b in range(7):
        db.insert(df.filter(pl.col("_b") == b).drop("_b"))
    assert db.labels().sort("uid", "stop_event").equals(one)


def test_state_roundtrip(tmp_path):
    df = random_stops(seed=3)
    db = pr.IncrementalDBSCAN()
    db.insert(df.filter(pl.col("stop_event") % 2 == 0))
    db.state.save(tmp_path)
    db2 = pr.IncrementalDBSCAN(state=pr.ClusterState.load(tmp_path))
    db2.insert(df.filter(pl.col("stop_event") % 2 == 1))
    assert db2.labels().sort("uid", "stop_event").equals(pr.cluster_stops(df).sort("uid", "stop_event"))


def test_connected_components_vs_scipy():
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as scc

    rng = np.random.default_rng(0)
    n = 3000
    a, b = rng.integers(0, n, 4000), rng.integers(0, n, 4000)
    got = pr.connected_components(pl.DataFrame({"uid": ["x"] * len(a), "a": a, "b": b}).cast({"a": pl.Int64, "b": pl.Int64}))
    _, lab = scc(coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)), directed=False)
    nodes = got["node"].to_numpy()
    roots = got["root"].to_numpy()
    for x, r in zip(nodes, roots):
        assert lab[x] == lab[r] and r == min(np.flatnonzero(lab == lab[x]))
