"""Map matching against a known grid and a brute-force HMM reference."""

import itertools
import math

import numpy as np
import polars as pl
import pytest

shapely = pytest.importorskip("shapely")
pytest.importorskip("scipy")

import polaroam as pr
from polaroam.geo import M_PER_DEG

LAT0, LON0 = 19.40, -99.15
KX = M_PER_DEG * math.cos(math.radians(LAT0))


def ll(x, y):
    """Metres east / north of the origin -> (lon, lat)."""
    return LON0 + x / KX, LAT0 + y / M_PER_DEG


def grid(n=6, step=100.0, bend=True):
    """Street centerlines of an n x n grid (two-way), one street with an extra vertex."""
    lines = []
    for i in range(n):
        for j in range(n):
            if i + 1 < n:
                lines.append([ll(i * step, j * step), ll((i + 1) * step, j * step)])
            if j + 1 < n:
                lines.append([ll(i * step, j * step), ll(i * step, (j + 1) * step)])
    if bend:  # a curved street from (0, 500) to (100, 600) through (20, 580)
        lines.append([ll(0, 500), ll(20, 580), ll(100, 600)])
    geoms = np.array([shapely.LineString(line) for line in lines], dtype=object)
    return pl.DataFrame({"geometry": shapely.to_wkt(geoms, rounding_precision=-1), "name": [f"s{k}" for k in range(len(lines))]})


def trajectory(path_xy, every=25.0, noise=4.0, seed=0, t0=1_700_000_000, speed=5.0):
    """Points every ``every`` m along a polyline (metres), with GPS noise; 1 point per (every/speed) s."""
    rng = np.random.default_rng(seed)
    line = shapely.LineString(path_xy)
    d = np.arange(0, line.length + 1e-9, every)
    xy = shapely.get_coordinates(shapely.line_interpolate_point(line, d)) + rng.normal(0, noise, (len(d), 2))
    coords = [list(ll(x, y)) for x, y in xy]
    ts = [t0 + int(k * every / speed) for k in range(len(d))]
    return coords, ts


def trips(*paths, **kw):
    rows = [trajectory(p, seed=i, **kw) for i, p in enumerate(paths)]
    return pl.DataFrame({"uid": [f"d{i}" for i in range(len(rows))], "trajectory_id": list(range(len(rows))),
                         "coordinates": [r[0] for r in rows], "timestamps": [r[1] for r in rows]},
                        schema_overrides={"coordinates": pl.List(pl.List(pl.Float64)), "timestamps": pl.List(pl.Int64)})


@pytest.fixture(scope="module")
def net():
    return pr.RoadNetwork(grid())


def test_network_from_lines(net):
    g = grid()
    assert net.edges.height == 2 * g.height  # every street two-way
    assert net.n_nodes == 36 + 1  # grid nodes + the curve's far end (100, 600)
    one = pr.RoadNetwork(g.with_columns(pl.Series("oneway", [True] + [False] * (g.height - 1))))
    assert one.edges.height == 2 * g.height - 1
    # u / v given, directed rows (as osmnx), WKB geometry
    e = net.edges.select("u", "v").with_columns(
        pl.Series("geometry", shapely.to_wkb(net._geoms)))
    assert pr.RoadNetwork(e, directed=True).edges.height == net.edges.height


def test_follows_known_route(net):
    # east along y=0 from x=0 to x=300, then north along x=300 to y=300
    tr = trips([(0, 0), (300, 0), (300, 300)])
    res = pr.match_trajectories(tr, net)
    t = res.trajectories.row(0, named=True)
    assert t["n_parts"] == 1 and t["n_matched"] == t["n_points"]
    assert t["matched_length_m"] == pytest.approx(600, abs=15)
    assert t["mean_distance_m"] < 10
    # the route covers 6 street blocks, in order, each (almost) entirely: 3 east, then 3 north
    cover = (res.edges.group_by("edge", maintain_order=True)
             .agg((pl.col("to_m") - pl.col("from_m")).sum().alias("m"), pl.col("u").first(), pl.col("v").first()))
    blocks = cover.filter(pl.col("m") > 50)
    assert blocks.height == 6 and cover.filter(pl.col("m") <= 50)["m"].sum() < 20  # GPS noise at the ends
    path = blocks["u"].to_list() + [blocks["v"][-1]]
    nodes = net._nodes.join(pl.DataFrame({"node_id": path}, schema={"node_id": net._nodes["node_id"].dtype}),
                            on="node_id", how="semi")
    assert nodes.height == 7  # a simple path through 7 intersections
    geom = shapely.from_wkt(t["geometry"])
    corner = shapely.Point(ll(300, 0))
    assert geom.distance(corner) < 1e-7  # passes exactly through the intersection
    # matched points lie on their edges
    pts = res.points
    on = shapely.distance(shapely.points(pts["matched_longitude"].to_numpy(), pts["matched_latitude"].to_numpy()),
                          net._geoms[pts["edge"].to_numpy()])
    assert np.all(on < 1e-9)


def test_curved_street_geometry_keeps_vertices(net):
    tr = trips([(0, 400), (0, 500), (20, 580), (100, 600)], every=15, noise=2)
    t = pr.match_trajectories(tr, net).trajectories.row(0, named=True)
    geom = shapely.from_wkt(t["geometry"])
    assert geom.distance(shapely.Point(ll(20, 580))) < 1e-7  # the street's interior vertex
    assert t["matched_length_m"] == pytest.approx(100 + math.hypot(20, 80) + math.hypot(80, 20), abs=20)


def test_gap_off_network_and_far_points(net):
    # second half 5 km away on another grid: no route, so matching restarts (2 parts)
    far = pr.RoadNetwork(pl.concat([grid(bend=False), grid(bend=False).with_columns(
        pl.Series("geometry", shapely.to_wkt(shapely.transform(
            shapely.from_wkt(grid(bend=False)["geometry"].to_numpy()), lambda c: c + [0.05, 0.0]))))]))
    a, ts_a = trajectory([(0, 0), (200, 0)])
    b, ts_b = trajectory([(0, 0), (200, 0)], seed=3)
    b = [[x + 0.05, y] for x, y in b]
    lost = [list(ll(250, 50))]  # 50 m from every street: no candidate, skipped
    tr = pl.DataFrame({"uid": ["d"], "trajectory_id": [0], "coordinates": [a + lost + b],
                       "timestamps": [ts_a + [ts_a[-1] + 10] + [ts_a[-1] + 20 + t - ts_b[0] for t in ts_b]]},
                      schema_overrides={"coordinates": pl.List(pl.List(pl.Float64))})
    t = pr.match_trajectories(tr, far, search_radius=40).trajectories.row(0, named=True)
    assert t["n_parts"] == 2 and t["n_matched"] == t["n_points"] - 1
    assert t["geometry"].startswith("MULTILINESTRING")


def test_modes_and_volumes(net):
    tr = trips([(0, 0), (300, 0)], [(0, 0), (300, 0)], [(0, 100), (0, 300)]).with_columns(
        pl.Series("mode_guess", ["walk", "motorized", "walk"]))
    res = pr.match_trajectories(tr, {"walk": net, "motorized": net})
    assert sorted(res.trajectories["mode_guess"].to_list()) == ["motorized", "walk", "walk"]
    vol = pr.edge_volumes(res.edges, net, by="mode_guess")
    assert vol["trajectories"].max() == 1 and {"geometry", "name"} <= set(vol.columns) | set(res.edges.columns)
    skip = pr.match_trajectories(tr, {"bike": net})
    assert skip.trajectories.height == 0


# ------------------------------------------------------------------ brute force reference
def reference_best(net, coords, ts, p):
    """Enumerate every candidate sequence; candidates and routes computed independently
    (shapely projection, full scipy shortest-path matrix)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    geoms = net._geoms
    e = net._edge_nodes.sort("edge")
    un, vn, length = e["un"].to_numpy(), e["vn"].to_numpy(), e["length_m"].to_numpy()
    n = net.n_nodes
    w = {}
    for a, b, L in zip(un, vn, length):
        if a != b:
            w[(a, b)] = min(w.get((a, b), np.inf), L)
    m = csr_matrix((list(w.values()), ([k[0] for k in w], [k[1] for k in w])), shape=(n, n))
    D = dijkstra(m, directed=True)

    def to_m(lon, lat, lat0):
        return (lon - LON0) * M_PER_DEG * math.cos(math.radians(lat0)), (lat - LAT0) * M_PER_DEG

    cands = []
    for lon, lat in coords:
        cs = []
        for ei, g in enumerate(geoms):
            xy = shapely.get_coordinates(g)
            pts = [to_m(x, y, lat) for x, y in xy]
            line = shapely.LineString(pts)
            P = shapely.Point(to_m(lon, lat, lat))
            d = line.distance(P)
            if d <= p.search_radius:
                frac = line.project(P) / line.length if line.length else 0.0
                cs.append((d, ei, frac * length[ei]))
        cs.sort()
        cands.append(cs[: p.max_candidates])
    best = (-np.inf, None)
    for seq in itertools.product(*cands):
        score = sum(-0.5 * (c[0] / p.sigma) ** 2 for c in seq)
        ok = True
        for k in range(1, len(seq)):
            (_, ea, oa), (_, eb, ob) = seq[k - 1], seq[k]
            lat1, lon1, lat2, lon2 = coords[k - 1][1], coords[k - 1][0], coords[k][1], coords[k][0]
            gc = 2 * 6371008.8 * math.asin(math.sqrt(math.sin(math.radians(lat2 - lat1) / 2) ** 2 + math.cos(
                math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(math.radians(lon2 - lon1) / 2) ** 2))
            same = ea == eb and ob >= oa - p.backtrack_m  # small step back on one edge: standing still
            route = max(ob - oa, 0.0) if same else length[ea] - oa + D[vn[ea], un[eb]] + ob
            if not np.isfinite(route) or route > gc * p.route_factor + 2 * p.search_radius:
                ok = False
                break
            score += -abs(route - gc) / p.beta
        if ok and score > best[0]:
            best = (score, [c[1] for c in seq])
    return best


@pytest.mark.parametrize("seed", range(15))
def test_viterbi_matches_brute_force(net, seed):
    rng = np.random.default_rng(seed)
    p = pr.MatchParams(search_radius=45, max_candidates=4, sigma=15, beta=60)
    start = rng.uniform(50, 450, 2)
    walk = [start]
    for _ in range(4):
        walk.append(walk[-1] + rng.normal(0, 45, 2))
    coords = [list(ll(x, y)) for x, y in walk]
    ts = [1_700_000_000 + 30 * k for k in range(len(walk))]
    tr = pl.DataFrame({"uid": ["d"], "trajectory_id": [0], "coordinates": [coords], "timestamps": [ts]},
                      schema_overrides={"coordinates": pl.List(pl.List(pl.Float64))})
    res = pr.match_trajectories(tr, net, p)
    _, ref_edges = reference_best(net, coords, ts, p)
    if ref_edges is None or res.trajectories["n_parts"][0] != 1 or res.trajectories["n_matched"][0] != len(coords):
        pytest.skip("random walk left the network or needed a restart")
    assert res.points.sort("step")["edge"].to_list() == ref_edges


def test_cli_match(tmp_path):
    from polaroam.cli import main

    grid().write_parquet(tmp_path / "roads.parquet")
    tr = trips([(0, 0), (300, 0)], [(0, 100), (0, 300)]).with_columns(pl.Series("mode_guess", ["walk", "bike"]))
    tr.write_parquet(tmp_path / "trips.parquet")
    assert main(["match", str(tmp_path / "trips.parquet"), str(tmp_path / "one"), "--network",
                 str(tmp_path / "roads.parquet")]) == 0
    assert pl.read_parquet(tmp_path / "one" / "trajectories" / "*.parquet")["n_parts"].to_list() == [1, 1]
    assert main(["match", str(tmp_path / "trips.parquet"), str(tmp_path / "modes"), "--network",
                 f"walk={tmp_path / 'roads.parquet'}", "--search-radius", "40"]) == 0
    out = pl.read_parquet(tmp_path / "modes" / "trajectories" / "*.parquet")
    assert out["mode_guess"].to_list() == ["walk"]  # no network given for bike
    assert any((tmp_path / "modes" / "edges").glob("*.parquet"))


def test_edge_volumes_min_fraction(net):
    # a trip ending 30 m into a block: counted on that block only without a coverage threshold
    tr = trips([(0, 0), (130, 0)], every=10, noise=0.5)
    res = pr.match_trajectories(tr, net)
    every = pr.edge_volumes(res.edges)
    half = pr.edge_volumes(res.edges, min_fraction=0.5)
    assert every.height == 2 and half.height == 1


def test_output_dir_matches_in_memory(net, tmp_path):
    tr = trips([(0, 0), (300, 0)], [(0, 100), (0, 300)], [(100, 0), (100, 400)]).with_columns(
        pl.Series("mode_guess", ["walk", "bike", "walk"]))
    p = pr.MatchParams(chunk_points=20)  # several chunks
    mem = pr.match_trajectories(tr, {"walk": net, "bike": net}, p)
    disk = pr.match_trajectories(tr, {"walk": net, "bike": net}, p, output_dir=tmp_path / "m")
    assert len(list((tmp_path / "m" / "edges").glob("*.parquet"))) > 2
    for name, key in [("trajectories", ["uid"]), ("points", ["uid", "step"]), ("edges", ["uid", "seq"])]:
        assert getattr(disk, name).collect().sort(key).equals(getattr(mem, name).sort(key)), name
