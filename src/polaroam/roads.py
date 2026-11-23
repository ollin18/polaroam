"""Map matching: snap trajectories to a road network.

A hidden Markov model (Newson & Krumm, 2009), vectorized over all trajectories:

1. **Candidates.** Each trajectory point is projected onto the road edges within
   ``search_radius`` (a grid join, as in the clustering), keeping the ``max_candidates``
   nearest edges. A candidate is a position on a *directed* edge: ``(edge, offset)``.
2. **Emission.** A candidate ``d`` meters from its point has log-probability
   ``-0.5 (d / sigma)^2``.
3. **Transition.** Between consecutive points, the route distance on the network between
   their candidates is compared with the straight distance between the points:
   ``-|route - straight| / beta``. Routes longer than ``route_factor`` times the straight
   distance (plus twice the search radius), or faster than ``max_speed_kmh``, are impossible.
   Node-to-node distances come from a bounded Dijkstra (numba) that runs only from the nodes
   needed, stops once its targets are reached, and runs sources in parallel.
4. **Viterbi.** The best candidate sequence is found step by step for all trajectories at
   once (polars joins). When no transition is possible (e.g. a gap the network cannot
   explain), the trajectory is split and matching restarts: the match has several parts.
5. **Paths.** Consecutive matched positions are joined by their shortest path on the network,
   which gives the matched line and the edges each trajectory traversed.

Points with no road within ``search_radius`` are skipped. Distances are geodesic (haversine)
for edge lengths and straight lines, and local planar for projecting a point onto a nearby
edge, so it works at any latitude and scale.

Install the ``roads`` extra (shapely, numba).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from .geo import M_PER_DEG, haversine

__all__ = ["MatchParams", "MatchResult", "RoadNetwork", "edge_volumes", "match_trajectories"]


@dataclass
class MatchParams:
    """Map-matching parameters. The defaults are tuned for LBS data (noisier than dedicated GPS):
    on one month of Mexico City, they match 86% of motorized trajectories in one piece.

    search_radius: roads farther than this from a point (m) are not candidates for it.
    max_candidates: nearest directed edges kept per point.
    sigma: GPS noise (m) in the emission probability.
    beta: tolerance (m) for the difference between route and straight distance.
    route_factor: a route longer than ``route_factor`` x the straight distance
        + 2 x ``search_radius`` is impossible.
    max_speed_kmh: a route faster than this between two timestamps is impossible (None: no limit).
    backtrack_m: on the same edge, moving back by up to this much (GPS jitter of a slow or
        stopped device) counts as standing still instead of looping around the block.
    chunk_points: points matched per chunk (bounds memory).
    """

    search_radius: float = 80.0
    max_candidates: int = 12
    sigma: float = 20.0
    beta: float = 100.0
    route_factor: float = 5.0
    max_speed_kmh: float | None = None
    backtrack_m: float = 40.0
    chunk_points: int = 200_000


def _geoms(col: pl.Series):
    import shapely

    if col.dtype == pl.Binary:
        return shapely.from_wkb(col.to_numpy())
    if col.dtype == pl.String:
        return shapely.from_wkt(col.to_numpy())
    return np.asarray(col.to_list(), dtype=object)


class RoadNetwork:
    """A directed road network for :func:`match_trajectories`.

    ``edges`` is a table of LineStrings (``geometry`` as WKB, WKT or shapely objects), one row
    per road edge, from any source (OpenStreetMap, a city's street centerlines...). Other
    columns (``highway``, ``name``...) are kept and returned with the matched edges.

    - With ``u`` and ``v`` columns, those are the node ids (each edge goes from ``u`` to
      ``v`` along its geometry). Otherwise nodes are the edge endpoints: lines must end at
      intersections (split at junctions, as street centerline datasets usually are).
    - ``directed=True`` uses each row as one direction only (e.g. osmnx edges, which list both
      directions of two-way streets). ``directed=False`` (default) makes every edge two-way,
      except those where the ``oneway`` column is true.

    >>> net = RoadNetwork(pl.read_parquet("streets.parquet"))                # doctest: +SKIP
    >>> net = RoadNetwork.from_osmnx(osmnx.graph_from_place("Coyoacán, Mexico"))  # doctest: +SKIP
    """

    def __init__(self, edges: pl.DataFrame, *, geometry: str = "geometry", u: str = "u", v: str = "v",
                 directed: bool = False, oneway: str | None = "oneway"):
        import shapely

        edges = edges.clone()
        geoms = _geoms(edges.get_column(geometry))
        kind = shapely.get_type_id(geoms)
        multi = kind == 5
        if multi.any():  # MultiLineStrings that form one line
            geoms[multi] = shapely.line_merge(geoms[multi])
            kind = shapely.get_type_id(geoms)
        if (kind != 1).any():
            raise ValueError(f"{int((kind != 1).sum())} road geometries are not single LineStrings.")
        attrs = edges.drop(geometry)
        has_nodes = u in edges.columns and v in edges.columns
        if not has_nodes:
            ends = np.stack([shapely.get_coordinates(shapely.get_point(geoms, 0)),
                             shapely.get_coordinates(shapely.get_point(geoms, -1))])
            key = pl.DataFrame({"x": np.round(ends[:, :, 0].ravel(), 7), "y": np.round(ends[:, :, 1].ravel(), 7)})
            node = key.join(key.unique(maintain_order=True).with_row_index("n"), on=["x", "y"],
                            how="left", maintain_order="left")["n"].to_numpy()
            attrs = attrs.with_columns(pl.Series(u, node[: len(geoms)]), pl.Series(v, node[len(geoms):]))
        attrs = attrs.with_row_index("source_row")
        if not directed:
            back = ~(attrs[oneway].cast(pl.Boolean).fill_null(False).to_numpy()) if oneway in attrs.columns \
                else np.ones(len(geoms), bool)
            rev = attrs.filter(pl.Series(back)).rename({u: v, v: u})
            rev = rev.select(attrs.columns)
            geoms = np.concatenate([geoms, shapely.reverse(geoms[back])])
            attrs = pl.concat([attrs.with_columns(pl.lit(False).alias("reversed")),
                               rev.with_columns(pl.lit(True).alias("reversed"))])
        edges_out = attrs.with_row_index("edge").with_columns(pl.col("edge").cast(pl.Int64))
        self.edges = edges_out.rename({u: "u", v: "v"}) if (u, v) != ("u", "v") else edges_out
        self._geoms = geoms
        self._build()

    @classmethod
    def from_osmnx(cls, graph) -> RoadNetwork:
        """From an osmnx graph (``osmnx.graph_from_place(..., network_type="drive")``)."""
        import osmnx as ox
        import shapely

        gdf = ox.graph_to_gdfs(graph, nodes=False).reset_index()
        cols = {c: gdf[c].map(lambda x: x if isinstance(x, (str, int, float, bool)) or x is None
                              else ",".join(map(str, x))).to_numpy()
                for c in ("highway", "name", "oneway") if c in gdf.columns}
        df = pl.DataFrame({"u": gdf["u"].to_numpy(), "v": gdf["v"].to_numpy(), **cols,
                           "geometry": shapely.to_wkb(gdf.geometry.to_numpy())})
        return cls(df, directed=True)

    @classmethod
    def read(cls, path: str | Path, **kwargs) -> RoadNetwork:
        """From a parquet or CSV file with a WKB / WKT ``geometry`` column."""
        p = str(path)
        return cls(pl.read_csv(p) if p.endswith(".csv") else pl.read_parquet(p), **kwargs)

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        import shapely

        from ._routing import csr

        xy, idx = shapely.get_coordinates(self._geoms, return_index=True)
        v = (pl.DataFrame({"edge": idx.astype(np.int64), "lon": xy[:, 0], "lat": xy[:, 1]})
             .with_columns(pl.int_range(pl.len()).over("edge").alias("i")))
        seg = (v.with_columns(pl.col("lon").shift(-1).over("edge").alias("lon2"),
                              pl.col("lat").shift(-1).over("edge").alias("lat2"))
               .drop_nulls("lon2")
               .with_columns(haversine("lat", "lon", "lat2", "lon2").alias("len"))
               .with_columns((pl.col("len").cum_sum().over("edge") - pl.col("len")).alias("cum0")))
        length = seg.group_by("edge").agg(pl.col("len").sum().alias("length_m"))
        self.edges = (self.edges.drop("length_m", strict=False)
                      .join(length, on="edge", how="left").with_columns(pl.col("length_m").fill_null(0.0)))
        self._segments = seg
        # vertex offsets along their edge; the last vertex is exactly at the edge length, so path
        # pieces ending at an edge's end compare equal to it
        self._vertices = (v.join(seg.select("edge", "i", pl.col("cum0").alias("cum")), on=["edge", "i"], how="left")
                          .join(length.rename({"length_m": "_end"}), on="edge", how="left")
                          .with_columns(pl.col("cum").fill_null(pl.col("_end")).fill_null(0.0)).drop("_end")
                          .sort("edge", "i"))
        # nodes and the routing graph (parallel edges: the shortest one)
        ids = pl.concat([self.edges["u"], self.edges["v"]]).unique(maintain_order=True)
        nodes = pl.DataFrame({"node_id": ids}).with_row_index("node")
        self._nodes = nodes
        e = (self.edges.select("edge", "u", "v", "length_m")
             .join(nodes.rename({"node_id": "u", "node": "un"}), on="u", how="left")
             .join(nodes.rename({"node_id": "v", "node": "vn"}), on="v", how="left"))
        self._edge_nodes = e.select("edge", pl.col("un").cast(pl.Int64), pl.col("vn").cast(pl.Int64), "length_m")
        best = (e.filter(pl.col("un") != pl.col("vn")).sort("length_m", "edge")
                .group_by("un", "vn", maintain_order=True).first())
        self._link = best.select(pl.col("un").cast(pl.Int64), pl.col("vn").cast(pl.Int64), "edge")
        n = nodes.height
        un = best["un"].to_numpy().astype(np.int64)
        self._graph = (*csr(n, un, best["vn"].to_numpy().astype(np.int64), best["length_m"].to_numpy()), n)
        self._arc_edge = best["edge"].to_numpy()[np.argsort(un, kind="stable")].astype(np.int64)  # edge per CSR arc
        self._index_cache: dict[float, tuple] = {}

    def _index(self, radius: float):
        """Edge pieces no longer than ``radius``, registered in every grid cell they touch."""
        if radius in self._index_cache:
            return self._index_cache[radius]
        lat_ref = min(89.0, float(self._segments.select(pl.max_horizontal(pl.col("lat").abs().max(),
                                                                         pl.col("lat2").abs().max())).item()))
        dlat = radius / M_PER_DEG
        dlon = radius / (M_PER_DEG * math.cos(math.radians(lat_ref)))
        n = (pl.col("len") / radius).ceil().clip(lower_bound=1).cast(pl.Int64)
        f0, f1 = pl.col("k") / pl.col("n"), (pl.col("k") + 1) / pl.col("n")
        pieces = (self._segments.with_columns(n.alias("n")).with_columns(pl.int_ranges(0, pl.col("n")).alias("k"))
                  .explode("k", empty_as_null=True)
                  .select("edge",
                          (pl.col("lon") + f0 * (pl.col("lon2") - pl.col("lon"))).alias("x1"),
                          (pl.col("lat") + f0 * (pl.col("lat2") - pl.col("lat"))).alias("y1"),
                          (pl.col("lon") + f1 * (pl.col("lon2") - pl.col("lon"))).alias("x2"),
                          (pl.col("lat") + f1 * (pl.col("lat2") - pl.col("lat"))).alias("y2"),
                          (pl.col("cum0") + f0 * pl.col("len")).alias("off"),
                          (pl.col("len") / pl.col("n")).alias("plen"))
                  .with_row_index("piece"))
        cells = (pieces.with_columns(
            pl.int_ranges((pl.min_horizontal("y1", "y2") / dlat).floor().cast(pl.Int64),
                          (pl.max_horizontal("y1", "y2") / dlat).floor().cast(pl.Int64) + 1).alias("ci"),
            pl.int_ranges((pl.min_horizontal("x1", "x2") / dlon).floor().cast(pl.Int64),
                          (pl.max_horizontal("x1", "x2") / dlon).floor().cast(pl.Int64) + 1).alias("cj"))
            .explode("ci", empty_as_null=True).explode("cj", empty_as_null=True))
        self._index_cache[radius] = (cells, dlat, dlon)
        return self._index_cache[radius]

    @property
    def n_nodes(self) -> int:
        return self._nodes.height

    def __repr__(self) -> str:
        return f"RoadNetwork({self.edges.height:,} directed edges, {self.n_nodes:,} nodes)"

    # ------------------------------------------------------------- routing
    def _node_distances(self, pairs: pl.DataFrame, predecessors: bool = False):
        """Shortest distances for ``src, dst, lim`` node pairs (inf beyond ``lim``).

        With ``predecessors``, also the node sequence of each path: a table ``pair, hop, node``
        (hop 0 is the target).
        """
        from ._routing import node_distances

        pairs = pairs.with_row_index("pair")
        src, dst = pairs["src"].to_numpy().astype(np.int64), pairs["dst"].to_numpy().astype(np.int64)
        lim = pairs["lim"].to_numpy().astype(np.float64)
        if not predecessors:
            return pairs.with_columns(pl.Series("d", node_distances(self._graph, src, dst, lim)))
        pair, hop, node = node_distances(self._graph, src, dst, lim, paths=True)
        path = pl.DataFrame({"pair": pair.astype(np.uint32), "hop": hop.astype(np.int64), "node": node})
        found = path.select("pair").unique().with_columns(pl.lit(0.0).alias("d"))
        return pairs.join(found, on="pair", how="left").with_columns(pl.col("d").fill_null(float("inf"))), path


@dataclass
class MatchResult:
    """Output of :func:`match_trajectories`.

    trajectories: one row per input trajectory: ``n_points``, ``n_matched`` (points with a
        road within the search radius), ``n_parts`` (1 unless matching had to restart),
        ``matched_length_m``, ``mean_distance_m`` (point to matched position) and the matched
        line ``geometry`` (WKT LINESTRING, or MULTILINESTRING when there are several parts).
    points: one row per matched point: its matched position, edge, offset along the edge and
        distance to the road.
    edges: every edge each trajectory traversed, in order, with the covered part
        (``from_m``, ``to_m``) and the network's edge attributes; see :func:`edge_volumes`.
    """

    trajectories: pl.DataFrame
    points: pl.DataFrame
    edges: pl.DataFrame
    extra: dict = field(default_factory=dict)


def _points(trajectories: pl.DataFrame) -> pl.DataFrame:
    """Explode trajectories (``coordinates`` [lon, lat] lists, ``timestamps``) into points."""
    t = trajectories.select("uid", "trajectory_id", "coordinates", "timestamps").with_row_index("key")
    return (t.explode("coordinates", "timestamps", empty_as_null=True)
            .with_columns(pl.int_range(pl.len()).over("key").alias("step"),
                          pl.col("coordinates").list.get(0).alias("lon"), pl.col("coordinates").list.get(1).alias("lat"),
                          pl.col("timestamps").alias("t"))
            .select("key", "step", "lon", "lat", "t").drop_nulls(["lon", "lat"]))


def _candidates(net: RoadNetwork, pts: pl.DataFrame, p: MatchParams) -> pl.DataFrame:
    cells, dlat, dlon = net._index(p.search_radius)
    r = p.search_radius
    q = (pts.lazy().with_columns((pl.col("lat") / dlat).floor().cast(pl.Int64).alias("_ci"),
                          (pl.col("lon") / dlon).floor().cast(pl.Int64).alias("_cj"))
         .join(pl.LazyFrame({"di": [-1, -1, -1, 0, 0, 0, 1, 1, 1], "dj": [-1, 0, 1] * 3}), how="cross")
         .select("key", "step", "lon", "lat", (pl.col("_ci") + pl.col("di")).alias("ci"),
                 (pl.col("_cj") + pl.col("dj")).alias("cj")))
    kx = (pl.col("lat") * (math.pi / 180)).cos() * M_PER_DEG
    ax, ay = (pl.col("x1") - pl.col("lon")) * kx, (pl.col("y1") - pl.col("lat")) * M_PER_DEG
    bx, by = (pl.col("x2") - pl.col("lon")) * kx, (pl.col("y2") - pl.col("lat")) * M_PER_DEG
    dx, dy = bx - ax, by - ay
    t = (-(ax * dx + ay * dy) / (dx * dx + dy * dy)).fill_nan(0.0).clip(0.0, 1.0)
    # streaming: the point x nearby-piece pairs are filtered batch by batch, never all in memory
    c = (q.join(cells.lazy(), on=["ci", "cj"])
         .with_columns(t.alias("_t"))
         .with_columns((ax + pl.col("_t") * dx).alias("_qx"), (ay + pl.col("_t") * dy).alias("_qy"))
         .with_columns((pl.col("_qx") ** 2 + pl.col("_qy") ** 2).sqrt().alias("dist"))
         .filter(pl.col("dist") <= r)
         .select("key", "step", "edge", "piece", "dist",
                 (pl.col("off") + pl.col("_t") * pl.col("plen")).alias("offset"),
                 (pl.col("lon") + pl.col("_qx") / kx).alias("mlon"),
                 (pl.col("lat") + pl.col("_qy") / M_PER_DEG).alias("mlat"))
         .collect(engine="streaming")
         .sort("key", "step", "edge", "dist", "piece")
         .unique(["key", "step", "edge"], keep="first", maintain_order=True)
         .sort("key", "step", "dist", "edge"))
    c = c.filter(pl.int_range(pl.len()).over("key", "step") < p.max_candidates).drop("piece")
    return c.with_columns(pl.int_range(pl.len()).over("key", "step").alias("c"))


def _match_chunk(net: RoadNetwork, pts: pl.DataFrame, p: MatchParams):
    cand = _candidates(net, pts, p)
    if cand.height == 0:
        return None
    # dense step numbers over the points that have candidates
    steps = (cand.select("key", "step").unique().sort("key", "step")
             .with_columns(pl.int_range(pl.len()).over("key").alias("s")))
    pts_m = pts.join(steps, on=["key", "step"])
    cand = (cand.join(steps, on=["key", "step"]).join(net._edge_nodes, on="edge", how="left")
            .with_columns((-0.5 * (pl.col("dist") / p.sigma) ** 2).alias("e"))
            .sort("key", "s", "c").with_row_index("_r").with_columns(pl.col("_r").cast(pl.Int64)))
    # straight distance and time between consecutive matched points
    gap = (pts_m.sort("key", "s").with_columns(
        haversine(pl.col("lat").shift(1).over("key"), pl.col("lon").shift(1).over("key"), "lat", "lon").alias("gc"),
        (pl.col("t") - pl.col("t").shift(1).over("key")).alias("dt"))
        .filter(pl.col("s") > 0).select("key", "s", "gc", "dt"))
    # transitions into step s come from candidates at step s - 1 (rows ``ga`` -> ``gb``)
    a = cand.select("key", pl.col("s") + 1, pl.col("_r").alias("ga"), pl.col("edge").alias("ea"),
                    pl.col("offset").alias("oa"), pl.col("vn").alias("va"), pl.col("length_m").alias("la"))
    b = cand.select("key", "s", pl.col("_r").alias("gb"), pl.col("edge").alias("eb"), pl.col("offset").alias("ob"),
                    pl.col("un").alias("ub"))
    maxroute = pl.col("gc") * p.route_factor + 2 * p.search_radius
    # all candidate pairs between consecutive points (up to max_candidates^2 per step): kept lazy
    # and run in the streaming engine, so only the node pairs and the possible transitions are stored
    pairs = (a.lazy().join(b.lazy(), on=["key", "s"]).join(gap.lazy(), on=["key", "s"])
             .with_columns(((pl.col("ea") == pl.col("eb")) & (pl.col("ob") >= pl.col("oa") - p.backtrack_m))
                           .alias("same"),
                           (maxroute - (pl.col("la") - pl.col("oa")) - pl.col("ob")).alias("lim")))
    need = (pairs.filter(~pl.col("same") & (pl.col("lim") >= 0))
            .group_by(pl.col("va").alias("src"), pl.col("ub").alias("dst")).agg(pl.col("lim").max())
            .collect(engine="streaming"))
    dist = net._node_distances(need) if need.height else need.with_columns(pl.lit(None, pl.Float64).alias("d"))
    route = (pl.when(pl.col("same")).then((pl.col("ob") - pl.col("oa")).clip(lower_bound=0.0))
             .otherwise(pl.col("la") - pl.col("oa") + pl.col("d") + pl.col("ob")))
    ok = pl.col("route").is_not_null() & pl.col("route").is_finite() & (pl.col("route") <= maxroute)
    if p.max_speed_kmh is not None:
        ok = ok & ((pl.col("dt") <= 0) | (pl.col("route") / pl.col("dt").cast(pl.Float64) <= p.max_speed_kmh / 3.6))
    tr = (pairs.join(dist.lazy().select(pl.col("src").alias("va"), pl.col("dst").alias("ub"), "d"),
                     on=["va", "ub"], how="left")
          .with_columns(route.alias("route"))
          .filter(ok)
          .select("ga", "gb", "d", (-(pl.col("route") - pl.col("gc")).abs() / p.beta).alias("w"))
          .collect(engine="streaming"))

    # ---- Viterbi (numba), one trajectory per task
    from ._routing import viterbi

    tr = tr.sort("gb", "ga")
    n = cand.height
    tptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(tr["gb"].to_numpy(), minlength=n), out=tptr[1:])
    key_ptr = np.append(cand.with_row_index("_i").group_by("key", maintain_order=True)
                        .agg(pl.col("_i").first())["_i"].to_numpy(), n).astype(np.int64)
    chosen, brk, back = viterbi(key_ptr, cand["s"].to_numpy().astype(np.int64), cand["e"].to_numpy(), tptr,
                                tr["ga"].to_numpy().astype(np.int64), tr["w"].to_numpy())
    ch = (cand.with_columns(pl.Series("brk", brk), pl.Series("_back", back))
          .filter(pl.Series(chosen))
          .join(tr.select(pl.col("ga").alias("_back"), pl.col("gb").alias("_r"), "d"), on=["_back", "_r"], how="left")
          .drop("_r", "_back")
          .sort("key", "s")
          .with_columns(pl.col("brk").cast(pl.Int64).cum_sum().over("key").alias("part")))
    # for the line: within a run on one edge, never step back (jitter within backtrack_m)
    run = ((pl.col("edge") != pl.col("edge").shift(1)) | pl.col("brk")).fill_null(True).cum_sum().over("key")
    ch = (ch.with_columns(run.alias("_run"))
          .with_columns(pl.col("offset").cum_max().over("key", "_run").alias("_line_off"))
          .with_columns(*[pl.when(pl.col("offset") == pl.col("_line_off")).then(pl.col(c)).otherwise(None)
                          .forward_fill().over("key", "_run").alias(f"line_{c}") for c in ("mlon", "mlat")])
          .drop("_run"))
    return ch, pts_m


def _paths(net: RoadNetwork, ch: pl.DataFrame) -> pl.DataFrame:
    """Edge pieces between consecutive matched positions: ``key, s, order, edge, from_m, to_m, last``."""
    prev = ch.select("key", pl.col("s") + 1, pl.col("edge").alias("ea"), pl.col("_line_off").alias("oa"),
                     pl.col("vn").alias("va"), pl.col("length_m").alias("la"))
    st = ch.filter(~pl.col("brk")).select("key", "s", pl.col("edge").alias("eb"), pl.col("_line_off").alias("ob"),
                                          pl.col("un").alias("ub"), "d").join(prev, on=["key", "s"])
    same = (pl.col("ea") == pl.col("eb")) & (pl.col("ob") >= pl.col("oa"))
    one = st.filter(same).select("key", "s", pl.lit(0, pl.Int64).alias("order"), pl.col("ea").alias("edge"),
                                 pl.col("oa").alias("from_m"), pl.col("ob").alias("to_m"), pl.lit(True).alias("last"))
    multi = st.filter(~same)
    pieces = [one]
    if multi.height:
        # the route length is known from the transition: search only that far
        pairs = (multi.group_by(pl.col("va").alias("src"), pl.col("ub").alias("dst"))
                 .agg((pl.col("d").max() * (1 + 1e-9) + 1e-6).alias("lim")))
        d, path = net._node_distances(pairs, predecessors=True)
        # path hops go backwards from dst: order them forward and turn node pairs into edges
        nodes = (path.join(d.select("pair", "src", "dst"), on="pair")
                 .sort("pair", "hop", descending=[False, True])
                 .with_columns(pl.col("node").shift(1).over("pair").alias("prev_node"),
                               pl.int_range(pl.len()).over("pair").alias("k"))
                 .filter(pl.col("prev_node").is_not_null())
                 .join(net._link.rename({"un": "prev_node", "vn": "node"}), on=["prev_node", "node"], how="left")
                 .select("src", "dst", "k", "edge"))
        mid = (multi.join(nodes, left_on=["va", "ub"], right_on=["src", "dst"], how="inner")
               .join(net._edge_nodes.select("edge", "length_m"), on="edge", how="left")
               .select("key", "s", pl.col("k").cast(pl.Int64).alias("order"), "edge",
                       pl.lit(0.0).alias("from_m"), pl.col("length_m").alias("to_m"), pl.lit(False).alias("last")))
        n_mid = mid.group_by("key", "s").agg(pl.len().cast(pl.Int64).alias("n"))
        first = multi.select("key", "s", pl.lit(0, pl.Int64).alias("order"), pl.col("ea").alias("edge"),
                             pl.col("oa").alias("from_m"), pl.col("la").alias("to_m"), pl.lit(False).alias("last"))
        lastp = (multi.join(n_mid, on=["key", "s"], how="left")
                 .select("key", "s", (pl.col("n").fill_null(0) + 1).alias("order"), pl.col("eb").alias("edge"),
                         pl.lit(0.0).alias("from_m"), pl.col("ob").alias("to_m"), pl.lit(True).alias("last")))
        pieces += [first, mid, lastp]
    return pl.concat(pieces).sort("key", "s", "order")


def match_trajectories(trajectories: pl.DataFrame | pl.LazyFrame, network: RoadNetwork | dict[str, RoadNetwork],
                       params: MatchParams | None = None, *, by: str = "mode_guess",
                       output_dir: str | Path | None = None, **kwargs) -> MatchResult:
    """Snap trajectories to a road network.

    ``trajectories`` is the output of the trajectory stage (``uid``, ``trajectory_id``,
    ``coordinates`` as ``[lon, lat]`` lists and ``timestamps``); keep every ping with
    ``simplify_tolerance=0`` for the most accurate matches. ``network`` is a
    :class:`RoadNetwork`, or a dict of networks per value of ``by`` (e.g.
    ``{"walk": walk_net, "bike": bike_net, "motorized": drive_net}``); trajectories whose
    ``by`` value has no network are skipped. Keyword arguments override
    :class:`MatchParams` fields.

    The results can be large (about 4 traversed edges per point). With ``output_dir``, each
    chunk is written as it is matched (``trajectories/``, ``points/``, ``edges/`` parquet
    parts) and nothing accumulates in memory; the result then holds LazyFrames scanning them.
    """
    p = params or MatchParams()
    if kwargs:
        p = MatchParams(**{**p.__dict__, **kwargs})
    df = trajectories.lazy().collect() if isinstance(trajectories, pl.LazyFrame) else trajectories
    nets = [(None, network)] if isinstance(network, RoadNetwork) else list(network.items())
    names = ("trajectories", "points", "edges")
    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        for n in names:
            (out / n).mkdir(parents=True, exist_ok=True)
            for old in (out / n).glob("part-*.parquet"):
                old.unlink()
    kept: list[list[pl.DataFrame]] = [[], [], []]
    part = 0
    for label, net in nets:
        sub = df if label is None else df.filter(pl.col(by) == label)  # one mode at a time
        if sub.height == 0:
            continue
        for tables in _match_chunks(sub, net, p, by):
            for i, t in enumerate(tables):
                if t.height == 0:
                    continue
                if out is not None:
                    t.write_parquet(out / names[i] / f"part-{part:05d}.parquet")
                else:
                    kept[i].append(t)
            part += 1
        del sub
    if out is not None:
        scan = [pl.scan_parquet(out / n / "*.parquet") if any((out / n).glob("*.parquet")) else pl.LazyFrame()
                for n in names]
        return MatchResult(*scan)
    return MatchResult(*(pl.concat(k, how="diagonal_relaxed") if k else pl.DataFrame() for k in kept))


def _match_chunks(df: pl.DataFrame, net: RoadNetwork, p: MatchParams, by: str):
    """Match ``df`` in chunks of about ``chunk_points`` points; yield each chunk's finished
    ``(trajectories, points, edges)`` tables."""
    keys = df.select("uid", "trajectory_id", *([by] if by in df.columns else [])).with_row_index("key")
    pts = _points(df)
    # chunks of whole trajectories with about chunk_points points
    sizes = pts.group_by("key").agg(pl.len().alias("n")).sort("key")
    sizes = sizes.with_columns((sizes["n"].cum_sum() // p.chunk_points).alias("chunk"))
    attrs = net.edges.drop("length_m")
    cols = ["uid", "trajectory_id", *([by] if by in keys.columns else []), "n_points", "n_matched", "n_parts",
            "matched_length_m", "mean_distance_m", "geometry"]
    for ks in sizes.partition_by("chunk"):
        ids = keys.join(ks.select("key"), on="key", how="semi")  # key, uid, trajectory_id (+ mode)
        tr = ids.join(ks.select("key", pl.col("n").cast(pl.UInt32).alias("n_points")), on="key", how="left")
        sub = pts.join(ks.select("key"), on="key", how="semi")
        res = _match_chunk(net, sub, p)
        if res is None:
            tr = tr.with_columns(pl.lit(0, pl.UInt32).alias("n_matched"), pl.lit(0, pl.UInt32).alias("n_parts"),
                                 pl.lit(0.0).alias("matched_length_m"), pl.lit(None, pl.Float64).alias("mean_distance_m"),
                                 pl.lit(None, pl.String).alias("geometry"))
            yield tr.select(cols), pl.DataFrame(), pl.DataFrame()
            continue
        ch, pts_m = res
        pieces = _paths(net, ch)
        points = (ids.join(ch.join(pts_m.select("key", "s", "step", "lon", "lat", "t"), on=["key", "s"])
                           .select("key", "step", pl.col("t").alias("timestamp"), pl.col("lon").alias("longitude"),
                                   pl.col("lat").alias("latitude"), pl.col("mlon").alias("matched_longitude"),
                                   pl.col("mlat").alias("matched_latitude"), "edge",
                                   pl.col("offset").alias("offset_m"), pl.col("dist").alias("distance_m"), "part"),
                           on="key").drop("key"))
        edges = (ids.join(pieces.join(ch.select("key", "s", "part"), on=["key", "s"]), on="key")
                 .sort("key", "s", "order")
                 .with_columns(pl.int_range(pl.len()).over("key").alias("seq"))
                 .join(net._edge_nodes.select("edge", "length_m"), on="edge", how="left")
                 .join(attrs, on="edge", how="left")
                 .drop("key", "order", "last"))
        # geometry: start point of each part, then for every piece its interior vertices and its end
        verts = (pieces.lazy().join(net._vertices.lazy(), on="edge")
                 .filter((pl.col("cum") > pl.col("from_m"))
                         & ((pl.col("cum") < pl.col("to_m")) | ((pl.col("cum") == pl.col("to_m")) & ~pl.col("last"))))
                 .select("key", "s", "order", "cum", "lon", "lat").collect(engine="streaming"))
        ends = (pieces.filter(pl.col("last")).join(ch.select("key", "s", "line_mlon", "line_mlat"), on=["key", "s"])
                .select("key", "s", "order", pl.lit(float("inf")).alias("cum"), pl.col("line_mlon").alias("lon"),
                        pl.col("line_mlat").alias("lat")))
        starts = ch.filter(pl.col("brk")).select("key", "s", pl.lit(-1, pl.Int64).alias("order"),
                                                 pl.lit(0.0).alias("cum"), pl.col("line_mlon").alias("lon"),
                                                 pl.col("line_mlat").alias("lat"))
        line = (pl.concat([starts, verts, ends]).join(ch.select("key", "s", "part"), on=["key", "s"])
                .sort("key", "part", "s", "order", "cum")
                .filter(~((pl.col("lon") == pl.col("lon").shift(1)) & (pl.col("lat") == pl.col("lat").shift(1)))
                        .over("key", "part").fill_null(True))
                .group_by("key", "part", maintain_order=True)
                .agg(pl.format("{} {}", "lon", "lat").str.join(", ").alias("_wkt"), pl.len().alias("_n"))
                .filter(pl.col("_n") >= 2))
        geo = line.group_by("key", maintain_order=True).agg(pl.col("_wkt"), pl.len().alias("n_parts")).with_columns(
            pl.when(pl.col("n_parts") == 1)
            .then(pl.concat_str(pl.lit("LINESTRING ("), pl.col("_wkt").list.first(), pl.lit(")")))
            .otherwise(pl.concat_str(pl.lit("MULTILINESTRING (("), pl.col("_wkt").list.join("), ("), pl.lit("))")))
            .alias("geometry")).drop("_wkt")
        stats = ch.group_by("key").agg(pl.len().cast(pl.UInt32).alias("n_matched"),
                                       pl.col("dist").mean().alias("mean_distance_m"))
        length = pieces.group_by("key").agg((pl.col("to_m") - pl.col("from_m")).sum().alias("matched_length_m"))
        tr = (tr.join(stats, on="key", how="left").join(length, on="key", how="left").join(geo, on="key", how="left")
              .with_columns(pl.col("n_matched").fill_null(0), pl.col("matched_length_m").fill_null(0.0),
                            pl.col("n_parts").fill_null(0).cast(pl.UInt32)))
        yield tr.select(cols), points, edges


def edge_volumes(edges: pl.DataFrame | pl.LazyFrame, network: RoadNetwork | None = None, by: str | None = None, *,
                 min_fraction: float = 0.0) -> pl.DataFrame:
    """Trajectories per edge (each trajectory counted once per edge), optionally per ``by``
    (e.g. ``mode_guess``); with ``network``, the edge ``geometry`` (WKB) is added.

    ``min_fraction`` counts a trajectory on an edge only if it covered at least that share of
    the edge's length (e.g. 0.5 so that a few meters at the start of a long highway edge
    do not light up the whole edge on a map).
    """
    keys = ["edge", *([by] if by else [])]
    per = (edges.lazy().select("uid", "trajectory_id", *keys, "from_m", "to_m", "length_m")
           .group_by("uid", "trajectory_id", *keys)
           .agg(((pl.col("to_m") - pl.col("from_m")).sum() / pl.col("length_m").first().clip(lower_bound=1e-9))
                .alias("_cover")))
    if min_fraction > 0:
        per = per.filter(pl.col("_cover") >= min_fraction)
    vol = (per.group_by(keys).agg(pl.len().cast(pl.UInt32).alias("trajectories"))
           .sort("trajectories", descending=True).collect(engine="streaming"))
    if network is not None:
        import shapely

        vol = vol.join(pl.DataFrame({"edge": network.edges["edge"],
                                     "geometry": shapely.to_wkb(network._geoms)}), on="edge", how="left")
    return vol
