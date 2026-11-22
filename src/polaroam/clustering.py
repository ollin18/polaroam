"""Per-device DBSCAN of stops, polars native and incremental.

DBSCAN (haversine, ``eps`` meters, ``min_points`` including the point itself)
is computed as a graph problem with column-wise operations only:

1. **Neighbors**: stops are hashed to a grid whose cells are at least ``eps``
   wide, so every pair within ``eps`` lies in the same or adjacent cells. A
   join on ``(uid, cell)`` over the 9 neighboring cells plus an exact
   haversine filter yields the ε-graph.
2. **Core points**: ``degree + 1 >= min_points``.
3. **Clusters**: connected components of the core–core graph, by min-label
   propagation with pointer jumping (a handful of joins per iteration).
4. **Border points** join the cluster of their nearest core neighbor
   (deterministic; with the default ``min_points=2`` there are no border
   points, and the result is identical to scikit-learn / BigQuery
   ``ST_CLUSTERDBSCAN``).

Incremental DBSCAN (in the spirit of Ester et al., 1998): inserting stops
only touches their ε-neighborhood. New edges can only increase degrees, turn
points into cores and *merge* clusters - never split them - so the state
(degree, core flag, component, nearest core) is updated using only the edges
of the new stops. Inserting stops in any number of batches gives exactly the
same clusters as clustering everything at once.

Output labels follow the gc-lbs convention: ``cluster_label`` is the dense
rank of the cluster centroid ``(lat, lon)`` within each device (``-1`` =
noise), ``cluster_latitude/longitude`` is the median of the member stops.
"""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from . import engine
from .config import ClusterParams
from .geo import M_PER_DEG, haversine, tie_exact_mean, with_group_row

CLUSTER_COLUMNS = [
    "uid", "stop_event", "start_timestamp", "end_timestamp", "stop_duration",
    "stop_latitude", "stop_longitude", "cluster_label", "cluster_counts",
    "cluster_latitude", "cluster_longitude",
]

_STOP_PAYLOAD = ["uid", "stop_event", "start_timestamp", "end_timestamp", "stop_duration",
                 "stop_latitude", "stop_longitude"]

_POINT_SCHEMA = {
    "uid": pl.String, "stop_event": pl.Int64, "start_timestamp": pl.Int64, "end_timestamp": pl.Int64,
    "stop_duration": pl.Int64, "stop_latitude": pl.Float64, "stop_longitude": pl.Float64,
    "_cy": pl.Int64, "_cx": pl.Int64, "_deg": pl.Int64, "_core": pl.Boolean,
    "_comp": pl.Int64, "_anchor": pl.Int64, "_adist": pl.Float64,
}
_DEVICE_SCHEMA = {"uid": pl.String, "_lat_ref": pl.Float64}
_OFFSETS = pl.DataFrame({"_oy": [-1, -1, -1, 0, 0, 0, 1, 1, 1], "_ox": [-1, 0, 1, -1, 0, 1, -1, 0, 1]},
                        schema={"_oy": pl.Int64, "_ox": pl.Int64})


def connected_components(edges: pl.DataFrame, use_cuda: bool = False) -> pl.DataFrame:
    """Components of an undirected graph given as ``uid, a, b`` (Int64 node ids).

    Returns ``uid, node, root`` with ``root`` the smallest node id of the
    component. Each round contracts every node into its smallest neighbor
    (with pointer jumping) and rewrites the edge list on the contracted
    graph, so dense groups collapse in the first round. Joins only.
    """
    out_schema = {"uid": edges.schema.get("uid", pl.String),
                  "node": pl.Int64, "root": pl.Int64}
    e = (edges.lazy().select("uid", pl.min_horizontal("a", "b").alias("a"), pl.max_horizontal("a", "b").alias("b"))
         .filter(pl.col("a") != pl.col("b")).unique())
    e = engine.collect(e, use_cuda)
    if e.height == 0:
        return pl.DataFrame(schema=out_schema)
    lab = (pl.concat([e.select("uid", pl.col("a").alias("node")), e.select("uid", pl.col("b").alias("node"))])
           .unique().with_columns(pl.col("node").alias("root")))
    while e.height:
        m = engine.collect(
            pl.concat([e.lazy(), e.lazy().select("uid", pl.col("b").alias("a"), pl.col("a").alias("b"))])
            .group_by("uid", "a").agg(pl.col("b").min().alias("t"))
            .select("uid", pl.col("a").alias("node"), pl.min_horizontal("a", "t").alias("t")), use_cuda)
        while True:  # pointer jumping: t <- t[t]
            m2 = (m.join(m.select("uid", pl.col("node").alias("t"), pl.col("t").alias("_t2")), on=["uid", "t"], how="left")
                  .with_columns(pl.coalesce("_t2", "t").alias("t")).drop("_t2"))
            if (m2.get_column("t") != m.get_column("t")).sum() == 0:
                break
            m = m2
        e = engine.collect(
            e.lazy().join(m.lazy().select("uid", pl.col("node").alias("a"), pl.col("t").alias("_ta")), on=["uid", "a"])
            .join(m.lazy().select("uid", pl.col("node").alias("b"), pl.col("t").alias("_tb")), on=["uid", "b"])
            .select("uid", pl.min_horizontal("_ta", "_tb").alias("a"), pl.max_horizontal("_ta", "_tb").alias("b"))
            .filter(pl.col("a") != pl.col("b")).unique(), use_cuda)
        lab = (lab.join(m.rename({"node": "root"}), on=["uid", "root"], how="left")
               .with_columns(pl.coalesce("t", "root").alias("root")).drop("t"))
    return lab


def _lat_ref(max_abs_lat: pl.Expr) -> pl.Expr:
    # Reference latitude (poleward of every stop of the device) for the lon cell width.
    return (((max_abs_lat + 0.01) / 5).ceil() * 5).clip(upper_bound=89.5)


def _cells(lf: pl.LazyFrame, eps: float) -> pl.LazyFrame:
    return lf.with_columns(
        (pl.col("stop_latitude") * (M_PER_DEG / eps)).floor().cast(pl.Int64).alias("_cy"),
        (pl.col("stop_longitude") * (pl.col("_lat_ref") * (math.pi / 180)).cos() * (M_PER_DEG / eps))
        .floor().cast(pl.Int64).alias("_cx"),
    )


class ClusterState:
    """Incremental DBSCAN state: every stop with its graph attributes."""

    def __init__(self, points: pl.DataFrame | None = None, devices: pl.DataFrame | None = None):
        self.points = points if points is not None else pl.DataFrame(schema=_POINT_SCHEMA)
        self.devices = devices if devices is not None else pl.DataFrame(schema=_DEVICE_SCHEMA)

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.points.write_parquet(d / "cluster_points.parquet")
        self.devices.write_parquet(d / "cluster_devices.parquet")

    @classmethod
    def load(cls, directory: str | Path) -> ClusterState:
        d = Path(directory)
        if not (d / "cluster_points.parquet").exists():
            return cls()
        return cls(pl.read_parquet(d / "cluster_points.parquet"), pl.read_parquet(d / "cluster_devices.parquet"))


class IncrementalDBSCAN:
    """Per-device DBSCAN that accepts new stops batch after batch.

    >>> db = IncrementalDBSCAN(ClusterParams(distance=20, min_points=2))
    >>> db.insert(stops_day_1)            # doctest: +SKIP
    >>> db.insert(stops_day_2)            # doctest: +SKIP
    >>> clustered = db.labels()           # doctest: +SKIP
    """

    def __init__(self, params: ClusterParams | None = None, state: ClusterState | None = None,
                 use_cuda: bool = False):
        self.params = params or ClusterParams()
        self.state = state or ClusterState()
        self.use_cuda = use_cuda

    # ------------------------------------------------------------------ insert
    def insert(self, stops: pl.DataFrame | pl.LazyFrame) -> None:
        """Add stops (``uid, stop_event, ..., stop_latitude, stop_longitude``)."""
        eps, min_pts, cuda = float(self.params.distance), int(self.params.min_points), self.use_cuda
        new = (stops.lazy().select(_STOP_PAYLOAD).cast({k: _POINT_SCHEMA[k] for k in _STOP_PAYLOAD})
               .join(self.state.points.lazy().select("uid", "stop_event"), on=["uid", "stop_event"], how="anti")
               .unique(subset=["uid", "stop_event"], keep="first").collect())
        if new.height == 0:
            return
        st = self.state
        active = new.select("uid").unique()
        # Integer device keys make the spatial self-join much cheaper than strings.
        dmap = active.sort("uid").with_row_index("_dev").with_columns(pl.col("_dev").cast(pl.Int64))

        def to_int(df: pl.DataFrame) -> pl.DataFrame:
            return df.join(dmap, on="uid").drop("uid").rename({"_dev": "uid"})

        def to_str(df: pl.DataFrame) -> pl.DataFrame:
            return df.rename({"uid": "_dev"}).join(dmap, on="_dev").drop("_dev")

        # Grid reference latitude per device (re-grid a device if it moved poleward).
        ref = (pl.concat([st.devices.join(active, on="uid", how="semi"),
                          new.group_by("uid").agg(_lat_ref(pl.col("stop_latitude").abs().max()).alias("_lat_ref"))])
               .group_by("uid").agg(pl.col("_lat_ref").max()))
        old_act = st.points.join(active, on="uid", how="semi")
        old_idle = st.points.join(active, on="uid", how="anti")
        old = to_int(_cells(old_act.lazy().drop("_cy", "_cx").join(ref.lazy(), on="uid"), eps).drop("_lat_ref").collect())
        nw = to_int(_cells(new.lazy().join(ref.lazy(), on="uid"), eps).drop("_lat_ref").collect())

        allp = pl.concat([old.select("uid", "stop_event", "stop_latitude", "stop_longitude", "_cy", "_cx",
                                     pl.lit(False).alias("_new")),
                          nw.select("uid", "stop_event", "stop_latitude", "stop_longitude", "_cy", "_cx",
                                    pl.lit(True).alias("_new"))])

        def neighbors(src: pl.DataFrame, dst: pl.DataFrame) -> pl.DataFrame:
            return engine.collect(
                src.lazy().select("uid", pl.col("stop_event").alias("a"), pl.col("stop_latitude").alias("_la"),
                                  pl.col("stop_longitude").alias("_loa"), "_cy", "_cx")
                .join(_OFFSETS.lazy(), how="cross")
                .with_columns((pl.col("_cy") + pl.col("_oy")).alias("_cy"), (pl.col("_cx") + pl.col("_ox")).alias("_cx"))
                .join(dst.lazy().rename({"stop_event": "b"}), on=["uid", "_cy", "_cx"])
                .filter((pl.col("a") < pl.col("b")) | (~pl.col("_new") if "_new" in dst.columns else (pl.col("a") != pl.col("b"))))
                .with_columns(haversine("_la", "_loa", "stop_latitude", "stop_longitude").alias("d"))
                .filter(pl.col("d") <= eps)
                .select("uid", "a", "b", "d", *(["_new"] if "_new" in dst.columns else [])), cuda)

        # 1-2. edges of the new stops (a is new, b is any) and updated degrees.
        E = neighbors(nw, allp)
        # (new-new pairs are kept once, with a < b)
        ends = pl.concat([E.select("uid", pl.col("a").alias("n")), E.select("uid", pl.col("b").alias("n"))])
        deg = ends.group_by("uid", "n").agg(pl.len().cast(pl.Int64).alias("_dd"))
        deg_new = deg.rename({"n": "a", "_dd": "_dn"})
        deg_old = deg.rename({"n": "b", "_dd": "_do"})
        old = (old.join(deg_old, left_on=["uid", "stop_event"], right_on=["uid", "b"], how="left")
               .with_columns((pl.col("_deg") + pl.col("_do").fill_null(0)).alias("_deg")).drop("_do")
               .with_columns(pl.col("_core").alias("_was_core"),
                             ((pl.col("_deg") + 1) >= min_pts).alias("_core")))
        nw = (nw.join(deg_new, left_on=["uid", "stop_event"], right_on=["uid", "a"], how="left")
              .with_columns(pl.col("_dn").fill_null(0).alias("_deg")).drop("_dn")
              .with_columns(((pl.col("_deg") + 1) >= min_pts).alias("_core"),
                            pl.lit(False).alias("_was_core")))

        # 3. points that just became core; their edges to old points.
        promoted = old.filter(pl.col("_core") & ~pl.col("_was_core"))
        if promoted.height and min_pts > 2:
            EX = neighbors(promoted, old.select("uid", "stop_event", "stop_latitude", "stop_longitude", "_cy", "_cx"))
        else:
            EX = pl.DataFrame(schema={"uid": pl.Int64, "a": pl.Int64, "b": pl.Int64, "d": pl.Float64})

        core = pl.concat([old.select("uid", "stop_event", "_core", "_comp"),
                          nw.select("uid", "stop_event", "_core", pl.lit(None, pl.Int64).alias("_comp"))])
        # Component node of a core point: its old component, or itself (seed).
        node = core.filter("_core").select("uid", "stop_event", pl.coalesce("_comp", "stop_event").alias("_node"))
        edges = pl.concat([E.select("uid", "a", "b", "d"), EX.select("uid", "a", "b", "d")])

        def tag(df: pl.DataFrame) -> pl.DataFrame:
            return (df.join(node.rename({"stop_event": "a", "_node": "_na"}), on=["uid", "a"], how="left")
                    .join(node.rename({"stop_event": "b", "_node": "_nb"}), on=["uid", "b"], how="left"))

        edges = tag(edges)
        cc = connected_components(
            edges.filter(pl.col("_na").is_not_null() & pl.col("_nb").is_not_null())
            .select("uid", pl.col("_na").alias("a"), pl.col("_nb").alias("b")), cuda)

        # 4. border candidates: non-core endpoint -> core endpoint.
        cand = pl.concat([
            edges.filter(pl.col("_na").is_null() & pl.col("_nb").is_not_null())
            .select("uid", pl.col("a").alias("p"), pl.col("b").alias("q"), "d"),
            edges.filter(pl.col("_nb").is_null() & pl.col("_na").is_not_null())
            .select("uid", pl.col("b").alias("p"), pl.col("a").alias("q"), "d"),
        ]).sort("uid", "p", "d", "q").unique(subset=["uid", "p"], keep="first")

        pts = pl.concat([old.drop("_was_core"), nw.drop("_was_core").with_columns(
            pl.lit(None, pl.Int64).alias("_comp"), pl.lit(None, pl.Int64).alias("_anchor"),
            pl.lit(None, pl.Float64).alias("_adist"))], how="diagonal_relaxed")
        pts = (
            pts.join(cand, left_on=["uid", "stop_event"], right_on=["uid", "p"], how="left")
            .with_columns(
                pl.when(pl.col("_core")).then(pl.col("stop_event"))
                .when(pl.col("q").is_not_null() & (pl.col("_anchor").is_null() | (pl.col("d") < pl.col("_adist"))
                                                   | ((pl.col("d") == pl.col("_adist")) & (pl.col("q") < pl.col("_anchor")))))
                .then(pl.col("q")).otherwise(pl.col("_anchor")).alias("_anchor"),
                pl.when(pl.col("_core")).then(0.0)
                .when(pl.col("q").is_not_null() & (pl.col("_anchor").is_null() | (pl.col("d") < pl.col("_adist"))
                                                   | ((pl.col("d") == pl.col("_adist")) & (pl.col("q") < pl.col("_anchor")))))
                .then(pl.col("d")).otherwise(pl.col("_adist")).alias("_adist"),
                pl.when(pl.col("_core")).then(pl.coalesce("_comp", "stop_event")).alias("_comp"),
            )
            .drop("q", "d")
            # 5. merge components
            .join(cc.rename({"node": "_comp", "root": "_root"}), on=["uid", "_comp"], how="left")
            .with_columns(pl.coalesce("_root", "_comp").alias("_comp")).drop("_root")
        )
        self.state = ClusterState(
            pl.concat([old_idle, to_str(pts).select(list(_POINT_SCHEMA))], how="vertical_relaxed").cast(_POINT_SCHEMA),
            pl.concat([st.devices.join(active, on="uid", how="anti"), ref]).cast(_DEVICE_SCHEMA),
        )

    # ------------------------------------------------------------------ output
    def labels(self, uids: pl.DataFrame | None = None) -> pl.DataFrame:
        """Clustered stops in the gc-lbs ``*_clustered`` schema."""
        p = self.state.points
        if uids is not None:
            p = p.join(uids.select("uid").unique(), on="uid", how="semi")
        return label_clusters(p, self.use_cuda)


def label_clusters(points: pl.DataFrame, use_cuda: bool = False) -> pl.DataFrame:
    """Turn the graph state into deterministic cluster labels and stats."""
    comp_of = points.filter("_core").select("uid", pl.col("stop_event").alias("_anchor"), pl.col("_comp").alias("_cl"))
    lf = points.lazy().join(comp_of.lazy(), on=["uid", "_anchor"], how="left")
    stats = (lf.filter(pl.col("_cl").is_not_null()).group_by("uid", "_cl").agg(
        tie_exact_mean("stop_latitude").alias("_clat"), tie_exact_mean("stop_longitude").alias("_clon"),
        pl.len().cast(pl.Int32).alias("_cnt"),
        pl.col("stop_latitude").median().alias("_mlat"), pl.col("stop_longitude").median().alias("_mlon"),
    ).sort("uid", "_clat", "_clon", "_cl").pipe(with_group_row, "uid", "cluster_label"))
    out = lf.join(stats, on=["uid", "_cl"], how="left").select(
        *_STOP_PAYLOAD,
        pl.col("cluster_label").fill_null(-1).cast(pl.Int32),
        pl.col("_cnt").fill_null(1).cast(pl.Int32).alias("cluster_counts"),
        pl.coalesce("_mlat", "stop_latitude").alias("cluster_latitude"),
        pl.coalesce("_mlon", "stop_longitude").alias("cluster_longitude"),
    ).sort("uid", "cluster_label", "start_timestamp")
    return engine.collect(out, use_cuda)


def cluster_stops(stops: pl.DataFrame | pl.LazyFrame, params: ClusterParams | None = None, *,
                  chunk_size: int | None = 256, use_cuda: bool = False, **kwargs) -> pl.DataFrame:
    """One-shot per-device DBSCAN of stops (gc-lbs ``dbscan.sh`` equivalent).

    Stops are inserted ``chunk_size`` stops per device at a time (by
    ``stop_event``); insertion is exact, so the result does not depend on it,
    but smaller chunks bound peak memory on dense clusters. ``None`` inserts
    everything at once. Keyword arguments override :class:`ClusterParams`.
    """
    p = params or ClusterParams()
    if kwargs:
        p = ClusterParams(**{**p.__dict__, **kwargs})
    db = IncrementalDBSCAN(p, use_cuda=use_cuda)
    df = engine.collect(stops.lazy(), use_cuda)
    if chunk_size is None:
        db.insert(df)
    else:
        df = df.with_columns(((pl.col("stop_event").rank("ordinal").over("uid") - 1) // chunk_size).alias("_chunk"))
        for k in range(int(df.get_column("_chunk").max() or 0) + 1):
            db.insert(df.filter(pl.col("_chunk") == k))
    return db.labels()
