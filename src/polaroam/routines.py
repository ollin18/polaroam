"""Routines: trips a device repeats, used to fill in what a single day does not show.

A **trip** is the move between two consecutive stays at different places. Location data
rarely sees a whole trip: on one month of Mexico City, 6% of trips were observed from stay to
stay, 32% partially and 61% not at all. But people repeat trips (home to work, work to the
gym...), and the days complement each other. A **routine** is a trip between the same two
places (DBSCAN locations) on several different days.

1. :func:`stay_trips`: trips between consecutive stays, with the trajectories inside each and
   how completely it was observed (``complete``, ``partial``, ``empty``).
2. :func:`find_routines`: routines, and each trip's ``mode``: the mode seen on the trip, or
   else the routine's usual mode.
3. :func:`routine_modes`: that mode back on the trajectories (for map matching by mode).
4. :func:`complete_trips`: each routine trip as one route on the road network from stay to
   stay. Observed stretches follow the matched points; unobserved stretches (gaps, the ends)
   follow the shortest path, with the streets the device used on *other* days of the routine
   made cheaper: the routine's route, learned from pieces seen on different days.

Imputed stretches are flagged, never mixed with what was observed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .geo import haversine

__all__ = ["CompletionResult", "complete_trips", "find_routines", "routine_modes", "stay_trips"]

#: Mode hierarchy for a trip with several modes (as in travel surveys: the "main" mode).
MODE_RANK = {"motorized": 3, "bike": 2, "walk": 1}
#: Typical speeds (km/h) to time an imputed start or end when the trip's own speed is unknown.
DEFAULT_SPEEDS = {"walk": 4.5, "bike": 13.0, "motorized": 20.0}


def _local(df: pl.DataFrame, col: str, timezone: str | None) -> pl.DataFrame:
    """``local_date`` and ``local_hour`` from epoch seconds ``col``, in ``timezone`` or each
    row's ``timezone`` column."""
    def conv(d: pl.DataFrame, tz: str) -> pl.DataFrame:
        t = pl.from_epoch(col, "s").dt.replace_time_zone("UTC").dt.convert_time_zone(tz)
        return d.with_columns(t.dt.date().alias("local_date"),
                              (t.dt.hour() + t.dt.minute() / 60).alias("local_hour"))
    if timezone is not None:
        return conv(df, timezone)
    if "timezone" not in df.columns:
        raise ValueError("stay_trips needs `timezone`, or a `timezone` column on the stays (add_local_time).")
    parts = [conv(p, p["timezone"][0]) for p in df.partition_by("timezone") if p["timezone"][0] is not None]
    return pl.concat(parts) if parts else conv(df.head(0), "UTC")


def stay_trips(stays: pl.DataFrame | pl.LazyFrame, trajectories: pl.DataFrame | pl.LazyFrame, *,
               timezone: str | None = None, min_distance_m: float = 200.0, end_gap_m: float = 500.0,
               hole_m: float = 1000.0, hole_s: float = 300.0) -> pl.DataFrame:
    """Trips between consecutive stays at least ``min_distance_m`` apart, and how completely
    each was observed.

    ``stays`` are clustered stays (:func:`cluster_stops`: ``cluster_label`` and the cluster
    centre); ``trajectories`` come from the trajectory stage. A trajectory belongs to the trip
    whose stays it lies between. ``state`` is:

    - ``complete``: one trajectory from the origin stay to the destination stay, starting and
      ending within ``end_gap_m`` of them, without a *hole* (consecutive points more than
      ``hole_m`` and ``hole_s`` apart);
    - ``partial``: pings between the stays, but not a complete trajectory;
    - ``empty``: no pings between the stays.

    ``mode_observed`` is the main mode of the trip's trajectories (motorized > bike > walk), or
    null if all are ``unknown``. Times are epoch seconds; ``local_date`` / ``local_hour`` (of
    the departure from the origin stay) use ``timezone``, or the stays' ``timezone`` column.
    """
    s = stays.lazy().collect() if isinstance(stays, pl.LazyFrame) else stays
    lat = pl.coalesce("cluster_latitude", "stop_latitude") if "cluster_latitude" in s.columns else pl.col("stop_latitude")
    lon = pl.coalesce("cluster_longitude", "stop_longitude") if "cluster_longitude" in s.columns else pl.col("stop_longitude")
    loc = (pl.when(pl.col("cluster_label") >= 0).then(pl.col("cluster_label")).cast(pl.Int64)
           if "cluster_label" in s.columns else pl.lit(None, pl.Int64))
    tz = [pl.col("timezone")] if timezone is None and "timezone" in s.columns else []
    s = (s.sort("uid", "start_timestamp")
         .select("uid", "stop_event", "start_timestamp", "end_timestamp", loc.alias("loc"), lat.alias("lat"),
                 lon.alias("lon"), *tz))
    nxt = [pl.col(c).shift(-1).over("uid").alias(f"d_{c}") for c in ("stop_event", "start_timestamp", "loc", "lat", "lon")]
    trips = (s.with_columns(nxt).filter(pl.col("d_stop_event").is_not_null())
             .select("uid", pl.col("stop_event").alias("trip_id"), pl.col("stop_event").alias("origin_stop_event"),
                     pl.col("d_stop_event").alias("destination_stop_event"),
                     pl.col("loc").alias("origin_location"), pl.col("d_loc").alias("destination_location"),
                     pl.col("lat").alias("origin_latitude"), pl.col("lon").alias("origin_longitude"),
                     pl.col("d_lat").alias("destination_latitude"), pl.col("d_lon").alias("destination_longitude"),
                     pl.col("end_timestamp").alias("depart_after"), pl.col("d_start_timestamp").alias("arrive_before"),
                     *tz)
             .with_columns(haversine("origin_latitude", "origin_longitude", "destination_latitude",
                                     "destination_longitude").alias("distance_m"),
                           (pl.col("arrive_before") - pl.col("depart_after")).alias("window_s"))
             .filter(pl.col("distance_m") >= min_distance_m))
    trips = _local(trips, "depart_after", timezone)
    if tz:
        trips = trips.drop("timezone")

    t = trajectories.lazy()
    names = t.collect_schema().names()
    mode = pl.col("mode_guess") if "mode_guess" in names else pl.lit("unknown")
    pts = (t.select("uid", "trajectory_id", "coordinates", "timestamps")
           .explode("coordinates", "timestamps", empty_as_null=True)
           .with_columns(pl.col("coordinates").list.get(0).alias("x"), pl.col("coordinates").list.get(1).alias("y")))
    holes = (pts.with_columns(haversine("y", "x", pl.col("y").shift(1).over("uid", "trajectory_id"),
                                        pl.col("x").shift(1).over("uid", "trajectory_id")).alias("_step"),
                              pl.col("timestamps").diff().over("uid", "trajectory_id").alias("_dt"))
             .group_by("uid", "trajectory_id")
             .agg(((pl.col("_step") > hole_m) & (pl.col("_dt") > hole_s)).any().alias("_hole")))
    tr = (t.select("uid", "trajectory_id", "origin_stop_event", "destination_stop_event", "start_timestamp",
                   "end_timestamp", "duration", mode.alias("_mode"),
                   pl.col("coordinates").list.first().alias("_first"), pl.col("coordinates").list.last().alias("_last"))
          .join(holes, on=["uid", "trajectory_id"], how="left")
          .sort("uid", "start_timestamp").collect(engine="streaming"))
    m = (tr.join_asof(trips.select("uid", "trip_id", "destination_stop_event", "depart_after", "arrive_before")
                      .rename({"destination_stop_event": "_d"}).sort("uid", "depart_after"),
                      left_on="start_timestamp", right_on="depart_after", by="uid", strategy="backward",
                      check_sortedness=False)
         .filter(pl.col("trip_id").is_not_null() & (pl.col("end_timestamp") <= pl.col("arrive_before") + 60)))
    rank = pl.col("_mode").replace_strict(MODE_RANK, default=0, return_dtype=pl.Int8)
    agg = (m.sort("start_timestamp").group_by("uid", "trip_id").agg(
        pl.col("trajectory_id").alias("trajectory_ids"),
        pl.len().cast(pl.UInt32).alias("n_trajectories"),
        ((pl.col("origin_stop_event") == pl.col("trip_id")) & (pl.col("destination_stop_event") == pl.col("_d")))
        .any().alias("_linked"),
        pl.col("duration").sum().alias("observed_s"),
        pl.col("_hole").any().alias("has_hole"),
        pl.col("_first").first(), pl.col("_last").last(),
        pl.col("_mode").sort_by(rank, descending=True).first().alias("_top"), rank.max().alias("_rank")))
    out = (trips.join(agg, on=["uid", "trip_id"], how="left")
           .with_columns(
               haversine("origin_latitude", "origin_longitude", pl.col("_first").list.get(1),
                         pl.col("_first").list.get(0)).alias("start_gap_m"),
               haversine("destination_latitude", "destination_longitude", pl.col("_last").list.get(1),
                         pl.col("_last").list.get(0)).alias("end_gap_m"),
               pl.col("n_trajectories").fill_null(0),
               pl.when(pl.col("_rank") > 0).then(pl.col("_top")).alias("mode_observed")))
    state = (pl.when(pl.col("trajectory_ids").is_null()).then(pl.lit("empty"))
             .when(pl.col("_linked") & ~pl.col("has_hole").fill_null(False)
                   & (pl.col("start_gap_m") <= end_gap_m) & (pl.col("end_gap_m") <= end_gap_m)).then(pl.lit("complete"))
             .otherwise(pl.lit("partial")))
    return out.with_columns(state.alias("state")).drop("_first", "_last", "_linked", "_top", "_rank").sort("uid", "trip_id")


def find_routines(trips: pl.DataFrame, *, min_days: int = 3, mode_min_trips: int = 2,
                  mode_min_share: float = 0.6) -> pl.DataFrame:
    """Routines: trips between the same two locations on at least ``min_days`` different days.

    Adds to :func:`stay_trips`' table:

    - ``routine_id`` (null outside a routine), ``routine_days``, ``routine_trips`` and
      ``routine_hour`` (median departure hour);
    - ``routine_mode``: the routine's usual mode, if at least ``mode_min_trips`` of its trips
      have a mode and at least ``mode_min_share`` of those share it;
    - ``mode``: the trip's own mode (``mode_observed``), or else the routine's;
      ``mode_source`` says which (``observed``, ``routine``, or null without a mode).

    Locations are DBSCAN clusters: stays outside any cluster are never part of a routine.
    """
    key = ["uid", "origin_location", "destination_location"]
    inr = trips.filter(pl.col("origin_location").is_not_null() & pl.col("destination_location").is_not_null())
    r = (inr.group_by(key).agg(pl.col("local_date").n_unique().cast(pl.UInt32).alias("routine_days"),
                               pl.len().cast(pl.UInt32).alias("routine_trips"),
                               pl.col("local_hour").median().alias("routine_hour"))
         .filter(pl.col("routine_days") >= min_days).sort(key)
         .with_row_index("routine_id").with_columns(pl.col("routine_id").cast(pl.Int64)))
    modes = (inr.join(r.select(*key, "routine_id"), on=key).filter(pl.col("mode_observed").is_not_null())
             .group_by("routine_id", "mode_observed").agg(pl.len().alias("_n"))
             .with_columns(pl.col("_n").sum().over("routine_id").alias("_tot"))
             .sort("routine_id", "_n", "mode_observed", descending=[False, True, False])
             .group_by("routine_id", maintain_order=True).first()
             .filter((pl.col("_tot") >= mode_min_trips) & (pl.col("_n") / pl.col("_tot") >= mode_min_share))
             .select("routine_id", pl.col("mode_observed").alias("routine_mode"),
                     (pl.col("_n") / pl.col("_tot")).alias("routine_mode_share")))
    r = r.join(modes, on="routine_id", how="left")
    out = trips.drop([c for c in r.columns if c not in key and c in trips.columns]).join(r, on=key, how="left")
    return out.with_columns(
        pl.coalesce("mode_observed", "routine_mode").alias("mode"),
        pl.when(pl.col("mode_observed").is_not_null()).then(pl.lit("observed"))
        .when(pl.col("routine_mode").is_not_null()).then(pl.lit("routine")).alias("mode_source"))


def routine_modes(trajectories: pl.DataFrame | pl.LazyFrame, trips: pl.DataFrame) -> pl.DataFrame:
    """Trajectories with a final ``mode`` and its ``mode_source``: the speed rule
    (``mode_guess``, source ``speed``), or, for ``unknown`` trajectories of a trip whose mode
    came from its routine, that mode (source ``routine``).

    Match by it with ``match_trajectories(trajs, networks, by="mode")``.
    """
    t = trajectories.lazy().collect() if isinstance(trajectories, pl.LazyFrame) else trajectories
    fill = (trips.filter(pl.col("mode_source") == "routine").select("uid", "trajectory_ids", pl.col("mode").alias("_rm"))
            .explode("trajectory_ids", empty_as_null=True).rename({"trajectory_ids": "trajectory_id"}))
    known = pl.col("mode_guess") != "unknown"
    return (t.join(fill, on=["uid", "trajectory_id"], how="left")
            .with_columns(pl.when(known).then(pl.col("mode_guess")).otherwise(pl.coalesce("_rm", pl.lit("unknown")))
                          .alias("mode"),
                          pl.when(known).then(pl.lit("speed")).when(pl.col("_rm").is_not_null())
                          .then(pl.lit("routine")).alias("mode_source"))
            .drop("_rm"))


# --------------------------------------------------------------------------- completion
@dataclass
class CompletionResult:
    """Output of :func:`complete_trips`.

    trips: one row per completed trip: lengths observed and imputed, the share of the imputed
        length on streets used on other days of the routine (``imputed_on_route``), estimated
        departure and arrival, and the route ``geometry`` (WKT; MULTILINESTRING if a gap could
        not be routed).
    edges: every edge of each route in order (``seq``), the covered stretch (``from_m``,
        ``to_m``), ``imputed``, and the network's edge attributes.
    routes: the routines' usual routes: per routine and edge, how many of its trips were
        *observed* on that edge (imputed stretches never count).
    """

    trips: pl.DataFrame
    edges: pl.DataFrame
    routes: pl.DataFrame


def _snap(net, df: pl.DataFrame, lat: str, lon: str, step: int, radius: float, k: int) -> pl.DataFrame:
    """Up to ``k`` nearest edge positions within ``radius`` of each row's point."""
    from .roads import MatchParams, _candidates

    pts = df.select("k", pl.lit(step, pl.Int64).alias("step"), pl.col(lon).alias("lon"), pl.col(lat).alias("lat"))
    c = _candidates(net, pts.rename({"k": "key"}), MatchParams(search_radius=radius, max_candidates=k))
    return c.select(pl.col("key").alias("k"), "edge", "offset", "c")


def _anchors(net, trips: pl.DataFrame, pts: pl.DataFrame, radius: float, backtrack_m: float) -> pl.DataFrame:
    """Per trip ``k``: anchor ``a`` = 0 (origin stay), matched points 1..n, n + 1 (destination
    stay); stays have up to 4 candidate positions (``c``). Columns ``k, a, c, edge, offset,
    lon, lat, t``."""
    # within a run on one edge, never step back (GPS jitter of a slow device)
    run = ((pl.col("edge") != pl.col("edge").shift(1)) | (pl.col("trajectory_id") != pl.col("trajectory_id").shift(1)))
    p = (pts.sort("k", "timestamp", "trajectory_id", "step")
         .with_columns(run.fill_null(True).cum_sum().over("k").alias("_run"))
         .with_columns(pl.col("offset_m").cum_max().over("k", "_run").alias("offset"))
         .with_columns(pl.int_range(1, pl.len() + 1).over("k").alias("a"))
         .select("k", "a", pl.lit(0, pl.Int64).alias("c"), "edge", "offset", pl.col("longitude").alias("lon"),
                 pl.col("latitude").alias("lat"), pl.col("timestamp").alias("t")))
    n = p.group_by("k").agg(pl.col("a").max().alias("_n"))
    o = (_snap(net, trips, "origin_latitude", "origin_longitude", 0, radius, 4)
         .join(trips.select("k", pl.col("origin_longitude").alias("lon"), pl.col("origin_latitude").alias("lat")), on="k")
         .select("k", pl.lit(0).alias("a"), "c", "edge", "offset", "lon", "lat", pl.lit(None, pl.Int64).alias("t")))
    d = (_snap(net, trips, "destination_latitude", "destination_longitude", 1, radius, 4)
         .join(trips.select("k", pl.col("destination_longitude").alias("lon"), pl.col("destination_latitude").alias("lat")), on="k")
         .join(n, on="k", how="left")
         .select("k", (pl.col("_n").fill_null(0) + 1).alias("a"), "c", "edge", "offset", "lon", "lat",
                 pl.lit(None, pl.Int64).alias("t")))
    cols = ["k", "a", "c", "edge", "offset", "lon", "lat", "t"]
    return pl.concat([x.select(cols).with_columns(pl.col("a").cast(pl.Int64), pl.col("c").cast(pl.Int64),
                                                  pl.col("t").cast(pl.Int64))
                      for x in (o, p, d)]).sort("k", "a", "c")


def _route(net, pairs: pl.DataFrame, routes, own, pref: float, max_detour: float, radius: float,
           backtrack_m: float) -> pl.DataFrame:
    """Best route for each anchor step ``(k, a)`` over its candidate pairs: pieces
    ``k, a, order, edge, from_m, to_m``. ``pairs``: ``k, a, ca, cb, ea, oa, eb, ob, gc, rg,
    tg, others``."""
    from ._routing import preferred_paths

    en = net._edge_nodes
    pairs = (pairs.join(en.select(pl.col("edge").alias("ea"), pl.col("vn").alias("va"), pl.col("length_m").alias("la")),
                        on="ea", how="left")
             .join(en.select(pl.col("edge").alias("eb"), pl.col("un").alias("ub")), on="eb", how="left")
             .with_columns(((pl.col("ea") == pl.col("eb")) & (pl.col("ob") >= pl.col("oa") - backtrack_m)).alias("same"))
             .with_row_index("pid"))
    same = pairs.filter(pl.col("same")).select("pid", (pl.col("ob") - pl.col("oa")).clip(lower_bound=0.0).alias("cost"))
    need = pairs.filter(~pl.col("same"))
    cost = same
    nodes = pl.DataFrame(schema={"pid": pl.UInt32, "k": pl.Int64, "node": pl.Int64})
    if need.height:
        lim = need.select(pl.col("gc") * max_detour + 2 * radius)["gc"].to_numpy()
        c, pair, hop, node = preferred_paths(
            net._graph, net._arc_edge, need["va"].to_numpy().astype(np.int64), need["ub"].to_numpy().astype(np.int64),
            lim.astype(np.float64), need["rg"].to_numpy().astype(np.int64), need["tg"].to_numpy().astype(np.int64),
            need["others"].to_numpy().astype(np.float64), routes, own, pref)
        cost = pl.concat([cost, need.select("pid", ((pl.col("la") - pl.col("oa")) + pl.Series(c) + pl.col("ob"))
                                            .alias("cost"))])
        nodes = (pl.DataFrame({"i": pair.astype(np.int64), "hop": hop.astype(np.int64), "node": node})
                 .join(need.select(pl.int_range(pl.len(), dtype=pl.Int64).alias("i"), "pid"), on="i")
                 .sort("pid", "hop", descending=[False, True])
                 .with_columns(pl.col("node").shift(1).over("pid").alias("prev"),
                               pl.int_range(pl.len()).over("pid").alias("order"))
                 .filter(pl.col("prev").is_not_null())
                 .join(net._link.rename({"un": "prev", "vn": "node"}), on=["prev", "node"], how="left")
                 .select("pid", pl.col("order").cast(pl.Int64), "edge"))
    best = (pairs.join(cost.filter(pl.col("cost").is_finite()), on="pid")
            .sort("k", "a", "cost", "ca", "cb").group_by("k", "a", maintain_order=True).first())
    one = best.filter(pl.col("same")).select("k", "a", pl.lit(0, pl.Int64).alias("order"), pl.col("ea").alias("edge"),
                                             pl.col("oa").alias("from_m"),
                                             pl.max_horizontal("oa", "ob").alias("to_m"), "ca", "cb")
    multi = best.filter(~pl.col("same"))
    first = multi.select("k", "a", pl.lit(0, pl.Int64).alias("order"), pl.col("ea").alias("edge"),
                         pl.col("oa").alias("from_m"), pl.col("la").alias("to_m"), "ca", "cb")
    mid = (multi.select("k", "a", "pid", "ca", "cb").join(nodes, on="pid")
           .join(en.select("edge", "length_m"), on="edge", how="left")
           .select("k", "a", (pl.col("order") + 1).alias("order"), "edge", pl.lit(0.0).alias("from_m"),
                   pl.col("length_m").alias("to_m"), "ca", "cb"))
    nmid = mid.group_by("k", "a").agg(pl.len().cast(pl.Int64).alias("_m"))
    last = (multi.join(nmid, on=["k", "a"], how="left")
            .select("k", "a", (pl.col("_m").fill_null(0) + 1).alias("order"), pl.col("eb").alias("edge"),
                    pl.lit(0.0).alias("from_m"), pl.col("ob").alias("to_m"), "ca", "cb"))
    return pl.concat([one, first, mid, last]).sort("k", "a", "order")


def _csr(df: pl.DataFrame, group: str, n: int):
    """``(ptr, edges, counts)`` of an ``group, edge, n`` table (groups ``0..n-1``)."""
    df = df.sort(group, "edge")
    ptr = np.zeros(n + 1, np.int64)
    if df.height:
        np.cumsum(np.bincount(df[group].to_numpy(), minlength=n), out=ptr[1:])
    return ptr, df["edge"].to_numpy().astype(np.int64), df["n"].to_numpy().astype(np.float64)


def _points_at(net, df: pl.DataFrame) -> pl.DataFrame:
    """``lon, lat`` of position ``m`` along ``edge`` for each row."""
    seg = net._segments.select("edge", "cum0", "len", "lon", "lat", "lon2", "lat2").sort("cum0")
    f = ((pl.col("m") - pl.col("cum0")) / pl.col("len")).fill_nan(0.0).clip(0.0, 1.0)
    return (df.with_row_index("_o").sort("m")
            .join_asof(seg, left_on="m", right_on="cum0", by="edge", strategy="backward", check_sortedness=False)
            .with_columns((pl.col("lon") + f * (pl.col("lon2") - pl.col("lon"))).alias("lon"),
                          (pl.col("lat") + f * (pl.col("lat2") - pl.col("lat"))).alias("lat"))
            .sort("_o").drop("_o", "cum0", "len", "lon2", "lat2"))


def _geometry(net, pieces: pl.DataFrame) -> pl.DataFrame:
    """WKT line per ``k`` from ordered pieces ``k, part, seq, edge, from_m, to_m``."""
    p = pieces.select("k", "part", "seq", "edge", "from_m", "to_m")
    ends = pl.concat([p.select("k", "part", "seq", "edge", pl.col("from_m").alias("m"), pl.lit(0).alias("w")),
                      p.select("k", "part", "seq", "edge", pl.col("to_m").alias("m"), pl.lit(2).alias("w"))])
    ends = _points_at(net, ends)
    mid = (p.join(net._vertices.select("edge", "cum", "lon", "lat"), on="edge")
           .filter((pl.col("cum") > pl.col("from_m")) & (pl.col("cum") < pl.col("to_m")))
           .select("k", "part", "seq", "edge", pl.col("cum").alias("m"), pl.lit(1).alias("w"), "lon", "lat"))
    pts = (pl.concat([ends.select(mid.columns), mid]).sort("k", "part", "seq", "w", "m")
           .with_columns(pl.col("lon").round(7), pl.col("lat").round(7))
           .filter(~((pl.col("lon") == pl.col("lon").shift(1)) & (pl.col("lat") == pl.col("lat").shift(1)))
                   .over("k", "part").fill_null(False)))
    line = (pts.group_by("k", "part", maintain_order=True)
            .agg(pl.format("{} {}", "lon", "lat").str.join(", ").alias("_w"), pl.len().alias("_n"))
            .filter(pl.col("_n") >= 2))
    return (line.group_by("k", maintain_order=True).agg(pl.col("_w"), pl.len().cast(pl.UInt32).alias("n_parts"))
            .with_columns(pl.when(pl.col("n_parts") == 1)
                          .then(pl.concat_str(pl.lit("LINESTRING ("), pl.col("_w").list.first(), pl.lit(")")))
                          .otherwise(pl.concat_str(pl.lit("MULTILINESTRING (("), pl.col("_w").list.join("), ("),
                                                   pl.lit("))")))
                          .alias("geometry")).drop("_w"))


def complete_trips(trips: pl.DataFrame, points, network, *, by: str = "mode", preference: float = 0.7,
                   gap_m: float = 500.0, gap_s: float = 120.0, fill_empty: bool = False, search_radius: float = 150.0,
                   max_detour: float = 3.0, backtrack_m: float = 40.0,
                   speeds_kmh: dict[str, float] | None = None) -> CompletionResult:
    """Complete routine trips on the road network, from the origin stay to the destination stay.

    ``trips`` come from :func:`find_routines`; ``points`` are the matched points of their
    trajectories (``MatchResult.points``, or the result itself) and ``network`` a
    :class:`RoadNetwork` or a dict of networks per value of ``by`` (as for
    :func:`match_trajectories`; points matched on another network, e.g. the walk to a bus stop
    on a motorized trip, are left out and their stretch is imputed).

    Every step between consecutive anchors (origin stay, matched points in time order,
    destination stay) is routed on the network. A step longer than ``gap_m`` that also took
    more than ``gap_s`` (or that starts or ends at a stay) is a stretch nobody observed: it is
    **imputed** and follows the shortest path with each edge's
    length lowered by ``preference`` x the share of the routine's *other* trips observed on it
    (0: plain shortest path; 0.7: a street used on every other day costs 30% of its length).
    The trip's own observations never count for its own route, so a single day cannot
    reinforce itself. Shorter steps are observed and follow the plain shortest path, like the
    map matching. Stays are snapped to the 4 nearest edges within ``search_radius`` and the
    cheapest one is used; steps whose route would be longer than ``max_detour`` x the straight
    distance (+ 2 x ``search_radius``) are left as gaps.

    Trips in a routine with a ``by`` value that has a network are completed; ``empty`` trips
    (no pings) only with ``fill_empty``: their route is the routine's usual one, and their
    times stay unknown. Departure and arrival of the others are estimated from the first and
    last observed points at the trip's observed speed (or ``speeds_kmh`` per mode).
    """
    from .roads import MatchResult, RoadNetwork

    if isinstance(points, MatchResult):
        points = points.points
    pts_all = points.lazy()
    nets = [(None, network)] if isinstance(network, RoadNetwork) else list(network.items())
    speeds = {**DEFAULT_SPEEDS, **(speeds_kmh or {})}
    t = trips.filter(pl.col("routine_id").is_not_null())
    if not fill_empty:
        t = t.filter(pl.col("state") != "empty")
    outs: list[tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]] = []
    for label, net in nets:
        sub = t if label is None else t.filter(pl.col(by) == label)
        if sub.height == 0:
            continue
        sub = sub.with_row_index("k").with_columns(pl.col("k").cast(pl.Int64))
        tmap = (sub.select("k", "uid", "trajectory_ids").explode("trajectory_ids", empty_as_null=True)
                .rename({"trajectory_ids": "trajectory_id"}).drop_nulls())
        p = pts_all if label is None or by not in pts_all.collect_schema().names() else pts_all.filter(pl.col(by) == label)
        p = (p.select("uid", "trajectory_id", "step", "timestamp", "longitude", "latitude", "edge", "offset_m")
             .join(tmap.lazy(), on=["uid", "trajectory_id"]).collect())
        anc = _anchors(net, sub, p, search_radius, backtrack_m)
        a0 = anc.select("k", pl.col("a") + 1, pl.col("c").alias("ca"), pl.col("edge").alias("ea"),
                        pl.col("offset").alias("oa"), pl.col("lon").alias("lona"), pl.col("lat").alias("lata"),
                        pl.col("t").alias("ta"))
        a1 = anc.select("k", "a", pl.col("c").alias("cb"), pl.col("edge").alias("eb"), pl.col("offset").alias("ob"),
                        pl.col("lon").alias("lonb"), pl.col("lat").alias("latb"), pl.col("t").alias("tb"))
        pairs = (a0.join(a1, on=["k", "a"])
                 .with_columns(haversine("lata", "lona", "latb", "lonb").alias("gc"))
                 .with_columns(((pl.col("gc") > gap_m) & ((pl.col("tb") - pl.col("ta") > gap_s).fill_null(True)))
                               .alias("imputed")))
        # trip -> its routine (rg) and itself (tg), in dense indices
        rid = sub.select("routine_id").unique().sort("routine_id").with_row_index("rg")
        sub = sub.join(rid, on="routine_id", how="left").with_columns(pl.col("rg").cast(pl.Int64))
        pairs = pairs.join(sub.select("k", "rg"), on="k").with_columns(pl.col("k").alias("tg"))
        nr, nt = rid.height, sub.height
        empty = (np.zeros(1, np.int64), np.empty(0, np.int64), np.empty(0, np.float64))

        # pass 1: observed steps, plain shortest paths
        obs = pairs.filter(~pl.col("imputed")).with_columns(pl.lit(0.0).alias("others"))
        po = _route(net, obs, (np.zeros(nr + 1, np.int64), *empty[1:]), (np.zeros(nt + 1, np.int64), *empty[1:]),
                    0.0, max_detour, search_radius, backtrack_m) if obs.height else None
        # the routines' routes: trips observed on each edge
        own = (po.select("k", "edge").unique() if po is not None
               else pl.DataFrame(schema={"k": pl.Int64, "edge": pl.Int64}))
        own = own.join(sub.select("k", "rg"), on="k")
        rcount = own.group_by("rg", "edge").agg(pl.len().cast(pl.Float64).alias("n"))
        observed_trips = own.group_by("rg").agg(pl.col("k").n_unique().alias("_nt"))
        others = (sub.select("k", "rg").join(observed_trips, on="rg", how="left")
                  .join(own.select("k").unique().with_columns(pl.lit(1).alias("_own")), on="k", how="left")
                  .select("k", (pl.col("_nt").fill_null(0) - pl.col("_own").fill_null(0)).cast(pl.Float64).alias("others")))
        # pass 2: imputed steps, preferring the routine's other trips' streets
        imp = pairs.filter(pl.col("imputed")).join(others, on="k", how="left")
        pi = _route(net, imp, _csr(rcount, "rg", nr),
                    _csr(own.select(pl.col("k").alias("tg"), "edge", pl.lit(1.0).alias("n")), "tg", nt),
                    preference, max_detour, search_radius, backtrack_m) if imp.height else None
        done = [x.with_columns(pl.lit(f).alias("imputed")) for x, f in ((po, False), (pi, True)) if x is not None]
        if not done:
            continue
        pieces = pl.concat(done)
        # stays: keep the chosen candidate; a step with no route starts a new part
        steps = pieces.select("k", "a", "ca", "cb").unique(["k", "a"])
        last_a = anc.group_by("k").agg(pl.col("a").max().alias("_last"))
        routed = pairs.select("k", "a").unique().join(steps.select("k", "a", pl.lit(True).alias("_ok")),
                                                     on=["k", "a"], how="left")
        part = (routed.sort("k", "a").with_columns(pl.col("_ok").fill_null(False))
                .with_columns((~pl.col("_ok")).cast(pl.Int64).cum_sum().over("k").alias("part")))
        pieces = (pieces.join(part.select("k", "a", "part"), on=["k", "a"]).sort("k", "a", "order")
                  .filter(pl.col("to_m") > pl.col("from_m"))
                  .with_columns(pl.int_range(pl.len()).over("k").alias("seq")))
        geo = _geometry(net, pieces)
        imp_steps = pairs.select("k", "a", "imputed").unique(["k", "a"])
        fill = (imp_steps.join(routed, on=["k", "a"]).group_by("k")
                .agg(pl.col("imputed").sum().cast(pl.UInt32).alias("n_gaps"),
                     (pl.col("imputed") & pl.col("_ok").fill_null(False)).sum().cast(pl.UInt32).alias("n_filled")))
        # imputed length on streets other days used
        on_route = (pieces.filter(pl.col("imputed")).join(sub.select("k", "rg"), on="k")
                    .join(rcount, on=["rg", "edge"], how="left")
                    .join(own.select("k", "edge", pl.lit(1.0).alias("_self")), on=["k", "edge"], how="left")
                    .with_columns(((pl.col("n").fill_null(0) - pl.col("_self").fill_null(0)) > 0).alias("_on"))
                    .group_by("k").agg(((pl.col("to_m") - pl.col("from_m")) * pl.col("_on")).sum().alias("_on_m")))
        lengths = pieces.group_by("k").agg(
            ((pl.col("to_m") - pl.col("from_m")) * ~pl.col("imputed")).sum().alias("observed_m"),
            ((pl.col("to_m") - pl.col("from_m")) * pl.col("imputed")).sum().alias("imputed_m"))
        ends = pieces.filter(pl.col("imputed")).join(last_a, on="k").group_by("k").agg(
            ((pl.col("to_m") - pl.col("from_m")) * (pl.col("a") == 1)).sum().alias("_m_start"),
            ((pl.col("to_m") - pl.col("from_m")) * (pl.col("a") == pl.col("_last"))).sum().alias("_m_end"))
        times = p.group_by("k").agg(pl.col("timestamp").min().alias("_t0"), pl.col("timestamp").max().alias("_t1"))
        default_v = (pl.col(by).replace_strict(speeds, default=10.0, return_dtype=pl.Float64) / 3.6
                     if by in sub.columns else pl.lit(10.0 / 3.6))
        obs_v = pl.col("observed_m") / (pl.col("_t1") - pl.col("_t0"))
        v = pl.when((pl.col("_t1") > pl.col("_t0")) & (pl.col("observed_m") > 0)).then(obs_v.clip(0.3, 35.0)) \
            .otherwise(default_v)
        tr = (sub.join(lengths, on="k", how="left").join(on_route, on="k", how="left").join(ends, on="k", how="left")
              .join(times, on="k", how="left").join(fill, on="k", how="left").join(geo, on="k", how="left")
              .with_columns(pl.col("observed_m").fill_null(0.0), pl.col("imputed_m").fill_null(0.0),
                            pl.col("n_parts").fill_null(0))
              .with_columns((pl.col("observed_m") + pl.col("imputed_m")).alias("length_m"),
                            (pl.col("_on_m").fill_null(0.0) / pl.col("imputed_m")).alias("imputed_on_route"),
                            v.alias("_v"))
              .with_columns((pl.col("imputed_m") / pl.col("length_m")).alias("imputed_share"),
                            pl.when(pl.col("_t0").is_not_null()).then(pl.max_horizontal(
                                "depart_after", (pl.col("_t0") - pl.col("_m_start").fill_null(0.0) / pl.col("_v"))
                                .round().cast(pl.Int64))).alias("depart_estimated"),
                            pl.when(pl.col("_t1").is_not_null()).then(pl.min_horizontal(
                                "arrive_before", (pl.col("_t1") + pl.col("_m_end").fill_null(0.0) / pl.col("_v"))
                                .round().cast(pl.Int64))).alias("arrive_estimated"))
              .join(others, on="k", how="left").rename({"others": "route_trips"})
              .with_columns(pl.col("route_trips").cast(pl.UInt32)))
        keep = ["uid", "trip_id", "routine_id", *([by] if by in sub.columns else []), "state", "n_gaps",
                "n_filled", "n_parts", "observed_m", "imputed_m", "length_m", "imputed_share", "imputed_on_route",
                "route_trips", "depart_after", "arrive_before", "depart_estimated", "arrive_estimated", "geometry"]
        edges = (pieces.join(sub.select("k", "uid", "trip_id", *([by] if by in sub.columns else [])), on="k")
                 .join(net._edge_nodes.select("edge", "length_m"), on="edge", how="left")
                 .join(net.edges.drop("length_m"), on="edge", how="left")
                 .sort("k", "seq").drop("k", "a", "order", "ca", "cb"))
        routes = (rcount.join(rid.with_columns(pl.col("rg").cast(pl.Int64)), on="rg")
                  .select("routine_id", *([pl.lit(label).alias(by)] if label is not None else []), "edge",
                          pl.col("n").cast(pl.UInt32).alias("trips")).sort("routine_id", "trips", descending=[False, True]))
        outs.append((tr.select(keep), edges, routes))
    if not outs:
        return CompletionResult(pl.DataFrame(), pl.DataFrame(), pl.DataFrame())
    return CompletionResult(*(pl.concat([o[i] for o in outs], how="diagonal_relaxed") for i in range(3)))
