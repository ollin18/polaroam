"""Trajectories: the movement between stays, as simplified polylines.

Pings that fall inside a stay ``[start_timestamp, end_timestamp]`` are stay
pings; the rest are *moving* pings. Consecutive moving pings of a device are
chained into one trajectory while

* they are less than ``max_time_gap`` seconds apart, and
* no stay happens in between.

The stay right before / after a trajectory (if it ended / starts within
``max_time_gap``) is used as origin / destination and its location is added as
the first / last vertex, so trajectories connect stays.

Lines are simplified with Douglas-Peucker (tolerance in meters). The
recursion is evaluated breadth-first: every iteration splits *all* current
segments of *all* trajectories at once with column-wise polars expressions,
which gives exactly the classic recursive result.
"""

from __future__ import annotations

import polars as pl

from . import engine
from .config import TrajectoryParams
from .geo import DEG, M_PER_DEG, haversine, with_group_row
from .io import PingSource, read_pings

TRAJECTORY_COLUMNS = [
    "uid", "trajectory_id", "origin_stop_event", "destination_stop_event",
    "start_timestamp", "end_timestamp", "duration", "n_pings", "n_points", "length_m",
    "speed_mean_kmh", "speed_median_kmh", "speed_p85_kmh", "speed_max_kmh", "speed_cv",
    "frac_slow", "straightness", "ping_gap_median_s", "n_jumps", "mode_guess",
    "geometry", "coordinates", "timestamps",
]
_FEATURE_COLUMNS = ["speed_mean_kmh", "speed_median_kmh", "speed_p85_kmh", "speed_max_kmh", "speed_cv",
                    "frac_slow", "ping_gap_median_s"]

_STOP_COLS = ["uid", "stop_event", "start_timestamp", "end_timestamp", "stop_latitude", "stop_longitude"]


def empty_trajectories() -> pl.DataFrame:
    return pl.DataFrame(schema={
        "uid": pl.String, "trajectory_id": pl.Int64, "origin_stop_event": pl.Int64,
        "destination_stop_event": pl.Int64, "start_timestamp": pl.Int64, "end_timestamp": pl.Int64,
        "duration": pl.Int64, "n_pings": pl.UInt32, "n_points": pl.UInt32, "length_m": pl.Float64,
        **{c: pl.Float64 for c in _FEATURE_COLUMNS}, "straightness": pl.Float64, "n_jumps": pl.UInt32,
        "mode_guess": pl.String, "geometry": pl.String, "coordinates": pl.List(pl.List(pl.Float64)),
        "timestamps": pl.List(pl.Int64),
    }).select(TRAJECTORY_COLUMNS)


def speed_features(vertices: pl.DataFrame, params: TrajectoryParams | None = None) -> pl.DataFrame:
    """Per-trajectory speed features from all (unsimplified) vertices.

    ``vertices``: ``_seg, _i, timestamp (ms), latitude, longitude, _step`` (meters from
    the previous vertex), sorted by ``_seg, _i``. Speeds (km/h) are *windowed*: for each
    vertex, the straight-line displacement to the first vertex at least ``speed_window``
    seconds later, divided by the elapsed time (GPS jitter cancels out instead of adding
    up, as it would with path length). Each window is weighted by the time to the next
    vertex, so the statistics are **time-weighted** ("share of the trip's time"),
    whatever the ping rate; ``speed_mean_kmh`` is the time-weighted mean. Intervals longer
    than ``speed_max_gap`` are not used (speed is not observed there). Steps faster than
    ``max_speed_kmh`` are GPS jumps: their windows are dropped and they are counted. All sums are per-trajectory prefix sums in row order, so the
    result is bit-identical between runs, engines and incremental / one-shot processing.
    """
    tp = params or TrajectoryParams()
    cap, slow, win = float(tp.max_speed_kmh), float(tp.slow_speed_kmh), int(float(tp.speed_window) * 1000)
    max_gap = float(tp.speed_max_gap) * 1000
    seg, ts = pl.col("_seg"), pl.col("timestamp")
    same_next = seg == seg.shift(-1)
    s = vertices.select("_seg", "_i", "timestamp", "latitude", "longitude", "_step").with_columns(
        pl.when(same_next).then(ts.shift(-1) - ts).alias("_dt"),               # ms to the next vertex
        pl.when(same_next).then(pl.col("_step").shift(-1)).alias("_dnext"),    # m to the next vertex
    ).with_columns(
        (pl.col("_dt").is_not_null()
         & (((pl.col("_dt") == 0) & (pl.col("_dnext") > 0))
            | ((pl.col("_dt") > 0) & (pl.col("_dnext") / pl.col("_dt") * 3600.0 > cap)))).alias("_jump")
    ).with_columns(
        ts.last().over("_seg").alias("_tlast"),
        pl.col("latitude").last().over("_seg").alias("_lalast"),
        pl.col("longitude").last().over("_seg").alias("_lolast"),
    )

    # window end: first vertex at least `speed_window` later (else the last vertex)
    right = (s.group_by("_seg", "timestamp", maintain_order=True)
             .agg(pl.col("latitude").last().alias("_laj"), pl.col("longitude").last().alias("_loj"))
             .rename({"timestamp": "_tj"}).sort("_tj", maintain_order=True))
    w = (s.with_columns((ts + win).alias("_tq")).sort("_tq", maintain_order=True)
         .join_asof(right, left_on="_tq", right_on="_tj", by="_seg", strategy="forward", check_sortedness=False)
         .sort("_seg", "_i")
         .with_columns(pl.coalesce("_tj", "_tlast").alias("_tj"), pl.coalesce("_laj", "_lalast").alias("_laj"),
                       pl.coalesce("_loj", "_lolast").alias("_loj"))
         .with_columns(pl.when(pl.col("_tj") > ts).then(
             haversine("latitude", "longitude", "_laj", "_loj") / (pl.col("_tj") - ts) * 3600.0).alias("_v")))
    valid = w.filter((pl.col("_dt") > 0) & (pl.col("_dt") <= max_gap) & ~pl.col("_jump")
                     & pl.col("_v").is_not_null() & (pl.col("_v") <= cap))
    wt = pl.col("_dt").cast(pl.Float64)

    tot = valid.with_columns(
        wt.cum_sum().over("_seg").alias("_W"), (wt * pl.col("_v")).cum_sum().over("_seg").alias("_WV"),
        (wt * (pl.col("_v") < slow).cast(pl.Float64)).cum_sum().over("_seg").alias("_WS"),
        pl.col("_v").cum_max().over("_seg").alias("_VMAX"),
    ).group_by("_seg", maintain_order=True).agg(pl.col("_W", "_WV", "_WS", "_VMAX").last()
    ).with_columns((pl.col("_WV") / pl.col("_W")).alias("_vbar"))
    var = (valid.join(tot.select("_seg", "_vbar"), on="_seg", maintain_order="left")
           .with_columns((wt * (pl.col("_v") - pl.col("_vbar")) ** 2).cum_sum().over("_seg").alias("_WD"))
           .group_by("_seg", maintain_order=True).agg(pl.col("_WD").last()))
    # time-weighted quantiles: sort speeds, accumulate time, first speed reaching q of the total
    q = (valid.sort("_seg", "_v", "_i").with_columns(wt.cum_sum().over("_seg").alias("_cw"))
         .join(tot.select("_seg", "_W"), on="_seg", maintain_order="left")
         .group_by("_seg", maintain_order=True).agg(
             pl.col("_v").filter(pl.col("_cw") >= 0.5 * pl.col("_W")).first().alias("speed_median_kmh"),
             pl.col("_v").filter(pl.col("_cw") >= 0.85 * pl.col("_W")).first().alias("speed_p85_kmh")))
    base = s.with_columns(pl.col("_jump").cast(pl.UInt32).cum_sum().over("_seg").alias("_nj")).group_by(
        "_seg", maintain_order=True).agg(
        pl.col("_nj").last().alias("n_jumps"),
        (pl.col("_dt").drop_nulls().median() / 1000).alias("ping_gap_median_s"),
    )
    return (base.join(tot, on="_seg", how="left").join(var, on="_seg", how="left").join(q, on="_seg", how="left")
            .with_columns(
                pl.col("_vbar").alias("speed_mean_kmh"),
                pl.col("_VMAX").alias("speed_max_kmh"),
                pl.when(pl.col("_vbar") > 0).then((pl.col("_WD") / pl.col("_W")).sqrt() / pl.col("_vbar")).alias("speed_cv"),
                (pl.col("_WS") / pl.col("_W")).alias("frac_slow"),
                pl.col("n_jumps").fill_null(0).cast(pl.UInt32))
            .select("_seg", *_FEATURE_COLUMNS, "n_jumps"))


def guess_mode(trajectories: pl.DataFrame | pl.LazyFrame, *, walk_p85_kmh: float = 8, walk_max_kmh: float = 15,
               bike_p85_kmh: float = 22,
               bike_max_kmh: float = 35, min_duration_s: float = 60, min_pings: int = 2,
               max_ping_gap_s: float = 300):
    """Rule-based baseline transport mode: ``walk``, ``bike``, ``motorized`` or ``unknown``.

    * ``unknown``: shorter than ``min_duration_s``, fewer than ``min_pings`` pings, no speeds, or
      pings too sparse to measure speed (median gap above ``max_ping_gap_s``: the "speed"
      then averages movement with standing still);
    * ``walk``: cruising speed (p85) below ``walk_p85_kmh``, never above ``walk_max_kmh``, and an
      overall trip speed (``length_m / duration``, gaps included) below ``walk_max_kmh``;
    * ``bike``: p85 below ``bike_p85_kmh``, never above ``bike_max_kmh``, and an overall trip
      speed below ``bike_max_kmh``;
    * ``motorized``: anything faster (cars, buses, metro, ...).

    A transparent starting point to replace with a trained model; recompute with other
    thresholds with ``guess_mode(trajectories, walk_p85_kmh=...)``.
    """
    p85, vmax = pl.col("speed_p85_kmh"), pl.col("speed_max_kmh")
    # Overall speed over the whole trip, unobserved gaps included: a lower bound of the moving
    # speed, so it catches distance covered across long gaps that the windowed speeds skip.
    overall = pl.when(pl.col("duration") > 0).then(pl.col("length_m") / pl.col("duration") * 3.6).otherwise(0.0)
    sparse = pl.col("ping_gap_median_s") > max_ping_gap_s
    expr = (pl.when((pl.col("duration") < min_duration_s) | (pl.col("n_pings") < min_pings) | p85.is_null()
                    | sparse.fill_null(True))
            .then(pl.lit("unknown"))
            .when((p85 < walk_p85_kmh) & (vmax < walk_max_kmh) & (overall < walk_max_kmh)).then(pl.lit("walk"))
            .when((p85 < bike_p85_kmh) & (vmax < bike_max_kmh) & (overall < bike_max_kmh)).then(pl.lit("bike"))
            .otherwise(pl.lit("motorized")).alias("mode_guess"))
    return trajectories.with_columns(expr)


def douglas_peucker(vertices: pl.DataFrame, tolerance: float, group: str = "_seg",
                    use_cuda: bool = False) -> pl.Series:
    """Douglas-Peucker ``keep`` mask for many polylines at once.

    ``vertices`` must be sorted by ``group, _i`` and hold planar coordinates
    ``_x, _y`` in meters. Returns a boolean Series aligned with ``vertices``.

    Breadth-first: each iteration splits every open segment of every line at
    its farthest vertex (if farther than ``tolerance``). Vertices of segments
    that did not split can never be kept and leave the working set, so later
    iterations only touch the few long, wiggly segments left.
    """
    i = pl.col("_i")
    if tolerance <= 0:
        return pl.Series("keep", [True] * vertices.height)
    v = engine.collect(vertices.lazy().select(group, "_i", "_x", "_y").with_row_index("_rid").with_columns(
        ((i == 0) | (i == i.max().over(group))).alias("_keep")), use_cuda)
    kept = [v.filter("_keep").get_column("_rid")]
    k = pl.col("_keep")
    while v.height:
        d = v.lazy().with_columns(
            pl.when(k).then(i).forward_fill().alias("_a"),
            pl.when(k).then(pl.col("_x")).forward_fill().alias("_ax"),
            pl.when(k).then(pl.col("_y")).forward_fill().alias("_ay"),
            pl.when(k).then(pl.col("_x")).backward_fill().alias("_bx"),
            pl.when(k).then(pl.col("_y")).backward_fill().alias("_by"),
        ).with_columns(
            (pl.col("_bx") - pl.col("_ax")).alias("_dx"), (pl.col("_by") - pl.col("_ay")).alias("_dy"),
        ).with_columns((pl.col("_dx") ** 2 + pl.col("_dy") ** 2).alias("_l2")).with_columns(
            pl.when(pl.col("_l2") > 0).then(
                (((pl.col("_x") - pl.col("_ax")) * pl.col("_dx") + (pl.col("_y") - pl.col("_ay")) * pl.col("_dy"))
                 / pl.col("_l2")).clip(0.0, 1.0)
            ).otherwise(0.0).alias("_t")
        ).with_columns(
            pl.when(~k).then(
                ((pl.col("_x") - pl.col("_ax") - pl.col("_t") * pl.col("_dx")) ** 2
                 + (pl.col("_y") - pl.col("_ay") - pl.col("_t") * pl.col("_dy")) ** 2).sqrt()
                # micrometer resolution: last-bit differences between CPU and GPU arithmetic
                # cannot flip equal-distance ties, which then break by position on both engines
                .mul(1e6).round().truediv(1e6)
            ).alias("_d")
        ).with_columns(pl.col("_d").max().over(group, "_a").alias("_dmax")).with_columns(
            ((pl.col("_d") == pl.col("_dmax")) & (pl.col("_d") > tolerance)).fill_null(False).alias("_cand")
        ).with_columns(
            (pl.col("_cand") & (pl.col("_cand").cast(pl.Int32).cum_sum().over(group, "_a") == 1)).alias("_new"),
            (pl.col("_dmax") > tolerance).fill_null(False).alias("_open"),
        )
        d = engine.collect(d, use_cuda)
        new = d.filter("_new")
        if new.height == 0:
            break
        kept.append(new.get_column("_rid"))
        # Keep kept vertices (segment endpoints) and the interior of segments that split.
        v = engine.collect(
            d.lazy().with_columns((k | pl.col("_new")).alias("_keep"))
            .filter(pl.col("_keep") | pl.col("_open"))
            .select("_rid", group, "_i", "_x", "_y", "_keep")
            # drop lines whose remaining vertices are all kept
            .filter((~pl.col("_keep")).cast(pl.Int8).max().over(group) > 0), use_cuda)
    ids = pl.concat(kept).sort()
    return pl.Series("keep", [False] * vertices.height).scatter(ids, True)


def build_trajectories(rows: pl.DataFrame | pl.LazyFrame, stops: pl.DataFrame | pl.LazyFrame,
                       params: TrajectoryParams | None = None, offsets: pl.DataFrame | None = None,
                       use_cuda: bool = False) -> pl.DataFrame:
    """Trajectories from canonical pings ``rows`` and the stays ``stops``.

    ``rows``: ``uid, timestamp (ms), latitude, longitude``.
    ``stops``: at least ``uid, stop_event, start_timestamp, end_timestamp,
    stop_latitude, stop_longitude`` (timestamps in seconds).
    ``offsets``: optional ``uid, n_trajectories`` to continue the numbering.
    """
    tp = params or TrajectoryParams()
    gap = float(tp.max_time_gap)
    cap_ms = float(tp.max_speed_kmh) / 3600.0          # m per ms
    st = stops.lazy().select(_STOP_COLS).unique(subset=["uid", "stop_event"]).sort("start_timestamp")
    prev = st.select("uid", *[pl.col(c).alias(f"_p_{c}") for c in _STOP_COLS[1:]])
    nxt = st.select("uid", *[pl.col(c).alias(f"_n_{c}") for c in _STOP_COLS[1:]])

    r = rows.lazy().with_columns((pl.col("timestamp") // 1000).alias("_ts")).sort("_ts")
    moving = (
        r.join_asof(prev, left_on="_ts", right_on="_p_start_timestamp", by="uid", strategy="backward",
                    check_sortedness=False)
        .filter(pl.col("_p_end_timestamp").is_null() | (pl.col("_ts") > pl.col("_p_end_timestamp")))
        .sort("uid", "timestamp", "latitude", "longitude")
        .with_columns(
            # GPS jump: faster than max_speed_kmh from the previous moving ping (same-second
            # pings count as one second apart). Jumps split trajectories like long gaps do.
            (haversine(pl.col("latitude").shift(1), pl.col("longitude").shift(1), "latitude", "longitude")
             > cap_ms * pl.max_horizontal(pl.col("timestamp") - pl.col("timestamp").shift(1), 1000)
             ).fill_null(False).alias("_jump"))
        .with_columns(
            ((pl.col("uid") != pl.col("uid").shift(1)).fill_null(True)
             | pl.col("_p_stop_event").ne_missing(pl.col("_p_stop_event").shift(1))
             | ((pl.col("timestamp") - pl.col("timestamp").shift(1)) > gap * 1000).fill_null(False)
             | pl.col("_jump")
             ).cast(pl.Int64).cum_sum().alias("_seg")
        )
    )
    moving = moving.collect()
    if moving.height == 0:
        return empty_trajectories()

    segs = (
        moving.lazy().group_by("_seg").agg(
            pl.col("uid").first(), pl.col("^_p_.*$").first(),
            pl.col("_ts").min().alias("_t0"), pl.col("_ts").max().alias("_t1"),
            pl.col("latitude").first().alias("_la0"), pl.col("longitude").first().alias("_lo0"),
            pl.col("latitude").last().alias("_la1"), pl.col("longitude").last().alias("_lo1"),
            pl.len().alias("n_pings"),
        )
        # pieces between the same two stays (split by jumps): only the first may connect to the
        # origin stay and only the last to the destination stay
        .sort("uid", "_t0", "_seg")
        .with_columns(pl.col("_t0").rank("ordinal").over("uid", "_p_stop_event").alias("_piece"),
                      pl.len().over("uid", "_p_stop_event").alias("_npieces"))
        .sort("_t1")
        .join_asof(nxt, left_on="_t1", right_on="_n_start_timestamp", by="uid", strategy="forward",
                   check_sortedness=False)
        .with_columns(
            (pl.col("_p_end_timestamp").is_not_null() & ((pl.col("_t0") - pl.col("_p_end_timestamp")) <= gap)
             & (pl.col("_piece") == 1)
             & (haversine("_p_stop_latitude", "_p_stop_longitude", "_la0", "_lo0")
                <= cap_ms * 1000 * pl.max_horizontal(pl.col("_t0") - pl.col("_p_end_timestamp"), 1))).alias("_o"),
            (pl.col("_n_start_timestamp").is_not_null() & ((pl.col("_n_start_timestamp") - pl.col("_t1")) <= gap)
             & (pl.col("_piece") == pl.col("_npieces"))
             & (haversine("_la1", "_lo1", "_n_stop_latitude", "_n_stop_longitude")
                <= cap_ms * 1000 * pl.max_horizontal(pl.col("_n_start_timestamp") - pl.col("_t1"), 1))).alias("_d"),
        )
        .with_columns(
            pl.when("_o").then(pl.col("_p_stop_event")).alias("origin_stop_event"),
            pl.when("_d").then(pl.col("_n_stop_event")).alias("destination_stop_event"),
        )
        .filter((pl.col("n_pings") >= tp.min_pings)
                & ((pl.col("n_pings") + pl.col("_o").cast(pl.UInt32) + pl.col("_d").cast(pl.UInt32)) >= 2))
        .collect()
    )
    if segs.height == 0:
        return empty_trajectories()

    v_cols = ["_seg", "_ord", "timestamp", "latitude", "longitude"]
    vertices = pl.concat([
        segs.filter("_o").select("_seg", pl.lit(0).alias("_ord"), (pl.col("_p_end_timestamp") * 1000).alias("timestamp"),
                                 pl.col("_p_stop_latitude").alias("latitude"), pl.col("_p_stop_longitude").alias("longitude")),
        moving.join(segs.select("_seg"), on="_seg", how="semi").select(
            "_seg", pl.lit(1).alias("_ord"), "timestamp", "latitude", "longitude"),
        segs.filter("_d").select("_seg", pl.lit(2).alias("_ord"), (pl.col("_n_start_timestamp") * 1000).alias("timestamp"),
                                 pl.col("_n_stop_latitude").alias("latitude"), pl.col("_n_stop_longitude").alias("longitude")),
    ], how="vertical_relaxed").select(v_cols).sort("_seg", "_ord", "timestamp", "latitude", "longitude")

    lat0 = pl.col("latitude").first().over("_seg") * DEG  # projection reference (exact on any engine)
    vertices = with_group_row(vertices, "_seg", "_i").with_columns(
        (pl.col("longitude") * lat0.cos() * M_PER_DEG).alias("_x"),
        (pl.col("latitude") * M_PER_DEG).alias("_y"),
        pl.when(pl.col("_seg") == pl.col("_seg").shift(1)).then(
            haversine(pl.col("latitude").shift(1), pl.col("longitude").shift(1), "latitude", "longitude")
        ).otherwise(0.0).alias("_step"),
    )
    vertices = vertices.with_columns(douglas_peucker(vertices, float(tp.simplify_tolerance), use_cuda=use_cuda))
    features = speed_features(vertices, tp)  # from every vertex, before simplification
    ends = vertices.group_by("_seg", maintain_order=True).agg(
        pl.col("latitude").first().alias("_la0"), pl.col("longitude").first().alias("_lo0"),
        pl.col("latitude").last().alias("_la1"), pl.col("longitude").last().alias("_lo1"))

    stats = engine.collect(
        vertices.lazy().group_by("_seg").agg(
            pl.col("timestamp").min().alias("_vs"), pl.col("timestamp").max().alias("_ve"),
            # integer micrometers: the sum does not depend on order (CPU == GPU)
            (pl.col("_step") * 1e6).round().cast(pl.Int64).sum().alias("_len_um"),
            pl.col("keep").cast(pl.UInt32).sum().alias("n_points"),
        ), use_cuda)
    # Text / list building stays on the CPU: float -> string formatting differs on the GPU.
    point = (pl.format("{} {} {}", "longitude", "latitude", pl.col("timestamp") // 1000) if tp.wkt_measure
             else pl.format("{} {}", "longitude", "latitude"))
    shapes = vertices.lazy().filter("keep").group_by("_seg", maintain_order=True).agg(
        point.str.join(", ").alias("_wkt"),
        pl.concat_list("longitude", "latitude").alias("coordinates"),
        (pl.col("timestamp") // 1000).alias("timestamps"),
    ).collect()
    lines = stats.join(shapes, on="_seg")

    out = segs.join(lines, on="_seg").join(features, on="_seg", how="left").join(ends, on="_seg").with_columns(
        (pl.col("_len_um") / 1e6).alias("length_m"),  # divided on the CPU: bit-identical for both engines
        (pl.col("_vs") // 1000).alias("start_timestamp"), (pl.col("_ve") // 1000).alias("end_timestamp"),
        pl.concat_str(pl.lit("LINESTRING M (" if tp.wkt_measure else "LINESTRING ("), pl.col("_wkt"), pl.lit(")"))
        .alias("geometry"),
    ).with_columns(
        (pl.col("end_timestamp") - pl.col("start_timestamp")).alias("duration"),
        pl.when(pl.col("length_m") > 0).then(
            (haversine("_la0", "_lo0", "_la1", "_lo1") / pl.col("length_m")).clip(upper_bound=1.0)
        ).alias("straightness"),
    ).filter(pl.col("length_m") >= float(tp.min_length_m)).pipe(guess_mode)
    if offsets is not None and offsets.height:
        out = out.join(offsets.select("uid", "n_trajectories"), on="uid", how="left")
    else:
        out = out.with_columns(pl.lit(0, pl.Int64).alias("n_trajectories"))
    out = out.sort("uid", "start_timestamp", "_seg").pipe(with_group_row, "uid", "_k", 1).with_columns(
        (pl.col("n_trajectories").fill_null(0) + pl.col("_k")).alias("trajectory_id"))
    return out.select(TRAJECTORY_COLUMNS).cast({"origin_stop_event": pl.Int64, "destination_stop_event": pl.Int64})


def extract_trajectories(pings: PingSource, stops: pl.DataFrame | pl.LazyFrame | str,
                         params: TrajectoryParams | None = None, *, column_map: dict[str, str] | None = None,
                         use_cuda: bool = False, **kwargs) -> pl.DataFrame:
    """Trajectories from raw pings and already computed stays.

    ``stops`` can be the output of :func:`polaroam.detect_stops`, of the
    clustering / home-work stages, or a parquet path to any of them.
    Keyword arguments override :class:`TrajectoryParams` fields.
    """
    tp = params or TrajectoryParams()
    if kwargs:
        tp = TrajectoryParams(**{**tp.__dict__, **kwargs})
    if isinstance(stops, str):
        stops = pl.scan_parquet(stops)
    rows = (read_pings(pings, column_map)
            .unique(subset=["uid", "latitude", "longitude", "timestamp"])
            .sort("uid", "timestamp", "latitude", "longitude"))
    return build_trajectories(rows, stops, tp, use_cuda=use_cuda)
