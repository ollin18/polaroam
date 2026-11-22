"""Stop (stay) detection, fully vectorized and incremental.

The algorithm is a line-by-line translation of the BigQuery / Sedona SQL used
in ``gc-lbs`` (``process_yearly_stays.sh``):

1. de-duplicate pings and order them per device;
2. a ping is *stationary* if it is within ``distance_threshold`` meters and
   ``time_threshold`` seconds of its next **or** previous ping;
3. a new stay starts at a stationary ping whose previous ping is not stationary
   or is farther than ``distance_threshold`` (a jump);
4. stays shorter than ``min_stop_duration`` are dropped; pings farther than
   ``distance_threshold`` from the stay median are removed and the stay is
   re-aggregated (and re-checked for duration).

Everything is expressed with column-wise polars expressions (shifts masked at
device boundaries, cumulative sums, group-by aggregations and joins); there is
no per-device Python code, so polars parallelizes it over all cores.

Incremental processing
----------------------
Only the last ping of a device has an undetermined ``stationary`` flag (it
depends on the next ping), so a stay is *final* once its last ping is not one
of the device's last two pings. :class:`StopDetector` emits final stays and
carries, per device, the pings of the still-open stay (plus the trajectory
tail and two context pings) to the next batch. Processing data day by day and
calling :meth:`StopDetector.flush` at the end yields exactly the stays of a
single run over all the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from . import engine
from .config import StopParams, TrajectoryParams
from .geo import haversine, with_group_row
from .io import PingSource, read_pings
from .quality import SinkOption, remove_sink_pings, sink_table

STOP_COLUMNS = [
    "uid", "stop_event", "start_timestamp", "end_timestamp", "stop_duration",
    "stop_latitude", "stop_longitude", "n_pings",
]

_CARRY_SCHEMA = {"uid": pl.String, "latitude": pl.Float64, "longitude": pl.Float64,
                 "timestamp": pl.Int64, "_ctx": pl.Boolean}
_DEVICE_SCHEMA = {"uid": pl.String, "n_stops": pl.Int64, "n_trajectories": pl.Int64,
                  "last_stop_event": pl.Int64, "last_start_timestamp": pl.Int64,
                  "last_end_timestamp": pl.Int64, "last_latitude": pl.Float64, "last_longitude": pl.Float64}


def label_pings(pings: pl.LazyFrame, p: StopParams) -> pl.LazyFrame:
    """Add the SQL ``stationary`` flag and stay id ``_ev`` to sorted pings.

    ``pings`` must be sorted by ``uid, timestamp``. Shifts are masked where
    the device changes, which reproduces ``LAG/LEAD OVER (PARTITION BY ...)``
    including SQL's three-valued logic (polars uses Kleene logic too).
    """
    D, T = float(p.distance_threshold), float(p.time_threshold)
    uid, lat, lon, ts = pl.col("uid"), pl.col("latitude"), pl.col("longitude"), pl.col("timestamp")
    same_next = uid.shift(-1) == uid
    same_prev = uid.shift(1) == uid
    return (
        pings.with_columns(
            pl.when(same_next).then(haversine(lat, lon, lat.shift(-1), lon.shift(-1))).alias("_dist"),
            pl.when(same_next).then((ts.shift(-1) - ts) / 1000.0).alias("_tdiff"),
        )
        .with_columns(
            pl.when(same_prev).then(pl.col("_dist").shift(1)).alias("_pdist"),
            pl.when(same_prev).then(pl.col("_tdiff").shift(1)).alias("_ptdiff"),
        )
        .with_columns(
            (
                ((pl.col("_dist") <= D) & ((pl.col("_tdiff") <= T) | pl.col("_tdiff").is_null()))
                | ((pl.col("_pdist") <= D) & ((pl.col("_ptdiff") <= T) | pl.col("_ptdiff").is_null()))
            ).alias("stationary")
        )
        .with_columns(pl.when(same_prev).then(pl.col("stationary").shift(1)).alias("_pstat"))
        .with_columns(
            pl.when(
                pl.col("stationary")
                & (pl.col("_pstat").is_null() | ~pl.col("_pstat") | (pl.col("_pdist") > D))
            ).then(1).otherwise(0).alias("_ev_start")
        )
        .with_columns(pl.col("_ev_start").cum_sum().over("uid").alias("_ev"))
        .pipe(with_group_row, "uid", "_row")
        .drop("_pdist", "_ptdiff", "_pstat", "_ev_start", "_dist")
    )


def _aggregate(points: pl.LazyFrame, min_dur_ms: float) -> pl.LazyFrame:
    # APPROX_QUANTILES(x, 100)[OFFSET(50)] / percentile_approx(x, 0.5) return an
    # observed value: the lower median.
    return (
        points.group_by("uid", "_ev")
        .agg(
            pl.col("timestamp").min().alias("_start"),
            pl.col("timestamp").max().alias("_end"),
            pl.col("latitude").quantile(0.5, "lower").alias("_mlat"),
            pl.col("longitude").quantile(0.5, "lower").alias("_mlon"),
            pl.col("_row").max().alias("_last_row"),
            pl.len().alias("n_pings"),
        )
        .filter((pl.col("_end") - pl.col("_start")) >= min_dur_ms)
    )


def stays_from_points(points: pl.LazyFrame, p: StopParams) -> pl.LazyFrame:
    """Stationary points (``uid, _ev, _row, timestamp, latitude, longitude``) -> stays."""
    min_dur_ms = float(p.min_stop_duration) * 1000
    initial = _aggregate(points, min_dur_ms).select("uid", "_ev", "_mlat", "_mlon")
    filtered = points.join(initial, on=["uid", "_ev"]).filter(
        haversine("latitude", "longitude", "_mlat", "_mlon") <= float(p.distance_threshold)
    ).drop("_mlat", "_mlon")
    return _aggregate(filtered, min_dur_ms)


def _finish_stops(raw: pl.DataFrame, devices: pl.DataFrame) -> pl.DataFrame:
    """Seconds resolution, global ``stop_event`` numbering (ROW_NUMBER by start)."""
    return (
        raw.join(devices.select("uid", "n_stops"), on="uid", how="left")
        .sort("uid", "_start", "_ev")
        .pipe(with_group_row, "uid", "_k", 1)
        .with_columns(
            (pl.col("n_stops").fill_null(0) + pl.col("_k")).alias("stop_event"),
            (pl.col("_start") // 1000).alias("start_timestamp"),
            (pl.col("_end") // 1000).alias("end_timestamp"),
        )
        .with_columns((pl.col("end_timestamp") - pl.col("start_timestamp")).alias("stop_duration"))
        .rename({"_mlat": "stop_latitude", "_mlon": "stop_longitude"})
    )


@dataclass
class StopState:
    """Per-device state carried between incremental batches."""

    carry: pl.DataFrame = field(default_factory=lambda: pl.DataFrame(schema=_CARRY_SCHEMA))
    devices: pl.DataFrame = field(default_factory=lambda: pl.DataFrame(schema=_DEVICE_SCHEMA))

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.carry.write_parquet(d / "carry.parquet")
        self.devices.write_parquet(d / "devices.parquet")

    @classmethod
    def load(cls, directory: str | Path) -> StopState:
        d = Path(directory)
        if not (d / "carry.parquet").exists():
            return cls()
        return cls(pl.read_parquet(d / "carry.parquet"), pl.read_parquet(d / "devices.parquet"))


@dataclass
class StopBatch:
    """Output of one :meth:`StopDetector.update` call."""

    stops: pl.DataFrame
    trajectories: pl.DataFrame | None = None


class StopDetector:
    """Stateful, incremental stop detector (optionally extracting trajectories).

    >>> det = StopDetector(StopParams(), trajectories=True)
    >>> for day in days:                       # doctest: +SKIP
    ...     out = det.update(pings_of(day))    # final stops / trajectories so far
    >>> last = det.flush()                     # close the still-open stays
    """

    def __init__(self, params: StopParams | None = None, *, trajectories: bool = False,
                 trajectory_params: TrajectoryParams | None = None, state: StopState | None = None,
                 use_cuda: bool = False):
        self.params = params or StopParams()
        self.trajectories = trajectories
        self.trajectory_params = trajectory_params or TrajectoryParams()
        self.state = state or StopState()
        self.use_cuda = use_cuda

    def update(self, pings: pl.LazyFrame | pl.DataFrame, flush: bool = False) -> StopBatch:
        """Process a batch of canonical pings; return newly finalized stops."""
        stops, trajs, self.state = process_batch(
            pings, self.state, self.params, self.trajectory_params,
            flush=flush, with_trajectories=self.trajectories, use_cuda=self.use_cuda,
        )
        return StopBatch(stops, trajs)

    def flush(self) -> StopBatch:
        """Finalize every open stay (call once, after the last batch)."""
        return self.update(pl.DataFrame(schema={k: v for k, v in _CARRY_SCHEMA.items() if k != "_ctx"}), flush=True)


def process_batch(pings: pl.LazyFrame | pl.DataFrame, state: StopState, p: StopParams,
                  tp: TrajectoryParams, *, flush: bool, with_trajectories: bool,
                  use_cuda: bool = False) -> tuple[pl.DataFrame, pl.DataFrame | None, StopState]:
    """One incremental step. Returns ``(stops, trajectories, new_state)``."""
    new = engine.collect(pings.lazy().select("uid", "latitude", "longitude", "timestamp"), use_cuda, streaming=True)
    carry = state.carry
    if flush:
        carry_active, carry_idle = carry, carry.clear()
    else:
        active = new.select("uid").unique()
        carry_active = carry.join(active, on="uid", how="semi")
        carry_idle = carry.join(active, on="uid", how="anti")

    combined = (
        pl.concat([carry_active.lazy(), new.lazy().with_columns(pl.lit(False).alias("_ctx"))], how="vertical_relaxed")
        .unique(subset=["uid", "latitude", "longitude", "timestamp"], keep="first", maintain_order=True)
        .sort("uid", "timestamp", "latitude", "longitude")
    )
    labeled = engine.collect(label_pings(combined, p), use_cuda)

    # --- which stays are final --------------------------------------------------
    pts = labeled.lazy().filter(pl.col("stationary") & (pl.col("_ev") > 0))
    dev = labeled.lazy().group_by("uid").agg(pl.len().cast(pl.Int64).alias("_n"),
                                              pl.col("_ctx").sum().cast(pl.Int64).alias("_nctx"))
    if flush:
        dev = dev.with_columns(pl.col("_n").alias("_s"))
    else:
        pen = labeled.lazy().filter(pl.col("_row") == pl.len().over("uid") - 2).filter(
            pl.col("stationary").fill_null(False)).select("uid", pl.col("_ev").alias("_ev_pen"))
        open_start = (pts.join(pen, on="uid").filter(pl.col("_ev") == pl.col("_ev_pen"))
                      .group_by("uid").agg(pl.col("_row").min().alias("_open")))
        dev = dev.join(open_start, on="uid", how="left").with_columns(
            pl.min_horizontal("_open", pl.col("_n") - 1).alias("_s")).drop("_open")
    dev = engine.collect(dev, use_cuda)

    groups = (pts.group_by("uid", "_ev")
              .agg(pl.col("_row").max().alias("_gmax"), (pl.col("_ctx").cast(pl.Int8).max() > 0).alias("_gctx"))
              .join(dev.lazy().select("uid", "_s"), on="uid")
              .filter((pl.col("_gmax") < pl.col("_s")) & ~pl.col("_gctx"))
              .select("uid", "_ev"))
    emit_pts = pts.join(groups, on=["uid", "_ev"], how="semi").select(
        "uid", "_ev", "_row", "timestamp", "latitude", "longitude")
    raw = engine.collect(stays_from_points(emit_pts, p), use_cuda)
    stops_full = _finish_stops(raw, state.devices)
    stops = stops_full.select(STOP_COLUMNS)

    # --- carry-over boundary c: after the last closed stay / long gap ----------
    gap_s = float(tp.max_time_gap)
    b_stop = stops_full.group_by("uid").agg(pl.col("_last_row").max().alias("_bs"))
    b_gap = (labeled.lazy().join(dev.lazy().select("uid", "_s"), on="uid")
             .filter((pl.col("_tdiff") > gap_s) & (pl.col("_row") < pl.col("_s")))
             .group_by("uid").agg(pl.col("_row").max().alias("_bg")))
    b_gap = engine.collect(b_gap, use_cuda)
    dev = (dev.join(b_stop, on="uid", how="left").join(b_gap, on="uid", how="left")
           .with_columns(pl.max_horizontal(pl.max_horizontal("_bs", "_bg") + 1, "_nctx").alias("_c")))

    if flush:
        new_carry = carry.clear()
        region = labeled.filter(~pl.col("_ctx"))
    else:
        lab_c = labeled.join(dev.select("uid", "_c"), on="uid")
        new_carry = (lab_c.filter(pl.col("_row") >= pl.col("_c") - 2)
                     .with_columns((pl.col("_row") < pl.col("_c")).alias("_ctx"))
                     .select(list(_CARRY_SCHEMA)))
        region = lab_c.filter(~pl.col("_ctx") & (pl.col("_row") < pl.col("_c")))

    # --- trajectories -----------------------------------------------------------
    trajs = None
    if with_trajectories:
        from .trajectories import build_trajectories

        prev = state.devices.filter(pl.col("last_stop_event").is_not_null()).select(
            "uid", pl.col("last_stop_event").alias("stop_event"),
            pl.col("last_start_timestamp").alias("start_timestamp"),
            pl.col("last_end_timestamp").alias("end_timestamp"),
            pl.col("last_latitude").alias("stop_latitude"), pl.col("last_longitude").alias("stop_longitude"))
        stop_set = pl.concat([prev, stops.select(prev.columns)], how="vertical_relaxed")
        trajs = build_trajectories(region.select("uid", "timestamp", "latitude", "longitude"),
                                   stop_set, tp, offsets=state.devices.select("uid", "n_trajectories"),
                                   use_cuda=use_cuda)

    # --- device counters --------------------------------------------------------
    last = (stops.sort("uid", "stop_event").group_by("uid").agg(pl.all().last())
            .select("uid", pl.col("stop_event").alias("_lse"), pl.col("start_timestamp").alias("_lst"),
                    pl.col("end_timestamp").alias("_let"), pl.col("stop_latitude").alias("_llat"),
                    pl.col("stop_longitude").alias("_llon")))
    counts = stops.group_by("uid").agg(pl.len().cast(pl.Int64).alias("_ns"))
    tcounts = (trajs.group_by("uid").agg(pl.len().cast(pl.Int64).alias("_nt")) if trajs is not None
               else pl.DataFrame(schema={"uid": pl.String, "_nt": pl.Int64}))
    touched = pl.concat([state.devices.select("uid"), dev.select("uid")]).unique()
    devices = (
        touched.join(state.devices, on="uid", how="left")
        .join(counts, on="uid", how="left").join(tcounts, on="uid", how="left").join(last, on="uid", how="left")
        .select(
            "uid",
            (pl.col("n_stops").fill_null(0) + pl.col("_ns").fill_null(0)).alias("n_stops"),
            (pl.col("n_trajectories").fill_null(0) + pl.col("_nt").fill_null(0)).alias("n_trajectories"),
            pl.coalesce("_lse", "last_stop_event").alias("last_stop_event"),
            pl.coalesce("_lst", "last_start_timestamp").alias("last_start_timestamp"),
            pl.coalesce("_let", "last_end_timestamp").alias("last_end_timestamp"),
            pl.coalesce("_llat", "last_latitude").alias("last_latitude"),
            pl.coalesce("_llon", "last_longitude").alias("last_longitude"),
        )
        .cast(_DEVICE_SCHEMA)
    )
    new_state = StopState(pl.concat([carry_idle, new_carry], how="vertical_relaxed").cast(_CARRY_SCHEMA), devices)
    return stops, trajs, new_state


def detect_stops(pings: PingSource, params: StopParams | None = None, *, trajectories: bool = False,
                 trajectory_params: TrajectoryParams | None = None, column_map: dict[str, str] | None = None,
                 remove_sinks: SinkOption = False, sinks_min_devices: int = 5000,
                 use_cuda: bool = False, **kwargs) -> pl.DataFrame | tuple[pl.DataFrame, pl.DataFrame]:
    """Detect stays in one shot (same result as the BigQuery/Sedona pipeline).

    ``pings`` is a path / glob / (Lazy)DataFrame; columns are mapped to
    ``uid, latitude, longitude, timestamp`` (see :func:`polaroam.io.read_pings`).
    Extra keyword arguments override :class:`StopParams` fields, e.g.
    ``detect_stops(df, distance_threshold=30)``.

    With ``trajectories=True`` returns ``(stops, trajectories)``.

    ``remove_sinks`` drops the pings at coordinate sinks first: ``True`` finds them in
    ``pings`` (coordinates shared by ``>= sinks_min_devices`` devices, see
    :func:`polaroam.find_sinks`); a parquet path or a DataFrame of ``latitude, longitude``
    uses a list computed beforehand. The default ``False`` keeps every ping, like gc-lbs.
    """
    p = params or StopParams()
    if kwargs:
        p = StopParams(**{**p.__dict__, **kwargs})
    lf = read_pings(pings, column_map)
    lf = remove_sink_pings(lf, sink_table(remove_sinks, lf, sinks_min_devices))
    det = StopDetector(p, trajectories=trajectories, trajectory_params=trajectory_params, use_cuda=use_cuda)
    out = det.update(lf, flush=True)
    return (out.stops, out.trajectories) if trajectories else out.stops
