"""Home / work classification of clustered, local-time stops (gc-lbs ``get_hw.sh``).

* **Home**: the cluster with most stops during home time (weekends, or
  weekday hours ``>= home_hour_evening`` or ``< home_hour_morning``) among
  clusters visited on at least ``home_min_days`` distinct days.
* **Work**: the cluster with most weekday stops between ``work_hour_start``
  and ``work_hour_end`` (inclusive) visited on at least ``work_min_days``
  distinct days during work time, different from home and at least
  ``min_distance_home_work`` meters away from it. Devices without home get no work.

Ties are broken by more distinct days, then by the lower ``cluster_label``.
"""

from __future__ import annotations

import polars as pl

from . import engine
from .config import HomeWorkParams
from .geo import exact_mean, haversine

HW_COLUMNS = [
    "uid", "stop_event", "start_timestamp", "end_timestamp", "stop_duration",
    "stop_latitude", "stop_longitude", "cluster_label", "cluster_counts", "cluster_latitude",
    "cluster_longitude", "timezone", "inputed_timezone", "stop_datetime", "end_stop_datetime",
    "date", "year", "month", "day", "hour", "weekday", "weekend", "end_date", "location_type",
]


def _best(lf: pl.LazyFrame, visits: str, days: str) -> pl.LazyFrame:
    # ROW_NUMBER() OVER (PARTITION BY uid ORDER BY visits DESC, days DESC, cluster_label) = 1
    return (lf.sort(["uid", visits, days, "cluster_label"], descending=[False, True, True, False])
            .unique(subset="uid", keep="first", maintain_order=True))


def home_work_clusters(local: pl.DataFrame | pl.LazyFrame, params: HomeWorkParams | None = None,
                       use_cuda: bool = False) -> pl.DataFrame:
    """One row per device: home / work cluster labels and coordinates."""
    p = params or HomeWorkParams()
    base = local.lazy().filter(pl.col("cluster_label") >= 0)
    days = base.group_by("uid", "cluster_label").agg(pl.col("date").n_unique().alias("unique_days_visited"))

    def candidates(mask: pl.Expr, prefix: str) -> pl.LazyFrame:
        return base.filter(mask).group_by("uid", "cluster_label").agg(
            exact_mean("cluster_latitude").alias("cluster_latitude"), exact_mean("cluster_longitude").alias("cluster_longitude"),
            pl.len().alias(f"{prefix}_time_visits"), pl.col("date").n_unique().alias(f"{prefix}_time_unique_days"))

    hour, weekend = pl.col("hour"), pl.col("weekend")
    home = _best(
        candidates(weekend | (hour >= p.home_hour_evening) | (hour < p.home_hour_morning), "home")
        .join(days, on=["uid", "cluster_label"]).filter(pl.col("unique_days_visited") >= p.home_min_days),
        "home_time_visits", "home_time_unique_days",
    ).select("uid", pl.col("cluster_label").alias("home_cluster_label"),
             pl.col("cluster_latitude").alias("home_latitude"), pl.col("cluster_longitude").alias("home_longitude"))

    work = _best(
        candidates(~weekend & (hour >= p.work_hour_start) & (hour <= p.work_hour_end), "work")
        .join(home, on="uid")
        .filter((pl.col("cluster_label") != pl.col("home_cluster_label"))
                & (pl.col("work_time_unique_days") >= p.work_min_days)
                & (haversine("cluster_latitude", "cluster_longitude", "home_latitude", "home_longitude")
                   >= p.min_distance_home_work)),
        "work_time_visits", "work_time_unique_days",
    ).select("uid", pl.col("cluster_label").alias("work_cluster_label"),
             pl.col("cluster_latitude").alias("work_latitude"), pl.col("cluster_longitude").alias("work_longitude"))

    return engine.collect(home.join(work, on="uid", how="left"), use_cuda)


def classify_home_work(local: pl.DataFrame | pl.LazyFrame, params: HomeWorkParams | None = None, *,
                       use_cuda: bool = False, **kwargs) -> pl.DataFrame:
    """Label every stop with ``location_type`` H (home), W (work) or O (other)."""
    p = params or HomeWorkParams()
    if kwargs:
        p = HomeWorkParams(**{**p.__dict__, **kwargs})
    hw = home_work_clusters(local, p, use_cuda)
    cl = pl.col("cluster_label")
    out = (local.lazy().join(hw.lazy().select("uid", "home_cluster_label", "work_cluster_label"), on="uid", how="left")
           .with_columns(
               pl.when(cl == -1).then(pl.lit("O"))
               .when(cl == pl.col("home_cluster_label")).then(pl.lit("H"))
               .when(cl == pl.col("work_cluster_label")).then(pl.lit("W"))
               .otherwise(pl.lit("O")).alias("location_type"))
           .select(HW_COLUMNS).sort("uid", "date", "start_timestamp"))
    return engine.collect(out, use_cuda)


def home_work_locations(local_or_hw: pl.DataFrame | pl.LazyFrame, params: HomeWorkParams | None = None,
                        use_cuda: bool = False) -> pl.DataFrame:
    """Compact table: ``uid, home_latitude, home_longitude, work_latitude, work_longitude``."""
    return home_work_clusters(local_or_hw, params, use_cuda).select(
        "uid", "home_latitude", "home_longitude", "work_latitude", "work_longitude").sort("uid")
