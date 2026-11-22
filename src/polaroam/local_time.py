"""Local-time columns for clustered stops (gc-lbs ``local_time.sh``).

The timezone is either fixed (``timezone="America/Mexico_City"``) or looked up
per stop with a point-in-polygon join against a ``tz,geometry`` CSV of WKT
polygons (the ``tz_codes.csv`` shipped with gc-lbs). Stops falling outside every
polygon use ``timezone`` as fallback (``inputed_timezone``).

The polygon lookup runs on the *unique* stop coordinates with shapely's
vectorized STRtree (install the ``tz`` extra); everything else is polars.
"""

from __future__ import annotations

import functools

import polars as pl

from . import engine

LOCAL_COLUMNS = [
    "timezone", "inputed_timezone", "stop_datetime", "end_stop_datetime", "date", "year", "month",
    "day", "hour", "weekday", "weekend", "end_date",
]


@functools.lru_cache(maxsize=4)
def _tz_tree(path: str, country_code: str | None):
    import shapely

    tz = pl.read_csv(path, infer_schema_length=0)
    if country_code and "country_code" in tz.columns:
        tz = tz.filter(pl.col("country_code") == country_code)
    tz = tz.filter(pl.col("geometry").str.contains(r"^\s*(MULTI)?POLYGON"))
    geoms = shapely.from_wkt(tz.get_column("geometry").to_numpy())
    shapely.prepare(geoms)  # prepared polygons: fast repeated point-in-polygon tests
    return geoms, tz.get_column("tz")


def lookup_timezones(points: pl.DataFrame, tz_polygons: str, country_code: str | None = None,
                     lat: str = "stop_latitude", lon: str = "stop_longitude") -> pl.DataFrame:
    """``points`` (unique lat/lon) -> same rows plus ``timezone`` (null if none)."""
    import shapely

    geoms, names = _tz_tree(str(tz_polygons), country_code)
    pts = shapely.points(points.get_column(lon).to_numpy(), points.get_column(lat).to_numpy())
    # Index the points and query with the (prepared) polygons: polygon.contains(point)
    # is ST_Within(point, polygon), and shapely prepares the query geometries.
    ig, ip = shapely.STRtree(pts).query(geoms, predicate="contains")
    # A point inside overlapping polygons takes the first one (no duplicated stops).
    hit = pl.DataFrame({"_i": ip, "_g": ig}).sort("_i", "_g").unique(subset="_i", keep="first")
    hit = hit.with_columns(names.gather(hit.get_column("_g")).alias("timezone"))
    return (points.with_row_index("_i").with_columns(pl.col("_i").cast(pl.Int64))
            .join(hit.select(pl.col("_i").cast(pl.Int64), "timezone"), on="_i", how="left").drop("_i"))


def add_local_time(stops: pl.DataFrame | pl.LazyFrame, timezone: str = "UTC", tz_polygons: str | None = None,
                   *, country_code: str | None = None, use_cuda: bool = False) -> pl.DataFrame:
    """Add local datetime, date parts, weekday (1=Sunday ... 7=Saturday) and weekend.

    ``timezone`` is the fixed timezone, or the fallback when ``tz_polygons`` is given.
    """
    df = engine.collect(stops.lazy(), use_cuda)
    if tz_polygons:
        coords = df.select("stop_latitude", "stop_longitude").unique()
        df = df.join(lookup_timezones(coords, tz_polygons, country_code), on=["stop_latitude", "stop_longitude"], how="left")
    else:
        df = df.with_columns(pl.lit(timezone).alias("timezone"))
    df = df.with_columns(pl.coalesce("timezone", pl.lit(timezone)).alias("inputed_timezone"))

    def local(col: str) -> pl.Expr:
        utc = pl.from_epoch(pl.col(col), time_unit="s").dt.replace_time_zone("UTC")
        zones = df.get_column("inputed_timezone").unique().sort().to_list()
        expr = None
        for z in zones:
            conv = utc.dt.convert_time_zone(z).dt.replace_time_zone(None)
            expr = pl.when(pl.col("inputed_timezone") == z).then(conv) if expr is None else \
                expr.when(pl.col("inputed_timezone") == z).then(conv)
        return expr.otherwise(None) if expr is not None else utc.dt.replace_time_zone(None)

    sd = pl.col("stop_datetime")
    return df.with_columns(local("start_timestamp").alias("stop_datetime"),
                           local("end_timestamp").alias("end_stop_datetime")).with_columns(
        sd.dt.date().alias("date"),
        sd.dt.year().cast(pl.Int32).alias("year"),
        sd.dt.month().cast(pl.Int32).alias("month"),
        sd.dt.day().cast(pl.Int32).alias("day"),
        sd.dt.hour().cast(pl.Int32).alias("hour"),
        (sd.dt.weekday() % 7 + 1).cast(pl.Int32).alias("weekday"),
        (sd.dt.weekday() >= 6).alias("weekend"),
        pl.col("end_stop_datetime").dt.date().alias("end_date"),
    )
