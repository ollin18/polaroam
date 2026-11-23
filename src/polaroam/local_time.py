"""Local-time columns for clustered stops (gc-lbs ``local_time.sh``).

The timezone is either fixed (``timezone="America/Mexico_City"``) or looked up
per stop in timezone polygons: ``tz_polygons="builtin"`` uses the polygons shipped
with polaroam (the ``tz_codes.csv`` of gc-lbs, 418 polygons, stored exactly), or pass
a ``tz,geometry`` CSV of WKT polygons. Stops falling outside every polygon use
``timezone`` as fallback (``inputed_timezone``).

The lookup is exact and fast: stop coordinates go on a lat/lon grid; a grid cell
lying inside a single polygon (and touching no other) gives its timezone by a plain
join, and only the points in cells crossing a border are tested against the polygons
(shapely's vectorized STRtree, ``tz`` extra). Classified cells are cached, so repeated
lookups only classify new cells.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import polars as pl

from . import engine

BUILTIN = "builtin"
_DATA = Path(__file__).parent / "data" / "timezones.npz"
#: grid cell size (degrees). ~0.1 deg was fastest for country-wide data (Mexico, Brazil)
CELL_DEG = 0.1

LOCAL_COLUMNS = [
    "timezone", "inputed_timezone", "stop_datetime", "end_stop_datetime", "date", "year", "month",
    "day", "hour", "weekday", "weekend", "end_date",
]


@functools.lru_cache(maxsize=1)
def _builtin():
    """Bundled polygons as ``(tz, country_code, shapely MultiPolygons)``."""
    import shapely
    from shapely import GeometryType

    z = np.load(_DATA)
    coords = np.cumsum(z["coords"], axis=0, dtype=np.int64) / 1e6
    geoms = shapely.from_ragged_array(GeometryType.MULTIPOLYGON, coords,
                                      (z["ring_offsets"], z["polygon_offsets"], z["geometry_offsets"]))
    return pl.Series("tz", z["tz"]), pl.Series("country_code", z["country_code"]), geoms


def builtin_timezones() -> pl.DataFrame:
    """The bundled timezone polygons: ``tz, country_code, geometry`` (WKB).

    418 polygons (the ``tz_codes.csv`` of gc-lbs). ``shapely.from_wkb(df["geometry"])``
    gives shapely geometries; to edit them, write a ``tz,geometry`` CSV with WKT and
    pass its path as ``tz_polygons``.
    """
    import shapely

    names, codes, geoms = _builtin()
    return pl.DataFrame([names, codes, pl.Series("geometry", shapely.to_wkb(geoms), dtype=pl.Binary)])


@functools.lru_cache(maxsize=4)
def _tz_tree(path: str, country_code: str | None):
    import shapely

    if path == BUILTIN:
        names, codes, geoms = _builtin()
        geoms = geoms.copy()
    else:
        tz = pl.read_csv(path, infer_schema_length=0)
        tz = tz.filter(pl.col("geometry").str.contains(r"^\s*(MULTI)?POLYGON"))
        names = tz.get_column("tz")
        codes = tz.get_column("country_code") if "country_code" in tz.columns else None
        geoms = shapely.from_wkt(tz.get_column("geometry").to_numpy())
    if country_code and codes is not None:
        keep = (codes == country_code).to_numpy()
        names, geoms = names.filter(keep), geoms[keep]
    shapely.prepare(geoms)  # prepared polygons: fast repeated point-in-polygon tests
    return geoms, names


def _point_in_polygon(points: pl.DataFrame, geoms, names: pl.Series, lat: str, lon: str) -> pl.DataFrame:
    """Exact lookup: ``points`` plus ``timezone`` (null outside every polygon)."""
    import shapely

    pts = shapely.points(points.get_column(lon).to_numpy(), points.get_column(lat).to_numpy())
    # Index the points and query with the (prepared) polygons: polygon.contains(point)
    # is ST_Within(point, polygon), and shapely prepares the query geometries.
    ig, ip = shapely.STRtree(pts).query(geoms, predicate="contains")
    # A point inside overlapping polygons takes the first one (no duplicated stops).
    hit = pl.DataFrame({"_i": ip, "_g": ig}).sort("_i", "_g").unique(subset="_i", keep="first")
    hit = hit.with_columns(names.gather(hit.get_column("_g")).alias("timezone"))
    return (points.with_row_index("_i").with_columns(pl.col("_i").cast(pl.Int64))
            .join(hit.select(pl.col("_i").cast(pl.Int64), "timezone"), on="_i", how="left", maintain_order="left")
            .drop("_i"))


_CELLS: dict[tuple, pl.DataFrame] = {}  # classified grid cells per (polygons, country, cell size)
_CELL_SCHEMA = {"_ci": pl.Int32, "_cj": pl.Int32, "_n": pl.UInt32, "_tz": pl.String}


def _classify_cells(cells: pl.DataFrame, geoms, names: pl.Series, cell: float) -> pl.DataFrame:
    """For each cell: how many polygons it touches (``_n``) and, when it lies in the
    interior of exactly one polygon, that polygon's timezone (``_tz``)."""
    import shapely

    ci, cj = cells.get_column("_ci").to_numpy(), cells.get_column("_cj").to_numpy()
    eps = 1e-9  # a slightly larger box: points are never outside their cell by float rounding
    boxes = shapely.box(cj * cell - eps, ci * cell - eps, (cj + 1) * cell + eps, (ci + 1) * cell + eps)
    tree = shapely.STRtree(boxes)
    b_hit = tree.query(geoms, predicate="intersects")[1]
    n_hit = np.bincount(b_hit, minlength=len(boxes)).astype(np.uint32)
    g_in, b_in = tree.query(geoms, predicate="contains_properly")
    ok = n_hit[b_in] == 1
    poly = np.full(len(boxes), -1, dtype=np.int64)
    poly[b_in[ok]] = g_in[ok]
    g = pl.Series("_g", poly)
    tz = names.gather(g.clip(lower_bound=0)).alias("_tz")
    return cells.with_columns(pl.Series("_n", n_hit), pl.when(g >= 0).then(tz).otherwise(None).alias("_tz"))


def lookup_timezones(points: pl.DataFrame, tz_polygons: str, country_code: str | None = None,
                     lat: str = "stop_latitude", lon: str = "stop_longitude", *,
                     cell_deg: float = CELL_DEG) -> pl.DataFrame:
    """``points`` (unique lat/lon) -> same rows plus ``timezone`` (null if outside every polygon).

    Exactly the result of testing every point against every polygon (a point inside
    overlapping polygons takes the first one), but only points near a border are tested.
    """
    geoms, names = _tz_tree(str(tz_polygons), country_code)
    key = (str(tz_polygons), country_code, cell_deg)
    p = points.with_columns((pl.col(lat) / cell_deg).floor().cast(pl.Int32).alias("_ci"),
                            (pl.col(lon) / cell_deg).floor().cast(pl.Int32).alias("_cj"))
    known = _CELLS.get(key, pl.DataFrame(schema=_CELL_SCHEMA))
    new = p.select("_ci", "_cj").unique().join(known, on=["_ci", "_cj"], how="anti")
    if new.height:
        known = pl.concat([known, _classify_cells(new, geoms, names, cell_deg)])
        _CELLS[key] = known
    p = p.with_row_index("_r").join(known, on=["_ci", "_cj"], how="left")
    border = (pl.col("_n") > 0) & pl.col("_tz").is_null()
    exact = _point_in_polygon(p.filter(border).select("_r", lat, lon), geoms, names, lat, lon)
    tz = pl.concat([p.filter(~border).select("_r", pl.col("_tz").alias("timezone")), exact.select("_r", "timezone")])
    return points.with_row_index("_r").join(tz, on="_r", how="left", maintain_order="left").drop("_r")


def timezone_table(stops: pl.DataFrame, tz_polygons: str, *, known: pl.DataFrame | None = None,
                   country_code: str | None = None) -> pl.DataFrame:
    """``stop_latitude, stop_longitude, timezone`` for every coordinate in ``stops``, plus
    the rows of ``known`` (a previous result): only new coordinates are looked up."""
    coords = stops.select("stop_latitude", "stop_longitude").unique()
    if known is not None:
        coords = coords.join(known, on=["stop_latitude", "stop_longitude"], how="anti")
    found = lookup_timezones(coords, tz_polygons, country_code)
    return found if known is None else pl.concat([known.select(found.columns), found])


def add_local_time(stops: pl.DataFrame | pl.LazyFrame, timezone: str = "UTC", tz_polygons: str | None = None,
                   *, country_code: str | None = None, timezones: pl.DataFrame | None = None,
                   use_cuda: bool = False) -> pl.DataFrame:
    """Add local datetime, date parts, weekday (1=Sunday ... 7=Saturday) and weekend.

    ``timezone`` is the fixed timezone, or the fallback when ``tz_polygons`` is given.
    ``tz_polygons="builtin"`` uses the polygons shipped with polaroam; a path uses a
    ``tz,geometry`` CSV of WKT polygons. ``timezones`` is an optional table of
    ``stop_latitude, stop_longitude, timezone`` already looked up (see
    :func:`timezone_table`); only coordinates missing from it are looked up.
    """
    df = engine.collect(stops.lazy(), use_cuda)
    if tz_polygons:
        tzs = timezone_table(df, tz_polygons, known=timezones, country_code=country_code)
        df = df.join(tzs, on=["stop_latitude", "stop_longitude"], how="left")
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
