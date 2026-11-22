"""H3 aggregation of locations (homes, work places, stays), with k-anonymity.

Uses the ``polars-h3`` plugin (``pip install polaroam[h3]``), so indexing is a
vectorized Rust expression like the rest of polaroam.

Fine resolutions isolate individuals: at resolution 10 (~66 m hexagon edge) a cell may
hold a single home. Every function here drops cells with fewer than ``min_count``
devices (k-anonymity, default 5) unless explicitly told otherwise.

Approximate hexagon sizes (average): resolution 8 - edge 531 m, 0.74 km²;
9 - 201 m, 0.11 km²; 10 - 76 m, 0.015 km²; 11 - 29 m, 0.002 km².
"""

from __future__ import annotations

import polars as pl


def _plh3():
    try:
        import polars_h3
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError("H3 support needs the polars-h3 plugin: pip install 'polaroam[h3]'") from e
    return polars_h3


def h3_cells(lat: str | pl.Expr, lon: str | pl.Expr, resolution: int) -> pl.Expr:
    """Expression: the H3 cell (UInt64) of each ``(lat, lon)`` at ``resolution``."""
    return _plh3().latlng_to_cell(lat, lon, resolution)


def with_geometry(cells: pl.DataFrame, cell: str = "h3_cell") -> pl.DataFrame:
    """Add cell center (``h3_latitude``, ``h3_longitude``), ``h3_index`` (hex string) and a WKT polygon."""
    h = _plh3()
    out = cells.with_columns(
        h.int_to_str(cell).alias("h3_index"),
        h.cell_to_lat(cell).alias("h3_latitude"),
        h.cell_to_lng(cell).alias("h3_longitude"),
        h.cell_to_boundary(cell).alias("_b"),
    )
    ring = pl.col("_b").list.eval(pl.format("{} {}", pl.element().list.get(1), pl.element().list.get(0)))
    return out.with_columns(
        pl.concat_str(pl.lit("POLYGON (("), ring.list.join(", "), pl.lit(", "),
                      pl.format("{} {}", pl.col("_b").list.first().list.get(1), pl.col("_b").list.first().list.get(0)),
                      pl.lit("))")).alias("geometry")
    ).drop("_b")


def h3_counts(df: pl.DataFrame | pl.LazyFrame, lat: str, lon: str, resolution: int = 9, *,
              count_unique: str | None = "uid", min_count: int = 5, geometry: bool = True) -> pl.DataFrame:
    """Count rows (or distinct ``count_unique`` values) per H3 cell.

    Cells with fewer than ``min_count`` are dropped (k-anonymity; ``min_count=1`` keeps all).
    """
    lf = df.lazy().filter(pl.col(lat).is_not_null() & pl.col(lon).is_not_null())
    agg = pl.col(count_unique).n_unique() if count_unique else pl.len()
    out = (lf.with_columns(h3_cells(lat, lon, resolution).alias("h3_cell"))
           .group_by("h3_cell").agg(agg.cast(pl.UInt32).alias("count"))
           .filter(pl.col("count") >= min_count).sort("h3_cell").collect())
    return with_geometry(out) if geometry else out


def home_work_h3(locations: pl.DataFrame | pl.LazyFrame, resolution: int = 9, *, min_count: int = 5,
                 geometry: bool = True) -> pl.DataFrame:
    """Homes and work places per H3 cell from a ``locations`` table.

    ``locations``: ``uid, home_latitude, home_longitude, work_latitude, work_longitude``
    (the pipeline's ``locations`` table or :func:`polaroam.home_work_locations`).

    Returns ``h3_cell, homes, workers`` (devices whose home / work is in the cell). A count
    below ``min_count`` is set to null, and cells where both are null are dropped.
    """
    lf = locations.lazy()
    homes = (lf.filter(pl.col("home_latitude").is_not_null())
             .group_by(h3_cells("home_latitude", "home_longitude", resolution).alias("h3_cell"))
             .agg(pl.col("uid").n_unique().cast(pl.UInt32).alias("homes")))
    work = (lf.filter(pl.col("work_latitude").is_not_null())
            .group_by(h3_cells("work_latitude", "work_longitude", resolution).alias("h3_cell"))
            .agg(pl.col("uid").n_unique().cast(pl.UInt32).alias("workers")))
    out = (homes.join(work, on="h3_cell", how="full", coalesce=True)
           .with_columns(pl.when(pl.col(c) >= min_count).then(pl.col(c)).alias(c) for c in ("homes", "workers"))
           .filter(pl.col("homes").is_not_null() | pl.col("workers").is_not_null())
           .sort("h3_cell").collect())
    return with_geometry(out) if geometry else out
