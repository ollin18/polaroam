"""Data-quality helpers.

**Coordinate sinks**: exact coordinates shared by many unrelated devices. These are
default geocodes (e.g. a city centroid used when only the IP or the country is known),
cell towers, or Wi-Fi positioning points. Real homes do not put thousands of devices on
the same 1e-6 degree point. In one month of Mexico City, 2,592 devices had their "home"
at exactly 19.4326, -99.1332 (the Zócalo, the city's textbook centroid).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import polars as pl

from .io import PingSource, read_pings

#: ``remove_sinks`` values: ``False`` (keep every ping, gc-lbs behavior), ``True`` (find the sinks
#: in the pings being processed), a parquet path or a DataFrame of ``latitude, longitude``.
SinkOption = bool | str | Path | pl.DataFrame | None


def coordinate_sinks(df: pl.DataFrame | pl.LazyFrame, lat: str = "latitude", lon: str = "longitude", *,
                     uid: str = "uid", min_devices: int = 50) -> pl.DataFrame:
    """Exact coordinates shared by at least ``min_devices`` distinct devices.

    Works on pings, stays (``stop_latitude``...) or locations (``home_latitude``...).
    Returns ``lat, lon, devices`` sorted by ``devices``, descending.
    """
    return (df.lazy().filter(pl.col(lat).is_not_null() & pl.col(lon).is_not_null())
            .group_by(lat, lon).agg(pl.col(uid).n_unique().cast(pl.UInt32).alias("devices"))
            .filter(pl.col("devices") >= min_devices).sort("devices", descending=True).collect())


def drop_sinks(df: pl.DataFrame | pl.LazyFrame, sinks: pl.DataFrame, lat: str = "latitude",
               lon: str = "longitude") -> pl.LazyFrame:
    """Remove rows located exactly at a sink coordinate (an anti-join on ``lat, lon``)."""
    s = sinks.lazy().select(pl.col(sinks.columns[0]).alias(lat), pl.col(sinks.columns[1]).alias(lon))
    return df.lazy().join(s, on=[lat, lon], how="anti")


def find_sinks(pings: PingSource, min_devices: int = 50, *, column_map: dict[str, str] | None = None) -> pl.DataFrame:
    """Coordinate sinks in a (possibly huge) ping source: ``latitude, longitude, devices``.

    Two streaming passes: count pings per exact coordinate (cheap), then count distinct
    devices only for coordinates with at least ``min_devices`` pings. Save the result and
    pass it as ``PipelineConfig(remove_sinks="sinks.parquet")`` (or ``detect_stops(...,
    remove_sinks=...)``) to drop those pings before stays and trajectories are computed;
    ``remove_sinks=True`` calls this function for you.
    """
    return _find_sinks(read_pings(pings, column_map), min_devices)


def _find_sinks(lf: pl.LazyFrame, min_devices: int) -> pl.DataFrame:
    """:func:`find_sinks` on pings already in the standard columns."""
    cand = (lf.group_by("latitude", "longitude").agg(pl.len().alias("_n"))
            .filter(pl.col("_n") >= min_devices).select("latitude", "longitude").collect(engine="streaming"))
    if cand.height == 0:
        return pl.DataFrame(schema={"latitude": pl.Float64, "longitude": pl.Float64, "devices": pl.UInt32})
    return (lf.join(cand.lazy(), on=["latitude", "longitude"], how="semi")
            .group_by("latitude", "longitude").agg(pl.col("uid").n_unique().cast(pl.UInt32).alias("devices"))
            .filter(pl.col("devices") >= min_devices).sort("devices", descending=True).collect(engine="streaming"))


def sink_table(option: SinkOption, pings: pl.LazyFrame, min_devices: int = 5000) -> pl.DataFrame | None:
    """Resolve a ``remove_sinks`` option to a ``latitude, longitude[, devices]`` table (``None``: keep all).

    ``pings`` must already be in the standard columns; it is only scanned when ``option is True``.
    """
    if option is None or option is False:
        return None
    if option is True:
        sinks = _find_sinks(pings, min_devices)
        if sinks.height == 0:
            warnings.warn(f"remove_sinks=True found no coordinate shared by >= {min_devices:,} devices; "
                          "no ping was removed. Lower sinks_min_devices for small datasets.", stacklevel=3)
        return sinks
    if isinstance(option, pl.DataFrame):
        return option
    if isinstance(option, (str, Path)):
        return pl.read_parquet(option)
    raise TypeError(f"remove_sinks must be a bool, a parquet path or a DataFrame, not {type(option).__name__}")


def remove_sink_pings(pings: pl.LazyFrame, sinks: pl.DataFrame | None) -> pl.LazyFrame:
    """Drop the pings located exactly at a sink (no-op for ``None``)."""
    if sinks is None:
        return pings
    s = sinks.lazy().select(pl.col("latitude").cast(pl.Float64), pl.col("longitude").cast(pl.Float64))
    return pings.join(s, on=["latitude", "longitude"], how="anti")
