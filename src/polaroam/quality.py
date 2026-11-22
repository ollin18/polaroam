"""Data-quality helpers.

**Coordinate sinks**: exact coordinates shared by many unrelated devices. These are
default geocodes (e.g. a city centroid used when only the IP or the country is known),
cell towers, or Wi-Fi positioning points. Real homes do not put thousands of devices on
the same 1e-6 degree point. In one month of Mexico City, 2,592 devices had their "home"
at exactly 19.4326, -99.1332 (the Zócalo, the city's textbook centroid).
"""

from __future__ import annotations

import polars as pl


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
