"""Geodesic helpers written as polars expressions (GPU-engine friendly)."""

from __future__ import annotations

import math

import polars as pl

#: Mean Earth radius (m), the one used by BigQuery ``ST_DISTANCE``.
EARTH_RADIUS_M = 6371008.8
DEG = math.pi / 180.0
#: Meters per degree of latitude on the sphere.
M_PER_DEG = EARTH_RADIUS_M * DEG


def _e(x: str | pl.Expr) -> pl.Expr:
    return pl.col(x) if isinstance(x, str) else x


def haversine(lat1: str | pl.Expr, lon1: str | pl.Expr, lat2: str | pl.Expr, lon2: str | pl.Expr) -> pl.Expr:
    """Great-circle distance in meters between two (lat, lon) columns in degrees."""
    p1, p2 = _e(lat1) * DEG, _e(lat2) * DEG
    dphi = p2 - p1
    dlmb = (_e(lon2) - _e(lon1)) * DEG
    a = (dphi / 2).sin() ** 2 + p1.cos() * p2.cos() * (dlmb / 2).sin() ** 2
    return 2 * EARTH_RADIUS_M * a.sqrt().clip(upper_bound=1.0).arcsin()


def with_group_row(frame, over: str | list[str], name: str, start: int = 0):
    """Row number within ``over`` groups (``start``-based) of a frame sorted by a unique key.

    Same as ``pl.int_range(pl.len()).over(over)`` but built from ``with_row_index`` +
    ordinal ``rank``, which the polars GPU engine supports.
    """
    return (frame.with_row_index("_ix")
            .with_columns((pl.col("_ix").rank("ordinal").over(over).cast(pl.Int64) - 1 + start).alias(name))
            .drop("_ix"))


def exact_mean(col: str, scale: float = 1e12) -> pl.Expr:
    """Group mean that is identical on CPU and GPU, whatever the summation order.

    Deviations from the group minimum are summed as fixed-point integers (1e-12 units),
    and integer addition does not depend on order. Identical values give exactly that
    value back. Use inside ``group_by(...).agg``.
    """
    x0 = pl.col(col).min()
    dev = ((pl.col(col) - x0) * scale).round().cast(pl.Int64).sum()
    return x0 + dev / (pl.len() * scale)


def tie_exact_mean(col: str, scale: float = 1e9) -> pl.Expr:
    """Group mean at ``1/scale`` resolution for *ranking*: exactly tied means give bit-equal
    results on any engine and in any summation order.

    Values are summed as integers (1e-9 degree units; the sums stay below 2**53, so they are
    exact as floats), then divided once. Two groups with the same exact mean therefore get
    the same key, and the sort falls through to the next key, as SQL / pandas do.
    """
    return (pl.col(col) * scale).round().cast(pl.Int64).sum() / pl.len() / scale
