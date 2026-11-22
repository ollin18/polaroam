"""Reading pings into the canonical schema and device sharding."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import polars as pl

#: Canonical ping schema used by every stage.
PING_SCHEMA = {"uid": pl.String, "latitude": pl.Float64, "longitude": pl.Float64, "timestamp": pl.Int64}

#: Common vendor names, tried in order when ``column_map`` does not say otherwise.
_ALIASES = {
    "uid": ["uid", "device_id", "caid", "ad_id", "aid", "user_id", "id"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon", "lng"],
    "timestamp": ["timestamp", "datetime", "ts", "time", "utc_timestamp", "epoch"],
}

PingSource = str | Path | pl.LazyFrame | pl.DataFrame | Iterable[str | Path]


def scan(source: PingSource, fmt: str = "auto", csv_separator: str = ",", has_header: bool = True) -> pl.LazyFrame:
    """Lazily scan parquet / csv files (globs and directories allowed)."""
    if isinstance(source, pl.DataFrame):
        return source.lazy()
    if isinstance(source, pl.LazyFrame):
        return source
    paths = [source] if isinstance(source, (str, Path)) else list(source)
    paths = [str(_expand_dir(Path(p))) for p in paths]
    if fmt == "auto":
        fmt = "csv" if any(".csv" in p for p in paths) else "parquet"
    if fmt == "parquet":
        return pl.scan_parquet(paths, hive_partitioning=False)
    return pl.concat(
        [pl.scan_csv(p, separator=csv_separator, has_header=has_header, infer_schema_length=10000) for p in paths]
    )


def _expand_dir(p: Path) -> Path:
    if p.is_dir():
        if any(p.rglob("*.parquet")):
            return p / "**" / "*.parquet"
        if any(p.rglob("*.csv*")):
            return p / "**" / "*.csv*"
    return p


def normalize_pings(lf: pl.LazyFrame, column_map: dict[str, str] | None = None) -> pl.LazyFrame:
    """Select / rename / cast to ``uid, latitude, longitude, timestamp`` (Unix ms).

    ``timestamp`` may be a Datetime, Unix seconds or Unix milliseconds
    (values below 1e11 are treated as seconds, as in the BigQuery pipeline).
    """
    names = lf.collect_schema()
    column_map = dict(column_map or {})
    exprs = []
    for std, aliases in _ALIASES.items():
        src = column_map.get(std)
        if src not in names:  # mapped column absent (e.g. already canonical): try known names
            src = next((a for a in aliases if a in names), None)
        if src is None:
            raise KeyError(f"Cannot find a column for '{std}'. Pass column_map={{'{std}': '<your column>'}}.")
        exprs.append((src, std, names[src]))

    out = []
    for src, std, dtype in exprs:
        c = pl.col(src)
        if std == "timestamp":
            if isinstance(dtype, pl.Datetime):
                c = c.dt.epoch("ms")
            elif dtype == pl.String:
                c = c.cast(pl.Float64)
            c = c.cast(pl.Int64)
            c = pl.when(c < 100_000_000_000).then(c * 1000).otherwise(c)
        elif std == "uid":
            c = c.cast(pl.String)
        else:
            c = c.cast(pl.Float64)
        out.append(c.alias(std))
    return lf.select(out).drop_nulls()


def read_pings(source: PingSource, column_map: dict[str, str] | None = None, fmt: str = "auto",
               csv_separator: str = ",", has_header: bool = True) -> pl.LazyFrame:
    """Scan ``source`` and return a LazyFrame with the canonical ping schema."""
    return normalize_pings(scan(source, fmt, csv_separator, has_header), column_map)


def shard_expr(n_shards: int, col: str = "uid") -> pl.Expr:
    """Stable (polars-version independent) device shard in ``[0, n_shards)``."""
    h = pl.col(col).str.encode("hex").str.slice(-14).str.to_integer(base=16).cast(pl.UInt64)
    return ((h * pl.lit(0x9E3779B97F4A7C15, pl.UInt64)) // (1 << 40) % n_shards).cast(pl.Int32)
