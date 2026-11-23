"""Reading pings into the canonical schema and device sharding."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import polars as pl

#: Canonical ping schema used by every stage. These four columns are required.
PING_SCHEMA = {"uid": pl.String, "latitude": pl.Float64, "longitude": pl.Float64, "timestamp": pl.Int64}

#: Common vendor names, tried in order (ignoring case) when ``column_map`` does not say otherwise.
ALIASES = {
    "uid": ["uid", "device_id", "caid", "maid", "ad_id", "advertiser_id", "advertising_id", "aid", "user_id",
            "userid", "device", "id"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon", "lng", "long"],
    "timestamp": ["timestamp", "utc_timestamp", "datetime", "ts", "time", "epoch", "unix_time", "unixtime",
                  "location_at", "event_time"],
}
_ALIASES = ALIASES  # backwards compatibility

#: Values per second for each Unix time unit, and the magnitude bounds used by ``timestamp_unit="auto"``:
#: below 1e11 is seconds, below 1e14 milliseconds, below 1e17 microseconds, otherwise nanoseconds
#: (unambiguous for any date after March 1973).
_PER_SECOND = {"s": 1, "ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}

ColumnMap = dict[str, "str | list[str]"]

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


def _find(names: list[str], candidates: list[str]) -> str | None:
    """First candidate present in ``names``: exact match first, then ignoring case."""
    for c in candidates:
        if c in names:
            return c
    lower: dict[str, str] = {}
    for n in names:
        lower.setdefault(n.lower(), n)
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def resolve_columns(names: list[str], column_map: ColumnMap | None = None) -> dict[str, str]:
    """Which input column holds each required field: ``{"uid": ..., "latitude": ..., ...}``.

    ``column_map`` values are a column name or a list of candidate names; names match
    exactly or ignoring case. Fields not in ``column_map`` are looked up among
    :data:`ALIASES`. A mapped name that is missing is an error (unless the input already
    has the standard name), so a typo never silently picks another column.
    """
    column_map = dict(column_map or {})
    unknown = set(column_map) - set(PING_SCHEMA)
    if unknown:
        raise KeyError(f"column_map keys must be among {list(PING_SCHEMA)}; got {sorted(unknown)}.")
    out: dict[str, str] = {}
    for std, aliases in ALIASES.items():
        if std in column_map:
            wanted = column_map[std]
            cands = [wanted] if isinstance(wanted, str) else list(wanted)
            src = _find(names, cands)
            if src is None and std in names:  # already in the standard schema (e.g. read twice)
                src = std
            if src is None:
                raise KeyError(f"column_map[{std!r}] = {wanted!r}: no such column. Available columns: {names}.")
        else:
            src = _find(names, aliases)
            if src is None:
                raise KeyError(f"Cannot find a column for {std!r} (tried {aliases}, ignoring case). Available "
                               f"columns: {names}. Pass column_map={{{std!r}: '<your column>'}}.")
        out[std] = src
    taken: dict[str, str] = {}
    for std, src in out.items():
        if src in taken:
            raise KeyError(f"Column {src!r} matched both {taken[src]!r} and {std!r}; set column_map explicitly.")
        taken[src] = std
    return out


def _to_float(c: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    if dtype == pl.String:  # " 19,4326 " -> 19.4326
        c = c.str.strip_chars().str.replace(",", ".", literal=True)
    return c.cast(pl.Float64)


def _unix_ms(c: pl.Expr, unit: str) -> pl.Expr:
    """Numeric Unix time in ``unit`` (or detected per value) -> Unix milliseconds (Int64)."""
    c = c.cast(pl.Float64)
    if unit == "auto":
        a = c.abs()
        scale = (pl.when(a < 1e11).then(1_000.0).when(a < 1e14).then(1.0)
                 .when(a < 1e17).then(1e-3).otherwise(1e-6))
    else:
        scale = pl.lit(1_000 / _PER_SECOND[unit])
    return (c * scale).round(0).cast(pl.Int64)


def _timestamp(lf: pl.LazyFrame, src: str, dtype: pl.DataType, unit: str, tz: str) -> pl.Expr:
    c = pl.col(src)
    if isinstance(dtype, pl.Datetime):
        if dtype.time_zone is None and tz != "UTC":
            c = c.dt.replace_time_zone(tz, ambiguous="earliest", non_existent="null")
        return c.dt.epoch("ms")
    if dtype == pl.Date:
        raise TypeError(f"Column {src!r} is a Date (no time of day); pings need a time.")
    if dtype.is_numeric():
        return _unix_ms(c, unit)
    if dtype in (pl.String, pl.Categorical, pl.Enum):
        c = c.cast(pl.String).str.strip_chars()
        # numbers stored as text, or date-time text? decide on a sample, then parse strictly
        sample = lf.select(c.alias("_t")).drop_nulls().head(1000).collect().get_column("_t")
        if sample.len() == 0 or sample.cast(pl.Float64, strict=False).null_count() == 0:
            return _unix_ms(c.cast(pl.Float64), unit)
        try:
            parsed = sample.str.to_datetime(strict=True)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Cannot parse timestamps in {src!r} (e.g. {sample[0]!r}): {e}. Convert them to "
                             "Unix time or a Datetime column first.") from None
        # text with an offset ("Z", "-06:00") is converted to UTC; text without one is read in ``tz``
        zone = "UTC" if parsed.dtype.time_zone is not None else tz
        return c.str.to_datetime(strict=True, time_unit="us", time_zone=zone, ambiguous="earliest").dt.epoch("ms")
    raise TypeError(f"Column {src!r} has type {dtype}; timestamps must be numbers, Datetime or text.")


def normalize_pings(lf: pl.LazyFrame, column_map: ColumnMap | None = None, *, timestamp_unit: str = "auto",
                    timestamp_timezone: str = "UTC") -> pl.LazyFrame:
    """Select, rename and cast to ``uid, latitude, longitude, timestamp`` (Unix ms).

    - Columns: ``column_map`` (a name or a list of candidate names per field), else
      :data:`ALIASES`; names match ignoring case. See :func:`resolve_columns`.
    - ``uid``: any type, as text (binary ids as hex).
    - ``latitude`` / ``longitude``: any numeric type, or text (``"19,43"`` accepted).
    - ``timestamp``: a Datetime, Unix time as numbers or text, or date-time text such as
      ``"2022-12-01 08:00:00"`` / ISO 8601. ``timestamp_unit`` is ``"s"``, ``"ms"``,
      ``"us"``, ``"ns"`` or ``"auto"`` (detected per value from its magnitude). Datetimes and
      text without a timezone are read in ``timestamp_timezone`` (default UTC).

    Values that cannot be converted raise an error instead of being dropped; rows with a
    missing (null) required field are dropped.
    """
    if timestamp_unit not in ("auto", *_PER_SECOND):
        raise ValueError(f"timestamp_unit must be 'auto', 's', 'ms', 'us' or 'ns'; got {timestamp_unit!r}.")
    schema = lf.collect_schema()
    cols = resolve_columns(schema.names(), column_map)
    uid, lat, lon, ts = (cols[k] for k in PING_SCHEMA)
    u = pl.col(uid)
    u = u.bin.encode("hex") if schema[uid] == pl.Binary else u.cast(pl.String)
    return lf.select(
        u.alias("uid"),
        _to_float(pl.col(lat), schema[lat]).alias("latitude"),
        _to_float(pl.col(lon), schema[lon]).alias("longitude"),
        _timestamp(lf, ts, schema[ts], timestamp_unit, timestamp_timezone).alias("timestamp"),
    ).drop_nulls()


def read_pings(source: PingSource, column_map: ColumnMap | None = None, fmt: str = "auto",
               csv_separator: str = ",", has_header: bool = True, *, timestamp_unit: str = "auto",
               timestamp_timezone: str = "UTC") -> pl.LazyFrame:
    """Scan ``source`` (files, globs, directories or a DataFrame) and return a LazyFrame with the
    canonical ping schema; see :func:`normalize_pings` for column names and types."""
    return normalize_pings(scan(source, fmt, csv_separator, has_header), column_map,
                           timestamp_unit=timestamp_unit, timestamp_timezone=timestamp_timezone)


def shard_expr(n_shards: int, col: str = "uid") -> pl.Expr:
    """Stable (polars-version independent) device shard in ``[0, n_shards)``."""
    h = pl.col(col).str.encode("hex").str.slice(-14).str.to_integer(base=16).cast(pl.UInt64)
    return ((h * pl.lit(0x9E3779B97F4A7C15, pl.UInt64)) // (1 << 40) % n_shards).cast(pl.Int32)
