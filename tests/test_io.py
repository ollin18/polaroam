"""Reading pings: column names (aliases, case, candidates) and type conversion."""

from datetime import datetime

import polars as pl
import pytest

import polaroam as pr
from polaroam.cli import _column_map

T = 1_669_881_600  # 2022-12-01 08:00:00 UTC
MS = [T * 1000, T * 1000 + 1500]  # the expected timestamps (Unix ms)


def frame(**cols) -> pl.DataFrame:
    base = {"uid": ["a", "b"], "latitude": [19.43, 19.44], "longitude": [-99.13, -99.14], "timestamp": MS}
    return pl.DataFrame({**base, **cols})


def read(df, **kw) -> pl.DataFrame:
    return pr.read_pings(df, **kw).collect()


def test_canonical_and_idempotent():
    df = frame()
    out = read(df)
    assert out.schema == pl.Schema(pr.io.PING_SCHEMA) and out.equals(df)
    assert read(out).equals(out)  # normalizing twice changes nothing


def test_aliases_ignore_case():
    df = pl.DataFrame({"Device_ID": ["a", "b"], "LAT": [19.43, 19.44], "Lng": [-99.13, -99.14], "Time": MS})
    assert read(df).equals(frame())


def test_column_map_names_and_candidates():
    df = pl.DataFrame({"who": ["a", "b"], "Y": [19.43, 19.44], "x": [-99.13, -99.14], "when": MS})
    cm = {"uid": "who", "latitude": ["lat", "y"], "longitude": ["lon", "X"], "timestamp": "WHEN"}
    assert read(df, column_map=cm).equals(frame())


def test_mapped_typo_is_an_error_not_a_fallback():
    df = frame().rename({"uid": "device_id"}).with_columns(pl.lit("x").alias("id"))
    with pytest.raises(KeyError, match="devce_id.*Available columns"):
        read(df, column_map={"uid": "devce_id"})
    with pytest.raises(KeyError, match="column_map keys"):
        read(df, column_map={"lat": "latitude"})


def test_missing_and_ambiguous_columns():
    with pytest.raises(KeyError, match="Cannot find a column for 'latitude'"):
        read(frame().drop("latitude"))
    with pytest.raises(KeyError, match="matched both"):
        read(frame(), column_map={"uid": "latitude"})


@pytest.mark.parametrize("ts", [
    pl.Series([float(T), T + 1.5]),                    # seconds, float
    [str(T), str(T + 1.5)],                            # seconds as text
    MS,                                                # milliseconds
    [MS[0] * 1000, MS[1] * 1000],                      # microseconds
    [MS[0] * 1_000_000, MS[1] * 1_000_000],            # nanoseconds
    ["2022-12-01 08:00:00", "2022-12-01 08:00:01.500"],
    ["2022-12-01T08:00:00Z", "2022-12-01T08:00:01.500Z"],
    ["2022-12-01T02:00:00-06:00", "2022-12-01T02:00:01.500-06:00"],
    pl.Series([datetime(2022, 12, 1, 8), datetime(2022, 12, 1, 8, 0, 1, 500_000)]),
    pl.Series([datetime(2022, 12, 1, 8), datetime(2022, 12, 1, 8, 0, 1, 500_000)]).dt.replace_time_zone("UTC")
      .dt.convert_time_zone("Asia/Tokyo"),
    pl.Series([datetime(2022, 12, 1, 8), datetime(2022, 12, 1, 8, 0, 1, 500_000)]).cast(pl.Datetime("ns")),
])
def test_timestamp_formats(ts):
    assert read(frame(timestamp=ts))["timestamp"].to_list() == MS


def test_timestamp_unit_and_timezone():
    # explicit unit: values too small for "auto" to call milliseconds
    assert read(frame(timestamp=[5, 6]), timestamp_unit="ms")["timestamp"].to_list() == [5, 6]
    assert read(frame(timestamp=[5, 6]))["timestamp"].to_list() == [5000, 6000]
    # naive local date-times
    local = ["2022-12-01 02:00:00", "2022-12-01 02:00:01.500"]
    assert read(frame(timestamp=local), timestamp_timezone="America/Mexico_City")["timestamp"].to_list() == MS
    naive = pl.Series([datetime(2022, 12, 1, 2), datetime(2022, 12, 1, 2, 0, 1, 500_000)])
    assert read(frame(timestamp=naive), timestamp_timezone="America/Mexico_City")["timestamp"].to_list() == MS
    # an explicit offset wins over timestamp_timezone
    aware = ["2022-12-01T08:00:00Z", "2022-12-01T08:00:01.500Z"]
    assert read(frame(timestamp=aware), timestamp_timezone="America/Mexico_City")["timestamp"].to_list() == MS
    with pytest.raises(ValueError, match="timestamp_unit"):
        read(frame(), timestamp_unit="minutes")


def test_bad_values_raise_instead_of_being_dropped():
    with pytest.raises(ValueError, match="Cannot parse timestamps"):
        read(frame(timestamp=["yesterday", "today"]))
    with pytest.raises(pl.exceptions.InvalidOperationError):
        read(frame(latitude=["19.43", "north"]))
    with pytest.raises(TypeError, match="Date"):
        read(frame(timestamp=pl.Series([datetime(2022, 12, 1)] * 2).dt.date()))


def test_coordinate_and_id_types():
    out = read(frame(latitude=[" 19,43", "19.44"], longitude=pl.Series([-99.13, -99.14], dtype=pl.Float32)))
    assert out["latitude"].to_list() == [19.43, 19.44]
    assert out["longitude"].to_list() == pytest.approx([-99.13, -99.14], abs=1e-5)
    assert read(frame(uid=[1, 2]))["uid"].to_list() == ["1", "2"]
    assert read(frame(uid=[b"\x01\xff", b"\x02"]))["uid"].to_list() == ["01ff", "02"]
    assert read(frame(uid=pl.Series(["a", "b"], dtype=pl.Categorical)))["uid"].to_list() == ["a", "b"]
    # rows with a missing required value are dropped
    assert read(frame(latitude=[None, 19.44])).height == 1


def test_pipeline_reads_vendor_schema(tmp_path, pings):
    """Vendor column names and microsecond datetimes in another timezone give the same stays."""
    vendor = pings.select(
        pl.col("uid").alias("Device_ID"), pl.col("latitude").cast(pl.String).alias("LAT"),
        pl.col("longitude").alias("Lng"),
        pl.from_epoch("timestamp", "ms").dt.replace_time_zone("UTC").dt.convert_time_zone("America/Bogota")
          .dt.cast_time_unit("us").alias("event_time"))
    vendor.write_parquet(tmp_path / "vendor.parquet")
    cfg = pr.PipelineConfig(timezone="America/Mexico_City")
    ref = pr.Pipeline(tmp_path / "ref", cfg, verbose=False)
    ref.run(pings)
    got = pr.Pipeline(tmp_path / "got", cfg, verbose=False)
    got.run(tmp_path / "vendor.parquet")
    assert got.stops().collect().equals(ref.stops().collect())


def test_cli_column_map():
    assert _column_map(["uid=device_id", "latitude=lat, y"]) == {"uid": "device_id", "latitude": ["lat", "y"]}
    assert _column_map(None) is None
    with pytest.raises(ValueError, match="timestamp_unit"):
        pr.PipelineConfig(timezone="UTC", timestamp_unit="minutes")
