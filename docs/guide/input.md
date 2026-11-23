# Input data

Every stage works on four required columns. polaroam finds them in your data and converts them
to one standard schema, whatever the vendor calls them or stores them as:

| Field | Standard type | Accepted input |
|---|---|---|
| `uid` | text | any type: text, integers, categoricals; binary ids become hex |
| `latitude`, `longitude` | Float64 | any number; text (`"19.4326"`, also `"19,4326"`) |
| `timestamp` | Unix ms (Int64) | Unix time in s, ms, µs or ns (numbers or text); Datetime (any unit or timezone); date-time text (`"2022-12-01 08:00:00"`, ISO 8601 with `Z` or an offset) |

Other columns are ignored. Rows with a missing (null) required value are dropped; values that
cannot be converted raise an error instead of disappearing silently.

```python
pings = pr.read_pings("raw/MX/*.parquet")                # parquet or csv; files, globs, directories
pings = pr.read_pings(df)                                # or a (Lazy)DataFrame
pings.collect_schema()   # uid: String, latitude: Float64, longitude: Float64, timestamp: Int64
```

Every function that takes pings (`Pipeline.run`, `detect_stops`, `extract_trajectories`,
`find_sinks`) reads them the same way.

!!! tip "Taking a subset of devices from a large dataset"
    Filter the raw id column, in its own type, *before* `read_pings`. Filtering after it means
    converting every id to text first: on 1.66B pings, taking 60,000 devices that way peaked at
    about 100 GB of RAM, against 8 GB (and 5 s) when filtering the raw column, which the parquet
    reader does as it scans.

    ```python
    raw = pl.scan_parquet("raw/*.parquet").filter(pl.col("device_id").is_in(ids.implode()))
    pings = pr.read_pings(raw, {"uid": "device_id"})
    ```

## Column names

Without instructions, polaroam looks for common names, **ignoring case**:

| Field | Names tried, in order |
|---|---|
| `uid` | uid, device_id, caid, maid, ad_id, advertiser_id, advertising_id, aid, user_id, userid, device, id |
| `latitude` | latitude, lat |
| `longitude` | longitude, lon, lng, long |
| `timestamp` | timestamp, utc_timestamp, datetime, ts, time, epoch, unix_time, unixtime, location_at, event_time |

So `LAT`, `Lng`, `Device_ID` or `Timestamp` work as they are. For anything else, say which
column holds each field with `column_map`, either one name or a list of candidates (useful when
the same code reads several vendors):

=== "Python"

    ```python
    cfg = pr.PipelineConfig(
        timezone="America/Mexico_City",
        column_map={"uid": "hashed_id", "latitude": ["y", "lat_deg"], "longitude": ["x", "lon_deg"]},
    )
    stops = pr.detect_stops("raw/*.parquet", column_map={"uid": "hashed_id"})
    ```

=== "Command line"

    ```bash
    polaroam run "raw/*.parquet" out --timezone America/Mexico_City \
        --col uid=hashed_id --col latitude=y,lat_deg --col longitude=x,lon_deg
    ```

A name in `column_map` that does not exist is an **error** listing the available columns. It
never falls back to another column, so a typo such as `{"uid": "devce_id"}` cannot silently pick
a column called `id`. One column matching two fields is an error too.

## Timestamps

| Input | Read as |
|---|---|
| Numbers or numeric text | Unix time (always UTC). The unit is detected from each value: below 10¹¹ seconds, below 10¹⁴ milliseconds, below 10¹⁷ microseconds, otherwise nanoseconds (unambiguous after March 1973). Set `timestamp_unit` to force one. |
| Datetime with a timezone | that instant |
| Text with an offset (`2022-12-01T08:00:00Z`, `...-06:00`) | that instant |
| Datetime or text without a timezone | local time in `timestamp_timezone` (default UTC) |

```python
# a vendor exporting local wall-clock times in Mexico City, and another with Unix microseconds
cfg = pr.PipelineConfig(timezone="America/Mexico_City", timestamp_timezone="America/Mexico_City")
cfg = pr.PipelineConfig(timezone="America/Mexico_City", timestamp_unit="us")
```

```bash
polaroam run raw out --timezone America/Mexico_City --timestamp-timezone America/Mexico_City
polaroam run raw out --timezone America/Mexico_City --timestamp-unit us
```

`timestamp_timezone` says how to *read* times without a timezone; `timezone` is where the stays
*are*, for local hours and weekdays ([local time](local-time.md)). They are often the same.

!!! warning "Unix times that are really local time"
    Some exports write the *local* wall-clock time as if it were a Unix time. polaroam reads Unix
    times as UTC, so every local time (home and work hours, weekdays, start hours) would be shifted
    by the UTC offset; stays, trajectories and speeds are not affected. Check one row against a
    column you trust: in the Mexico City quadrant data, `timestamp = 1670275719` reads as
    21:28:39, which is the local `date`, while `utc_date` is 03:28:39 the next day. Use the UTC
    column instead:

    ```python
    cfg = pr.PipelineConfig(timezone="America/Mexico_City", column_map={"timestamp": "utc_date"})
    ```

    A quick symptom: pings per local hour that peak around noon and stay high all night.

!!! tip "Seconds before 1973"
    Automatic detection reads a value below 10¹¹ as seconds. Data starting at the epoch
    (`0`, `5`, `6`... milliseconds, as in synthetic tests) needs `timestamp_unit="ms"`.
