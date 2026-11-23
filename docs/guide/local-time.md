# Local time

Home and work are defined by local hours and weekdays, so every stay gets a local time
(gc-lbs `local_time.sh`).

=== "Fixed timezone"

    ```python
    local = pr.add_local_time(clusters, timezone="America/Mexico_City")
    ```

    Every stay uses one timezone. No extra dependency is needed.

=== "Timezone polygons (bundled)"

    ```python
    local = pr.add_local_time(clusters, timezone="UTC", tz_polygons="builtin")
    ```

    Each stay takes the timezone of the polygon that contains it. `timezone` is the fallback
    for stays outside every polygon (at sea). Needs the `tz` extra (shapely).

=== "Your own polygons"

    ```python
    local = pr.add_local_time(clusters, timezone="UTC", tz_polygons="my_timezones.csv")
    ```

    A CSV with a `tz` column and a WKT `geometry` column (optionally `country_code`).

## The bundled polygons

polaroam ships the world timezone polygons of the gc-lbs pipeline (`tz_codes.csv`: 418
polygons, 1.0M vertices), so `tz_polygons="builtin"` needs no external file. They are stored
in 5.2 MB instead of 21.7 MB, without losing precision: every coordinate has 6 decimals, so they
are kept as integer microdegrees and decode to exactly the same floats as the CSV.

```python
tz = pr.builtin_timezones()              # tz, country_code, geometry (WKB)
tz.filter(pl.col("country_code") == "BR")["tz"].to_list()
# ['America/Araguaina', 'America/Bahia', 'America/Belem', ...]
```

`scripts/build_timezones.py` rebuilds the bundled file from a CSV.

## How the lookup works

Large countries span several timezones (Mexico 4 offsets, Brazil 4), and a year of stays
has hundreds of millions of coordinates. Testing each one against complex polygons would be
slow, and almost all of them are nowhere near a border. So:

1. Stay coordinates go on a 0.1° grid.
2. A grid cell inside a single timezone polygon, touching no other, gives its timezone with a
   plain Polars join.
3. Only the points in cells crossing a border (1–5% of them) are tested against the polygons,
   with a shapely STRtree and prepared polygons.

The result is **exactly** the same as testing every point: the tests check it point by point,
including points on cell edges and on polygon vertices. With 10M points spread over a whole
country:

| 10M points | Every point tested | Grid | Grid, cells already classified |
|---|---|---|---|
| Mexico | 8.0 s | 1.4 s | 1.0 s |
| Brazil | 7.8 s | 1.3 s | 0.8 s |

Stays cluster in cities, so real data has fewer cells and gains more. A point on the boundary of
overlapping polygons takes the first polygon: stays are never duplicated (the SQL `LEFT JOIN`
could duplicate them).

**In the pipeline, each coordinate is looked up once.** A stay's coordinate never changes, so
each shard keeps a coordinate → timezone table in its state. Daily runs look up only that day's
new stays, instead of every stay of every earlier day.

## Added columns

| Column | Type | Description |
|---|---|---|
| `timezone` | str | polygon timezone (null if none, or the fixed one) |
| `inputed_timezone` | str | timezone used |
| `stop_datetime`, `end_stop_datetime` | datetime | local start / end (naive) |
| `date`, `end_date` | date | local start / end date |
| `year`, `month`, `day`, `hour` | i32 | local start parts |
| `weekday` | i32 | 1 = Sunday … 7 = Saturday (BigQuery `DAYOFWEEK`) |
| `weekend` | bool | Saturday or Sunday |
