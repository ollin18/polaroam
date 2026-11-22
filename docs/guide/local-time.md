# Local time

Home and work are defined by local hours and weekdays, so every stay gets a local time
(gc-lbs `local_time.sh`).

=== "Fixed timezone"

    ```python
    local = pr.add_local_time(clusters, timezone="America/Mexico_City")
    ```

    Every stay uses one timezone. No extra dependency is needed.

=== "Timezone polygons"

    ```python
    local = pr.add_local_time(clusters, timezone="America/Mexico_City",
                              tz_polygons="tz_codes.csv")
    ```

    Each stay takes the timezone of the polygon that contains it. `timezone` is the fallback
    for stays outside every polygon. Needs the `tz` extra (shapely).

The polygon lookup runs on the **unique** stay coordinates only. It uses a shapely STRtree of
the points queried with the *prepared* timezone polygons, so 280k points against the whole
world file take under a second. A point on the boundary of overlapping polygons takes the first
polygon: stays are never duplicated (the SQL `LEFT JOIN` could duplicate them).

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
