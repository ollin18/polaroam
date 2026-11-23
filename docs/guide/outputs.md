# Output tables

All tables are parquet, one file per shard (and per batch for stays and trajectories). See
[The pipeline](pipeline.md#output-layout) for the directory layout.

## `stops`

| Column | Type | Description |
|---|---|---|
| `uid` | str | device id |
| `stop_event` | i64 | stay number per device, 1, 2, … in time order |
| `start_timestamp`, `end_timestamp` | i64 | Unix seconds |
| `stop_duration` | i64 | seconds |
| `stop_latitude`, `stop_longitude` | f64 | lower median of the stay's pings |
| `n_pings` | u32 | pings in the stay |

## `clusters` (gc-lbs `*_clustered`)

`uid, stop_event, start_timestamp, end_timestamp, stop_duration, stop_latitude, stop_longitude`,
plus:

| Column | Type | Description |
|---|---|---|
| `cluster_label` | i32 | per-device cluster rank by centroid; -1 = noise |
| `cluster_counts` | i32 | stays in the cluster (1 for noise) |
| `cluster_latitude`, `cluster_longitude` | f64 | median of the member stays |

## `home_work` (gc-lbs `*_HW`, same 24 columns and types)

| Column | Type | Description |
|---|---|---|
| uid | str | device id |
| stop_event | i64 | stay number per device |
| start_timestamp / end_timestamp | i64 | Unix seconds |
| stop_duration | i64 | seconds |
| stop_latitude / stop_longitude | f64 | stay location |
| cluster_label | i32 | DBSCAN cluster (-1 = noise) |
| cluster_counts | i32 | stays in the cluster |
| cluster_latitude / cluster_longitude | f64 | cluster median location |
| timezone | str | polygon timezone (null if none) |
| inputed_timezone | str | timezone used |
| stop_datetime / end_stop_datetime | datetime | local start / end |
| date, year, month, day, hour | date / i32 | local start parts |
| weekday | i32 | 1 = Sunday … 7 = Saturday |
| weekend | bool | Saturday or Sunday |
| end_date | date | local end date |
| location_type | str | `H` home, `W` work, `O` other |

## `locations`

One row per device with a home: `uid, home_latitude, home_longitude, work_latitude,
work_longitude` (work is null without a work location).

## `trajectories`

| Column | Type | Description |
|---|---|---|
| `uid`, `trajectory_id` | str, i64 | device and trajectory number |
| `origin_stop_event`, `destination_stop_event` | i64 | connected stays |
| `start_timestamp`, `end_timestamp`, `duration` | i64 | Unix seconds; seconds |
| `n_pings`, `n_points` | u32 | moving pings; vertices kept |
| `length_m` | f64 | path length (m) |
| `speed_mean_kmh`, `speed_median_kmh`, `speed_p85_kmh`, `speed_max_kmh` | f64 | time-weighted speeds |
| `speed_cv`, `frac_slow` | f64 | variability; share of slow time |
| `straightness` | f64 | start-to-end distance / length |
| `ping_gap_median_s` | f64 | median time between pings |
| `n_jumps` | u32 | GPS jumps removed |
| `mode_guess` | str | `walk`, `bike`, `motorized`, `unknown` |
| `geometry` | str | WKT `LINESTRING` (or `LINESTRING M`) |
| `coordinates` | list[list[f64]] | `[[lon, lat], …]` |
| `timestamps` | list[i64] | Unix seconds per vertex |

## `matched` (map matching)

`match_trajectories` returns three tables (or writes them under `output_dir` as
`trajectories/`, `points/` and `edges/`): one matched line per trajectory, each point's matched
position, and the route's edges in order. Columns are in [Map matching](map-matching.md#outputs).

## Trips and routines

| Table | From | One row per |
|---|---|---|
| trips | `stay_trips`, then `find_routines` | move between two consecutive stays ≥ 200 m apart, with its `state`, `routine_id` and `mode` |
| `done.trips` | `complete_trips` | routine trip, completed from stay to stay: observed and imputed length, estimated times, geometry |
| `done.edges` | `complete_trips` | edge of a completed route, in order, flagged `imputed` |
| `done.routes` | `complete_trips` | edge of a routine's usual streets, with the trips observed on it |

Columns are in [Routines and trip completion](routines.md#completing-trips).

!!! tip "Home and work, gc-lbs style"
    The gc-lbs README extracts home and work per device with SQL. In PolaRoam, that is the
    `locations` table, or:

    ```python
    pr.home_work_locations(pipe.home_work())
    ```
