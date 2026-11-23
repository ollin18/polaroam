# Quick start

This page goes from a file of pings to stays, trips with speeds and modes, places and home /
work, first step by step and then with the pipeline.

## 1. Pings

Any parquet or CSV file, glob, directory or Polars (Lazy)DataFrame with a device id, latitude,
longitude and time works. Common column names are found automatically; map others with
`column_map`.

```python
import polaroam as pr

pings = pr.read_pings("data/veraset_movement_416.snappy.parquet")   # LazyFrame
pings.head(3).collect()
```

```text
┌────────────────────────────┬────────────┬────────────┬───────────────┐
│ uid                        ┆ latitude   ┆ longitude  ┆ timestamp     │
│ str                        ┆ f64        ┆ f64        ┆ i64           │
╞════════════════════════════╪════════════╪════════════╪═══════════════╡
│ 9049025233cdb7e88910f9b109 ┆ -23.465675 ┆ -46.423027 ┆ 1577836872000 │
│ …                          ┆ …          ┆ …          ┆ …             │
```

Timestamps become Unix milliseconds. The source can be a Datetime, Unix seconds or Unix
milliseconds (values below 1e11 are read as seconds, as in gc-lbs).

## 2. Stays and trajectories

```python
stops, trips = pr.detect_stops(pings, trajectories=True,
                               trajectory_params=pr.TrajectoryParams(min_pings=2, min_length_m=100))
```

`stops` has one row per stay: `uid, stop_event, start_timestamp, end_timestamp, stop_duration,
stop_latitude, stop_longitude, n_pings`. `trips` has one row per trajectory, with the
simplified line, per-vertex timestamps and [speed features](../guide/speed-and-mode.md):

```python
trips.select("uid", "trajectory_id", "length_m", "duration", "speed_p85_kmh", "speed_cv", "mode_guess")
```

## 3. Places, local time, home and work

```python
places = pr.cluster_stops(stops)                                  # DBSCAN, 20 m, min 2
local = pr.add_local_time(places, timezone="America/Sao_Paulo")   # required: see Local time
hw = pr.classify_home_work(local, home_min_days=7, work_min_days=4)
homes = pr.home_work_locations(local)                             # one row per device
```

## 4. The same with the pipeline

`Pipeline` chains every step, writes parquet tables and keeps the state for incremental runs:

```python
cfg = pr.PipelineConfig(
    timezone="America/Sao_Paulo",
    trajectories=pr.TrajectoryParams(min_pings=2, min_length_m=100),
)
pipe = pr.Pipeline("out/example", cfg, trajectories=True)
pipe.run("data/veraset_movement_416.snappy.parquet")

import polars as pl

pipe.locations().collect()          # LazyFrames over the output tables
pipe.trajectories_table().filter(pl.col("mode_guess") == "bike").collect()
```

!!! note "Reading the tables"
    `pipe.stops()`, `pipe.trajectories_table()`, `pipe.clusters()`, `pipe.home_work()` and
    `pipe.locations()` return Polars **LazyFrames**, so filters and selections are pushed down
    into the parquet scan.

## 5. Command line

```bash
polaroam run data/veraset_movement_416.snappy.parquet out/example \
    --timezone America/Sao_Paulo --trajectories --min-pings 2 --min-length 100
```

## Next steps

- [The pipeline](../guide/pipeline.md): one-shot vs incremental, output layout, retries.
- [Speed and transport mode](../guide/speed-and-mode.md): what the speed features mean.
- [Map matching](../guide/map-matching.md): trajectories on the streets.
- [Routines and trip completion](../guide/routines.md): repeated trips, their mode, and partial trips completed.
- [Example data, end to end](../examples/example-data.md): every figure, with the code.
