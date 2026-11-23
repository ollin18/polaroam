# Configuration

Every stage takes a parameter dataclass, or the same fields as keyword arguments:

```python
pr.detect_stops(pings, pr.StopParams(distance_threshold=30))
pr.detect_stops(pings, distance_threshold=30)                  # same
```

`PipelineConfig` groups them for the [pipeline](pipeline.md):

```python
cfg = pr.PipelineConfig(
    timezone="America/Mexico_City",          # required, unless find_timezones=True
    find_timezones=False,
    stops=pr.StopParams(distance_threshold=20, time_threshold=3600, min_stop_duration=300),
    dbscan=pr.ClusterParams(distance=20, min_points=2),
    home_work=pr.HomeWorkParams(home_min_days=7, work_min_days=4, home_hour_evening=22,
                                home_hour_morning=7, work_hour_start=8, work_hour_end=18),
    trajectories=pr.TrajectoryParams(min_pings=2, min_length_m=100),
    column_map={"uid": "device_id"},
    n_shards=24,
)
```

## gc-lbs config files

`PipelineConfig.from_json` reads the gc-lbs `pipeline_config.json` as is. With a label, it uses
that country's section; `stops.shards` becomes `n_shards`.

```python
cfg = pr.PipelineConfig.from_json("pipeline_config.json", "MX", column_map={"uid": "device_id"})
```

A flat polaroam file (the same keys without `countries`) works too, and it can also hold a
`trajectories` section.

## All parameters

| Stage | Parameter | Default | Meaning |
|---|---|---|---|
| Stays | `distance_threshold` | 20 m | max distance between consecutive pings of a stay; max distance to the median |
| | `time_threshold` | 3600 s | max gap between consecutive pings of a stay |
| | `min_stop_duration` | 300 s | minimum stay duration |
| DBSCAN | `distance` | 20 m | epsilon (haversine) |
| | `min_points` | 2 | min samples, the stay itself included |
| Home / work | `home_min_days` | 14 | min distinct days a cluster is visited to be home |
| | `home_hour_evening` / `home_hour_morning` | 19 / 8 | home time: weekends, or weekday hours `≥ 19` or `< 8` |
| | `work_min_days` | 8 | min distinct work-time days to be work |
| | `work_hour_start` / `work_hour_end` | 9 / 18 | work time: weekdays, `9 ≤ hour ≤ 18` |
| | `min_distance_home_work` | 50 m | min distance between home and work |
| Trajectories | `max_time_gap` | 1800 s | split at longer gaps; max gap to the origin / destination stay |
| | `simplify_tolerance` | 30 m | Douglas-Peucker tolerance (0 keeps every ping) |
| | `min_pings` | 1 | minimum moving pings per trajectory |
| | `min_length_m` | 100 m | minimum trajectory length (0 keeps all) |
| | `speed_window` | 30 s | minimum window for speeds |
| | `speed_max_gap` | 300 s | longer gaps are not used for speeds |
| | `max_speed_kmh` | 250 | faster steps are GPS jumps and split trajectories |
| | `slow_speed_kmh` | 7 | threshold for `frac_slow` |
| | `wkt_measure` | False | `LINESTRING M` with time as the measure |
| Pipeline | `timezone` | **required** | timezone of the data; with `find_timezones`, the fallback (UTC if not given). Without either, the pipeline suggests one ([local time](local-time.md#not-sure-which-timezone)) |
| | `find_timezones` | False | look up each stay's timezone: `True` (bundled polygons) or a `tz,geometry` CSV. Only for data spanning several timezones |
| | `column_map` | {} | your column per field: a name or a list of candidates, matched ignoring case ([input data](input.md)) |
| | `timestamp_unit` | auto | Unix time unit of numeric timestamps: `s`, `ms`, `us`, `ns`, or detected per value |
| | `timestamp_timezone` | UTC | how to read date-times without a timezone |
| | `n_shards` | 1 | device groups processed independently |
| | `remove_sinks` | False | drop pings at coordinate sinks first: `True` finds them, or a parquet path ([sinks](spatial.md#data-quality-coordinate-sinks)) |
| | `sinks_min_devices` | 5000 | devices on one exact coordinate to call it a sink (with `remove_sinks=True`) |

`guess_mode` thresholds are keyword arguments of [`guess_mode`](speed-and-mode.md#mode_guess-a-baseline),
not configuration, because they can be re-applied without recomputing anything.
