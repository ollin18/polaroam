# Home and work

Every stay gets a `location_type`: `H` (home), `W` (work) or `O` (other), following gc-lbs
`get_hw.sh`.

```python
hw = pr.classify_home_work(local)                               # per-stay labels
hw = pr.classify_home_work(local, home_min_days=7, work_min_days=4)
homes = pr.home_work_locations(local)                           # one row per device
```

## The rules

**Home** is the cluster with the most stays during *home time* (weekends, or weekday hours
`≥ home_hour_evening` or `< home_hour_morning`). Only clusters visited on at least
`home_min_days` distinct days qualify.

**Work** is the cluster with the most weekday stays between `work_hour_start` and
`work_hour_end` (inclusive). It must also be:

- visited on at least `work_min_days` distinct days during work time;
- different from home;
- at least `min_distance_home_work` meters from home.

Devices without a home get no work.

Ties are broken by more distinct days, then by the lower `cluster_label`. Noise stays
(`cluster_label = -1`) are always `O`.

| Parameter | Default | gc-lbs country configs | Meaning |
|---|---|---|---|
| `home_min_days` | 14 | 7 | min distinct days visited |
| `home_hour_evening` | 19 | 22 | home time starts (weekdays) |
| `home_hour_morning` | 8 | 7 | home time ends (weekdays) |
| `work_min_days` | 8 | 4 | min distinct work-time days |
| `work_hour_start` / `work_hour_end` | 9 / 18 | 8 / 18 | work time (weekdays, inclusive) |
| `min_distance_home_work` | 50 m | 50 m | min distance between home and work |

!!! tip "Short periods"
    The defaults (14 / 8 days) suit a year of data. For a month, use the gc-lbs country values
    (7 / 4), for example with `PipelineConfig.from_json("pipeline_config.json", "MX")`.

## Outputs

`classify_home_work` returns the 24 columns of the gc-lbs `*_HW` table (see [Output
tables](outputs.md)). `home_work_locations` returns one row per device with a home:

| Column | Description |
|---|---|
| `uid` | device id |
| `home_latitude`, `home_longitude` | home cluster location |
| `work_latitude`, `work_longitude` | work cluster location (null without work) |

![Local hour when stays start, by location type](../assets/figures/distributions_light.png#only-light)
![Local hour when stays start, by location type](../assets/figures/distributions_dark.png#only-dark)

The bottom-right panel shows when stays start by location type in the example data: home stays
peak in the evening, work stays during office hours.
