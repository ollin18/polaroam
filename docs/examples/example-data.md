# Example data, end to end

The repository's example file, `data/veraset_movement_416.snappy.parquet`, has **11.3M pings
from 500 devices** in five cities (Mexico City, Bogotá, São Paulo, Manila, Jakarta; ~100
devices each), from January 2020 to December 2021. Every figure on this page comes from
`examples/make_figures.py`.

## Run it

```python
import polaroam as pr

cfg = pr.PipelineConfig(
    find_timezones=True,                              # devices span five timezones
    timezone="UTC",                                   # fallback: stays outside every polygon
    trajectories=pr.TrajectoryParams(min_pings=2, min_length_m=100),
)
pipe = pr.Pipeline("examples/output", cfg, trajectories=True)
pipe.run("data/veraset_movement_416.snappy.parquet")
```

```text
[polaroam] batch all shard 1/1: 368,114 stops, 164,608 trajectories (40s)
```

| Output | Rows |
|---|---|
| stays | 368,114 |
| trajectories (≥ 2 pings, ≥ 100 m) | 164,608 |
| devices with a home | 498 |

## Where people stay

Stays per hexagon, log scale. Each city has ~100 devices over two years:

![Stays per hexagon in five cities](../assets/figures/stays_map_light.png#only-light)
![Stays per hexagon in five cities](../assets/figures/stays_map_dark.png#only-dark)

## Distributions

```python
stops = pipe.stops().collect()
hw = pipe.home_work().collect()
trips = pipe.trajectories_table().collect()
```

![Distributions](../assets/figures/distributions_light.png#only-light)
![Distributions](../assets/figures/distributions_dark.png#only-dark)

- **Stay duration**: median 48 min, with peaks at the ping cadence of the source.
- **Stays per active day**: ~3 per device-day, with a long tail of very mobile devices.
- **Trajectory length**: ~100 m to ~100 km. Douglas-Peucker at 30 m keeps about half of the
  vertices.
- **Start hour by location type**: home stays start in the evening, work stays in the morning
  (local time from the timezone polygons).

## Trajectories

All simplified trajectories of the São Paulo devices, drawn as faint overlapping lines. The
city's road network emerges from ~100 devices:

![São Paulo trajectories](../assets/figures/trajectories_map_light.png#only-light){ width="560" }
![São Paulo trajectories](../assets/figures/trajectories_map_dark.png#only-dark){ width="560" }

One trip, from raw pings to stays and the simplified line (relative frame):

![One trip](../assets/figures/simplification_light.png#only-light){ width="560" }
![One trip](../assets/figures/simplification_dark.png#only-dark){ width="560" }

## Speed and mode

```python
import polars as pl

trips.group_by("mode_guess").agg(
    pl.len(), pl.col("speed_p85_kmh").median(), pl.col("speed_max_kmh").median(),
    pl.col("length_m").median(),
)
```

| Mode | Trajectories | Median p85 | Median max | Median length |
|---|---|---|---|---|
| unknown | 69,216 | | | |
| motorized | 52,684 | 37.0 km/h | 68.5 km/h | 6.3 km |
| walk | 36,381 | 3.1 km/h | 5.2 km/h | 0.4 km |
| bike | 6,327 | 15.2 km/h | 19.0 km/h | 2.5 km |

![Speed features by mode](../assets/figures/mode_features_light.png#only-light)
![Speed features by mode](../assets/figures/mode_features_dark.png#only-dark)

- **Cruising speed (p85, 30 s windows)** by mode (top left). Motorized starts at 22 km/h and
  peaks around 30 km/h. Walks reach past 8 km/h on these short windows: on sparse, noisy pings
  one hop between two fixes reads as a fast window, so walk and bike are told apart on the
  robust cruising speed instead (windows of at least 3 pings), see
  [Speed and mode](../guide/speed-and-mode.md#mode_guess-a-baseline).
- **Speed CV vs p85** (top right): most trips are stop-and-go (CV ~1). Steady fast trips (low CV,
  high p85) are a distinct minority, mostly highways and rail.
- **Mode share by local hour** (bottom left): motorized trips dominate the morning and evening
  peaks; slow movement dominates the night.
- **Trip length by mode** (bottom right): median dot, middle 50% bar and 10th–90th percentile
  line. Medians are 0.4 km (walk), 2.5 km (bike) and 6.3 km (motorized).

Speed over time for one trip of each class (shaded: gaps > 5 min, not used):

![Speed profiles](../assets/figures/speed_profiles_light.png#only-light)
![Speed profiles](../assets/figures/speed_profiles_dark.png#only-dark)

See [Speed and transport mode](../guide/speed-and-mode.md) for the definitions and caveats.

## Reproduce the figures

```bash
pip install -e ".[plot,tz]"
python examples/make_figures.py data/veraset_movement_416.snappy.parquet --find-timezones
```
