# Mexico City at scale

A full month of location data for Mexico City: **1.66 billion pings from 1.58 million
devices**, December 2022 (the `filtered_localized` quadrant dataset, 22 GB of parquet in 32
daily partitions). The run uses the gc-lbs MX configuration.

```python
import polaroam as pr

cfg = pr.PipelineConfig.from_json("pipeline_config.json", "MX", n_shards=24)
pipe = pr.Pipeline("/data/out/MX_2022_12", cfg, trajectories=True, use_cuda=True)
pipe.run("/data/Berkeley/MX/data_quadrant/filtered_localized/*/*.parquet")
```

| Output | Rows |
|---|---|
| stays | 22,391,432 |
| trajectories | 21,029,009 |
| devices with a home | 370,200 |
| devices with a work place | 58,766 |

## Stays, homes and work places

Counts per ~150 m cell, log scale. The quadrant dataset is clipped to the city's boundary:

![Stays, homes and work places in Mexico City](../assets/figures/cdmx_stays_light.png#only-light)
![Stays, homes and work places in Mexico City](../assets/figures/cdmx_stays_dark.png#only-dark)

Homes cover the whole residential city. Work places concentrate along the central corridors.
With one month of data (7 / 4 day thresholds), one device in four with a home also has a work
place.

## The city drawn by its trajectories

Every vertex of the 21M simplified trajectories (103M points), as a density raster. The road
network appears from movement alone:

![Trajectory density in Mexico City](../assets/figures/cdmx_flows_light.png#only-light){ width="640" }
![Trajectory density in Mexico City](../assets/figures/cdmx_flows_dark.png#only-dark){ width="640" }

## Three ways, one answer

| Run | Wall time | Peak RAM | Peak GPU memory |
|---|---|---|---|
| One-shot, CPU | 24.4 min | 24.9 GB | |
| One-shot, GPU (RTX 3090) | **15.9 min** | 22.1 GB | 13.1 GB |
| 32 daily batches + `finalize` (CPU) | 40.5 min | **3.4 GB** | |

All three produce **identical** tables (stays, trajectories, clusters, home/work, locations),
compared exactly, row by row.

```python
# day by day
import glob

pipe = pr.Pipeline("/data/out/MX_2022_12_daily", cfg, trajectories=True)
for d in sorted(glob.glob("/data/Berkeley/MX/data_quadrant/filtered_localized/date_trunc=*")):
    pipe.update(f"{d}/*.parquet", batch_id=d.split("=")[-1], compute_home_work=False)
pipe.finalize()
```

Day by day, each batch processes one day of pings (~52M) plus the carried state. The days
run in 40–100 s each on the CPU, growing as more stays accumulate in the DBSCAN state.

The three runs above were made before the trajectory speed features were added (they include
every micro-trajectory: 21.0M trajectories). A rerun with the current version, the speed
features and `min_pings=2, min_length_m=100` gave the same 22.4M stays and **11.2M
trajectories** in 18.9 min on the GPU (peak RAM 24.7 GB).

## Speed and mode at scale

```python
cfg = pr.PipelineConfig.from_json("pipeline_config.json", "MX", n_shards=24,
                                  trajectories=pr.TrajectoryParams(min_pings=2, min_length_m=100))
```

| Mode | Trajectories | Median p85 | Median length |
|---|---|---|---|
| motorized | 5,246,347 | 40.7 km/h | 8.5 km |
| unknown | 2,642,458 | 2.8 km/h | 2.6 km |
| walk | 2,112,460 | 0.9 km/h | 2.6 km |
| bike | 1,231,102 | 13.8 km/h | 2.0 km |

![Speed features by mode, Mexico City](../assets/figures/cdmx_modes_light.png#only-light)
![Speed features by mode, Mexico City](../assets/figures/cdmx_modes_dark.png#only-dark)

!!! warning "What the baseline gets wrong here"
    - **Motorized** and **bike** look plausible: cruising speeds of 41 and 14 km/h, trips of
      8.5 and 2.0 km.
    - **`walk` is mostly not walking.** It has a median p85 of 0.9 km/h over 2.6 km: slow
      drift and sparse sampling.
    - **A single fast window** (still under the 250 km/h jump cap) sends some otherwise-slow
      noisy trips to `motorized`: the small bump at p85 ≈ 1 km/h.

    Use `mode_guess` as a baseline and a feature set, not as ground truth. Filter on
    `ping_gap_median_s`, `straightness` and `n_jumps`, lower `max_speed_kmh` for a city, or
    train a model on labeled trips. See [Speed and transport mode](../guide/speed-and-mode.md).

## Homes and work places on H3

Fine-grained heat maps with [H3](../guide/spatial.md), k-anonymous (cells with fewer than 5
devices hidden):

```python
cells = pipe.locations_h3(resolution=10)          # or pr.home_work_h3(locations, 10)
```

![Homes, work places and jobs vs residents on H3](../assets/figures/cdmx_h3_light.png#only-light)
![Homes, work places and jobs vs residents on H3](../assets/figures/cdmx_h3_dark.png#only-dark)

![Roma, Condesa and Juárez at block level](../assets/figures/cdmx_h3_zoom_light.png#only-light)
![Roma, Condesa and Juárez at block level](../assets/figures/cdmx_h3_zoom_dark.png#only-dark)

!!! danger "18% of homes sit on coordinate sinks"
    - **2,592 devices** had their home at exactly 19.4326, -99.1332, the Zócalo: the city's
      default geocode.
    - Five other single points held 1,000–1,500 homes each.
    - **65,963 homes (18%)** sit on exact coordinates shared by 50 or more devices.

    The maps above exclude them. Find and remove them with `pr.coordinate_sinks` and
    `pr.drop_sinks`, ideally on the pings *before* stay detection
    ([details](../guide/spatial.md#data-quality-coordinate-sinks)).

## Reproduce the figures

```bash
python examples/mexico_city_figures.py /data/out/MX_2022_12            # maps and H3
python examples/mexico_city_figures.py /data/out/MX_2022_12 --modes    # speed / mode figure
```
