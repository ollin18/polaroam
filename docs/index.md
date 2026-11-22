---
hide:
  - navigation
  - toc
---

<div class="pr-hero" markdown>
<div markdown>

# Mobility pings to stays, trips and homes

<p class="pr-lead">PolaRoam turns raw location pings (LBS / GPS) into <b>stays</b>, <b>trajectories with
speeds and transport mode</b>, recurrent <b>places</b> and <b>home / work</b> locations, using pure,
vectorized <a href="https://pola.rs">Polars</a>. All at once or day by day, on CPU or CUDA,
with the same results as the gc-lbs BigQuery pipeline.</p>

[Get started](getting-started/quickstart.md){ .md-button .md-button--primary }
[User guide](guide/index.md){ .md-button }
[GitHub](https://github.com/ollin18/polaroam){ .md-button }

</div>
<img class="pr-logo" src="assets/logo.png" alt="PolaRoam logo">
</div>

<div class="pr-stats">
<div class="pr-stat"><b>1.66B</b><span>pings of a month of Mexico City, processed in 16 min on one GPU</span></div>
<div class="pr-stat"><b>3.4 GB</b><span>peak RAM processing that month day by day</span></div>
<div class="pr-stat"><b>=</b><span>day by day, all at once, CPU and GPU give identical tables</span></div>
<div class="pr-stat"><b>0</b><span>per-device Python loops: everything is column-wise Polars</span></div>
</div>

![Trajectory density of Mexico City](assets/figures/cdmx_flows_light.png#only-light){ width="420" align=right }
![Trajectory density of Mexico City](assets/figures/cdmx_flows_dark.png#only-dark){ width="420" align=right }

## What you get

<div class="grid cards" markdown>

-   :material-map-marker-radius:{ .lg .middle } **Stays**

    ---

    The gc-lbs stay rules, line by line: stationary pings, jump splits, median
    locations and outlier filtering. Bit-identical to the SQL on deterministic data.

    [:octicons-arrow-right-24: Stays](guide/stays.md)

-   :material-vector-polyline:{ .lg .middle } **Trajectories**

    ---

    The movement between stays as simplified lines that connect origin and destination,
    with per-vertex timestamps and `LINESTRING M`.

    [:octicons-arrow-right-24: Trajectories](guide/trajectories.md)

-   :material-speedometer:{ .lg .middle } **Speed and mode**

    ---

    Time-weighted, GPS-noise-robust speeds (median, p85, max, CV, slow share) and a
    transparent walk / bike / motorized baseline.

    [:octicons-arrow-right-24: Speed and mode](guide/speed-and-mode.md)

-   :material-home-city:{ .lg .middle } **Places, home and work**

    ---

    Per-device DBSCAN of stays (incremental), local time from timezone polygons, and
    home / work labels.

    [:octicons-arrow-right-24: Home and work](guide/home-work.md)

-   :material-calendar-sync:{ .lg .middle } **Incremental**

    ---

    Process one day at a time. Open stays carry over and DBSCAN updates in place; the
    result equals one run over all the data.

    [:octicons-arrow-right-24: Incremental processing](guide/incremental.md)

-   :material-expansion-card:{ .lg .middle } **CUDA**

    ---

    `use_cuda=True` runs every query on the GPU through cudf-polars, with automatic CPU
    fallback and identical results.

    [:octicons-arrow-right-24: CUDA](guide/cuda.md)

</div>

## Thirty seconds

```python
import polaroam as pr

stops, trips = pr.detect_stops("pings/*.parquet", trajectories=True)
places = pr.cluster_stops(stops)                                   # per-device DBSCAN
local = pr.add_local_time(places, timezone="America/Mexico_City")
hw = pr.classify_home_work(local)                                  # H / W / O per stay
trips.select("uid", "length_m", "speed_p85_kmh", "mode_guess", "geometry")
```

Or the whole thing, all at once or one day at a time, with disk-backed state:

=== "All at once"

    ```python
    pipe = pr.Pipeline("out/MX", pr.PipelineConfig(timezone="America/Mexico_City", n_shards=24),
                       trajectories=True, use_cuda=True)
    pipe.run("raw/MX/**/*.parquet")
    ```

=== "Day by day"

    ```python
    pipe = pr.Pipeline("out/MX", pr.PipelineConfig(timezone="America/Mexico_City"), trajectories=True)
    for day in days:
        pipe.update(f"raw/MX/date={day}/*.parquet", batch_id=day)
    pipe.finalize()
    ```

=== "Command line"

    ```bash
    polaroam run "raw/MX/**/*.parquet" out/MX --timezone America/Mexico_City --trajectories --cuda
    ```

![Stays per hexagon in five cities](assets/figures/stays_map_light.png#only-light)
![Stays per hexagon in five cities](assets/figures/stays_map_dark.png#only-dark)
