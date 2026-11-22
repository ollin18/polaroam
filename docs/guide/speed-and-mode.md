# Speed and transport mode

Every trajectory carries speed features computed from **all** of its pings (before
simplification), plus a transparent baseline guess of the transport mode.

```python
trips.select(
    "speed_mean_kmh", "speed_median_kmh", "speed_p85_kmh", "speed_max_kmh", "speed_cv",
    "frac_slow", "straightness", "ping_gap_median_s", "n_jumps", "mode_guess",
)
```

## How speeds are measured

Location pings are noisy (tens of meters) and irregular (seconds to minutes apart). Speeds
between consecutive pings are therefore unreliable, and averaging them treats a burst of pings
the same as a long interval. PolaRoam measures speed in three steps.

**1. Windowed displacement.** For each vertex \(i\), let \(j(i)\) be the first vertex at least
\(\Delta\) = `speed_window` (30 s) later. The window speed is the straight-line displacement over
the elapsed time:

\[
v_i = \frac{d\big(p_i,\, p_{j(i)}\big)}{t_{j(i)} - t_i}
\]

Over ≥ 30 s, GPS jitter cancels instead of adding up. With path length, a walker with 20 m of
noise every 2 s would look like 36 km/h; with displacement it reads under 8 km/h.

**2. Time weights.** Each window is weighted by the time until the next ping,
\(w_i = t_{i+1} - t_i\), so the statistics describe *the trip's time*, whatever the ping rate.

**3. Unobserved time is excluded.**

- Intervals longer than `speed_max_gap` (300 s) are not used: a speed averaged over a 20-minute
  gap mixes moving and waiting. Such gaps are often the time between leaving a stay and the
  first moving ping.
- Steps faster than `max_speed_kmh` (250 km/h) are **GPS jumps**: they split trajectories
  (see [Trajectories](trajectories.md)). A jump between a stay and the first or last ping
  stops that stay from being used as the trip's origin or destination. Any jump that remains
  inside a trip has its window dropped and is counted in `n_jumps`.

With \(W = \sum_i w_i\) over the valid windows:

| Column | Definition | Mode signal |
|---|---|---|
| `speed_mean_kmh` | \(\bar v = \sum_i w_i v_i / W\) | overall pace |
| `speed_median_kmh` | the \(v\) at which half of \(W\) is slower | typical speed, lowered by stops |
| `speed_p85_kmh` | the \(v\) at which 85% of \(W\) is slower | **cruising speed**: the best single separator |
| `speed_max_kmh` | \(\max_i v_i\) | peaks; also data quality |
| `speed_cv` | \(\sqrt{\sum_i w_i (v_i - \bar v)^2 / W}\,/\,\bar v\) | steady vs stop-and-go |
| `frac_slow` | \(\sum_{v_i < 7} w_i / W\) | walking, waiting |
| `straightness` | start-to-end distance / `length_m` | detours, loops, wandering near a place |
| `ping_gap_median_s` | median time between pings | how far the speeds can be trusted |
| `n_jumps` | steps faster than `max_speed_kmh` | data quality |

!!! question "Why p85 and max, but not p95?"
    The **median** is pulled down by red lights, traffic and stops: a bus or a car in traffic
    can have a walking-like median. **p85** is the robust "cruising speed" (traffic engineering
    uses the 85th percentile for the same reason): it ignores the slow parts and is not driven
    by a single bad point. **Max** comes from one window, so it is the least robust, but it
    catches short fast segments and data problems. With a median of ~18 pings per trip, a
    **p95** would be essentially the first or second highest window, almost the same as max,
    so it is not computed.

!!! info "Reading `speed_cv`"
    The CV is the time-weighted standard deviation of the window speeds divided by their mean.

    - **Low (~0.2–0.4):** steady motion, such as a car on a highway, the metro between
      stations, or steady cycling.
    - **High (≳ 0.8):** stop-and-go, such as a bus with stops or congested traffic.
    - Walking sits in between, at low absolute speed.

    It is most useful together with p85. High p85 with high CV suggests a bus or city traffic;
    high p85 with low CV suggests a highway or metro.

![Speed features by mode](../assets/figures/mode_features_light.png#only-light)
![Speed features by mode](../assets/figures/mode_features_dark.png#only-dark)

## Speed profiles of real trips

Each panel shows a real trip from the example data, recomputed at full resolution:

- the line is the window speed over time, and the dashed line is the trip's p85;
- shaded areas are gaps longer than 5 minutes, excluded from the statistics.

![Speed over time for a walk, a bike ride and a motorized trip](../assets/figures/speed_profiles_light.png#only-light)
![Speed over time for a walk, a bike ride and a motorized trip](../assets/figures/speed_profiles_dark.png#only-dark)

These panels show what LBS data looks like: irregular sampling, gaps, and the occasional
implausible spike (the 220 km/h tail of the motorized trip is below the 250 km/h jump cap).
Lower `max_speed_kmh` for urban studies, and use `ping_gap_median_s` and `n_jumps` to filter
trips before modeling.

## `mode_guess`: a baseline

`mode_guess` is a transparent rule-based baseline, meant to be replaced by a trained model:

| Label | Rule (defaults) |
|---|---|
| `unknown` | duration < 60 s, fewer than 2 pings, no valid speed, **or median ping gap > 300 s** (too sparse to measure speed) |
| `walk` | p85 < 8 km/h, max < 15 km/h, **and** overall speed < 15 km/h |
| `bike` | p85 < 22 km/h, max < 35 km/h, **and** overall speed < 35 km/h |
| `motorized` | anything faster (car, bus, metro, …) |

The max and overall conditions matter:

- **Max:** a trip that waits for 20 minutes and then drives for 2 has a walking-like p85, but
  no walker reaches 60 km/h.
- **Overall:** the overall speed (`length_m / duration`, gaps included) catches distance
  covered *across* gaps longer than `speed_max_gap`, which the windowed speeds rightly skip. It
  is a lower bound of the moving speed, since the duration includes waiting: 30 km in 20
  minutes is not a walk, however slow the observed windows look.

Re-run the rules with your own thresholds without recomputing anything:

```python
trips = pr.guess_mode(trips, walk_p85_kmh=7, walk_max_kmh=12, bike_p85_kmh=25,
                      bike_max_kmh=40, max_ping_gap_s=180)
```

In the example data (`min_pings=2`, `min_length_m=100`), 42% of trajectories are `unknown`
because their pings are minutes apart. The rest look plausible:

| Mode | Trajectories | Median p85 | Median max | Median length | Longest |
|---|---|---|---|---|---|
| motorized | 52,684 | 37.0 km/h | 68.5 km/h | 6.3 km | 535 km (intercity) |
| walk | 26,212 | 2.1 km/h | 3.3 km/h | 0.3 km | 27.7 km |
| bike | 16,496 | 12.4 km/h | 18.4 km/h | 1.6 km | 59.6 km |

!!! warning "`walk` is not only walking"
    The `walk` class is slower than real walking (~4–5 km/h). Much of it is lingering or GPS
    drift near a place that did not qualify as a stay (pings scattered beyond 20 m). Use
    `straightness`, `length_m` and `frac_slow` to separate real walks, or train a model on these
    features with labeled trips.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `speed_window` | 30 s | minimum window for speeds |
| `speed_max_gap` | 300 s | longer gaps are not used for speeds |
| `max_speed_kmh` | 250 | faster steps are GPS jumps |
| `slow_speed_kmh` | 7 | threshold for `frac_slow` |

All features are computed with per-trajectory prefix sums in row order. The results are
bit-identical across runs, CPU and GPU, and day-by-day vs one-shot processing.
