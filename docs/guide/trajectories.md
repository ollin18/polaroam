# Trajectories

Pings that are not part of a stay are **moving** pings. PolaRoam chains them into trajectories
that connect stays, simplifies the lines, and keeps the time of every vertex.

```python
# while detecting stays
stops, trips = pr.detect_stops(pings, trajectories=True,
                               trajectory_params=pr.TrajectoryParams(min_pings=2, min_length_m=100))

# or later, from stays that are already processed (stops, clusters or home_work tables)
trips = pr.extract_trajectories(pings, "out/MX/home_work/*.parquet", min_pings=2, min_length_m=100)
```

## How trajectories are built

- A ping inside a stay's `[start, end]` is a stay ping. Any other ping is a moving ping.
- Consecutive moving pings of a device form **one trajectory** while three things hold:
    - they are less than `max_time_gap` (30 min) apart;
    - no stay happens between them;
    - the step between them is **not a GPS jump** (faster than `max_speed_kmh`).
- The stay just before (after) the trajectory, if it ended (starts) within `max_time_gap`,
  becomes its **origin (destination)**. Its location is added as the first (last) vertex, so
  trajectories connect stays. When jumps split the movement between two stays into pieces,
  only the first piece connects to the origin and only the last to the destination, and only
  if reaching that stay is itself not a jump.
- The line is simplified with **Douglas-Peucker** (`simplify_tolerance` meters).

![Raw pings, stays and the simplified trajectory of one trip](../assets/figures/simplification_light.png#only-light){ width="620" }
![Raw pings, stays and the simplified trajectory of one trip](../assets/figures/simplification_dark.png#only-dark){ width="620" }

### Douglas-Peucker, vectorized

The classic algorithm is recursive. PolaRoam evaluates it **breadth-first**: each iteration
splits *every* open segment of *every* trajectory at its farthest vertex, using forward and
backward fills and a grouped max. Vertices of segments that stopped splitting leave the working
set, so later iterations only touch the few long, wiggly segments left. The result equals the
recursive algorithm, which the tests check, including ties and collinear points. Distances are
compared at micrometer resolution, so CPU and GPU arithmetic cannot break ties differently.

## Columns

| Column | Type | Description |
|---|---|---|
| `uid`, `trajectory_id` | str, i64 | device and trajectory number (1, 2, … in time order) |
| `origin_stop_event`, `destination_stop_event` | i64 | connected stays (null if none within `max_time_gap`) |
| `start_timestamp`, `end_timestamp`, `duration` | i64 | Unix seconds; seconds |
| `n_pings` | u32 | moving pings |
| `n_points` | u32 | vertices kept after simplification |
| `length_m` | f64 | path length over all vertices (m) |
| `coordinates` | list[list[f64]] | `[[lon, lat], …]`: every kept vertex, not just the ends |
| `timestamps` | list[i64] | Unix seconds of each vertex, aligned with `coordinates` |
| `geometry` | str | WKT `LINESTRING`, or `LINESTRING M (lon lat t, …)` with `wkt_measure=True` |
| speed features, `mode_guess` | | see [Speed and transport mode](speed-and-mode.md) |

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `max_time_gap` | 1800 s | split at longer gaps; max gap to the origin / destination stay |
| `simplify_tolerance` | 30 m | Douglas-Peucker tolerance. **0 keeps every ping** (full resolution) |
| `min_pings` | 1 | minimum moving pings (2 drops single-ping "trips") |
| `min_length_m` | 100 m | minimum path length (drops GPS jitter near stays; 0 keeps all) |
| `wkt_measure` | False | write `LINESTRING M` with time as the measure |
| `speed_window`, `speed_max_gap`, `max_speed_kmh`, `slow_speed_kmh` | | see [Speed and transport mode](speed-and-mode.md) |

!!! tip "Micro-trajectories"
    About 40% of raw "trajectories" are tiny: a single moving ping between two stays, or GPS
    jitter just outside a stay's 20 m radius (median length ~54 m in the example data).
    `min_length_m` (100 m by default) removes them; `min_pings=2` also drops single-ping trips.
    Set `min_length_m=0` to keep everything.

!!! danger "Why jumps split trajectories"
    LBS pings can **ping-pong** between the true position and a distant fallback point (a cell
    tower, a default geocode), many times within a few minutes. Before jump splitting, a month
    of Mexico City had 103k "walks" longer than 100 km. Their median duration was 3.4 minutes,
    with 18 jumps each and straightness ≈ 0: every bounce added kilometers of fake path. A jump
    now ends the trajectory just as a long gap does, so ping-pong breaks into short pieces that
    the length filter removes.

## Full resolution vs simplified

Simplification keeps the shape: in the example data, trips over 5 km keep a median of 12
vertices out of 18 pings. For analysis that needs every ping, use `simplify_tolerance=0`. The
speed features are always computed from every ping, before simplification, whatever the
tolerance.

## Incremental

Trajectories are emitted once they are closed, ending at a stay or at a gap longer than
`max_time_gap`. Open tails are carried to the next batch with the open stays, so day-by-day
processing yields the same trajectories, ids and features as one run (see [Incremental
processing](incremental.md)).

![Simplified trajectories in São Paulo](../assets/figures/trajectories_map_light.png#only-light){ width="560" }
![Simplified trajectories in São Paulo](../assets/figures/trajectories_map_dark.png#only-dark){ width="560" }
