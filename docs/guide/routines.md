# Routines and trip completion

Location data rarely sees a whole trip. A phone reports when an app asks for its location, so
one day may show the start of the commute, another the middle, another nothing at all. On one
month of Mexico City, **6% of trips** (moves between two stays at least 200 m apart) were
observed from stay to stay, 32% in part and **61% not at all**.

But people repeat trips, and the days complement each other. polaroam finds **routines**, trips
between the same two places on several days, and uses them to:

- give a **mode** to trips that have none of their own;
- **complete** each routine trip on the road network, from stay to stay, filling what a day
  did not show with the streets the device used on its *other* days.

```python
import polaroam as pr

trips = pr.stay_trips(clusters, trajectories, timezone="America/Mexico_City")
trips = pr.find_routines(trips, min_days=3)            # routine_id, routine mode, final `mode`
trajs = pr.routine_modes(trajectories, trips)          # the mode back on the trajectories

nets = {"walk": walk_net, "bike": bike_net, "motorized": drive_net}
matched = pr.match_trajectories(trajs, nets, by="mode")
done = pr.complete_trips(trips, matched, nets)         # by="mode"
done.trips    # one route per trip: observed and imputed length, estimated times, geometry
done.edges    # the route's edges in order, each flagged `imputed`
done.routes   # each routine's usual streets: trips observed on each edge
```

`clusters` are the clustered stays (`cluster_stops`, or the pipeline's `clusters/` table);
`trajectories` come from the trajectory stage. For completion, extract them with every ping
(`simplify_tolerance=0`), as for [map matching](map-matching.md).

## Trips between stays

`stay_trips` turns consecutive stays into trips and puts each trajectory in the trip whose
stays it lies between. Each trip gets a `state`:

| `state` | Meaning |
|---|---|
| `complete` | one trajectory from the origin stay to the destination stay, ending within `end_gap_m` (500 m) of both, without a hole (two consecutive points more than 1 km and 5 minutes apart) |
| `partial` | pings between the stays, but not a complete trajectory |
| `empty` | no pings between the stays |

It also gives the trip's own mode, `mode_observed`: the main mode of its trajectories
(motorized > bike > walk, as travel surveys count a walk to the bus as a bus trip), or null if
all are `unknown`.

## Routines and their mode

`find_routines` groups trips by origin and destination location (DBSCAN clusters). A pair seen
on at least `min_days` different days is a routine. Stays outside any cluster are never part of
a routine.

A routine's **usual mode** is the most common mode of its trips, if at least `mode_min_trips`
(2) of its trips have a mode and at least `mode_min_share` (60%) of those agree. Each trip's
final `mode` is its own mode, or else its routine's; `mode_source` says which (`observed` or
`routine`). `routine_modes` passes the routine's mode on to the `unknown` trajectories of those
trips (`mode_source = "routine"`), so they can be map-matched on the right network.

!!! note "Departure time is a weak signal"
    In Mexico City, only half of a routine's trips left within an hour of its usual time
    (`routine_hour`). Routines are defined by place; the hour is reported, not required.

## Completing trips

`complete_trips` builds one route per routine trip, on the network of its mode, through
*anchors*: the origin stay, the trip's matched points in time order, and the destination stay.
Every step between consecutive anchors is routed on the network.

- **Observed steps.** Consecutive points close in space or time (≤ `gap_m`, 500 m, or ≤
  `gap_s`, 2 minutes apart) follow the plain shortest path, as in map matching.
- **Imputed steps.** A longer step with no point in between, or a step to or from a stay,
  was not observed. It follows the shortest path with each street made cheaper by how often
  the routine's **other** trips were observed on it:

    \[
    \text{cost}(e) = \text{length}(e) \times \Big(1 - p \cdot \frac{n_e}{N}\Big)
    \]

    where \(n_e\) is the number of other trips of the routine observed on edge \(e\), \(N\)
    the number of other trips with observations, and \(p\) = `preference` (0.7). A street the
    device used on every other day costs 30% of its length, so the route follows the device's
    habit when there is one, and the shortest path otherwise.

The trip's own observations never count for its own route, and imputed stretches never count
for anyone's: a routine's route is learned only from what was seen, so guesses cannot reinforce
each other. Stays are snapped to their 4 nearest edges within `search_radius` (150 m) and the
cheapest is used. A step whose route would be longer than `max_detour` (3) × the straight
distance is left as a gap: the route then has several parts (MULTILINESTRING).

`empty` trips have no anchor between the two stays. With `fill_empty=True` they get the
routine's usual route, but no times. For the other trips, departure and arrival are estimated
from the first and last observed points, at the trip's observed speed.

| Output | Columns |
|---|---|
| `done.trips` | `uid, trip_id, routine_id, mode, state, n_gaps, n_filled, n_parts, observed_m, imputed_m, length_m, imputed_share, imputed_on_route, route_trips, depart_after, arrive_before, depart_estimated, arrive_estimated, geometry` |
| `done.edges` | `uid, trip_id, mode, seq, edge, from_m, to_m, length_m, imputed, part` + the network's edge columns |
| `done.routes` | `routine_id, mode, edge, trips` |

`imputed_on_route` is the share of the imputed length on streets the routine's other trips
used, and `route_trips` the number of those trips: together, a confidence for the imputation.
`edge_volumes(done.edges, by="mode")` counts trips per street, observed and imputed together;
filter on `imputed` to count only what was seen.

## Does it work? A hold-out test

On the Mexico City example (60,000 devices, full-resolution trajectories), one complete trip of
at least 1 km per routine lost the pings of the middle 60% of its length. It was matched and
completed again, and the imputed stretch compared with the route matched from all its pings
(1,889 trips, length-weighted):

| `preference` | Hidden route recovered (recall) | Imputed route correct (precision) |
|---|---|---|
| 0 (shortest path) | 35% | 51% |
| 0.4 | 48% | 67% |
| **0.7 (default)** | **51%** | **70%** |
| 0.9 | 51% | 70% |

Following the device's own streets recovers about half again as much of the hidden route as
the shortest path does. Most test trips were motorized (1,615; walk 252, bike 22): treat the
walk and bike numbers (44% and 67% recall at 0.7) as indicative.

!!! warning "Completion makes people look more regular"
    Only devices with routines get completed trips, and their imputed stretches follow their
    habits. Analyses of route choice or variability should use observed edges only
    (`imputed == False`), or report both.

Run `examples/mexico_city_routines.py` to reproduce the numbers; see the
[Mexico City example](../examples/mexico-city.md#routines).
