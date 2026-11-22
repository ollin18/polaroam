# Clustering (DBSCAN)

Stays at the same place on different days are grouped into **clusters** (recurrent places)
with per-device DBSCAN, as in gc-lbs (`dbscan.sh`, `ST_CLUSTERDBSCAN`).

```python
clusters = pr.cluster_stops(stops)                         # eps 20 m, min_points 2
clusters = pr.cluster_stops(stops, distance=25, min_points=3)
```

| Parameter | Default | Meaning |
|---|---|---|
| `distance` | 20 m | DBSCAN epsilon (haversine) |
| `min_points` | 2 | DBSCAN `min_samples`, the stay itself included |

## DBSCAN as a graph, in Polars

1. **Neighbors.** Stays are hashed to a grid whose cells are at least `eps` wide. The cell
   width in longitude uses each device's most poleward latitude, so it is valid anywhere.
   Every pair within `eps` is therefore in the same or adjacent cells. A join on
   `(device, cell)` over the 9 neighboring cells, plus an exact haversine filter, gives the
   ε-graph.
2. **Core points** have `degree + 1 ≥ min_points`.
3. **Clusters** are the connected components of the core–core graph. They are computed by
   repeated *contraction*: every node merges into its smallest neighbor (with pointer
   jumping) and the edge list is rewritten on the contracted graph. Dense groups, like the
   hundreds of stays at home, collapse in the first round, so later rounds run on a tiny
   graph.
4. **Border points** (non-core with a core neighbor) join the cluster of their nearest core
   point. With the default `min_points=2` there are no border points, and the result equals
   scikit-learn and the gc-lbs clusters exactly.

## Incremental DBSCAN

Inserting stays can only add edges, raise degrees, turn points into cores and *merge*
clusters. It never splits one (Ester et al., 1998). `IncrementalDBSCAN` therefore updates
the state (degree, core flag, component, nearest core) using only the edges of the new stays
and of points that just became core:

```python
db = pr.IncrementalDBSCAN(pr.ClusterParams(distance=20, min_points=2))
db.insert(stops_day_1)
db.insert(stops_day_2)
clusters = db.labels()              # identical to cluster_stops(all stays)
```

Inserting in any number of batches gives exactly the same clusters as one run. `cluster_stops`
itself inserts in chunks of `chunk_size=256` stays per device: the result is the same, and peak
memory stays bounded on devices with very dense clusters.

## Labels

As in gc-lbs:

- `cluster_label` is the rank of the cluster centroid (mean latitude, then longitude) within
  the device, starting at 0. Noise is `-1`.
- `cluster_counts` is the number of stays in the cluster (1 for noise).
- `cluster_latitude` / `cluster_longitude` is the median of the member stays (the stay's own
  location for noise).

!!! note "Deterministic labels"
    Centroids are compared with an exact fixed-point mean (1e-9° resolution). Clusters whose
    centroid latitudes are tied, which is common when coordinates are rounded to 5 decimals,
    are ordered by longitude. The labels are then the same on CPU and GPU and for any batch
    order. gc-lbs orders such ties by floating-point noise, so the *numbering* of those
    clusters may differ from it. Membership, counts and coordinates do not.
