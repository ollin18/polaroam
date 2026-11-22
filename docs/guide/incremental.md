# Incremental processing

LBS data usually arrives one day at a time. PolaRoam processes each batch once, carries a small
state forward, and produces **exactly** the tables of a single run over all the data.

```python
pipe = pr.Pipeline("out/MX", cfg, trajectories=True)
for day in days:
    pipe.update(f"raw/MX/date={day}/*.parquet", batch_id=day)
pipe.finalize()
```

On a full month of Mexico City (1.66B pings), 32 daily batches plus `finalize` gave tables
identical to the one-shot run, row by row: 22.4M stays, 21.0M trajectories and 370k home/work
locations. Peak memory was 3.4 GB, against 24.9 GB for the one-shot run.

## Why stays can be incremental

Only the **last ping** of a device has an undetermined *stationary* flag, because it depends on
the next ping, which hasn't arrived yet. A stay is therefore final once it ends before the
device's last two pings. Each batch:

1. emits the stays that became final;
2. carries, per device, only the pings of the still-open stay, plus two context pings (the
   flags of the next pings depend on them);
3. continues the per-device counters (`stop_event`, `trajectory_id`).

Devices without new pings are not touched. `finalize()` closes the stays still open, using the
same end-of-data rule as a single run.

```mermaid
sequenceDiagram
    participant D1 as Day 1 pings
    participant S as Carried state
    participant D2 as Day 2 pings
    D1->>S: final stays emitted; open stay + 2 context pings carried
    S->>D2: prepended to day 2
    D2->>S: newly final stays emitted; new open tail carried
    Note over S: finalize() closes what is still open
```

## Why trajectories can be incremental

A trajectory is final once it ends at a final stay or at a gap longer than `max_time_gap`. The
carry boundary is placed right after the last such point. The open tail of moving pings
therefore travels with the open stay, and the last emitted stay is kept as the origin of the
next trajectory.

## Why DBSCAN can be incremental

Inserting stays can only add ε-edges, raise degrees, turn points into cores and **merge**
clusters, never split them. `IncrementalDBSCAN` updates the graph state using only the edges of
the new stays and of points that just became core. Cluster labels are recomputed
deterministically after each insert. See [Clustering](clustering.md).

## Rules for the batches

- **Time order per device.** A batch must not contain pings older than what was already
  carried for that device. Late pings are processed, but they cannot change stays that were
  already emitted.
- **Any split works.** Days, hours, or arbitrary cuts all give the same result.
- **Retries are idempotent.** Each shard is committed atomically, and re-running a committed
  `batch_id` is a no-op.
- **Home / work is recomputed** from all stays after each batch (or at the end with
  `compute_home_work=False` and `refresh_home_work()`).

!!! tip "CPU for daily updates"
    A day is small, so per-batch overhead dominates and the GPU gives little. Use the GPU for
    backfills and one-shot runs.
