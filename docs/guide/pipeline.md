# The pipeline

`Pipeline` chains every stage, writes parquet tables and keeps the incremental state on disk.

```python
import polaroam as pr

cfg = pr.PipelineConfig(
    timezone="America/Bogota",                 # required (or find_timezones=True)
    column_map={"uid": "device_id"},           # optional: your column names
    n_shards=8,                                # optional: split devices to bound memory
    trajectories=pr.TrajectoryParams(min_pings=2, min_length_m=100),
)
pipe = pr.Pipeline("out/CO", cfg, trajectories=True, use_cuda=True)
```

## All at once

```python
pipe.run("raw/CO/**/*.parquet")                 # overwrite=True to replace previous outputs
```

## Day by day

```python
for day in days:
    pipe.update(f"raw/CO/date={day}/*.parquet", batch_id=day)
pipe.finalize()                                  # closes stays still open after the last day
```

Day by day plus `finalize()` gives **exactly** the same tables as `run()` (see
[Incremental processing](incremental.md)).

- **Retries are safe.** Each shard of a batch is committed atomically and recorded in
  `manifest.json`. Re-running a committed `batch_id` does nothing, so a failed daily job can
  simply be retried; it resumes at the first uncommitted shard.
- **Order matters per device.** Batches must arrive in time order (e.g. one day at a time).
- **Clusters and home/work** are recomputed from the accumulated stays after every batch. Pass
  `compute_home_work=False` to skip that during a backfill and call
  `pipe.refresh_home_work()` at the end. These steps work on stays, not pings, and are cheap.
- **The configuration is fixed per output directory.** Opening it with a different config
  raises an error; use a new directory or `pipe.reset()`.

## Reading the results

```python
pipe.stops()                # LazyFrames over the output tables
pipe.trajectories_table()
pipe.clusters()
pipe.home_work()
pipe.locations()
```

## Output layout

```text
out/CO/
├── manifest.json                              config, polars version, committed batches
├── stops/batch=<id>/shard=<k>.parquet         stays, appended per batch
├── trajectories/batch=<id>/shard=<k>.parquet  trajectories, appended per batch
├── clusters/shard=<k>.parquet                 clustered stays (current state)
├── home_work/shard=<k>.parquet                stays with local time and H / W / O
├── locations/shard=<k>.parquet                one row per device: home and work
├── _state/shard=<k>/                          incremental state (carried pings, DBSCAN graph)
└── _split/batch=<id>/                         temporary: the batch split by shard
```

With `n_shards > 1`, each batch is first written once, partitioned by device shard, in one
streaming pass over the input. Each shard then reads only its own pings. This is the
equivalent of the gc-lbs "slim" table and avoids re-scanning the input once per shard. The
`_split` directory is deleted when the batch is committed.

See [Output tables](outputs.md) for every column.

## Lower-level building blocks

The pipeline is a thin layer over stateful objects you can use directly:

```python
det = pr.StopDetector(pr.StopParams(), trajectories=True)
db = pr.IncrementalDBSCAN(pr.ClusterParams())
for batch in batches:
    out = det.update(batch)          # stays and trajectories that became final
    db.insert(out.stops)
last = det.flush()
db.insert(last.stops)
clusters = db.labels()

det.state.save("state/")
db.state.save("state/")              # pr.StopState.load(...), pr.ClusterState.load(...)
```
