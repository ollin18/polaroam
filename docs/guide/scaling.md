# Scaling and memory

Two knobs bound memory: **shards** (devices processed in groups) and **batches** (time
processed in pieces). Results do not depend on either.

## Shards

```python
cfg = pr.PipelineConfig(timezone="America/Mexico_City", n_shards=24)        # --shards 24
```

Devices are split into `n_shards` groups by a stable hash of `uid`: the last 7 bytes of the id,
mixed with a multiplicative hash. The assignment does not change across Polars versions, so
on-disk state stays valid. Each shard is processed independently, and peak memory scales with
one shard.

Before processing, each batch is written once, partitioned by shard, with Polars' streaming
partitioned sink (one pass over the input). Each shard then reads only its own pings instead
of re-scanning the input.

| Mexico City month (1.66B pings), 24 shards | Peak RAM | Peak GPU memory |
|---|---|---|
| One-shot, CPU | 24.9 GB | |
| One-shot, GPU | 22.1 GB | 13.1 GB |
| Daily batches, CPU | **3.4 GB** | |

A rule of thumb from this run: ~69M pings per shard used ~25 GB of RAM on the CPU and ~13 GB on
the GPU. Size `n_shards` so that one shard fits comfortably.

## Batches

[Incremental processing](incremental.md) bounds memory by time instead: each `update()` holds
one batch plus the small carried state. A month processed day by day peaked at 3.4 GB.

## DBSCAN chunks

Dense clusters (hundreds of stays at home) make many ε-edges. `cluster_stops(chunk_size=256)`
inserts stays per device in chunks. The result is identical (insertion is exact) and peak
memory is bounded.

## Parallelism

Everything is column-wise Polars, so a single process uses all cores (the Mexico City run
averaged ~14 busy cores). On one machine there is no need for multiprocessing.

## Several machines

Devices never interact: stays, trajectories, DBSCAN, home / work and routines are all per
device. So the shards can run on different machines, all writing to **one output directory**
on a shared file system (e.g. a cluster's scratch). Each process takes some shards:

```python
pipe = pr.Pipeline("/scratch/out/MX", cfg, trajectories=True)
pipe.update(day_files, batch_id=day, shards=[0, 4, 8])   # this process's shards
pipe.finalize(shards=[0, 4, 8])
```

A shard's tables, state and commit marker (`_commits/batch=<id>/shard=<k>`) are written only
by the process that has it, so processes never overwrite each other, and a failed one is
simply re-run: committed shards are skipped. The result is identical to one process (tested
with three processes at once). `pipe.status()` (or `polaroam status OUT`) lists the committed
shards per batch.

Each process needs its own shards' pings. Either every process reads the whole batch once and
keeps its own shards (simplest), or one job splits the batch once first, and the others read
only their part:

```bash
polaroam update "raw/2023-11-01/*.parquet" out/MX --batch-id 2023-11-01 --split-only   # once
polaroam update "raw/2023-11-01/*.parquet" out/MX --batch-id 2023-11-01 --only-shards 0-7
```

Use the shared split when the input is slow to read (gzipped CSV) or many processes run.

### A SLURM job array (e.g. Savio)

On a SLURM cluster no extra software is needed: each array task takes the shards `k` with
`k % SLURM_ARRAY_TASK_COUNT == SLURM_ARRAY_TASK_ID` (`--only-shards slurm`) and runs every day
in order.

```bash
#!/bin/bash
#SBATCH --array=0-7 --nodes=1 --exclusive --time=24:00:00
OUT=/global/scratch/users/$USER/out/MX
for day in $(cat days.txt); do
  polaroam update "raw/date=$day/*.parquet" $OUT --timezone America/Mexico_City --shards 64 \
      --trajectories --batch-id $day --skip-home-work --only-shards slurm || exit 1
done
polaroam finalize $OUT --only-shards slurm
```

`--exclusive` gives each task a whole node: Polars uses all its cores.

### Ray

[Ray](https://docs.ray.io) does the same from one Python script, with dynamic scheduling and
retries. Start a Ray cluster across the nodes of a SLURM job (head on the first node,
`ray start --address` on the others), then:

```python
import os
import ray
import polaroam as pr

ray.init(address="auto")
cfg = pr.PipelineConfig(timezone="America/Mexico_City", n_shards=64)
OUT = "/global/scratch/users/me/out/MX"


@ray.remote(num_cpus=32)  # a whole node: Polars and numba use its cores
def shard_group(shards, days):
    os.environ.setdefault("POLARS_MAX_THREADS", "32")
    pipe = pr.Pipeline(OUT, cfg, trajectories=True, verbose=False)
    for day, files in days:
        pipe.update(files, batch_id=day, shards=shards, compute_home_work=False)
    pipe.finalize(shards=shards)
    return shards


groups = [list(range(k, 64, 8)) for k in range(8)]
ray.get([shard_group.remote(g, days) for g in groups])
```

Give each task a whole node (`num_cpus` = its cores, or `num_gpus=1` with `use_cuda=True`) so
Polars, numba and Ray do not compete for cores, and keep data on the shared file system, not in
Ray's object store: tasks pass only shard numbers and paths.

Map matching and routines parallelize the same way: split the trajectories by device
(`shard_expr`), run `match_trajectories` / `polaroam routines` on each part, and sum
`edge_volumes` over the parts.
