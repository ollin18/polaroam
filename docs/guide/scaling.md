# Scaling and memory

Two knobs bound memory: **shards** (devices processed in groups) and **batches** (time
processed in pieces). Results do not depend on either.

## Shards

```python
cfg = pr.PipelineConfig(n_shards=24)        # --shards 24
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
averaged ~14 busy cores). There is no need for multiprocessing. To use several machines, give
each a disjoint set of shards (`shard_expr(n) == k`) and a separate output directory.
