# Benchmarks

Machine: AMD Ryzen 9 5950X (16 cores, 32 threads), 125 GB RAM, NVIDIA RTX 3090 (24 GB).
Software: Polars 1.44, cudf-polars 26.10, Python 3.12.

## Example data: 11.3M pings, 500 devices

| Step | CPU | `use_cuda=True` |
|---|---|---|
| Stays | 4.3 s | **2.4 s** |
| Stays + trajectories (with speed features) | 10.5 s | **7.4 s** |
| DBSCAN (368k stays) | 9.9 s | **6.0 s** |
| Local time + home / work | 0.1 s | 0.2 s |
| Full `polaroam run` (timezone polygons, trajectories) | 33 s | **20 s** |
| Peak RAM of the full run (1 shard) | 17.3 GB | 13.7 GB |

For comparison, on a Mexico City subset (26M pings, 24.6k devices), the full pipeline took
19 s on the CPU and 12 s on the GPU with PolaRoam (including trajectories), against 48 s for the
Sedona port of gc-lbs (without trajectories) on the same machine.

## Mexico City: 1.66B pings, 1.58M devices, one month

| Run | Wall time | Peak RAM | Peak GPU memory |
|---|---|---|---|
| One-shot, CPU, 24 shards | 24.4 min | 24.9 GB | |
| One-shot, GPU, 24 shards | **15.9 min** | 22.1 GB | 13.1 GB |
| 32 daily batches + `finalize`, CPU, 24 shards | 40.5 min | **3.4 GB** | |

- **Split:** 99 s, one streaming pass over 22 GB of parquet, writing 24 shard partitions.
- **One-shot:** each shard (~69M pings) takes ~56 s on the CPU and ~35 s on the GPU.
- **Daily batches:** 40–100 s per day. The days get slower as the DBSCAN state grows, because
  the state is rewritten after each batch.
- **Equality:** the three runs produce identical tables. These runs predate the trajectory
  speed features (see [Mexico City](../examples/mexico-city.md)).

## History

| Change | Effect |
|---|---|
| Contraction-based connected components (instead of label propagation) | DBSCAN 155 s → 31 s on the example data |
| Vectorized, shrinking Douglas-Peucker | 21 s → 4.4 s on 3M vertices |
| Prepared polygons for timezones | 403 s → 0.7 s for 280k points |
| Polars 1.8 → 1.44 | stays 7.5 s → 4.3 s, DBSCAN 17.5 s → 9.9 s |
| cudf-polars 24.10 → 26.10, in-memory executor | stays fully on the GPU: 4.3 s → 2.4 s |
| One-pass shard split | each shard reads its own files instead of re-scanning the input |
