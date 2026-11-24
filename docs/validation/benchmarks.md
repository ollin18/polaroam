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

All at once (`polaroam run`, 24 shards, trajectories with speed features, coordinate sinks
removed), on 32 cores and one RTX 3090:

| Run | Wall time | Peak RAM | Peak GPU memory |
|---|---|---|---|
| CPU | 23.9 min | 20.8 GB | |
| GPU (`--cuda`) | 15.4 min | 22.5 GB | 11.5 GB |
| Hybrid (`--cuda --hybrid`) | **14.2 min** | 40.5 GB | 11.5 GB |

- **Split:** 1.7–2.9 min, one streaming pass over 22 GB of parquet, writing 24 shard partitions.
- **Per shard** (~69M pings): ~55 s on the CPU, ~31 s on the GPU.
- **Hybrid** runs two shards at a time, one on each engine (here 15 on the GPU, 9 on the CPU).
  It saves only 8% over the GPU alone, for almost twice the RAM: side by side, each shard took
  ~50 s on the GPU and ~100 s on the CPU, because the GPU path still reads the parquet and builds
  the trajectories on the same cores. It pays off on machines with more cores per GPU.
- **Equality:** the three runs produce identical tables (stays, clusters, trajectories,
  home/work, locations), compared exactly, shard by shard.

Day by day, before the trajectory speed features existed (sinks kept):

| Run | Wall time | Peak RAM |
|---|---|---|
| One-shot, CPU | 24.4 min | 24.9 GB |
| 32 daily batches + `finalize`, CPU | 40.5 min | **3.4 GB** |

- **Daily batches:** 40–100 s per day. The days get slower as the DBSCAN state grows, because
  the state is rewritten after each batch. Both runs produced identical tables.

## History

| Change | Effect |
|---|---|
| Contraction-based connected components (instead of label propagation) | DBSCAN 155 s → 31 s on the example data |
| Vectorized, shrinking Douglas-Peucker | 21 s → 4.4 s on 3M vertices |
| Prepared polygons for timezones | 403 s → 0.7 s for 280k points |
| Polars 1.8 → 1.44 | stays 7.5 s → 4.3 s, DBSCAN 17.5 s → 9.9 s |
| cudf-polars 24.10 → 26.10, in-memory executor | stays fully on the GPU: 4.3 s → 2.4 s |
| One-pass shard split | each shard reads its own files instead of re-scanning the input |
