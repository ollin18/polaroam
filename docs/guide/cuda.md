# CUDA

```python
pr.cuda_available()                         # True if cudf-polars + a CUDA device work
pr.detect_stops(df, use_cuda=True)          # every function takes use_cuda
pr.Pipeline("out", cfg, use_cuda=True)      # polaroam run ... --cuda
```

With `use_cuda=True`, every query goes to the Polars GPU engine (cudf-polars) with
`raise_on_fail=False`. Supported plans run on the GPU, and anything else falls back to the CPU
automatically. If CUDA or cudf-polars is not available, you get one warning and the CPU engine.

## What runs where

With **cudf-polars ≥ 26.4** every polaroam query runs on the GPU:

- stay labeling (shifts, windows) and stay aggregation;
- the DBSCAN neighbor join and connected components;
- the Douglas-Peucker iterations;
- home / work.

Three small steps stay on the CPU on purpose:

| Step | Why on the CPU |
|---|---|
| `join_asof` assigning pings to stays | not supported by the GPU engine |
| WKT text and coordinate lists | the GPU formats floats as text differently; outputs must match |
| speed features | prefix sums in row order make them bit-identical across runs |

Older cudf-polars versions (< 26.4) have no `shift` and run most of stay detection on the CPU.

## Executor

PolaRoam uses cudf-polars' single-GPU **in-memory** executor by default. On these window- and
join-heavy queries it was about 8× faster than the multi-partition streaming executor, which is
the cudf-polars default. Each shard must fit in GPU memory: one month of Mexico City split into
24 shards peaked at 13.1 GB on a 24 GB card. Options:

```python
pr.configure_gpu(device=1)                  # another GPU
pr.configure_gpu(executor="streaming")      # shards larger than GPU memory
```

## Identical results

GPU and CPU outputs are **identical**, compared exactly, on every table of the full Mexico City
month. Floating-point details that could differ between engines are made order-independent:

- **Sums and averages** used for cluster ranking and home/work locations are exact fixed-point
  integer sums, and trajectory lengths are summed as integer micrometers.
- **Douglas-Peucker** compares distances at micrometer resolution, so equal distances tie the
  same way on both engines.
- **Speed features** are computed on the CPU in row order.

## Speedups

| Data | CPU | GPU |
|---|---|---|
| Example data, stays (11.3M pings) | 4.3 s | 2.4 s |
| Example data, DBSCAN (368k stays) | 9.9 s | 6.0 s |
| Example data, full `polaroam run` | 33 s | 20 s |
| Mexico City month, one-shot (1.66B pings) | 24.4 min | 15.9 min |

See [Benchmarks](../validation/benchmarks.md) for details.
