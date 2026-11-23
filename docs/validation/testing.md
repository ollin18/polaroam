# Test suite

```bash
pip install -e ".[dev]"
pytest -q                    # 111 tests; GPU tests are skipped without CUDA
ruff check src tests examples
```

## What is checked

| Area | Oracle / check | File |
|---|---|---|
| Stays | an independent **row-by-row Python port of the BigQuery SQL**, for three parameter sets, on synthetic data with float32 rounding and tied timestamps | `test_stops.py` |
| Timestamp units | ms, s and Datetime inputs give the same stays | `test_stops.py` |
| Incremental stays and trajectories | 15 random time cuts + flush == one shot, for three seeds; state save / load round trip | `test_stops.py` |
| DBSCAN, `min_points=2` | identical partition to **scikit-learn** DBSCAN (haversine), for ε = 10, 20, 50 m, including high latitudes | `test_clustering.py` |
| DBSCAN, `min_points` = 3, 5 | identical core partition and noise as scikit-learn (border assignment documented) | `test_clustering.py` |
| Incremental DBSCAN | 7 random insertion batches == one shot, for `min_points` 2, 3, 4 | `test_clustering.py` |
| Connected components | identical to **SciPy** on a random graph | `test_clustering.py` |
| Douglas-Peucker | identical to the **recursive** algorithm on 200 lines, including collinear and tied points | `test_trajectories_hw.py` |
| Trajectories | connect real stays; extraction from stored stays == extraction during detection | `test_trajectories_hw.py` |
| Local time | weekday convention, fixed and polygon timezones, fallback | `test_trajectories_hw.py` |
| Pipeline | daily batches in 3 shards (with retried batches) == one-shot `run`, for all five tables | `test_trajectories_hw.py` |
| Speed features | known answers: constant speeds, stop-and-go time weighting, GPS jumps, 2-second noisy pings, long gaps, short fast endings | `test_speed.py` |
| Timestamps and WKT M | vertex times aligned with coordinates; `LINESTRING M` measures | `test_speed.py` |
| Mode guess | walk / bike / motorized on known trips; custom thresholds | `test_speed.py` |
| Input | column names and aliases, timestamp formats and units, bad values raise | `test_io.py` |
| Timezones | built-in polygons equal the gc-lbs CSV; exact grid lookup; suggestions when no timezone is given | `test_timezones.py` |
| Coordinate sinks | found, removed in detection and the pipeline, reused across batches | `test_sinks.py`, `test_h3.py` |
| H3 | counts with k-anonymity, cell geometry and resolution | `test_h3.py` |
| Map matching | follows a known route on a street grid; HMM path identical to **brute force** over all candidate sequences; gaps and off-network points; modes and volumes; chunked output == in memory | `test_roads.py` |
| Routines | trip states, routine mode passed to trips without one; completion follows the device's habit over a shorter street; empty trips filled with and without preference; the `routines` command | `test_routines.py` |
| CUDA | every stage on GPU == CPU **exactly**; incremental on GPU == one-shot on CPU | `test_cuda.py` |

## Large-scale checks

Beyond the suite, the full Mexico City month (1.66B pings) was run three ways (CPU one-shot,
GPU one-shot, 32 daily batches) with identical results, and compared with Sedona (see [Parity
with gc-lbs](parity.md)).
