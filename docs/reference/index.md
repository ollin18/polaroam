# API reference

Everything below is importable from the top-level package:

```python
import polaroam as pr
```

| Area | Main entry points |
|---|---|
| [Pipeline](pipeline.md) | `Pipeline`, `BatchSummary` |
| [Stays](stops.md) | `detect_stops`, `StopDetector`, `StopState` |
| [Clustering](clustering.md) | `cluster_stops`, `IncrementalDBSCAN`, `ClusterState`, `connected_components` |
| [Trajectories](trajectories.md) | `extract_trajectories`, `build_trajectories`, `speed_features`, `guess_mode`, `douglas_peucker` |
| [Local time](local_time.md) | `add_local_time`, `lookup_timezones` |
| [Home and work](home_work.md) | `classify_home_work`, `home_work_locations`, `home_work_clusters` |
| [H3 aggregation](h3.md) | `home_work_h3`, `h3_counts`, `h3_cells` |
| [Data quality](quality.md) | `coordinate_sinks`, `drop_sinks` |
| [Configuration](config.md) | `PipelineConfig`, `StopParams`, `ClusterParams`, `HomeWorkParams`, `TrajectoryParams` |
| [Input and sharding](io.md) | `read_pings`, `shard_expr` |
| [Engine](engine.md) | `cuda_available`, `configure_gpu`, `collect` |
| [Geometry helpers](geo.md) | `haversine`, `exact_mean`, `tie_exact_mean`, `with_group_row` |
