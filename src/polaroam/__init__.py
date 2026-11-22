"""PolaRoam: stay detection, trajectories and home/work locations with Polars.

Quick start::

    import polaroam as pr

    stops, trajs = pr.detect_stops("pings.parquet", trajectories=True)
    clusters = pr.cluster_stops(stops)
    local = pr.add_local_time(clusters, timezone="America/Mexico_City")
    hw = pr.classify_home_work(local)

or the whole thing, one-shot or day by day, with :class:`Pipeline`.
"""

from .clustering import ClusterState, IncrementalDBSCAN, cluster_stops, connected_components
from .config import ClusterParams, HomeWorkParams, PipelineConfig, StopParams, TrajectoryParams
from .engine import configure_gpu, cuda_available
from .h3 import h3_cells, h3_counts, home_work_h3
from .home_work import classify_home_work, home_work_clusters, home_work_locations
from .io import read_pings, shard_expr
from .local_time import add_local_time
from .pipeline import Pipeline
from .quality import coordinate_sinks, drop_sinks, find_sinks
from .stops import StopDetector, StopState, detect_stops
from .trajectories import build_trajectories, douglas_peucker, extract_trajectories, guess_mode, speed_features

__author__ = "Ollin Langle"
__version__ = "0.3.0"

__all__ = [
    "ClusterParams",
    "ClusterState",
    "HomeWorkParams",
    "IncrementalDBSCAN",
    "Pipeline",
    "PipelineConfig",
    "StopDetector",
    "StopParams",
    "StopState",
    "TrajectoryParams",
    "add_local_time",
    "build_trajectories",
    "classify_home_work",
    "cluster_stops",
    "configure_gpu",
    "connected_components",
    "coordinate_sinks",
    "cuda_available",
    "detect_stops",
    "douglas_peucker",
    "drop_sinks",
    "extract_trajectories",
    "find_sinks",
    "guess_mode",
    "h3_cells",
    "h3_counts",
    "home_work_clusters",
    "home_work_h3",
    "home_work_locations",
    "read_pings",
    "shard_expr",
    "speed_features",
]
