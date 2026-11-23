"""Parameter containers for every stage of the pipeline.

Defaults mirror the BigQuery / Sedona LBS pipelines (``gc-lbs``), so a
``pipeline_config.json`` written for those pipelines can be loaded directly
with :meth:`PipelineConfig.from_json`.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import InitVar, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class StopParams:
    """Stop (stay) detection parameters.

    distance_threshold: max distance (m) between consecutive pings of a stay.
    time_threshold: max gap (s) between consecutive pings of a stay.
    min_stop_duration: minimum stay duration (s).
    """

    distance_threshold: float = 20
    time_threshold: float = 3600
    min_stop_duration: float = 300


@dataclass
class ClusterParams:
    """DBSCAN parameters used to group stops into recurrent locations.

    distance: DBSCAN epsilon (m).
    min_points: DBSCAN ``min_samples`` (the point itself counts).
    """

    distance: float = 20
    min_points: int = 2


@dataclass
class HomeWorkParams:
    """Home / work classification parameters (hours are local time)."""

    home_min_days: int = 14
    home_hour_evening: int = 19
    home_hour_morning: int = 8
    work_min_days: int = 8
    work_hour_start: int = 9
    work_hour_end: int = 18
    min_distance_home_work: float = 50


@dataclass
class TrajectoryParams:
    """Trajectory extraction parameters.

    max_time_gap: two consecutive non-stop pings further apart than this (s)
        belong to different trajectories. Also the max gap between a stop and
        the first/last moving ping for the stop to be used as origin/destination.
    simplify_tolerance: Douglas-Peucker tolerance (m). ``0`` keeps every ping (full resolution).
    min_pings: minimum number of moving pings in a trajectory (e.g. 2 drops single-ping "trips").
    min_length_m: minimum path length (m) of a trajectory (drops GPS jitter around stays; 0 keeps all).
    speed_window: speeds are measured over windows of at least this many seconds (GPS error
        dominates the speed of pings only a few seconds apart).
    speed_max_gap: gaps between pings longer than this (s) are not used for speeds: a speed
        averaged over a long gap mixes moving and waiting (often the time between leaving a
        stay and the first moving ping).
    max_speed_kmh: steps faster than this are GPS jumps: they split trajectories (like gaps longer
        than ``max_time_gap``) and are excluded from speeds.
    slow_speed_kmh: speed below which time counts as slow (``frac_slow``; ~walking).
    wkt_measure: write ``geometry`` as ``LINESTRING M`` with the Unix time (s) as measure.
    """

    max_time_gap: float = 1800
    simplify_tolerance: float = 30
    min_pings: int = 1
    min_length_m: float = 100
    speed_window: float = 30
    speed_max_gap: float = 300
    max_speed_kmh: float = 250
    slow_speed_kmh: float = 7
    wkt_measure: bool = False


@dataclass
class PipelineConfig:
    """Full pipeline configuration.

    timezone: IANA timezone of the data (e.g. ``"America/Mexico_City"``), used for every
        stay; with ``find_timezones`` it is the fallback for stays outside every polygon
        (``"UTC"`` if not given). Required unless ``find_timezones`` is set: the pipeline
        raises :class:`polaroam.TimezoneRequired` with a suggestion from a sample of the data.
    find_timezones: look up each stay's timezone: ``True`` uses the polygons bundled with
        polaroam, a path a ``tz,geometry`` CSV of WKT polygons. Costs one lookup per stay
        coordinate, so use it only for data known to span several timezones.
    tz_polygons: deprecated alias of ``find_timezones``.

    remove_sinks: drop the pings at coordinate sinks (default geocodes, cell towers,
        Wi-Fi points) before anything else. ``False`` keeps every ping (gc-lbs behavior);
        ``True`` finds them in the data with :func:`polaroam.find_sinks` (in incremental runs,
        in the first batch) and saves them as ``sinks.parquet`` in the output directory;
        a path uses a parquet of ``latitude, longitude`` computed beforehand.
    sinks_min_devices: distinct devices on one exact coordinate to call it a sink
        (only used with ``remove_sinks=True``).
    drop_coordinates: deprecated alias of ``remove_sinks=<path>``.
    """

    timezone: str | None = None
    find_timezones: bool | str = False
    stops: StopParams = field(default_factory=StopParams)
    dbscan: ClusterParams = field(default_factory=ClusterParams)
    home_work: HomeWorkParams = field(default_factory=HomeWorkParams)
    trajectories: TrajectoryParams = field(default_factory=TrajectoryParams)
    column_map: dict[str, str] = field(default_factory=dict)
    n_shards: int = 1
    remove_sinks: bool | str = False
    sinks_min_devices: int = 5000
    drop_coordinates: InitVar[str | None] = None
    tz_polygons: InitVar[str | None] = None

    def __post_init__(self, drop_coordinates: str | None, tz_polygons: str | None) -> None:
        if tz_polygons is not None:
            warnings.warn("tz_polygons is deprecated; use find_timezones=True (bundled polygons) or "
                          "find_timezones=<csv path>.", DeprecationWarning, stacklevel=3)
            self.find_timezones = True if tz_polygons == "builtin" else str(tz_polygons)
        if isinstance(self.find_timezones, Path):
            self.find_timezones = str(self.find_timezones)
        if self.find_timezones == "builtin":
            self.find_timezones = True
        if not isinstance(self.find_timezones, (bool, str)):
            raise TypeError("find_timezones must be True/False or the path of a tz,geometry CSV.")
        if self.timezone is not None:
            from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

            try:
                ZoneInfo(self.timezone)
            except (ZoneInfoNotFoundError, ValueError):
                raise ValueError(f"Unknown timezone {self.timezone!r}: use an IANA name such as "
                                 "'America/Mexico_City'.") from None
        if drop_coordinates is not None:
            warnings.warn("drop_coordinates is deprecated; use remove_sinks=<path>.", DeprecationWarning,
                          stacklevel=3)
            if self.remove_sinks not in (False, drop_coordinates):
                raise ValueError("Pass remove_sinks or drop_coordinates, not both.")
            self.remove_sinks = str(drop_coordinates)
        if not isinstance(self.remove_sinks, (bool, str)):
            if isinstance(self.remove_sinks, Path):
                self.remove_sinks = str(self.remove_sinks)
            else:
                raise TypeError("PipelineConfig.remove_sinks must be True/False or a parquet path "
                                "(a DataFrame is accepted by detect_stops / extract_trajectories).")

    @classmethod
    def from_json(cls, path: str | Path, label: str | None = None, **overrides: Any) -> PipelineConfig:
        """Load a ``pipeline_config.json`` (gc-lbs format or a flat polaroam one).

        With ``label`` the parameters under ``countries[label]`` are used.
        """
        raw = json.loads(Path(path).read_text())
        section = raw["countries"][label] if label is not None else raw
        kw: dict[str, Any] = {
            "stops": StopParams(**_known(StopParams, section.get("stops", {}))),
            "dbscan": ClusterParams(**_known(ClusterParams, section.get("dbscan", {}))),
            "home_work": HomeWorkParams(**_known(HomeWorkParams, section.get("home_work", {}))),
            "trajectories": TrajectoryParams(**_known(TrajectoryParams, section.get("trajectories", {}))),
        }
        if "timezone" in section:
            kw["timezone"] = section["timezone"]
        for key in ("find_timezones", "column_map", "n_shards", "remove_sinks", "sinks_min_devices"):
            if key in section:
                kw[key] = section[key]
        if "tz_polygons" in section and "find_timezones" not in section:  # older configs
            kw["find_timezones"] = section["tz_polygons"] or False
        if "drop_coordinates" in section and "remove_sinks" not in section:  # older configs
            kw["remove_sinks"] = section["drop_coordinates"]
        if "shards" in section.get("stops", {}):
            kw["n_shards"] = int(section["stops"]["shards"])
        kw.update(overrides)
        return cls(**kw)

    @property
    def tz_source(self) -> str | None:
        """Polygons for the timezone lookup: ``"builtin"``, a CSV path, or ``None`` (fixed timezone)."""
        if self.find_timezones is True:
            return "builtin"
        return self.find_timezones or None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _known(cls: type, values: dict) -> dict:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in values.items() if k in names}
