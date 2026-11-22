"""End-to-end pipeline: pings -> stops (+ trajectories) -> clusters -> local time -> home/work.

``Pipeline.run`` processes everything at once; ``Pipeline.update`` processes
one batch (e.g. one day) and keeps the incremental state on disk, so running
day by day and then ``finalize()`` gives the same tables as ``run``.

Output directory layout::

    out/
      manifest.json                      config + committed batches
      stops/batch=<id>/shard=<k>.parquet         stays, appended per batch
      trajectories/batch=<id>/shard=<k>.parquet  trajectories, appended per batch
      clusters/shard=<k>.parquet                 clustered stops (current)
      home_work/shard=<k>.parquet                stops with local time + H/W/O
      locations/shard=<k>.parquet                one row per device: home & work
      _state/shard=<k>/...                       incremental state

Devices are split into ``n_shards`` groups by a stable hash of ``uid``; each
shard is processed independently, which bounds memory for very large inputs.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from . import engine
from .clustering import ClusterState, IncrementalDBSCAN
from .config import ClusterParams, HomeWorkParams, PipelineConfig, StopParams, TrajectoryParams
from .home_work import classify_home_work, home_work_locations
from .io import PingSource, read_pings, shard_expr
from .local_time import add_local_time
from .stops import StopDetector, StopState

_TABLES = ("stops", "trajectories", "clusters", "home_work", "locations", "_state", "_split")


@dataclass
class BatchSummary:
    batch_id: str
    n_stops: int
    n_trajectories: int
    seconds: float


class Pipeline:
    """Stay detection to home/work, one-shot or incremental.

    >>> pipe = Pipeline("out", PipelineConfig(timezone="America/Bogota"), trajectories=True)
    >>> pipe.run("pings/*.parquet")                         # doctest: +SKIP
    >>> # or, incrementally
    >>> for day in days:                                    # doctest: +SKIP
    ...     pipe.update(f"pings/date={day}/*.parquet", batch_id=day)
    >>> pipe.finalize()                                     # doctest: +SKIP
    >>> pipe.home_work().collect()                          # doctest: +SKIP
    """

    def __init__(self, output_dir: str | Path, config: PipelineConfig | None = None, *,
                 trajectories: bool = False, use_cuda: bool = False, verbose: bool = True):
        self.dir = Path(output_dir)
        self.use_cuda = use_cuda
        self.verbose = verbose
        manifest = self._read_manifest()
        if manifest is not None:
            stored = _config_from_dict(manifest["config"])
            if config is not None and config.to_dict() != stored.to_dict():
                raise ValueError(f"{self.dir} was created with a different config. Use a new output_dir "
                                 "or Pipeline.reset().")
            self.config = stored
            self.trajectories = manifest["trajectories"]
            if trajectories and not self.trajectories:
                raise ValueError("This pipeline was created without trajectories; use a new output_dir.")
        else:
            self.config = config or PipelineConfig()
            self.trajectories = trajectories

    # ------------------------------------------------------------------ public
    def run(self, pings: PingSource, *, overwrite: bool = False, column_map: dict[str, str] | None = None) -> BatchSummary:
        """Process all pings at once (resets previous outputs if ``overwrite``)."""
        if self._read_manifest() is not None:
            if not overwrite:
                raise FileExistsError(f"{self.dir} already holds a pipeline; pass overwrite=True.")
            self.reset()
        return self.update(pings, batch_id="all", flush=True, column_map=column_map)

    def update(self, pings: PingSource, *, batch_id: str | None = None, flush: bool = False,
               compute_home_work: bool = True, column_map: dict[str, str] | None = None) -> BatchSummary:
        """Process one batch of pings (e.g. a day), committing it per shard.

        Batches must come in time order per device. Re-running a committed
        ``batch_id`` is a no-op, so a failed daily job can simply be retried.
        """
        t0 = time.time()
        cfg = self.config
        manifest = self._read_manifest() or self._new_manifest()
        if manifest.get("finalized"):
            raise RuntimeError("Pipeline already finalized; start a new output_dir to add more data.")
        batch_id = str(batch_id if batch_id is not None else f"{len(manifest['batches']):05d}")
        done = set(manifest["batches"].get(batch_id, []))
        if len(done) == cfg.n_shards:
            self._log(f"batch {batch_id} already committed; skipping")
            return BatchSummary(batch_id, 0, 0, 0.0)

        lf = read_pings(pings, column_map or cfg.column_map)
        split = self._split(lf, batch_id) if cfg.n_shards > 1 else None
        n_stops = n_traj = 0
        for k in range(cfg.n_shards):
            if k in done:
                continue
            part = lf if split is None else self._shard_part(split, k)
            sdir = self.dir / "_state" / f"shard={k}"
            det = StopDetector(cfg.stops, trajectories=self.trajectories, trajectory_params=cfg.trajectories,
                               state=StopState.load(sdir), use_cuda=self.use_cuda)
            out = det.update(part, flush=flush)
            db = IncrementalDBSCAN(cfg.dbscan, ClusterState.load(sdir), use_cuda=self.use_cuda)
            db.insert(out.stops)

            self._write(out.stops, "stops", batch_id, k)
            if self.trajectories:
                self._write(out.trajectories, "trajectories", batch_id, k)
            if compute_home_work or flush:
                self._refresh_home_work(db, k)

            tmp = self.dir / "_state" / f"shard={k}.tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            det.state.save(tmp)
            db.state.save(tmp)
            shutil.rmtree(sdir, ignore_errors=True)
            tmp.rename(sdir)

            manifest["batches"].setdefault(batch_id, []).append(k)
            if flush and len(manifest["batches"][batch_id]) == cfg.n_shards:
                manifest["finalized"] = True
            self._write_manifest(manifest)
            n_stops += out.stops.height
            n_traj += out.trajectories.height if out.trajectories is not None else 0
            self._log(f"batch {batch_id} shard {k + 1}/{cfg.n_shards}: {out.stops.height:,} stops"
                      + (f", {out.trajectories.height:,} trajectories" if out.trajectories is not None else "")
                      + f" ({time.time() - t0:.0f}s)")
        if split is not None:
            shutil.rmtree(split, ignore_errors=True)
        return BatchSummary(batch_id, n_stops, n_traj, time.time() - t0)

    def _split(self, lf: pl.LazyFrame, batch_id: str) -> Path:
        """Write the batch once, partitioned by device shard (one streaming pass over the input),
        so each shard reads only its own pings instead of re-scanning everything."""
        d = self.dir / "_split" / f"batch={batch_id}"
        if (d / "_SUCCESS").exists():  # retry of a partially committed batch
            return d
        shutil.rmtree(d, ignore_errors=True)
        t = time.time()
        lf.with_columns(shard_expr(self.config.n_shards).alias("_shard")).sink_parquet(
            pl.PartitionBy(d, key="_shard", include_key=False), mkdir=True, engine="streaming")
        d.mkdir(parents=True, exist_ok=True)  # an empty batch writes no partitions
        (d / "_SUCCESS").touch()
        self._log(f"batch {batch_id}: split into {self.config.n_shards} shards ({time.time() - t:.0f}s)")
        return d

    @staticmethod
    def _shard_part(split: Path, k: int) -> pl.LazyFrame:
        files = sorted((split / f"_shard={k}").glob("*.parquet"))
        if not files:
            return pl.LazyFrame(schema={"uid": pl.String, "latitude": pl.Float64, "longitude": pl.Float64,
                                        "timestamp": pl.Int64})
        return pl.scan_parquet(files, hive_partitioning=False)

    def finalize(self) -> BatchSummary:
        """Close every open stay / trajectory. Call once after the last batch."""
        empty = pl.LazyFrame(schema={"uid": pl.String, "latitude": pl.Float64, "longitude": pl.Float64,
                                     "timestamp": pl.Int64})
        return self.update(empty, batch_id="finalize", flush=True)

    def refresh_home_work(self) -> None:
        """Recompute clusters / home-work tables from the current state."""
        for k in range(self.config.n_shards):
            db = IncrementalDBSCAN(self.config.dbscan, ClusterState.load(self.dir / "_state" / f"shard={k}"),
                                   use_cuda=self.use_cuda)
            self._refresh_home_work(db, k)

    def reset(self) -> None:
        """Delete every table and the state written by polaroam in ``output_dir``."""
        for t in _TABLES:
            shutil.rmtree(self.dir / t, ignore_errors=True)
        (self.dir / "manifest.json").unlink(missing_ok=True)

    # readers
    def stops(self) -> pl.LazyFrame:
        return self._scan("stops")

    def trajectories_table(self) -> pl.LazyFrame:
        return self._scan("trajectories")

    def clusters(self) -> pl.LazyFrame:
        return self._scan("clusters")

    def home_work(self) -> pl.LazyFrame:
        return self._scan("home_work")

    def locations(self) -> pl.LazyFrame:
        return self._scan("locations")

    def locations_h3(self, resolution: int = 9, min_count: int = 5) -> pl.DataFrame:
        """Homes and work places per H3 cell (k-anonymous), see :func:`polaroam.h3.home_work_h3`."""
        from .h3 import home_work_h3

        return home_work_h3(self.locations(), resolution, min_count=min_count)

    # ----------------------------------------------------------------- private
    def _refresh_home_work(self, db: IncrementalDBSCAN, k: int) -> None:
        cfg = self.config
        clusters = db.labels()
        local = add_local_time(clusters, cfg.timezone, cfg.tz_polygons, use_cuda=self.use_cuda)
        hw = classify_home_work(local, cfg.home_work, use_cuda=self.use_cuda)
        self._write(clusters, "clusters", None, k)
        self._write(hw, "home_work", None, k)
        self._write(home_work_locations(local, cfg.home_work, self.use_cuda), "locations", None, k)

    def _write(self, df: pl.DataFrame, table: str, batch_id: str | None, k: int) -> None:
        d = self.dir / table if batch_id is None else self.dir / table / f"batch={batch_id}"
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f".shard={k}.parquet.tmp"
        df.write_parquet(tmp)
        tmp.replace(d / f"shard={k}.parquet")

    def _scan(self, table: str) -> pl.LazyFrame:
        d = self.dir / table
        files = sorted(d.rglob("shard=*.parquet"))
        if not files:
            raise FileNotFoundError(f"No '{table}' output in {self.dir}")
        return pl.scan_parquet(files, hive_partitioning=False)

    def _read_manifest(self) -> dict | None:
        f = self.dir / "manifest.json"
        return json.loads(f.read_text()) if f.exists() else None

    def _new_manifest(self) -> dict:
        return {"config": self.config.to_dict(), "trajectories": self.trajectories,
                "polars": pl.__version__, "batches": {}, "finalized": False}

    def _write_manifest(self, m: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / "manifest.json.tmp"
        tmp.write_text(json.dumps(m, indent=2))
        tmp.replace(self.dir / "manifest.json")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[polaroam] {msg}" + (" (cuda)" if engine.resolve(self.use_cuda) else ""))


def _config_from_dict(d: dict) -> PipelineConfig:
    return PipelineConfig(
        timezone=d["timezone"], tz_polygons=d.get("tz_polygons"),
        stops=StopParams(**d["stops"]), dbscan=ClusterParams(**d["dbscan"]),
        home_work=HomeWorkParams(**d["home_work"]), trajectories=TrajectoryParams(**d["trajectories"]),
        column_map=d.get("column_map", {}), n_shards=d.get("n_shards", 1),
    )
