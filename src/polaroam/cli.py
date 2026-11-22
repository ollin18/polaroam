"""Command line interface.

    polaroam run      INPUT OUTPUT [options]   # everything at once
    polaroam update   INPUT OUTPUT [options]   # one incremental batch (e.g. a day)
    polaroam finalize OUTPUT                   # close open stays after the last batch
    polaroam trajectories PINGS STOPS OUT.parquet
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

from .config import PipelineConfig


def _config(a: argparse.Namespace) -> PipelineConfig | None:
    if a.config:
        cfg = PipelineConfig.from_json(a.config, a.label)
    elif any(v is not None for v in (a.timezone, a.tz_polygons, a.shards, a.distance, a.time, a.min_duration,
                                     a.drop_coordinates,
                                     a.dbscan_dist, a.min_points, *_traj_args(a).values())) or a.col:
        cfg = PipelineConfig()
    else:
        return None
    over = {}
    if a.timezone:
        over["timezone"] = a.timezone
    if a.tz_polygons:
        over["tz_polygons"] = a.tz_polygons
    if a.shards:
        over["n_shards"] = a.shards
    if a.drop_coordinates:
        over["drop_coordinates"] = a.drop_coordinates
    if a.col:
        over["column_map"] = dict(c.split("=", 1) for c in a.col)
    stops = {k: v for k, v in (("distance_threshold", a.distance), ("time_threshold", a.time),
                               ("min_stop_duration", a.min_duration)) if v is not None}
    db = {k: v for k, v in (("distance", a.dbscan_dist), ("min_points", a.min_points)) if v is not None}
    if stops:
        over["stops"] = dataclasses.replace(cfg.stops, **stops)
    if db:
        over["dbscan"] = dataclasses.replace(cfg.dbscan, **db)
    tr = {k: v for k, v in _traj_args(a).items() if v is not None}
    if tr:
        over["trajectories"] = dataclasses.replace(cfg.trajectories, **tr)
    return dataclasses.replace(cfg, **over)


def _add_traj_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("trajectories")
    g.add_argument("--max-gap", type=float, help="split trajectories at gaps longer than this (s)")
    g.add_argument("--tolerance", type=float, help="Douglas-Peucker tolerance (m); 0 = keep every ping")
    g.add_argument("--min-pings", type=int, help="minimum moving pings per trajectory")
    g.add_argument("--min-length", type=float, help="minimum trajectory length (m)")
    g.add_argument("--speed-window", type=float, help="speed window (s)")
    g.add_argument("--max-speed", type=float, help="GPS-jump speed cap (km/h)")
    g.add_argument("--wkt-measure", action="store_true", default=None, help="write LINESTRING M (time as measure)")


def _traj_args(a: argparse.Namespace) -> dict:
    return {"max_time_gap": a.max_gap, "simplify_tolerance": a.tolerance, "min_pings": a.min_pings,
            "min_length_m": a.min_length, "speed_window": a.speed_window, "max_speed_kmh": a.max_speed,
            "wkt_measure": a.wkt_measure}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="polaroam", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "update"):
        p = sub.add_parser(name)
        p.add_argument("input", help="parquet/csv file, glob or directory of pings")
        p.add_argument("output", help="output directory")
        p.add_argument("--config", help="pipeline_config.json (gc-lbs format accepted)")
        p.add_argument("--label", help="country/label inside --config, e.g. CO")
        p.add_argument("--timezone", help="fixed (or fallback) timezone, e.g. America/Bogota")
        p.add_argument("--tz-polygons", help="CSV with tz,geometry(WKT) columns for per-stop timezones")
        p.add_argument("--shards", type=int, help="split devices into N groups (bounded memory)")
        p.add_argument("--drop-coordinates", help="parquet of latitude,longitude to drop (coordinate sinks)")
        p.add_argument("--col", action="append", help="column mapping, e.g. --col uid=device_id")
        p.add_argument("--distance", type=float, help="stop distance threshold (m)")
        p.add_argument("--time", type=float, help="stop time threshold (s)")
        p.add_argument("--min-duration", type=float, help="minimum stop duration (s)")
        p.add_argument("--dbscan-dist", type=float, help="DBSCAN epsilon (m)")
        p.add_argument("--min-points", type=int, help="DBSCAN min points")
        p.add_argument("--trajectories", action="store_true", help="also extract trajectories")
        p.add_argument("--cuda", action="store_true", help="use the polars GPU engine when available")
        _add_traj_args(p)
        if name == "run":
            p.add_argument("--overwrite", action="store_true")
        else:
            p.add_argument("--batch-id", help="batch identifier (e.g. the date); re-running it is a no-op")
            p.add_argument("--skip-home-work", action="store_true", help="do not refresh home/work after this batch")
    f = sub.add_parser("finalize")
    f.add_argument("output")
    f.add_argument("--cuda", action="store_true")
    t = sub.add_parser("trajectories", help="trajectories from pings + already computed stops")
    t.add_argument("pings")
    t.add_argument("stops", help="parquet with stops (stops, clusters or home_work table)")
    t.add_argument("out")
    t.add_argument("--cuda", action="store_true")
    _add_traj_args(t)
    k = sub.add_parser("sinks", help="find coordinate sinks (default geocodes, towers) in pings")
    k.add_argument("pings")
    k.add_argument("out", help="parquet to write (latitude, longitude, devices)")
    k.add_argument("--min-devices", type=int, default=5000)
    k.add_argument("--col", action="append", help="column mapping, e.g. --col uid=device_id")
    a = ap.parse_args(argv)

    from .pipeline import Pipeline

    if a.cmd == "sinks":
        from .quality import find_sinks

        cm = dict(c.split("=", 1) for c in a.col) if a.col else None
        s = find_sinks(a.pings, a.min_devices, column_map=cm)
        s.write_parquet(a.out)
        print(f"{s.height:,} sink coordinates -> {a.out}")
        return 0
    if a.cmd == "run":
        pipe = Pipeline(a.output, _config(a), trajectories=a.trajectories, use_cuda=a.cuda)
        s = pipe.run(a.input, overwrite=a.overwrite)
    elif a.cmd == "update":
        pipe = Pipeline(a.output, _config(a), trajectories=a.trajectories, use_cuda=a.cuda)
        s = pipe.update(a.input, batch_id=a.batch_id, compute_home_work=not a.skip_home_work)
    elif a.cmd == "finalize":
        s = Pipeline(a.output, use_cuda=a.cuda).finalize()
    else:
        from .trajectories import extract_trajectories

        tr = extract_trajectories(a.pings, a.stops, use_cuda=a.cuda,
                                  **{k: v for k, v in _traj_args(a).items() if v is not None})
        tr.write_parquet(a.out)
        print(f"{tr.height:,} trajectories -> {a.out}")
        return 0
    print(f"done: {s.n_stops:,} stops, {s.n_trajectories:,} trajectories in {s.seconds:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
