"""Command line interface.

    polaroam run      INPUT OUTPUT [options]   # everything at once
    polaroam update   INPUT OUTPUT [options]   # one incremental batch (e.g. a day)
    polaroam finalize OUTPUT                   # close open stays after the last batch
    polaroam trajectories PINGS STOPS OUT.parquet
    polaroam match    TRAJECTORIES OUT_DIR --network ROADS.parquet  # snap trajectories to roads
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

from .config import PipelineConfig


def _config(a: argparse.Namespace) -> PipelineConfig | None:
    if a.config:
        cfg = PipelineConfig.from_json(a.config, a.label)
    elif any(v is not None for v in (a.timezone, a.find_timezones, a.tz_polygons, a.shards, a.distance, a.time, a.min_duration,
                                     a.drop_coordinates, a.remove_sinks, a.sinks_min_devices,
                                     a.dbscan_dist, a.min_points, a.timestamp_unit, a.timestamp_timezone, *_traj_args(a).values())) or a.col:
        cfg = PipelineConfig()
    else:
        return None
    over = {}
    if a.timezone:
        over["timezone"] = a.timezone
    if a.find_timezones is not None and a.tz_polygons is not None:
        raise SystemExit("use --find-timezones, not both it and --tz-polygons")
    find = a.find_timezones if a.find_timezones is not None else a.tz_polygons
    if find is not None:
        over["find_timezones"] = True if find in (True, "builtin") else find
    if a.shards:
        over["n_shards"] = a.shards
    sinks = _sink_option(a)
    if sinks is not None:
        over["remove_sinks"] = sinks
    if a.sinks_min_devices is not None:
        over["sinks_min_devices"] = a.sinks_min_devices
    if a.col:
        over["column_map"] = _column_map(a.col)
    if a.timestamp_unit:
        over["timestamp_unit"] = a.timestamp_unit
    if a.timestamp_timezone:
        over["timestamp_timezone"] = a.timestamp_timezone
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


def _column_map(items: list[str] | None) -> dict[str, str | list[str]] | None:
    """``["uid=device_id", "latitude=lat,y"]`` -> ``{"uid": "device_id", "latitude": ["lat", "y"]}``."""
    if not items:
        return None
    out: dict[str, str | list[str]] = {}
    for item in items:
        std, _, cols = item.partition("=")
        names = [c.strip() for c in cols.split(",") if c.strip()]
        out[std.strip()] = names[0] if len(names) == 1 else names
    return out


def _add_sink_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("coordinate sinks")
    g.add_argument("--remove-sinks", nargs="?", const=True, metavar="FILE",
                   help="drop pings at coordinate sinks: alone, find them in the input; "
                        "with FILE, use a parquet of latitude,longitude (e.g. from `polaroam sinks`)")
    g.add_argument("--sinks-min-devices", type=int,
                   help="devices on one exact coordinate to call it a sink (default 5000)")
    g.add_argument("--drop-coordinates", metavar="FILE", help=argparse.SUPPRESS)  # old name of --remove-sinks FILE


def _sink_option(a: argparse.Namespace) -> bool | str | None:
    if a.remove_sinks is not None and a.drop_coordinates is not None:
        raise SystemExit("use --remove-sinks FILE or --drop-coordinates FILE, not both")
    return a.remove_sinks if a.remove_sinks is not None else a.drop_coordinates


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
        p.add_argument("--timezone", help="timezone of the data, e.g. America/Bogota (required unless "
                                          "--find-timezones; then the fallback)")
        p.add_argument("--find-timezones", nargs="?", const=True, metavar="CSV",
                       help="look up each stay's timezone: alone, in the bundled polygons; with CSV, in a "
                            "tz,geometry (WKT) file. Only for data spanning several timezones")
        p.add_argument("--tz-polygons", help=argparse.SUPPRESS)  # old name of --find-timezones
        p.add_argument("--shards", type=int, help="split devices into N groups (bounded memory)")
        p.add_argument("--col", action="append",
                       help="column mapping, repeatable: --col uid=device_id, or candidates --col latitude=lat,y")
        p.add_argument("--timestamp-unit", choices=["auto", "s", "ms", "us", "ns"],
                       help="Unix time unit of numeric timestamps (default: auto, from each value)")
        p.add_argument("--timestamp-timezone", help="timezone of date-times without an offset (default UTC)")
        p.add_argument("--distance", type=float, help="stop distance threshold (m)")
        p.add_argument("--time", type=float, help="stop time threshold (s)")
        p.add_argument("--min-duration", type=float, help="minimum stop duration (s)")
        p.add_argument("--dbscan-dist", type=float, help="DBSCAN epsilon (m)")
        p.add_argument("--min-points", type=int, help="DBSCAN min points")
        p.add_argument("--trajectories", action="store_true", help="also extract trajectories")
        p.add_argument("--cuda", action="store_true", help="use the polars GPU engine when available")
        _add_sink_args(p)
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
    _add_sink_args(t)
    _add_traj_args(t)
    k = sub.add_parser("sinks", help="find coordinate sinks (default geocodes, towers) in pings")
    k.add_argument("pings")
    k.add_argument("out", help="parquet to write (latitude, longitude, devices)")
    k.add_argument("--min-devices", type=int, default=5000)
    k.add_argument("--col", action="append", help="column mapping, e.g. --col uid=device_id")
    mt = sub.add_parser("match", help="snap trajectories to a road network (map matching)")
    mt.add_argument("trajectories", help="parquet of trajectories (keep every ping: --tolerance 0)")
    mt.add_argument("out", help="output directory: trajectories, points and edges parquet")
    mt.add_argument("--network", action="append", required=True, metavar="[MODE=]ROADS",
                    help="road table (parquet/csv, WKB or WKT geometry); per mode: --network walk=walk.parquet "
                         "--network motorized=drive.parquet")
    mt.add_argument("--by", default="mode_guess", help="column choosing the network (default mode_guess)")
    mt.add_argument("--directed", action="store_true", help="road rows are one direction each (osmnx edges)")
    for name, kind in (("search-radius", float), ("max-candidates", int), ("sigma", float), ("beta", float),
                       ("route-factor", float), ("max-speed", float)):
        mt.add_argument(f"--{name}", type=kind)
    a = ap.parse_args(argv)

    from .local_time import TimezoneRequired
    from .pipeline import Pipeline

    try:
        return _run(a, Pipeline)
    except TimezoneRequired as e:
        print(f"polaroam: {e}", file=sys.stderr)
        return 2


def _run(a: argparse.Namespace, Pipeline) -> int:
    if a.cmd == "match":
        return _match(a)
    if a.cmd == "sinks":
        from .quality import find_sinks

        cm = _column_map(a.col)
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

        tr = extract_trajectories(a.pings, a.stops, use_cuda=a.cuda, remove_sinks=_sink_option(a) or False,
                                  sinks_min_devices=a.sinks_min_devices or 5000,
                                  **{k: v for k, v in _traj_args(a).items() if v is not None})
        tr.write_parquet(a.out)
        print(f"{tr.height:,} trajectories -> {a.out}")
        return 0
    print(f"done: {s.n_stops:,} stops, {s.n_trajectories:,} trajectories in {s.seconds:.1f}s")
    return 0



def _match(a: argparse.Namespace) -> int:
    from pathlib import Path

    import polars as pl

    from .roads import MatchParams, RoadNetwork, match_trajectories

    nets = {}
    for item in a.network:
        mode, sep, path = item.rpartition("=")
        nets[mode if sep else None] = RoadNetwork.read(path, directed=a.directed)
    network = nets[None] if list(nets) == [None] else {k: v for k, v in nets.items() if k is not None}
    over = {k: v for k, v in (("search_radius", a.search_radius), ("max_candidates", a.max_candidates),
                              ("sigma", a.sigma), ("beta", a.beta), ("route_factor", a.route_factor),
                              ("max_speed_kmh", a.max_speed)) if v is not None}
    out = Path(a.out)
    res = match_trajectories(pl.scan_parquet(a.trajectories), network, MatchParams(**over), by=a.by, output_dir=out)
    t = res.trajectories.select(pl.len(), (pl.col("n_parts") == 1).sum()).collect().row(0)
    print(f"{t[0]:,} trajectories matched ({t[1]:,} in one piece) -> {out}/{{trajectories,points,edges}}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
