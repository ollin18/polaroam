# Command line

Installing the package provides a `polaroam` command (also `python -m polaroam.cli`).

```bash
polaroam run      INPUT OUTPUT [options]     # everything at once
polaroam update   INPUT OUTPUT [options]     # one incremental batch (e.g. a day)
polaroam finalize OUTPUT                     # close open stays after the last batch
polaroam sinks    INPUT SINKS.parquet        # find coordinate sinks (default geocodes, towers)
polaroam trajectories PINGS STOPS OUT.parquet [options]
```

## Examples

```bash
# gc-lbs style flags
polaroam run "raw/CO/*.parquet" out/CO --timezone America/Bogota --trajectories \
    --col uid=device_id --distance 20 --time 3600 --min-duration 300 --dbscan-dist 20 --min-points 2

# reuse a gc-lbs pipeline_config.json (country section), on the GPU, in 24 shards
polaroam run raw/MX out/MX --config pipeline_config.json --label MX --shards 24 --cuda

# drop coordinate sinks (default geocodes, towers) found in the input
polaroam run "raw/MX/*.parquet" out/MX --config pipeline_config.json --label MX --remove-sinks

# day by day
polaroam update "raw/MX/date=2022-12-01/*.parquet" out/MX --timezone America/Mexico_City \
    --trajectories --batch-id 2022-12-01
polaroam finalize out/MX

# trajectories from stays computed earlier: full resolution, no micro-trips, time-aware WKT
polaroam trajectories "raw/MX/**/*.parquet" "out/MX/stops/**/*.parquet" trips.parquet \
    --tolerance 0 --min-pings 2 --min-length 100 --wkt-measure
```

## Options

| Option | Applies to | Meaning |
|---|---|---|
| `--config FILE`, `--label L` | run, update | load a `pipeline_config.json` (gc-lbs format accepted) |
| `--timezone TZ` | run, update | timezone of the data (required unless `--find-timezones`; then the fallback) |
| `--find-timezones [CSV]` | run, update | look up each stay's timezone: alone, in the bundled polygons; with `CSV`, in that file |
| `--shards N` | run, update | split devices into N groups |
| `--remove-sinks [FILE]` | run, update, trajectories | drop pings at coordinate sinks first: alone, find them in the input (saved to `OUTPUT/sinks.parquet`); with `FILE`, use that list |
| `--sinks-min-devices N` | run, update, trajectories | devices on one exact coordinate to call it a sink (default 5000) |
| `--min-devices N` | sinks | devices sharing an exact coordinate to call it a sink (default 5000) |
| `--col std=yours[,other]` | run, update, sinks | column mapping, repeatable (`--col uid=aid`, `--col latitude=lat,y`) |
| `--timestamp-unit U` | run, update | `s`, `ms`, `us`, `ns` or `auto` (default) |
| `--timestamp-timezone TZ` | run, update | how to read date-times without a timezone (default UTC) |
| `--distance`, `--time`, `--min-duration` | run, update | stay parameters (m, s, s) |
| `--dbscan-dist`, `--min-points` | run, update | DBSCAN parameters |
| `--trajectories` | run, update | also extract trajectories |
| `--max-gap`, `--tolerance`, `--min-pings`, `--min-length` | all | trajectory parameters |
| `--speed-window`, `--max-speed`, `--wkt-measure` | all | speed features and WKT |
| `--cuda` | all | use the GPU engine when available |
| `--overwrite` | run | replace an existing output directory |
| `--batch-id ID` | update | batch name (e.g. the date); re-running it is a no-op |
| `--skip-home-work` | update | do not refresh home/work after this batch |
