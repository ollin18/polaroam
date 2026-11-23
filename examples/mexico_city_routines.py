"""Routines example for the "Mexico City at scale" docs page.

    python examples/mexico_city_routines.py /data/out/MX_2022_12 --osm /data/osm/cdmx \\
        --sample examples/output/matching/sample_60000.parquet

Uses the full-resolution trajectories of the map-matching example's device sample
(examples/mexico_city_matching.py) and the month's clustered stays:

1. Trips between consecutive stays, routines (same two places on >= 3 days), routine modes.
2. Routine trips matched (each on its mode's network) and completed from stay to stay.
3. A hold-out test: one complete trip per routine loses the middle of its pings and is
   completed again, with and without the routine's preference; the hidden stretch is compared
   with the route matched from all the pings.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import polars as pl

import polaroam as pr
from polaroam.geo import haversine

NETWORKS = {"walk": "walk", "bike": "bike", "motorized": "drive"}
TZ = "America/Mexico_City"


def thin(tr: pl.DataFrame, keep: float = 0.2) -> pl.DataFrame:
    """Drop the pings between the first and last ``keep`` of each trajectory's length."""
    p = (tr.select("uid", "trajectory_id", "coordinates", "timestamps").explode("coordinates", "timestamps")
         .with_columns(pl.col("coordinates").list.get(0).alias("x"), pl.col("coordinates").list.get(1).alias("y"))
         .with_columns(haversine("y", "x", pl.col("y").shift(1).over("uid", "trajectory_id"),
                                 pl.col("x").shift(1).over("uid", "trajectory_id")).fill_null(0.0).alias("_d"))
         .with_columns((pl.col("_d").cum_sum().over("uid", "trajectory_id")
                        / pl.col("_d").sum().over("uid", "trajectory_id")).alias("_f"))
         .filter((pl.col("_f") <= keep) | (pl.col("_f") >= 1 - keep))
         .group_by("uid", "trajectory_id", maintain_order=True).agg("coordinates", "timestamps"))
    return tr.drop("coordinates", "timestamps").join(p, on=["uid", "trajectory_id"])


def overlap(pred: pl.DataFrame, truth: pl.DataFrame) -> pl.DataFrame:
    """Length-weighted agreement of imputed edges with the true hidden edges, per trip."""
    key = ["uid", "trip_id"]
    pe = pred.group_by(*key, "edge").agg((pl.col("to_m") - pl.col("from_m")).sum().alias("p"))
    te = truth.group_by(*key, "edge").agg((pl.col("to_m") - pl.col("from_m")).sum().alias("t"))
    j = pe.join(te, on=[*key, "edge"], how="full", coalesce=True).with_columns(pl.col("p", "t").fill_null(0.0))
    return j.group_by(key).agg(pl.min_horizontal("p", "t").sum().alias("hit"), pl.col("p").sum().alias("p"),
                               pl.col("t").sum().alias("t"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--osm", required=True)
    ap.add_argument("--sample", required=True, help="full-resolution trajectories (mexico_city_matching.py)")
    ap.add_argument("--work", default="examples/output/routines")
    ap.add_argument("--out", default="docs/assets/figures")
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    stats = {}

    t0 = time.time()
    tr = pl.read_parquet(a.sample)
    uids = tr.select("uid").unique()
    stays = (pl.scan_parquet(glob.glob(f"{a.output_dir}/clusters/**/*.parquet", recursive=True))
             .join(uids.lazy(), on="uid", how="semi").collect())
    trips = pr.find_routines(pr.stay_trips(stays, tr, timezone=TZ))
    tm = pr.routine_modes(tr, trips)
    print(f"{trips.height:,} trips from {uids.height:,} devices ({time.time() - t0:.0f}s)")
    inr = trips.filter(pl.col("routine_id").is_not_null())
    stats["trips"] = trips.height
    stats["state"] = dict(trips.group_by("state").len().rows())
    stats["in_routine"] = inr.height
    stats["routines"] = inr["routine_id"].n_unique()
    stats["routine_devices"] = inr["uid"].n_unique()
    stats["routine_state"] = dict(inr.group_by("state").len().rows())
    stats["mode_source"] = dict(trips.group_by("mode_source").len().rows())
    stats["trajectory_mode_source"] = dict(tm.group_by("mode_source").len().rows())
    stats["mode_from_routine"] = dict(trips.filter(pl.col("mode_source") == "routine").group_by("mode").len().rows())
    print(json.dumps(stats, indent=1, default=str))

    nets = {m: pr.RoadNetwork(pl.read_parquet(f"{a.osm}/{k}.parquet"), directed=True) for m, k in NETWORKS.items()}
    ids = inr.select("uid", pl.col("trajectory_ids").alias("trajectory_id")).explode("trajectory_id").drop_nulls()
    rt = tm.join(ids, on=["uid", "trajectory_id"], how="semi")
    t0 = time.time()
    res = pr.match_trajectories(rt, nets, by="mode", output_dir=work / "matched")
    points = res.points.collect()
    print(f"matched {rt.height:,} routine trajectories ({time.time() - t0:.0f}s)")

    t0 = time.time()
    done = pr.complete_trips(trips, points, nets, fill_empty=True)
    print(f"completed {done.trips.height:,} trips ({time.time() - t0:.0f}s)")
    done.trips.write_parquet(work / "completed_trips.parquet")
    done.edges.write_parquet(work / "completed_edges.parquet")
    done.routes.write_parquet(work / "routes.parquet")
    c = done.trips
    stats["completed"] = {s: {"trips": int(g.height), "one_part": float((g["n_parts"] == 1).mean()),
                              "imputed_share_median": float(g["imputed_share"].median()),
                              "imputed_on_route_median": float(g["imputed_on_route"].drop_nans().median() or 0),
                              "with_route_trips": float((g["route_trips"] > 0).mean())}
                          for (s,), g in c.group_by("state")}

    # ---- hold-out: one complete trip per routine with other observed trips
    t0 = time.time()
    cand = (c.filter((pl.col("state") == "complete") & (pl.col("route_trips") >= 2) & (pl.col("n_parts") == 1)
                     & (pl.col("length_m") >= 1000))
            .join(trips.select("uid", "trip_id", "trajectory_ids", "mode"), on=["uid", "trip_id"])
            .filter(pl.col("trajectory_ids").list.len() == 1)
            .sample(fraction=1.0, shuffle=True, seed=1).unique("routine_id", keep="first"))
    test_ids = cand.select("uid", pl.col("trajectory_ids").list.first().alias("trajectory_id"))
    thinned = thin(rt.join(test_ids, on=["uid", "trajectory_id"], how="semi"))
    rm = pr.match_trajectories(thinned, nets, by="mode").points
    pts_ho = pl.concat([points.join(test_ids, on=["uid", "trajectory_id"], how="anti"), rm.select(points.columns)])
    truth = done.edges.join(cand.select("uid", "trip_id"), on=["uid", "trip_id"], how="semi")
    rows = []
    for pref in (0.0, 0.4, 0.7, 0.9):
        ho = pr.complete_trips(trips.join(cand.select("routine_id"), on="routine_id", how="semi"), pts_ho, nets,
                               preference=pref)
        e = ho.edges.join(cand.select("uid", "trip_id"), on=["uid", "trip_id"], how="semi")
        observed = e.filter(~pl.col("imputed")).select("uid", "trip_id", "edge").unique()
        hidden = truth.join(observed, on=["uid", "trip_id", "edge"], how="anti")
        o = overlap(e.filter(pl.col("imputed")), hidden).join(cand.select("uid", "trip_id", "mode"), on=["uid", "trip_id"])
        for mode, g in [("all", o), *[(m, o.filter(pl.col("mode") == m)) for m in NETWORKS]]:
            rows.append({"preference": pref, "mode": mode, "trips": g.height,
                         "recall": g["hit"].sum() / g["t"].sum(), "precision": g["hit"].sum() / g["p"].sum(),
                         "recall_median": (g["hit"] / g["t"]).median()})
    ho_tab = pl.DataFrame(rows)
    print(f"hold-out on {cand.height:,} trips ({time.time() - t0:.0f}s)")
    with pl.Config(tbl_rows=30):
        print(ho_tab)
    ho_tab.write_parquet(work / "holdout.parquet")
    stats["holdout"] = ho_tab.to_dicts()
    (work / "stats.json").write_text(json.dumps(stats, indent=1, default=str))


if __name__ == "__main__":
    main()
