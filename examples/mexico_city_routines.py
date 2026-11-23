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

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shapely
from make_figures import THEMES, merc, style
from matplotlib.collections import LineCollection
from mexico_city_figures import blend

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


def lines(net, edges):
    xy, idx = shapely.get_coordinates(net._geoms[np.asarray(edges)], return_index=True)
    x, y = merc(xy[:, 0], xy[:, 1])
    return np.split(np.column_stack([x, y]), np.flatnonzero(np.diff(idx)) + 1)


def pick(c: pl.DataFrame, n: int = 3) -> pl.DataFrame:
    """Partial motorized trips with a well-supported imputation, from different routines."""
    return (c.filter((pl.col("state") == "partial") & (pl.col("mode") == "motorized") & (pl.col("n_parts") == 1)
                     & (pl.col("route_trips") >= 3) & pl.col("imputed_share").is_between(0.35, 0.75)
                     & (pl.col("imputed_on_route") >= 0.8) & pl.col("length_m").is_between(6000, 12000)
                     & (pl.col("observed_m") + pl.col("imputed_m") < 1.6 * pl.col("distance_m")))
            .sort("uid", "trip_id").sample(n=n, seed=3))


def draw_trip(ax, net, trip, edges, routes, pts, t, clip_m=1000.0):
    """One trip, without its first and last ``clip_m`` (no stays, no basemap: nobody's places)."""
    e = edges.sort("seq").with_columns((pl.col("to_m") - pl.col("from_m")).cum_sum().alias("_c"))
    total = e["_c"][-1]
    keep = e.filter((pl.col("_c") > clip_m) & (pl.col("_c") < total - clip_m))
    xy = np.concatenate(lines(net, keep["edge"]))
    pad = 350
    x0, x1, y0, y1 = xy[:, 0].min() - pad, xy[:, 0].max() + pad, xy[:, 1].min() - pad, xy[:, 1].max() + pad
    # square view
    cx, cy, h = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) / 2
    x0, x1, y0, y1 = cx - h, cx + h, cy - h, cy + h
    b = shapely.bounds(net._geoms)
    lon0, lat0 = merc_inv(x0, y0)
    lon1, lat1 = merc_inv(x1, y1)
    near = np.flatnonzero((b[:, 2] >= lon0) & (b[:, 0] <= lon1) & (b[:, 3] >= lat0) & (b[:, 1] <= lat1))
    ax.add_collection(LineCollection(lines(net, near), colors=t["grid"], linewidths=0.6, zorder=1))
    own = edges.filter(~pl.col("imputed")).select("edge").unique().with_columns(pl.lit(1).alias("_own"))
    other = (routes.filter((pl.col("routine_id") == trip["routine_id"]) & (pl.col("mode") == trip["mode"]))
             .join(own, on="edge", how="left")
             .with_columns((pl.col("trips") - pl.col("_own").fill_null(0)).alias("n")).filter(pl.col("n") > 0))
    full = np.concatenate(lines(net, e["edge"]))
    ends = np.array([full[0], full[-1]])
    if other.height:  # never near the trip's ends
        mid = shapely.get_coordinates(shapely.line_interpolate_point(net._geoms[other["edge"].to_numpy()], 0.5,
                                                                     normalized=True))
        mx, my = merc(mid[:, 0], mid[:, 1])
        other = other.filter(pl.Series(np.hypot(mx[:, None] - ends[:, 0], my[:, None] - ends[:, 1]).min(axis=1)
                                       > clip_m))
    if other.height:
        ax.add_collection(LineCollection(lines(net, other["edge"]), colors=blend(t["series"][2], t["surface"], 0.62),
                                         linewidths=4 + 2.5 * other["n"].to_numpy() / other["n"].max(), zorder=2,
                                         capstyle="butt", joinstyle="round"))
    for imp, color, ls in ((False, t["series"][0], "-"), (True, t["series"][1], (0, (2.2, 1.6)))):
        k = keep.filter(pl.col("imputed") == imp)
        if k.height:
            ax.add_collection(LineCollection(lines(net, k["edge"]), colors=color, linewidths=2.0, linestyles=ls,
                                             zorder=3, capstyle="round"))
    # the trip's pings, away from its ends (the stays)
    px, py = merc(pts["longitude"].to_numpy(), pts["latitude"].to_numpy())
    far = np.hypot(px[:, None] - ends[:, 0], py[:, None] - ends[:, 1]).min(axis=1) > clip_m
    ok = far & (px > x0) & (px < x1) & (py > y0) & (py < y1)
    ax.scatter(px[ok], py[ok], s=9, color=t["ink"], edgecolor=t["surface"], linewidth=0.6, zorder=4)
    ax.set_xlim(x0, x1), ax.set_ylim(y0, y1), ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"{trip['length_m'] / 1000:.1f} km · {trip['imputed_share']:.0%} imputed · "
                 f"{trip['route_trips']} other days", fontsize=10.5, fontweight="normal", color=t["ink2"])


def merc_inv(x, y):
    R = 6378137.0
    return np.degrees(x / R), np.degrees(2 * np.arctan(np.exp(y / R)) - np.pi / 2)


def draw_holdout(ax, ho: pl.DataFrame, t):
    h = ho.filter(pl.col("mode") == "all").sort("preference")
    x = h["preference"].to_numpy()
    for col, label, color in (("precision", "Imputed route correct (precision)", t["series"][0]),
                              ("recall", "Hidden route recovered (recall)", t["series"][1])):
        y = h[col].to_numpy() * 100
        ax.plot(x, y, color=color, lw=2, marker="o", ms=6, mec=t["surface"], mew=1.5)
        ax.annotate(label, (x[-1], y[-1]), xytext=(0, 9), textcoords="offset points", ha="right",
                    fontsize=9.5, color=t["ink2"])
        for xi, yi in ((x[0], y[0]), (x[2], y[2])):
            ax.annotate(f"{yi:.0f}%", (xi, yi), xytext=(0, -15), textcoords="offset points", ha="center",
                        fontsize=9, color=t["ink"])
    ax.set_xticks(x, [f"{v:g}" + (" (shortest path)" if v == 0 else " (default)" if v == 0.7 else "") for v in x],
                  fontsize=8.5)
    ax.set_xlim(-0.08, 0.98), ax.set_ylim(20, 85)
    ax.set_ylabel("% of route length")
    ax.set_xlabel("preference for the routine's streets")
    ax.set_title(f"Hold-out: {h['trips'][0]:,} trips, middle 60% of pings removed", fontsize=10.5,
                 fontweight="normal", color=t["ink2"])


def figures(work: Path, nets, out: str):
    c = pl.read_parquet(work / "completed_trips.parquet")
    edges = pl.read_parquet(work / "completed_edges.parquet")
    routes = pl.read_parquet(work / "routes.parquet")
    ho = pl.read_parquet(work / "holdout.parquet")
    trips = pl.read_parquet(work / "trips.parquet")
    pts = pl.scan_parquet(work / "matched" / "points" / "*.parquet")
    chosen = pick(c.join(trips.select("uid", "trip_id", "distance_m"), on=["uid", "trip_id"]))
    for name, t in THEMES.items():
        style(t)
        fig, axes = plt.subplots(2, 2, figsize=(11, 11.6))
        fig.subplots_adjust(left=0.03, right=0.98, top=0.9, bottom=0.06, wspace=0.12, hspace=0.16)
        for ax, trip in zip(axes.flat[:3], chosen.iter_rows(named=True)):
            e = edges.filter((pl.col("uid") == trip["uid"]) & (pl.col("trip_id") == trip["trip_id"]))
            tid = trips.filter((pl.col("uid") == trip["uid"]) & (pl.col("trip_id") == trip["trip_id"]))
            ids = tid["trajectory_ids"][0].to_list()
            p = pts.filter((pl.col("uid") == trip["uid"]) & pl.col("trajectory_id").is_in(ids)).collect()
            draw_trip(ax, nets["motorized"], trip, e, routes, p, t)
        draw_holdout(axes.flat[3], ho, t)
        fig.suptitle("Completing partial trips along the device's usual streets", x=0.03, ha="left",
                     fontsize=13, fontweight="bold", color=t["ink"], y=0.985)
        handles = [plt.Line2D([], [], color=t["series"][0], lw=2, label="observed route"),
                   plt.Line2D([], [], color=t["series"][1], lw=2, ls=(0, (2.2, 1.6)), label="imputed route"),
                   plt.Line2D([], [], color=blend(t["series"][2], t["surface"], 0.62), lw=6, label="streets seen on other days"),
                   plt.Line2D([], [], color=t["ink"], marker="o", ls="", ms=4, label="pings")]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.025, 0.955), ncol=4, fontsize=9.5)
        fig.text(0.98, 0.012, "Three motorized trips from the Mexico City sample, without their first and last km. "
                 "Streets © OpenStreetMap contributors (ODbL)", ha="right", fontsize=8, color=t["muted"])
        fig.savefig(f"{out}/cdmx_routines_{name}.png", dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--osm", required=True)
    ap.add_argument("--sample", required=True, help="full-resolution trajectories (mexico_city_matching.py)")
    ap.add_argument("--work", default="examples/output/routines")
    ap.add_argument("--out", default="docs/assets/figures")
    ap.add_argument("--figures-only", action="store_true", help="redraw the figure from the saved results")
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    stats = {}
    if a.figures_only:
        nets = {"motorized": pr.RoadNetwork(pl.read_parquet(f"{a.osm}/drive.parquet"), directed=True)}
        figures(work, nets, a.out)
        return

    t0 = time.time()
    tr = pl.read_parquet(a.sample)
    uids = tr.select("uid").unique()
    stays = (pl.scan_parquet(glob.glob(f"{a.output_dir}/clusters/**/*.parquet", recursive=True))
             .join(uids.lazy(), on="uid", how="semi").collect())
    trips = pr.find_routines(pr.stay_trips(stays, tr, timezone=TZ))
    tm = pr.routine_modes(tr, trips)
    trips.write_parquet(work / "trips.parquet")
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
    figures(work, nets, a.out)


if __name__ == "__main__":
    main()
