"""Map matching example for the "Mexico City at scale" docs page.

    python examples/download_osm.py /data/osm/cdmx                     # OSM: drive, bike, walk
    python examples/mexico_city_matching.py /data/out/MX_2022_12 --osm /data/osm/cdmx \\
        --pings "/data/Berkeley/MX/data_quadrant/filtered_localized/*/*.parquet" --sinks mx_sinks.parquet

1. A random sample of devices; their trajectories re-extracted at full resolution
   (``simplify_tolerance=0``) from the month's stays.
2. Each mode matched on its own OpenStreetMap network: walk, bike, drive (motorized).
3. Trajectories per street segment, per mode: an aggregate over thousands of devices.
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shapely
from make_figures import THEMES, merc, style
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from mexico_city_figures import MAP_BG, MAP_LINE, blend, extent, grid_mask, inside, outline, vertex_density

import polaroam as pr

NETWORKS = {"walk": "walk", "bike": "bike", "motorized": "drive"}


def sample_trajectories(out: str, pings: str, sinks: str | None, n_devices: int, path: Path) -> pl.DataFrame:
    if path.exists():
        return pl.read_parquet(path)
    traj = pl.scan_parquet(glob.glob(f"{out}/trajectories/**/shard=*.parquet", recursive=True))
    uids = traj.select("uid").unique().collect().sample(n=n_devices, seed=1)["uid"].implode()
    stops = pl.scan_parquet(glob.glob(f"{out}/stops/**/shard=*.parquet", recursive=True))
    t = time.time()
    tr = pr.extract_trajectories(pr.read_pings(pings).filter(pl.col("uid").is_in(uids)),
                                 stops.filter(pl.col("uid").is_in(uids)).collect(),
                                 simplify_tolerance=0, min_pings=2, min_length_m=100, remove_sinks=sinks or False)
    tr.write_parquet(path)
    print(f"{tr.height:,} full-resolution trajectories from {n_devices:,} devices ({time.time() - t:.0f}s)")
    return tr


def summary(trajs: pl.DataFrame, tr: pl.DataFrame) -> pl.DataFrame:
    m = trajs.join(tr.select("uid", "trajectory_id", "length_m"), on=["uid", "trajectory_id"])
    return (m.group_by("mode_guess").agg(
        pl.len().alias("trajectories"),
        (pl.col("n_matched").sum() / pl.col("n_points").sum()).alias("points_matched"),
        (pl.col("n_parts") == 1).mean().alias("one_part"),
        (pl.col("n_parts") == 0).mean().alias("no_road"),
        pl.col("mean_distance_m").median().alias("median_distance_m"),
        (pl.col("matched_length_m").sum() / 1000).alias("matched_km"))
        .sort("trajectories", descending=True))


def undirected_volumes(vol: pl.DataFrame, net: pr.RoadNetwork) -> pl.DataFrame:
    """Both directions of a street together: trajectories per street segment."""
    e = net.edges.select("edge", "u", "v")
    return (vol.join(e, on="edge")
            .with_columns(pl.min_horizontal("u", "v").alias("a"), pl.max_horizontal("u", "v").alias("b"))
            .group_by("a", "b").agg(pl.col("trajectories").sum(), pl.col("edge").first()))


def segments(net: pr.RoadNetwork, edges: np.ndarray):
    g = net._geoms[edges]
    xy, idx = shapely.get_coordinates(g, return_index=True)
    x, y = merc(xy[:, 0], xy[:, 1])
    cut = np.flatnonzero(np.diff(idx)) + 1
    return np.split(np.column_stack([x, y]), cut)


def bare(ax, t):
    x0, x1, y0, y1 = extent()
    ax.set_xlim(x0, x1), ax.set_ylim(y0, y1), ax.set_aspect("equal")
    outline(ax, t, lw=0.8, color=MAP_LINE)
    ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def ramp(mode, color):
    """Quiet in a dark shade of the mode's hue, the busiest in white."""
    return LinearSegmentedColormap.from_list(mode, [
        (0.0, blend(color, MAP_BG, 0.7)), (0.45, color), (0.82, blend(color, "#ffffff", 0.55)), (1.0, "#ffffff")])


def draw_matched(ax, net, v, mode, color, t):
    """Trajectories per street segment, on the faint network."""
    ax.set_facecolor(MAP_BG)
    mid = shapely.get_coordinates(shapely.line_interpolate_point(net._geoms, 0.5, normalized=True))
    base = np.flatnonzero(inside(mid[:, 0], mid[:, 1]))
    ax.add_collection(LineCollection(segments(net, base), colors="#34342f", linewidths=0.15))
    v = v.filter(pl.Series(inside(*shapely.get_coordinates(shapely.line_interpolate_point(
        net._geoms[v["edge"].to_numpy()], 0.5, normalized=True)).T))).sort("trajectories")
    n = v["trajectories"].to_numpy().astype(float)
    vmax = max(n.max(), 2)
    f = np.clip(np.log(n) / np.log(vmax), 0, 1)
    ax.add_collection(LineCollection(segments(net, v["edge"].to_numpy()),
                                     colors=ramp(mode, color)(LogNorm(vmin=1, vmax=vmax)(n)),
                                     linewidths=0.12 + 1.1 * f ** 2))
    bare(ax, t)


def draw_density(ax, coords, mode, color, t, bins=900):
    """Heat map of the observed positions (trajectory vertices): the same colors, bins and scale as
    the trajectory heat maps of mexico_city_figures.fig_mode_maps."""
    h = vertex_density(coords, bins)
    x0, x1, y0, y1 = extent()
    cmap = LinearSegmentedColormap.from_list(
        mode, [MAP_BG, blend(color, MAP_BG, 0.55), color, blend(color, "#ffffff", 0.6), "#ffffff"])
    ax.set_facecolor(MAP_BG)
    mm = np.ma.masked_where((h < 1) | ~grid_mask(h.shape), h)
    ax.imshow(mm, extent=(x0, x1, y0, y1), origin="lower", cmap=cmap, interpolation="antialiased",
              norm=LogNorm(vmin=1, vmax=max(np.percentile(h[h > 0], 99), 2)))
    bare(ax, t)


def figure(rows, t):
    x0, x1, y0, y1 = extent()
    width, top, bottom, gap = 16.0, 1 - 0.11 / rows, 0.035 / rows, 0.012
    panel_w = width * (1 - 2 * gap) / 3
    height = rows * panel_w * (y1 - y0) / (x1 - x0) / (top - bottom) * 1.04
    fig, axes = plt.subplots(rows, 3, figsize=(width, height), squeeze=False)
    fig.subplots_adjust(left=0.004, right=0.996, wspace=0.03, hspace=0.07, top=top, bottom=bottom)
    return fig, axes


def fig_matched(nets, vols, counts, t, path):
    fig, axes = figure(1, t)
    for ax, mode, color in zip(axes[0], NETWORKS, t["series"]):
        draw_matched(ax, nets[mode], vols[mode], mode, color, t)
        ax.set_title(f"{mode} · {counts[mode]:,} trajectories", color=t["ink"], fontsize=12, loc="left")
    fig.suptitle("Trajectories matched to the streets, each mode on its OpenStreetMap network "
                 "(walk, bike, drive): trajectories per street segment, log scale",
                 x=0.006, y=0.985, ha="left", fontsize=12.5, fontweight="bold", color=t["ink"])
    fig.text(0.995, 0.005, "Streets © OpenStreetMap contributors (ODbL)", ha="right", va="bottom",
             fontsize=8, color=t["muted"])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_compare(tr, nets, vols, counts, t, path):
    """Same trajectories: heat map of their raw positions (top) vs matched streets (bottom)."""
    fig, axes = figure(2, t)
    for j, (mode, color) in enumerate(zip(NETWORKS, t["series"])):
        draw_density(axes[0, j], tr.filter(pl.col("mode_guess") == mode)["coordinates"], mode, color, t)
        axes[0, j].set_title(f"{mode} · heat map of the raw positions", color=t["ink"], fontsize=12, loc="left")
        draw_matched(axes[1, j], nets[mode], vols[mode], mode, color, t)
        axes[1, j].set_title(f"{mode} · matched to the streets", color=t["ink"], fontsize=12, loc="left")
    fig.suptitle("Before and after map matching: the same trajectories (a random 4% of the devices), "
                 "log scale", x=0.006, y=0.993, ha="left", fontsize=12.5, fontweight="bold", color=t["ink"])
    fig.text(0.995, 0.003, "Streets © OpenStreetMap contributors (ODbL)", ha="right", va="bottom",
             fontsize=8, color=t["muted"])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--osm", required=True, help="directory with drive/bike/walk.parquet (examples/download_osm.py)")
    ap.add_argument("--pings", required=True)
    ap.add_argument("--sinks")
    ap.add_argument("--devices", type=int, default=60_000)
    ap.add_argument("--work", default="examples/output/matching")
    ap.add_argument("--out", default="docs/assets/figures")
    ap.add_argument("--reuse", action="store_true", help="reuse saved matches (redraw the figure only)")
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    tr = sample_trajectories(a.output_dir, a.pings, a.sinks, a.devices, work / f"sample_{a.devices}.parquet")
    nets = {m: pr.RoadNetwork(pl.read_parquet(f"{a.osm}/{k}.parquet"), directed=True) for m, k in NETWORKS.items()}
    for m, net in nets.items():
        print(m, net)
    matched = work / "matched"
    if not (a.reuse and (matched / "edges").exists()):
        t0 = time.time()
        # each chunk is written as it is matched: memory stays flat however large the results
        pr.match_trajectories(tr, nets, output_dir=matched)
        print(f"matched in {time.time() - t0:.0f}s", flush=True)
    trajs = pl.scan_parquet(matched / "trajectories" / "*.parquet").drop("geometry").collect()
    edges = pl.scan_parquet(matched / "edges" / "*.parquet")
    print(f"{trajs.height:,} trajectories, {int(trajs['n_points'].sum()):,} points")
    with pl.Config(tbl_cols=10, tbl_width_chars=160):
        print(summary(trajs, tr))
    vols, counts = {}, {}
    for m, net in nets.items():
        vols[m] = undirected_volumes(pr.edge_volumes(edges.filter(pl.col("mode_guess") == m), min_fraction=0.5), net)
        counts[m] = trajs.filter((pl.col("mode_guess") == m) & (pl.col("n_parts") > 0)).height
    style(THEMES["light"])  # one image for both site themes: white page, dark map panel
    fig_matched(nets, vols, counts, THEMES["light"], f"{a.out}/cdmx_matched.png")
    fig_compare(tr, nets, vols, counts, THEMES["light"], f"{a.out}/cdmx_matched_vs_raw.png")


if __name__ == "__main__":
    main()
