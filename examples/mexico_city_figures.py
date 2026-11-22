"""Figures for the "Mexico City at scale" docs page.

    python examples/mexico_city_figures.py /data/tmp/polaroam_quadrant/oneshot_cpu

Reads a polaroam output directory (one month, 1.66B pings) shard by shard and draws
aggregated maps only: densities over hundreds of thousands of devices.
"""

from __future__ import annotations

import argparse
import glob

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from make_figures import ATTRIBUTION, THEMES, basemap, fig_mode_features, merc, style
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

import polaroam as pr

LAT, LON = (19.20, 19.60), (-99.34, -98.94)


def extent():
    x, y = merc(np.array(LON), np.array(LAT))
    return x[0], x[1], y[0], y[1]


def hist2d(lat, lon, bins):
    x, y = merc(lon, lat)
    x0, x1, y0, y1 = extent()
    h, _, _ = np.histogram2d(y, x, bins=bins, range=[[y0, y1], [x0, x1]])
    return h


def load(out):
    bins_pts, bins_flow = 260, 900
    stays = np.zeros((bins_pts, bins_pts))
    home = np.zeros_like(stays)
    work = np.zeros_like(stays)
    flow = np.zeros((bins_flow, bins_flow))
    n = {"stays": 0, "home": 0, "work": 0, "trajectories": 0, "vertices": 0, "devices": 0}
    for f in sorted(glob.glob(f"{out}/clusters/shard=*.parquet")):
        c = pl.read_parquet(f, columns=["uid", "stop_latitude", "stop_longitude"])
        stays += hist2d(c["stop_latitude"].to_numpy(), c["stop_longitude"].to_numpy(), bins_pts)
        n["stays"] += c.height
        n["devices"] += c["uid"].n_unique()
    for f in sorted(glob.glob(f"{out}/locations/shard=*.parquet")):
        loc = pl.read_parquet(f)
        h = loc.drop_nulls("home_latitude")
        w = loc.drop_nulls("work_latitude")
        home += hist2d(h["home_latitude"].to_numpy(), h["home_longitude"].to_numpy(), bins_pts)
        work += hist2d(w["work_latitude"].to_numpy(), w["work_longitude"].to_numpy(), bins_pts)
        n["home"] += h.height
        n["work"] += w.height
    for f in sorted(glob.glob(f"{out}/trajectories/**/shard=*.parquet", recursive=True)):
        v = pl.read_parquet(f, columns=["coordinates"]).explode("coordinates").drop_nulls()
        xy = np.asarray(v["coordinates"].to_list())
        flow += hist2d(xy[:, 1], xy[:, 0], bins_flow)
        n["trajectories"] += pl.read_parquet(f, columns=["uid"]).height
        n["vertices"] += len(xy)
    return stays, home, work, flow, n


def fig_stays(stays, home, work, n, t, path):
    cmap = LinearSegmentedColormap.from_list("seq", t["seq"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    fig.subplots_adjust(left=0.005, right=0.95, wspace=0.03, top=0.86, bottom=0.05)
    x0, x1, y0, y1 = extent()
    panels = [(stays, f"All stays ({n['stays'] / 1e6:.1f}M)"), (home, f"Homes ({n['home']:,})"),
              (work, f"Work places ({n['work']:,})")]
    for ax, (h, title) in zip(axes, panels):
        ax.set_xlim(x0, x1), ax.set_ylim(y0, y1)
        basemap(ax, t)
        m = np.ma.masked_where(h < 1, h)
        im = ax.imshow(m, extent=(x0, x1, y0, y1), origin="lower", cmap=cmap, alpha=0.88, zorder=2,
                       norm=LogNorm(vmin=1, vmax=max(h.max(), 2)), interpolation="nearest")
        ax.set_title(title, color=t["ink"]), ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01, shrink=0.8)
        cb.outline.set_visible(False)
    fig.suptitle(f"Mexico City, December 2022: {n['devices'] / 1e6:.2f}M devices "
                 "(count per ~150 m cell, log scale)", x=0.008, y=0.98, ha="left", fontsize=13,
                 fontweight="bold", color=t["ink"])
    fig.text(0.008, 0.012, ATTRIBUTION, fontsize=7, color=t["muted"])
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_flows(flow, n, t, path):
    x0, x1, y0, y1 = extent()
    # from the surface color up the sequential ramp (light: toward dark blue, dark: toward light blue)
    cmap = LinearSegmentedColormap.from_list("flow", [t["surface"], *t["seq"]])
    fig, ax = plt.subplots(figsize=(9, 9))
    m = np.ma.masked_where(flow < 1, flow)
    ax.set_facecolor(t["surface"])
    ax.imshow(m, extent=(x0, x1, y0, y1), origin="lower", cmap=cmap,
              norm=LogNorm(vmin=1, vmax=np.percentile(flow[flow > 0], 99.8)), interpolation="antialiased")
    ax.set_xlim(x0, x1), ax.set_ylim(y0, y1)
    ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{n['trajectories'] / 1e6:.1f}M trajectories, {n['vertices'] / 1e6:.0f}M simplified vertices",
                 color=t["ink"], fontsize=12)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


ZOOM_LAT, ZOOM_LON = (19.400, 19.428), (-99.185, -99.150)   # Roma - Condesa - Juárez, ~3.5 km


def hexes(cells: pl.DataFrame):
    """Mercator polygons for H3 cells (boundaries from polars-h3)."""
    import polars_h3 as plh3

    b = cells.select(plh3.cell_to_boundary("h3_cell").alias("b"))["b"].to_list()
    polys = []
    for ring in b:
        r = np.asarray(ring)
        x, y = merc(r[:, 1], r[:, 0])
        polys.append(np.column_stack([x, y]))
    return polys


def count_ticks(cb, vmin, vmax):
    """Plain round-number ticks on a log colorbar."""
    ticks = [v for v in (5, 10, 20, 50, 100, 200, 500, 1000) if vmin <= v <= vmax]
    cb.set_ticks(ticks, labels=[str(v) for v in ticks])
    cb.minorticks_off()


def draw_hex(ax, cells, values, cmap, norm, t, ext):
    x0, x1, y0, y1 = ext
    ax.set_xlim(x0, x1), ax.set_ylim(y0, y1)
    basemap(ax, t)
    pc = PolyCollection(hexes(cells), array=np.asarray(values, float), cmap=cmap, norm=norm,
                        edgecolors="none", alpha=0.9, zorder=2)
    ax.add_collection(pc)
    ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    return pc


def clean_locations(loc, min_devices=50):
    """Blank homes / work places sitting on coordinate sinks (default geocodes, towers)."""
    hs = pr.coordinate_sinks(loc, "home_latitude", "home_longitude", min_devices=min_devices)
    ws = pr.coordinate_sinks(loc, "work_latitude", "work_longitude", min_devices=min_devices)
    def blank(df, sinks, a, b):
        bad = sinks.select(pl.col(a), pl.col(b), pl.lit(True).alias("_sink"))
        return (df.join(bad, on=[a, b], how="left")
                .with_columns(pl.when(pl.col("_sink")).then(None).otherwise(pl.col(a)).alias(a),
                              pl.when(pl.col("_sink")).then(None).otherwise(pl.col(b)).alias(b)).drop("_sink"))
    out = blank(blank(loc, hs, "home_latitude", "home_longitude"), ws, "work_latitude", "work_longitude")
    return out, int(hs["devices"].sum()), int(ws["devices"].sum())


def fig_h3(loc, t, path, k=5):
    seq = LinearSegmentedColormap.from_list("seq", t["seq"])
    div = LinearSegmentedColormap.from_list(
        "div", ["#1c5cab", "#86b6ef", "#f0efec" if t is THEMES["light"] else "#383835", "#ec835a", "#c93b3a"])
    loc, nh, nw = clean_locations(loc)
    h10 = pr.home_work_h3(loc, 10, min_count=k, geometry=False).drop_nulls("homes")
    w9 = pr.home_work_h3(loc, 9, min_count=k, geometry=False).drop_nulls("workers")
    b8 = pr.home_work_h3(loc, 8, min_count=k, geometry=False).drop_nulls(["homes", "workers"])
    H, W = b8["homes"].sum(), b8["workers"].sum()
    bal = np.log2((b8["workers"].to_numpy() / W) / (b8["homes"].to_numpy() / H))

    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2))
    fig.subplots_adjust(left=0.005, right=0.995, wspace=0.05, top=0.83, bottom=0.12)
    ext = extent()
    q = lambda s: float(np.percentile(s.to_numpy(), 99))
    panels = [
        (h10, h10["homes"], seq, LogNorm(k, q(h10["homes"])), f"Homes, H3 res 10 (~76 m): {h10['homes'].sum():,}"),
        (w9, w9["workers"], seq, LogNorm(k, q(w9["workers"])), f"Work places, H3 res 9 (~200 m): {w9['workers'].sum():,}"),
        (b8, np.clip(bal, -3, 3), div, Normalize(-3, 3), "Jobs vs residents, H3 res 8 (~530 m)"),
    ]
    for ax, (cells, vals, cmap, norm, title) in zip(axes, panels):
        pc = draw_hex(ax, cells, vals, cmap, norm, t, ext)
        ax.set_title(title, color=t["ink"], fontsize=10.5)
        cb = fig.colorbar(pc, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, shrink=0.8)
        cb.outline.set_visible(False)
        if cmap is div:
            cb.set_ticks([-3, 0, 3], labels=["8x more residents", "city average", "8x more jobs"])
        else:
            cb.set_label("devices per cell (log)", color=t["ink2"])
            count_ticks(cb, norm.vmin, norm.vmax)
    fig.suptitle("Where Mexico City lives and works", x=0.008, y=0.985, ha="left", fontsize=13, fontweight="bold",
                 color=t["ink"])
    fig.text(0.008, 0.92, f"Cells with fewer than {k} devices are hidden. Homes / work places on coordinate sinks "
             f"(exact points shared by 50+ devices: {nh:,} homes, {nw:,} work places) are excluded.",
             fontsize=9, color=t["ink2"])
    fig.text(0.008, 0.008, ATTRIBUTION, fontsize=7, color=t["muted"])
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_h3_zoom(loc, t, path, k=5):
    seq = LinearSegmentedColormap.from_list("seq", t["seq"])
    loc, _, _ = clean_locations(loc)
    x, y = merc(np.array(ZOOM_LON), np.array(ZOOM_LAT))
    ext = (x[0], x[1], y[0], y[1])
    box = (pl.col("h3_latitude").is_between(*ZOOM_LAT)) & (pl.col("h3_longitude").is_between(*ZOOM_LON))
    h11 = pr.home_work_h3(loc, 11, min_count=k).drop_nulls("homes").filter(box)
    w10 = pr.home_work_h3(loc, 10, min_count=k).drop_nulls("workers").filter(box)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2))
    fig.subplots_adjust(left=0.005, right=0.995, wspace=0.04, top=0.88, bottom=0.12)
    for ax, cells, col, title in [(axes[0], h11, "homes", "Homes, H3 res 11 (~29 m)"),
                                  (axes[1], w10, "workers", "Work places, H3 res 10 (~76 m)")]:
        vmax = max(float(np.percentile(cells[col].to_numpy(), 99)), k + 1)
        pc = draw_hex(ax, cells, cells[col], seq, LogNorm(k, vmax), t, ext)
        ax.set_title(title, color=t["ink"], fontsize=11)
        cb = fig.colorbar(pc, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, shrink=0.7)
        cb.outline.set_visible(False), cb.set_label("devices per cell (log)", color=t["ink2"])
        count_ticks(cb, k, vmax)
    fig.suptitle(f"Roma, Condesa and Juárez at block level (cells with fewer than {k} devices are hidden)",
                 x=0.008, y=0.98, ha="left", fontsize=13, fontweight="bold", color=t["ink"])
    fig.text(0.008, 0.008, ATTRIBUTION, fontsize=7, color=t["muted"])
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--out", default="docs/assets/figures")
    ap.add_argument("--only-h3", action="store_true")
    ap.add_argument("--modes", action="store_true", help="mode-feature figure from the trajectories (needs v0.3+ output)")
    a = ap.parse_args()
    if a.modes:
        cols = ["uid", "origin_stop_event", "start_timestamp", "length_m", "speed_p85_kmh", "speed_cv", "mode_guess"]
        tr = pl.read_parquet(glob.glob(f"{a.output_dir}/trajectories/**/shard=*.parquet", recursive=True), columns=cols)
        tz = (tr.select("uid", pl.col("origin_stop_event").alias("stop_event")).drop_nulls().unique()
              .with_columns(pl.lit("America/Mexico_City").alias("inputed_timezone")))
        print(tr["mode_guess"].value_counts().sort("count", descending=True))
        print(tr.group_by("mode_guess").agg(pl.len(), pl.col("speed_p85_kmh").median(), pl.col("length_m").median())
              .sort("len", descending=True))
        for mode, t in THEMES.items():
            style(t)
            fig_mode_features(tr, tz, t, f"{a.out}/cdmx_modes_{mode}.png")
        return
    loc = pl.read_parquet(glob.glob(f"{a.output_dir}/locations/shard=*.parquet"))
    for mode, t in THEMES.items():
        style(t)
        fig_h3(loc, t, f"{a.out}/cdmx_h3_{mode}.png")
        fig_h3_zoom(loc, t, f"{a.out}/cdmx_h3_zoom_{mode}.png")
    if a.only_h3:
        return
    stays, home, work, flow, n = load(a.output_dir)
    print(n)
    for mode, t in THEMES.items():
        style(t)
        fig_stays(stays, home, work, n, t, f"{a.out}/cdmx_stays_{mode}.png")
        fig_flows(flow, n, t, f"{a.out}/cdmx_flows_{mode}.png")
        print("figures:", mode)


if __name__ == "__main__":
    main()
