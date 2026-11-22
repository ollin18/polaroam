"""Figures for the "Mexico City at scale" docs page.

    python examples/mexico_city_figures.py /data/tmp/polaroam_quadrant/oneshot_cpu

Reads a polaroam output directory (one month, 1.66B pings) shard by shard and draws
aggregated maps only: densities over hundreds of thousands of devices.
"""

from __future__ import annotations

import argparse
import functools
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from make_figures import ATTRIBUTION, R_MERC, THEMES, basemap, fig_mode_features, merc, style
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

import polaroam as pr

BOUNDARY = Path(__file__).parent / "data" / "cdmx_boundary.geojson"   # INEGI Marco Geoestadístico, CVE_ENT 09


@functools.lru_cache(maxsize=1)
def boundary():
    import shapely

    geo = json.loads(BOUNDARY.read_text())["features"][0]["geometry"]
    poly = shapely.geometry.shape(geo)
    shapely.prepare(poly)
    return poly


def extent(margin=0.02):
    lo0, la0, lo1, la1 = boundary().bounds
    x, y = merc(np.array([lo0, lo1]), np.array([la0, la1]))
    mx, my = (x[1] - x[0]) * margin, (y[1] - y[0]) * margin
    return x[0] - mx, x[1] + mx, y[0] - my, y[1] + my


def inside(lon, lat):
    import shapely

    return shapely.contains_xy(boundary(), np.asarray(lon), np.asarray(lat))


def grid_mask(shape):
    """True for raster cells whose center is inside the city boundary."""
    x0, x1, y0, y1 = extent()
    ny, nx = shape
    xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    ys = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    X, Y = np.meshgrid(xs, ys)
    lon = np.degrees(X / R_MERC)
    lat = np.degrees(np.arctan(np.sinh(Y / R_MERC)))
    return inside(lon, lat)


def outline(ax, t, lw=1.2):
    b = boundary()
    for poly in getattr(b, "geoms", [b]):
        lo, la = np.asarray(poly.exterior.coords).T
        x, y = merc(lo, la)
        ax.plot(x, y, color=t["ink2"], lw=lw, zorder=4)


def hist2d(lat, lon, bins):
    x, y = merc(lon, lat)
    x0, x1, y0, y1 = extent()
    h, _, _ = np.histogram2d(y, x, bins=bins, range=[[y0, y1], [x0, x1]])
    return h


MODES = ["walk", "bike", "motorized"]


def hot_cmap():
    """Multi-hue sequential scale for counts on a dark basemap: inferno without its near-black
    start, so the lowest counts stay visible (dark purple -> red -> orange -> pale yellow)."""
    import matplotlib
    from matplotlib.colors import ListedColormap

    return ListedColormap(matplotlib.colormaps["inferno"](np.linspace(0.18, 1.0, 256)), name="hot")


MODE_COLS = ["speed_p85_kmh", "speed_max_kmh", "duration", "n_pings", "ping_gap_median_s", "length_m"]


def vertex_density(coords: pl.Series, bins: int):
    """Density of observed positions: the vertices of the trajectories (real pings and stay
    anchors, on actual streets), not the straight chords between sparse pings."""
    xy = np.asarray(pl.DataFrame({"c": coords}).explode("c").drop_nulls()["c"].to_list())
    if len(xy) == 0:
        return np.zeros((bins, bins))
    return hist2d(xy[:, 1], xy[:, 0], bins)


def load(out):
    bins_pts, bins_flow = 260, 900
    stays = np.zeros((bins_pts, bins_pts))
    home = np.zeros_like(stays)
    work = np.zeros_like(stays)
    flow = np.zeros((bins_flow, bins_flow))
    by_mode = {m: np.zeros((bins_flow, bins_flow)) for m in MODES}
    n = {"stays": 0, "home": 0, "work": 0, "trajectories": 0, "vertices": 0, "devices": 0, **{m: 0 for m in MODES}}
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
        tr = pr.guess_mode(pl.read_parquet(f, columns=["coordinates", "n_points", *MODE_COLS]))
        flow += vertex_density(tr["coordinates"], bins_flow)
        for m in MODES:
            sub_ = tr.filter(pl.col("mode_guess") == m)
            by_mode[m] += vertex_density(sub_["coordinates"], bins_flow)
            n[m] += sub_.height
        n["trajectories"] += tr.height
        n["vertices"] += int(tr["n_points"].sum())
    return stays, home, work, flow, by_mode, n


def fig_stays(stays, home, work, n, t, path):
    cmap = hot_cmap()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    fig.subplots_adjust(left=0.005, right=0.95, wspace=0.03, top=0.86, bottom=0.05)
    x0, x1, y0, y1 = extent()
    panels = [(stays, f"All stays ({n['stays'] / 1e6:.1f}M)"), (home, f"Homes ({n['home']:,})"),
              (work, f"Work places ({n['work']:,})")]
    for ax, (h, title) in zip(axes, panels):
        ax.set_xlim(x0, x1), ax.set_ylim(y0, y1)
        basemap(ax, t)
        m = np.ma.masked_where((h < 1) | ~grid_mask(h.shape), h)
        vmax = max(float(np.percentile(h[h > 0], 99.5)), 2)
        im = ax.imshow(m, extent=(x0, x1, y0, y1), origin="lower", cmap=cmap, alpha=0.95, zorder=2,
                       norm=LogNorm(vmin=1, vmax=vmax), interpolation="nearest")
        outline(ax, t)
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
    # dark surface -> deep blue -> bright blue -> white (most traveled)
    cmap = LinearSegmentedColormap.from_list("flow", [t["surface"], "#104281", "#2a78d6", "#86b6ef", "#ffffff"])
    fig, ax = plt.subplots(figsize=(9, 9))
    m = np.ma.masked_where((flow < 1) | ~grid_mask(flow.shape), flow)
    ax.set_facecolor(t["surface"])
    ax.imshow(m, extent=(x0, x1, y0, y1), origin="lower", cmap=cmap,
              norm=LogNorm(vmin=1, vmax=np.percentile(flow[flow > 0], 99)), interpolation="antialiased")
    ax.set_xlim(x0, x1), ax.set_ylim(y0, y1)
    outline(ax, t, lw=0.8)
    ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{n['trajectories'] / 1e6:.1f}M trajectories in one month ({n['vertices'] / 1e6:.0f}M observed positions)",
                 color=t["ink"], fontsize=11)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


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


def clip_cells(cells, values):
    """Keep H3 cells whose center is inside the city boundary."""
    import polars_h3 as plh3

    c = cells.select(plh3.cell_to_lng("h3_cell").alias("lo"), plh3.cell_to_lat("h3_cell").alias("la"))
    keep = inside(c["lo"].to_numpy(), c["la"].to_numpy())
    return cells.filter(pl.Series(keep)), np.asarray(values)[keep]


def draw_hex(ax, cells, values, cmap, norm, t, ext):
    cells, values = clip_cells(cells, values)
    x0, x1, y0, y1 = ext
    ax.set_xlim(x0, x1), ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")   # true proportions (polygons do not imply an aspect)
    basemap(ax, t)
    pc = PolyCollection(hexes(cells), array=np.asarray(values, float), cmap=cmap, norm=norm,
                        edgecolors="none", alpha=0.9, zorder=2)
    ax.add_collection(pc)
    outline(ax, t)
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
    seq = hot_cmap()
    div = LinearSegmentedColormap.from_list(
        "div", ["#3987e5", "#86b6ef", "#f0efec" if t is THEMES["light"] else "#4a4a46", "#ec835a", "#e34948"])
    loc, nh, nw = clean_locations(loc)
    h10 = pr.home_work_h3(loc, 10, min_count=k, geometry=False).drop_nulls("homes")
    w10 = pr.home_work_h3(loc, 10, min_count=k, geometry=False).drop_nulls("workers")
    n_h, n_w = loc["home_latitude"].is_not_null().sum(), loc["work_latitude"].is_not_null().sum()
    b8 = pr.home_work_h3(loc, 8, min_count=k, geometry=False).drop_nulls(["homes", "workers"])
    H, W = b8["homes"].sum(), b8["workers"].sum()
    bal = np.log2((b8["workers"].to_numpy() / W) / (b8["homes"].to_numpy() / H))

    fig, axes = plt.subplots(1, 3, figsize=(15, 8.0))
    fig.subplots_adjust(left=0.005, right=0.995, wspace=0.05, top=0.87, bottom=0.10)
    ext = extent()
    q = lambda s: float(np.percentile(s.to_numpy(), 99))
    panels = [
        (h10, h10["homes"], seq, LogNorm(k, q(h10["homes"])),
         f"Homes, H3 res 10 (~76 m): {h10['homes'].sum():,} ({h10['homes'].sum() / n_h:.0%} shown)"),
        (w10, w10["workers"], seq, LogNorm(k, q(w10["workers"])),
         f"Work places, H3 res 10 (~76 m): {w10['workers'].sum():,} ({w10['workers'].sum() / n_w:.0%} shown)"),
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
    fig.text(0.008, 0.935, f"Cells with fewer than {k} devices are hidden. Homes / work places on coordinate sinks "
             f"(exact points shared by 50+ devices: {nh:,} homes, {nw:,} work places) are excluded.",
             fontsize=9, color=t["ink2"])
    fig.text(0.008, 0.008, ATTRIBUTION, fontsize=7, color=t["muted"])
    fig.savefig(path, dpi=130)
    plt.close(fig)


def blend(c1, c2, f):
    """Mix two hex colors (f = share of c2)."""
    from matplotlib.colors import to_hex, to_rgb

    return to_hex([(1 - f) * x + f * y for x, y in zip(to_rgb(c1), to_rgb(c2))])


def fig_mode_maps(by_mode, n, t, path):
    """City-wide trajectory density for each mode_guess class, each in its own hue."""
    x0, x1, y0, y1 = extent()
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.4))
    fig.subplots_adjust(left=0.005, right=0.995, wspace=0.02, top=0.86, bottom=0.03)
    for ax, m, color in zip(axes, MODES, t["series"]):
        h = by_mode[m]
        # dark surface -> deep hue -> hue -> pale hue -> white (most traveled)
        cmap = LinearSegmentedColormap.from_list(
            m, [t["surface"], blend(color, t["surface"], 0.55), color, blend(color, "#ffffff", 0.6), "#ffffff"])
        mm = np.ma.masked_where((h < 1) | ~grid_mask(h.shape), h)
        ax.set_facecolor(t["surface"])
        ax.imshow(mm, extent=(x0, x1, y0, y1), origin="lower", cmap=cmap, interpolation="antialiased",
                  norm=LogNorm(vmin=1, vmax=max(np.percentile(h[h > 0], 99), 2)))
        ax.set_xlim(x0, x1), ax.set_ylim(y0, y1)
        outline(ax, t, lw=0.8)
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{m}: {n[m] / 1e6:.2f}M trajectories", color=t["ink"], fontsize=12)
    fig.suptitle("Where each mode travels (baseline mode_guess; observed positions, log scale)",
                 x=0.008, y=0.985, ha="left", fontsize=13, fontweight="bold", color=t["ink"])
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
        cols = ["uid", "origin_stop_event", "start_timestamp", "speed_cv", *MODE_COLS]
        tr = pr.guess_mode(pl.read_parquet(glob.glob(f"{a.output_dir}/trajectories/**/shard=*.parquet", recursive=True),
                                           columns=cols))
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
    # one dark rendering (hot colors = most) used in both site themes
    style(THEMES["dark"])
    fig_h3(loc, THEMES["dark"], f"{a.out}/cdmx_h3.png")
    if a.only_h3:
        return
    stays, home, work, flow, by_mode, n = load(a.output_dir)
    print(n)
    style(THEMES["dark"])
    fig_stays(stays, home, work, n, THEMES["dark"], f"{a.out}/cdmx_stays.png")
    # trajectory maps: one dark rendering (white = most traveled), used in both site themes
    style(THEMES["dark"])
    fig_flows(flow, n, THEMES["dark"], f"{a.out}/cdmx_flows.png")
    fig_mode_maps(by_mode, n, THEMES["dark"], f"{a.out}/cdmx_mode_maps.png")
    print("figures: trajectory maps")


if __name__ == "__main__":
    main()
