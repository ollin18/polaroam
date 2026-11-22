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
from matplotlib.colors import LinearSegmentedColormap, LogNorm

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


def outline(ax, t, lw=1.2, color=None):
    b = boundary()
    for poly in getattr(b, "geoms", [b]):
        lo, la = np.asarray(poly.exterior.coords).T
        x, y = merc(lo, la)
        ax.plot(x, y, color=color or t["ink2"], lw=lw, zorder=4)


#: Trajectory maps: white page and black text, only the map panel is dark so white-hot paths stand out.
MAP_BG, MAP_LINE = "#1a1a19", "#898781"


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
    bins_flow = 900
    stay_lon, stay_lat = [], []
    flow = np.zeros((bins_flow, bins_flow))
    by_mode = {m: np.zeros((bins_flow, bins_flow)) for m in MODES}
    n = {"stays": 0, "home": 0, "work": 0, "trajectories": 0, "vertices": 0, "devices": 0, **{m: 0 for m in MODES}}
    for f in sorted(glob.glob(f"{out}/clusters/shard=*.parquet")):
        c = pl.read_parquet(f, columns=["uid", "stop_latitude", "stop_longitude"])
        stay_lon.append(c["stop_longitude"].to_numpy())
        stay_lat.append(c["stop_latitude"].to_numpy())
        n["stays"] += c.height
        n["devices"] += c["uid"].n_unique()
    for f in sorted(glob.glob(f"{out}/trajectories/**/shard=*.parquet", recursive=True)):
        tr = pr.guess_mode(pl.read_parquet(f, columns=["coordinates", "n_points", *MODE_COLS]))
        flow += vertex_density(tr["coordinates"], bins_flow)
        for m in MODES:
            sub_ = tr.filter(pl.col("mode_guess") == m)
            by_mode[m] += vertex_density(sub_["coordinates"], bins_flow)
            n[m] += sub_.height
        n["trajectories"] += tr.height
        n["vertices"] += int(tr["n_points"].sum())
    return np.concatenate(stay_lon), np.concatenate(stay_lat), flow, by_mode, n


DS_BG = "#0e0e12"   # datashader panel background (as in the haystac maps)


def ds_extent(margin=0.02):
    """City bounds in degrees, with a margin."""
    lo0, la0, lo1, la1 = boundary().bounds
    mx, my = (lo1 - lo0) * margin, (la1 - la0) * margin
    return lo0 - mx, lo1 + mx, la0 - my, la1 + my


def ds_image(lon, lat, cmap, width=900, spread=0):
    """datashader point density, histogram-equalised, clipped to the city boundary."""
    import datashader as ds
    import datashader.transfer_functions as tf
    import pandas as pd

    lon0, lon1, lat0, lat1 = ds_extent()
    h = max(1, round(width * (lat1 - lat0) / (lon1 - lon0)))   # square pixels in degrees
    cvs = ds.Canvas(plot_width=width, plot_height=h, x_range=(lon0, lon1), y_range=(lat0, lat1))
    agg = cvs.points(pd.DataFrame({"lon": np.asarray(lon, float), "lat": np.asarray(lat, float)}), "lon", "lat")
    X, Y = np.meshgrid(agg.coords["lon"].values, agg.coords["lat"].values)
    agg = agg.where(inside(X, Y) & (agg > 0))                         # outside the city: transparent
    img = tf.shade(agg, cmap=cmap, how="eq_hist", min_alpha=0)
    if spread:   # grow isolated points so sparse layers stay visible (datashader's usual remedy)
        img = tf.spread(img, px=spread, shape="circle")
    img = tf.set_background(img, DS_BG)
    return np.asarray(img.to_pil()), (lon0, lon1, lat0, lat1)


def ds_panel(ax, img, ext, title, subtitle=""):
    """A datashader map on a white page: dark panel, true proportions, city outline."""
    ax.imshow(img, extent=ext, origin="upper", interpolation="auto",
              aspect=1 / np.cos(np.radians((ext[2] + ext[3]) / 2)))
    b = boundary()
    for poly in getattr(b, "geoms", [b]):
        lo, la = np.asarray(poly.exterior.coords).T
        ax.plot(lo, la, color=MAP_LINE, lw=0.8)
    ax.set_xlim(ext[0], ext[1]), ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    if subtitle:
        ax.text(0.5, -0.02, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=10.5,
                color=THEMES["light"]["ink2"])


def inferno_list(lo=0.12):
    import matplotlib

    return [matplotlib.colors.to_hex(c) for c in matplotlib.colormaps["inferno"](np.linspace(lo, 1.0, 12))]


def fig_ds_stays(lon, lat, n, path):
    img, ext = ds_image(lon, lat, inferno_list(), width=1000)
    fig, ax = plt.subplots(figsize=(9.5, 11))
    ds_panel(ax, img, ext, f"Where people stay: {n['stays'] / 1e6:.1f}M stays, {n['devices'] / 1e6:.2f}M devices",
             "datashader, histogram-equalised · December 2022")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_ds_home_work(loc, path):
    loc, nh, _nw = clean_locations(loc)
    h = loc.drop_nulls("home_latitude")
    w = loc.drop_nulls("work_latitude")
    fig, axes = plt.subplots(1, 2, figsize=(15, 9.5))
    panels = [(h, "home_latitude", "home_longitude", f"Homes: {h.height:,}",
               f"homes on coordinate sinks excluded ({nh:,})"),
              (w, "work_latitude", "work_longitude", f"Work places: {w.height:,}",
               "datashader, histogram-equalised")]
    for ax, (d, la, lo, title, sub_) in zip(axes, panels):
        img, ext = ds_image(d[lo], d[la], inferno_list(), width=700, spread=1)
        ds_panel(ax, img, ext, title, sub_)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_flows(flow, n, t, path):
    x0, x1, y0, y1 = extent()
    # dark surface -> deep blue -> bright blue -> white (most traveled)
    cmap = LinearSegmentedColormap.from_list("flow", [MAP_BG, "#104281", "#2a78d6", "#86b6ef", "#ffffff"])
    fig, ax = plt.subplots(figsize=(9, 9))
    m = np.ma.masked_where((flow < 1) | ~grid_mask(flow.shape), flow)
    ax.set_facecolor(MAP_BG)
    ax.imshow(m, extent=(x0, x1, y0, y1), origin="lower", cmap=cmap,
              norm=LogNorm(vmin=1, vmax=np.percentile(flow[flow > 0], 99)), interpolation="antialiased")
    ax.set_xlim(x0, x1), ax.set_ylim(y0, y1)
    outline(ax, t, lw=0.8, color=MAP_LINE)
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
    ticks = [v for v in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) if vmin <= v <= vmax]
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


def fig_h3(loc, t, path, k=1):
    seq = hot_cmap()
    loc, nh, nw = clean_locations(loc)
    h10 = pr.home_work_h3(loc, 10, min_count=k, geometry=False).drop_nulls("homes")
    w10 = pr.home_work_h3(loc, 10, min_count=k, geometry=False).drop_nulls("workers")

    fig, axes = plt.subplots(1, 2, figsize=(11, 8.0))
    fig.subplots_adjust(left=0.005, right=0.995, wspace=0.05, top=0.87, bottom=0.10)
    ext = extent()
    q = lambda s: float(np.percentile(s.to_numpy(), 99))
    panels = [
        (h10, h10["homes"], seq, LogNorm(k, q(h10["homes"])), f"Homes, H3 res 10 (~76 m): {h10['homes'].sum():,}"),
        (w10, w10["workers"], seq, LogNorm(k, q(w10["workers"])),
         f"Work places, H3 res 10 (~76 m): {w10['workers'].sum():,}"),
    ]
    for ax, (cells, vals, cmap, norm, title) in zip(axes, panels):
        pc = draw_hex(ax, cells, vals, cmap, norm, t, ext)
        ax.set_title(title, color=t["ink"], fontsize=10.5)
        cb = fig.colorbar(pc, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, shrink=0.8)
        cb.outline.set_visible(False)
        cb.set_label("devices per cell (log)", color=t["ink2"])
        count_ticks(cb, norm.vmin, norm.vmax)
    fig.suptitle("Where Mexico City lives and works", x=0.008, y=0.985, ha="left", fontsize=13, fontweight="bold",
                 color=t["ink"])
    hidden = f"Cells with fewer than {k} devices are hidden. " if k > 1 else ""
    fig.text(0.008, 0.935, f"{hidden}Homes / work places on coordinate sinks "
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
            m, [MAP_BG, blend(color, MAP_BG, 0.55), color, blend(color, "#ffffff", 0.6), "#ffffff"])
        mm = np.ma.masked_where((h < 1) | ~grid_mask(h.shape), h)
        ax.set_facecolor(MAP_BG)
        ax.imshow(mm, extent=(x0, x1, y0, y1), origin="lower", cmap=cmap, interpolation="antialiased",
                  norm=LogNorm(vmin=1, vmax=max(np.percentile(h[h > 0], 99), 2)))
        ax.set_xlim(x0, x1), ax.set_ylim(y0, y1)
        outline(ax, t, lw=0.8, color=MAP_LINE)
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
    for mode, t in THEMES.items():
        style(t)
        fig_h3(loc, t, f"{a.out}/cdmx_h3_{mode}.png")
    if a.only_h3:
        return
    stay_lon, stay_lat, flow, by_mode, n = load(a.output_dir)
    print(n)
    # datashader maps (as in the haystac project): one image for both site themes
    style(THEMES["light"])
    fig_ds_stays(stay_lon, stay_lat, n, f"{a.out}/cdmx_stays.png")
    fig_ds_home_work(loc, f"{a.out}/cdmx_home_work.png")
    # trajectory maps: one image for both site themes: white page, black text, dark map panel
    style(THEMES["light"])
    fig_flows(flow, n, THEMES["light"], f"{a.out}/cdmx_flows.png")
    fig_mode_maps(by_mode, n, THEMES["light"], f"{a.out}/cdmx_mode_maps.png")
    print("figures: trajectory maps")


if __name__ == "__main__":
    main()
