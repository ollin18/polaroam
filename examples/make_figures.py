"""Build the README figures from the example data.

    python examples/make_figures.py data/veraset_movement_416.snappy.parquet \
        --tz-polygons path/to/tz_codes.csv

Maps only show aggregates over many devices (density, overlaid trajectories);
the single-device panel is drawn in a relative frame (km from an arbitrary
origin) without basemap, so no individual location is disclosed.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm

import polaroam as pr
import polaroam.geo

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "series": ["#2a78d6", "#eb6834", "#1baf7a"], "tiles": "light_nolabels",
              "seq": ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"], "line_alpha": 0.05},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "series": ["#3987e5", "#d95926", "#199e70"], "tiles": "dark_nolabels",
             "seq": ["#104281", "#256abf", "#3987e5", "#86b6ef", "#cde2fb"], "line_alpha": 0.07},
}
CITIES = {"America/Mexico_City": "Mexico City", "America/Bogota": "Bogotá", "America/Sao_Paulo": "São Paulo",
          "Asia/Manila": "Manila", "Asia/Jakarta": "Jakarta"}
LOCATION = {"H": "Home", "W": "Work", "O": "Other"}
R_MERC = 6378137.0
ATTRIBUTION = "Basemap: Esri World Gray Canvas (Esri, HERE, Garmin, © OpenStreetMap contributors)"


def merc(lon, lat):
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    return R_MERC * np.radians(lon), R_MERC * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def style(t):
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"], "font.size": 10,
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "text.color": t["ink"], "axes.labelcolor": t["ink2"], "axes.edgecolor": t["axis"],
        "xtick.color": t["muted"], "ytick.color": t["muted"], "axes.grid": True, "grid.color": t["grid"],
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "legend.labelcolor": t["ink2"], "axes.axisbelow": True,
    })


def basemap(ax, t):
    try:
        import contextily as cx
        variant = "Canvas/World_Light_Gray_Base" if t["tiles"] == "light_nolabels" else "Canvas/World_Dark_Gray_Base"
        url = f"https://server.arcgisonline.com/ArcGIS/rest/services/{variant}/MapServer/tile/{{z}}/{{y}}/{{x}}"
        cx.add_basemap(ax, crs="EPSG:3857", source=url, attribution=False, zorder=0)
    except Exception as e:  # noqa: BLE001 - offline: keep the plain surface
        print("basemap skipped:", e)


def city_extent(df, lat="stop_latitude", lon="stop_longitude", q=0.03):
    la = df.select(pl.col(lat).quantile(q).alias("a"), pl.col(lat).quantile(1 - q).alias("b")).row(0)
    lo = df.select(pl.col(lon).quantile(q).alias("a"), pl.col(lon).quantile(1 - q).alias("b")).row(0)
    x, y = merc(lo, la)
    cx, cy, half = (x[0] + x[1]) / 2, (y[0] + y[1]) / 2, max(x[1] - x[0], y[1] - y[0]) / 2 * 1.05
    return cx - half, cx + half, cy - half, cy + half


def fig_stop_density(hw, t, out):
    cmap = LinearSegmentedColormap.from_list("seq", t["seq"])
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.9))
    fig.subplots_adjust(left=0.005, right=0.95, wspace=0.04, top=0.84, bottom=0.05)
    for ax, (tz, name) in zip(axes, CITIES.items()):
        d = hw.filter(pl.col("inputed_timezone") == tz)
        x0, x1, y0, y1 = city_extent(d)
        x, y = merc(d["stop_longitude"], d["stop_latitude"])
        ax.set_xlim(x0, x1), ax.set_ylim(y0, y1)
        basemap(ax, t)
        hb = ax.hexbin(x, y, gridsize=55, extent=(x0, x1, y0, y1), cmap=cmap, norm=LogNorm(1, 3000),
                       mincnt=1, linewidths=0, alpha=0.92, zorder=2)
        ax.set_title(name, color=t["ink"]), ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(0.02, 0.02, f"{d['uid'].n_unique()} devices · {d.height:,} stays", transform=ax.transAxes,
                fontsize=8, color=t["ink2"], zorder=3)
    cb = fig.colorbar(hb, cax=fig.add_axes([0.957, 0.12, 0.008, 0.66]))
    cb.set_label("stays per cell", color=t["ink2"]), cb.outline.set_visible(False)
    fig.text(0.01, 0.01, ATTRIBUTION, fontsize=7, color=t["muted"])
    fig.suptitle("Where people stay: detected stays per hexagon (log scale)", x=0.008, y=0.98, ha="left",
                 fontsize=13, fontweight="bold", color=t["ink"])
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_trajectories(tr, hw, t, out, tz="America/Sao_Paulo"):
    uids = hw.filter(pl.col("inputed_timezone") == tz).select("uid").unique()
    d = tr.join(uids, on="uid", how="semi").filter(pl.col("length_m").is_between(300, 60_000))
    x0, x1, y0, y1 = city_extent(hw.filter(pl.col("inputed_timezone") == tz), q=0.04)
    segs = []
    for coords in d["coordinates"].to_list():
        c = np.asarray(coords)
        x, y = merc(c[:, 0], c[:, 1])
        segs.append(np.column_stack([x, y]))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(x0, x1), ax.set_ylim(y0, y1)
    basemap(ax, t)
    ax.add_collection(LineCollection(segs, colors=t["series"][0], linewidths=0.7, alpha=t["line_alpha"] * 3, zorder=2))
    ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{CITIES[tz]}: {d.height:,} simplified trajectories of {d['uid'].n_unique()} devices",
                 color=t["ink"], fontsize=12)
    ax.text(0.01, 0.01, ATTRIBUTION, transform=ax.transAxes, fontsize=7,
            color=t["muted"])
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_distributions(stops, tr, hw, t, out):
    c1 = t["series"][0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4))
    ax = axes[0, 0]
    dur = stops["stop_duration"].to_numpy() / 3600
    ax.hist(dur, bins=np.logspace(np.log10(5 / 60), np.log10(72), 60), color=c1, edgecolor=t["surface"], linewidth=0.6)
    ax.set_xscale("log"), ax.set_xticks([1 / 6, 0.5, 1, 3, 8, 24, 72], ["10m", "30m", "1h", "3h", "8h", "1d", "3d"])
    ax.set_title("Stay duration"), ax.set_ylabel("stays")
    med = float(np.median(dur))
    ax.axvline(med, color=t["ink2"], lw=1, ls="--"), ax.text(med * 1.1, ax.get_ylim()[1] * 0.92, f"median {med * 60:.0f} min",
                                                             color=t["ink2"], fontsize=9)

    ax = axes[0, 1]
    per_day = (hw.group_by("uid", "date").len().group_by("uid").agg(pl.col("len").mean())["len"].to_numpy())
    ax.hist(per_day, bins=40, color=c1, edgecolor=t["surface"], linewidth=0.6)
    ax.set_title("Stays per active day (per device)"), ax.set_ylabel("devices"), ax.set_xlabel("mean stays / day")

    ax = axes[1, 0]
    L = tr["length_m"].to_numpy() / 1000
    L = L[(L > 0.05) & (L < 200)]
    ax.hist(L, bins=np.logspace(np.log10(0.05), np.log10(200), 60), color=c1, edgecolor=t["surface"], linewidth=0.6)
    ax.set_xscale("log"), ax.set_xticks([0.1, 1, 10, 100], ["100 m", "1 km", "10 km", "100 km"])
    ax.set_title("Trajectory length"), ax.set_ylabel("trajectories")
    ratio = 1 - tr["n_points"].sum() / (tr["n_pings"].sum() + 2 * tr.height)
    ax.text(0.98, 0.92, f"Douglas-Peucker (30 m) keeps {1 - ratio:.0%} of vertices", transform=ax.transAxes,
            ha="right", color=t["ink2"], fontsize=9)

    ax = axes[1, 1]
    h = hw.group_by("location_type", "hour").len().with_columns(
        (pl.col("len") / pl.col("len").sum().over("location_type")).alias("share")).sort("hour")
    for (lt, color) in zip(["H", "W", "O"], t["series"]):
        s = h.filter(pl.col("location_type") == lt)
        ax.plot(s["hour"], s["share"] * 100, color=color, lw=2, label=LOCATION[lt])
        ax.text(s["hour"][-1] + 0.3, s["share"][-1] * 100, LOCATION[lt], color=t["ink2"], fontsize=9, va="center")
    ax.set_xticks(range(0, 24, 3), [f"{x:02d}h" for x in range(0, 24, 3)])
    ax.set_xlim(-0.5, 25.5)
    ax.set_title("Local hour when stays start, by location type"), ax.set_ylabel("% of stays of that type")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 0.98), ncols=3, fontsize=9, borderaxespad=0.6)
    ax.set_title("Local hour when stays start, by location type", pad=26)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_simplification(pings, stops, tr, t, out, seed=7):
    # A busy device-day, drawn in a relative frame (km from an arbitrary origin).
    cand = tr.filter(pl.col("n_pings").is_between(25, 120) & pl.col("length_m").is_between(4000, 25000)
                     & pl.col("origin_stop_event").is_not_null() & pl.col("destination_stop_event").is_not_null())
    cand = cand.with_columns(pl.col("coordinates").list.first().alias("_a"), pl.col("coordinates").list.last().alias("_b"))
    cand = cand.with_columns(pr.geo.haversine(pl.col("_a").list.get(1), pl.col("_a").list.get(0),
                                              pl.col("_b").list.get(1), pl.col("_b").list.get(0)).alias("_straight"))
    cand = cand.filter(pl.col("_straight") / pl.col("length_m") > 0.75)
    row = cand.sample(1, seed=seed).row(0, named=True)
    uid, t0, t1 = row["uid"], row["start_timestamp"] - 1800, row["end_timestamp"] + 1800
    p = pings.filter((pl.col("uid") == uid) & (pl.col("timestamp") // 1000).is_between(t0, t1))
    st = stops.filter((pl.col("uid") == uid) & (pl.col("end_timestamp") >= t0) & (pl.col("start_timestamp") <= t1))
    lat0, lon0 = p["latitude"].mean(), p["longitude"].mean()
    k = 111.195

    def rel(la, lo):
        return (np.asarray(lo) - lon0) * k * math.cos(math.radians(lat0)), (np.asarray(la) - lat0) * k

    fig, ax = plt.subplots(figsize=(8, 6))
    x, y = rel(p["latitude"], p["longitude"])
    ax.scatter(x, y, s=14, color=t["muted"], alpha=0.7, linewidths=0, zorder=2, label=f"raw pings ({p.height})")
    c = np.asarray(row["coordinates"])
    sx, sy = rel(c[:, 1], c[:, 0])
    ax.plot(sx, sy, color=t["series"][0], lw=2, zorder=3, marker="o", ms=5, label=f"simplified trajectory ({len(c)} vertices)")
    qx, qy = rel(st["stop_latitude"], st["stop_longitude"])
    ax.scatter(qx, qy, s=160, color=t["series"][1], edgecolor=t["surface"], linewidth=2, zorder=4, label="stays")
    for (a, b, d) in zip(qx, qy, st["stop_duration"].to_list()):
        ax.annotate(f"{d / 60:.0f} min", (a, b), xytext=(8, 6), textcoords="offset points", color=t["ink2"], fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("km east (relative)"), ax.set_ylabel("km north (relative)")
    ax.set_title("From pings to stays and a simplified trajectory (one trip)")
    ax.legend(loc="best")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


MODES = ["walk", "bike", "motorized"]


def fig_mode_features(tr, hw, t, out):
    """Speed features by mode_guess: cruising speed, variability, hour of day, length."""
    cl = dict(zip(MODES, t["series"]))
    d = tr.filter(pl.col("mode_guess").is_in(MODES))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6))

    ax = axes[0, 0]
    bins = np.logspace(0, np.log10(150), 50)
    for m in MODES:
        x = d.filter(pl.col("mode_guess") == m)["speed_p85_kmh"].to_numpy()
        ax.hist(x, bins=bins, histtype="step", lw=2, color=cl[m], label=f"{m} ({len(x):,})")
    for thr, lab in [(8, "8 km/h"), (22, "22 km/h")]:
        ax.axvline(thr, color=t["muted"], lw=1, ls="--")
        ax.text(thr * 1.05, ax.get_ylim()[1] * 0.93, lab, color=t["ink2"], fontsize=8)
    ax.set_xscale("log"), ax.set_xticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
    ax.set_xlabel("cruising speed, p85 (km/h)"), ax.set_ylabel("trajectories")
    ax.set_title("Cruising speed separates the modes"), ax.legend(loc="upper left", fontsize=9)

    ax = axes[0, 1]
    c = tr.filter(pl.col("speed_p85_kmh").is_not_null() & pl.col("speed_cv").is_not_null()
                  & (pl.col("speed_p85_kmh") > 0.5))
    cmap = LinearSegmentedColormap.from_list("seq", t["seq"])
    hb = ax.hexbin(c["speed_p85_kmh"], c["speed_cv"].clip(upper_bound=2.5), xscale="log", gridsize=45, cmap=cmap,
                   norm=LogNorm(), mincnt=1, linewidths=0)
    ax.set_xticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"]), ax.grid(False)
    ax.set_xlabel("cruising speed, p85 (km/h)"), ax.set_ylabel("speed CV (std / mean)")
    ax.set_title("Steady vs stop-and-go")
    cb = fig.colorbar(hb, ax=ax, pad=0.01)
    cb.set_label("trajectories", color=t["ink2"]), cb.outline.set_visible(False)

    ax = axes[1, 0]
    tz = hw.select("uid", pl.col("stop_event").alias("origin_stop_event"), "inputed_timezone").unique(
        subset=["uid", "origin_stop_event"])
    h = (d.join(tz, on=["uid", "origin_stop_event"]).with_columns(pl.from_epoch("start_timestamp", "s")
         .dt.replace_time_zone("UTC").alias("_t")))
    hours = []
    for z in h["inputed_timezone"].unique().to_list():
        hz = h.filter(pl.col("inputed_timezone") == z)
        hours.append(hz.with_columns(pl.col("_t").dt.convert_time_zone(z).dt.hour().alias("hour")).select("mode_guess", "hour"))
    hh = pl.concat(hours).group_by("hour", "mode_guess").len().with_columns(
        (pl.col("len") / pl.col("len").sum().over("hour") * 100).alias("share")).sort("hour")
    for m in MODES:
        s = hh.filter(pl.col("mode_guess") == m)
        ax.plot(s["hour"], s["share"], color=cl[m], lw=2, label=m)
        ax.text(s["hour"][-1] + 0.3, s["share"][-1], m, color=t["ink2"], fontsize=9, va="center")
    ax.set_xticks(range(0, 24, 3), [f"{x:02d}h" for x in range(0, 24, 3)]), ax.set_xlim(-0.5, 26)
    ax.set_ylabel("% of trajectories starting that hour"), ax.set_title("Mode share by local hour")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 0.98), ncols=3, fontsize=9, borderaxespad=0.6)
    ax.set_title("Mode share by local hour", pad=26)

    ax = axes[1, 1]
    meds = []
    for m in MODES:
        x = np.sort(d.filter(pl.col("mode_guess") == m)["length_m"].to_numpy() / 1000)
        ax.plot(x, np.arange(1, len(x) + 1) / len(x) * 100, color=cl[m], lw=2, label=m)
        med = float(np.median(x))
        ax.plot([med], [50], "o", ms=7, color=cl[m], mec=t["surface"], mew=2, zorder=5)
        meds.append(f"{m} {med:.1f} km")
    ax.text(0.98, 0.04, "medians: " + " · ".join(meds), transform=ax.transAxes, ha="right", fontsize=8.5,
            color=t["ink2"])
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xscale("log"), ax.set_xticks([0.1, 1, 10, 100], ["100 m", "1 km", "10 km", "100 km"])
    ax.axhline(50, color=t["muted"], lw=1, ls="--")
    ax.set_ylabel("% of trajectories shorter than"), ax.set_title("Trip length by mode")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def window_speeds(c, ts, window=30, max_gap=300):
    """Windowed speeds (km/h) from full-resolution vertices, as in polaroam.speed_features."""
    lon, lat, ts = np.asarray([p[0] for p in c]), np.asarray([p[1] for p in c]), np.asarray(ts, float)
    j = np.searchsorted(ts, ts + window, side="left").clip(max=len(ts) - 1)
    ok = ts[j] > ts
    r = 6371008.8
    p1, p2 = np.radians(lat), np.radians(lat[j])
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon[j] - lon) / 2) ** 2
    d = 2 * r * np.arcsin(np.sqrt(np.minimum(a, 1)))
    v = np.where(ok, d / np.maximum(ts[j] - ts, 1) * 3.6, np.nan)
    gap = np.append(np.diff(ts), 0) > max_gap  # speed not observed across long gaps
    v[gap] = np.nan
    return ts, v


def fig_speed_profiles(pings, stops, tr, t, out, seed=3):
    """One trip per mode: windowed speed over time, with the p85 'cruising speed'."""
    cl = dict(zip(MODES, t["series"]))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=False)
    for ax, m in zip(axes, MODES):
        lo, hi = {"walk": (500, 3000), "bike": (2000, 8000), "motorized": (5000, 30000)}[m]
        cand = tr.filter((pl.col("mode_guess") == m) & pl.col("length_m").is_between(lo, hi)
                         & (pl.col("n_pings") >= pl.col("duration") / 40)       # >= 1.5 pings / minute
                         & pl.col("duration").is_between(300, 3600) & (pl.col("n_jumps") == 0)
                         & (pl.col("straightness") > 0.6))
        row = cand.sample(1, seed=seed).row(0, named=True)
        p = pings.filter(pl.col("uid") == row["uid"])
        full = pr.extract_trajectories(p, stops.filter(pl.col("uid") == row["uid"]), simplify_tolerance=0)
        r = full.filter(pl.col("start_timestamp") == row["start_timestamp"]).row(0, named=True)
        ts, v = window_speeds(r["coordinates"], r["timestamps"])
        mins = (ts - ts[0]) / 60
        ax.step(mins, v, where="post", color=cl[m], lw=1.8)
        gaps = np.flatnonzero(np.isnan(v[:-1]))
        for g in gaps:  # shade unobserved gaps
            ax.axvspan(mins[g], mins[g + 1], color=t["grid"], alpha=0.6, lw=0)
        ax.axhline(r["speed_p85_kmh"], color=t["ink2"], lw=1, ls="--")
        ax.text(mins[-1], r["speed_p85_kmh"], f"p85 {r['speed_p85_kmh']:.0f} km/h", color=t["ink2"],
                fontsize=8.5, va="bottom", ha="right")
        ax.set_ylim(0, max(np.nanmax(v), r["speed_p85_kmh"]) * 1.25)
        ax.set_title(f"{m}: {r['length_m'] / 1000:.1f} km in {r['duration'] / 60:.0f} min")
        ax.set_xlabel("minutes since start")
        ax.text(0.02, 0.95, f"CV {r['speed_cv']:.2f} · slow {r['frac_slow']:.0%} · {r['n_pings']} pings"
                + (" · shaded: gaps > 5 min" if len(gaps) else ""),
                transform=ax.transAxes, fontsize=8.5, color=t["ink2"], va="top")
    axes[0].set_ylabel("speed (km/h, 30 s windows)")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pings")
    ap.add_argument("--tz-polygons")
    ap.add_argument("--work-dir", default="examples/output")
    ap.add_argument("--out", default="docs/assets/figures")
    a = ap.parse_args()
    cfg = pr.PipelineConfig(timezone="UTC", tz_polygons=a.tz_polygons,
                            trajectories=pr.TrajectoryParams(min_pings=2, min_length_m=100))
    pipe = pr.Pipeline(a.work_dir, cfg, trajectories=True)
    if not (Path(a.work_dir) / "manifest.json").exists():
        pipe.run(a.pings)
    stops, tr, hw = pipe.stops().collect(), pipe.trajectories_table().collect(), pipe.home_work().collect()
    tr = pr.guess_mode(tr)  # current rules, whatever version wrote the table
    pings = pr.read_pings(a.pings).collect()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for mode, t in THEMES.items():
        style(t)
        fig_stop_density(hw, t, f"{a.out}/stays_map_{mode}.png")
        fig_trajectories(tr, hw, t, f"{a.out}/trajectories_map_{mode}.png")
        fig_distributions(stops, tr, hw, t, f"{a.out}/distributions_{mode}.png")
        fig_simplification(pings, stops, tr, t, f"{a.out}/simplification_{mode}.png")
        fig_mode_features(tr, hw, t, f"{a.out}/mode_features_{mode}.png")
        fig_speed_profiles(pings, stops, tr, t, f"{a.out}/speed_profiles_{mode}.png")
        print("figures:", mode)


if __name__ == "__main__":
    main()
