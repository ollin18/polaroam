"""Compare one month of LBS trajectories with the official Origin-Destination Survey (EOD 2017).

    python examples/mexico_city_validation.py /data/out/MX_2022_12 \\
        --eod /data/eod_2017_csv/tviaje_eod2017/conjunto_de_datos/tviaje.csv \\
        --districts DistritosEODHogaresZMVM2017.shp

EOD 2017 (INEGI): household survey of the Mexico City metropolitan area; one row per trip with its
modes, start and end time, origin and destination district (194 districts) and expansion factor.
Both sides are restricted to weekday trips whose origin and destination lie in Mexico City
districts covered by the LBS data. On the LBS side a *trip* is a trajectory that goes from one
stay to another (both stays known), moves at least 200 m and has no GPS jumps: the survey does not
record a phone's jitter around a place, nor loops back to the same place. Only regular weekdays are
used (default: 2-22 December 2022, without the 12th, Guadalupe's day; --days and --exclude for other
periods). The comparison is indicative: 2017 vs the LBS period, people's reported trips vs app-location
trajectories (one trajectory per movement between two stays). --home-only keeps the devices with a home
location, closer to the survey's population of residents.

    # six months of ITESM data, Mexico City proper, residents only
    python examples/mexico_city_validation.py /data/out/ITESM --label "LBS Oct 2023-Mar 2024" --no-box \
        --days 2023-10-02:2024-03-22 --exclude 2023-11-01:2023-11-02,2023-11-20,2023-12-12,2023-12-16:2024-01-07,2024-02-05,2024-03-18 \
        --home-only --eod ... --districts ... --name cdmx_eod_itesm_home
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shapely
from make_figures import THEMES, style

from polaroam.geo import haversine

MODES = ["walk", "bike", "motorized"]
LBS = "LBS Dec 2022"  # label of the LBS side (--label)
HOME_ONLY = False
BOX = (19.198, 19.5895, -99.350, -98.9596)  # LBS coverage: lat0, lat1, lon0, lon1
MOTOR = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 17, 18, 19, 20]  # EOD p5_14 codes; 7 bike, 14 walk, 16 bicitaxi


def districts(path: str, cdmx_ids: set[str], box: tuple | None = BOX) -> pl.DataFrame:
    """CDMX districts (at least 95% inside the LBS coverage box, if any), in WGS84."""
    import geopandas as gpd

    g = gpd.read_file(path)  # UTM 14N: areas in m2
    g = g[g["Distrito"].isin(cdmx_ids)]
    if box is not None:
        lat0, lat1, lon0, lon1 = box
        b = gpd.GeoSeries([shapely.box(lon0, lat0, lon1, lat1)], crs=4326).to_crs(g.crs).iloc[0]
        covered = g.geometry.intersection(b).area / g.geometry.area
        g = g[covered >= 0.95]
    g = g.to_crs(4326)
    return pl.DataFrame({"district": g["Distrito"].to_numpy(), "name": g["Descripcio"].to_numpy(),
                         "geometry": shapely.to_wkb(g.geometry.to_numpy())})


def eod_trips(path: str) -> pl.DataFrame:
    v = pl.read_csv(path, infer_schema_length=0)
    motor = pl.any_horizontal([pl.col(f"p5_14_{i:02d}") == "1" for i in MOTOR])
    bike = (pl.col("p5_14_07") == "1") | (pl.col("p5_14_16") == "1")
    minutes = lambda h, m: pl.col(h).cast(pl.Int32) * 60 + pl.col(m).cast(pl.Int32)
    return (v.filter((pl.col("p5_3") == "1") & (pl.col("p5_9_1") != "99") & (pl.col("p5_10_1") != "99"))
            .select(pl.col("dto_origen").alias("o"), pl.col("dto_dest").alias("d"),
                    pl.col("p5_7_7").alias("o_state"), pl.col("p5_12_7").alias("d_state"),
                    pl.col("factor").cast(pl.Float64).alias("w"),
                    pl.col("p5_9_1").cast(pl.Int32).alias("hour"),
                    ((minutes("p5_10_1", "p5_10_2") - minutes("p5_9_1", "p5_9_2")) % 1440).alias("minutes"),
                    pl.when(motor).then(pl.lit("motorized")).when(bike).then(pl.lit("bike"))
                    .when(pl.col("p5_14_14") == "1").then(pl.lit("walk")).alias("mode")))


MIN_DISPLACEMENT = 200  # m between the first and the last position


def _dates(spec: str) -> list[dt.date]:
    """``"2023-12-16:2024-01-07,2024-02-05"`` -> every date in those ranges."""
    out = []
    for part in filter(None, spec.split(",")):
        a, _, b = part.partition(":")
        d0 = dt.date.fromisoformat(a)
        d1 = dt.date.fromisoformat(b) if b else d0
        out += [d0 + dt.timedelta(days=i) for i in range((d1 - d0).days + 1)]
    return out


def regular_weekday(t: pl.Expr, days: str = "2022-12-02:2022-12-22", exclude: str = "2022-12-12") -> pl.Expr:
    """Monday-Friday within ``days`` and not in ``exclude``. Default: 2-22 December 2022, without
    the 12th (Guadalupe's day, a night of pilgrimage); the first day starts mid-stay and the last
    days are the holidays."""
    a, _, b = days.partition(":")
    d = t.dt.date()
    return ((t.dt.weekday() <= 5) & d.is_between(dt.date.fromisoformat(a), dt.date.fromisoformat(b or a))
            & ~d.is_in(_dates(exclude)))


def lbs_trips(out: str, dist: pl.DataFrame, days: str = "2022-12-02:2022-12-22", exclude: str = "2022-12-12",
              home_only: bool = False) -> pl.DataFrame:
    """Weekday trajectories with origin / destination district (first / last vertex)."""
    geoms = shapely.from_wkb(dist["geometry"].to_numpy())
    tree = shapely.STRtree(geoms)
    ids = dist["district"].to_numpy()

    def which(lon, lat):
        pts = shapely.points(lon, lat)
        ip, ig = tree.query(pts, predicate="within")
        res = np.full(len(pts), None, dtype=object)
        res[ip] = ids[ig]
        return pl.Series(res.tolist(), dtype=pl.String)

    homes = (pl.scan_parquet(glob.glob(f"{out}/locations/**/shard=*.parquet", recursive=True)).select("uid").collect()
             if home_only else None)
    parts = []
    for f in sorted(glob.glob(f"{out}/trajectories/**/shard=*.parquet", recursive=True)):
        cols = ["uid", "start_timestamp", "duration", "mode_guess", "coordinates", "origin_stop_event",
                "destination_stop_event", "n_jumps"]
        a_, b_ = pl.col("coordinates").list.first(), pl.col("coordinates").list.last()
        t = pl.read_parquet(f, columns=cols)
        if homes is not None:
            t = t.join(homes, on="uid", how="semi")
        t = (t.filter(pl.col("mode_guess").is_in(MODES))
             .with_columns(pl.from_epoch("start_timestamp", "s").dt.replace_time_zone("UTC")
                           .dt.convert_time_zone("America/Mexico_City").alias("_t"))
             .filter(regular_weekday(pl.col("_t"), days, exclude))
             .filter(pl.col("origin_stop_event").is_not_null() & pl.col("destination_stop_event").is_not_null()
                     & (pl.col("n_jumps") == 0)
                     & (haversine(a_.list.get(1), a_.list.get(0), b_.list.get(1), b_.list.get(0)) >= MIN_DISPLACEMENT))
             .with_columns(a_.alias("_a"), b_.alias("_b")))
        a, b = np.array(t["_a"].to_list()), np.array(t["_b"].to_list())
        parts.append(t.select(pl.col("mode_guess").alias("mode"), pl.col("_t").dt.hour().cast(pl.Int32).alias("hour"),
                              (pl.col("duration") / 60).alias("minutes"))
                     .with_columns(which(a[:, 0], a[:, 1]).alias("o"), which(b[:, 0], b[:, 1]).alias("d"),
                                   pl.lit(1.0).alias("w"))
                     .drop_nulls(["o", "d"]))
    return pl.concat(parts)


def spearman(x, y) -> float:
    rx, ry = pl.Series(x).rank().to_numpy(), pl.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def weighted_median(values, weights) -> float:
    o = np.argsort(values)
    c = np.cumsum(np.asarray(weights)[o])
    return float(np.asarray(values)[o][np.searchsorted(c, c[-1] / 2)])


def compare(eod: pl.DataFrame, lbs: pl.DataFrame, ids: list[str]):
    rows, hours, origins = [], {}, {}
    grid = pl.DataFrame({"o": ids})
    for name, df in (("EOD 2017", eod), (LBS, lbs)):
        tot = df["w"].sum()
        for m in MODES:
            s = df.filter(pl.col("mode") == m)
            rows.append({"source": name, "mode": m, "share": s["w"].sum() / tot,
                         "median_minutes": weighted_median(s["minutes"].to_numpy(), s["w"].to_numpy())})
            h = s.group_by("hour").agg(pl.col("w").sum()).sort("hour")
            prof = np.zeros(24)
            prof[h["hour"].to_numpy()] = h["w"].to_numpy()
            hours[(name, m)] = prof / prof.sum()
            o = grid.join(s.group_by("o").agg(pl.col("w").sum()), on="o", how="left").fill_null(0)
            origins[(name, m)] = o["w"].to_numpy()
    stats = pl.DataFrame(rows)
    corr = []
    for m in MODES:
        corr.append({"mode": m,
                     "hour_profile_r": float(np.corrcoef(hours[("EOD 2017", m)], hours[(LBS, m)])[0, 1]),
                     "origins_spearman": spearman(origins[("EOD 2017", m)], origins[(LBS, m)])})
    # district-to-district flows, all modes
    fe = eod.group_by("o", "d").agg(pl.col("w").sum().alias("eod"))
    fl = lbs.group_by("o", "d").agg(pl.col("w").sum().alias("lbs"))
    od = fe.join(fl, on=["o", "d"], how="full", coalesce=True).fill_null(0)
    od_rho = spearman(od["eod"].to_numpy(), od["lbs"].to_numpy())
    return stats, pl.DataFrame(corr), hours, origins, od_rho, od.height


def fig_validation(stats, corr, hours, origins, t, path):
    c_eod, c_lbs = t["ink2"], t["series"][0]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.08, hspace=0.42, wspace=0.28)
    h = np.arange(24)
    for j, m in enumerate(MODES):
        ax = axes[0, j]
        ax.plot(h, 100 * hours[("EOD 2017", m)], color=c_eod, lw=2, label="EOD 2017 (survey)")
        ax.plot(h, 100 * hours[(LBS, m)], color=c_lbs, lw=2, label=LBS)
        r = corr.filter(pl.col("mode") == m)["hour_profile_r"][0]
        sh = {s: stats.filter((pl.col("source") == s) & (pl.col("mode") == m))["share"][0]
              for s in ("EOD 2017", LBS)}
        ax.set_title(f"{m} · share EOD {sh['EOD 2017']:.0%}, LBS {sh[LBS]:.0%}", loc="left", fontsize=11)
        ax.text(0.98, 0.95, f"r = {r:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=10, color=t["ink2"])
        ax.set_xticks([0, 6, 12, 18, 23]), ax.set_xlim(0, 23)
        ax.set_xlabel("start hour (local)")
        if j == 0:
            ax.set_ylabel("% of the mode's trips")
            ax.legend(frameon=False, fontsize=9, loc="upper left")
        ax2 = axes[1, j]
        e, lb = origins[("EOD 2017", m)], origins[(LBS, m)]
        ok = (e > 0) & (lb > 0)
        ax2.scatter(e[ok], lb[ok], s=14, color=c_lbs, alpha=0.75, edgecolor=t["surface"], linewidth=0.4)
        ax2.set_xscale("log"), ax2.set_yscale("log")
        rho = corr.filter(pl.col("mode") == m)["origins_spearman"][0]
        ax2.set_title(f"{m} · trips per origin district", loc="left", fontsize=11)
        ax2.text(0.04, 0.95, f"Spearman ρ = {rho:.2f}", transform=ax2.transAxes, va="top", fontsize=10,
                 color=t["ink2"])
        ax2.set_xlabel("EOD 2017 (expanded trips per weekday)")
        if j == 0:
            ax2.set_ylabel(f"{LBS} trajectories (weekdays)")
    fig.suptitle("LBS trajectories vs the official Origin-Destination Survey (EOD 2017), Mexico City",
                 x=0.07, y=0.975, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.07, 0.905, "Weekday trips between Mexico City districts covered by the LBS data. LBS trips: "
             "stay to stay, ≥ 200 m, no GPS jumps, regular weekdays;\nmode is the baseline mode_guess"
             + ("; devices with a home location only." if HOME_ONLY else "."),
             fontsize=10, color=t["ink2"])
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--eod", required=True)
    ap.add_argument("--districts", required=True)
    ap.add_argument("--out", default="docs/assets/figures")
    ap.add_argument("--name", default="cdmx_eod", help="figure file name: <name>_{light,dark}.png")
    ap.add_argument("--label", default="LBS Dec 2022", help="name of the LBS data in the figure")
    ap.add_argument("--days", default="2022-12-02:2022-12-22", help="first:last date of the weekdays used")
    ap.add_argument("--exclude", default="2022-12-12", help="dates or first:last ranges to leave out, comma-separated")
    ap.add_argument("--no-box", action="store_true", help="the LBS data cover all of Mexico City (no coverage box)")
    ap.add_argument("--home-only", action="store_true", help="only devices with a home location")
    a = ap.parse_args()
    global LBS, HOME_ONLY
    LBS, HOME_ONLY = a.label, a.home_only
    eod = eod_trips(a.eod)
    cdmx_ids = set(eod.filter(pl.col("o_state") == "09")["o"].unique().to_list())
    dist = districts(a.districts, cdmx_ids, None if a.no_box else BOX)
    ids = sorted(dist["district"].to_list())
    eod = eod.filter(pl.col("o").is_in(ids) & pl.col("d").is_in(ids) & pl.col("mode").is_not_null())
    lbs = lbs_trips(a.output_dir, dist, a.days, a.exclude, a.home_only)
    print(f"{len(ids)} districts; EOD {eod.height:,} survey trips ({eod['w'].sum():,.0f} expanded); "
          f"LBS {lbs.height:,} weekday trajectories")
    stats, corr, hours, origins, od_rho, n_pairs = compare(eod, lbs, ids)
    with pl.Config(tbl_rows=20, tbl_width_chars=140):
        print(stats.pivot(on="source", index="mode", values=["share", "median_minutes"]))
        print(corr)
    print(f"district-to-district flows (all modes): Spearman ρ = {od_rho:.2f} over {n_pairs:,} pairs")
    for theme, t in THEMES.items():
        style(t)
        fig_validation(stats, corr, hours, origins, t, f"{a.out}/{a.name}_{theme}.png")


if __name__ == "__main__":
    main()
