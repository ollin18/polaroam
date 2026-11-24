"""Validate polaroam's stays, home/work locations and trajectories against the EOD 2017 survey.

    python examples/mexico_city_survey_validation.py /data/out/MX_2022_12 \\
        --eod-dir .../cdmx_survey --districts DistritosEODHogaresZMVM2017.shp --name cdmx_survey_dec

The survey-validation suite used for the LBS stop data (OD heat maps, expansion factors,
departure-time profiles by trip type), redone with polaroam's tables. Every trip-level comparison
is made twice: with **stays** (a trip is two consecutive stays at least 200 m apart; it leaves
when the first stay ends) and with **trajectories** (the pings between the two stays; it
leaves at the first moving ping).

1. **Population and expansion.** Residents per district (devices whose home is there) vs the
   survey's population; expansion factor = population / residents.
2. **Home-work OD.** Devices' home district -> work district, expanded by the home district's
   factor, vs the survey's commuters (household district -> destination of the first work trip).
3. **Trip OD.** Weekday trips between districts: survey vs stays vs trajectories.
4. **Start time by trip type.** HBW (home <-> work; first home->work and last work->home of each
   day), HBO (home <-> other), NHB (neither end at home) and all trips. For stays, the departure
   is unknown between the end of one stay and the start of the next: it is taken as the end of
   the stay (*raw*), uniformly in the gap (*uniform*), or from the survey's own hourly profile
   within the gap (*informed*, as in the original suite: it pulls the LBS towards the survey, so
   it is not an independent test). Trajectories need none of this.
5. **Durations** by trip type, and the **commute**: home-work distance and travel time to work.

Trip types come from the pipeline's ``home_work`` table (location_type H / W / O of each stay).
On the survey side a trip's origin is home if it starts at home, work if the person's previous trip
was to work, other otherwise; its destination is home if it ends at home, work if its purpose is
work. Weekdays only, both ends in Mexico City districts covered by the LBS data. By default only
devices with a home location (the population the survey describes); --all-devices adds the rest
(they only contribute NHB and all trips).
"""

from __future__ import annotations

import argparse
import glob
import pickle
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shapely
from make_figures import THEMES, style
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from mexico_city_validation import BOX, districts, regular_weekday

from polaroam.geo import haversine

TZ = "America/Mexico_City"
TYPES = ["HBW", "HBO", "NHB", "all"]
METHODS = ["raw", "uniform", "informed"]
MIN_DISPLACEMENT = 200  # m between the two stays / the trajectory's ends
MAX_MIN = 600  # duration histograms: 1-minute bins up to 10 h
LBS = "LBS"


# --------------------------------------------------------------------------------------- survey


def eod(eod_dir: str, ids: list[str]):
    """Weekday survey trips with origin / destination activity, population and home-work pairs."""
    d = Path(eod_dir)
    v = pl.read_csv(d / "tviaje_eod2017/conjunto_de_datos/tviaje.csv", infer_schema_length=0)
    v = (v.filter(pl.col("p5_3") == "1")
         .with_columns(pl.col("n_via").cast(pl.Int32), pl.col("factor").cast(pl.Float64).alias("w"))
         .sort("id_soc", "n_via")
         .with_columns(pl.col("p5_13").shift(1).over("id_soc").alias("prev")))
    act_o = (pl.when(pl.col("p5_6") == "01").then(pl.lit("H")).when(pl.col("prev") == "02").then(pl.lit("W"))
             .otherwise(pl.lit("O")))
    act_d = (pl.when(pl.col("p5_11a") == "01").then(pl.lit("H")).when(pl.col("p5_13") == "02").then(pl.lit("W"))
             .otherwise(pl.lit("O")))
    minutes = lambda h, m: pl.col(h).cast(pl.Int32) * 60 + pl.col(m).cast(pl.Int32)
    trips = (v.filter((pl.col("p5_9_1") != "99") & (pl.col("p5_10_1") != "99"))
             .select("id_soc", "n_via", "w", pl.col("dto_origen").alias("o"), pl.col("dto_dest").alias("d"),
                     act_o.alias("o_act"), act_d.alias("d_act"), pl.col("p5_9_1").cast(pl.Int32).alias("hour"),
                     ((minutes("p5_10_1", "p5_10_2") - minutes("p5_9_1", "p5_9_2")) % 1440).alias("minutes"))
             .filter(pl.col("o").is_in(ids) & pl.col("d").is_in(ids)))
    viv = pl.read_csv(d / "tvivienda_eod2017/conjunto_de_datos/tvivienda.csv", infer_schema_length=0)
    pop = (viv.select(pl.col("distrito").alias("district"),
                      (pl.col("p1_1").cast(pl.Float64) * pl.col("factor").cast(pl.Float64)).alias("population"))
           .group_by("district").agg(pl.col("population").sum()).filter(pl.col("district").is_in(ids)))
    sdem = pl.read_csv(d / "tsdem_eod2017/conjunto_de_datos/tsdem.csv", infer_schema_length=0,
                       columns=["id_soc", "distrito"])
    hw = (v.filter(pl.col("p5_13") == "02").group_by("id_soc").agg(pl.all().sort_by("n_via").first())
          .join(sdem, on="id_soc")
          .select(pl.col("distrito").alias("home"), pl.col("dto_dest").alias("work"), "w")
          .filter(pl.col("home").is_in(ids) & pl.col("work").is_in(ids)))
    return trips, pop, hw


def trip_type(o: str = "o_act", d: str = "d_act") -> pl.Expr:
    o, d = pl.col(o), pl.col(d)
    return (pl.when(((o == "H") & (d == "W")) | ((o == "W") & (d == "H"))).then(pl.lit("HBW"))
            .when((o == "H") | (d == "H")).then(pl.lit("HBO")).otherwise(pl.lit("NHB")).alias("type"))


def commutes(df: pl.DataFrame, by: list[str], t: str) -> pl.DataFrame:
    """HBW trips of one day: the first home->work and the last work->home of each ``by`` group."""
    hw = df.filter((pl.col("o_act") == "H") & (pl.col("d_act") == "W")).sort(t).group_by(by).first()
    wh = df.filter((pl.col("o_act") == "W") & (pl.col("d_act") == "H")).sort(t).group_by(by).last()
    return pl.concat([hw, wh.select(hw.columns)])


# ------------------------------------------------------------------------------------- geography


class Districts:
    def __init__(self, shp: str, ids: list[str]):
        import geopandas as gpd

        g = gpd.read_file(shp)
        g = g[g["Distrito"].isin(ids)].sort_values("Distrito")  # UTM 14N
        self.ids = g["Distrito"].to_numpy()
        self.gdf = g.to_crs(4326)
        c = g.geometry.centroid
        self.cx, self.cy = c.x.to_numpy(), c.y.to_numpy()
        self.radius = shapely.minimum_bounding_radius(g.geometry.to_numpy())  # m, for trips within a district
        self.tree = shapely.STRtree(self.gdf.geometry.to_numpy())

    def which(self, lon, lat) -> pl.Series:
        pts = shapely.points(np.asarray(lon, float), np.asarray(lat, float))
        ip, ig = self.tree.query(pts, predicate="within")
        res = np.full(len(pts), None, dtype=object)
        res[ip] = self.ids[ig]
        return pl.Series(res.tolist(), dtype=pl.String)

    def distance(self, a: pl.Series, b: pl.Series) -> np.ndarray:
        """Centroid-to-centroid distance in m; a district's bounding radius within it."""
        idx = {k: i for i, k in enumerate(self.ids)}
        ia, ib = np.array([idx[x] for x in a]), np.array([idx[x] for x in b])
        dd = np.hypot(self.cx[ia] - self.cx[ib], self.cy[ia] - self.cy[ib])
        return np.where(ia == ib, self.radius[ia], dd)


# ------------------------------------------------------------------------------------------- LBS


def _local(c: str) -> pl.Expr:
    return pl.from_epoch(c, "s").dt.replace_time_zone("UTC").dt.convert_time_zone(TZ)


def _tod(c: pl.Expr) -> pl.Expr:
    return c.dt.hour().cast(pl.Int64) * 3600 + c.dt.minute().cast(pl.Int64) * 60 + c.dt.second().cast(pl.Int64)


def informed(t0: np.ndarray, t1: np.ndarray, prof: np.ndarray, rng) -> np.ndarray:
    """Departure (seconds of the day) between t0 and t1, its hour drawn from the survey profile."""
    h0, h1 = t0 // 3600, t1 // 3600
    ok = (np.arange(24)[None, :] >= h0[:, None]) & (np.arange(24)[None, :] <= h1[:, None])
    p = np.where(ok, prof[None, :] + 1e-6, 0.0).cumsum(axis=1)
    h = (p < rng.random(len(t0))[:, None] * p[:, -1:]).sum(axis=1)
    return np.clip(h * 3600 + rng.integers(0, 60, len(t0)) * 60, t0, t1)


class Acc:
    """Sums over shards: hour profiles, duration histograms, OD counts."""

    def __init__(self):
        self.hours = defaultdict(lambda: np.zeros(24))
        self.minutes = defaultdict(lambda: np.zeros(MAX_MIN + 1))
        self.od = defaultdict(list)
        self.n = defaultdict(int)

    def add(self, src, df, hour_cols, minutes):
        for typ, part in [*df.partition_by("type", as_dict=True).items(), (("all",), None)]:
            typ = typ[0]
            part = df if part is None else part
            if typ == "HBW":
                continue  # added from the day's commutes only
            self._add(src, typ, part, hour_cols, minutes)

    def _add(self, src, typ, part, hour_cols, minutes):
        for method, col in hour_cols.items():
            self.hours[(src, method, typ)] += np.bincount(part[col].to_numpy(), minlength=24)[:24]
        m = np.clip(np.rint(part[minutes].to_numpy()), 0, MAX_MIN).astype(int)
        self.minutes[(src, typ)] += np.bincount(m, minlength=MAX_MIN + 1)
        self.n[(src, typ)] += part.height
        if typ == "all":
            self.od[src].append(part.group_by("o", "d").len())


def lbs(out: str, dist: Districts, days: str, exclude: str, all_devices: bool, profiles: dict, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    acc = Acc()
    loc = pl.scan_parquet(glob.glob(f"{out}/locations/**/*.parquet", recursive=True)).collect()
    homes = loc.select("uid")
    shards = sorted(glob.glob(f"{out}/home_work/**/shard=*.parquet", recursive=True),
                    key=lambda f: int(re.search(r"shard=(\d+)", f).group(1)))
    for i, f in enumerate(shards):
        k = re.search(r"shard=(\d+)", f).group(1)
        s = pl.read_parquet(f, columns=["uid", "stop_event", "start_timestamp", "end_timestamp", "stop_latitude",
                                        "stop_longitude", "location_type"])
        if not all_devices:
            s = s.join(homes, on="uid", how="semi")
        # ---- stays: consecutive stays at least 200 m apart, same regular weekday
        nx = lambda c: pl.col(c).shift(-1).over("uid")
        st = (s.sort("uid", "stop_event")
              .with_columns(nx("start_timestamp").alias("t1"), nx("stop_latitude").alias("lat1"),
                            nx("stop_longitude").alias("lon1"), nx("location_type").alias("d_act"))
              .rename({"location_type": "o_act"})
              .filter(pl.col("t1").is_not_null() & (pl.col("t1") > pl.col("end_timestamp")))
              .filter(haversine(pl.col("stop_latitude"), pl.col("stop_longitude"), pl.col("lat1"), pl.col("lon1"))
                      >= MIN_DISPLACEMENT)
              .with_columns(_local("end_timestamp").alias("dep"), _local("t1").alias("arr"))
              .filter(regular_weekday(pl.col("dep"), days, exclude) & (pl.col("dep").dt.date() == pl.col("arr").dt.date()))
              .with_columns(trip_type(), pl.col("dep").dt.date().alias("date"), _tod(pl.col("dep")).alias("t0"),
                            _tod(pl.col("arr")).alias("tt1"), ((pl.col("t1") - pl.col("end_timestamp")) / 60).alias("min")))
        st = st.with_columns(dist.which(st["stop_longitude"], st["stop_latitude"]).alias("o"),
                             dist.which(st["lon1"], st["lat1"]).alias("d")).drop_nulls(["o", "d"])
        t0, t1 = st["t0"].to_numpy(), st["tt1"].to_numpy()
        typ = st["type"].to_numpy()
        inf = np.empty(len(t0), dtype=np.int64)
        for name in ("HBW", "HBO", "NHB"):
            m = typ == name
            inf[m] = informed(t0[m], t1[m], profiles[name], rng)
        st = st.with_columns(pl.Series("h_raw", t0 // 3600), pl.Series("h_uniform", (t0 + rng.random(len(t0)) * (t1 - t0)) // 3600),
                             pl.Series("h_informed", inf // 3600)).with_columns(pl.col("^h_.*$").cast(pl.Int64))
        cols = {m: f"h_{m}" for m in METHODS}
        acc.add("stays", st, cols, "min")
        acc._add("stays", "HBW", commutes(st, ["uid", "date"], "dep"), cols, "min")
        # ---- trajectories of this shard, every batch
        types = s.select("uid", "stop_event", "location_type")
        parts = []
        for tf in glob.glob(f"{out}/trajectories/**/shard={k}.parquet", recursive=True):
            a_, b_ = pl.col("coordinates").list.first(), pl.col("coordinates").list.last()
            tr = (pl.read_parquet(tf, columns=["uid", "start_timestamp", "end_timestamp", "duration", "n_jumps",
                                               "origin_stop_event", "destination_stop_event", "coordinates"])
                  .filter(pl.col("origin_stop_event").is_not_null() & pl.col("destination_stop_event").is_not_null()
                          & (pl.col("n_jumps") == 0))
                  .join(types.rename({"stop_event": "origin_stop_event", "location_type": "o_act"}),
                        on=["uid", "origin_stop_event"])
                  .join(types.rename({"stop_event": "destination_stop_event", "location_type": "d_act"}),
                        on=["uid", "destination_stop_event"])
                  .with_columns(a_.list.get(0).alias("lon0"), a_.list.get(1).alias("lat0"),
                                b_.list.get(0).alias("lon1"), b_.list.get(1).alias("lat1"))
                  .drop("coordinates")
                  .filter(haversine(pl.col("lat0"), pl.col("lon0"), pl.col("lat1"), pl.col("lon1")) >= MIN_DISPLACEMENT)
                  .with_columns(_local("start_timestamp").alias("dep"), _local("end_timestamp").alias("arr"))
                  .filter(regular_weekday(pl.col("dep"), days, exclude) & (pl.col("dep").dt.date() == pl.col("arr").dt.date())))
            parts.append(tr)
        if parts:
            tr = pl.concat(parts)
            tr = (tr.with_columns(dist.which(tr["lon0"], tr["lat0"]).alias("o"), dist.which(tr["lon1"], tr["lat1"]).alias("d"))
                  .drop_nulls(["o", "d"])
                  .with_columns(trip_type(), pl.col("dep").dt.date().alias("date"),
                                pl.col("dep").dt.hour().cast(pl.Int64).alias("h"), (pl.col("duration") / 60).alias("min")))
            acc.add("trajectories", tr, {"observed": "h"}, "min")
            acc._add("trajectories", "HBW", commutes(tr, ["uid", "date"], "dep"), {"observed": "h"}, "min")
        print(f"  shard {k} ({i + 1}/{len(shards)}): {st.height:,} stay trips, {tr.height if parts else 0:,} trajectories",
              flush=True)
    # ---- home / work locations
    loc = loc.with_columns(dist.which(loc["home_longitude"], loc["home_latitude"]).alias("home"))
    loc = loc.with_columns(pl.when(pl.col("work_latitude").is_not_null())
                           .then(dist.which(loc["work_longitude"].fill_null(0), loc["work_latitude"].fill_null(0)))
                           .alias("work"))
    residents = loc.drop_nulls("home").group_by("home").len("residents")
    hw = loc.drop_nulls(["home", "work"])
    exact = haversine(hw["home_latitude"], hw["home_longitude"], hw["work_latitude"], hw["work_longitude"]).to_numpy()
    od = {src: pl.concat(v).group_by("o", "d").agg(pl.col("len").sum().alias("n")) for src, v in acc.od.items()}
    return {"hours": dict(acc.hours), "minutes": dict(acc.minutes), "n": dict(acc.n), "od": od,
            "residents": residents, "hw": hw.group_by("home", "work").len("n"),
            "hw_dist": hw.select("home", "work"), "hw_exact_m": exact}


# ---------------------------------------------------------------------------------- comparisons


def dtw(x, y) -> float:
    """Dynamic time warping distance (L1) between two hour profiles in %, divided by 48 as in the
    original suite."""
    n, m = len(x), len(y)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = abs(x[i - 1] - y[j - 1]) + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))


def spearman(x, y) -> float:
    return float(np.corrcoef(pl.Series(x).rank().to_numpy(), pl.Series(y).rank().to_numpy())[0, 1])


def logr(x, y) -> float:
    ok = (x > 0) & (y > 0)
    return float(np.corrcoef(np.log10(x[ok]), np.log10(y[ok]))[0, 1])


def wquantile(v, w, q):
    o = np.argsort(v)
    c = np.cumsum(np.asarray(w, float)[o])
    return float(np.asarray(v)[o][np.searchsorted(c, q * c[-1])])


def survey_side(trips: pl.DataFrame):
    t = trips.with_columns(trip_type())
    hours, minutes, n = {}, {}, {}
    for typ in TYPES:
        part = commutes(t, ["id_soc"], "n_via") if typ == "HBW" else t if typ == "all" else t.filter(pl.col("type") == typ)
        h = np.zeros(24)
        g = part.group_by("hour").agg(pl.col("w").sum())
        h[g["hour"].to_numpy() % 24] = g["w"].to_numpy()
        hours[typ] = h
        m = np.clip(part["minutes"].to_numpy(), 0, MAX_MIN)
        minutes[typ] = np.bincount(m, weights=part["w"].to_numpy(), minlength=MAX_MIN + 1)
        n[typ] = part.height
    return hours, minutes, n


def matrix(df: pl.DataFrame, o: str, d: str, v: str, ids) -> np.ndarray:
    idx = {k: i for i, k in enumerate(ids)}
    M = np.zeros((len(ids), len(ids)))
    np.add.at(M, ([idx[x] for x in df[o]], [idx[x] for x in df[d]]), df[v].to_numpy())
    return M


def compute(a) -> dict:
    ids = sorted(districts(a.districts, set(_cdmx(a.eod_dir)), None if a.no_box else BOX)["district"].to_list())
    dist = Districts(a.districts, ids)
    trips, pop, hw_eod = eod(a.eod_dir, ids)
    s_hours, s_min, s_n = survey_side(trips)
    prof = {k: v / v.sum() for k, v in s_hours.items()}
    L = lbs(a.output_dir, dist, a.days, a.exclude, a.all_devices, prof)
    # population and expansion
    P = pop.join(L["residents"].rename({"home": "district"}), on="district", how="left").fill_null(0)
    P = P.with_columns(pl.when(pl.col("residents") > 0).then(pl.col("population") / pl.col("residents"))
                       .alias("expansion"))
    # home-work OD
    E = P.select(pl.col("district").alias("home"), "expansion")
    hw_l = L["hw"].join(E, on="home").with_columns((pl.col("n") * pl.col("expansion")).alias("expanded"))
    hw_e = hw_eod.group_by("home", "work").agg(pl.col("w").sum())
    M_l, M_e = matrix(hw_l, "home", "work", "expanded", ids), matrix(hw_e, "home", "work", "w", ids)
    # trip OD
    eod_all = trips.group_by("o", "d").agg(pl.col("w").sum())
    T = {"EOD 2017": matrix(eod_all, "o", "d", "w", ids)}
    for src in ("stays", "trajectories"):
        T[src] = matrix(L["od"][src], "o", "d", "n", ids)
    # commute distances, district level (both) and exact (LBS)
    d_eod = dist.distance(hw_eod["home"], hw_eod["work"])
    d_lbs = dist.distance(L["hw_dist"]["home"], L["hw_dist"]["work"])
    w_lbs = L["hw_dist"].join(E, on="home", how="left")["expansion"].fill_null(0).to_numpy()
    return {"ids": ids, "gdf": dist.gdf[["Distrito", "geometry"]], "pop": P, "M_hw": (M_e, M_l), "T": T,
            "s_hours": s_hours, "s_min": s_min, "s_n": s_n, "L": {k: L[k] for k in ("hours", "minutes", "n")},
            "commute": {"eod": (d_eod, hw_eod["w"].to_numpy()), "lbs": (d_lbs, w_lbs), "exact": L["hw_exact_m"]}}


def _cdmx(eod_dir: str) -> list[str]:
    v = pl.read_csv(Path(eod_dir) / "tviaje_eod2017/conjunto_de_datos/tviaje.csv", infer_schema_length=0,
                    columns=["p5_7_7", "dto_origen"])
    return v.filter(pl.col("p5_7_7") == "09")["dto_origen"].unique().to_list()


def summary(R) -> pl.DataFrame:
    rows = []
    for typ in TYPES:
        e = R["s_hours"][typ] / R["s_hours"][typ].sum() * 100
        for src, methods in (("stays", METHODS), ("trajectories", ["observed"])):
            for m in methods:
                h = R["L"]["hours"].get((src, m, typ))
                if h is None or h.sum() == 0:
                    continue
                h = h / h.sum() * 100
                mins = R["L"]["minutes"][(src, typ)]
                rows.append({"type": typ, "source": src, "departure": m, "trips": R["L"]["n"][(src, typ)],
                             "hour_r": float(np.corrcoef(e, h)[0, 1]), "dtw": dtw(e, h),
                             "median_min": float(np.searchsorted(np.cumsum(mins), mins.sum() / 2))})
        sm = R["s_min"][typ]
        rows.append({"type": typ, "source": "EOD 2017", "departure": "reported", "trips": R["s_n"][typ],
                     "hour_r": 1.0, "dtw": 0.0, "median_min": float(np.searchsorted(np.cumsum(sm), sm.sum() / 2))})
    return pl.DataFrame(rows)


def od_stats(R) -> pl.DataFrame:
    rows = []
    M_e, M_l = R["M_hw"]
    eye = np.eye(len(R["ids"]), dtype=bool)
    for name, (A, B) in {"home-work": (M_e, M_l), "trips: stays": (R["T"]["EOD 2017"], R["T"]["stays"]),
                         "trips: trajectories": (R["T"]["EOD 2017"], R["T"]["trajectories"])}.items():
        rows.append({"od": name, "spearman_all_pairs": spearman(A.ravel(), B.ravel()),
                     "log_r": logr(A.ravel(), B.ravel()), "log_r_intra": logr(A[eye], B[eye]),
                     "log_r_inter": logr(A[~eye], B[~eye]), "intra_share_eod": A[eye].sum() / A.sum(),
                     "intra_share_lbs": B[eye].sum() / B.sum()})
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------------------- figures


def seq(t):
    c = ["#e8f0fb", t["series"][0], "#0d2b52"] if t["surface"] > "#8" else ["#17263b", t["series"][0], "#d7e6fa"]
    cm = LinearSegmentedColormap.from_list("seq", c)
    cm.set_bad(t["surface"])
    return cm


def _title(fig, title, sub, t):
    fig.suptitle(title, x=0.06, y=0.975, ha="left", fontsize=13, fontweight="bold", color=t["ink"])
    fig.text(0.06, 0.925, sub, fontsize=9.5, color=t["ink2"], va="top")


def fig_population(R, t, path):
    g = R["gdf"].merge(R["pop"].to_pandas(), left_on="Distrito", right_on="district")
    _population(g, R["pop"], "district", "districts", "EOD 2017", "household members × expansion factor", t, path)


def fig_population_ageb(R, t, path):
    A = R["ageb"]
    g = A["gdf"].merge(A["pop"].to_pandas(), on="geomid")
    _population(g, A["pop"], "AGEB", "urban AGEBs", "Census 2020", "residents counted by INEGI", t, path,
                f"{A['in_ageb']:.1%} of the devices with a home in Mexico City live in an urban AGEB. "
                f"{(A['pop']['residents'] == 0).sum():,} AGEBs have no device.")


def _population(g, P, unit, units, src, how, t, path, extra=""):
    many = P.height > 300
    fig, ax = plt.subplots(2, 3, figsize=(13, 8.6), gridspec_kw={"height_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.07, hspace=0.3, wspace=0.25)
    g = g.assign(population=g["population"].where(g["population"] > 0), residents=g["residents"].where(g["residents"] > 0))
    for a, col, name in zip(ax[0], ["population", "residents", "expansion"],
                            [f"population ({src})", "LBS residents (devices with a home)",
                             "expansion factor = population / residents"]):
        v = g[col].dropna().to_numpy()
        norm = LogNorm(*np.percentile(v, [2, 98])) if many else LogNorm()
        g.plot(column=col, ax=a, cmap=seq(t), norm=norm, legend=True, edgecolor=t["surface"],
               linewidth=0.05 if many else 0.4, legend_kwds={"shrink": 0.7},
               missing_kwds={"color": t["grid"], "edgecolor": t["grid"], "linewidth": 0.05})
        a.set_title(name, loc="left", fontsize=10.5), a.set_axis_off()
    e = P["expansion"].drop_nulls().to_numpy()
    ax[1, 0].hist(e, bins=np.logspace(np.log10(e.min()), np.log10(e.max()), 40 if many else 30), color=t["series"][0],
                  edgecolor=t["surface"], linewidth=0.6 if not many else 0.3)
    ax[1, 0].set_xscale("log"), ax[1, 0].set_xlabel("expansion factor"), ax[1, 0].set_ylabel(units)
    ax[1, 0].set_title(f"expansion factor · median {np.median(e):,.0f}", loc="left", fontsize=10.5)
    x, y = P["residents"].to_numpy().astype(float), P["population"].to_numpy()
    ok = (x > 0) & (y > 0)
    kw = {"s": 5, "alpha": 0.5, "edgecolor": "none"} if many else {"s": 16, "edgecolor": t["surface"], "linewidth": 0.4}
    a = ax[1, 1]
    a.scatter(x[ok], y[ok], color=t["series"][0], **kw)
    a.set_xscale("log"), a.set_yscale("log")
    a.set_xlabel(f"{LBS} residents"), a.set_ylabel(f"population ({src})")
    a.set_title(f"residents vs population · Spearman ρ = {spearman(x[ok], y[ok]):.2f}", loc="left", fontsize=10.5)
    a = ax[1, 2]
    px, py = y[ok] / y[ok].sum() * 100, x[ok] / x[ok].sum() * 100
    a.scatter(px, py, color=t["series"][0], **kw)
    if many:
        lo, hi = min(px.min(), py.min()) * 0.8, max(px.max(), py.max()) * 1.2
        a.set_xscale("log"), a.set_yscale("log")
        for f, ls in ((1, "--"), (2, ":"), (0.5, ":")):
            a.plot([lo, hi], [lo * f, hi * f], color=t["muted"], lw=1, ls=ls)
        title = f"share per {unit} (- - equal, ··· ×2 and ×½)"
    else:
        lo, hi = 0, max(px.max(), py.max()) * 1.05
        a.plot([lo, hi], [lo, hi], color=t["muted"], lw=1, ls="--")
        title = f"share per {unit} (dashed: equal)"
    a.set_xlim(lo, hi), a.set_ylim(lo, hi)
    a.set_xlabel("% of the population"), a.set_ylabel(f"% of {LBS} residents")
    a.set_title(title, loc="left", fontsize=10.5)
    _title(fig, f"Where devices live vs where people live, Mexico City {units}",
           f"{LBS}: devices whose home location is in the {unit}. {src}: {how}. "
           f"Scaled residents (× expansion factor) equal\nthe population by construction. {extra}", t)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def ageb_population(out: str, path: str) -> dict:
    """Devices with a home per urban AGEB, against the census population of the AGEB."""
    import geopandas as gpd

    g = gpd.read_file(path).to_crs(4326)  # geomid, population, geometry
    loc = pl.scan_parquet(glob.glob(f"{out}/locations/**/*.parquet", recursive=True)).select(
        "home_longitude", "home_latitude").collect()
    tree = shapely.STRtree(g.geometry.to_numpy())
    pts = shapely.points(loc["home_longitude"].to_numpy(), loc["home_latitude"].to_numpy())
    ip, ig = tree.query(pts, predicate="within")
    ids = g["geomid"].to_numpy()
    res = pl.DataFrame({"geomid": ids[ig]}).group_by("geomid").len("residents")
    P = (pl.DataFrame({"geomid": ids, "population": g["population"].to_numpy().astype(float)})
         .join(res, on="geomid", how="left").with_columns(pl.col("residents").fill_null(0))
         .with_columns(pl.when((pl.col("residents") > 0) & (pl.col("population") > 0))
                       .then(pl.col("population") / pl.col("residents")).alias("expansion")))
    city = gpd.read_file(Path(__file__).parent / "data/cdmx_boundary.geojson").to_crs(4326).union_all()
    n_city = int(shapely.contains_xy(city, loc["home_longitude"].to_numpy(), loc["home_latitude"].to_numpy()).sum())
    return {"gdf": g[["geomid", "geometry"]], "pop": P, "in_ageb": len(np.unique(ip)) / n_city}


def _heat(ax, M, t, norm, title):
    im = ax.imshow(np.where(M > 0, M, np.nan), cmap=seq(t), norm=norm, interpolation="nearest")
    ax.set_title(title, loc="left", fontsize=10.5)
    ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    return im


def _scatter(ax, A, B, t, xl, yl, title):
    eye = np.eye(len(A), dtype=bool)
    for m, c, name in ((~eye, t["series"][1], "between districts"), (eye, t["series"][0], "within a district")):
        ok = (A[m] > 0) & (B[m] > 0)
        ax.scatter(A[m][ok], B[m][ok], s=6 if name[0] == "b" else 16, alpha=0.5 if name[0] == "b" else 0.9,
                   color=c, edgecolor="none", label=name)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlabel(xl), ax.set_ylabel(yl), ax.set_title(title, loc="left", fontsize=10.5)
    ax.text(0.04, 0.96, f"log r = {logr(A.ravel(), B.ravel()):.2f} (within {logr(A[eye], B[eye]):.2f}, "
            f"between {logr(A[~eye], B[~eye]):.2f})\nSpearman ρ, all pairs = {spearman(A.ravel(), B.ravel()):.2f}",
            transform=ax.transAxes, va="top", fontsize=9, color=t["ink2"])
    ax.legend(loc="lower right", fontsize=8.5, markerscale=1.5)


def fig_hw_od(R, t, path):
    M_e, M_l = R["M_hw"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    fig.subplots_adjust(left=0.04, right=0.98, top=0.8, bottom=0.12, wspace=0.18)
    hi = max(M_e.max(), M_l.max())
    norm = LogNorm(hi / 1e4, hi)
    _heat(ax[0], M_e, t, norm, "EOD 2017: commuters (expanded)")
    im = _heat(ax[1], M_l, t, norm, f"{LBS}: devices × expansion factor")
    for a in ax[:2]:
        a.set_xlabel("work district →"), a.set_ylabel("home district →")
    fig.colorbar(im, ax=ax[:2], shrink=0.8, pad=0.02).outline.set_visible(False)
    _scatter(ax[2], M_e, M_l, t, "EOD 2017 commuters", f"{LBS} (expanded)", "pairs of districts")
    _title(fig, "Home-work flows between Mexico City districts",
           f"{LBS}: devices with a home and a work location (pipeline `locations`), expanded by their home district. "
           "EOD 2017: household district → destination\nof each worker's first weekday work trip. Districts in survey order (001…).",
           t)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_trip_od(R, t, path):
    T = R["T"]
    S = {k: v / v.sum() * 100 for k, v in T.items()}
    fig = plt.figure(figsize=(15, 9.4))
    gs = fig.add_gridspec(2, 3, left=0.05, right=0.98, top=0.86, bottom=0.07, hspace=0.32, wspace=0.22)
    hi = max(v.max() for v in S.values())
    norm = LogNorm(hi / 1e4, hi)
    for j, (k, title) in enumerate((("EOD 2017", "EOD 2017 (expanded weekday trips)"), ("stays", f"{LBS} stays"),
                                    ("trajectories", f"{LBS} trajectories"))):
        a = fig.add_subplot(gs[0, j])
        im = _heat(a, S[k], t, norm, title)
        a.set_xlabel("destination district →"), a.set_ylabel("origin district →")
    fig.colorbar(im, ax=fig.axes, shrink=0.45, pad=0.01, anchor=(0, 1), label="% of trips").outline.set_visible(False)
    for j, k in enumerate(("stays", "trajectories")):
        _scatter(fig.add_subplot(gs[1, j]), S["EOD 2017"], S[k], t, "EOD 2017 (% of trips)", f"{LBS} {k} (% of trips)",
                 f"{k} vs survey")
    a = fig.add_subplot(gs[1, 2])
    eye = np.eye(len(R["ids"]), dtype=bool)
    vals = [S[k][eye].sum() for k in S]
    a.bar(range(3), vals, color=[t["ink2"], t["series"][0], t["series"][1]], width=0.6)
    for i, v in enumerate(vals):
        a.text(i, v + 0.8, f"{v:.0f}%", ha="center", fontsize=9.5, color=t["ink2"])
    a.set_xticks(range(3), ["EOD 2017", "stays", "trajectories"]), a.set_ylim(0, max(vals) * 1.2)
    a.set_title("trips within one district", loc="left", fontsize=10.5), a.set_ylabel("% of trips")
    _title(fig, "Weekday trips between Mexico City districts: survey, stays and trajectories",
           f"{LBS} stays: consecutive stays ≥ 200 m apart, same regular weekday. Trajectories: from one stay to another, "
           "≥ 200 m, no GPS jumps. Shares of each source's trips.", t)
    fig.savefig(path, dpi=130)
    plt.close(fig)


NAMES = {"HBW": "home ↔ work (HBW)", "HBO": "home ↔ other (HBO)", "NHB": "non-home-based (NHB)", "all": "all trips"}


def fig_hours(R, t, path):
    fig, ax = plt.subplots(2, 4, figsize=(16, 8.2))
    fig.subplots_adjust(left=0.05, right=0.985, top=0.8, bottom=0.08, hspace=0.38, wspace=0.22)
    h = np.arange(24)
    c_s, c_t = t["series"][0], t["series"][1]
    for j, typ in enumerate(TYPES):
        a = ax[0, j]
        e = R["s_hours"][typ] / R["s_hours"][typ].sum() * 100
        a.plot(h, e, color=t["ink2"], lw=2.4, label="EOD 2017")
        notes = []
        for m, ls in (("raw", "-"), ("uniform", "--"), ("informed", ":")):
            v = R["L"]["hours"][("stays", m, typ)]
            v = v / v.sum() * 100
            a.plot(h, v, color=c_s, lw=1.6, ls=ls, label=f"stays, {m}")
            notes.append(f"stays {m}: r {np.corrcoef(e, v)[0, 1]:.2f}, DTW {dtw(e, v):.2f}")
        v = R["L"]["hours"][("trajectories", "observed", typ)]
        v = v / v.sum() * 100
        a.plot(h, v, color=c_t, lw=2, label="trajectories")
        notes.append(f"trajectories: r {np.corrcoef(e, v)[0, 1]:.2f}, DTW {dtw(e, v):.2f}")
        a.set_title(NAMES[typ], loc="left", fontsize=11)
        top = a.get_ylim()[1]
        a.text(0.02, 0.98, "\n".join(notes), transform=a.transAxes, va="top", fontsize=7.8, color=t["ink2"])
        a.set_xticks([0, 6, 12, 18, 23]), a.set_xlim(0, 23), a.set_ylim(0, top * 1.45)
        a.set_xlabel("departure hour (local)")
        if j == 0:
            a.set_ylabel("% of trips")
        b = ax[1, j]
        x = np.arange(MAX_MIN + 1)
        for k, c, lw, name in ((R["s_min"][typ], t["ink2"], 2.4, "EOD 2017 (reported)"),
                               (R["L"]["minutes"][("stays", typ)], c_s, 1.6, "stays (gap between them)"),
                               (R["L"]["minutes"][("trajectories", typ)], c_t, 2, "trajectories")):
            cdf = np.cumsum(k) / k.sum()
            b.plot(x[1:], cdf[1:] * 100, color=c, lw=lw, label=f"{name} · median {np.searchsorted(cdf, 0.5)} min")
        b.set_xscale("log"), b.set_xlim(1, MAX_MIN), b.set_ylim(0, 100)
        b.set_xlabel("trip duration (minutes)"), b.legend(fontsize=7.8, loc="upper left")
        if j == 0:
            b.set_ylabel("% of trips (cumulative)")
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.05, 0.875), ncol=5,
               fontsize=9, frameon=False)
    _title(fig, "When trips start and how long they take, by trip type",
           f"{LBS}, regular weekdays. Stays: the departure lies between two stays; raw = end of the first stay, uniform = "
           "anywhere in the gap,\ninformed = drawn from the survey's hourly profile within the gap (not independent of the "
           "survey). Trajectories: first moving ping. HBW: first home→work and last work→home of each day.", t)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_commute(R, t, path):
    C = R["commute"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.8, bottom=0.13, wspace=0.22)
    a = ax[0]
    for (d, w), c, lw, name in ((C["eod"], t["ink2"], 2.4, "EOD 2017, district to district"),
                                (C["lbs"], t["series"][0], 2, f"{LBS}, district to district"),
                                ((C["exact"], np.ones(len(C["exact"]))), t["series"][1], 1.6, f"{LBS}, home to work location")):
        o = np.argsort(d)
        cdf = np.cumsum(w[o]) / w.sum()
        a.plot(d[o] / 1000, cdf * 100, color=c, lw=lw, label=f"{name} · median {wquantile(d, w, 0.5) / 1000:.1f} km")
    a.set_xscale("log"), a.set_xlim(0.1, 50), a.set_ylim(0, 100)
    a.set_xlabel("home-work distance (km)"), a.set_ylabel("% of workers (cumulative)")
    a.set_title("distance to work", loc="left", fontsize=11), a.legend(fontsize=8.5, loc="upper left")
    b = ax[1]
    x = np.arange(MAX_MIN + 1)
    sm = R["s_min"]["HBW"]
    for k, c, lw, name in ((sm, t["ink2"], 2.4, "EOD 2017 (reported)"),
                           (R["L"]["minutes"][("stays", "HBW")], t["series"][0], 1.6, "stays (gap)"),
                           (R["L"]["minutes"][("trajectories", "HBW")], t["series"][1], 2, "trajectories")):
        cdf = np.cumsum(k) / k.sum()
        b.plot(x[1:], cdf[1:] * 100, color=c, lw=lw, label=f"{name} · median {np.searchsorted(cdf, 0.5)} min")
    b.set_xscale("log"), b.set_xlim(1, MAX_MIN), b.set_ylim(0, 100)
    b.set_xlabel("commute duration, HBW trips (minutes)"), b.set_title("travel time to and from work", loc="left", fontsize=11)
    b.legend(fontsize=8.5, loc="upper left")
    _title(fig, "The commute: distance and travel time",
           "District to district: centroid distance, or the district's bounding radius within it (as in the survey "
           f"analysis), {LBS} weighted by\nthe expansion factor. Home to work location: straight line between the "
           "device's two locations.", t)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("output_dir")
    ap.add_argument("--eod-dir", required=True, help="folder with the EOD 2017 tviaje_, tvivienda_ and tsdem_eod2017 folders")
    ap.add_argument("--districts", required=True, help="DistritosEODHogaresZMVM2017.shp")
    ap.add_argument("--out", default="docs/assets/figures")
    ap.add_argument("--name", default="cdmx_survey")
    ap.add_argument("--label", default="LBS Dec 2022")
    ap.add_argument("--days", default="2022-12-02:2022-12-22")
    ap.add_argument("--exclude", default="2022-12-12")
    ap.add_argument("--no-box", action="store_true", help="the LBS data cover all of Mexico City")
    ap.add_argument("--all-devices", action="store_true", help="also devices without a home location")
    ap.add_argument("--cache", help="pickle of the computed results: reused if it exists")
    ap.add_argument("--agebs", required=True, help="urban AGEBs: GeoJSON with geomid and the census population")
    a = ap.parse_args()
    global LBS
    LBS = a.label
    if a.cache and Path(a.cache).exists():
        R = pickle.loads(Path(a.cache).read_bytes())
    else:
        R = compute(a)
        if a.cache:
            Path(a.cache).write_bytes(pickle.dumps(R))
    if "ageb" not in R:
        R["ageb"] = ageb_population(a.output_dir, a.agebs)
        if a.cache:
            Path(a.cache).write_bytes(pickle.dumps(R))
    with pl.Config(tbl_rows=40, tbl_width_chars=160, float_precision=3):
        print(f"{len(R['ids'])} districts")
        print(od_stats(R))
        print(summary(R))
    for theme, t in THEMES.items():
        style(t)
        fig_population_ageb(R, t, f"{a.out}/{a.name}_population_ageb_{theme}.png")
        for name, f in (("population", fig_population), ("hw_od", fig_hw_od), ("trip_od", fig_trip_od),
                        ("hours", fig_hours), ("commute", fig_commute)):
            f(R, t, f"{a.out}/{a.name}_{name}_{theme}.png")


if __name__ == "__main__":
    main()
