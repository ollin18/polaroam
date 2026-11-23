"""Bundled timezone polygons, the grid lookup and the pipeline's timezone cache."""

import dataclasses
from pathlib import Path

import numpy as np
import polars as pl
import pytest

shapely = pytest.importorskip("shapely")

import polaroam as pr
from polaroam import local_time as lt

GC_LBS_CSV = Path(__file__).resolve().parents[2] / "bigquery_lbs" / "gc-lbs" / "tz_codes.csv"

CITIES = [  # (lat, lon, tz)
    (19.4326, -99.1332, "America/Mexico_City"),
    (21.1619, -86.8515, "America/Cancun"),
    (32.5149, -117.0382, "America/Tijuana"),
    (29.0729, -110.9559, "America/Hermosillo"),
    (-23.5505, -46.6333, "America/Sao_Paulo"),
    (-3.1190, -60.0217, "America/Manaus"),
    (4.7110, -74.0721, "America/Bogota"),
    (14.5995, 120.9842, "Asia/Manila"),
    (-6.2088, 106.8456, "Asia/Jakarta"),
    (0.0, -30.0, None),  # Atlantic: outside every polygon
]


def exact(points, polygons="builtin", country_code=None):
    """Reference: every point tested against every polygon."""
    geoms, names = lt._tz_tree(polygons, country_code)
    return lt._point_in_polygon(points, geoms, names, "stop_latitude", "stop_longitude")


def test_builtin_polygons_known_cities():
    tz = pr.builtin_timezones()
    assert tz.height == 418 and tz["tz"].n_unique() > 300
    pts = pl.DataFrame({"stop_latitude": [c[0] for c in CITIES], "stop_longitude": [c[1] for c in CITIES]})
    assert pr.lookup_timezones(pts, "builtin")["timezone"].to_list() == [c[2] for c in CITIES]


@pytest.mark.skipif(not GC_LBS_CSV.exists(), reason="gc-lbs tz_codes.csv not available")
def test_builtin_equals_gc_lbs_csv():
    tz = pl.read_csv(GC_LBS_CSV, infer_schema_length=0).filter(pl.col("geometry").str.contains(r"^\s*(MULTI)?POLYGON"))
    src = shapely.from_wkt(tz["geometry"].to_numpy())
    src = np.array([shapely.MultiPolygon([g]) if g.geom_type == "Polygon" else g for g in src], dtype=object)
    b = pr.builtin_timezones()
    assert b["tz"].to_list() == tz["tz"].to_list() and b["country_code"].to_list() == tz["country_code"].to_list()
    assert shapely.equals_exact(shapely.from_wkb(b["geometry"].to_numpy()), src, tolerance=0).all()  # bit-identical coordinates


@pytest.mark.parametrize("cell", [1.0, 0.1, 0.013])
def test_grid_lookup_is_exact(cell):
    """Points dense along borders, on cell edges, and on polygon vertices give the brute-force answer."""
    rng = np.random.default_rng(1)
    geoms = shapely.from_wkb(pr.builtin_timezones().filter(pl.col("country_code").is_in(["MX", "BR", "US"]))["geometry"].to_numpy())
    verts = shapely.get_coordinates(geoms)[::50]
    near = verts + rng.normal(0, 0.05, verts.shape)                       # around borders
    grid = np.round(near / cell) * cell                                   # exactly on cell edges
    x0, y0, x1, y1 = shapely.total_bounds(geoms)
    uni = np.column_stack([rng.uniform(x0, x1, 50_000), rng.uniform(y0, y1, 50_000)])
    xy = np.vstack([near, grid, verts, uni])
    pts = pl.DataFrame({"stop_latitude": xy[:, 1], "stop_longitude": xy[:, 0]}).unique(maintain_order=True)
    lt._CELLS.clear()
    got = lt.lookup_timezones(pts, "builtin", cell_deg=cell)
    assert got.equals(exact(pts))
    assert got["timezone"].n_unique() > 20
    # second call reuses the classified cells and still agrees
    assert lt.lookup_timezones(pts.reverse(), "builtin", cell_deg=cell).equals(exact(pts.reverse()))


def test_csv_path_and_country_filter(tmp_path):
    tz = pr.builtin_timezones().filter(pl.col("country_code").is_in(["MX", "CO"]))
    csv = tmp_path / "tz.csv"
    tz.with_columns(pl.Series("geometry", shapely.to_wkt(shapely.from_wkb(tz["geometry"].to_numpy()), rounding_precision=-1))).write_csv(csv)
    pts = pl.DataFrame({"stop_latitude": [c[0] for c in CITIES], "stop_longitude": [c[1] for c in CITIES]})
    by_csv = pr.lookup_timezones(pts, str(csv))
    assert by_csv.equals(exact(pts, str(csv)))
    assert by_csv["timezone"].to_list()[:4] == [c[2] for c in CITIES[:4]] and by_csv["timezone"][4] is None
    only_co = pr.lookup_timezones(pts, "builtin", "CO")["timezone"].to_list()
    assert only_co[6] == "America/Bogota" and only_co[0] is None


def test_pipeline_timezone_cache(tmp_path, pings, monkeypatch):
    """Daily runs look up each stay coordinate once; results equal a one-shot run."""
    cfg = pr.PipelineConfig(find_timezones=True, n_shards=2,
                            home_work=pr.HomeWorkParams(home_min_days=3, work_min_days=2))
    one = pr.Pipeline(tmp_path / "one", cfg, verbose=False)
    one.run(pings)
    assert set(one.home_work().collect()["timezone"].unique()) == {"America/Mexico_City"}

    looked_up = []
    real = lt.lookup_timezones

    def counting(points, *a, **k):
        looked_up.append(points.height)
        return real(points, *a, **k)

    monkeypatch.setattr(lt, "lookup_timezones", counting)
    inc = pr.Pipeline(tmp_path / "inc", dataclasses.replace(cfg), verbose=False)
    days = pings.with_columns(pl.from_epoch("timestamp", "ms").dt.date().alias("_d"))
    for i, ((d,), g) in enumerate(days.sort("_d").group_by("_d", maintain_order=True)):
        # every third day skips home/work: the cache must survive those batches
        inc.update(g.drop("_d"), batch_id=str(d), compute_home_work=i % 3 != 1)
    inc.finalize()
    for name, key in [("clusters", ["uid", "stop_event"]), ("home_work", ["uid", "stop_event"]), ("locations", ["uid"])]:
        assert getattr(one, name)().collect().sort(key).equals(getattr(inc, name)().collect().sort(key)), name
    n_coords = inc.stops().select("stop_latitude", "stop_longitude").unique().collect().height
    assert sum(looked_up) == n_coords  # each coordinate looked up exactly once
    cached = sum(pl.read_parquet(f).height for f in (tmp_path / "inc" / "_state").glob("shard=*/timezones.parquet"))
    assert cached == n_coords


def test_timezone_required_suggests_from_sample(tmp_path, pings):
    """No timezone and no lookup: fail before any work, suggesting the data's timezone."""
    pipe = pr.Pipeline(tmp_path / "out", pr.PipelineConfig(), verbose=False)
    with pytest.raises(pr.TimezoneRequired, match='timezone="America/Mexico_City"') as e:
        pipe.run(pings)
    msg = str(e.value)
    assert "random sample of 100 points" in msg and "find_timezones=True" in msg
    assert "several UTC offsets" not in msg  # one timezone: lookup not recommended
    assert not (tmp_path / "out").exists()  # nothing was written
    with pytest.raises(pr.TimezoneRequired):
        pr.add_local_time(pr.detect_stops(pings.head(5000)))


def test_suggestion_for_several_timezones():
    cdmx = pl.DataFrame({"latitude": 19.4 + np.arange(60) * 1e-3, "longitude": [-99.13] * 60})
    tijuana = pl.DataFrame({"latitude": 32.5 + np.arange(40) * 1e-3, "longitude": [-117.03] * 40})
    pts = pl.concat([cdmx, tijuana])
    s = pr.suggest_timezone(pts)
    assert s.rows() == [("America/Mexico_City", s["utc_offset"][0], 60), ("America/Tijuana", s["utc_offset"][1], 40)]
    msg = str(lt.timezone_required(pts))
    assert 'timezone="America/Mexico_City"' in msg and "several UTC offsets" in msg


def test_timezone_options():
    with pytest.raises(ValueError, match="Unknown timezone"):
        pr.PipelineConfig(timezone="Mexico City")
    with pytest.warns(DeprecationWarning):
        assert pr.PipelineConfig(tz_polygons="builtin").find_timezones is True
    with pytest.warns(DeprecationWarning):
        assert pr.PipelineConfig(tz_polygons="x.csv").tz_source == "x.csv"
    assert pr.PipelineConfig(find_timezones=True).tz_source == "builtin"
    assert pr.PipelineConfig(timezone="UTC").tz_source is None


def test_cli_timezone_required(tmp_path, pings, capsys):
    from polaroam.cli import main

    src = tmp_path / "p.parquet"
    pings.write_parquet(src)
    assert main(["run", str(src), str(tmp_path / "out")]) == 2
    assert 'timezone="America/Mexico_City"' in capsys.readouterr().err
    assert main(["run", str(src), str(tmp_path / "out"), "--find-timezones"]) == 0
    assert set(pr.Pipeline(tmp_path / "out").home_work().collect()["inputed_timezone"].unique()) == {"America/Mexico_City"}
