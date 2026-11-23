"""H3 aggregation and k-anonymity."""

import polars as pl
import pytest

import polaroam as pr

pytest.importorskip("polars_h3")


def locations():
    # 12 homes in one ~block, 3 in another, 7 workers in one place, 1 elsewhere
    rows = ([(f"a{i}", 19.4326, -99.1332, 19.4200, -99.1700) for i in range(7)]
            + [(f"b{i}", 19.4326, -99.1332, None, None) for i in range(5)]
            + [(f"c{i}", 19.3000, -99.2000, None, None) for i in range(3)]
            + [("d", 19.5000, -99.1000, 19.2500, -99.0500)])
    return pl.DataFrame(rows, schema=["uid", "home_latitude", "home_longitude", "work_latitude", "work_longitude"],
                        orient="row")


def test_counts_and_k_anonymity():
    out = pr.home_work_h3(locations(), resolution=10, min_count=5)
    assert out["homes"].drop_nulls().to_list() == [12]           # the 3-home and 1-home cells are suppressed
    assert out["workers"].drop_nulls().to_list() == [7]          # the single worker is suppressed
    raw = pr.home_work_h3(locations(), resolution=10, min_count=1, geometry=False)
    assert raw["homes"].sum() == 16 and raw["workers"].sum() == 8


def test_geometry_and_resolution():
    import polars_h3 as plh3

    out = pr.home_work_h3(locations(), resolution=9, min_count=1)
    assert out.select(plh3.get_resolution("h3_cell")).to_series().unique().to_list() == [9]
    g = out["geometry"][0]
    pts = g.removeprefix("POLYGON ((").removesuffix("))").split(", ")
    assert len(pts) == 7 and pts[0] == pts[-1]                   # closed hexagon ring
    assert out["h3_index"].str.len_chars().unique().to_list() == [15]


def test_h3_counts_generic(pings):
    stays = pr.detect_stops(pings)
    c = pr.h3_counts(stays, "stop_latitude", "stop_longitude", resolution=8, min_count=1, geometry=False)
    assert c["count"].sum() >= stays["uid"].n_unique()            # devices counted per cell
    c2 = pr.h3_counts(stays, "stop_latitude", "stop_longitude", resolution=8, count_unique=None, min_count=1)
    assert c2["count"].sum() == stays.height


def test_coordinate_sinks():
    loc = locations()
    sinks = pr.coordinate_sinks(loc, "home_latitude", "home_longitude", min_devices=10)
    assert sinks.rows() == [(19.4326, -99.1332, 12)]
    kept = pr.drop_sinks(loc, sinks, "home_latitude", "home_longitude").collect()
    assert kept.height == 4 and 19.4326 not in kept["home_latitude"].to_list()


def test_find_sinks_and_pipeline_drop(tmp_path, pings):
    # plant a default-geocode point shared by 20 devices
    devs = pings["uid"].unique().sort().head(20).to_list()
    t0 = int(pings["timestamp"].min())
    planted = pl.DataFrame({"uid": devs, "latitude": [19.4326] * 20, "longitude": [-99.1332] * 20,
                            "timestamp": [t0 + 1000 * i for i in range(20)]})
    p = pl.concat([pings, planted])
    sinks = pr.find_sinks(p, min_devices=15)
    assert sinks.select("latitude", "longitude").rows() == [(19.4326, -99.1332)]
    sinks.write_parquet(tmp_path / "sinks.parquet")
    cfg = pr.PipelineConfig(timezone="America/Mexico_City", remove_sinks=str(tmp_path / "sinks.parquet"))
    pipe = pr.Pipeline(tmp_path / "out", cfg, verbose=False)
    pipe.run(p)
    assert pipe.stops().collect().sort("uid", "stop_event").equals(pr.detect_stops(pings).sort("uid", "stop_event"))
