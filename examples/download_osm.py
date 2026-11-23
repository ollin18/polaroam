"""Download the OpenStreetMap road networks of Mexico City (drive, bike, walk) with osmnx and
save them as polaroam road tables (parquet: u, v, length, highway, oneway, geometry as WKB).

    pip install osmnx
    python examples/download_osm.py /data/osm/cdmx

Data (c) OpenStreetMap contributors, ODbL 1.0.
"""

import json
import sys
import time
from pathlib import Path

import osmnx as ox
import polars as pl
import shapely

BOUNDARY = Path(__file__).parent / "data" / "cdmx_boundary.geojson"


def main(out: str) -> None:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gj = json.loads(BOUNDARY.read_text())
    geom = gj["features"][0]["geometry"] if gj.get("type") == "FeatureCollection" else gj
    poly = shapely.geometry.shape(geom)
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(out_dir / "cache")
    for kind in ("drive", "bike", "walk"):
        t = time.time()
        g = ox.graph_from_polygon(poly, network_type=kind, simplify=True, retain_all=False)
        edges = ox.graph_to_gdfs(g, nodes=False).reset_index()
        hw = edges["highway"].map(lambda h: h if isinstance(h, str) else ",".join(map(str, h)))
        df = pl.DataFrame({
            "u": edges["u"].to_numpy(), "v": edges["v"].to_numpy(), "length": edges["length"].to_numpy(),
            "highway": hw.to_numpy(), "oneway": edges["oneway"].astype(bool).to_numpy(),
            "geometry": shapely.to_wkb(edges.geometry.to_numpy()),
        })
        df.write_parquet(out_dir / f"{kind}.parquet")
        print(f"{kind}: {g.number_of_nodes():,} nodes, {df.height:,} directed edges ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
