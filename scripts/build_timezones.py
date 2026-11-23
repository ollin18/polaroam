"""Build ``src/polaroam/data/timezones.npz`` from a ``tz,geometry,country_code`` CSV of WKT polygons.

    python scripts/build_timezones.py path/to/tz_codes.csv

Coordinates are stored as delta-encoded int32 microdegrees: the source has at most 6
decimals, so decoding (``cumsum / 1e6``) gives back exactly the floats parsed from the WKT.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import shapely
from shapely import GeometryType

OUT = Path(__file__).resolve().parents[1] / "src" / "polaroam" / "data" / "timezones.npz"


def main(csv: str) -> None:
    tz = (pl.read_csv(csv, infer_schema_length=0)
          .filter(pl.col("geometry").str.contains(r"^\s*(MULTI)?POLYGON")))  # skips a repeated header row
    decimals = tz["geometry"].str.extract_all(r"\.\d+").list.eval(pl.element().str.len_chars()).list.max().max() - 1
    assert decimals <= 6, f"coordinates with {decimals} decimals cannot be stored exactly as microdegrees"
    geoms = shapely.from_wkt(tz["geometry"].to_numpy())
    multi = np.array([shapely.MultiPolygon([g]) if g.geom_type == "Polygon" else g for g in geoms], dtype=object)
    kind, coords, offsets = shapely.to_ragged_array(multi)
    assert kind == GeometryType.MULTIPOLYGON
    q = np.round(coords * 1e6).astype(np.int64)
    assert np.array_equal(q / 1e6, coords) and np.abs(q).max() < 2**31
    deltas = np.diff(q, axis=0, prepend=np.zeros((1, 2), np.int64)).astype(np.int32)
    np.savez_compressed(OUT, tz=tz["tz"].to_numpy().astype(str), country_code=tz["country_code"].to_numpy().astype(str),
                        coords=deltas, ring_offsets=offsets[0], polygon_offsets=offsets[1],
                        geometry_offsets=offsets[2])
    back = shapely.from_ragged_array(GeometryType.MULTIPOLYGON, np.cumsum(deltas, axis=0, dtype=np.int64) / 1e6, offsets)
    assert shapely.equals_exact(back, multi, tolerance=0).all()
    print(f"{len(tz)} polygons, {len(coords):,} vertices -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1])
