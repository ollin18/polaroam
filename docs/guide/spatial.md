# Heat maps with H3

Homes, work places and stays can be aggregated on [H3](https://h3geo.org) hexagons with
`polaroam.h3`, through the Polars-native [polars-h3](https://github.com/Filimoa/polars-h3)
plugin (`pip install -e ".[h3]"`, Python ≥ 3.11).

```python
cells = pr.home_work_h3(pipe.locations(), resolution=10)       # homes and workers per cell
cells = pipe.locations_h3(resolution=10)                         # the same, from a pipeline

stays = pr.h3_counts(pipe.stops(), "stop_latitude", "stop_longitude", resolution=9)
```

Each row has the cell id (`h3_cell`, UInt64, and `h3_index`, the usual hex string), its center
and a WKT `POLYGON`. The output opens directly in QGIS or GeoPandas:

```python
import geopandas as gpd

gdf = gpd.GeoDataFrame(cells.to_pandas(), geometry=gpd.GeoSeries.from_wkt(cells["geometry"].to_list()),
                       crs="EPSG:4326")
```

## k-anonymity

Fine resolutions isolate individuals: at resolution 10 a cell can hold a single home. **Cells
with fewer than `min_count` devices (default 5) are dropped.** In `home_work_h3`, a count below
`min_count` is set to null, and the cell is dropped when both counts are null.
`min_count=1` disables the protection, for internal use only.

How much survives depends on resolution and density. In one month of Mexico City (370k homes,
58.8k work places), with k = 5:

| H3 res | Hexagon edge | Homes kept | Workers kept |
|---|---|---|---|
| 8 | ~531 m | 100% | 100% |
| 9 | ~201 m | 100% | 92% |
| 10 | ~76 m | 92% | 44% |
| 11 | ~29 m | 66% | 23% |

Homes support block-level maps (res 10–11). Work places, which are fewer, are best at res 9.

![Homes, work places and jobs vs residents in Mexico City](../assets/figures/cdmx_h3_light.png#only-light)
![Homes, work places and jobs vs residents in Mexico City](../assets/figures/cdmx_h3_dark.png#only-dark)

The right panel compares jobs with residents per res-8 cell. It plots
\(\log_2\big((\text{workers}_c / W) / (\text{homes}_c / H)\big)\), where \(W\) and \(H\) are
the city totals: red cells have more jobs than residents relative to the city, blue cells the
opposite. A coarse resolution keeps both counts above k everywhere. At finer resolutions,
requiring ≥ k workers would select only job-dense cells and bias the map red.

![Roma, Condesa and Juárez at block level](../assets/figures/cdmx_h3_zoom_light.png#only-light)
![Roma, Condesa and Juárez at block level](../assets/figures/cdmx_h3_zoom_dark.png#only-dark)

## Data quality: coordinate sinks

Some coordinates are shared by thousands of unrelated devices. These are default geocodes
(e.g. a city centroid used when only the IP is known), cell towers, or Wi-Fi positioning
points. Stays and homes detected there are artifacts, and they dominate fine-grained maps.

```python
sinks = pr.coordinate_sinks(pipe.locations(), "home_latitude", "home_longitude", min_devices=50)
clean = pr.drop_sinks(pipe.locations(), sinks, "home_latitude", "home_longitude")
```

In one month of Mexico City:

- **2,592 devices** had their "home" at exactly 19.4326, -99.1332: the Zócalo, the textbook
  centroid of the city.
- Five more single points held 1,000–1,500 homes each.
- Overall, **65,963 homes (18%)** sit on exact coordinates shared by 50 or more devices.

At ~1 m coordinate precision, that many devices on the same point means snapped locations,
not tall buildings. The maps above exclude them.

!!! warning "Sinks affect stays, not only maps"
    A device whose pings snap to a tower or a default point can get a stay, and a home, there.
    `coordinate_sinks` and `drop_sinks` also work on pings (`latitude` / `longitude`). Filtering
    pings at sink coordinates *before* stay detection removes these artifacts at the source;
    choose `min_devices` for your data (more devices, more legitimate sharing).
