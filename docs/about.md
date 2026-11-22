# About

PolaRoam is developed by Ollin Langle. It is a Polars-native, incremental reimplementation of
the gc-lbs location-based-services pipeline (BigQuery, with an Apache Sedona port), extended
with trajectories, speed features and transport mode.

## Credits

- **Stays, DBSCAN, local time and home / work** follow the gc-lbs BigQuery pipeline and its
  Apache Sedona port.
- **Incremental clustering** follows Ester, Kriegel, Sander, Wimmer and Xu, *Incremental
  Clustering for Mining in a Data Warehousing Environment*, VLDB 1998.
- **Line simplification** is Douglas and Peucker, *Algorithms for the reduction of the number
  of points required to represent a digitized line or its caricature*, 1973.
- Built on [Polars](https://pola.rs) and [cudf-polars](https://docs.rapids.ai/api/cudf/stable/cudf_polars/).
- Basemaps: Esri World Gray Canvas (Esri, HERE, Garmin, © OpenStreetMap contributors).

## Privacy

Mobility pings are personal data. The figures in this documentation only show aggregates over
many devices, and single-trip illustrations use a relative frame without a basemap. When
publishing results, aggregate home and work locations and avoid showing individual
trajectories on a map.

## License

MIT.
