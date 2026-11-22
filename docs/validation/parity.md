# Parity with gc-lbs

The reference is the **Apache Sedona port** of the gc-lbs BigQuery pipeline, run locally on the
same pings. BigQuery itself was not run. The two engines can differ slightly on edge cases:
BigQuery uses `ST_DISTANCE`, `APPROX_QUANTILES` and `ST_CLUSTERDBSCAN`, while Sedona uses
`ST_DistanceSphere`, `percentile_approx` and scikit-learn.

## Example data (11.3M pings, 500 devices)

| Stage | Result |
|---|---|
| Stays | **bit-identical** (times and median coordinates) on every device whose pings have distinct timestamps |
| DBSCAN, given the same stays | identical: every label, count and median |
| Local time + home / work, given the same clusters | identical, including the 24-column schema and types |

## Mexico City subset (24,588 devices, 26M pings, MX configuration)

| Stage | Result |
|---|---|
| Stays, devices without tied timestamps (7,857 devices) | **identical** |
| Stays, all devices | 95.5% identical (see below) |
| DBSCAN, given the same stays | identical except the numbering of exactly tied clusters (18 of 351k stays) |
| Home / work, given the same clusters | identical |
| Home / work locations, end to end | 97.7% identical |
| Time | PolaRoam 19 s (CPU) / 12 s (GPU); Sedona 48 s |

## Pings with the same timestamp

**68%** of the Mexico City devices have pings that share a second but differ in coordinates.
SQL's `ORDER BY timestamp` leaves their order arbitrary, so the stays (and everything
downstream) depend on how the engine happens to order them. Running **Sedona twice** with
different partitioning shows it:

| Comparison | Stays identical | Home / work locations identical |
|---|---|---|
| Sedona vs Sedona (two runs) | 95.8% | 97.9% |
| PolaRoam vs Sedona | 95.5% | 97.7% |

PolaRoam disagrees with Sedona at the same rate Sedona disagrees with itself. PolaRoam orders
tied pings by latitude and longitude, so it always gives the same answer. To make gc-lbs
reproducible too, add the same tie-break to its query: `ORDER BY timestamp, latitude,
longitude`.

## Cluster numbering ties

Clusters are numbered by centroid (mean latitude, then longitude). Coordinates rounded to 5
decimals often give two clusters of one device **exactly** the same mean latitude. gc-lbs then
orders them by floating-point noise in its `AVG`. PolaRoam compares centroids with an exact
fixed-point mean and orders ties by longitude, which keeps labels identical on CPU and GPU and
across batch orders. Only the numbering of those clusters can differ from gc-lbs; their stays,
counts and coordinates do not.
