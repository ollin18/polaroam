# Stays

A **stay** (stop) is a period in which a device remains in one place. PolaRoam detects stays
with the rules of the gc-lbs BigQuery query (`process_yearly_stays.sh`), translated line by line
into Polars expressions.

```python
stops = pr.detect_stops(pings)                                         # defaults: 20 m, 3600 s, 300 s
stops = pr.detect_stops(pings, distance_threshold=30, min_stop_duration=600)
stops = pr.detect_stops(pings, pr.StopParams(20, 3600, 300), column_map={"uid": "device_id"})
```

## The rules

1. **De-duplicate and order.** Identical pings are removed, and each device's pings are
   ordered by time.
2. **Stationary pings.** A ping is stationary if it is within `distance_threshold` meters
   *and* `time_threshold` seconds of its next ping, **or** of its previous ping.
3. **Stay starts.** A new stay starts at a stationary ping whose previous ping is not
   stationary, or is farther than `distance_threshold` (a jump between two places).
4. **Duration.** Stays shorter than `min_stop_duration` are dropped.
5. **Outliers.** The stay location is the median of its pings. Pings farther than
   `distance_threshold` from it are removed, the stay is re-aggregated (median, start, end),
   and the duration is checked again.

| Parameter | Default | Meaning |
|---|---|---|
| `distance_threshold` | 20 m | max distance between consecutive pings of a stay; max distance to its median |
| `time_threshold` | 3600 s | max gap between consecutive pings of a stay |
| `min_stop_duration` | 300 s | minimum stay duration |

Distances are haversine on a sphere of radius 6,371,008.8 m (BigQuery's `ST_DISTANCE`). The stay
location is the *lower median* of its pings, an observed value, like BigQuery's
`APPROX_QUANTILES(x, 100)[OFFSET(50)]` and Spark's `percentile_approx`.

## How it is vectorized

SQL's `LAG` / `LEAD OVER (PARTITION BY device ORDER BY time)` becomes a global shift masked
where the device changes. SQL's three-valued logic (NULL for the first and last ping of a
device) maps directly to Polars' Kleene logic:

```python
same_next = pl.col("uid").shift(-1) == pl.col("uid")
dist = pl.when(same_next).then(haversine(lat, lon, lat.shift(-1), lon.shift(-1)))
stationary = ((dist <= D) & ((tdiff <= T) | tdiff.is_null())) \
           | ((prev_dist <= D) & ((prev_tdiff <= T) | prev_tdiff.is_null()))
```

The stay id is a cumulative sum of stay starts per device. The aggregations are group-bys and a
join back to the medians. There is no per-device Python code, so Polars uses every core, and the
same query runs on the [GPU](cuda.md).

## Output

| Column | Type | Description |
|---|---|---|
| `uid` | str | device id |
| `stop_event` | i64 | stay number per device, 1, 2, … in time order |
| `start_timestamp`, `end_timestamp` | i64 | Unix seconds |
| `stop_duration` | i64 | seconds |
| `stop_latitude`, `stop_longitude` | f64 | lower median of the stay's pings |
| `n_pings` | u32 | pings in the stay (after outlier removal) |

!!! info "Pings with the same timestamp"
    Many sources report two pings in the same second with different coordinates. SQL's
    `ORDER BY timestamp` leaves their order arbitrary, so BigQuery and Spark can give different
    stays run to run. PolaRoam orders ties by latitude and longitude and is always
    deterministic. See [Parity with gc-lbs](../validation/parity.md).

![Stay duration and stays per day](../assets/figures/distributions_light.png#only-light)
![Stay duration and stays per day](../assets/figures/distributions_dark.png#only-dark)
