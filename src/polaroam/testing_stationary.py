import polars as pl
import os
import numpy as np
import bisect
from sklearn.neighbors import BallTree
from scipy.spatial import ConvexHull, QhullError

from infomap import Infomap
from sklearn.cluster import DBSCAN

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

data_dir = os.path.join("/","data","Berkeley","MX","data_quadrant")
df = pl.scan_parquet(os.path.join(data_dir,"filtered_localized","**","*.parquet"))
df.collect_schema()

df = df.filter(pl.col("uid")==4787392011392)

input_df = df.collect()

def haversine_polars(lat1, lon1):
    # Convert latitude and longitude from degrees to radians
    lat1_rad = pl.col(lat1).radians()
    lon1_rad = pl.col(lon1).radians()
    lat2_rad = pl.col(lat1).shift(-1).radians()
    lon2_rad = pl.col(lon1).shift(-1).radians()

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
            ((dlat / 2).sin()).pow(2) +
            lat1_rad.cos() * lat2_rad.cos() * ((dlon / 2).sin()).pow(2)

         )

    c = 2 * a.sqrt().arcsin()

    # Radius of Earth in meters
    R = 6371000
    distance = R * c

    return distance

def euclidean_polars(lat1, lon1):
    lat1_col = pl.col(lat1)
    lon1_col = pl.col(lon1)
    lat2_col = pl.col(lat1).shift(-1)
    lon2_col = pl.col(lon1).shift(-1)

    # Haversine formula
    dlat = lat2_col - lat1_col
    dlon = lon2_col - lon1_col

    distance = (
            (dlat.pow(2) + dlon.pow(2)).sqrt()
         )

    return distance

# Example usage with Polars DataFrame
def calculate_distances(coords, distance_metric):
    dict_dist = {"haversine" : haversine_polars
                 ,"euclidean" : euclidean_polars
                 }
    if distance_metric in dict_dist:
        coords = coords.with_columns([
            dict_dist[distance_metric](
                'latitude', 'longitude'
            ).alias('distance'),
            (pl.col('timestamp').shift(-1) - pl.col('timestamp')).alias('time_diff')
        ])
    else:
        raise ValueError("Unsupported distance metric")

    return coords

coords = input_df.select(['uid', 'latitude', 'longitude', 'timestamp'])
coords = coords.sort(["uid","timestamp"])

distance_metric = "haversine"
r_C = 20
max_staying_time = 3600

coords = calculate_distances(coords, distance_metric)

coords = coords.with_columns([
    (pl.col('distance') <= r_C).alias('within_radius'),
    (pl.col('time_diff').is_null() | (pl.col('time_diff') <= max_staying_time)).alias('within_time')
])

# Find clusters of points that meet the stationary criteria
coords = coords.with_columns([
(pl.col('within_radius') & pl.col('within_time')).alias('stationary')
])

coords = coords.with_columns([
    (pl.col('stationary') & (~pl.col('stationary').shift(1, fill_value=False))).cast(pl.Int32).alias('event_change')
])

# Create event IDs by cumulatively summing up the transitions
#  coords = coords.with_columns([
#      (pl.col('event_change').cum_sum().alias('event_id'))
#  ])

coords = coords.with_columns([
    pl.when(pl.col('stationary'))
    .then(pl.col('event_change').cum_sum())
    .otherwise(-1)
    .alias('event_id')
])

# Filter events based on min_size and min_staying_time
min_size = 2
min_staying_time = 300

event_stats = coords.group_by('event_id').agg([
    pl.col('event_id').count().alias('event_size'),
    pl.col('time_diff').sum().alias('total_time')
]).filter(
    (pl.col('event_size') >= min_size) & (pl.col('total_time') >= min_staying_time)
)

# Join coords with event_stats to filter valid event_ids
#  coords = coords.join(
#      event_stats.select("event_id"),
#      on="event_id",
#      how="left"
#  )

coords = coords.filter(pl.col("event_id") != -1).join(
    event_stats.select("event_id"),
    on="event_id",
    how="inner"
)

# Now replace null event_ids with -1
coords = coords.with_columns(
    pl.col("event_id").fill_null(-1)
)

coords = coords.with_columns([
    pl.when(pl.col('stationary'))
    .then(pl.col('event_change').cum_sum())
    .otherwise(-1)
    .alias('event_id')
])

coords = coords.with_columns(
    pl.concat_list(["latitude", "longitude"]).list.to_array(2).alias("stat_coords")
)

results = coords.select(
    pl.col("uid"),
    pl.col("event_id").cast(pl.Int64).alias("stop_events"),
    pl.col("stat_coords").alias("event_maps"),
    pl.col('timestamp')
)

labels_and_coords = results.filter(pl.col("stop_events") != -1)
labels_and_coords = labels_and_coords.with_columns(pl.col("event_maps").arr.get(0).alias("latitude"),
                pl.col("event_maps").arr.get(1).alias("longitude"))

median_coords = (labels_and_coords
                       .group_by(["uid", "stop_events"], maintain_order=True)
                       .agg(pl.col("latitude").median()
                            , pl.col("longitude").median()
                            , pl.col("timestamp").min().alias("start_timestamp")
                            , pl.col("timestamp").max().alias("end_timestamp")
                        )
                    )


def downsample(df, min_spacial_resolution):
    if min_spacial_resolution > 0:
        coords_df = (
            df.with_columns(
                ((pl.col("latitude") / min_spacial_resolution).round() * min_spacial_resolution),
                ((pl.col("longitude") / min_spacial_resolution).round() * min_spacial_resolution)
            ).with_columns(
                pl.concat_list(["latitude", "longitude"]).alias("coords")
            )
        )
    else:
        coords_df = df.with_columns([
        pl.concat_list(["latitude", "longitude"]).alias('coords')
    ])

    coords_df = coords_df.with_row_index("index")

    unique_coords_df = (coords_df.group_by(["uid", "latitude", "longitude"], maintain_order=True)
                        .agg(pl.col('index')
                            , pl.col('stop_events') # this changed
                            , pl.len().alias("count")
                            , pl.col('coords').first()
                            , pl.col('start_timestamp')
                            , pl.col('end_timestamp')
                            )
                        )

    return unique_coords_df

coords_df = median_coords.with_columns([
        pl.concat_list(["latitude", "longitude"]).alias('coords')
    ])

coords_df = coords_df.with_row_index("index")

unique_coords_df = (coords_df.group_by(["uid", "latitude", "longitude"], maintain_order=True)
                    .agg(pl.col('index')
                        , pl.col('stop_events') # this changed
                        , pl.len().alias("count")
                        , pl.col('coords').first()
                        , pl.col('start_timestamp')
                        , pl.col('end_timestamp')
                        )
                    )
unique_coords_df.filter(pl.col("count")>1)


min_spacial_resolution = 0
r2=20
num_threads=30

stat_coords = downsample(median_coords, min_spacial_resolution)
# Infomap
stat_coords = stat_coords.lazy()


def cluster_dbscan(coords, r2, distance_metric="haversine", num_threads=1):
    # Extract the coordinates from the Polars DataFrame
    coords_list = np.array(coords.select("coords").to_numpy().tolist()).reshape(-1, 2)

    # Convert to radians if using haversine distance
    if distance_metric == "haversine":
        coords_list = np.radians(coords_list)
        r2 = r2 / 6371000  # Earth radius in meters
    # Apply DBSCAN
    db = DBSCAN(eps=r2, min_samples=2, metric=distance_metric, n_jobs=num_threads).fit(coords_list)
    labels = db.labels_

    return labels



output_schema = {"uid":pl.String
                 ,"stop_events":pl.Int64
                 ,"inverse_indices":pl.List(pl.UInt32)
                 ,"latitude":pl.Float64
                 ,"longitude":pl.Float64
                 ,"start_timestamp":pl.Int64
                 ,"end_timestamp":pl.Int64
                 ,"stop_locations":pl.Int64
                 }
stat_labels = (stat_coords
                        .group_by("uid", maintain_order=True)
                        .map_groups(lambda df: pl.DataFrame({
                            "uid":df.select(pl.col('uid'))
                            ,"stop_events":df.select(pl.col("stop_events"))
                            ,"inverse_indices":df.select(pl.col("index"))
                            ,"latitude":df.select(pl.col('latitude'))
                            ,"longitude":df.select(pl.col('longitude'))
                            ,"start_timestamp":df.select(pl.col("start_timestamp"))
                            ,"end_timestamp":df.select(pl.col('end_timestamp'))
                            ,"stop_locations":cluster_dbscan(df
                                                , r2
                                                , distance_metric
                                                , num_threads)
                                            })
                                        , schema = output_schema
                                    )
                                )
#  stat_labels = stat_labels.collect()
stat_labels.filter(pl.col("stop_locations") > 0)
##################
##################
################## This has to be added!
##################
filtered_df = stat_labels.filter((pl.col("stop_events").list.len() > 1) & (pl.col("stop_locations") == -1))

filtered_df_with_new_locations = filtered_df.with_columns(
    (pl.col("stop_locations").max() + pl.arange(1, pl.len() + 1)).alias("new_stop_locations")
)


stat_labels = stat_labels.join(
    filtered_df_with_new_locations.select(["uid", "stop_events", "new_stop_locations"]),  # Select relevant columns
    on=["uid", "stop_events"],  # Join on these columns to ensure proper update
    how="left"  # Keep all rows, but update the filtered ones
).with_columns(
    # Update stop_locations with new_stop_locations where applicable
    pl.when(pl.col("new_stop_locations").is_not_null())
    .then(pl.col("new_stop_locations"))
    .otherwise(pl.col("stop_locations"))
    .alias("stop_locations")
).drop("new_stop_locations")
stat_labels.collect()

##################
##################
##################
##################


stop_labels = stat_labels.explode(["inverse_indices","stop_events","start_timestamp","end_timestamp"]).sort("inverse_indices")

stat_labels.filter(pl.col("stop_locations") != -1).collect()
stop_labels.filter(pl.col("stop_locations") != -1).collect()

stop_labels = stop_labels.lazy()
output_schema = {"uid":pl.String
                 ,"stop_locations":pl.Int64
                 ,"cluster_counts":pl.UInt32
                 ,"cluster_latitude":pl.Float64
                 ,"cluster_longitude":pl.Float64
                 }
stop_medoid = (stop_labels
               .group_by(["uid","stop_locations"], maintain_order=True)
               .map_groups(lambda df: pl.DataFrame({
                            "uid":df.select(pl.col('uid').first())
                            ,"stop_locations":df.select(pl.col("stop_locations").first())
                            ,"cluster_counts":df.select(pl.len())
                            ,"cluster_latitude":df.select(pl.col("latitude")).median()
                            ,"cluster_longitude":df.select(pl.col("longitude")).median()
                        })
                    , schema = output_schema
                    )
                    .with_columns(
                        cluster_counts = pl.when(pl.col("stop_locations") == -1)
                        .then(1)
                        .otherwise(pl.col("cluster_counts"))
                    )
               )

stop_labels = stop_labels.join(stop_medoid, on=["uid", "stop_locations"], how="left")
stop_labels = stop_labels.collect()
stop_labels.collect_schema()
stop_labels.select(["stop_locations","cluster_counts"]).unique()

