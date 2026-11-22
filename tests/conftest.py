import math

import numpy as np
import polars as pl
import pytest

R = 6371008.8


def synthetic_pings(n_devices: int = 25, days: int = 20, seed: int = 0) -> pl.DataFrame:
    """Devices alternating between home / work / random places, with GPS noise,
    gaps, duplicated timestamps and travel pings in between."""
    rng = np.random.default_rng(seed)
    rows = []
    t0 = 1_600_000_000
    for d in range(n_devices):
        base = np.array([19.4 + rng.normal(0, 0.05), -99.1 + rng.normal(0, 0.05)])
        home = base
        work = base + rng.normal(0, 0.02, 2)
        others = [base + rng.normal(0, 0.03, 2) for _ in range(4)]
        t = t0 + int(rng.integers(0, 3600))
        pos = home
        while t < t0 + days * 86400:
            hour = (t // 3600) % 24
            target = home if hour < 8 or hour >= 19 else (work if rng.random() < 0.7 else others[rng.integers(4)])
            # travel
            n_travel = int(rng.integers(0, 12))
            for k in range(n_travel):
                f = (k + 1) / (n_travel + 1)
                p = pos + f * (target - pos) + rng.normal(0, 0.0005, 2)
                t += int(rng.integers(20, 400))
                rows.append((f"dev{d:03d}", p[0], p[1], t))
            pos = target
            # stay
            n_stay = int(rng.integers(1, 40))
            for _ in range(n_stay):
                noise = rng.normal(0, 6 / 111_000, 2) if rng.random() > 0.05 else rng.normal(0, 60 / 111_000, 2)
                t += int(rng.integers(30, 900))
                rows.append((f"dev{d:03d}", pos[0] + noise[0], pos[1] + noise[1], t))
                if rng.random() < 0.03:  # same timestamp, other coordinate
                    rows.append((f"dev{d:03d}", pos[0] + noise[1], pos[1] + noise[0], t))
            if rng.random() < 0.1:
                t += int(rng.integers(3600, 20000))  # long gap
    df = pl.DataFrame(rows, schema={"uid": pl.String, "latitude": pl.Float64, "longitude": pl.Float64,
                                    "timestamp": pl.Int64}, orient="row")
    # round like float32 sources so distance ties happen
    return df.with_columns(pl.col("latitude").cast(pl.Float32).cast(pl.Float64),
                           pl.col("longitude").cast(pl.Float32).cast(pl.Float64),
                           pl.col("timestamp") * 1000)


@pytest.fixture(scope="session")
def pings() -> pl.DataFrame:
    return synthetic_pings()


def hav(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))
