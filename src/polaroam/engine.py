"""Execution engine selection (CPU or CUDA through ``cudf-polars``).

Every collect in polaroam goes through :func:`collect`. With ``use_cuda=True``
the query is handed to the polars GPU engine; any query (or part of it) the GPU
engine does not support transparently falls back to the CPU engine, so results
are identical either way.
"""

from __future__ import annotations

import functools
import warnings

import polars as pl


@functools.lru_cache(maxsize=1)
def cuda_available() -> bool:
    """True when ``cudf-polars`` is installed and a CUDA device can run a query."""
    try:
        import cudf_polars  # noqa: F401
    except Exception:  # noqa: BLE001 - any import problem means no GPU
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pl.LazyFrame({"a": [1, 2]}).select(pl.col("a").sum()).collect(
                engine=pl.GPUEngine(raise_on_fail=True)
            )
        return True
    except Exception:  # noqa: BLE001 - any failure means the GPU engine is unusable
        return False


_warned = False

#: Options passed to ``pl.GPUEngine``. The single-GPU "in-memory" executor is much
#: faster than cudf-polars' default multi-partition "streaming" executor for the
#: window/join-heavy queries of polaroam (each shard is meant to fit in GPU memory).
GPU_OPTIONS: dict = {"executor": "in-memory"}


def configure_gpu(**options) -> None:
    """Set ``pl.GPUEngine`` options, e.g. ``configure_gpu(device=1)`` or
    ``configure_gpu(executor="streaming")`` for shards larger than GPU memory."""
    GPU_OPTIONS.clear()
    GPU_OPTIONS.update(options)


def resolve(use_cuda: bool | None) -> bool:
    """Return whether the GPU will actually be used (warns once if unavailable)."""
    global _warned
    if not use_cuda:
        return False
    if cuda_available():
        return True
    if not _warned:
        warnings.warn("use_cuda=True but cudf-polars/CUDA is not available; using the CPU engine.", stacklevel=3)
        _warned = True
    return False


def collect(lf: pl.LazyFrame | pl.DataFrame, use_cuda: bool | None = False, streaming: bool = False) -> pl.DataFrame:
    """Collect a LazyFrame on the GPU when requested and available, else on the CPU.

    ``streaming=True`` uses the CPU streaming engine (bounded memory), meant for
    order-independent scans such as reading and filtering pings.
    """
    if isinstance(lf, pl.DataFrame):
        return lf
    if resolve(use_cuda):
        with warnings.catch_warnings():
            # cudf-polars warns for every node it hands back to the CPU engine.
            warnings.simplefilter("ignore")
            return lf.collect(engine=pl.GPUEngine(raise_on_fail=False, **GPU_OPTIONS))
    return lf.collect(engine="streaming") if streaming else lf.collect()
