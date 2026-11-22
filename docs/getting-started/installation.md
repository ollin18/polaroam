# Installation

PolaRoam needs **Python ≥ 3.10** and **Polars ≥ 1.35** (tested with Polars 1.44). The core
depends only on Polars and NumPy. Everything else is an optional extra.

```bash
git clone https://github.com/ollin18/polaroam.git
cd polaroam
python -m venv ~/.venvs/polaroam && source ~/.venvs/polaroam/bin/activate
pip install -e .
```

## Optional extras

| Extra | Installs | For |
|---|---|---|
| `tz` | `shapely>=2.1` | per-stay timezones from polygons ([Local time](../guide/local-time.md)) |
| `h3` | `polars-h3>=0.7` (Python ≥ 3.11) | H3 heat maps ([Heat maps](../guide/spatial.md)) |
| `cuda` | `cudf-polars-cu12>=26.4` | the GPU engine on CUDA 12 ([CUDA](../guide/cuda.md)) |
| `cuda13` | `cudf-polars-cu13>=26.4` | the GPU engine on CUDA 13 |
| `plot` | `matplotlib`, `contextily` | the example figures |
| `docs` | `mkdocs-material`, `mkdocstrings[python]` | building this site |
| `dev` | `pytest`, `scikit-learn`, `scipy`, `shapely`, `ruff` | the test suite |

```bash
pip install -e ".[tz,cuda]"        # typical GPU workstation
pip install -e ".[tz,plot,dev]"    # development
```

!!! tip "Use a dedicated virtual environment"
    cudf-polars pins a narrow Polars range (26.10 needs Polars 1.35–1.44). Keep PolaRoam in its
    own environment so other projects are not forced onto that range.

## Check the installation

```python
import polaroam as pr

print(pr.__version__)
print("CUDA available:", pr.cuda_available())
```

```bash
pytest -q      # with the dev extra: 35 tests, GPU tests skipped without CUDA
```

## Timezone polygons

Per-stay timezones need a CSV with a `tz` column and a WKT `geometry` column, like the
`tz_codes.csv` of the gc-lbs pipeline (world timezone polygons, ~21 MB). It is not bundled.
Pass its path as `tz_polygons=` or `--tz-polygons`. If every stay is in one timezone, a fixed
`timezone=` needs no file and no shapely.
