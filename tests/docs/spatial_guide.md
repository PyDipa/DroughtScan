# Spatial Extension

This section documents the **spatial analysis tools** available in Drought-Scan.
While the standard workflow aggregates gridded data to a single basin-average time series,
the spatial extension computes the **SPI-like index and the SIDI at every grid point**
within the basin, producing spatially distributed maps at a target date.

---

## 1. Overview

When an object is initialized from a gridded NetCDF file (with a shapefile), the library
retains the full 3D grid (`n_months × n_rows × n_cols`) before spatial aggregation. The
spatial methods replicate the same pipeline as the basin-level index — standardization,
weighted ensemble mean, standardization on the baseline — independently at each valid
grid point, so a pixel's SIDI is exactly what `ds.SIDI` would be for a basin made of that
pixel alone.

| Class | Grid used | Maps of |
|---|---|---|
| `Precipitation` | `ds.Pgrid` | SPI, SIDI |
| `Pet` | `ds.PETgrid` | SPETI, SIDI |
| `Balance` | `ds.prec_grid − ds.pet_grid` (pixel-wise P − PET) | SPEI, SIDI |
| `Temperature` | `ds.Tgrid` | STI, SIDI |

Objects built from `ts`/`m_cal` (no grid), `Streamflow` and `Teleindex` have no spatial
maps and raise a clear error. A `Balance` whose P and PET grids have different spatial
resolutions cannot be mapped pixel by pixel (a warning is issued at import).

The maps are stored in the object and drawn with `plot_spatial()`.

> **Note on performance**
> Computing the index at many scales for every grid point is computationally intensive.
> The library parallelizes the computation across all available CPU cores with `joblib`,
> and skips degenerate grid points (all-NaN, e.g. outside the shapefile, or constant
> series) before starting. An estimated completion time is printed at the beginning of
> the run.

**Two methods, two products, one date.**

- `spatial_sidi()` → `ds.SIDI_grid` (the SIDI, all weighting schemes) and `ds.spatial_K`;
- `spatial_spi()` → `ds.SPI_grid` (the SPI-like index at the requested windows) and
  `ds.reverse_SPI_grid` (its native-unit deficit/surplus).

All the maps stored on the object refer to **one** date, `ds.spatial_timestamp` (§4).

---

## 2. `spatial_sidi`

### 2.1 Signature

```python
ds.spatial_sidi(
    timestamp=None,
    K=None,
    seasonal_params=None,
    agg=None,
    seasons=None
)
```

#### Parameters

- `timestamp` : tuple `(month, year)`, optional
  Target date. Default: the stored `ds.spatial_timestamp` if a previous spatial call set
  one, else the last month of `ds.m_cal` (§4).

- `K` : int or sequence of 5 int, optional
  Month-scale(s) of the SIDI ensemble: one value for all five weighting schemes, or one
  per scheme (as `analyze_correlation`'s `best_k_per_weight`). `ds.K` is never modified.
  Default: **whatever calibration the object carries** — `ds.seasonal_params` if a
  seasonal SIDI was committed, else `ds.optimal_k` if a global one was, else `ds.K` — so
  the grid matches `ds.SIDI`. A printed note says which calibration is followed.

- `seasonal_params` : dict, optional
  A seasonal calibration to map **without committing it**. Two shapes are accepted:
  - the `set_optimal_SIDI_seasonal` wrapper (i.e. `ds.seasonal_params`), which carries
    `'seasons'` and `'config'`;
  - the raw output of `analyze_correlation_seasonal()`. Then also pass `agg=` (or
    `seasons=`) so months can be mapped to seasons; it falls back to a committed
    `ds.seasonal_params['agg']` if present.

  `K` is resolved to the per-scheme optima (`best_k_per_weight`) of the season
  containing `timestamp`, overriding any `K` passed, and each pixel is standardized on
  that season's baseline months only — the same per-season reference as the seasonal
  SIDI.

- `agg` : str, optional
  `'quarter'`, `'semiannual'`, `'four-monthly'`, `'monthly'` or `'custom'`. Only used with
  the raw `analyze_correlation_seasonal` output; must match the `agg` it was computed with.

- `seasons` : dict, optional
  `{season_name: [month_ints]}` for `agg='custom'` — same role as `agg`.

### 2.2 Stored attributes

- **`ds.SIDI_grid`** : ndarray `(n_rows, n_cols, 5)` — the SIDI at the target date, one
  layer per weighting scheme (`ds.SIDI_grid[:, :, weight_index]`).
- **`ds.spatial_K`** : the K used (scalar or one per scheme).
- **`ds.spatial_timestamp`** : ndarray `(2,)`, `[month, year]`.

Grid points with degenerate series remain `np.nan`.

> **Gotcha**: `spatial_sidi` stores into `ds` and returns `None`. Do not write
> `A = ds.spatial_sidi(...)`.

### 2.3 Example

```python
import drought_scan as DS

shape_path = 'tests/data/bacino_pontelagoscuro.shp'
prec_path  = 'tests/data/LAPrec1871.v1.1.nc'

ds = DS.Precipitation(
    prec_path=prec_path,
    shape_path=shape_path,
    start_baseline_year=1900,
    end_baseline_year=1950,
    basin_name='Po'
)

ds.spatial_sidi()                                 # last month of the record
ds.spatial_sidi(timestamp=(8, 2003), K=36)        # August 2003, explicit K
print(ds.SIDI_grid.shape)                         # (n_rows, n_cols, 5)
print(ds.spatial_timestamp)                       # [8, 2003]
```

### 2.4 Mapping a calibrated SIDI

**A. Known K, nothing committed.** Pass `K` directly:

```python
ds.spatial_sidi(K=8)
ds.plot_spatial(var='SIDI', weight_index=0)
```

**B. Global calibration committed** (`set_optimal_SIDI(..., overwrite=True)`).
`spatial_sidi()` with no `K` uses `ds.optimal_k`, and `plot_spatial` defaults to
`ds.optimal_weight_index`:

```python
ds.set_optimal_SIDI(optimal_k=A['best_k_per_weight'], optimal_weight_index=0, overwrite=True)
ds.spatial_sidi()
ds.plot_spatial(var='SIDI')
```

**C. Seasonal calibration.** A single global `K` cannot replicate the seasonal SIDI: each
season has its own K per scheme and its own baseline months. Both routes resolve the
season from `timestamp`.

*Way 1 — without committing.* Pass the raw `analyze_correlation_seasonal()` output and its
`agg`; `ds.SIDI` and `ds.seasonal_params` are left untouched:

```python
S = ds.analyze_correlation_seasonal(streamflow, agg='semiannual')
ds.spatial_sidi(seasonal_params=S, agg='semiannual', timestamp=(1, 2022))
ds.plot_spatial(var='SIDI', weight_index=0)
```

*Way 2 — commit first.* After `set_optimal_SIDI_seasonal(overwrite=True)`,
`spatial_sidi()` with no argument detects it and reuses `ds.seasonal_params`; after a
seasonal calibration `plot_spatial` defaults to `weight_index=0` (equal weights):

```python
ds.set_optimal_SIDI_seasonal(S, agg='semiannual', overwrite=True)
ds.spatial_sidi(timestamp=(1, 2022))
ds.plot_spatial(var='SIDI')
```

> If both `K` and `seasonal_params` are passed, `seasonal_params` wins.

---

## 3. `spatial_spi`

### 3.1 Purpose

For each valid grid point and each requested window W (accumulation scale), computes the
SPI-like index at the target date and its **native-unit reverse**: the deficit/surplus
over the last W months, relative to the index = 0 reference, obtained with the exact
inverse of the pixel's fitted distribution (the same `spi_to_native` used at basin level,
whatever the `calculation_method`: zero-inflated Gamma, Pearson III, KDE, z-score, or a
registered custom method). Units are millimetres for the cumulative variables. The windows
are **not** capped by `K`.

### 3.2 Signature

```python
ds.spatial_spi(
    windows=None,      # list of int (months); default [24, 36, 60, 120]
    timestamp=None     # default: stored spatial_timestamp, else last month (§4)
)
```

### 3.3 Stored attributes

- **`ds.SPI_grid`** : dict `{window: ndarray (n_rows, n_cols)}` — the SPI-like index.
- **`ds.reverse_SPI_grid`** : dict `{window: ndarray (n_rows, n_cols)}` — the native-unit
  deficit/surplus. Real values everywhere the index is defined, **including** inside the
  near-neutral band: `plot_spatial` greys that band out at draw time, the data are not
  zeroed. `NaN` only where the index is undefined (record shorter than the window, gaps).
- **`ds.spatial_timestamp`** : ndarray `(2,)`.

### 3.4 Example

```python
ds.spatial_spi(windows=[12, 36, 60, 120])
print(list(ds.SPI_grid))                 # [12, 36, 60, 120]
print(ds.reverse_SPI_grid[60].shape)     # (n_rows, n_cols)
```

---

## 4. The spatial timestamp

The two methods share one rule, so all the maps on the object always refer to the same
date:

1. **Which date.** `timestamp=(month, year)` if given; otherwise the stored
   `ds.spatial_timestamp` (set by any previous `spatial_sidi` / `spatial_spi` call);
   otherwise the last month of `ds.m_cal`.
2. **When the date changes**, every product of the *other* method is removed, with a
   warning: `spatial_sidi` at a new date removes `SPI_grid` and `reverse_SPI_grid`;
   `spatial_spi` at a new date removes `SIDI_grid` and `spatial_K`. Rerun that method
   for the new date.

```python
ds.spatial_sidi(timestamp=(8, 2003))   # SIDI_grid at Aug 2003
ds.spatial_spi(windows=[12])           # inherits Aug 2003 — SIDI_grid kept
ds.spatial_spi(windows=[12], timestamp=(9, 2003))
# UserWarning: spatial timestamp changed from [8 2003] to [9 2003]: the spatial_sidi()
# products (SIDI_grid, spatial_K) referred to the old date and have been removed ...
```

---

## 5. `plot_spatial`

### 5.1 Signature

```python
ds.plot_spatial(
    var='SIDI',          # 'SIDI', 'SPI' or 'reverse_spi'
    weight_index=None,   # SIDI layer; default as every other SIDI consumer (below)
    month_scale=None,    # required for 'SPI' / 'reverse_spi': a key of SPI_grid
    ax=None,             # external axes for multi-panel figures
    title=None,          # default: index name and date
    cmap=None,           # default: the library's drought palette (reversed if threshold > 0)
    neutral_band=0.5     # 'reverse_spi' only: grey out |index| < neutral_band (0/None: off)
)
```

- `weight_index` default: the committed `optimal_weight_index` after a global
  calibration, `0` after a seasonal one, else the construction `weight_index` (2 unless
  set) — see the [User Guide](user_guide.md) §9.1.3.
- Labels use the object's index name (SPI, SPETI, SPEI, STI, ...).
- `var='CDN'` is not accepted: the library does not map the CDN.

Returns the `matplotlib.axes.Axes`.

### 5.2 Examples

```python
ds.plot_spatial()                                  # SIDI, default layer
ds.plot_spatial(var='SPI', month_scale=12)         # SPI-12
ds.plot_spatial(var='reverse_spi', month_scale=60) # deficit/surplus over 60 months
ds.plot_spatial(var='SIDI', weight_index=0, title='SIDI (equal weights) — August 2003')

import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
ds.plot_spatial(var='SIDI', ax=axs[0])
ds.plot_spatial(var='SPI', month_scale=36, ax=axs[1])
ds.plot_spatial(var='SPI', month_scale=12, ax=axs[2])
plt.tight_layout()
```

---

## 6. Practical notes

- **Compute once per date, plot many times.** `plot_spatial` reads the stored maps; to
  change date, rerun the spatial methods (§4).
- **Baseline consistency.** The pixels use the object's `start_baseline_year` /
  `end_baseline_year`; keep it consistent with the basin-level analysis.
- **Computation time.** On a standard laptop (8 cores), ~2500 valid grid points at K=36
  take approximately 5–10 minutes for `spatial_sidi`.
- **Degenerate points.** Points with all-NaN or constant series are excluded and stay
  `np.nan`; the number of valid points and any per-point failure are printed at runtime.
