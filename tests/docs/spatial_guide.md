# Spatial Extension

This section documents the **spatial analysis tools** available in Drought-Scan.  
While the standard workflow aggregates gridded precipitation to a single basin-average time series,
the spatial extension allows computing **SPI and SIDI at every grid point** within the basin,
producing spatially distributed maps of drought conditions at a target date.

---

## 1. Overview

When a `Precipitation` object is initialized from a NetCDF file, the library internally retains
the full 3D precipitation grid (`ds.Pgrid`, shape: `n_months × n_rows × n_cols`) before spatial
aggregation. The spatial methods leverage this grid to replicate the same SPI/SIDI pipeline
independently at each valid grid point.

Output maps are stored in the DSO and can be visualized with `plot_spatial()`.

> **Note on performance**  
> Computing SPI at K temporal scales for every grid point is computationally intensive.
> The library automatically parallelizes the computation across all available CPU cores
> using `joblib`, and skips masked or degenerate grid points (e.g., outside the shapefile,
> constant series) before starting. An estimated completion time is printed at the beginning
> of the run.

---

## 2. `spatial_sidi`

### 2.1 Purpose

Computes, for each valid grid point within `ds.Pgrid`:

- **SPI** at selected temporal scales — **moved to `spatial_spi` in 4.0.0**
- **SIDI** across all implemented weighting schemes

All results are stored as spatial arrays in the DSO at a target timestamp.

---

### 2.2 Method signature

```python
ds.spatial_sidi(
    timestamp=None,
    K=None,
    seasonal_params=None,
    agg=None,
    seasons=None
)
```

> **Renamed in 4.0.0.** This method was `spatial_maps()`. It is now `spatial_sidi()`,
> mirroring `spatial_spi()`: one computes the gridded SIDI, the other the gridded SPI.

#### Parameters

> **Changed in 4.0.0.** `month_scales` is gone: `spatial_sidi` computes the SIDI
> grid only. SPI maps (and their millimetre-equivalent reverse) come from
> `spatial_spi`, which is not capped by `K`. Both methods used to write `SPI_grid`,
> so whichever ran last overwrote the other's.

<!-- removed parameter -->
- ~~`month_scales`~~ : list of int, optional  
  Temporal scales for which SPI maps are stored.  
  Default: `[1, 3, 6, 12, 18, 24]`.  
  All values must be ≤ `K`; scales exceeding `K` are silently ignored with a warning.

- `timestamp` : tuple `(month, year)`, optional  
  Target date for the output maps.  
  Default: last available timestamp in `ds.m_cal`.  
  Example: `timestamp=(9, 2022)` for September 2022.

- `K` : int, optional  
  Maximum temporal scale. Overrides `ds.K` for this computation only.  
  After the method completes, `ds.K` is restored to its original value.

- `seasonal_params` : dict, optional  
  A seasonal calibration to map **without committing it** to the object. Two
  shapes are accepted:
  - the `set_optimal_SIDI_seasonal` wrapper (i.e. `ds.seasonal_params`), which
    carries `'seasons'` and `'config'` — the month→season mapping travels inside it;
  - the raw output of `analyze_correlation_seasonal()` (keys are season names). In
    this case also pass `agg=` (or `seasons=`) so months can be mapped to seasons;
    it falls back to a committed `ds.seasonal_params['agg']` if present.
  `K` is resolved to the per-scheme optima (`best_k_per_weight`, else `best_k`) of
  the season containing `timestamp`, overriding any `K` passed explicitly, and each
  pixel is standardized on that season's baseline months only. A warning reports
  the suggested `weight_index` to use in `plot_spatial`.
  See [§2.6 Using an optimized SIDI](#26-using-an-optimized-sidi) below.

- `agg` : str, optional  
  Aggregation scheme (`'quarter'`, `'semiannual'`, `'four-monthly'`, `'monthly'`).
  Used only when `seasonal_params` is the raw `analyze_correlation_seasonal` output.
  Must match the `agg` that output was computed with.

- `seasons` : dict, optional  
  `{season_name: [month_ints]}` for `agg='custom'` — same role as `agg`.

---

### 2.3 Stored attributes

After running `spatial_sidi`, the following attributes are added to the DSO:

- **`ds.SIDI_grid`** : ndarray, shape `(n_rows, n_cols, n_weights)`  
  SIDI at the target timestamp for all weighting schemes.  
  Access a specific weight with: `ds.SIDI_grid[:, :, weight_index]`  
  Default display weight: `weight_index=2` (geometrically decreasing).

(`ds.SPI_grid` is written by **`spatial_spi`**, not by `spatial_sidi` — see §3.)

- **`ds.spatial_timestamp`** : ndarray `(2,)`, i.e. `[month, year]`  
  The timestamp corresponding to the stored maps.

Grid points outside the shapefile or with degenerate series (all-NaN, zero variance)
remain `np.nan` in all output arrays.

---

### 2.4 Example usage

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

# Compute the spatial SIDI grid at the last available timestamp
ds.spatial_sidi()

# Compute at a specific date, with an explicit K
ds.spatial_sidi(
    timestamp=(8, 2003),   # August 2003
    K=36
)

# Inspect outputs
print("SIDI grid shape:", ds.SIDI_grid.shape)      # (n_rows, n_cols, n_weights)
# SPI maps come from spatial_spi:
ds.spatial_spi(windows=[12])
print("SPI-12 map shape:", ds.SPI_grid[12].shape)  # (n_rows, n_cols)
print("Timestamp:", ds.spatial_timestamp)           # [8, 2003]
```

---

### 2.5 Accessing specific weighting schemes

SIDI is computed for all five weighting schemes (see [User Guide](user_guide.md), Section 3, `weight_index`).
You can extract any of them:

```python
import matplotlib.pyplot as plt

# Default: geometrically decreasing weights (weight_index=2)
sidi_map = ds.SIDI_grid[:, :, 2]

# Equal weights
sidi_equal = ds.SIDI_grid[:, :, 0]
```

---

### 2.6 Using an optimized SIDI

`spatial_sidi` computes SIDI for **all** weighting schemes at a given `K`. Since
4.0.0 it **follows whatever calibration the object carries** when `K` is not
given: `ds.seasonal_params` if a seasonal SIDI was committed, else `ds.optimal_k`
if a global one was, else `ds.K`. You only need to be explicit when you want to
map a calibration the object has *not* committed.

**A. You already know the optimal `K` and `weight_index`.**  
No need to call any `set_optimal_*` method — pass `K` directly:

```python
# Suppose a previous analyze_correlation() found K=8, weight_index=0 as optimal
ds.spatial_sidi(K=8)
ds.plot_spatial(var='SIDI', weight_index=0)
```

**B. You called `set_optimal_SIDI(overwrite=True)` on the point-scale object.**  
`ds.SIDI` is the optimized series and `ds.optimal_k` / `ds.optimal_weight_index`
are stored on the instance. `spatial_sidi()` with no `K` picks them up
automatically, so the grid matches `ds.SIDI` (a warning states which `K` it used):

```python
ds.set_optimal_SIDI(optimal_k=8, optimal_weight_index=0, overwrite=True)

ds.spatial_sidi()                       # uses optimal_k=8
ds.plot_spatial(var='SIDI', weight_index=ds.optimal_weight_index)
```

**C. Seasonal optimization — two ways.**  
A single global `K` cannot replicate a seasonal mosaic: each season has its own
optimal `K` (one per weighting scheme) and its own baseline months. Both routes
below resolve the season from `timestamp` and standardize each pixel on that
season's baseline months.

*Way 1 — map a seasonal calibration without committing it.* Pass the raw output
of `analyze_correlation_seasonal()` straight to `spatial_sidi`, together with the
same `agg`. `ds.SIDI` and `ds.seasonal_params` are left untouched.

```python
A = ds.analyze_correlation_seasonal(streamflow, agg='semiannual')

ds.spatial_sidi(seasonal_params=A, agg='semiannual', timestamp=(1, 2022))
# UserWarning: Using seasonal K=[...] for season 'autwin' (month 1),
# standardized on that season's baseline months.
# Plot with weight_index=3 for consistency.
ds.plot_spatial(var='SIDI', weight_index=A['autwin']['col_best_weight'])
```

*Way 2 — commit the seasonal SIDI first, then map it.* After
`set_optimal_SIDI_seasonal(overwrite=True)`, `ds.SIDI` is the `(time, 5)`
per-scheme seasonal array; `spatial_sidi()` with no argument detects it and
re-uses `ds.seasonal_params`.

```python
ds.set_optimal_SIDI_seasonal(A, agg='semiannual', overwrite=True)

ds.spatial_sidi(timestamp=(1, 2022))    # auto: reuses ds.seasonal_params
ds.plot_spatial(var='SIDI',
                weight_index=ds.seasonal_params['config']['autwin']['col_best_weight'])
```

You can also pass `ds.seasonal_params` (the wrapper) explicitly — equivalent to
Way 2's auto-detection. When passing the raw `analyze_correlation_seasonal`
output you **must** also give `agg=` (or `seasons=`), otherwise months cannot be
mapped to seasons and a `ValueError` is raised.

> **Note**  
> If both `K` and `seasonal_params` are passed together, `seasonal_params` wins
> silently — the season-resolved `K` overrides the explicit `K`.

> **Gotcha**  
> `spatial_sidi` stores into `ds` and returns `None`. `A = ds.spatial_sidi(...)`
> overwrites `A` with `None` — keep the calls on separate lines.

---

## 3. `spatial_spi`

### 3.1 Purpose

Computes pixel-wise **deficit/surplus maps** at a target timestamp. For each valid
grid point, the method computes the SPI-like index at each requested window scale
and converts it to native units (mm) via the pixel-level reverse-gamma calibration
(`c2r`), consistent with `deficit_from_spi`. This replaces an earlier linear
approximation (`std_to_mm`) with the same non-linear, month-specific transform used
elsewhere in the framework — see the [Changelog](../CHANGELOG.md), v3.6.1.

Anomalies with `|SPI| < 0.5` at the target timestamp are set to `0.0`, filtering out
meteorologically negligible deficits/surpluses, consistent with `plot_cdn_trends`.
The CDN itself (cumulative sum of SPI-1) is still computed per pixel and stored
internally, but rolling trend significance is no longer used to gate the output —
the physical magnitude of the SPI-based anomaly is the sole criterion.

---

### 3.2 Method signature

```python
ds.spatial_spi(
    windows=None,
    timestamp=None
)
```

#### Parameters

- `windows` : list of int, optional  
  Moving window sizes in months.  
  Default: `[24, 36, 60, 120]`.

- `timestamp` : tuple `(month, year)`, optional  
  Target date for the output maps.  
  Default: last available timestamp in `ds.m_cal`.

---

### 3.3 Stored attributes

After running `spatial_spi`, the following attribute is added:

- **`ds.reverse_SPI_grid`** : dict `{window: ndarray (n_rows, n_cols)}`  
  mm-equivalent deficit/surplus over each window at the target timestamp, derived
  from the reverse-gamma transform of the SPI-like index (see §3.1).  
  Pixels with `|SPI| < 0.5` at the target timestamp are set to `0.0`.

The `spatial_timestamp` attribute is also updated. If it differs from a previously
stored timestamp, `SIDI_grid` is invalidated — rerun `spatial_sidi`
for consistency.

> **Note**  
> Every call to `spatial_spi` explicitly discards any previously stored
> `reverse_SPI_grid` before recomputing, so rerunning with different `windows` always
> reflects a clean recalculation rather than a partial update.

---

### 3.4 Example usage

```python
# Compute CDN trend maps at the last available timestamp
ds.spatial_spi(windows=[36, 60, 120])

# Inspect
print("Available windows:", list(ds.reverse_SPI_grid.keys()))
print("Trend 60-month shape:", ds.reverse_SPI_grid[60].shape)
```

---

## 4. `plot_spatial`

### 4.1 Purpose

Visualizes a spatial map of SIDI, SPI, or CDN trends overlaid on the basin shapefile boundary.
The colormap is automatically centered on zero and uses a discrete drought/surplus scale.

> **Note**  
> This method requires that `spatial_sidi` (for SIDI) or `spatial_spi` (for SPI / reverse SPI)
> has been run before.

---

### 4.2 Method signature

```python
ds.plot_spatial(
    var='SIDI',
    weight_index=2,
    month_scale=None,
    ax=None,
    title=None
)
```

#### Parameters

- `var` : `'SIDI'`, `'SPI'`, or `'reverse_spi'`, default `'SIDI'`  
  Which index to display.

- `weight_index` : int, default `2`  
  Weighting scheme to display when `var='SIDI'`.  
  Ignored when `var='SPI'` or `var='reverse_spi'`.

- `month_scale` : int, optional  
  Required when `var='SPI'`: must be one of the keys in `ds.SPI_grid`.  
  Required when `var='reverse_spi'`: must be one of the keys in `ds.reverse_SPI_grid`.

- `ax` : `matplotlib.axes.Axes`, optional  
  External axes for integration in multi-panel figures.  
  If `None`, a new figure is created.

- `title` : str, optional  
  Custom title. If `None`, the method auto-generates a title with the index name and timestamp.

#### Returns
`matplotlib.axes.Axes`

---

### 4.3 Example usage

```python
# Default: SIDI map with weight_index=2
ds.plot_spatial()

# SPI-12 map
ds.plot_spatial(var='SPI', month_scale=12)

# CDN trend map (60-month window, in mm)
ds.plot_spatial(var='reverse_spi', month_scale=60)

# Custom title
ds.plot_spatial(var='SIDI', weight_index=0, title='SIDI (equal weights) — August 2003')

# Integration in a multi-panel figure
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
ds.plot_spatial(var='SIDI', ax=axs[0])
ds.plot_spatial(var='SPI', month_scale=3,  ax=axs[1])
ds.plot_spatial(var='SPI', month_scale=12, ax=axs[2])
plt.tight_layout()
```

---

## 5. Practical notes

- **Compute once per timestamp, visualize many times.**  
  `spatial_sidi` stores results for a single target timestamp. Once computed,
  you can call `plot_spatial` as many times as you want on different variables, scales
  or weight schemes without recomputing — as long as you are exploring the same timestamp.
  To change the target date, run `spatial_sidi` again with a new `timestamp`.

- **Baseline consistency.**  
  The spatial computation uses the same `start_baseline_year` and `end_baseline_year` defined
  at initialization. For spatial maps to be comparable with the basin-average SIDI,
  keep the baseline consistent.

- **Computation time.**  
  The method prints an estimated completion time before starting, based on the number of valid
  grid points and available CPU cores. On a standard laptop (8 cores), a basin of ~2500 valid
  grid points at K=36 takes approximately 5–10 minutes.

- **`K` override.**  
  Passing `K` to `spatial_sidi` does not permanently change `ds.K`.
  The original value is restored after the method completes, leaving the DSO in its original state.

- **Grid points outside the basin.**  
  All grid points that fall outside the shapefile or contain degenerate series (e.g., constant
  precipitation, all-NaN) are automatically excluded from computation and set to `np.nan`
  in the output maps. The number of valid points processed is printed at runtime.

- **Timestamp consistency between `spatial_sidi` and `spatial_spi`.**  
  If you change the timestamp between calls, the library warns you and invalidates the
  previously stored grids. Always rerun both methods if you need maps and trends at the
  same timestamp.