# Visualization Toolkit
This section documents the main **diagnostic and visualization utilities** available in Drought-Scan.



## 1) Initial setup (from files)

Recall that to run Drought-Scan and initialize a Drought-Scan Object (DSO) you need at least:
- A **precipitation dataset** in NetCDF format.  
- A **shapefile** delimiting the hydrographic basin of interest.  

The tool will automatically:
1. Select the gridded data that fall within the shapefile.  
2. Aggregate them spatially (area-weighted average).  
3. Aggregate them temporally on a **monthly basis**. 

```python
import drought_scan as DS
 
shape_path = 'tests/data/bacino_pontelagoscuro.shp'
prec_path  = 'tests/data/LAPrec1871.v1.1.nc'

ds = DS.Precipitation(
    prec_path=prec_path,
    shape_path=shape_path,
    start_baseline_year=1900,
    end_baseline_year=1950,
    basin_name='Po'  # only used for labeling/plots
)

```
---

## 2) Plot the SPI heatmap and CDN

The core visualization is provided by `plot_scan`, which shows:

- Heatmap of SPI (scales 1–K).
- SIDI series (weighted multi-scale index).
- CDN (Cumulative Deviation from Normal).

```python
# Default visualization
ds.plot_scan()
```

### Customization options

```python
ds.plot_scan(optimal_k=10)           # Highlight an optimal integration timescale
ds.plot_scan(weight_index=4)         # Use geometrically increasing weights for SIDI
ds.plot_scan(year_ext=(2000,2010))   # Zoom in on a specific period
ds.plot_scan(plot_order='HSC')       # set the order of the subplots: H > heatmap; S = SIDI ==(D(SPI)); C=CDN
ds.plot_scan(split_plot=True)        # plot Heatmap, SIDI and CDN in single plots 
ds.plot_scan(saveplot=True)          # Automatically save the figure in working directory
```

Figures can also be saved manually:

```python
from drought_scan.utils import savefig

ds.plot_scan(year_ext=(2000,2010)) 
savefig('PATH/my_figure.png')
```

---

## 3) Identify severe drought events

Use the threshold to detect events:

```python
print("Default threshold:", ds.threshold)
ds.threshold = -1.5  # Custom threshold

tstartid, tendid, duration, deficit = ds.severe_events_old()
print("Severe events started at:", ds.m_cal[tstartid])
```

### Show multiple events

```python
ds.severe_events_old(max_events=10, labels=True)
```


---

## 4) Equivalent precipitation for a target SPI

`normal_values()` returns precipitation amounts equivalent to SPI=0.
Use `spi_to_native()` to convert any SPI value to precipitation.

```python
print("Normal precipitation values:", ds.normal_values())

equivalent_precipitation = ds.spi_to_native(-1.5, month_scale=18, ref_month=3)
print(f"Equivalent precipitation for SPI18=-1.5 in March: {equivalent_precipitation}")

# and the other direction
print(ds.native_to_spi(1000.0, month_scale=18, ref_month=3))
```

`spi_to_native` / `native_to_spi` invert the fitted distribution exactly — the
Gamma's zero-inflated ppf for `f_spi`, Pearson III for `f_spei`, the KDE
cumulative for `f_kde`, mean+std for `f_zscore`.

> **Deprecated:** `ds.c2r_index` holds a degree-3 polynomial approximation of the
> same mapping, evaluated with `np.polyval`. It is kept for backward compatibility
> only and is not used by any live computation: its error is non-trivial in the
> distribution tails, and it does not represent the point mass at zero at all.
> Do not use it for new code.
**Visual inspection with `plot_spi_fit()`** 

For exploratory purposes, the method `plot_spi_fit(K, month)` provides a graphical representation of the regression curves used to link SPI values to raw data (precipitation, PET, or balance).
It plots the fitted relationship for a given month-scale K and reference month, with SPI values on the vertical axis and equivalent raw values on the horizontal axis. The color scale reflects the SPI domain (–3 to +3).
```python
# Example: plot the fitted curve for SPI-3 in June

ds.plot_spi_fit(K=3, month=6) 
 
# Optionally, the method can return the full 3D matrix of equivalent values used for plotting (K × SPI domain × months). Assign the result to a variable to access it:
K = 12
month = 6
mm = ds.plot_spi_fit(K=K, month=month, return_data=True)
# This allows extracting equivalent raw values for specific month and scale, e.g.:
spi_target = -1.5
idx = np.where(spi_target<=np.arange(-3, 3.2, 0.2))[0][0]
equivalent_precipitation = mm[K-1, idx, month-1]  # SPI-12, June
print(equivalent_precipitation)
```
---

## 5) Trend detection in CDN

Detect long-term positive/negative cycles with a rolling window (default=60 months).

```python
window = 36
R = ds.find_trends(window=window)

# Arrays returned:
# 'trend': -1 (negative), 0 (none), 1 (positive)
# 'slope': slope coefficient
# 'p_value': statistical significance
# 'delta': cumulative change of the CDN over the window (standardized units)

# Example: trend status in Nov 2017
date_idx = np.where((ds.m_cal[:,0]==11) & (ds.m_cal[:,1]==2017))[0][0]
print(f"Trend at Nov 2017 (W={window}m): "
      f"direction={R['trend'][date_idx]}, p-value={R['p_value'][date_idx]:.3f}")

# To convert the trend into physical units (mm or m^3), see
# `deficit_from_spi` in the subsection below.

```

**Using external variables**

By default, `find_trends` operates on CDN, but it can also be applied to any external time series aligned with the same calendar:

```python
import numpy as np
n = 600
rng = np.random.default_rng(0)

t = np.arange(n)
my_timeseries = 0.5*np.sin(2*np.pi*t/50) + 0.3*np.sin(2*np.pi*t/200) + rng.normal(0, 0.2, n)

R = ds.find_trends(var=my_timeseries, window=48)
```

### Plot trends

`plot_trends` visualizes the CDN curve together with the cumulative water
deficit/surplus estimated over one or more moving windows. Bars on the right
axis report the deficit in **physical units** — millimetres for `Precipitation`,
cubic metres for `Streamflow` — inferred automatically from the object type.
Bars are set to zero wherever no statistically significant monotonic trend is
detected, so a magnitude is shown only during phases of significant change.

```python
ds.plot_trends()                          # default window
ds.plot_trends(windows=[36, 60])          # multiple windows
ds.plot_trends(windows=[36], show_spi=True)  # overlay the SPI series at that scale
```

When `show_spi=True`, the SPI-like series at the window scale is overlaid on a
third axis. Scales beyond `ds.K` are computed on the fly. Gaps from missing
values in the input series appear as breaks in the SPI line.

You can still provide an external axis for multi-panel figures:

```python
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 1, figsize=(8, 6))
ds.plot_trends(windows=[36], ax=axs[0])
ds.plot_trends(windows=[60], ax=axs[1])
```

> **Note**: `plot_trends` always visualizes the CDN trend; it does not accept a
> custom variable. Use `find_trends(var=...)` to compute trends on external
> series.

### Cumulative deficit/surplus in physical units

Two complementary methods quantify the cumulative water deficit/surplus over a
moving window, both expressed in native units (mm for `Precipitation`, total m^3
for `Streamflow`) relative to the SPI=0 reference (which equals
`normal_values()`):

```python
# Statistical-rarity estimate (used internally by plot_trends): maps the SPI-like
# anomaly at the window scale back to native units via the exact inverse of the
# fitted distribution (spi_to_native), relative to the SPI=0 reference.
deficit = ds.deficit_from_spi(window=36)

# Direct observation-based water balance: sum of monthly (obs - normal) anomalies.
volume = ds.volume_anomaly_rolling(window=36)
```

`deficit_from_spi` reflects the **statistical rarity** of the accumulated anomaly
and is symmetric in SPI space (though, once converted back to physical units via
the non-linear gamma calibration, it remains physically asymmetric — e.g.
precipitation is bounded below by zero). `volume_anomaly_rolling` is the **direct
physical water balance** and serves as a sanity check and complementary metric.
The two correlate strongly but diverge at extremes.

```python
# Example: deficit at the peak of a window
import numpy as np
d = ds.deficit_from_spi(window=36)
idx = np.nanargmin(d)
print(f"Peak deficit over 36 months: {d[idx]:.3e} "
      f"({'m^3' if hasattr(ds, 'BFI') else 'mm'}) at {ds.m_cal[idx]}")
```
---


## 6) Monthly profiles of input data

Visualize the intra-annual cycle of the input variable through monthly profiles (by default: precipitation if the instance is initialized with Precipitation, snowfall if initialized with Snowfall, etc.).
The method shows a central reference line and the interquartile range (IQR) for each month over the baseline years. The name of the variable must be specified by the user.

When plotting the DSO's own series (`var=None`, the default), the central line
is `normal_values()`'s per-month "normal" — the exact inverse of the SPI-like
index at 0 (via `spi_to_native`), the same SPI=0 reference `deficit_from_spi`
and `volume_anomaly_rolling` use — **not** the raw arithmetic mean. This
matters for skewed and/or zero-inflated variables (typical for precipitation),
where the mean and the SPI=0 "normal" (closer to the median) can differ
noticeably. When plotting an arbitrary `var` instead (e.g. `ds.spi_like_set[0]`
below), there is no associated fit to invert, so the central line falls back
to the plain baseline arithmetic mean. The IQR/10-90 percentile bands are
always the plain empirical baseline percentiles in both cases.

```python
ds.plot_monthly_profile(var_name='P')
ds.plot_monthly_profile(var_name='P', highlight_years=[2017, 2018])
```

Cumulative profiles are useful for snow-dominated (nival) regimes or for highlighting multi-year accumulation.
```python
ds.plot_monthly_profile(var_name='P', cumulate=True, highlight_years=[2017, 2018])
```

For winter-relevant variables, the plot can be shifted and centered on the hydrological year (from August to July) using the `season_shift=True` option: 
```python
ds.plot_monthly_profile(var_name='P', season_shift=True, highlight_years=[2017, 2018])
```

User can choose other variables from the DSO to be plotted, such as `ds.spi_like_set[0]` (SPI-1):
 
```python
ds.plot_monthly_profile(var=ds.spi_like_set[0], var_name='SPI1', season_shift=True, highlight_years=[2000])
```

**Advanced options**

- Assign an external axis for multiple plots (e.g., in subplots):

```python
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 1, figsize=(8, 6))
ds.plot_monthly_profile(ax=axs[0])
ds.plot_monthly_profile(season_shift=True, ax=axs[1])
```

---

## 7) Basin boundary

The method `plot_boundary` displays the basin shapefile on an equal-area projection
with coordinate frame.

```python
ds.plot_boundary()

# Customization
ds.plot_boundary(buffer_deg=0.3, facecolor='lightblue', edgecolor='navy', linewidth=2)

# Integration in a multi-panel figure
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

fig, axs = plt.subplots(1, 2, figsize=(14, 6),
                         subplot_kw={'projection': ccrs.PlateCarree()})
ds.plot_boundary(ax=axs[0])
```

---

## 8) Precipitation–Streamflow covariates

After optimizing the SIDI with `analyze_correlation` and `set_optimal_SIDI` (see [User Guide](user_guide.md), Section 8.1),
the method `plot_covariates` shows the optimal SIDI and SQI1 time series side by side.

```python
# Requires: ds.set_optimal_SIDI(optimal_k, optimal_weight_index, overwrite=True)
ds.plot_covariates(streamflow)

# Zoom on a period
ds.plot_covariates(streamflow, year_ext=(2000, 2019))

# Split into separate figures
ds.plot_covariates(streamflow, split_plot=True)
```

---

## 9) Annual time series comparison (Streamflow only)

The method `plot_annual_ts` (available on `Streamflow` objects) compares annual aggregates
of streamflow with an external driver (Precipitation, Pet, or Balance).

```python
# Compare annual streamflow with annual precipitation (starting from August)
streamflow.plot_annual_ts(ds, starting_month=8)

# Using standardized values instead of raw
streamflow.plot_annual_ts(ds, values='std')
```

**Parameters:**
- `DSO`: a DroughtScan object (Precipitation, Pet, or Balance) to compare against.
- `starting_month` (int, default 8): month at which each 12-month window starts.
- `values` (`'abs'` or `'std'`): use raw series or standardized (SPI-like) values.

---

## 10) Baseflow Index (Streamflow only)

The method `BFI` computes the Baseflow Index using the Institute of Hydrology method
(Gustard et al., 1992). Requires **daily** streamflow data.

```python
bfi, baseflow = streamflow.BFI(block_size=5, plot=True)
print(f"Baseflow Index: {bfi:.3f}")
```

---

## 11) Export data for external plotting

The method `export_scan_plot_csv` exports the minimum data needed to replicate
the `plot_scan` visualization in another workspace (e.g., R, MATLAB).

```python
ds.export_scan_plot_csv(weight_index=2, name='Po_basin', out_dir='exports')
```

This creates CSV files for `m_cal`, `CDN`, `spi_like_set`, `SIDI`, and a JSON metadata file.

---

## 12) Spatial visualization

For spatially distributed maps of SPI, SIDI, and CDN trends at every grid point
within the basin, see the [Spatial Guide](spatial_guide.md).

Key methods:
- `ds.spatial_maps()` — compute gridded SPI and SIDI at a target timestamp.
- `ds.spatial_spi()` — compute pixel-wise SPI maps and their millimetre-equivalent reverse.
- `ds.plot_spatial()` — visualize the output maps.