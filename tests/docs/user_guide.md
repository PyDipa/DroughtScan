# Example Usage

This section shows how to initialize a **Drought-Scan** analysis object (DSO), with practical notes on the most important options and how they change the behavior compared to defaults.

> **IMPORTANT NOTE**  
> Make sure you are running **the same Python interpreter** where DroughtScan and its dependencies have been installed.  
> 
> For example, if you installed with:
> ```bash
> python3.10 -m pip install .
> ```
> then you must also start your session with:
> ```bash
> python3.10
> ```
> or set the proper interpreter `python3.10` on your IDE and not another Python version.

> **Interactive plots (pop-up windows)**  
> If you want matplotlib figures to open in interactive windows instead of inline,
> add these lines **before** importing drought_scan:
> ```python
> import matplotlib
> matplotlib.use('Qt5Agg')
> import matplotlib.pyplot as plt
> ```
> This requires `PyQt5` (install with `pip install PyQt5`).
> In Jupyter notebooks this is not needed — figures render inline by default.


---

## 1) Minimal setup (from files)

To run Drought-Scan you need at least:
- A **precipitation dataset** in NetCDF format.  
- A **shapefile** delimiting the hydrographic basin of interest.  

Note: In `test/data` you will find some dataset for running the following examples

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

print("Aggregated precipitation (ts):", ds.ts.shape)
print("Monthly calendar (m_cal):", ds.m_cal[:5])
print("SPI multi-scale set:", ds.spi_like_set.shape)
print("SIDI (by 5 weighting scheme):", ds.SIDI.shape)
print("CDN (cumulative SPI1 from the starting baseline year):", ds.CDN.shape)
```

What happens here:
- The library reads the NetCDF precipitation, clips/aggregates it over the basin shapefile, builds a monthly calendar (`m_cal`), and computes SPI (1–K), SIDI, and CDN over the **baseline** `start_baseline_year:end_baseline_year`.

You can access the data as shown in the prints. For example, `ds.ts` is the monthly precipitation time series imported and aggregated at the river basin scale, while `ds.CDN` is the Cumulative Deviation from Normal. 
For the full list of variables and methods explore examples and usage notes below. You can also find 
a full list in the [README](https://github.com/PyDipa/DroughtScan/blob/main/README.md) file.


---

## 2) Direct arrays instead of files

You can also bypass I/O entirely and pass your own arrays:

```python
import numpy as np
import drought_scan as DS

# Example: 600 months of synthetic precipitation (positive) and a matching calendar
ts = np.random.gamma(shape=2.0, scale=30.0, size=600)          # (T,)
years = np.repeat(np.arange(1975, 2025), 12)[:600]
months = np.tile(np.arange(1, 13), 50)[:600]
m_cal = np.column_stack([months, years])                        # (T,2) -> [month, year]


ds = DS.Precipitation(
    ts=ts,
    m_cal=m_cal,
    shape_path=shape_path,
    basin_name='Po_fantasy',
    start_baseline_year=1981,
    end_baseline_year=2010
)
```

This is useful for customized pre-processing pipelines or when your data is already basin-aggregated.

---

## 3) Key parameters (and how to choose them)

Below are the most impactful options, with defaults and when you might want to change them:

- **`K` (int)** — *maximum temporal scale for SPI/SIDI*  
  Default: `36`.  
  Interpretation: `K` sets the longest memory of your indices.  
  - `K=36` (3 years): good general-purpose horizon for basin-scale drought.  
  - `K=60` (5 years): emphasizes slow/structural deficits (useful for long-term storage anomalies or policy assessments).  
  - `K=24` (2 years): focuses on shorter integrated dynamics.  
  In short, larger `K` → longer memory and smoother signals.

- **`threshold` (float)** — *severity threshold for events (e.g., on SIDI)*  
  Default: `-1`.  
  Meaning: events below the threshold are flagged as severe.  
  - `-1` corresponds to 1 standard deviation below the mean of a standardized index.  
  - In the Po River case study (see paper), `-1` proved effective for **severe** drought identification; adjust for your basin by comparing the SIDI with some observed impact variable.
  
- **`calculation_method` (callable)** — *index family for standardization*  
  Default: `f_kde` (non-parametric fit via Gaussian Kernel Density Estimation, Silverman bandwidth).  
  Available (in `drought_scan.utils`):
  - `f_spi` → standardization via **Gamma** distribution. Best for **positive, right-skewed** data (e.g., precipitation). Generally used for SPI.
  - `f_spei` → **Pearson III** distribution. Handles **real-valued, negative and/or skewed** data. Generally used for SPEI. Also suited for precipitation.
  - `f_kde` → Non-parametric standardization using Gaussian KDE *(default for all classes since v3.1.0)*.
  - `f_zscore` → standard z-score. Best when data are **approximately Gaussian** (real-valued); no parametric skew modeling.

  Practical guidance:  
  - Use `f_spi` for precipitation-like data when you want Gamma-based SPI.  
  - Use `f_spei` for **SPEI-style** applications (precip–PET, can be negative).  
  - Use `f_kde` (default) for any type of data, especially when unsure about the parametric family.
  - Use `f_zscore` when you trust normality and prefer a simpler transform.

  To verify which distribution best fits your data, see the diagnostic tools in
  [statistics_tools.md](statistics_tools.md).

- **`weight_index` (int)** — *weighting scheme for SIDI aggregation across scales*  
  Default: `2` (geometrically decreasing).  
  Options:
  - `0`: equal weights  
  - `1`: linear decreasing  
  - `2`: **geometrically decreasing** *(default; favors recent months)*  
  - `3`: linear increasing  
  - `4`: geometrically increasing  

  In practice, decreasing schemes (1–2) often improve responsiveness to recent conditions while preserving multi-scale context.

- **`start_baseline_year`, `end_baseline_year` (int)** — *climatological baseline*  
  Choose a stable, representative period (e.g., **1981–2010**). The baseline impacts index standardization and, consequently, event thresholds.

- **`index_name` (str)** — label used in outputs/plots (default `'SPI'`).

- **`verbose` (bool)** — print initialization details (default `True`).

**Defaults**

- **Baseline**: a stable and representative climatological period of at least 30 years is recommended. 50 years is good.  
Using the same baseline across precipitation and streamflow analyses ensures comparability of results.

- **Threshold**: a good starting point is **–1**, corresponding to one standard deviation below the mean of the standardized index.
This level is widely used in drought monitoring (e.g. to flag *severe drought*).  
You can easily adjust the threshold (e.g. –1.5 for stricter detection) and test the impact on event identification.

- These defaults are directly used by the method:

```python
ds.severe_events_old()
```
and can be visualized through:

```python
ds.plot_scan()
ds.plot_trends()
```

which show how baseline and threshold affect the scan of SPI/SIDI/CDN time series and the detection of drought episodes.
 
Please see the [Visualization Guide](visualization_guide.md) for further details about plotting methods.

---

## 4) Using different index families

Switch to **SPEI-like** behavior (Pearson III) or plain z-score:
By specifying the index name the plots will have the proper labels. 

```python
from drought_scan.utils import f_spi, f_zscore, f_spei, f_kde
import drought_scan as DS
from functools import partial

shape_path = 'tests/data/bacino_pontelagoscuro.shp'
prec_path  = 'tests/data/LAPrec1871.v1.1.nc'

ds = DS.Precipitation(
    prec_path=prec_path,
    shape_path=shape_path,
    start_baseline_year=1981,
    end_baseline_year=2010,
    basin_name='Po',
    calculation_method=f_spei,     # Pearson III
    index_name='SPI (Pearson3)'
)

ds2 = DS.Precipitation(
    prec_path=prec_path,
    shape_path=shape_path,
    start_baseline_year=1981,
    end_baseline_year=2010,
    basin_name='Po',
    calculation_method=f_zscore,   # z-score
    index_name='SPI (Zscore)'
)

ds3 = DS.Precipitation(
    prec_path=prec_path,
    shape_path=shape_path,
    start_baseline_year=1981,
    end_baseline_year=2010,
    basin_name='Po',
    calculation_method=partial(f_kde, log_transform=True),   #  default is False
    index_name='SPI (f_kde) '
)
# user can easily check the calibration obtained by the methods
ref_month = 3 #reference month (march in the example)
k = 6 # month-scale
ds.plot_spi_fit(K=k,month=ref_month)

ds2.plot_spi_fit(K=k,month=ref_month)

ds3.plot_spi_fit(K=k,month=ref_month)
```

---

## 5) Choosing `K` and `threshold` by intent

- **Operational monitoring** (recent conditions matter):  
  `K=24–36`, `weight_index=2`, `threshold=-1` (severe).  
- **Risk screening / structural deficits**:  
  `K=48–60`, consider testing `threshold` between `-1` and `-1.5` depending on desired sensitivity.  
- **Research sensitivity analysis**:  
  grid-search over `K ∈ {24,36,48,60}` and weighting schemes to see stability of drought episodes in your basin.

---

## 6) Quick inspection / DIY plotting
You can always extract raw arrays and build your own plots.

Note that ds.SIDI holds 5 time-series, one for each weighting scheme:
  - `0`: equal weights  
  - `1`: linear decreasing  
  - `2`: **geometrically decreasing** *(default; favors recent months)*  
  - `3`: linear increasing  
  - `4`: geometrically increasing  

```python
import matplotlib.pyplot as plt


# Example: SIDI (by equal_weights) and CDN time series
weight_index = 0
fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(ds.SIDI[:,weight_index], label='SIDI')
ax.axhline(-1, ls='--', label='Severe threshold')
ax.legend(); ax.set_title('SIDI (standardized)'); ax.grid(True)

fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(ds.CDN, label='CDN')
ax.legend(); ax.set_title('Cumulative Deviation from Normal (CDN)'); ax.grid(True)

# Example: SPI heatmap (1..K)
spi = ds.spi_like_set  # shape: (K, T) with K scales stacked
K = spi.shape[0] # or ds.K
fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(spi, aspect='auto',
               extent=[0, spi.shape[1], 1, K])
ax.set_ylabel('Scale (months)')
ax.set_xlabel('Time (index)')
ax.set_title('SPI 1–K heatmap')
fig.colorbar(im, ax=ax, label='SPI')
plt.tight_layout()
```

Please see the [Visualization Guide](visualization_guide.md) for further details about plotting methods.

---

## 7) Trends and deficit quantification on the CDN

The Cumulative Deviation from Normal (`ds.CDN`) integrates the standardized
anomaly over time, so it is a natural starting point to identify
**multi-year cycles of drought or wet conditions** and to translate them into
**physical water deficits / surpluses**.

### Detecting trends

`find_trends(windows=W)` returns the water deficit/surplus accumulated over a
moving window of `W` months, in native units, obtained from the **reverse SPI**:
the SPI-like index at accumulation scale `W`, mapped back through the fitted
distribution (`spi_to_native`) relative to the SPI = 0 reference. These are
exactly the bars `plot_trends()` draws.

```python
R = ds.find_trends(windows=[12, 36])
# R[36]['spi']            : SPI-like index at scale 36
# R[36]['anomaly']        : deficit (<0) / surplus (>0), mm for Precipitation,
#                           m³ of total volume for Streamflow
# R[36]['anomaly_masked'] : the same, zeroed where |SPI36| < 0.5 (the near-neutral
#                           band), i.e. what gets plotted
# R[36]['unit']           : 'mm' or 'm3'
```

> Until 2026-09 this method ran a rolling OLS regression on the CDN and reported
> slopes and p-values. The CDN is a cumulative sum, so those p-values were
> spurious by construction, and `plot_trends()` had already stopped using them.
> For a genuine test of a sustained wet/dry phase, use
> `utils.statistics._rolling_phase_test`, which tests the **mean of SPI-1**
> (approximately serially independent) over a window against zero.

Larger `W` filters short-term oscillations and emphasizes structural cycles.
For typical basin-scale analyses, **W = 36–60 months** captures multi-annual
drought / pluvial episodes that matter for water resource planning.

### Quantifying the deficit/surplus in physical units

Detecting a trend is one thing; communicating its magnitude as
"X mm of missing rainfall" or "Y million m³ of missing discharge"
requires conversion from standardized to physical units. Drought-Scan
offers two complementary methods, both returning native units (mm for
`Precipitation`, total m³ for `Streamflow`, derived units for `Pet`/`Balance`):

- **`deficit_from_spi(window)`** — *statistical-rarity perspective.*
  The SPI-like index at the matching accumulation scale is converted back
  to native units via the calibrated inverse transform, taking SPI = 0 as
  reference (which by construction equals `normal_values()`). This is the
  method used internally by `plot_trends` (see
  [Visualization Guide §5](visualization_guide.md#5-trend-detection-in-cdn)).
  Anchors the deficit to the **statistical exceptionality** of the event,
  preserving symmetry between dry and wet tails in standardized space.

- **`volume_anomaly_rolling(window)`** — *physical water-balance perspective.*
  Direct summation of monthly `(obs − normal)` anomalies over the window.
  Returns the **observed physical deficit/surplus** — the quantity that
  water managers, irrigation boards, and ecological-flow assessments
  recognise. No statistical transformation involved.

The two estimates correlate strongly but **diverge at extremes**, especially
for precipitation, which is bounded below by zero (you cannot rain less than
nothing) but unbounded above. The SPI-based method keeps the two tails
balanced in standardized space; the direct sum is more honest about the
physical asymmetry of the variable.

### Which one should I use?

A practical guide:

| Goal | Recommended method |
|------|-------------------|
| Reporting headline numbers in papers and outreach | `deficit_from_spi` |
| Reservoir / water-balance accounting (real cubic metres) | `volume_anomaly_rolling` |
| Cross-basin comparison of drought severity | `deficit_from_spi` |
| Sanity check on either method | use both, compare |

In the spirit of robust analysis, **reporting both** in a methods section is
often the cleanest choice: the SPI-based estimate as the primary,
statistically-anchored figure, and the direct volumetric anomaly as the
observation-grounded counterpart.

```python
window = 36

# Statistically-anchored deficit
d_spi = ds.deficit_from_spi(window=window)

# Physically-observed deficit
d_obs = ds.volume_anomaly_rolling(window=window)

# Peak event comparison
import numpy as np
idx = np.nanargmin(d_spi)
print(f"At {ds.m_cal[idx]} over the last {window} months:")
print(f"  deficit_from_spi:        {d_spi[idx]:.3e}")
print(f"  volume_anomaly_rolling:  {d_obs[idx]:.3e}")
```

For visualization, see `plot_trends` in the
[Visualization Guide §5](visualization_guide.md#5-trend-detection-in-cdn),
which combines the CDN curve with the deficit bars in a single figure.

> **Note on hydrological regimes**: divergences between the two methods, and
> between the deficit of precipitation and the deficit of streamflow, often
> carry interpretive value. A streamflow deficit much larger than the
> corresponding precipitation deficit may signal **destocking from cryospheric
> or groundwater reservoirs** (glaciers, snowpack, aquifers); a streamflow
> deficit much smaller may signal **buffering by lake regulation or
> reservoir operation**. These contrasts are valuable diagnostic information,
> not artefacts.
---


## 8) Streamflow (SQI), Pet and Balance (SPEI) classes
For drought analysis based on other standardized indices like SQI, 
SPEI or SPETI you can use the corresponding `Streamflow`, `Balance` and `Pet` classes. 
They share the same initialization philosophy: provide `ts/m_cal` **or** file paths, 
set `K`, `baseline`, `calculation_method`, and optionally a `threshold` aligned with your risk definition. Outputs include **SQI/SPEI/SPETI** (SPI-like arrays), **SIDI**, and **CDN** computed by using the 1-month scale of the obtained index.



The **substantial differences** are limited to:

- **Data source and I/O**  
  - *Precipitation*, *Pet* and *Balance*: typically read **NetCDF** variables, with possible names defined internally.  
  - *Streamflow*: accepts **CSV/Excel** point series, with utilities for gap-filling.  

Note: If daily data are detected in Streamflow, they are **averaged** to monthly means. In contrast, `Precipitation`, `Pet`, and `Balance` are **accumulated** over the month.


- **Domain of values and recommended methods**  
  All classes default to `f_kde` (non-parametric, since v3.1.0). However, depending on the
  variable's statistical properties, you may prefer a parametric method:
  - *Precipitation*: strictly positive, strongly skewed → `f_spi` (Gamma) is the classical choice.  
  - *Streamflow*: strictly positive, strongly skewed, with possible zeros/missing → `f_spi` (Gamma), with special handling for gaps.  
  - *Pet*: positive but less skewed → `f_kde` (default) or `f_zscore`.  
  - *Balance (P–PET)*: can be negative as well as positive → `f_spei` (Pearson III) is the standard SPEI transform.  

  Use the diagnostic tools in [statistics_tools.md](statistics_tools.md) to verify which
  distribution best fits your data.

- **Interpretation**  
  - *Precipitation*: meteorological drought (SPI/SIDI/CDN).  
  - *Streamflow*: hydrological drought (SQI/SIDI/CDN), directly comparable to precipitation through correlation.  
  - *Pet*: climatic driver that can be used on its own or combined with precipitation.  
  - *Balance*: meteorological drought using the input for SPEI-like indices.  

In practice this means that, aside from the different input format and sensible defaults,
all classes are **symmetric**: once initialized they provide the same workflow and
diagnostic outputs, allowing the user to compare meteorological, hydrological and climatic
drought signals under a unified framework.



```python
import drought_scan as DS
shape_path = 'tests/data/bacino_pontelagoscuro.shp'
river_path = 'tests/data/ARPAE_Q_month.csv'
tb1 = 1961
tb2 = 2020
streamflow = DS.Streamflow(data_path = river_path,
                        shape_path=shape_path,
                        start_baseline_year=tb1,
                        end_baseline_year=tb2,
                        basin_name = 'Po')

 
```


## 9) Streamflow (SQI) — symmetry with Precipitation

Precipitation and streamflow are intrinsically linked as part of the hydrological cycle and represent key indicators for understanding drought. A reduction in precipitation can directly lead to decreased river discharge, reduced groundwater recharge, and lower reservoir storage. This extends the impacts of drought on water availability over time, often with a delayed effect. Drought-Scan explicitly analyzes this relationship through the correlation between the Standardized Drought Integration Index (SIDI), derived from SPI, and the one-month Streamflow Drought Index (SQI1).

## 9.1) Reproducibility tips

- Fix your **baseline** and stick to it across runs for fair comparisons between Precipitation and Streamflow.  
- Streamflow data formats accepted are CSV or Excel.
- Run the `analyze_correlation` method on the driver object (Precipitation, Pet, or Balance).
- Recompute the optimal SIDI.

The `analyze_correlation` method compares drought indices (SIDI) with the streamflow standardized index (SQI1) in order to identify the temporal scale and weighting scheme that maximize their correlation.
It works by testing different month-scales (K) and weighting functions applied to the SPI ensemble, then calculating the coefficient of determination (R²) against the streamflow SQI1.

What it does:
- Finds the overlapping time period between the driver and streamflow data.
- Computes SIDI values for multiple temporal scales (K) and weighting schemes (equal, linear, geometric).
- Evaluates the correlation (R²) between each SIDI configuration and the streamflow SQI1.
- Identifies the best K and weighting scheme that maximize correlation.
- Optionally produces plots showing how R² varies with K across weighting schemes, the relationship between the optimized SIDI and SQI1, and a diagnostic scan plot with the optimal configuration.

NOTE: This optimization task does not require that the SIDI and SQI1 time series cover the same time interval, thus facilitating analyses even for situations where the streamflow data are shorter or only partially temporally overlapped with the precipitation data.

```python
import drought_scan as DS
shape_path = 'tests/data/bacino_pontelagoscuro.shp'
prec_path  = 'tests/data/LAPrec1871.v1.1.nc'
river_path = 'tests/data/ARPAE_Q_month.csv'
# ------------------ 
print("\n--- Precipitation-to-Streamflow Analysis ---")
# define the baseline, it must be the same for precipitation and streamflow analysis
tb1 = 1961
tb2 = 2000
ds = DS.Precipitation(
    prec_path=prec_path,
    shape_path=shape_path,
    start_baseline_year=tb1,
    end_baseline_year=tb2,
    basin_name='Po'
)



streamflow = DS.Streamflow(data_path = river_path,
                        shape_path=shape_path,
                        start_baseline_year=tb1,
                        end_baseline_year=tb2,
                        basin_name = 'Po')



# let's look to the SIDI vs SQI1 correlation:
A = ds.analyze_correlation(streamflow, plot=True)
# NB: dots can be coloured by season (April-October and November-March):
A = ds.analyze_correlation(streamflow, plot=True, plot_mode='seasonal')
# or by month
A = ds.analyze_correlation(streamflow, plot=True, plot_mode='monthly')



# if desired, SIDI can be recomputed with optimal K and weight_index and become a proxy for SQI1
ds.set_optimal_SIDI(
    optimal_k=A['best_k'],
    optimal_weight_index=A['col_best_weight'],
    overwrite=True
)

# Once the optimal SIDI has been recalculated and OVERWRITTEN it is possible to plot the opt-SIDI/SQI1 covariates:
ds.plot_covariates(streamflow, year_ext=(2000, 2019))


# Option 2 (no overwrite): get the full SIDI matrix for K=best_k and pick a column
SIDI_matrix = ds.recalculate_SIDI(K=A['best_k'])              # shape: (time, n_weightings)
sidi_opt    = SIDI_matrix[:, A['col_best_weight']]            # 1D vector (time,)

```

**Each weighting scheme has its own optimal K.** `A['best_k']` is the K of the single
best (K, weight) pair — applying it to all five columns leaves the other four
calibrated at a scale that is not their own. `analyze_correlation` also reports the
optimum of every scheme, so each column of SIDI can keep its own scale:

```python
A['best_k_per_weight']            # ndarray (5,) — optimal K of each weighting scheme
A['max_correlation_per_weight']   # ndarray (5,) — R² each scheme reaches at its own K
A['MatCorr']                      # ndarray (K, 5) — the full R² surface

# every column optimized on its own terms; col_best_weight stays the one to read
ds.set_optimal_SIDI(
    optimal_k=A['best_k_per_weight'],
    optimal_weight_index=A['col_best_weight'],
    overwrite=True
)
```

Passing a single integer keeps the historical behaviour, so existing code is unaffected.

### 9.1.1) Seasonal correlation analysis

The method `analyze_correlation_seasonal` repeats the same optimization **per season**,
allowing different K and weighting schemes for different parts of the year.
This is especially useful in basins where the precipitation-streamflow relationship
varies seasonally (e.g., snowmelt-dominated winters vs rain-fed summers).

```python
# Seasonal analysis (quarterly)
seasonal_corr = ds.analyze_correlation_seasonal(streamflow, agg='quarter', plot=True)

# Available aggregation modes: 'quarter', 'semiannual', 'four-monthly', 'monthly', 'custom'
# For custom seasons:
my_seasons = {'wet': [10, 11, 12, 1, 2, 3], 'dry': [4, 5, 6, 7, 8, 9]}
seasonal_corr = ds.analyze_correlation_seasonal(streamflow, agg='custom',
                                                 seasons=my_seasons, plot=True)

# Apply seasonal optimization
ds.set_optimal_SIDI_seasonal(seasonal_corr, agg='quarter', overwrite=True)
```

For the **gridded** seasonal SIDI, pass the same `seasonal_corr` (with its `agg`)
straight to `spatial_sidi` — no need to commit it first:

```python
ds.spatial_sidi(seasonal_params=seasonal_corr, agg='quarter')
```

See the [Spatial Guide](spatial_guide.md) §2.6 for both the committed and
non-committed routes.

### 9.1.2) Uncertainty on R²(w, K): the block bootstrap

`analyze_correlation` / `analyze_correlation_seasonal` return one R²(w, K)
surface. It came from one particular stretch of history; a slightly different but
equally believable weather record would have moved the numbers, and maybe changed
which K looks best. Pass `n_boot > 0` and the method rebuilds the whole
calculation on many resampled histories, then reports how much the answer moves:

```python
A = ds.analyze_correlation(
    streamflow,
    n_boot=500,            # 0 (default) -> behaviour and return value unchanged
    block_length=None,     # L in months; default max(24, 2*K) rounded to whole years
    ci=(2.5, 97.5),        # band percentiles
    circular=True,         # wrap blocks around the series end
    random_state=0,
)
A["MatCorr"]        # (K, 5)      the point estimate, exactly as before
A["MatCorr_ci"]     # (2, K, 5)   [lo, hi] percentile band
A["MatCorr_boot"]   # (B, K, 5)   every replica's surface (for your own stats)
A["boot_meta"]      # dict: block_length, n_blocks, contaminated_fraction (K,), ...
A["summary"]        # cluster table: cluster / K / sub-cluster / R2 — see below
```

`analyze_correlation_seasonal(..., n_boot=500)` builds **one** replica set on the
continuous overlap and slices it per season, so each season's dict gains
`"R2_boot"`, `"R2_ci"` and `"summary"`, plus a top-level `result["summary"]`
table with one row per season.

---

#### Why a *block* bootstrap and not the classic one

The classic bootstrap resamples **one month at a time**, as if months were
independent. They are not: rainfall and streamflow persist from one month to the
next, and SPI/SQI are themselves moving averages that share input months between
consecutive points by construction. Month-by-month resampling would break that
link and make the data look richer in independent information than it is, giving
an artificially narrow uncertainty. The fix is to resample **whole blocks of
consecutive months**.

#### Step 1 — Block length `L`

`L` must be long enough to contain the system's memory:

> `L` = the larger of 24 months (2 years) and 2 × the maximum `K` tested, then
> rounded to the nearest whole number of years.

Twice `K`, not `K`: every join between two blocks produces a small unusable zone
`K-1` months wide (Step 5). With `L = K` almost the whole block would be that
zone; with `L = 2K` the damaged part stays a limited fraction and the rest is
clean.

**Whole-year blocks** (multiples of 12 months): SPI and SQI are computed
calendar-month by calendar-month (January's fit uses only past Januaries, etc.).
A block that cut a year in half would distort the balance of the 12 months in the
synthetic series. Whole-year blocks keep every replica on a valid calendar; the
library lays the blocks end-to-end under a fresh contiguous calendar so that each
synthetic month equals its real month.

#### Step 2 — The pool of blocks

From the real series (N years long) every run of `block_years` consecutive years
is a candidate block: one starting at year 1, one at year 2, and so on — these
overlap (the *moving block bootstrap*). With `circular=True` the series wraps
after its last year, so years near the ends are not systematically under-sampled.

#### Step 3 — Resample rainfall and streamflow **together**

Blocks are drawn at random, with replacement, until the original length is
covered. **The same span of years is always drawn for both series** — never
independently — otherwise the rainfall→streamflow link the analysis exists to
measure would be broken and the result would be meaningless.

#### Step 4 — Rebuild everything from scratch

Nothing is reused: on each synthetic series the whole chain is recomputed —
SPI at every scale `K`, then SQI1, then the full R²(w, K) surface. Only this way
does the estimated uncertainty reflect how the *entire procedure* would move.

#### Step 5 — The seams, and how they are handled

At every block join the synthetic series places side by side two moments that
were never consecutive in reality (e.g. October of one year followed by March of
another). Individual values are fine; the problem is the **K-month moving
average** (SPI_K): near a seam it blends months that were not really adjacent,
producing a physically meaningless accumulation.

The fix: the months within `K-1` steps of a seam are **discarded (set to NaN)**
before R² is computed for that scale. The larger `K`, the more months are dropped
near each seam — unavoidable, since a longer window reaches further back.

**Practical consequence:** as `K` grows, fewer valid points remain, so a wider
band at high `K` does **not** necessarily mean "more physical uncertainty" — it
can partly just mean "less data left, for a technical reason". The library prints
a table (also in `boot_meta["contaminated_fraction"]` and
`boot_meta["eff_n_per_scale"]`) of how much was dropped and how much survives at
each `K`. **Read that table before interpreting the width of a band.**

#### Step 6 — Repeat

Steps 3–5 are repeated `n_boot` times. A few hundred to a couple of thousand is
usual; the library's interactive default is 500. (The diagnostics site build
leaves it off by default — it re-runs the full pipeline `B` times and is slow
with `f_kde` at large `K`.)

#### Step 7 — The confidence interval

For each (w, K) cell, the `ci` percentiles (default 2.5 / 97.5) of the `n_boot`
values are the 95% band: *"reran under a slightly different but statistically
similar history, 95% of the time R² would land in here."*

#### Step 8 — Seasons: one bootstrap, sliced afterward

Do **not** bootstrap each season on its own months only: a long-`K` SPI computed
in (say) January reaches back into the previous February–December, so cutting the
season out *before* rebuilding SPI would lose that memory. The procedure is: one
bootstrap on the whole continuous series (Steps 1–6), then — only at the end,
after SPI/SQI1 are rebuilt — keep the months of the season of interest and
compute R² on that subset.

#### The `summary` table — clusters and sub-clusters

`analyze_correlation_seasonal(..., n_boot>0)` builds a table with **one row per
(season, K cluster, R² sub-cluster)** at `result["summary"]` (a pandas
DataFrame); the raw per-season pieces are in `result[season]["summary"]`.
`analyze_correlation` builds the same with `season = "whole period"`. It is also
printed.

For **each** of the 5 weighting schemes the bootstrap gives its **peak**: the
highest R² it reaches over all K, the K where that happens, and a 95 % CI for
**both** — the R² CI (percentiles of the per-replica peak R²) and the **K CI**
(percentiles of the per-replica peak K, a whole-number range). Those are the
horizontal + vertical arms of the **cross** in the peak figure.

**Level 1 — clusters, by timescale.** Instead of grouping schemes by how well
they score, group them by **the K they peak at**. Two schemes that peak at
(nearly) the same K are describing the same *response mechanism* of the basin.

- *Build:* two schemes are linked when **each one's peak K falls inside the
  other's K CI** (symmetric, on K); the clusters are the connected groups. A
  scheme whose peak K sits clearly outside the others' K CIs stays **on its own**.
  A flat, unresolved season (every K CI is wide) links everything into one
  cluster with a wide box.
- *Per cluster:* `K` = the **median** over the cluster's schemes of their peak K;
  `K_CI` = percentiles of the per-replica median. Median, not maximum → no
  "winner's curse".

**Level 2 — sub-clusters, by response.** Within a cluster that holds more than
one scheme, ask whether those schemes reach the **same R²** or not. The **same
rule** is re-applied, now on peak R²: two schemes stay together when each one's
peak R² falls inside the other's **R² CI**; schemes that reach clearly separated
R² split into their own sub-cluster. A one-scheme cluster has a single
sub-cluster (that scheme's own R² CI).

- *Per sub-cluster:* `R2` = the median peak R² over its schemes; `R2_CI` =
  percentiles of the per-replica median. Sub-clusters are numbered by
  **decreasing R²**, so `sub-cluster` 1 is always the strongest.

**Cluster reference — a one-name handle.** Two or more schemes lumped into one
K cluster describe the same mechanism, but `K`/`R2` above are cluster
**medians**, not any one scheme's own numbers. For a quick label, each cluster
also carries a *reference scheme*: for a **2-scheme** cluster, the member with
the **shorter** peak K (the conservative read — the shorter memory already
captures the shared signal); for a **3+-scheme** cluster, the member whose
peak K is **closest to the cluster median** (ties → shorter K). No scheme is
privileged, not even EW. It is reported with **that scheme's own** peak K and
R², each with its own bootstrap CI (`ref_K_CI` / `ref_R2_CI`) — not the
cluster's median values. Purely a label: the SIDI itself keeps using each
scheme's own K regardless of which one is picked as reference.

The printed recap (below the table) shows one line per cluster with this
reference, scheme names abbreviated for that line only (`ew`, `ldw`, `geodw`,
`liw`, `geoiw` — the table itself keeps the full `WEIGHT_LABELS` names), and
the cluster's median K in parentheses as a reminder of how the reference was
picked:

```
  cluster reference (2 schemes -> shorter K; more -> nearest the cluster median K):
    DJF        cluster 1:  ref = liw  K=4 [3, 6]  R2=0.664 [0.602, 0.752]   (cluster median K = 4.0)
```

| column | plain meaning |
|---|---|
| `cluster` | K-cluster ordinal (1, 2, …), by increasing `K`. No interpretive label — the box is meant to speak for itself. |
| `K` `[K_CI]` | the cluster's typical response time in months, with its whole-number CI — the box's K-extent in the peak/cluster figure. Repeated on each sub-cluster row. |
| `ref_scheme` | a one-name handle for the cluster (see above). Repeated per row. |
| `ref_K` `[ref_K_CI]` | that reference scheme's **own** peak K and its bootstrap CI (not the cluster median). Repeated per row. |
| `ref_R2` `[ref_R2_CI]` | that reference scheme's **own** peak R² and its bootstrap CI. Repeated per row. |
| `sub-cluster` | ordinal (1, 2, …) within the cluster, by decreasing `R2`. One row only when the cluster's schemes do not differ in R². |
| `families` | the weighting schemes in that sub-cluster. |
| `R2` `[R2_CI]` | how well that sub-cluster explains the season, typically, with its CI — the box's R²-extent for that sub-cluster in the peak/cluster figure. |

Reading it: one cluster → a single response scale (a wide `K_CI` = the record
cannot pin it down). Several clusters → distinct mechanisms at different K. Two
or more sub-clusters inside one cluster → schemes that share a scale but not a
strength; one sub-cluster → they agree.

The per-season dict also has `peak_by_family` — `peak_R2`, `argmax_K`, `peak_CI`
and `K_CI` for every scheme, if you want them directly.

#### What the figures show (with `n_boot>0`)

The peak/cluster figure — **the first figure** of `analyze_correlation` /
`analyze_correlation_seasonal`, and the *only* R²(k) figure on the diagnostic
site (global and seasonal calibration alike): the R²(k) curves, plus a dark grey
**cross** at each scheme's peak (horizontal arm = the peak-K CI, vertical arm =
the peak-R² CI), plus each **K cluster** as one or more very transparent
**boxes**: all share the cluster's `K_CI` width and are stacked at the `R2_CI`
height of each response **sub-cluster**, each tinted with that sub-cluster's
leading-scheme hue and framed by light-grey dotted CI reference lines. A cluster
that does not sub-split by R² shows a single box. The plain R²(k) curves — each
with its per-cell percentile band — are still produced, demoted to the last
figure.

#### In one sentence

Many plausible alternative histories are rebuilt by resampling whole consecutive
chunks of rainfall-and-streamflow **together**, the entire method is recomputed
on each, the points where resampling glued together moments that were never
really adjacent are discarded, and what is reported is not just where the peak
falls but how stable that choice is — an interval, not a bare point.

### 9.1.3) Understanding SIDI optimization states

After running `analyze_correlation` or `analyze_correlation_seasonal`, the SIDI
can be optimized in two ways. Understanding the difference is important because
it affects how downstream methods (plotting, gap filling, forecasting) select
the correct SIDI.

**Global optimization** (`set_optimal_SIDI`):
the same K and weight_index are applied to all months.

```python
A = ds.analyze_correlation(streamflow)
ds.set_optimal_SIDI(A['best_k'], A['col_best_weight'], overwrite=True)

# After this call:
#   ds.optimal_k             → int (the chosen K)
#   ds.optimal_weight_index  → int (the chosen column of SIDI)
#   ds.SIDI                  → shape (N, 5), read column ds.optimal_weight_index
#   ds.is_seasonal_sidi      → False
```

With a scalar K, **only the `optimal_weight_index` column is calibrated at its own
optimum** — the other four are computed at a K chosen for a different scheme, so they
should not be read. To keep all five interpretable, pass one K per scheme:

```python
ds.set_optimal_SIDI(A['best_k_per_weight'], A['col_best_weight'], overwrite=True)

# After this call:
#   ds.optimal_k             → ndarray (5,) — one K per weighting scheme
#   ds.optimal_weight_index  → int (the column to read by default)
#   ds.SIDI                  → shape (N, 5), every column at its own optimal K
```

**Seasonal optimization** (`set_optimal_SIDI_seasonal`):
each season gets its own K, for each weighting scheme. Two products come out of
it and both are kept — see the note below.

```python
S = ds.analyze_correlation_seasonal(streamflow, agg='quarter')
ds.set_optimal_SIDI_seasonal(S, agg='quarter', overwrite=True)

# After this call:
#   ds.seasonal_params       → dict with per-season config
#   ds.SIDI                  → shape (N, 5), one column per weighting scheme,
#                              each at ITS OWN per-season K
#   ds.SIDI_seasonal_best    → shape (N,), the single best-(K, weight)-per-season
#                              series (the real-time monitoring index)
#   ds.is_seasonal_sidi      → True
#   ds.optimal_k             → does NOT exist (K varies by season)
```

> **Changed in 4.0.0.** `ds.SIDI` used to be the 1-D `SIDI_seasonal_best` series
> tiled across five identical columns. `ds.SIDI` now always means the same thing —
> `(time, 5)` with the column being the weighting scheme — and the 1-D series lives
> alongside it in `ds.SIDI_seasonal_best`. Code that read `ds.SIDI[:, 0]` to get
> "the" seasonal series must read `ds.SIDI_seasonal_best` instead.

The five per-scheme seasonal series are what the water-balance tables are built
from; get them for a non-committed object with:

```python
seasons = {"DJF": [12, 1, 2], "MAM": [3, 4, 5],
           "JJA": [6, 7, 8],  "SON": [9, 10, 11]}

S = ds.analyze_correlation_seasonal(streamflow, seasons=seasons, plot=False)
all_schemes = ds.recalculate_SIDI_seasonal_all_schemes(S, seasons)   # (N, 5)
best_only   = ds.recalculate_SIDI_seasonal(S, seasons)               # (N,)
```

**How downstream code selects SIDI:**

| Method | What it reads |
|--------|---------------|
| `plot_scan(weight_index=w)` | `ds.SIDI[:, w]` — works in both cases |
| `plot_covariates(streamflow)` | auto-selects from `optimal_weight_index` or seasonal |
| `gap_filling(ds)` | requires `overwrite=True`; reads the active SIDI |
| ESM scenarios | re-applies optimization automatically after recalculation |

**Rule of thumb**: always call `set_optimal_SIDI` or `set_optimal_SIDI_seasonal`
with `overwrite=True`. If you want to go back to the default SIDI,
re-initialize the Precipitation object.


## 9.2) Streamflow Gap Filling
Observed streamflow time series may contain **missing values** due to monitoring gaps or sensor errors.  
The method `gap_filling` of the `Streamflow` class allows you to fill short gaps and preserve continuity in index calculation.

**Concept.** Gaps in the streamflow record are reconstructed **using the precipitation-based SIDI** that best explains SQI1.  
You must first **optimize the SIDI configuration** on the Precipitation object with `set_optimal_SIDI` (with `overwrite=True`), then pass the Precipitation object to `Streamflow.gap_filling`.

```python
# Assuming ds (Precipitation) and streamflow (Streamflow) are already initialized
# and A holds the results from analyze_correlation:

print("Best K:", A['best_k'], "Best weight index (SIDI):", A['col_best_weight'])

# 1) Overwrite SIDI on the precipitation object with optimal settings
ds.set_optimal_SIDI(
    optimal_k=A['best_k'],
    optimal_weight_index=A['col_best_weight'],
    overwrite=True
)

# 2) Gap filling — pass the precipitation object (whose SIDI is now optimized)
streamflow.gap_filling(ds)
```

## 9.3) Month-wise SPIₖ–SQI₁ Correlation (`spi_sqi_corr`)

The method `spi_sqi_corr` provides a detailed month-by-month diagnostic of how 
drought conditions propagate into hydrological drought.

While `analyze_correlation` identifies the *optimal* multi-scale SIDI configuration,
`spi_sqi_corr` focuses on the **raw physical relationship** between:

- each accumulation scale index at scale **k**,  
- and the streamflow one-month index **SQI₁**,  

across the **12 calendar months**.

This allows the user to quantify seasonal differences in drought propagation and 
identify which time-scales are most influential for river discharge in each month.

### What the method computes

- Automatically finds the overlapping period between the driver and Streamflow.
- For each month (Jan…Dec) and each scale k = 1…K:
  - computes the Pearson correlation ρ(index_k, SQI₁),
  - retains only statistically significant correlations (p < 0.05),
  - stores the determination coefficient **R² = ρ²**.
- Returns a **12 × K matrix** of R² values.
- Optionally produces a contour heatmap to visualize the propagation patterns.

### When to use it

Use `spi_sqi_corr` when you need:

- a **diagnostic map** of meteorological → hydrological drought propagation,  
- identification of **seasonally dependent response times**,  
- insight on which time-scales dominate in specific months,  
- comparison of catchments with different hydrological memory,  
- validation before SIDI optimization.

It is especially useful in catchments where snowmelt, reservoir regulation or 
irrigation withdrawals create **seasonal asymmetries** between precipitation and discharge.

> **Note**: `spi_sqi_corr` is available on `Precipitation`, `Pet`, and `Balance` objects.
> The method name is the same for all classes.

### Example

```python
import drought_scan as DS

shape_path = 'tests/data/bacino_pontelagoscuro.shp'
prec_path  = 'tests/data/LAPrec1871.v1.1.nc'
river_path = 'tests/data/ARPAE_Q_month.csv'

# Initialize Precipitation and Streamflow with the same baseline
tb1, tb2 = 1961, 2000

prec = DS.Precipitation(
    prec_path=prec_path,
    shape_path=shape_path,
    start_baseline_year=tb1,
    end_baseline_year=tb2,
    basin_name='Po'
)

streamflow = DS.Streamflow(
    data_path=river_path,
    shape_path=shape_path,
    start_baseline_year=tb1,
    end_baseline_year=tb2,
    basin_name='Po'
)

# Compute the month-wise SPIk–SQI1 correlation matrix (R²)
R2 = prec.spi_sqi_corr(streamflow, plot=True)

print("Shape of R² matrix:", R2.shape)   # Expected: (12, K)
```
---


## 10) Pet and Balance utilities 

## 10.1) PET analysis (Potential Evapotranspiration)

PET datasets can be analyzed directly with the `Pet` class.  
An example NetCDF file is provided in `tests/ERA5_monthly_pev.nc`. The workflow mirrors the precipitation setup.

```python
import drought_scan as DS
shape_path = 'tests/data/bacino_pontelagoscuro.shp'
pet_path = 'tests/data/ERA5_monthly_pev.nc'
tb1 = 1953
tb2 = 2003
pet = DS.Pet(
    data_path=pet_path,
    shape_path=shape_path,
    start_baseline_year=tb1,
    end_baseline_year=tb2,
    basin_name='Po'
)

print("PET time series shape:", pet.ts.shape)
print("SPI-like PET indices:", pet.spi_like_set.shape)
print("SIDI from PET:", pet.SIDI.shape)

```


Use PET as an independent climatic driver or combine it with precipitation to build water balance indicators.

---

## 10.2) Balance (P–PET) > SPEI

The `Balance` class computes the **monthly climatic water balance** (precipitation (P) minus potential evapotranspiration (PET)).  
This is the standard input for SPEI index, which captures drought as a function of both supply (P) and virtual water demand (PET).

NOTE 1: using gridded data for P and PET with different spatial resolutions is not a problem: data are first imported, aggregated spatially over the basin and then, when single monthly timeseries are ready for P and PET, derive the P-PET timeseries used to initialize the instance of *Balance*.

NOTE 2: using input data for P and PET covering a different time-span is not a problem: the script selects only data on a common timestamp which is reported in `.m_cal`. 
```python
import drought_scan as DS
prec_path = 'tests/data/LAPrec1871.v1.1.nc'
pet_path = 'tests/data/ERA5_monthly_pev.nc'
shape_path = 'tests/data/bacino_pontelagoscuro.shp'
tb1 = 1961
tb2 = 2000
spei = DS.Balance(
    prec_path=prec_path,
    pet_path=pet_path,
    shape_path=shape_path,
    start_baseline_year=tb1,
    end_baseline_year=tb2,
    basin_name='Po',
)

print("length of spei timeseries (P–PET):", spei.ts.shape)
print("shape of the SPEI-like indices (1–K months):", spei.spi_like_set.shape)
print(f"time-span: {spei.m_cal}")
```

This setup is particularly useful in climate change studies, where increasing PET may exacerbate drought even under stable precipitation.



## 11) Temperature class

The `Temperature` class extends the same philosophy used for `Precipitation`, `Pet`, `Balance`, and `Streamflow`, 
but is specialized for temperature datasets.

- **Input handling**  
  - Accepts **NetCDF** temperature datasets (daily or monthly).  

- **Defaults**  
  - The default `calculation_method` is `f_kde` (non-parametric), consistent with all other classes since v3.1.0.
  - Since temperature is generally close to Gaussian and can take both positive and negative values,
    `f_zscore` is also a natural choice and can be passed explicitly.
  - The default `threshold` is `+1` (positive anomalies flag warm events).

- **Interpretation**  
  - Provides standardized indices of temperature variability, which can be used as an independent drought driver 
    (e.g., heat stress episodes) or in conjunction with other classes.  
  - Outputs include the usual **SPI-like set** (in this case, temperature indices), **SIDI**, and **CDN** 
    computed from the 1-month scale.

```python
import drought_scan as DS
shape_path = 'tests/data/bacino_pontelagoscuro.shp'
temp_path = 'tests/data/ERA5_monthly_t2m.nc'
tb1 = 1953
tb2 = 2024
Temp = DS.Temperature(
    data_path=temp_path,
    shape_path=shape_path,
    start_baseline_year=tb1,
    end_baseline_year=tb2,
    basin_name='Po'
)

print("T time series shape:", Temp.ts.shape)
print("SPI-like T indices:", Temp.spi_like_set.shape)
print("SIDI from T:", Temp.SIDI.shape)
```
---

## 12) Teleindex class

The `Teleindex` class is meant for **large-scale climate drivers** (e.g., Niño3.4, NAO, AO, IOD), provided as a
single time series with a calendar. It reuses the common pipeline (SPI-like multi-scale set, **SIDI**, **CDN**),
but differs from the hydro-meteorological classes in a few key ways.

- **Input handling**
  - Accepts `ts` + `m_cal` directly **or** `data_path` via `import_timeseries(...)`.
  - No shapefile or spatial aggregation: teleconnections are basin-agnostic, exogenous drivers.
  - If **daily** data are detected, values are **averaged to monthly means** (not summed).
    This mirrors `Temperature` (monthly mean), while `Precipitation`, `Pet` and `Balance` are monthly **sums**.

- **Defaults and normalization**
  - Default `calculation_method` is `f_kde` (non-parametric), consistent with all other classes since v3.1.0.
    You can switch to `f_zscore` if the series is close to Gaussian, or to `f_spei` for robustness with skewed series.
  - Set `index_name` to the specific driver (e.g., `"Niño3.4"`, `"NAO"`) for clear labeling in plots.

- **Purpose and interpretation**
  - Produces a **SPI-like multi-scale set** of the teleconnection, its **SIDI** (weighted multi-scale integration),
    and **CDN** (from the 1-month scale).
  - Intended for **diagnostics and coupling** with basin indicators (e.g., correlation/lag analysis with SIDI from
    precipitation or SQI1 from streamflow) and for **predictor design** in ML workflows.

- **Practical notes**
  - Prefer **raw (non-standardized)** teleconnection series as input; the class will standardize them using the
    selected method over your chosen baseline.
  - Keep **baseline years** consistent with other classes when you plan cross-comparisons.


```python
import numpy as np
import drought_scan as DS
from drought_scan.utils import f_zscore

# Example: 600 months of synthetic data and a matching calendar
ts = np.random.gamma(shape=2.0, scale=30.0, size=600)

years = np.repeat(np.arange(1975, 2025), 12)[:600]
months = np.tile(np.arange(1, 13), 50)[:600]
m_cal = np.column_stack([months, years])

tb1 = 1975
tb2 = 2024

index = DS.Teleindex(ts=ts, m_cal=m_cal, start_baseline_year=tb1,
                     end_baseline_year=tb2, calculation_method=f_zscore,
                     index_name='my_index', verbose=False)
```

---

## Further documentation

- [Visualization Guide](visualization_guide.md) — plotting methods and customization options.
- [Spatial Guide](spatial_guide.md) — gridded SPI/SIDI maps at every grid point within the basin.
- [Statistical Diagnostics](statistics_tools.md) — distribution fitting, goodness-of-fit tests, and standardization tools.
- [Common Errors](common_errors.md) — typical errors and how to fix them.