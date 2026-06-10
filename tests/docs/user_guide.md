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
  Default: `2` (logarithmically decreasing).  
  Options:
  - `0`: equal weights  
  - `1`: linear decreasing  
  - `2`: **logarithmically decreasing** *(default; favors recent months)*  
  - `3`: linear increasing  
  - `4`: logarithmically increasing  

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
ds.severe_events()
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
  - `2`: **logarithmically decreasing** *(default; favors recent months)*  
  - `3`: linear increasing  
  - `4`: logarithmically increasing  

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

`find_trends(window=W)` applies a rolling linear regression to the CDN over a
moving window of `W` months and flags **monotonic, statistically significant**
trends (p < 0.05). Returns four arrays:

```python
window = 36
R = ds.find_trends(window=window)
# R['trend']  : -1 negative, 0 none, 1 positive
# R['slope']  : slope coefficient of the regression
# R['p_value']: p-value of the trend test
# R['delta']  : cumulative CDN change over the window (slope × W), in standardized units
```

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
- Computes SIDI values for multiple temporal scales (K) and weighting schemes (equal, linear, logarithmic).
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
### 9.1.2) Understanding SIDI optimization states

After running `analyze_correlation` or `analyze_correlation_seasonal`, the SIDI
can be optimized in two ways. Understanding the difference is important because
it affects how downstream methods (plotting, gap filling, forecasting) select
the correct SIDI.

**Global optimization** (`set_optimal_SIDI`):
a single K and weight_index are applied to all months.

```python
A = ds.analyze_correlation(streamflow)
ds.set_optimal_SIDI(A['best_k'], A['col_best_weight'], overwrite=True)

# After this call:
#   ds.optimal_k             → int (the chosen K)
#   ds.optimal_weight_index  → int (the chosen column of SIDI)
#   ds.SIDI                  → shape (N, 5), read column ds.optimal_weight_index
#   ds.is_seasonal_sidi      → False
```

**Seasonal optimization** (`set_optimal_SIDI_seasonal`):
each season gets its own K and weight_index. The resulting SIDI is a single
series (tiled to 5 identical columns for backward compatibility).

```python
S = ds.analyze_correlation_seasonal(streamflow, agg='quarter')
ds.set_optimal_SIDI_seasonal(S, agg='quarter', overwrite=True)

# After this call:
#   ds.seasonal_params       → dict with per-season config
#   ds.SIDI                  → shape (N, 5), all 5 columns identical
#   ds.is_seasonal_sidi      → True
#   ds.optimal_k             → does NOT exist (K varies by season)
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