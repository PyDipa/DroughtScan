# Example Usage

This section shows how to initialize a **Drought-Scan** analysis object (DSO), with practical notes on the most important options and how they change the behavior compared to defaults.

> **IMPORTANT NOTE**  
> Make sure you are running **the same Python interpreter** where DroughtScan and its dependencies have been installed.  
> 
> For example, if you installed with:
> ```bash
> python3.10 -m pip install .
> ```
> then you must also start your session with :
> ```bash
> python3.10
> ```
> or set the proper interpreter `python3.10` on your IDE and not  other version Python version  
 


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

You can access to the data as shown in the prints. For examples, ds.ts is the monhtly precipation time series imported and aggregated at the river basin scale, while ds.CDN is the Cumulative Deviation from Normal. 
For the full list of variables and methods explore examples and usage notes below. You can also find 
a full list in the [README](https://github.com/PyDipa/DroughtScan/blob/main/README.md) file 


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
    basin_name='My Basin',
    start_baseline_year=1981,
    end_baseline_year=2010
)
```

This is useful for customized pre-processing pipelines or when your data is already basin-aggregated.

---

## 3) Key parameters (and how to choose them)

Below are the most impactful options, with defaults and when you might want to change them:

- **`K` (int)** — *maximum temporal scale for SPI/SIDI*  
  Default: library default (commonly `K=36`).  
  Interpretation: `K` sets the longest memory of your indices.  
  - `K=36` (3 years): good general-purpose horizon for basin-scale drought.  
  - `K=60` (5 years): emphasizes slow/structural deficits (useful for long-term storage anomalies or policy assessments).  
  - `K=24` (2 years): focuses on shorter integrated dynamics.  
  In short, larger `K` → longer memory and smoother signals.

- **`threshold` (float)** — *severity threshold for events (e.g., on SIDI)*  
  Default: `-1`.  
  Meaning: events below the threshold are flagged as severe.  
  - `-1` corresponds to 1 standard deviation below the mean of a standardized index.  
  - In the  Po River case study (see paper ), `-1` proved effective for **severe** drought identification; adjust for your basin by comparing the SIDI with some observed impact variable.
  
- **`calculation_method` (callable)** — *index family for standardization*  
  Default: `f_spi` (Gamma fit).  
  Available (in `utils.py`):
  - `f_spi` → a function to standardize data according to a **Gamma** distribution. Best for **positive, right-skewed** data (e.g., precipitation). Works fine on positive near-normal as well. Generally used for the calculation of SPI.
  - `f_spei` → **Pearson III** distribution. Handles **real-valued, negative and/or skewed** data; Gnerally used for **SPEI**. Works fine on negative, positive near-normal as well, being also well suited for precipitation data.
  - `f_zscore` → standard z-score. Best when data are **approximately Gaussian** (real-valued); no parametric skew modeling.

  Practical guidance:  
  - Use `f_spi` for precipitation-like data (eg.  SPI/SIDI.  
  - Use `f_spei` for **SPEI-style** applications (precip–PET, can be negative).  
  - Use `f_zscore` when you trust normality and prefer a simpler transform.

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

- **Baseline**: a stable and representative climatological period of at least 30-years is recommended. 50-years in good.  
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
 
Please see the *visualization_guide.md* for further details about plotting methods

---

## 4) Using different index families

Switch to **SPEI-like** behavior (Pearson III) or plain z-score:
By specifing the Index name the plots will have the proper labels. 

```python
from drought_scan.utils import f_spei, f_zscore
import drought_scan as DS
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

Please see the *visualization_guide.md* for further details about plotting methods

---


## 7) Streamflow (SQI), Pet and Balance (SPEI) classes
For drought analysis based on other standardiezed indices like  SQI, SPEI or SPETI you can use the corresponding  `Streamflow`, `Balance ` and `Pet`  classes. They shares the same initialization philosophy; provide `ts/m_cal` **or** file paths, set `K`, `baseline`, `calculation_method` (`f_spi` on positive flows as Streamflow, f_spei for the P-PET balance and f_zscore for PET), and optionally a `threshold` aligned with your risk definition. Outputs include **SQI/SPEI/SPETI** (SPI-like arrays), **SIDI**, and **CDN** computed by using the 1-month scale of the obtained index.


The **substantial differences** are limited to:

- **Data source and I/O**  
  - *Precipitation*, *Pet* and *Balance*: typically read **NetCDF** variables, with possible names defined internally.  
  - *Streamflow*: accepts **CSV/Excel** point series, with utilities for gap-filling and direct reassignment of `ts/m_cal`.  

Note: If daily data are detected in Streamflow, they are **averaged** to monthly means. In contrast, `Precipitation`, `Pet`, and `Balance` are **accumulated** over the month.


- **Domain of values and defaults**  
  - *Precipitation*: strictly positive, strongly skewed → default `Gamma` (`f_spi`).  
  - *Streamflow*: strictly positive, strongly skewed, with possible zeros/missing → default `Gamma` (`f_spi`), with special handling for gaps.  
  - *Pet*: positive but less skewed → defaults often `z-score` (`f_zscore`) or `Pearson III`.  
  - *Balance (P–PET)*: can be negative as well as positive → default `Pearson III` (`f_spei`), i.e. the usual SPEI transform.  

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


## 8) Streamflow (SQI) — symmetry with Precipitation

Precipitation and streamflow are intrinsically linked as part of the hydrological cycle and represent key indicators for understanding drought. A reduction in precipitation can directly lead to decreased river discharge, reduced groundwater recharge, and lower reservoir storage. This extends the impacts of drought on water availability over time, often with a delayed effect. Drought-Scan explicitly analyzes this relationship through the correlation between the Standardized Drought Integration Index (SIDI), derived from SPI, and the one-month Streamflow Drought Index (SQI1)

## 8.1) Reproducibility tips

- Fix your **baseline** and stick to it across runs for fair comparisons between Precipitation and Streamflow.  
- Streamflow data format acceped are CSV or Excel
- run the Precipitation method "analyze_correlation";
- recompute the optimal SIDI

The "analize_correlation" Precipitation method  compares precipitation-based drought indices (SIDI) with the streamflow standardiezed index (SQI1) in order to identify the temporal scale and weighting scheme that maximize their correlation.
It works by testing different month-scales (K) and weighting functions applied to the precipitation SPI ensemble, then calculating the coefficient of determination (R²) against the streamflow SPI1.
What it does:
Finds the overlapping time period between precipitation and streamflow data.
Computes SIDI values for multiple temporal scales (K) and weighting schemes (equal, linear, logarithmic).
Evaluates the correlation (R²) between each SIDI configuration and the streamflow SPI1.
Identifies the best K and weighting scheme that maximize correlation.
Optionally produces plots showing:
How R² varies with K across weighting schemes.
The relationship between the optimized SIDI and SQI1.
A diagnostic scan plot with the optimal configuration.

NOTE: This optimization task does not require that the SIDI and SQI1 time series cover the same time interval, thus facilitating analyses even for situations where the streamflow data are shorter or only partially temporally overlapped with the precipitation data.

```python
import drought_scan as DS
shape_path = 'tests/data/bacino_pontelagoscuro.shp'
prec_path  = 'tests/data/LAPrec1871.v1.1.nc'
river_path = 'tests/data/ARPAE_Q_month.csv'
# ------------------ 
print("\n--- Precipitation-to-Streamflow Analysis ---")
# define the baseline, it must be the same for preciptiation and streafmlow analysis
tb1 = 1961
tb2 = 2000
ds = DS.Precipitation(
    prec_path=prec_path,
    shape_path=shape_path,
    start_baseline_year=tb1,
    end_baseline_year=tb2,
    basin_name='Po'  # only used for labeling/plots
)



streamflow = DS.Streamflow(data_path = river_path,
                        shape_path=shape_path,
                        start_baseline_year=tb1,
                        end_baseline_year=tb2,
                        basin_name = 'Po')



# let's look to the SIDI vs SQI1 correlation:
A = ds.analyze_correlation(streamflow,plot=True)
# NB: dots can be coloured by season (April-October and November-March):
A = ds.analyze_correlation(streamflow,plot=True,yellow=False,seasonal=True)
# or by month
A = ds.analyze_correlation(streamflow,plot=True,yellow=False,seasonal=False)


# if desiderd, SIDI can be recompiuted with optimal K and weight_index and became a proxy for SQI1
ds.recalculate_SIDI(K=A['best_k'],weight_index=A['col_best_weight'],overwrite=True)
```


## 8.2) Streamflow Gap Filling
Observed streamflow time series may contain **missing values** due to monitoring gaps or sensor errors.  
The method `gap_filling` of the `Streamflow` class allows you to fill short gaps and preserve continuity in index calculation.

**Concept.** Gaps in the streamflow record are reconstructed **using the precipitation‑based SIDI** that best explains SQI1.  
You must first **optimize the SIDI configuration** against the streamflow with `analyze_correlation`, then pass those settings to `gap_filling`.

```python
# we have previously run A = ds.analyze_correlation(streamflow,plot=True)
# So A holds the results from the optimization method:
print("Best K:", A['best_k'], "Best weight index (SIDI):", A['col_best_weight'])

# 4) Gap filling (SIDI-guided) — uses the precipitation object and the optimal settings
streamflow.gap_filling(ds, K=A['best_k'], weight_index=A['col_best_weight'])

```
---


## 9) Pet and Balance  utilities 

## 9.1) PET analysis (Potential Evapotranspiration)

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


Use PET as an independent climatic driver or combine it with precipitation to build water balance indicators

---

## 9.2) Balance (P–PET) > SPEI

The `Balance` class computes the **monthly climatic water balance** (precipitation (P) minus potential evapotranspirtion (PET).  
This is the standard input for SPEI index, which capture drought as a function of both supply (P) and virtual water demand (PET).

NOTE 1: using gridded data for P and PET with different spatial resolutions is not a problem: data are first imported, aggregated spatially over the basin and then, when single monthly timeseries are ready for P and PET, derive the P-PET  timeseries used to initialize the instance of *Balance*.

NOTE 2: using input data for P and PET covering a different time-span is not a problem: the script select only data on a common timestamp wich is reported in *.m_cal*. 
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



## 10 Temperature class

The `Temperature` class extends the same philosophy used for `Precipitation`, `Pet`, `Balance`, and `Streamflow`, 
but is specialized for temperature datasets.

- **Input handling**  
  - Accepts **NetCDF** temperature datasets (daily or monthly).  

- **Defaults**  
  - Since temperature is generally close to Gaussian and can take both positive and negative values, 
    the default `calculation_method` is the **z-score** (`f_zscore`).  
  - Alternative methods (`f_spei`, `f_spi`) can still be assigned if desired, but are less common.  

- **Interpretation**  
  - Provides standardized indices of temperature variability, which can be used as an independent drought driver 
    (e.g., heat stress episodes) or in conjunction with other classes.  
  - Outputs include the usual **SPI-like set** (in this case, temperature indices), **SIDI**, and **CDN** 
    computed from the 1-month scale.
  - 
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

### Teleindex class

The `Teleindex` class is meant for **large-scale climate drivers** (e.g., Niño3.4, NAO, AO, IOD), provided as a
single time series with a calendar. It reuses the common pipeline (SPI-like multi-scale set, **SIDI**, **CDN**),
but differs from the hydro-meteorological classes in a few key ways.

- **Input handling**
  - Accepts `ts` + `m_cal` directly **or** `data_path` via `import_timeseries(...)`.
  - No shapefile or spatial aggregation: teleconnections are basin-agnostic, exogenous drivers.
  - If **daily** data are detected, values are **averaged to monthly means** (not summed).
    This mirrors `Temperature` (monthly mean), while `Precipitation`, `Pet` and `Balance` are monthly **sums**.

- **Defaults and normalization**
  - Default `calculation_method` is **Pearson III** (`f_spei`) to accommodate signed, potentially skewed indices.
    You can switch to `f_zscore` if the series is close to Gaussian, or keep `f_spei` for robustness.
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
  - The helper method `assign_streamflow_data(...)` updates the internal series and recomputes indices; despite the
    name, it simply assigns a generic teleconnection time series (daily or monthly) and aggregates to monthly if needed.


```python
import numpy as np
import drought_scan as DS
from drought_scan.utils import f_zscore

# Example: 600 months of synthetic precipitation (positive) and a matching calendar
ts = np.random.gamma(shape=2.0, scale=30.0, size=600)          # 

years = np.repeat(np.arange(1975, 2025), 12)[:600]
months = np.tile(np.arange(1, 13), 50)[:600]
m_cal = np.column_stack([months, years])                        # 

tb1 = 1975
tb2 = 2024


index = DS.Teleindex(ts=ts, m_cal = m_cal,start_baseline_year=tb1,
                        end_baseline_year=tb2,calculation_method=f_zscore,
                     index_name='my_index',
                   verbose = False)
```