# Example Usage

This section shows how to initialize a **Drought-Scan** analysis object (DSO), with practical notes on the most important options and how they change the behavior compared to defaults.

## 1) Minimal setup (from files)

To run Drought-Scan you need at least:
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

print("Aggregated precipitation (ts):", ds.ts.shape)
print("Monthly calendar (m_cal):", ds.m_cal[:5])
print("SPI multi-scale set:", ds.spi_like_set.shape)
print("SIDI (by 5 weighting scheme):", ds.SIDI.shape)
print("CDN (cumulative SPI1 from the starting baseline year):", ds.CDN.shape)
```

What happens here:
- The library reads the NetCDF precipitation, clips/aggregates it over the basin shapefile, builds a monthly calendar (`m_cal`), and computes SPI (1–K), SIDI, and CDN over the **baseline** `start_baseline_year:end_baseline_year`.

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

ds = DS.Precipitation(
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

---


## 7) Streamflow (SQI), Pet and Balance (SPEI) classes
For drought analysis based on other standardiezed indices like  SQI, SPEI or SPETI you can use the corresponding  `Streamflow`, `Balance ` and `Pet`  classes. They shares the same initialization philosophy; provide `ts/m_cal` **or** file paths, set `K`, `baseline`, `calculation_method` (`f_spi` on positive flows as Streamflow, f_spei for the P-PET balance and f_zscore for PET), and optionally a `threshold` aligned with your risk definition. Outputs include **SQI/SPEI/SPETI** (SPI-like arrays), **SIDI**, and **CDN** computed by using the 1-month scale of the obtained index.

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

# Note:  in Streamflow it is possible to assign or update the time series (ts) and calendar (m_cal) if provided by user.  SPI and SIDI are recomputed acordingly. 

# EXEMPLE
# ts = np.random.gamma(shape=2.0, scale=30.0, size=600)          # (T,)
# years = np.repeat(np.arange(1975, 2025), 12)[:600]
# months = np.tile(np.arange(1, 13), 50)[:600]
# m_cal = np.column_stack([months, years])  
# streamflow.assign_streamflow_data(ts=ts,m_cal=m_cal)
```


## 8) Streamflow (SQI) — symmetry with Precipitation

Precipitation and streamflow are intrinsically linked as part of the hydrological cycle and represent key indicators for understanding drought. A reduction in precipitation can directly lead to decreased river discharge, reduced groundwater recharge, and lower reservoir storage. This extends the impacts of drought on water availability over time, often with a delayed effect. Drought-Scan explicitly analyzes this relationship through the correlation between the Standardized Drought Integration Index (SIDI), derived from SPI, and the one-month Streamflow Drought Index (SQI1)

## 8.1) Reproducibility tips

- Fix your **baseline** and stick to it across runs for fair comparisons between Precipitation and Streamflow.  
- Streamflow data format acceped are CSV or EXCELL
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
# So A holds the results from the optimitation method:
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

## 3) Balance (P–PET) > SPEI

The `Balance` class computes the **monthly climatic water balance** (precipitation minus PET).  
This is the standard input for SPEI index, which capture drought as a function of both supply (P) and virtual water demand (PET).

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

print("Balance time series (P–PET):", balance.ts.shape)
print("SPEI-like indices (1–K months):", balance.spi_like_set.shape)

```

This setup is particularly useful in climate change studies, where increasing PET may exacerbate drought even under stable precipitation.





---

