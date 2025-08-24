# Visualization toolkit
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
ds.plot_scan(weight_index=4)         # Use logarithmically increasing weights for SIDI
ds.plot_scan(xlim=(2000,2010))       # Zoom in on a specific period
ds.plot_scan(reverse_color=True)     # Reverse colormap (useful for SPEI-like indices)
ds.plot_scan(reverse_color=True, xlim=(2000,2012))
ds.plot_scan(saveplot=False)          # Automatically save in working directory
```

Figures can also be saved manually:

```python
from drought_scan.utils.visualization import savefig
savefig('test.png')
```

---

## 3) Identify severe drought events

Use the threshold to detect events:

```python
print("Default threshold:", ds.threshold)
ds.threshold = -1.5  # Custom threshold

tstartid, tendid, duration, deficit = ds.severe_events()
print("Severe events started at:", ds.m_cal[tstartid])
```

### Show multiple events

```python
ds.severe_events(max_events=10, labels=True)
```


---

## 4) Equivalent precipitation for a target SPI

`normal_values()` returns precipitation amounts equivalent to SPI=0.  
Use regression coefficients (`c2r_index`) to convert an SPI value to precipitation.

```python
print("Normal precipitation values:", ds.normal_values())

coeff = ds.c2r_index
spi_value = -1.5
month_index = 3   # March
scale_index = 18  # SPI18

equivalent_precipitation = np.polyval(coeff[scale_index-1, month_index-1, :], spi_value)
print(f"Equivalent precipitation for SPI18=-1.5 in March: {equivalent_precipitation}")
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
# 'delta': cumulative change (slope * window size)

# Example: change up to Nov 2017
date_idx = np.where((ds.m_cal[:,0]==11) & (ds.m_cal[:,1]==2017))[0][0]
delta_unit = R['delta'][date_idx]

std_to_mm = np.mean([np.polyval(coeff[0, m, :], 1) - ds.normal_values()[m] for m in range(12)])
print(f'in Nov 2017 there was a trend over the last {window} months for a total of {delta_unit*std_to_mm} mm gain/lost')
```

### Plot trends

```python
ds.plot_trends()                        # default window
ds.plot_trends(windows=[50, 120])       # custom multiple windows
```
---

## 6) Monthly profiles of input data

Visualize intra-annual cycle of precipitation:

```python
ds.plot_monthly_profile()
ds.plot_monthly_profile(highlight_years=[2017,2018])
```

Cumulative profiles are useful for snow-dominated (nival) regimes or multi-year analysis:

```python
ds.plot_monthly_profile(cumulate=True, two_year=True, highlight_years=[2017,2018])
```

---
