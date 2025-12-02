# Statistical Diagnostics & Distribution Fitting Tools

Drought-Scan includes a set of advanced statistical tools designed to help the user 
assess whether a dataset follows a specific probability distribution, and to quantify 
how well parametric transforms (Gamma, Pearson III, Gaussian) represent the empirical data.

These tools are meant to support:
- SPI/SPEI/SQI design and debugging  
- baseline evaluation  
- distribution selection  
- exploratory data analysis  
- quality control of precipitation or streamflow series  
- local tuning of parametric transforms  

The methods introduced here are **generic**, meaning they can be applied to any 1D dataset 
(monthly precipitation, PET, streamflow, groundwater levels, residuals, etc.).

---

## 1. `test_standardization`

### 1.1 Purpose

`test_standardization` performs a **quick diagnostic** to suggest a suitable family of
standardization / transformation for a given dataset, optionally by **groups**.

It combines:

- **Skewness**  
- **Normality test** (D'Agostino & Pearson)  
- A simple **rule-based recommendation**:
  - Gamma
  - Pearson III
  - Gaussian (z-score)
- A **Kolmogorov–Smirnov (KS) test** against the recommended distribution
- A simple *error / goodness* percentage based on the KS statistic

This is useful when you want a **first screening** of the proper standardization
method, before committing to Gamma / Pearson III / Gaussian in your SPI/SQI/SPEI
pipeline.

---

### 1.2 Function signature

```python
result = test_standardization(data, groups=None)
```

#### Parameters

- `data` : array-like  
  1D numeric dataset to be analyzed.

- `groups` : array-like or `None`, optional  
  Optional grouping labels (e.g. months, seasons, regimes).  
  Must have the same length as `data` if provided.  
  If not provided, the analysis is performed on the full dataset.

#### Returns

- If `groups` is `None`  
  A dictionary with keys:

  - `"skewness"`: sample skewness of the dataset  
  - `"normality_p_value"`: p-value from normality test  
  - `"recommendation"`: textual suggestion (Gamma / Pearson III / Gaussian / Unclear)  
  - `"KS_statistic"`: KS test statistic vs selected distribution  
  - `"KS_p_value"`: p-value of the KS test  
  - `"error_percent"`: `100 * KS_statistic`  
  - `"goodness_percent"`: `100 * (1 - KS_statistic)`

- If `groups` is provided  
  A dictionary of dictionaries:

  ```python
  {
      group_value_1: { ... same fields as above ... },
      group_value_2: { ... },
      ...
  }
  ```

Each inner dict contains the same fields as in the ungrouped case, computed
on the subset of `data` belonging to that group.

---

### 1.3 Recommendation logic (summary)

Internally, the function:

1. Computes **skewness** and runs a **normality test**.  
2. Applies simple rules such as:
   - If all data are **positive** and skewness is **strongly positive** → recommend **Gamma**.  
   - If skewness is large (positive or negative) and data are not strictly positive → recommend **Pearson III**.  
   - If the normality test does **not** reject normality → recommend **Gaussian (z-score)**.  
   - Otherwise → mark as **Unclear — further analysis needed**.
3. Runs a **KS test** against the recommended distribution and maps the KS statistic
   to `error_percent` and `goodness_percent`.

This does **not replace** a full statistical analysis, but provides a consistent,
fast and reproducible first check.

---

### 1.4 Example usage

```python
import numpy as np
import drought_scan as DS
from drought_scan.utils.statistics import test_standardization
# load data
shape_path = 'tests/data/bacino_pontelagoscuro.shp'
prec_path  = 'tests/data/LAPrec1871.v1.1.nc'

ds = DS.Precipitation(
    prec_path=prec_path,
    shape_path=shape_path,
    start_baseline_year=1900,
    end_baseline_year=1950,
    basin_name='Po'  # only used for labeling/plots
)

# Example: February precipitation over the basin
month = 2
ii = np.where(ds.m_cal[:,0]==month)
data = ds.ts[ii]

result = test_standardization(data)

print("Skewness:", result["skewness"])
print("Normality p-value:", result["normality_p_value"])
print("Recommended family:", result["recommendation"])
print("KS statistic:", result["KS_statistic"])
print("Goodness (%):", result["goodness_percent"])

#---------------------------------------
# elaboration grouped by season or month
# --------------------------------------
# Analyze each month separately
data = ds.ts
results_by_month = test_standardization(data, groups=ds.m_cal[:,0])

print(results_by_month[1])  # diagnostics for January
print(results_by_month[7])  # diagnostics for July

```

This is particularly useful to check whether your **distributional assumption**
(e.g. Gamma) holds **uniformly across all months**, or whether some months
or seasons are better modeled by another family.

---

## 2. `fit_distribution_stats`

### 2.1 Purpose

`fit_distribution_stats` is a more **general and configurable** diagnostic tool that:

- Fits one of the following distributions to your data:
  - Gamma
  - Pearson III
  - Gaussian (normal)
- Computes:
  - **Skewness**
  - **Normality test** p-value
  - **MLE parameters** of the chosen distribution
  - **KS statistic and p-value**
  - **Log-likelihood**
  - **AIC** (Akaike Information Criterion)
  - Error / goodness percentages from KS statistic
- Optionally repeats this analysis **per group** (e.g. by month, season, regime).

It is the recommended tool when you need a **quantitative comparison** between
distributional assumptions (via KS, log-likelihood and AIC), rather than just a
rule-based suggestion.

---

### 2.2 Function signature

```python
stats_dict = fit_distribution_stats(
    data,
    dist="gamma",
    groups=None,
    shift_for_gamma=True
)
```

#### Parameters

- `data` : array-like  
  Input numeric dataset.

- `dist` : {`"gamma"`, `"pearson3"`, `"gaussian"`}, default `"gamma"`  
  Target distribution used for fitting and KS testing.

- `groups` : array-like or `None`, optional  
  Grouping labels, same length as `data` if provided.  
  If `None`, the analysis is done on the full dataset.  
  If provided, the analysis is repeated independently for each unique
  group value.

- `shift_for_gamma` : bool, default `True`  
  If `True` and `dist == "gamma"`, the data are shifted by `+1` internally
  (i.e. `data_shifted = data + 1`) to avoid issues with zeros or negative values
  when fitting a Gamma distribution.

---

### 2.3 Returns

- If `groups` is `None`  
  A dictionary with keys:

  - `"distribution"`: selected distribution name (`"gamma"`, `"pearson3"`, `"gaussian"`)  
  - `"skewness"`  
  - `"normality_p_value"`  
  - `"params"`: fitted parameters, as a dict. For example:
    - Gamma / Pearson III: `{"shape": ..., "loc": ..., "scale": ..., ...}`
    - Gaussian: `{"mu": ..., "sigma": ...}`
  - `"KS_statistic"`  
  - `"KS_p_value"`  
  - `"error_percent"`: `100 * KS_statistic`  
  - `"goodness_percent"`: `100 * (1 - KS_statistic)`  
  - `"log_likelihood"`: sum of log-PDF values  
  - `"AIC"`: Akaike Information Criterion (`2*k - 2*logL`, with `k` = number of parameters)

- If `groups` is provided  
  A nested dictionary:

  ```python
  {
      group_value_1: {... stats for subset ...},
      group_value_2: {...},
      ...
  }
  ```

Each inner dict has the same structure as in the ungrouped case.

---

### 2.4 Example usage

#### 2.4.1 Fit a Gamma distribution to the precipitation data 

```python
import numpy as np
import drought_scan as DS
from drought_scan.utils.statistics import fit_distribution_stats
 
# load data
shape_path = 'tests/data/bacino_pontelagoscuro.shp'
prec_path  = 'tests/data/LAPrec1871.v1.1.nc'

ds = DS.Precipitation(
    prec_path=prec_path,
    shape_path=shape_path,
    start_baseline_year=1900,
    end_baseline_year=1950,
    basin_name='Po'  # only used for labeling/plots
)

month = 2
ii = np.where(ds.m_cal[:,0]==month)
data = ds.ts[ii]

stats_gamma = fit_distribution_stats(data, dist="gamma")

print("Fitted Gamma parameters:", stats_gamma["params"])
print("KS statistic:", stats_gamma["KS_statistic"])
print("AIC:", stats_gamma["AIC"])

# ---------------------------------------------
# Fit Gamma separately for each calendar month
# ---------------------------------------------
data=ds.ts
months = ds.m_cal[:,0]
# Fit Gamma separately for each calendar month
stats_by_month = fit_distribution_stats(
    data,
    dist="gamma",
    groups=months)
    
```

This is useful to detect whether **some months violate the usual Gamma assumption**
more strongly than others, or to support a **season-dependent choice** of the
standardization family.

---

## 3. Suggested workflow

A typical diagnostic workflow might be:

1. Use `test_standardization` to get a **first recommendation** for the distribution
   family (Gamma / Pearson III / Gaussian), optionally by groups.

2. Use `fit_distribution_stats` to **quantify** how good that choice is:
   - KS statistic and p-value  
   - log-likelihood and AIC  
   - parameter estimates  

3. Optionally compare **multiple distributions** (e.g. Gamma vs Gaussian) using
   **AIC** and **KS**, and decide which one is more appropriate for your basin,
   variable and temporal aggregation.

These tools do not enforce any specific decision, but provide a **transparent,
reproducible and quantitative basis** for choosing how to standardize your
hydroclimatic time series.



