# Statistical Diagnostics & Distribution Fitting Tools

Drought-Scan includes a set of statistical tools for assessing distributional assumptions,
quantifying goodness-of-fit, and standardizing hydroclimatic time series.

These tools support:
- SPI / SPEI / SQI design and debugging
- baseline evaluation and distribution selection
- exploratory data analysis and quality control
- local tuning of parametric transforms

All functions are **generic**: they accept any 1-D dataset (monthly precipitation,
PET, streamflow, groundwater levels, residuals, etc.).

---

## Overview of the pipeline

The statistics module is organized as a **three-stage pipeline**:

1. **Diagnose** → `test_standardization()` fits all four distribution families
   (Gaussian, Gamma, Pearson III, KDE) and recommends the best one.
2. **Zoom** → `fit_distribution_stats()` fits a *single* distribution of your
   choice and returns detailed goodness-of-fit statistics.
3. **Act** → `standardize_data()` transforms the raw data into standard-normal
   scores (z-scores) using the Probability Integral Transform (PIT).

Additional utilities:
- `plot_cdf_comparison()` — standalone visual diagnostic (empirical vs theoretical CDF/PDF).
- `find_overlap()` — find the common temporal window between two calendar arrays.

---

## 1. `test_standardization`

### 1.1 Purpose

Fits **all four** distribution families to the input data and selects the best one
based on objective criteria. This replaces any rule-based heuristic: the recommendation
comes directly from the data.

The four families always evaluated are:
- **Gaussian** (normal)
- **Gamma**
- **Pearson type III**
- **Gaussian KDE** (non-parametric, Silverman bandwidth)

**Primary selection criterion**: lowest Kolmogorov–Smirnov statistic *D* (scale-free,
valid for all four families including KDE).

**Secondary criterion**: lowest AIC (Akaike Information Criterion), computed for the
three parametric families only (KDE is excluded because no canonical number of parameters
is defined).

---

### 1.2 Function signature

```python
from drought_scan.utils.statistics import test_standardization

result = test_standardization(
    data,
    groups=None,
    shift_for_gamma=True,
    plot=True,
    n_bootstrap=0,
    seed=None
)
```

#### Parameters

- `data` : array-like
  1-D numeric dataset. Temporal aggregation (e.g., SPI-3/6/12) must be
  performed **before** calling this function — results reflect the input scale as-is.

- `groups` : array-like or `None`, optional
  Grouping labels (e.g., months, seasons). Same length as `data`.
  If provided, the analysis is run independently per group.

- `shift_for_gamma` : bool, default `True`
  Add +1 to data before Gamma fitting to handle zeros.

- `plot` : bool, default `True`
  Generate empirical vs theoretical CDF/PDF comparison figures
  (all four families overlaid, best highlighted).

- `n_bootstrap` : int, default `0`
  If > 0, compute Lilliefors-corrected KS p-values via parametric bootstrap.
  Recommended ≥ 999 for stable estimates.

- `seed` : int or `None`
  Random seed for bootstrap reproducibility.

#### Returns

- If `groups` is `None` — a dictionary:

  | Key                  | Type   | Description                                   |
  |----------------------|--------|-----------------------------------------------|
  | `"skewness"`         | float  | Sample skewness                               |
  | `"normality_p_value"`| float  | p-value from D'Agostino & Pearson test        |
  | `"fits"`             | dict   | `{family: fit_dict}` for all four families    |
  | `"best_by_KS"`       | str    | Family with lowest KS statistic               |
  | `"best_by_AIC"`      | str    | Family with lowest AIC (parametric only)      |
  | `"recommendation"`   | str    | Primary recommendation (= `best_by_KS`)      |

  Each entry in `"fits"` is a dictionary with keys:
  `distribution`, `params`, `KS_statistic`, `KS_p_value`,
  `log_likelihood`, `AIC`, `k_params`, `error_percent`, `goodness_percent`.
  If `n_bootstrap > 0`, also `KS_p_value_bootstrap`.

- If `groups` is provided — a nested dictionary: `{group_label: <same dict as above>}`.

> **Note on KS p-values.**
> When parameters are estimated from the same sample (MLE), classical KS p-values
> are anti-conservative (Lilliefors 1967). The KS statistic *D* itself remains a valid
> distance metric. For formal inference, use `n_bootstrap ≥ 999`.

---

### 1.3 Example usage

```python
import numpy as np
import drought_scan as DS
from drought_scan.utils.statistics import test_standardization

# --- Load data ---
shape_path = 'tests/data/bacino_pontelagoscuro.shp'
prec_path  = 'tests/data/LAPrec1871.v1.1.nc'

ds = DS.Precipitation(
    prec_path=prec_path,
    shape_path=shape_path,
    start_baseline_year=1900,
    end_baseline_year=1950,
    basin_name='Po'
)

# --- Single-month analysis (e.g., February) ---
month = 2
ii = np.where(ds.m_cal[:, 0] == month)
data = ds.ts[ii]

result = test_standardization(data)

print("Recommended family:", result["recommendation"])
print("Best by AIC:",        result["best_by_AIC"])

# Access KS statistic for each family
for family, fit in result["fits"].items():
    if fit is not None:
        print(f"  {family}: D={fit['KS_statistic']:.4f}, "
              f"goodness={fit['goodness_percent']:.1f}%")

# --- Grouped analysis (by month) ---
results_by_month = test_standardization(ds.ts, groups=ds.m_cal[:, 0])

# Check recommendation for January and July
print("January:", results_by_month[1]["recommendation"])
print("July:",    results_by_month[7]["recommendation"])
```

### 1.4 Interpreting the results: KS vs AIC

It is normal for `best_by_KS` and `best_by_AIC` to disagree. They measure
different things:

- **KS statistic *D*** measures the maximum pointwise distance between the
  empirical and theoretical CDF. It is sensitive to the *worst local mismatch*
  — a distribution can fit 95% of the data perfectly but score poorly if one
  tail diverges.
- **AIC** evaluates the overall likelihood of the data under the fitted model,
  penalized by the number of parameters. It rewards *global* fit and parsimony.

A practical example: if Gamma has D = 0.08 and Pearson III has D = 0.21 but
lower AIC, it means Pearson III explains the bulk of the data marginally better
in a likelihood sense, but has a much worse worst-case mismatch in the tails.

**Rule of thumb for drought indices**: prefer the KS-based recommendation
(`best_by_KS`). SPI/SPEI are built on the CDF transform, so the CDF shape
matters more than overall likelihood. A distribution that deviates in the
tails will distort exactly the extreme events you are trying to detect.

The `goodness_percent` field (= 100 × (1 − D)) is a quick readable metric:
values above 90% indicate a strong fit; below 80% suggest that the family
may not be appropriate for that dataset or month.

---

## 2. `fit_distribution_stats`

### 2.1 Purpose

Fits a **single specified distribution** and returns detailed goodness-of-fit
statistics. Use this after `test_standardization` to zoom in on a specific family,
or to compare two families head-to-head via KS and AIC.

---

### 2.2 Function signature

```python
from drought_scan.utils.statistics import fit_distribution_stats

stats = fit_distribution_stats(
    data,
    dist="gamma",
    groups=None,
    shift_for_gamma=True,
    plot=True,
    n_bootstrap=0,
    seed=None
)
```

#### Parameters

- `data` : array-like — input dataset.
- `dist` : `{"gaussian", "gamma", "pearson3", "kde"}`, default `"gamma"` — target distribution.
- `groups` : array-like or `None` — grouping labels.
- `shift_for_gamma` : bool, default `True` — add +1 before Gamma fitting.
- `plot` : bool, default `True` — show empirical vs theoretical CDF/PDF for the chosen distribution.
- `n_bootstrap` : int, default `0` — Lilliefors-corrected p-value (≥ 999 recommended).
- `seed` : int or `None` — random seed.

#### Returns

- If `groups` is `None` — a dictionary:

  | Key                      | Type   | Description                                     |
  |--------------------------|--------|-------------------------------------------------|
  | `"distribution"`         | str    | Distribution name                               |
  | `"skewness"`             | float  | Sample skewness                                 |
  | `"normality_p_value"`    | float  | p-value from normality test                     |
  | `"params"`               | dict   | Fitted parameters (see below)                   |
  | `"KS_statistic"`         | float  | KS test statistic                               |
  | `"KS_p_value"`           | float  | Classical KS p-value                            |
  | `"KS_p_value_bootstrap"` | float  | Bootstrap-corrected p-value (if `n_bootstrap>0`)|
  | `"log_likelihood"`       | float  | Sum of log-PDF values                           |
  | `"AIC"`                  | float  | Akaike Information Criterion (`NaN` for KDE)    |
  | `"k_params"`             | int    | Number of estimated parameters (`None` for KDE) |
  | `"error_percent"`        | float  | `100 × D`                                       |
  | `"goodness_percent"`     | float  | `100 × (1 − D)`                                |

  **`params` structure by distribution:**
  - Gamma: `{"shape", "loc", "scale", "shift_applied"}`
  - Pearson III: `{"shape", "loc", "scale"}`
  - Gaussian: `{"mu", "sigma"}`
  - KDE: `{"bw_method", "bw_factor", "n_fit", "_kde_cdf"}`

- If `groups` is provided — `{group_label: <same dict>}`.

---

### 2.3 Example usage

```python
from drought_scan.utils.statistics import fit_distribution_stats

# Fit Gamma to February precipitation
month = 2
ii = np.where(ds.m_cal[:, 0] == month)
data = ds.ts[ii]

stats_gamma = fit_distribution_stats(data, dist="gamma")
print("KS statistic:", stats_gamma["KS_statistic"])
print("AIC:", stats_gamma["AIC"])
print("Fitted params:", stats_gamma["params"])

# Compare Gamma vs Pearson III via AIC
stats_p3 = fit_distribution_stats(data, dist="pearson3")
print(f"Gamma AIC: {stats_gamma['AIC']:.1f}  vs  Pearson III AIC: {stats_p3['AIC']:.1f}")

# Fit Gamma separately for each calendar month
stats_by_month = fit_distribution_stats(
    ds.ts, dist="gamma", groups=ds.m_cal[:, 0]
)
```

---

## 3. `standardize_data`

### 3.1 Purpose and when to use it

Transforms raw data into **standard-normal scores** (z-scores) using the
Probability Integral Transform (PIT). This is the same logic underlying
SPI (McKee 1993) and SPEI (Vicente-Serrano 2010):

1. Fit the recommended distribution to the data.
2. Map each observation *x* → *p* = F(*x*)  (CDF value ∈ (0, 1)).
3. Map *p* → *z* = Φ⁻¹(*p*)  (standard-normal quantile).

### When to use `standardize_data` vs re-initializing a DroughtScan object

There are **two distinct workflows** depending on your goal:

**Workflow A — Within DroughtScan (recommended for drought indices).**
The distribution used for SPI/SPEI/SQI is controlled by the `calculation_method`
parameter at initialization. The diagnostic pipeline is:

```python
# 1. Initialize with default method (f_spi → Gamma)
ds = DS.Precipitation(prec_path=..., shape_path=..., ...)

# 2. Check if Gamma is appropriate
from drought_scan.utils.statistics import test_standardization
analysis = test_standardization(ds.ts, groups=ds.m_cal[:, 0])

# 3. If another family fits better, re-initialize with the appropriate method
from drought_scan.utils import f_spei   # Pearson III
ds = DS.Precipitation(prec_path=..., shape_path=..., calculation_method=f_spei, ...)
```

Available calculation methods: `f_spi` (Gamma), `f_spei` (Pearson III),
`f_zscore` (Gaussian), `f_kde` (KDE).

**Workflow B — Standalone (for arbitrary data).**
Use `standardize_data` when you need to standardize a time series that is
*not* managed by a DroughtScan object (e.g., groundwater levels, soil moisture,
temperature anomalies, or any external dataset):

```python
from drought_scan.utils.statistics import test_standardization, standardize_data

analysis = test_standardization(my_data, groups=my_months)
result   = standardize_data(my_data, analysis, groups=my_months)
z_scores = result["z_scores"]
```

---

### 3.2 Function signature

```python
from drought_scan.utils.statistics import standardize_data

result = standardize_data(
    data,
    analysis_result,
    groups=None,
    plot=True
)
```

#### Parameters

- `data` : array-like — original (non-standardized) values.
- `analysis_result` : dict — output of `test_standardization()`.
  If `groups` was used, it must be the corresponding nested dict.
- `groups` : array-like or `None` — same grouping vector used in
  `test_standardization()`. Each group is standardized independently.
- `plot` : bool, default `True` — diagnostic scatter (original vs z-scores)
  and histogram of z-scores vs N(0,1) reference.

#### Returns

A dictionary:

| Key                | Type       | Description                                    |
|--------------------|------------|------------------------------------------------|
| `"z_scores"`       | ndarray    | Standardized values (NaN preserved)            |
| `"distribution"`   | str / dict | Distribution used (str if ungrouped, dict if grouped) |
| `"params"`         | dict       | Fitted parameters (or `{group: params}`)       |
| `"cdf_values"`     | ndarray    | Intermediate CDF values F(x)                   |
| `"recommendation"` | str / dict | Distribution used (same as `"distribution"`)   |

---

### 3.3 Example — Workflow A (DroughtScan re-initialization)

```python
import numpy as np
import drought_scan as DS
from drought_scan.utils.statistics import test_standardization, fit_distribution_stats
from drought_scan.utils import f_spi, f_spei, f_zscore, f_kde

# 1. Initialize with default (Gamma)
ds = DS.Precipitation(
    prec_path='tests/data/LAPrec1871.v1.1.nc',
    shape_path='tests/data/bacino_pontelagoscuro.shp',
    start_baseline_year=1900,
    end_baseline_year=1950,
    basin_name='Po'
)

# 2. DIAGNOSE — which distribution fits best, month by month?
analysis = test_standardization(ds.ts, groups=ds.m_cal[:, 0])

for m in range(1, 13):
    rec = analysis[m]["recommendation"]
    D   = analysis[m]["fits"][rec]["KS_statistic"]
    print(f"  Month {m:2d}: {rec:10s}  (D = {D:.4f})")

# 3. ZOOM — inspect February more closely
stats_feb = fit_distribution_stats(ds.ts[ds.m_cal[:, 0] == 2], dist="gamma")

# 4. ACT — if the diagnosis suggests a different family, re-initialize
#    Example: diagnosis recommends Pearson III overall → use f_spei
ds = DS.Precipitation(
    prec_path='tests/data/LAPrec1871.v1.1.nc',
    shape_path='tests/data/bacino_pontelagoscuro.shp',
    start_baseline_year=1900,
    end_baseline_year=1950,
    basin_name='Po',
    calculation_method=f_spei   # Pearson III
)
```

### 3.4 Example — Workflow B (standalone standardization)

```python
import numpy as np
from drought_scan.utils.statistics import test_standardization, standardize_data

# Any 1-D dataset not managed by DroughtScan
my_data   = ...   # e.g., groundwater levels, soil moisture
my_months = ...   # grouping vector (e.g., month labels 1–12)

# Diagnose
analysis = test_standardization(my_data, groups=my_months, plot=False)

# Standardize
result = standardize_data(my_data, analysis, groups=my_months)

z_scores = result["z_scores"]
print("Mean z:", np.nanmean(z_scores))    # ≈ 0
print("Std z:",  np.nanstd(z_scores))     # ≈ 1
```

---

## 4. `plot_cdf_comparison`

### 4.1 Purpose

Standalone visual diagnostic: overlays the **empirical CDF** and **empirical PDF**
(histogram) with the theoretical curves for a single specified distribution.
Useful for inspecting where the KS error is concentrated (lower tail, central body,
upper tail).

---

### 4.2 Function signature

```python
from drought_scan.utils.statistics import plot_cdf_comparison

plot_cdf_comparison(
    data,
    dist="gamma",
    params=None,
    shift_for_gamma=True,
    unit=None
)
```

#### Parameters

- `data` : array-like — input dataset.
- `dist` : `{"gamma", "pearson3", "gaussian"}` — distribution family.
- `params` : dict or `None` — pre-computed parameters from `fit_distribution_stats()`.
  If `None`, parameters are fitted internally.
- `shift_for_gamma` : bool, default `True` — consistent with `fit_distribution_stats`.
- `unit` : str or `None` — label for the x-axis (e.g., `"mm"`, `"m³/s"`).
  Defaults to `"value"`.

---

### 4.3 Example usage

```python
from drought_scan.utils.statistics import fit_distribution_stats, plot_cdf_comparison

# Fit and inspect
stats_gamma = fit_distribution_stats(data, dist="gamma", plot=False)
plot_cdf_comparison(data, dist="gamma", params=stats_gamma["params"], unit="mm")

# Quick check without pre-fitting (params estimated internally)
plot_cdf_comparison(data, dist="pearson3", unit="mm")
```

---

## 5. Suggested workflow

A typical diagnostic workflow:

1. **Initialize** a DroughtScan object with the default `calculation_method`.

2. **`test_standardization`** → get a data-driven recommendation for the best
   distribution family, optionally by groups (months, seasons).

3. **`fit_distribution_stats`** → quantify the fit for a specific family
   (KS, AIC, log-likelihood, parameter estimates). Compare multiple families
   head-to-head if needed.

4. **`plot_cdf_comparison`** → visual inspection of where the theoretical
   distribution deviates from the empirical one.

5. **Act**:
   - *Within DroughtScan* → re-initialize the object with the appropriate
     `calculation_method` (`f_spi`, `f_spei`, `f_zscore`, `f_kde`).
   - *Standalone data* → use `standardize_data()` to apply the PIT transform.

These tools do not enforce any specific decision, but provide a **transparent,
reproducible, and quantitative basis** for choosing how to standardize your
hydroclimatic time series.

> **Note on trend analysis.**
> Trend detection on the CDN or on any other time series is available through
> the DroughtScan methods `find_trends()` and `plot_trends()` (see the
> [Visualization Guide](visualization_guide.md) and the
> [User Guide](user_guide.md) for details and examples).

---

## 6. Utility functions

### `find_overlap`

```python
from drought_scan.utils import find_overlap

idx1, idx2 = find_overlap(m_cal1, m_cal2)
```

Finds the temporal overlap between two calendar arrays `(N, 2)` with columns
`[month, year]`. Returns index arrays for the overlapping period in each calendar.
Raises `ValueError` if no overlap exists.

Typical use case: when you have a `Precipitation` and a `Streamflow` object
initialized from different datasets with different temporal coverages, and you
need to extract the common period for joint analysis:

```python
import drought_scan as DS
from drought_scan.utils import find_overlap

prec = DS.Precipitation(prec_path=..., shape_path=..., ...)
flow = DS.Streamflow(start_baseline_year=..., end_baseline_year=..., ...)

idx_p, idx_f = find_overlap(prec.m_cal, flow.m_cal)

# Common-period time series
prec_common = prec.ts[idx_p]
flow_common = flow.ts[idx_f]
```