# Common Errors and How to Fix Them

This page lists some typical errors you may encounter when initializing or running **Drought-Scan**, together with their meaning and suggested solutions.  

---

## 1. Using `f_spi` with non-positive data

**Context**:  
The default `calculation_method` for all classes is `f_kde` (non-parametric, since v3.1.0).
However, if you explicitly set `calculation_method=f_spi`, this assumes **non-negative values**
(zero-inflated Gamma: exact zeros are handled as a point mass, negative values are not).

If your time series contains **negative numbers**, the object is rejected at construction:

```text
ValueError: Negative distribution detected (12% of values < 0). f_spi (zero-inflated Gamma) is defined only on positive, right-skewed data. Re-run with calculation_method=f_spei (Pearson III) or calculation_method=f_kde for real-valued variables such as the P-PET balance.
```

**Solution**:  
- Check your data: precipitation and discharge values should normally be positive.  
- If your data can be negative (e.g. anomalies, balances), switch to a different calculation method such as:  
  - `f_spei` (Pearson III)  
  - `f_zscore` (Gaussian)  
  - `f_kde` (non-parametric, the default)

```python
from drought_scan.utils import f_spei
ds = DS.Precipitation(ts=my_ts, m_cal=my_mcal, shape=my_shape, calculation_method=f_spei)
```

---

## 2. Streamflow initialization from CSV/Excel

**Context**:  
When reading streamflow from CSV/Excel files, sometimes the **date column is not parsed correctly** due to hidden characters or unsupported formats.  
You may see:  

```text
UnboundLocalError: cannot access local variable 'date_column' where it is not associated with a value
```

**Solution**:  
- Try saving your file in a clean format (e.g. export CSV again, or convert to Excel `.xlsx`).  
- Ensure that the date column uses one of the **supported formats**:  

```text
YYYY-MM-DD        (e.g., 1960-12-31)
YYYY/MM/DD        (e.g., 1960/12/31)
DD-MM-YYYY        (e.g., 31-12-1960)
DD/MM/YYYY        (e.g., 31/12/1960)
DD/MM/YY          (e.g., 31/12/60)
YYYYMMDD          (e.g., 19601231)
DD MMM YYYY       (e.g., 01 Dec 2023)
MMM DD, YYYY      (e.g., Dec 01, 2023)
YYYY-DOY          (Julian day, e.g., 2023-365)
YYYY.MM.DD        (e.g., 1960.12.31)
DD.MM.YYYY        (e.g., 31.12.1960)
YYYY/MM           (year+month, e.g., 1960/12)
HH:MM:SS          (time, e.g., 23:59:59)
ISO8601           (e.g., 1960-12-31T23:59:59Z)
```

**Example**: if your file has `31-12-1960`, it will be recognized as `DD-MM-YYYY`.  

---
## 3. Inconsistent baseline years or timestamp gaps

**Context**:  
When initializing a class (e.g., `Precipitation`, `Streamflow`, `Balance`), you may encounter:  

```text
ValueError: Start baseline year 1950 not found in `m_cal`.
ValueError: End baseline year 2030 not found in `m_cal`.
```
**Possible causes**:
The baseline years you specified are not fully included in your data.
The original file contains missing or malformed timestamps (e.g., empty cells or missing months/years), which breaks the alignment between ts and m_cal.

**Solution**:
Double-check that your baseline period falls entirely within the data range.
Carefully inspect your input file to ensure that all months/years are present and that there are no missing rows or invalid date entries.
If you find gaps, correct the file (e.g., add the missing dates with NaN values or interpolate) before reloading it.

---
## 4. Too few data points for fitting distributions

**Context**:
Some methods (e.g., `f_spi` with a Gamma distribution) require a sufficiently large dataset to fit the probability distribution.
If you provide only a very short series (say, 10 years → 10 values per month), the fit of the monthly distribution may fail to converge.

**Solution**:
Use more data whenever possible (ideally several decades of monthly values).
If you have a short record and still want to test the system, switch to a simpler method such as `f_zscore`, which does not require fitting a parametric distribution.

```python
from drought_scan.utils import f_zscore
ds = DS.Precipitation(ts=my_ts, m_cal=my_mcal, shape=my_shape, calculation_method=f_zscore)
```

Alternatively, the default `f_kde` (non-parametric) is generally more robust than Gamma on short series, though it still requires a reasonable sample size.

---
## 5. Unknown `calculation_method`

**Context**: the method passed is not one of the registered standardization methods:

```text
ValueError: Unknown calculation_method <function my_method ...>: expected one of f_spi (zero-inflated Gamma), f_spei (Pearson III), f_kde (Gaussian KDE), f_zscore (z-score) ...
```

**Solution**: use a built-in method (optionally wrapped in `functools.partial`), or register
your own with `drought_scan.utils.drought_indices.register_standardization` (see the
[User Guide](user_guide.md) §3).

---
## 6. Methods that need a calibrated SIDI

`plot_covariates`, `ds_balance` and `Streamflow.gap_filling` read the SIDI calibrated against
the streamflow, and raise if the object still carries the default one. Run the correlation
analysis and commit it first:

```python
A = prec.analyze_correlation(streamflow)
prec.set_optimal_SIDI(A['best_k_per_weight'], A['col_best_weight'], overwrite=True)
# or, seasonal:
S = prec.analyze_correlation_seasonal(streamflow, agg='quarter')
prec.set_optimal_SIDI_seasonal(S, agg='quarter', overwrite=True)
```

Only `Precipitation`, `Pet` and `Balance` can be calibrated; `Streamflow`, `Temperature`
and `Teleindex` raise a `TypeError` on the correlation and calibration methods.

---
## 7. Spatial methods without a grid

```text
ValueError: No spatial grid for this Precipitation object: the spatial methods need an object built from a gridded NetCDF file (with a shapefile), not from ts/m_cal.
```

`spatial_sidi`, `spatial_spi` and `plot_spatial` need the gridded data kept at import
(`Precipitation`, `Pet`, `Balance`, `Temperature` built from NetCDF files). A `Balance`
whose P and PET grids have different resolutions cannot be mapped pixel by pixel.
Likewise, `Streamflow.BFI()` needs the daily series and raises if the object was built
from monthly `ts`/`m_cal` instead of `data_path`.

---
## 8. General advice

- Always verify that your **baseline years** are included in the data (otherwise index computation may fail).  
- Ensure that `ts` (time series) and `m_cal` (calendar) arrays have the same length and no gaps in the timestamp.  
- When possible, **visualize the raw data** (before computing indices) to catch unexpected values.
- Use the diagnostic tools in [statistics_tools.md](statistics_tools.md) to verify which distribution best fits your data before choosing a `calculation_method`.