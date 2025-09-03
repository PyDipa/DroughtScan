# Common Errors and How to Fix Them

This page lists some typical errors you may encounter when initializing or running **Drought-Scan**, together with their meaning and suggested solutions.  

---

## 1. Using `f_spi` with non-positive data

**Context**:  
By default, `Precipitation` and `Streamflow` objects use `calculation_method=f_spi`, which assumes **strictly positive values** (Gamma distribution).  

If your time series contains **zeros or negative numbers**, you may get:  

```text
ValueError: The function value at x=nan is NaN; solver cannot continue
```

**Solution**:  
- Check your data: precipitation and discharge values should normally be positive.  
- If your data can be negative (e.g. anomalies, balances), switch to a different calculation method such as:  
  - `f_spei` (Pearson III)  
  - `f_zscore` (Gaussian)  

```python
from drought_scan.utils import f_spei
ds = DS.Precipitation(ts=my_ts, m_cal=my_mcal, calculation_method=f_spei)
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

## 3. General advice

- Always verify that your **baseline years** are included in the data (otherwise index computation may fail).  
- Ensure that `ts` (time series) and `m_cal` (calendar) arrays have the same length.  
- When possible, **visualize the raw data** (before computing indices) to catch unexpected values.  
