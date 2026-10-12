# Drought Scan
[![PyPI version](https://img.shields.io/pypi/v/droughtscan.svg)](https://pypi.org/project/droughtscan/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17318415.svg)](https://doi.org/10.5281/zenodo.17318415)

## Overview
**Drought Scan** is a Python library implementing a multi-temporal and basin-scale approach for drought analysis. It is designed to provide advanced tools for evaluating drought severity and trends at the river basin scale by integrating meteorological and hydrological data.

The methodology is described in the article:  
*Building a framework for a synoptic overview of drought* ([Read the article](https://www.sciencedirect.com/science/article/pii/S0048969724081063)),
and is continuously developed within the activities of Drought Central ([DroughtCentral](https://droughtcentral.it)).

---

## Key Features
- Calculation of standardized drought indices (e.g., SPI, SQI, SPEI, SPETI, STI) via parametric (zero-inflated Gamma, Pearson III, z-score) or non-parametric (KDE) methods — or your own, registered once.
- Integration of precipitation, streamflow, PET, temperature, and teleconnection data for basin-level analysis.
- Multi-temporal scales for flexibility in drought assessment.
- Calibration of the multi-scale index (SIDI) against streamflow, global or seasonal, with block-bootstrap confidence intervals.
- Benchmarks of the integrated index against linear and parametric rainfall–runoff kernels.
- Spatial extension: gridded SPI-like/SIDI maps at every grid point within the basin.
- Statistical diagnostics for distribution selection and goodness-of-fit evaluation.
- Visualization tools for drought monitoring, deficit quantification, and seasonal analysis.

For examples and usage notes see:
- [User Guide](https://github.com/PyDipa/DroughtScan/blob/main/tests/docs/user_guide.md) — How to initialize and use DroughtScan objects.
- [Visualization Guide](https://github.com/PyDipa/DroughtScan/blob/main/tests/docs/visualization_guide.md) — Plotting methods and customization.
- [Spatial Guide](https://github.com/PyDipa/DroughtScan/blob/main/tests/docs/spatial_guide.md) — Gridded SPI-like/SIDI maps and their native-unit reverse.
- [Statistical Diagnostics](https://github.com/PyDipa/DroughtScan/blob/main/tests/docs/statistics_tools.md) — Distribution fitting and standardization tools.
- [Common Errors](https://github.com/PyDipa/DroughtScan/blob/main/tests/docs/common_errors.md) — Troubleshooting guide.
---
## Installation


### Option 1:
DroughtScan is available on **[PyPI](https://pypi.org/project/droughtscan/)**.
To install the latest stable version:

```bash
pip install droughtscan
```

### Option 2: Clone and install locally
 
Drought Scan can be installed directly from this repository.

Note:
DroughtScan requires Python ≥3.9.
If multiple Python versions are installed (e.g. 3.10 and 3.12), make sure pip and python refer to the same interpreter. You can check it by running

```bash
python --version
pip --version
```

The following instructions will download the package to your working directory (`pwd`). If you wish to download the package to a specific path, first navigate to the desired location with the terminal.

```bash
git clone https://github.com/PyDipa/DroughtScan.git
cd DroughtScan
pip install .
```

### Option 3: Install directly from GitHub (no local clone)
```bash
pip install git+https://github.com/PyDipa/DroughtScan.git
```
To use a specific python interpreter, say for example Python 3.10, for option 2 use:

```bash
python3.10 -m pip install . 
```
for option 3:
```bash
python3.10 -m pip install git+https://github.com/PyDipa/DroughtScan.git
```

Dependencies listed in the repository will be installed automatically in your Python environment during the installation process. 
Refer to the pyproject.toml file for more details about the DroughtScan package.

## What Drought-Scan Does

Drought-Scan provides an **end-to-end framework** for monitoring and analyzing drought conditions at the basin scale.  
It combines **statistical drought indices**, **quantitative analysis**, and **visualization tools** into a single Python package.

### Core Capabilities
- **Data handling**: Organizes meteorological and hydrological time series (precipitation, streamflow, PET, temperature, teleconnections) into a consistent calendar (`m_cal`) and spatial framework (shapefiles).
- **Drought indices**:
  - **SPI (Standardized Precipitation Index)** from 1 to K months (default K=36).
  - **SIDI (Standardized Integrated Drought Index)**: a weighted multi-scale index, standardized to mean 0 and variance 1.
  - **CDN (Cumulative Deviation from Normal)**: integrates long-term memory of anomalies by cumulating the standard index at 1-month scale.
  - **SQI (Standardized Streamflow Index)**: SPI-like indicator based on river discharge.
  - **SPEI / SPETI**: Balance- and PET-based drought indices.
- **Standardization methods**: zero-inflated Gamma (`f_spi`), Pearson III (`f_spei`), Gaussian KDE (`f_kde`, default), z-score (`f_zscore`); custom methods via `register_standardization`. Exact forward and inverse transforms (`native_to_spi` / `spi_to_native`).
- **Calibration against streamflow**: the K and weighting scheme of the SIDI that best explain SQI₁ (also per season and with a time lag), with block-bootstrap CIs; benchmarks (OLS lag kernels, Nash-type Gamma kernel, IHACRES routing). R² is the squared Pearson correlation throughout.
- **Visualization**: the three "pillars" of drought monitoring (heatmap, SIDI, CDN), plus deficit/surplus bars, monthly profiles, spatial maps, and precipitation–streamflow diagnostics.
- **Spatial extension**: SPI-like and SIDI maps (and the native-unit deficit/surplus) at every grid point within the basin, for precipitation, PET, P−PET and temperature, with parallel processing.

---

## Available Classes

| Class | Input | Index | Use case |
|-------|-------|-------|----------|
| `Precipitation` | NetCDF or arrays | SPI | Meteorological drought |
| `Streamflow` | CSV/Excel or arrays | SQI | Hydrological drought |
| `Pet` | NetCDF or arrays | SPETI | Evapotranspiration analysis |
| `Balance` | NetCDF (P + PET) | SPEI | Water balance drought |
| `Temperature` | NetCDF or arrays | STI | Temperature variability |
| `Teleindex` | CSV/Excel or arrays | (custom) | Large-scale climate drivers |

All classes share the same base (`BaseDroughtAnalysis`) and produce the same core outputs: `ts`, `m_cal`, `spi_like_set`, `SIDI`, `CDN`. Some analyses only apply to some classes — see the capability table in the [User Guide](https://github.com/PyDipa/DroughtScan/blob/main/tests/docs/user_guide.md) §8.

---

## Core Attributes
- **`ts`**: monthly time series (precipitation, streamflow, etc.).  
- **`m_cal`**: calendar aligned with the time series, shape (N, 2) with [month, year].  
- **`spi_like_set`**: set of SPI-like indices at scales 1–K, shape (K, N).  
- **`SIDI`**: Standardized Integrated Drought Index, shape (N, 5) for 5 weighting schemes.  
- **`CDN`**: Cumulative Deviation from Normal, shape (N,).  
- **`basin_name`**: name of the basin under analysis.  
- **`index_name`**: name of the standardized index (e.g., 'SPI', 'SQI').  
- **`shape`**: basin geometry (GeoDataFrame).  
- **`area_kmq`**: area of the basin in km².  
- **`K`**: maximum SPI scale (default 36).  
- **`threshold`**: severity threshold (default −1; +1 for `Pet` and `Temperature`).
- **`weight_index`**: default SIDI column before any calibration (default 2).
- **`calibration_mode`**: `None`, `'global'` or `'seasonal'` — which calibration `SIDI` carries.
- **`Pgrid` / `PETgrid` / `Tgrid` / `prec_grid`, `pet_grid`**: input gridded data within the basin (when initialized from NetCDF).

---

## Main Methods

### All classes (BaseDroughtAnalysis)
- **`plot_scan()`**: full overview (heatmap, SIDI, CDN).
- **`plot_monthly_profile()`**: climatology plot (monthly profile) of the input variable.
- **`plot_boundary()`**: basin shapefile on an equal-area projection.
- **`normal_values()`**: "normal" values of the climatology (index = 0, exact inverse).
- **`native_to_spi()` / `spi_to_native()`**: exact conversion between native units and the index.
- **`find_trends()`**: deficit/surplus over moving windows from the reverse SPI (the numbers behind `plot_trends`).
- **`plot_trends()`**: CDN with deficit/surplus bars for multiple window sizes.
- **`deficit_from_spi()` / `volume_anomaly_rolling()`**: statistical and direct physical deficit over a window.
- **`plot_spi_fit()`**: plot the fitted relationship between index values and the raw variable.
- **`recalculate_SIDI()`**: recompute SIDI with a custom K (one value, or one per weighting scheme).
- **`export_scan_plot_csv()`**: export data needed to replicate `plot_scan` in external tools.
- **`severe_events_old()`**: obsolete, kept only to reproduce Di Paola et al. (2025).

### Precipitation, Pet, Balance (drivers of a streamflow)
- **`analyze_correlation(streamflow)`**: R² of SIDI vs SQI₁ for every (K, weighting scheme); optional block bootstrap.
- **`analyze_correlation_seasonal(streamflow)`** / **`analyze_correlation_seasonal_rolling(streamflow)`**: per season, and on moving windows of years.
- **`analyze_correlation_lagged(streamflow, lag)`**: the same with the SIDI leading SQI₁ by `lag` months.
- **`set_optimal_SIDI()`** / **`set_optimal_SIDI_seasonal()`**: commit a global or seasonal calibration (mutually exclusive).
- **`plot_covariates(streamflow)`**: calibrated SIDI alongside SQI₁.
- **`spi_sqi_corr(streamflow)`** / **`spi_sqi_corr_lagged(streamflow)`**: month-wise SPIₖ–SQI₁ R² matrix, without or with a time lag.
- **Benchmarks**: `benchmark_convolution()`, `Dspi_free()`, `benchmark_nash()`, `benchmark_ihacres()`, `analyze_convolution_irf()`.

### Gridded objects (Precipitation, Pet, Balance, Temperature from NetCDF)
- **`spatial_sidi()`**: gridded SIDI at a target date (follows the committed calibration).
- **`spatial_spi()`**: gridded SPI-like index and its native-unit deficit/surplus.
- **`plot_spatial()`**: maps of SIDI, the SPI-like index, or its reverse.

### Precipitation only
- **`ds_balance(streamflow)`** / **`ds_balance_all_schemes(streamflow)`**: monthly water balance of the basin — the part of the streamflow anomaly the precipitation accounts for, and the part it does not.
- **`import_forecast()`**: import forecast ensembles.

### Streamflow only
- **`gap_filling(precipitation)`**: reconstruct missing streamflow from a calibrated SIDI.
- **`plot_annual_ts(DSO)`**: compare annual streamflow with an external driver.
- **`BFI()`**: Baseflow Index (requires the daily file, `data_path`).

> **Note**: internal methods (prefixed with `_`) are used for calculations and should not be called directly.  
> For detailed reference and examples, see the [User Guide](https://github.com/PyDipa/DroughtScan/blob/main/tests/docs/user_guide.md) and [Visualization Guide](https://github.com/PyDipa/DroughtScan/blob/main/tests/docs/visualization_guide.md).

---

## License

DroughtScan is distributed under the [GNU GPL v3.0](LICENSE) for academic and
non-commercial research use.

For any commercial or revenue-generating application, a separate commercial
license is required. A separate commercial license can be arranged **outside** this repository 
and **does not alter** the open-source terms of the GPL for this codebase.  
For inquiries: arianna.dipaola@cnr.it

## Authors

- **Arianna Di Paola** CNR-IBE, Italy — Lead developer and maintainer; arianna.dipaola@cnr.it
- **Massimiliano Pasqui** CNR-IBE, Italy — Feedback, scientific guidance, methodological validation and review.
- **Ramona Magno** CNR-IBE, Italy — Feedback, scientific guidance, methodological validation and review.
- **Leandro Rocchi** CNR-IBE, Italy — Technical support.
