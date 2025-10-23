# Drought Scan
[![PyPI version](https://img.shields.io/pypi/v/droughtscan.svg)](https://pypi.org/project/droughtscan/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17318415.svg)](https://doi.org/10.5281/zenodo.17318415)

## Overview
**Drought Scan** is a Python library implementing a multi-temporal and basin-scale approach for drought analysis. It is designed to provide advanced tools for evaluating drought severity and trends at the river basin scale by integrating meteorological and hydrological data.

The methodology is described in the article:  
*Building a framework for a synoptic overview of drought* ([Read the article](https://www.sciencedirect.com/science/article/pii/S0048969724081063)).
and is continuously developed within the activities of Drought Central ([DroughtCentral](https://droughtcentral.it)).

---

## Key Features
- Calculation of standardized drought indices (e.g., SPI, SQI, SPEI,etc).
- Integration of precipitation and streamflow data for basin-level analysis.
- Multi-temporal scales for flexibility in drought assessment.
- Possibility of generating synthetic graphs and seasonal trend analysis.


for examples and usage notes see:
- [User Guide](https://github.com/PyDipa/DroughtScan/blob/main/tests/docs/user_guide.md) → Demonstrates how to initialize a Drought-Scan Object
- [Visualization Guide](https://github.com/PyDipa/DroughtScan/blob/main/tests/docs/visualization_guide.md) → Demonstrates how to use some visualization methods

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
To use a specific  python interpreter for option1, say for example Python 3.10, use: 

```bash
python3.10 -m pip install . 
```
for option 2:
```bash
python3.10 -m pip install git+https://github.com/PyDipa/DroughtScan.git
```

Dependencies listed in the repository will be installed automatically in your Python environment during the installation process. 
Refer to the pyproject.toml file for more details about the DroughtScan package.

## What Drought-Scan Does

Drought-Scan provides an **end-to-end framework** for monitoring and analyzing drought conditions at the basin scale.  
It combines **statistical drought indices**, **quantitative analysis**  and **visualization tools**  into a single Python package.

### Core Capabilities
- **Data handling**: Organizes meteorological and hydrological time series (precipitation, streamflow, external predictors) into a consistent calendar (`m_cal`) and spatial framework (shapefiles of provinces/basins).
- **Drought indices**:
  - **SPI (Standardized Precipitation Index)** from 1 to K months (default K=36).
  - **SIDI (Standardized Integrated Drought Index)**: a weighted multi-scale index, standardized to mean 0 and variance 1.
  - **CDN (Cumulative Deviation from Normal)**: integrates long-term memory of anomalies by cumulating the standard index at 1-month scale.
  - **SQI (Standardized Streamflow Index)**: SPI-like indicator based on river discharge.
- **Visualization**: Provides the three “pillars” of drought monitoring:
  1. Heatmap of SPI(SQI/SPEI-like) 1–K set.
  2. SIDI as a compact synthesis across scales.
  3. CDN as a long-memory diagnostic.
- **precipitation to streamflow analysis**: Allows joint analysis of precipitation- and streamflow-based indices (e.g., SIDI vs SQI) to measure the strength and the responding time of the hydrographic basin to drought events. 

## The `DroughtScan` Object

When you initialize a `DroughtScan` object, it stores both the **input data** and the **derived drought indicators**.  
It acts as the main container of the framework, holding attributes and methods for analysis, visualization, and forecasting.

### Core Attributes
- **`ts`**: monthly precipitation (or streamflow) time series.  
- **`m_cal`**: calendar aligned with the time series.  
- **`spi_like_set`**: set of SPI1–K series (default K=36).  
- **`SIDI`**: Standardized Integrated Drought Index (weighted ensemble of SPI1–K).  
- **`CDN`**: Cumulative Deviation from Normal (cumulative sum of SPI1).  
- **`basin_name`**: name of the basin under analysis.  
- **`index_name`** name of the spi-like standardized index (default = 'SPI')
- **`shape`**: basin geometry.  
- **`area_kmq`**: area of the basin.  
- **`K`**: maximum SPI scale (default 36).  
- **`threshold`**: default threshold for severe drought (−1).
- **`Pgrid`**: input gridded data within the basin.


### Main Methods
- **`plot_scan()`**: full DS overview (heatmap, SIDI, CDN).  
- **`plot_monthly_profile()`**:climatology plot (monthly profile) of the input variable
- **`normal_values()`**: Compute the "normal" values of the climatology using the inverse function of the SPI-like index.
- **`find_trends()`** Analyze trends in the CDN using rolling windows and linear regression (without any plot)
- **`plot_trends()`**: search, quantify, and plot trends in specific moving windows of the CDN curve
- **`severe_events()`**: list and plot severe drought events, ordered by magnitude or duration
- **`plot_spi_fit()`** plot the fitted relationship between the SPI values and the raw variable
- **`recalculate_SIDI()`**: recompute SIDI with custom subset (K) of spi-like set ranging from 1 to K  


ONLY FOR PRECIPITATION WITH AVAILABLE STREAMFLOW DATA:
- **`analyze_correlation()`**: find the combination of month-scales and weights that maximize the correlaiotn between SIDI and SQI1
- **`set_optimal_SIDI()`**: recompute SIDI with the optimal subset of the spi-like set as provied by `analyze_correlation().
- **`plot_covariates()`**: plot the time series of the covariate: optimal_SIDI along with the target variable (generaly SQI1)

ONLY FOR STREAMFLOW:

-**`gap_filling()`**:Reconstruct monthly streamflow gaps thanks to the best correlation with precipitation data found out in `analyze_correlation()`.

-**`plot_annual_ts(DSO)`**: plot annual timeseries along with annual time series of selected Drought Scan Object (DSO) among Precipitation, Pet, and Balance

> **Note**: internal methods (prefixed with `_`) are used for calculations and should not be called directly by the user.  
> For a detailed reference and usage examples, see the for examples and usage notes see the 
[User Guide](https://github.com/PyDipa/DroughtScan/blob/main/tests/docs/user_guide.md) and  [Visualization Guide](https://github.com/PyDipa/DroughtScan/blob/main/tests/docs/visualization_guide.md)

## License

DroughtScan is distributed under the [GNU GPL v3.0](LICENSE) for academic and
non-commercial research use.

For any commercial or revenue-generating application, a separate commercial
license is required. A separate commercial license can be arranged **outside** this repository 
and **does not alter** the open-source terms of the GPL for this codebase.  
For inquiries: arianna.dipaola@cnr.it

## Authors

- **Arianna Di Paola** CNR-IBE, Italy — Lead developer and maintainer; arianna.dipaola@cnr.it
- **Massimiliano Pasqui** CNR-IBE, Italy — Feedback,   scientific guidance, methodological validation and review.
- **Ramona Magno** CNR-IBE, Italy — Feedback, scientific guidance, methodological validation and review.
- **Leando Rocchi** CNR-IBE, Italy — technical support
