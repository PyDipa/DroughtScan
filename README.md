# Drought Scan

## Overview
**Drought Scan** is a Python library implementing a multi-temporal and basin-scale approach for drought analysis. It is designed to provide advanced tools for evaluating drought severity and trends at the river basin scale by integrating meteorological and hydrological data.

The methodology is described in the article:  
*"A novel framework for multi-temporal and basin-scale drought analysis"* ([Read the article](https://www.sciencedirect.com/science/article/pii/S0048969724081063?via%3Dihub)).

---

## Key Features
- Calculation of standardized drought indices (e.g., SPI, SQI, SPEI,etc).
- Integration of precipitation and streamflow data for basin-level analysis.
- Multi-temporal scales for flexibility in drought assessment.
- Possibility of generating synthetic graphs and seasonal trend analysis.

    See **tests/user_guide.md** and **tests/visualization_guide.md** for usage notes and examples
---

## Note on Installation
**Drought Scan** will soon be available on PyPI. 

To use Drought-Scan, you can install it directly from GitHub:

```bash
git clone https://github.com/PyDipa/DroughtScan.git
cd DroughtScan
pip install .
```
Alternatively, you can use the following command to install directly from the repository:

```bash
pip install git+https://github.com/PyDipa/Drought-Scan.git
```

Ensure that all dependencies listed in the repository are installed in your Python environment. Refer to the requirements.txt file for more details.

## What Drought-Scan Does

Drought-Scan provides an **end-to-end framework** for monitoring and analyzing drought conditions at the basin scale.  
It combines **statistical drought indices**, **quantitative analysis**  and **visualization tools"  into a single Python package.

### Core Capabilities
- **Data handling**: Organizes meteorological and hydrological time series (precipitation, streamflow, external predictors) into a consistent calendar (`m_cal`) and spatial framework (shapefiles of provinces/basins).
- **Drought indices**:
  - **SPI (Standardized Precipitation Index)** from 1 to K months (default K=36).
  - **SIDI or gotic D (Standardiezed Integrated Drought Index)**: a weighted multi-scale index, standardized to mean 0 and variance 1.
  - **CDN (Cumulative Deviation from Normal)**: integrates long-term memory of anomalies by cumulating the standard index at 1-month scale.
  - **SQI (Standardized Streamflow Index)**: SPI-like indicator based on river discharge.
- **Visualization**: Provides the three “pillars” of drought monitoring:
  1. Heatmap of SPI(SQI/SPEI-like) 1–K set.
  2. SIDI as a compact synthesis across scales.
  3. CDN as a long-memory diagnostic.
- **precipitation to streamflow analysis**: Allows joint analysis of precipitation- and streamflow-based indices (e.g., SIDI vs SQI) to measure the strength and the responding time of the hydrografic basin to drought events. 

