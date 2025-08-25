# Drought Scan

## Overview
**Drought Scan** is a Python library implementing a multi-temporal and basin-scale approach for drought analysis. It is designed to provide advanced tools for evaluating drought severity and trends at the river basin scale by integrating meteorological and hydrological data.

The methodology is described in the article:  
*"A novel framework for multi-temporal and basin-scale drought analysis"* ([Read the article](https://www.sciencedirect.com/science/article/pii/S0048969724081063)).
and is continuously developed within the activities of Drought Central ([DroughtCentral](https://droughtcentral.it)).

---

## Key Features
- Calculation of standardized drought indices (e.g., SPI, SQI, SPEI,etc).
- Integration of precipitation and streamflow data for basin-level analysis.
- Multi-temporal scales for flexibility in drought assessment.
- Possibility of generating synthetic graphs and seasonal trend analysis.

for examples and usage notes see: 
- [User Guide](tests/docs/user_guide.md)→ Demonstrates how to initialize a Drough-Scan Object
- [Visualization Guide](tests/docs/visualization_guide.md) → Demonstrates how to use some visualization methods.

---
## Installation

> **Note:** DroughtScan will soon be available on PyPI.  
> Until then, it can be installed directly from this repository.

### Option 1: Clone and install locally
```bash
git clone https://github.com/PyDipa/DroughtScan.git
cd DroughtScan
pip install .
```

Option 2: Install directly from GitHub (no local clone)
```bash
pip install git+https://github.com/PyDipa/DroughtScan.git
```

Ensure that all dependencies listed in the repository are installed in your Python environment. Refer to the pyproject.toml.txt file for more details.

## What Drought-Scan Does

Drought-Scan provides an **end-to-end framework** for monitoring and analyzing drought conditions at the basin scale.  
It combines **statistical drought indices**, **quantitative analysis**  and **visualization tools**  into a single Python package.

### Core Capabilities
- **Data handling**: Organizes meteorological and hydrological time series (precipitation, streamflow, external predictors) into a consistent calendar (`m_cal`) and spatial framework (shapefiles of provinces/basins).
- **Drought indices**:
  - **SPI (Standardized Precipitation Index)** from 1 to K months (default K=36).
  - **SIDI or gotic D (Standardized Integrated Drought Index)**: a weighted multi-scale index, standardized to mean 0 and variance 1.
  - **CDN (Cumulative Deviation from Normal)**: integrates long-term memory of anomalies by cumulating the standard index at 1-month scale.
  - **SQI (Standardized Streamflow Index)**: SPI-like indicator based on river discharge.
- **Visualization**: Provides the three “pillars” of drought monitoring:
  1. Heatmap of SPI(SQI/SPEI-like) 1–K set.
  2. SIDI as a compact synthesis across scales.
  3. CDN as a long-memory diagnostic.
- **precipitation to streamflow analysis**: Allows joint analysis of precipitation- and streamflow-based indices (e.g., SIDI vs SQI) to measure the strength and the responding time of the hydrographic basin to drought events. 

## Authors

- **Arianna Di Paola** CNR-IBE, Italy — Lead developer and maintainer; arianna.dipaola@cnr.it
- **Massimiliano Pasqui** CNR-IBE, Italy — Feedback,   scientific guidance, methodological validation and review.
- **Ramona Magno** CNR-IBE, Italy — Feedback, scientific guidance, methodological validation and review.
- **Leando Rocchi** CNR-IBE, Italy — technical support