# Drought-Scan

## Overview
**Drought Scan** is a Python library implementing a multi-temporal and basin-scale approach for drought analysis. It is designed to provide advanced tools for evaluating drought severity and trends at the river basin scale by integrating meteorological and hydrological data.

The methodology is described in the article:  
*"A novel framework for multi-temporal and basin-scale drought analysis"* ([Read the article](https://www.sciencedirect.com/science/article/pii/S0048969724081063?via%3Dihub)).

---

## Key Features
- Calculation of standardized drought indices (e.g., SPI, SQI).
- Integration of precipitation and streamflow data for basin-level analysis.
- Multi-temporal scales for flexibility in drought assessment.
- Possibility of generating synthetic graphs and seasonal trend scenarios (What-If scenarios).

    See **docs/DetailedUsage.md** for usage notes and examples
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
