---
title: 'DroughtScan: A multi-temporal and basin-scale approach for drought analysis'
tags:
  - Drought
  - Hydrology
  - Standardized precipitation index
  - Standardized streamflow index
  - Climate services
  - Python

authors:
  - name: Arianna Di Paola
    orcid: https://orcid.org/0000-0001-9050-4787
    affiliation: 1
  - name: Massimiliano Pasqui
    orcid: https://orcid.org/0000-0002-0926-362X
    affiliation: 1
  - name: Ramona Magno
    orcid: https://orcid.org/0000-0001-5170-2852
    affiliation: 2
  - name: Leandro Rocchi
    orcid: https://orcid.org/0000-0003-4613-8550
    affiliation: 2
affiliations:
  - name: National Research Council of Italy (CNR) - Institute of BioEconomy (IBE) - Rome, Italy
    index: 1
  - name: National Research Council of Italy (CNR) - Institute of BioEconomy (IBE) - Florence, Italy
    index: 2
---

## Summary

**DroughtScan** is a Python package for multiscale drought diagnosis based on standardized indices derived from climate and hydrological variables such as precipitation (P), streamflow (Q), or potential evapotranspiration (PET). The system relies on the joint analysis of monthly precipitation and, where available, mean river flow at the basin closure section, interpreted respectively as the input and output of the hydrological balance.
The library enables computation, visualization, and analysis of drought metrics at the basin scale, with particular emphasis on multi-temporal indices and their integration. Each input variable is standardized into a continuous set of monthly timescales (e.g. from P to the Standardized Precipitation Index, SPI1–36), producing a family of SPI or **SPI-like** indices. From this ensemble the library derives two key indicators: the **SIDI** (Standardized Integrated Drought Index), defined as a weighted mean across all scales to facilitate drought monitoring, communication, and climate services; and the **CDN** (Cumulative Deviation from Normal), computed from the shortest timescale (e.g. SPI1) to track the cumulative persistence of anomalies and the long-term memory of the system. Together, these indices provide a comprehensive overview of drought severity, duration, and memory effects

The software supports gridded **NetCDF** input for atmospheric variables such as P and PET, as well as **CSV** or **Excel** formats for tabular data such as Q. Input data are spatially aggregated over the selected hydrographic basin (provided in **GeoJSON** or **Shapefile** format) and, when gridded, are further aggregated into monthly time series. The package allows flexible selection of standardization methods based on well-established distributions (**Gamma, Pearson III, Gaussian**) and includes ready-to-use visualization tools such as **SPI heatmaps**, **SIDI–CDN plots**, **trend detection**, and intra-annual precipitation profiles. Additional methods enable direct correlation between precipitation-based and streamflow-based indices, supporting hydrological consistency checks and gap-filling procedures

Developed within the scientific framework **DroughtCentral**, DroughtScan is designed for operational drought monitoring and climate impact studies. Its modular design and transparent implementation support reproducibility and extensibility, making it suitable for use in both research and policy-oriented contexts.

## Statement of need

There is a growing demand for reproducible and customizable tools to monitor and assess drought dynamics at multiple temporal and spatial scales. While several standardized indices are used in scientific and operational settings, few open-source tools provide a consistent and extensible framework for multiscale drought analysis, including index computation, visualization, event detection, and integration with hydrological observations.

DroughtScan addresses this gap by offering a Python-based solution for computing SPI, SIDI, and CDN over customizable baselines and timescales, with built-in support for shapefile-based spatial aggregation and streamflow diagnostics. The library bridges the methodological foundations of DroughtCentral with a usable, documented, and extensible codebase. It is particularly suited for researchers and practitioners working on hydroclimatic risk assessment, drought early warning, and climate services.

## Functionality

DroughtScan provides:
- Calculation of SPI-like sets and SIDI over a customizable number of timescales (K);
- Support for different standardization distributions (Gamma, Pearson III, Gaussian);
- Computation of CDN (Cumulative Deviation from Normal of the 1-month standardized index);
- Detection and ranking of severe drought events;
- Integrated correlation between precipitation- and streamflow-based indices (e.g. SIDI vs. SQI);
- Visualization tools (SPI heatmaps, SIDI/CDN trends, intra-annual profiles);

From an operational and climate-service perspective, DroughtScan is designed to:

- Analyze the hydrological memory of the system and dentify drought precursors responsible for major critical events;
- Summarize the intensity and duration of water crises in an easily interpretable form;
- Provide objective measures of precipitation trends over user-defined timescales (e.g., 3, 5, or 10 years);
- Assess the propagation of meteorological drought into hydrological drought, detecting both the strength and response time of drought signals in the meteorological-to-hydrological continuum;
- Reconstruct monthly streamflow in the absence of recent observations, provided a historical series is available for calibration
- Distinguish between streamflow deficits caused by meteorological drought and those of anthropogenic origin

For methodological details, see @DiPaola2025.

## Acknowledgments

This software was developed in the context of the **DroughtCentral** project, which provided the scientific framework for the SIDI index and the multiscale drought analysis methodology. The implementation was led by Arianna Di Paola, with contributions and technical guidance from members of the DroughtCentral team. For methodological details, see the reference paper below.

## References

Di Paola, A., Di Giuseppe, E., Magno, R., Quaresima, S., Rocchi, L., Rapisardi, E., Pavan, V., Tornatore, F., Leoni, P., & Pasqui, M. (2025).  
*Building a framework for a synoptic overview of drought.*  
Science of The Total Environment, 958, 177949.  
https://doi.org/10.1016/j.scitotenv.2024.177949  
