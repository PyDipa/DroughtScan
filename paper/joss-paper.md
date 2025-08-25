---
title: 'DroughtScan: A multi-temporal and basin-scale approach for drought analysis'
tags:
  - drought
  - SPI
  - SIDI
  - hydrology
  - Python
  - climate services
authors:
  - name: Arianna Di Paola
    orcid: https://orcid.org/0000-0001-9050-4787
    affiliation: 1
  - name: Massimiliano Pasqui
    orcid: https://orcid.org/0000-0002-0926-362X
    affiliation: 2
  - name: Ramona Magno
    orcid: https://orcid.org/0000-0001-5170-2852
    affiliation: 2
  - name: Leandro Rocchi
    orcid: https://orcid.org/0000-0003-4613-8550
    affiliation: 2
affiliations:
  - name: Independent researcher and software developer
    index: 1
  - name: DroughtCentral Research Group, CNR-IBE, Florence, Italy
    index: 2
---

## Summary

**DroughtScan** is a Python toolkit for multiscale drought diagnosis based on standardized indices derived from precipitation and streamflow data. The library enables computation, visualization, and analysis of drought metrics at the basin scale, with particular focus on multi-temporal integration. It implements the **SIDI** (Standardized Integrated Drought Index), a weighted aggregation of SPI over multiple timescales, alongside CDN (Cumulative Deviation from Normal) and streamflow-based indices (SQI), supporting a comprehensive overview of drought severity, duration, and memory effects.

The software supports both gridded NetCDF input and pre-aggregated time series, and allows flexible selection of standardization methods (Gamma, Pearson III, z-score). It includes ready-to-use visualization tools such as SPI heatmaps, SIDI-CDN plots, trend detection, and intra-annual precipitation profiles. Additional methods allow direct correlation between precipitation-based and streamflow-based indices, enabling hydrological consistency checks and gap filling.

Developed within the scientific framework of the **DroughtCentral** project, DroughtScan is designed for operational drought monitoring, climate impact studies, and integration into seasonal forecasting workflows. Its modular design and transparent implementation support reproducibility and extensibility, making it suitable for use in both research and policy-oriented contexts.

## Statement of need

There is a growing demand for reproducible and customizable tools to monitor and assess drought dynamics at multiple temporal and spatial scales. While several standardized indices are used in scientific and operational settings, few open-source tools provide a consistent and extensible framework for multiscale drought analysis, including index computation, visualization, event detection, and integration with hydrological observations.

DroughtScan addresses this gap by offering a Python-based solution for computing SPI, SIDI, and CDN over customizable baselines and timescales, with built-in support for shapefile-based spatial aggregation and streamflow diagnostics. The library bridges the methodological foundations of DroughtCentral with a usable, documented, and extensible codebase. It is particularly suited for researchers and practitioners working on hydroclimatic risk assessment, drought early warning, and climate services.

## Functionality

DroughtScan provides:

- Calculation of SPI and SIDI over customizable timescales (`K`)
- Support for different standardization distributions (Gamma, Pearson III, z-score)
- Computation of CDN (Cumulative Deviation from Normal)
- Detection and ranking of severe drought events
- Integrated correlation between precipitation and streamflow-based indices (e.g., SIDI and SQI)
- Visualization tools (SPI heatmaps, SIDI/CDN trends, intra-annual profiles)
- Compatibility with machine learning and Earth system model forecasts

## Acknowledgments

This software was developed in the context of the **DroughtCentral** project, which provided the scientific framework for the SIDI index and the multiscale drought analysis methodology. The implementation was led by Arianna Di Paola, with contributions and technical guidance from members of the DroughtCentral team. We thank the broader team for their conceptual contributions and scientific support. For methodological details, see the reference paper below.

## References

Di Paola, A., Di Giuseppe, E., Magno, R., Quaresima, S., Rocchi, L., Rapisardi, E., Pavan, V., Tornatore, F., Leoni, P., & Pasqui, M. (2025).  
*Building a framework for a synoptic overview of drought.*  
Science of The Total Environment, 958, 177949.  
https://doi.org/10.1016/j.scitotenv.2024.177949  
