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

Effective drought management represents one of the most urgent challenges for climate change adaptation and sustainable water resource planning [@PereiraCardenal2016; @Olmstead2014]. The environmental, agricultural, and economic impacts of prolonged drought events manifest across the entire hydrological cycle, affecting surface and groundwater availability, aquifer recharge, and the balance of large river basins [@Hao2013; @Mishra2010; @Raposo2023; @VanLoon2015].

In particularly vulnerable regions such as the Mediterranean—where agriculture accounts for 64% to 80% of total water withdrawals [@Rey2011; @Dono2013; @Dono2016]—the increasing severity and frequency of drought events highlights the need for operational tools that provide timely, relevant, and interpretable information to support decision-making processes.

Traditional drought monitoring relies on standardized indices such as the Standardized Precipitation Index (SPI) [@McKee1993] and the Standardized Streamflow Index (SQI) [@Telesca2012]. While widely adopted due to their simplicity and comparability, these indicators offer a static snapshot of the system that depends heavily on the selected timescale and often fail to account for antecedent conditions or cumulative stress [@VanLoon2015]. Furthermore, the understanding of drought propagation from meteorological anomalies to hydrological impacts remains limited.

To address these limitations, the **Drought Scan (DS)** framework was developed [@DiPaola2025]. DS introduces two complementary indicators designed to integrate short- and long-term drought signals:  
– the **Standardized Integrated Drought Index (SIDI)**, which condenses a full set of SPI timescales (1 to N months) into a single weighted index that can be optionally optimized against observed streamflow (SQI1);  
– and the **Cumulative Deviation from Normal (CDN)**, computed as the cumulative sum of SPI1, which captures hydrological memory by highlighting prolonged phases of surplus or deficit. While the SIDI focuses on temporally integrated responses to precipitation, the CDN offers an intuitive view of storage dynamics and long-term system stress.

Building on this conceptual foundation, we present **DroughtScan**, an open-source Python package that implements the DS framework for reproducible, basin-scale drought analysis. The library supports input from gridded climate data (e.g., precipitation, potential evapotranspiration) and streamflow series in tabular format. It provides tools for computing SPI/SQI across multiple monthly timescales, deriving SIDI and CDN, applying different standardization distributions (Gamma, Pearson III, Gaussian), and visualizing results through SPI heatmaps, SIDI–CDN overlays, seasonal profiles, and trend analysis.

Input data are spatially aggregated over user-defined hydrographic units (in Shapefile or GeoJSON format), enabling consistent analysis at basin scale. DroughtScan also allows correlation between precipitation- and streamflow-based indices, supporting hydrological consistency checks and proxy construction for data-scarce regions.

The DS framework has been successfully applied in the Po River basin [@DiPaola2025], where the Pontelagoscuro closure section is used operationally by the River Basin Authority. The modular design of the software supports adaptation to diverse hydrological settings and use cases, from research to policy support.

Developed within the scientific context of **DroughtCentral**, DroughtScan offers a concrete example of a climate service tool: it translates complex drought dynamics into integrated and communicable indicators, facilitating monitoring, communication, and seasonal outlooks for drought risk.


## Statement of need

There is a growing demand for reproducible and customizable tools to monitor and assess drought dynamics at multiple temporal and spatial scales. While several standardized indices are used in scientific and operational settings, few open-source tools provide a consistent and extensible framework for multiscale drought analysis, including index computation, visualization, event detection, and integration with hydrological observations.

DroughtScan addresses this gap by offering a Python-based solution for computing SPI, SIDI, and CDN over customizable baselines, timescales, and over user-defined hydrographic units with built-in support for shapefile-based spatial aggregation and streamflow diagnostics. The library bridges the methodological foundations of DroughtCentral with a usable, documented, and extensible codebase. It is particularly suited for researchers and practitioners working on hydroclimatic risk assessment, drought early warning, and climate services at the river basin scale.

## Functionality

DroughtScan provides:
- Calculation of SPI-like sets and SIDI over a customizable number of timescales (K);
- Support for different standardization distributions (Gamma, Pearson III, Gaussian);
- Computation of reference climatological values via inverse standardization;
- Computation of CDN (Cumulative Deviation from Normal of the 1-month standardized index);
- Search, quantify, and plot trends in specific moving windows of the CDN curve
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



## Acknowledgments

This software was developed in the context of the **DroughtCentral** framework (www.droughtcentral.it), which provided the scientific basis for the SIDI index and the multiscale drought analysis methodology. The implementation was led by Arianna Di Paola, with contributions and technical guidance from members of the DroughtCentral team. For methodological details, see the reference paper below.
