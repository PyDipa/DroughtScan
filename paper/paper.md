---
title: 'DroughtScan: A python package for multi-temporal and basin-scale drought analysis'
tags:
  - Drought
  - Hydrology
  - Standardized precipitation index
  - Standardized streamflow index
  - Climate services
  - Python

authors:
  - name: Arianna Di Paola
    orcid: 0000-0001-9050-4787
    affiliation: 1
  - name: Massimiliano Pasqui
    orcid: 0000-0002-0926-362X
    affiliation: 1
  - name: Ramona Magno
    orcid: 0000-0001-5170-2852
    affiliation: 2
  - name: Leandro Rocchi
    orcid: 0000-0003-4613-8550
    affiliation: 2
affiliations:
  - name: National Research Council of Italy (CNR) - Institute of BioEconomy (IBE) - Rome, Italy
    index: 1
  - name: National Research Council of Italy (CNR) - Institute of BioEconomy (IBE) - Florence, Italy
    index: 2
date: 10 October 2025
bibliography: paper.bib
---

## Summary

Effective drought management represents one of the most urgent challenges for climate change adaptation and sustainable water resource planning [@PereiraCardenal2016; @Olmstead2014]. The environmental, agricultural, and economic impacts of prolonged drought events manifest across the entire hydrological cycle, affecting surface and groundwater availability, aquifer recharge, and the balance of large river basins [@Hao2013; @Mishra2010; @Raposo2023; @VanLoon2015].

In particularly vulnerable regions such as the Mediterranean — where agriculture accounts for the largest share of total water withdrawals [@Malek2017] — the increasing severity and frequency of drought events highlight the need for operational tools that provide timely, relevant, and interpretable information to support decision-making processes.


Traditional drought monitoring relies on standardized indices such as the Standardized Precipitation Index (SPI) [@McKee1993] and the Standardized Streamflow Index (SQI) [@Telesca2012]. While widely adopted due to their simplicity and comparability, these indicators offer a static snapshot of the system that depends heavily on the selected timescale and often fail to account for antecedent conditions or cumulative stress [@VanLoon2015]. Furthermore, the understanding of drought propagation from meteorological anomalies to hydrological impacts remains limited.

To address these limitations, the **Drought Scan (DS)** framework was developed [@DiPaola2025] and later implemented as a modular end-to-end  python package. DS is an analytical framework for monitoring and quantitative drought analysis at the catchment scale.  The  DS (i) ingests and standardizes monthly hydro-climatic drivers (e.g., precipitation, PET, or P–PET/SPEI) into SPI-like families over 1 to K months, (ii) provides diagnostics and visualization to inspect triggers, propagation, and closures (heatmaps, intra-annual profiles, moving-window trends, and more), and (iii) detects and profiles severe events via thresholding and summary statistics (onset, duration, accumulated deficits).

DS introduces two complementary indicators designed to integrate short- and long-term drought signals:  
– the **Standardized Integrated Drought Index** ($\mathbb{D}_{\{\mathrm{SPI}\}}$ in the original article [@DiPaola2025], codenamed **SIDI** in the library), which condenses a full set of SPI timescales (1 to N months) into a single weighted index that can be optionally optimized against observed streamflow (SQI1);  

– and the **Cumulative Deviation from Normal (CDN)**, computed as the cumulative sum of SPI1, which captures hydrological memory by highlighting prolonged phases of surplus or deficit. While the SIDI focuses on temporally integrated responses to precipitation, the CDN offers an intuitive view of storage dynamics and long-term system stress.

Building on this conceptual foundation, we present **DroughtScan**, an open-source Python package that implements the DS framework for reproducible, basin-scale drought analysis. The library accepts inputs either as in-memory arrays/data frames provided by the user, or directly from common file formats—gridded climate data in NetCDF (e.g., precipitation, potential evapotranspiration) and streamflow time series in tabular files (CSV, Excel). It provides tools for computing SPI/SQI across multiple monthly timescales, deriving SIDI and CDN, applying different standardization distributions (Gamma, Pearson III, Gaussian), and visualizing results through SPI heatmaps, SIDI and CDN time series, momthly seasonal profiles, and trend analysis.

Input data are spatially aggregated over user-defined hydrographic units (in Shapefile or GeoJSON format), enabling consistent analysis at basin scale. DroughtScan also allows correlation between precipitation- and streamflow-based indices, supporting hydrological consistency checks and proxy construction for data-scarce regions.

The DS framework has been successfully applied in the Po River basin [@DiPaola2025], where the Pontelagoscuro closure section is used operationally by the River Basin Authority. The modular design of the software supports adaptation to diverse hydrological settings and use cases, from research to policy support. DroughtScan is modular by design: beyond precipitation-related SPI, it can ingest any monthly hydro-climatic driver or combination (e.g., Potential Evapotranspiration, PET, or P–PET), enabling SPI-like families derived from inputs other than precipitation—such as the widely used Standardized Precipitation–Evapotranspiration Index (SPEI)[@VicenteSerrano2010]. The same SIDI machinery then applies generically when the standardized index set is SPEI (i.e., SPEI from 1 to K), with identical weighting/optimization options and visualization workflows.

Developed within the scientific context of DroughtCentral (www.DroughtCentral.it), DroughtScan offers a concrete example of a climate service tool: it translates complex drought dynamics into integrated and communicable indicators, facilitating monitoring, communication, and seasonal outlooks for drought risk.

## Statement of need

Several drought indicators (e.g., SPI, SQI, SPEI) are widely used, but operational users often face two gaps: (i) integrating signals across many timescales without inspecting dozens of series, and (ii) contextualizing present anomalies against multi-year wet/dry phases that modulate risk and recovery. Existing open-source tools rarely offer a unified workflow that computes multi-scale indices, condenses them into a single impact-oriented indicator, and couples this with a memory metric.

DroughtScan fills this gap by providing: (a) immediate computation of a full set of SPI-like indices and practical visualization via a heatmap; (b) **SIDI**, a weighted integration of the SPI-like family (i.e., from 1 to K month-scales) that can be aligned with hydrological impact (SQI1); (c) **CDN**, an intuitive memory curve that tracks cumulative standardized anomalies; and (d) additional utilities for drought diagnostics and event profiling. This combination helps identify drought precursors, assess persistence, and communicate conditions in a compact, reproducible form suitable for basin-scale climate services [@DiPaola2025].
Crucially, DroughtScan treats the “SPI-like” computation as a pluggable step: users can replace precipitation with PET, or form P–PET to obtain SPEI[@VicenteSerrano2010], without changing the downstream pipeline (SIDI computation, CDN, diagnostics). This modularity supports basin-specific analysis and impact alignment, while preserving a consistent, transparent interface for analysis and communication


## Functionality

DroughtScan provides:
- Computation of SPI-like sets and SIDI over a customizable number of monthly timescales (K).
- Support for different standardization distributions (Gamma, Pearson III, Gaussian).
- Inverse standardization to retrieve reference climatological values.
- CDN (cumulative sum of SPI1) and moving-window trend analysis on CDN.
- Detection and ranking of severe drought events via thresholding on SIDI.
- Precipitation–streamflow linkage: correlations between SIDI and SQI throught a simple data-driven algorithm.
- Visualization: SPI heatmaps, SIDI and CDN time series, intra-annual profiles, and more.


Operationally, it enables users to:
- Analyze hydrological memory and identify drought precursors for major events.
- Summarize intensity and duration of water crises in an interpretable way.
- Quantify multi-year precipitation trends over user-defined horizons (e.g., 3, 5, 10 years).
- Assess meteorological-to-hydrological propagation (strength and response time).
- Reconstruct monthly streamflow where recent observations are missing (given historical data for calibration).
- Distinguish streamflow deficits of meteorological origin from those driven by human activities.


## Acknowledgments
This software was developed in the context of the **DroughtCentral** framework (www.droughtcentral.it), which provided the scientific basis for the SIDI index and the multiscale drought analysis methodology. The implementation was led by Arianna Di Paola, with contributions and technical guidance from members of the DroughtCentral team. For methodological details, see the reference paper below.
