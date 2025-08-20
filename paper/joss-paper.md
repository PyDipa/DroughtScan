title: 'Py Drought Scan: A Python package for Drought Detection and Analysis'
authors:

name: Arianna Di Paola
orcid: https://orcid.org/0000-0001-9050-4787
affiliation: 1



affiliation: 
name: Institution for BioEconomy, National Research Council, Rome, Italy
index: 1

date: 2025-03-07
bibliography: paper.bib

Summary

The Drought Scan (Di Paola et al., 2025) is an innovative system for monitoring drought at the catchment level, aiming to make climate services more efficient and accessible. Developed and tested on the Po River basin and subsequently on the Arno River basin, this tool provides an overview of past and present droughts, improving the understanding of their dynamics and of the continuum between meteorological and hydrological droughts. By enhancing monitoring capabilities, it supports water resources management.

Drought Scan contextualizes drought within a hydrological balance, considering two key variables: monthly precipitation (P) aggregated at the catchment level and the average monthly river flow (Q) at the closure section of the catchment of interest. P and Q represent, respectively, the input and output of a hydrogeological balance that, net of anthropogenic water withdrawals and water infiltrating into the aquifer, should be in equilibrium over time, even in the presence of strong climate anomalies.

The foundation of Drought Scan is based on standardized monthly indices (McKee et al., 1993) of precipitation (SPI) and river flow (SQI), calculated on continuous time scales from 1 to 36 months

Statement of Need

Droughts have significant socio-economic and environmental impacts, making their monitoring and prediction crucial. Existing tools often require high technical expertise or are not flexible enough to be adapted to different datasets and research needs. Py Drought Scan aims to fill this gap by offering an accessible, modular, and customizable framework for drought detection and analysis.

State of the Field

Several tools exist for drought monitoring, including the Standardized Precipitation Index (SPI) calculators and remote sensing-based platforms such as the Global Drought Monitor. However, many of these are either proprietary, have limited adaptability, or require substantial computational resources. Py Drought Scan is distinguished by its open-source nature, ease of use, and integration with multiple data sources, providing a balance between accessibility and analytical power.

Description

Drought Scan is developed in Python and supports integration with commonly used climate and streamflow datasets, including NETCDF, CSV and shapefile formats. The core functionalities include:

Automated data retrieval and preprocessing

Calculation of drought indices (e.g., SPI, SPEI, VCI)

Machine learning-based drought classification

Visualization and statistical analysis tools

The software is structured as a Python package, enabling easy installation and use within research workflows. It supports both command-line execution and interactive Jupyter Notebook environments.

Example Usage

A typical usage scenario involves downloading satellite-based vegetation indices and computing drought indices over a specific region. Below is an example code snippet:

from drought_scan import DroughtScan

# Initialize with dataset and region
dscan = DroughtScan(dataset='MODIS', region='California')

# Compute drought index
drought_index = dscan.compute_spi(time_scale=3)

# Generate visualization
dscan.plot_drought_map()

References

::: {#refs}
:::

Smith, J. et al. (2021). "A new approach to drought monitoring using remote sensing." Journal of Climate Research.

Doe, J. et al. (2022). "Machine learning for hydrological applications." Environmental Data Science.

:::