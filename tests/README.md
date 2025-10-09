# Test Suite

This directory contains **unit tests and example usage** for Drought Scan.

### 📂 Subdirectories:
- `data/` → Contains **test datasets**:
- the **shapefile** of the Po basin; 
- the precipitation dataset **LAPrec**: This dataset provides **gridded monthly precipitation** over the **Alpine region** (eight countries), available in two versions—one starting in **1871** (85 station series) and another in **1901** (165 series)—reconstructed by combining long-term station data and high‑resolution gridded observations via optimal interpolation.
Per maggiori dettagli, consulta la pagina ufficiale su Copernicus: [https://cds.climate.copernicus.eu/datasets/insitu‑gridded‑observations‑alpine‑precipitation?tab=overview](https://cds.climate.copernicus.eu/datasets/insitu‑gridded‑observations‑alpine‑precipitation?tab=overview)


- **Po River streamflow data** ([mc/sec]), collected at Pontelagoscuro
gauging's closure section, are retrieved from Agenzia Regionale per la
Prevenzione, l'Ambiente e l'Energia dell'Emilia-Romagna (ARPAE, http
s://www.arpae.it/it) as daily average for the period December 1961 – october 2024.


- the **Potential Evapotranspiration (pev)** and **air temperature (t2m)**from **ERA5-Land**: these dataset are the monthly reanalysis for potential evaporation and air temperture, provided by the Copernicus Climate Data Store. They covers the period from 1950 to present, with data available at monthly resolution on a 0.1° (~9 km) grid, and your request restricts the domain to Europe/North Africa (60°N–24°N, 12°W–40°E). Data are delivered in NetCDF format, consistent with the ERA5-Land archive conventions, and represent monthly averages of reanalysis fields produced with the ECMWF IFS model. 
Official dataset page: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means


 
- `docs/` → Contains **user guides**:
### 📄 Quick Start:
- `docs/user_guide.md` → Demonstrates how to use the library.
 `docs/visualization_guide.md` → Demonstrates how to use some visualization methods.
-  `docs/common_errors.md` → Demonstrates how to solve some potential iussues
- `__init__.py` → Initializes the test suite.


