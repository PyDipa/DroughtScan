import os
from drought_scan.core_old import Precipitation, Streamflow

# Check the current directory
print("Current directory:", os.getcwd())

# Define paths to test data
shape_path = 'tests/data/po_basin_piacenza.shp'
prec_path = 'tests/data/LAPrec1871.v1.1.nc'
river_path = 'tests/data/ARPAE_Q_month.csv'

# ------------------ Precipitation Analysis ------------------
print("\n--- Precipitation Analysis ---")

# Initialize the Precipitation object
ds_prec = Precipitation(data_path=prec_path,
                        shape_path=shape_path,
                        start_baseline_year=1900,
                        end_baseline_year=1950)

# Explore attributes
print("Aggregated precipitation time series:", ds_prec.ts)
print("Monthly calendar:", ds_prec.m_cal)
print("SPI set for multiple scales:", ds_prec.spi_like_set)
print("Cumulative deviation normalized (CDN):", ds_prec.CDN)

# Plot the SPI heatmap and CDN
ds_prec.plot_scan()

# Customize the SPI calculation
ds_prec.plot_scan(optimal_k=10)  # Optimal number of months
ds_prec.plot_scan(weight_index=4)  # Logarithmically increasing weights

# Identify severe drought events
print("Default threshold:", ds_prec.threshold)
ds_prec.threshold = -1.5  # Set a custom threshold
tstartid, tendid, duration, deficit = ds_prec.severe_events_deficits()
print("Severe events started at:", ds_prec.m_cal[tstartid])

# Visualize severe events
ds_prec.plot_severe_events(tstartid=tstartid, duration=duration, deficit=deficit)
ds_prec.plot_severe_events(tstartid=tstartid, duration=duration, deficit=deficit, max_events=10, labels=True)

# What-If Scenario from the last timestamp
month,year = ds_prec.m_cal[-1]
ds_prec.what_if_scenario(year=year, month=month, window=12)

# Additional tools
print("Normal precipitation values:", ds_prec.normal_values())
coeff = ds_prec.c2r_index
spi_value = -1.5
month_index = 3   # March
scale_index = 18  # SPI18
equivalent_precipitation = np.polyval(coeff[scale_index-1, month_index-1, :], spi_value)
print(f"Equivalent precipitation for SPI18=-1.5 in March: {equivalent_precipitation}")

# ------------------ Streamflow Analysis ------------------
print("\n--- Streamflow Analysis ---")

# Initialize the Streamflow object
ds_streamflow = Streamflow(precipitation_instance=ds_prec)
print("Baseline years (inherited):", ds_streamflow.start_baseline_year, ds_streamflow.end_baseline_year)

# Override baseline if necessary
ds_streamflow = Streamflow(precipitation_instance=ds_prec, start_baseline_year=1963, end_baseline_year=1994)

# Load streamflow data
ds_streamflow.load_streamflow(river_path)

# Upload data manually (alternative)
# Example: Q = [...] and m_cal = [...]
# ds_streamflow.upload_data(ts=Q, m_cal=m_cal)

# Visualize streamflow analysis
ds_streamflow.plot_scan(weight_index=2)

# Simulate a What-If Scenario for Streamflow
month, year = ds_streamflow.m_cal[-1]
ds_streamflow.what_if_scenario(month=month, year=year, window=3)
