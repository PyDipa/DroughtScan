"""
## Example Usage

This section shows how to initialize a **Drought-Scan** analysis object (DSO) for precipitation,
with practical notes on the most important options and how they change the behavior compared to defaults.

### 1) Minimal setup (from files)
"""
import os
import drought_scan as DS
import numpy as np
# Check the current directory
print("Current directory:", os.getcwd())


# Define paths to test data
shape_path = 'tests/data/po_basin_piacenza.shp'
prec_path = 'tests/data/LAPrec1871.v1.1.nc'
river_path = 'tests/data/ARPAE_Q_month.csv'


# ------------------ Precipitation Analysis ------------------
print("\n--- Precipitation Analysis ---")

# Initialize the Precipitation object
ds = DS.Precipitation(prec_path=prec_path,
                        shape_path=shape_path,
                        start_baseline_year=1900,
                        end_baseline_year=1950,
                   basin_name = 'Po')

# Explore attributes and make graph by yourself
print("Aggregated precipitation time series:", ds.ts)
print("Monthly calendar:", ds.m_cal)
print("SPI set for multiple scales:", ds.spi_like_set)
print("Cumulative deviation normalized (CDN):", ds.CDN)

""" 
What happens here:
The library reads the NetCDF precipitation, clips/aggregates it over the basin shapefile, 
builds a monthly calendar (m_cal), and computes SPI (1–K), SIDI, 
and CDN over the baseline start_baseline_year:end_baseline_year
"""

# Plot the SPI heatmap and CDN
ds.plot_scan()

# Customize the SPI calculation
ds.plot_scan(optimal_k=10)  # Optimal number of months
ds.plot_scan(weight_index=4)  # Logarithmically increasing weights
ds.plot_scan(xlim=(2000,2010)) #highlight a specif period
# setting saveplot = True the figure is automaticallly saved the working directory
# otherwise each figure can be save by run savefig
ds.plot_scan(reverse_color=True,saveplot=False) #for SPEI and Drought Indices where high values indicate drought condiction
ds.plot_scan(reverse_color=True,xlim=(2000,2012))

from drought_scan.utils.visualization import savefig
savefig('test.png')

# Identify severe drought events
print("Default threshold:", ds.threshold)
ds.threshold = -1.5  # Set a custom threshold
tstartid, tendid, duration, deficit = ds.severe_events()
print("Severe events started at:", ds.m_cal[tstartid])

# Visualize max n severe events e track the numbers for decifics
ds.severe_events( max_events=10, labels=True)




# # What-If Scenario from the last timestamp
# month,year = ds.m_cal[-1]
# ds.what_if_scenario(year=year, month=month, window=12)

# Additional tools
print("Normal precipitation values:", ds.normal_values())
# normal precipitation values are ammount of precipitation giving spi==0
coeff = ds.c2r_index
spi_value = -1.5
month_index = 3   # March
scale_index = 18  # SPI18
equivalent_precipitation = np.polyval(coeff[scale_index-1, month_index-1, :], spi_value)
print(f"Equivalent precipitation for SPI18=-1.5 in March: {equivalent_precipitation}")

# find trends in the CDN (cumulative SPI1) using a rollling window (default is window=60):
# The CDN suggest that precipitation has quite irregular cycles of dry, stantionary or wet periods.
# custumable window allows the active cycle detection
window = 36 #months
R = ds.find_trends(window=window)
#'trend': Array with -1 (negative trend), 0 (no trend), 1 (positive trend).
# 'slope': Array with slope coefficients.
# 'p_value': Array with p-values.
# 'delta': Array with the cumulative change (slope * window size)
# Arrays has the same lenght of ds.m_cal
# to know the cumulative change in - say Nov. 2017 - over the last "window":
# find the id corresponsing to the date
date_idx = np.where((ds.m_cal[:,0]==11) & (ds.m_cal[:,1]==2017))[0][0]
# extract the delta_unit: the number of standard deviation (1 unit == 1 std) changed
delta_unit = R['delta'][date_idx]
# moltply the delta_unit for the average value of SPI1==1 (1 standard deviation)
std_to_mm = np.mean([np.polyval(coeff[0, m, :], 1) - ds.normal_values()[m] for m in range(12)])
# amount of change: (note that in this case the precipitation ds.ts is expressed as mm)
print(f'in Nov 2017 there was a trend over the last {window} months for a total of  {delta_unit*std_to_mm} mm gain/lost')

# you can also plot this amont by
ds.plot_trends()
# a set  custum windows
ds.plot_trends(windows=[50,120])

# last, plot a monthly profile of input data:
ds.plot_monthly_profile()
ds.plot_monthly_profile(highlight_years=[2017,2018])
# the same graph could be done for cumulative values (good for nival precipitation)
# and/or over 24 monthe
ds.plot_monthly_profile(cumulate=True,two_year=True,highlight_years=[2017,2018])

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
