"""
author: PyDipa
# © 2025 Arianna Di Paola
# License: GNU General Public License v3.0 (GPLv3)

Data Input/Output Utilities.

This module provides functions for:
- **Loading and processing meteorological data** (NetCDF, CSV).
- **Aggregating time-series data** (e.g., daily to monthly precipitation).
- **Applying spatial masks** for regional analysis.
- **Handling missing values** in climate datasets.

Main functions:
- `import_netcdf_for_cumulative_variable()`: Reads precipitation data from NetCDF.
- `create_mask()`: Generates a spatial mask for selected study areas.
- `load_shape()`: Loads and reprojects shapefiles for spatial analysis.
- `get_regex_for_date_format()`: Returns regex for matching date formats.
- `check_datetime()`: Checks if a string matches common date formats.
- `detect_delimiter()`: Detects delimiters in CSV files.
- `extract_variable()`: Extracts a variable from a NetCDF dataset.

Used by: `core.py`, `drought_indices.py`.
"""

import re
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Helvetica'
import geopandas as gpd
import regionmask
import netCDF4 as nc
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime
import pandas as pd

# from datetime import datetime,timedelta

# Ensure compatibility with GeoPandas
os.environ['USE_PYGEOS'] = '0'

def get_regex_for_date_format(date_format):
    """
    Returns the regex corresponding to a given date format string.

    Args:
        date_format (str): Date format expressed as a string (e.g., 'YYYY-MM-DD').

    Returns:
        str: The regex corresponding to the format or None if not found.
    """
    # Mapping dictionary: Literal format -> Regex
    format_to_regex = {
        "YYYY-MM-DD": r'^\d{4}-\d{2}-\d{2}$',
        "YYYY/MM/DD": r'^\d{4}/\d{2}/\d{2}$',
        "DD-MM-YYYY": r'^\d{2}-\d{2}-\d{4}$',
        "DD/MM/YYYY": r'^\d{2}/\d{2}/\d{4}$',
        "DD/MM/YY": r'^\d{2}/\d{2}/\d{2}$',
        "YYYYMMDD": r'^\d{4}\d{2}\d{2}$',
        "DD MMM YYYY": r'^\d{2}\s\w{3}\s\d{4}$',    # e.g., 01 Dec 2023
        "MMM DD, YYYY": r'^\w{3}\s\d{2},\s\d{4}$',  # e.g., Dec 01, 2023
        "YYYY-DOY": r'^\d{4}-\d{3}$',               # Julian day, e.g., 2023-365
        "YYYY.MM.DD": r'^\d{4}\.\d{2}\.\d{2}$',
        "DD.MM.YYYY": r'^\d{2}\.\d{2}\.\d{4}$',
        "YYYY/MM": r'^\d{4}/\d{2}$',                # Year and month
        "HH:MM:SS": r'^\d{2}:\d{2}:\d{2}$',
        "ISO8601": r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$',  # ISO 8601
        "YYYY-MM-DD": r'^\d{4}-\d{2}-\d{2}$' #e.g., 1960-12-31
    }

    # Return the regex if the format matches any of the defined patterns
    return format_to_regex.get(date_format, None)

def check_datetime(text):
    """
    Checks if the input text matches one of the most common date formats.

    Args:
        text (str): A string to be checked.

    Returns:
        bool: True if the text matches a recognized date format, otherwise False.
    """
    # Dictionary of formats to regex
    format_to_regex = {
        "YYYY-MM-DD": r'^\d{4}-\d{2}-\d{2}$',
        "YYYY/MM/DD": r'^\d{4}/\d{2}/\d{2}$',
        "DD-MM-YYYY": r'^\d{2}-\d{2}-\d{4}$',
        "DD/MM/YYYY": r'^\d{2}/\d{2}/\d{4}$',
        "DD/MM/YY": r'^\d{2}/\d{2}/\d{2}$',
        "YYYYMMDD": r'^\d{4}\d{2}\d{2}$',
        "DD MMM YYYY": r'^\d{2}\s\w{3}\s\d{4}$',
        "MMM DD, YYYY": r'^\w{3}\s\d{2},\s\d{4}$',
        "YYYY-DOY": r'^\d{4}-\d{3}$',
        "YYYY.MM.DD": r'^\d{4}\.\d{2}\.\d{2}$',
        "DD.MM.YYYY": r'^\d{2}\.\d{2}\.\d{4}$',
        "YYYY/MM": r'^\d{4}/\d{2}$',
        "HH:MM:SS": r'^\d{2}:\d{2}:\d{2}$',
        "ISO8601": r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$'
    }

    # Itera su tutti i regex per verificare se il testo corrisponde a un formato
    for regex in format_to_regex.values():
        if re.match(regex, text):
            return True
    return False
def detect_delimiter(line):
    """
    Detects the most likely delimiter in a CSV line.

    Args:
        line (str): A line from the CSV file.

    Returns:
        str: The most likely delimiter.
    """
    # List of common delimiters to check
    delimiters = ['\t', ';', ',', '|']
    best_delimiter = None
    max_columns = 0

    # Iterate through possible delimiters
    for delim in delimiters:
        columns = line.split(delim)
        # Update the best delimiter if this one produces more columns
        if len(columns) > max_columns:
            max_columns = len(columns)
            best_delimiter = delim
    # return max(delimiters, key=lambda delim: len(line.split(delim)))
    return best_delimiter


def extract_variable(data, possible_names):
    """
    Extract a variable from a NetCDF dataset by checking possible names.

    Args:
        data (Dataset): Opened NetCDF dataset.
        possible_names (list): List of potential variable names (strings)

    Returns:
        ndarray: Extracted variable array.
    """
    # check of the inputs:
    if not hasattr(data, "variables"):
        raise TypeError("The provided `data` is not a valid NetCDF dataset.")
    if not isinstance(possible_names, list) or not all(isinstance(name, str) for name in possible_names):
        raise TypeError("`possible_names` must be a list of strings.")

    # looking for variables names in the NetCDF file
    for name in possible_names:
        if name in data.variables:
            try:
                return np.array(data[name])
            except Exception as e:
                raise ValueError(f"Error extracting variable '{name}': {e}")
    # raise ValueError(f"None of {possible_names} found in NetCDF variables.")

def create_mask(shape,LAT, LON):
    """
    Create a mask of the region defined by the shapefile.

    Args:
        LAT (ndarray): Latitude grid  (2D array).
        LON (ndarray): Longitude grid  (2D array).

    Returns:
        ndarray: Mask array where 0 indicates the region of interest.
    """
    lat_steps, lon_steps = LAT.shape[0], LON.shape[1]
    lat_grid = np.linspace(np.min(LAT), np.max(LAT), lat_steps)
    lon_grid = np.linspace(np.min(LON), np.max(LON), lon_steps)

    mask = regionmask.mask_geopandas(shape, lon_grid, lat_grid)
    return np.flipud(mask)

def import_netcdf_for_cumulative_variable(file_path, possible_names,shape,verbose):
    """
    Loads precipitation oe PET data from a NetCDF file and applies spatial aggregation.

    Args:
        file_path (str): Path to the NetCDF file.
        possible_names (list): List of possible variable names.
        shape (GeoDataFrame): Spatial mask for regional data extraction.
        verbose (bool, optional): If True, displays additional information.

    Returns:
        tuple:
            - ts (ndarray): Aggregated time series.
            - m_cal (ndarray): Monthly and yearly timestamps.
            - Pgrid (ndarray): Precipitation data grid.

    Raises:
        FileNotFoundError: If the NetCDF file is not found.
        RuntimeError: If data processing encounters an error.
    """
    try:
        with nc.Dataset(file_path, 'r') as data:
            # Extract latitudes and longitudes
            Lat = extract_variable(data, ['latitude', 'lat', 'LAT'])
            Lon = extract_variable(data, ['longitude', 'lon', 'LON'])

            # Create 2D grid if coordinates are 1D
            LAT, LON = np.meshgrid(Lat, Lon, indexing='ij') if Lat.ndim == 1 else (Lat, Lon)

            # Load precipitation data
            # tp_var = next(var for var in data.variables if data[var].ndim == 3)
            Pgrid = extract_variable(data,possible_names)
            # Pgrid = np.array(data['tp'][:], dtype=float)
            Pgrid[Pgrid < 0] = 0  # Mask invalid values

            # Create time metadata
            try:
                dates = nc.num2date(data['time'][:], units=data['time'].units, calendar=data['time'].calendar)
            except IndexError:
                dates = nc.num2date(data['valid_time'][:], units=data['valid_time'].units, calendar=data['valid_time'].calendar)
            m_cal = np.array([[date.month, date.year] for date in dates])

            time_diffs = np.diff(dates)
            # Converti in giorni
            days_diffs = np.array([td.days for td in time_diffs])
            # days = np.array([date.day for date in dates])
            # if np.all(days_diffs <= 1) or np.median(days_diffs) == 1:

            # Check temporal resolution and aggregate if daily
            if np.median(days_diffs) >= 28 and np.median(days_diffs) <= 31:
                pass
            else:
                print("Data appears to have daily resolution. Aggregating to monthly.")
                years = np.unique(m_cal[:, 1])
                Pgrid_m = np.empty((len(years) * 12, *Pgrid.shape[1:]))
                Pgrid_m[:] = np.nan

                for yr_idx, year in enumerate(years):
                    for month in range(1, 13):
                        month_indices = np.where((m_cal[:, 1] == year) & (m_cal[:, 0] == month))[0]
                        if len(month_indices) > 0:
                            monthly_sum = np.nansum(Pgrid[month_indices, :, :], axis=0)
                            Pgrid_m[yr_idx * 12 + month - 1, :, :] = monthly_sum

                Pgrid = Pgrid_m
                m_cal = np.array([[m, y] for y in years for m in range(1, 13)])

            # Flip LAT if necessary
            if LAT[0, 0] < LAT[1, 0]:
                LAT = np.flipud(LAT)
                Pgrid = np.flip(Pgrid, axis=1)

            # Create a regional mask
            mask = create_mask(shape,LAT, LON)
            if mask is None:
                raise ValueError("Failed to create regional mask.")
            if mask.shape != LAT.shape:
                raise ValueError(
                    f"Mismatch between mask shape {mask.shape} and grid spatial shape {Pgrid.shape[1:]}.")

            if verbose==True:
                print(f'Regional mask created: mask shape {mask.shape}, grid shape {Pgrid.shape}')
                check = Pgrid[1, :, :] if len(np.shape(Pgrid))==3 else Pgrid[0,0,:,:]
                plt.figure()
                plt.imshow(check, cmap='viridis')
                plt.imshow(mask, cmap='jet_r')
                plt.title('Overlay of data field and river basin mask')
                plt.show()

            # Aggregate precipitation timeseries over the basin
            if len(np.shape(Pgrid))>3: #forecast/multi members dataset
                ts = np.array(
                    [[np.nanmean(Pgrid[t, m, mask >= 0]) for m in range(Pgrid.shape[1])] for t in range(Pgrid.shape[0])])
            elif len(np.shape(Pgrid))==3:
                # ts = np.array([np.nanmean(Pgrid[i, mask >= 0]) for i in range(Pgrid.shape[0])])
                ts = np.nanmean(Pgrid[:, mask >= 0], axis=1)

    except FileNotFoundError:
        raise FileNotFoundError(f"NetCDF file not found at: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error importing data: {e}")


    return ts, m_cal, Pgrid

def load_shape(shape_path):
    """
    Load and reproject the shapefile to WGS84 (EPSG:4326).

    This method loads a shapefile, checks its coordinate reference system (CRS),
    and reprojects it to WGS84 (EPSG:4326) if necessary. If the CRS is not set,
    it assigns EPSG:4326 as the default CRS.

    Args:
        shape_path (str): Path to the shapefile.

    Returns:
    GeoDataFrame: Shapefile data reprojected to WGS84.
    """
    try:
        # Verifica se il file esiste
        if not os.path.exists(shape_path):
            raise FileNotFoundError(f"The shapefile '{shape_path}' does not exist.")

        # Carica lo shapefile
        shape = gpd.read_file(shape_path)

        # Verifica o imposta il CRS
        if shape.crs is None:
            shape = shape.set_crs('epsg:4326')
        elif shape.crs.to_string() != 'EPSG:4326':
            shape = shape.to_crs('epsg:4326')

        return shape
    except Exception as e:
        raise ValueError(f"Error loading shapefile: {e}")
def import_timeseries(data_path):
    var_name, starting_date, ending_date = get_teleindex_info(data_path)
    data = nc.Dataset(data_path)
    # Estraiamo mese e anno dal timestamp iniziale
    mm,yr =  starting_date.month,  starting_date.year
    row1 = np.where((m_cal[:, 0] == mm) & (m_cal[:, 1] == yr))[0][0]
    # Estraiamo mese e anno dal timestamp finale
    mm,yr, anno = ending_date.month, ending_date.year
    row2 = np.where((m_cal[:, 0] == mm) & (m_cal[:, 1] == yr))[0][0]
    # try:
    #     predittori[row1:row2+1,i] = data[var_name[1]][:]
    # except ValueError:
    #     print(f'discontinuità rilevate in {lista[i]} -  {Vars[i][1]}')

def load_streamflow_from_csv(file_path, date_col=None, value_col=None):
        """
        Load and process streamflow data from a csv file.

        This method automatically detects the file format, delimiter, and column structure,
        then processes the data to create a usable streamflow time series. If the data is
        daily, it aggregates it to monthly averages.

        Args:
            file_path (str): Path to the streamflow data file.
            date_col (str, optional): Name of the column containing date values. If None, it is auto-detected.
            value_col (str, optional): Name of the column containing streamflow values. If None, it is auto-detected.

        Returns:
            None: Updates `self.ts` and `self.m_cal` attributes, and recalculates derived attributes.

        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        with open(file_path, "r", encoding="utf-8", errors="replace") as file:
            lines = file.readlines()  # Legge tutte le righe del file

        skip_rows = 0  # Contatore righe da saltare
        while skip_rows < len(lines):

            # explore data getting the first 5 rows starting from skip_rows
            test_block = lines[skip_rows:skip_rows + 5]

            # Try separating columns using a space, comma or semicolon.
            split_lines = [line.strip().replace(",", ".").split() for line in test_block]

            # Convert the first two columns to numeric arrays
            split_lines = np.array(split_lines,dtype=object)
            # print(f'skip_rows = {skip_rows}')
            if split_lines.ndim>1:
                if (skip_rows==0) | (skip_rows==1):
                    skip_rows=0
                delimiter = detect_delimiter(lines[skip_rows + 1])
                break
            else:
                skip_rows += 1

        # !tail -n 20 file_path

        end_row = 1 if skip_rows==0 else skip_rows
        while end_row < len(lines):

            # explore data getting the first 5 rows starting from skip_rows
            test_block = lines[end_row:end_row + 5]

            # Try separating columns using a space, comma or semicolon.
            split_lines = [line.strip().replace(",", ".").split() for line in test_block]

            # Convert the first two columns to numeric arrays
            split_lines = np.array(split_lines,dtype=object)
            # print(f'skip_rows = {skip_rows}')
            if split_lines.ndim==1:
                break
            else:
                end_row += 1

        if len(lines)-end_row==0: #no skipfooter
            skip_footer = 0
        elif len(lines)-end_row>0:
            skip_footer = len(lines)-end_row-skip_rows
        else:
            print('check the footer of the csv file')

        sospetti = ('#', '@', '--', '//')  # aggiungi qui altri prefissi sospetti
        if lines[skip_rows].lstrip().startswith(sospetti):
            skip_rows=skip_rows+1

        df = pd.read_csv(
            file_path,encoding="utf-8",
            encoding_errors="ignore",
            delimiter=delimiter,
            skiprows=skip_rows,
            skipfooter=skip_footer,
            na_values=['-9999', '-999.000', '@'],
            engine='python',
            index_col=False,
            header=0
        )

        # Remove extra spaces from column names
        # df.columns = df.columns.str.strip()
        # remove any potential colums of only nan
        df = df.dropna(axis=1, how='all')

        df_example =  df.iloc[0].to_numpy()

        # Auto-detect date and value columns if not provided

        for col in range(30):
            try:
                check = check_datetime(df_example[col][0:10])
                if date_col is None and (check == True or ":" in df_example[col]):
                    date_column = df_example[col]
                    date_col = col

            except IndexError:
                pass

        # for col in range(5):
        #     try:
        #         check = check_datetime(df_example[col][0:10])
        #         if value_col is None and (check ==False and ":" not in df_example[col]):
        #             value_column = df_example[col]
        #     except IndexError:
        #         pass
        for col in range(30):
            try:
                check = check_datetime(df_example[col][0:10])
                pass
            except IndexError:
                try:
                    df_example[col]
                    if value_col is None:
                        value_column = df_example[col]
                        value_col = col
                except IndexError:
                    pass

        value_col, date_col = None,None
        if value_col is None:
            value_col = [col for col in df.columns if df[col].eq(float(value_column)).any()][0]
        if date_col is None:
            date_col = [col for col in df.columns if df[col].eq(date_column).any()][0]

        try:
            df[value_col].mean()
        except TypeError:
            decimal = ','
            df = pd.read_csv(
                file_path, encoding="utf-8", encoding_errors="ignore",
                delimiter=delimiter,
                skiprows=skip_rows,
                na_values=['-9999', '-999.000', '@'],
                engine='python',
                decimal=decimal,
            )

        # remove any potential colums of only nan
        df = df.dropna(axis=1, how='all')

        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.dropna(subset=[date_col, value_col])

        # Aggregate daily data to monthly averages if needed
        if df[date_col].dt.day.unique().size > 1:
            print("Risoluzione giornaliera rilevata: aggrego a medie mensili.")
            # -------------------------------------------------------
            # # Here to counts the number of monthly observations
            # grouped = df.set_index(date_col).resample('ME')[value_col]
            # count_valid = grouped.count()
            # count_total = df.set_index(date_col).resample('ME')[value_col].size()
            # ratio = count_valid / count_total
            # monthly_mean = grouped.mean()
            # # choosign at least 70% of availabel days
            # monthly_mean[ratio >= 0.7].reset_index()
            # -------------------------------------------------------------

            df = df.resample('ME', on=date_col)[value_col].mean().reset_index()

        # Update class attributes

        ts = df[value_col].values
        m_cal = np.column_stack((df[date_col].dt.month, df[date_col].dt.year))

        # Welcome and guidance messages
        print("#########################################################################")
        print("streamflow data has been imported successfully.")
        print(f"data starts from {m_cal[0]} and ends on {m_cal[-1]}.")
        print("#########################################################################")
        print("Run the following class methods to access key functionalities:\n")
        print(" >>> ._plot_scan(): to plot the sqiset heatmap and D_{SPI} \n ")
        print("*************** Alternatively, you can access to: \n >>> streamflow.ts (Q timeseries), \n >>> streamflow.spi_like_set (SQI (1:K) timeseries) \n >>> streamflow.SIDI (D_{SQI}) \n to visualize the data your way or proceed with further analyses!")

        return ts,m_cal

def load_streamflow_from_excel(file_path, date_col=None, value_col=None):
    df = pd.read_excel(file_path)


    if all(col in df.columns for col in  ['Giorno', 'Mese', 'Anno']):
        df['data'] = pd.to_datetime(dict(year=df['Anno'], month=df['Mese'], day=df['Giorno']))
        date_col = 'data'
        date_format = ['Giorno', 'Mese', 'Anno']
    if all(col in df.columns for col in ['gg', 'mm', 'aaaa']):
        df['data'] = pd.to_datetime(dict(year=df['aaaa'], month=df['mm'], day=df['gg']))
        date_col = 'data'
        date_format= ['gg', 'mm', 'aaaa']
    elif date_col is None:
        raise ValueError("Colonne di data non trovate e 'date_col' non specificato.")

    # Individua colonna con il valore se non specificata
    if value_col is None:
        # Se c'è solo una colonna numerica esclusa la data, la usiamo
        value_candidates = df.select_dtypes(include='object').columns.tolist() + \
                           df.select_dtypes(include='float').columns.tolist()
        value_candidates = [col for col in value_candidates if col not in date_format + [date_col]]
        if len(value_candidates) == 1:
            value_col = value_candidates[0]
        if any('mc/s' in col for col in value_candidates):
            value_col = next(col for col in value_candidates if 'mc/s' in col)
        elif any('value' in col for col in value_candidates):
            value_col = next(col for col in value_candidates if 'value' in col)
        else:
            raise ValueError("Impossibile determinare la colonna dei valori: specificare 'value_col'.")

    # Conversione a stringa e pulizia iniziale
    df[value_col] = df[value_col].astype(str).str.strip().str.replace(',', '.', regex=False)

    # Sostituzione di valori problematici con NaN
    na_values = ['-9999', '-999.000', '@', '-', '- ', '', ' ']
    df[value_col] = df[value_col].replace(na_values, np.nan)

    # Conversione sicura a float
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')

    # Imposta a NaN i valori negativi
    df.loc[df[value_col] < 0, value_col] = np.nan
    # Rimuove righe con date o valori mancanti
    df = df.dropna(subset=[date_col, value_col])

    # Aggrega a medie mensili se la risoluzione è giornaliera
    if df[date_col].dt.day.nunique() > 1:
        print("Risoluzione giornaliera rilevata: aggrego a medie mensili.")
        df = df.resample('ME', on=date_col)[value_col].mean().reset_index()

    # Estrai ts e m_cal
    ts = df[value_col].values
    m_cal = np.column_stack((df[date_col].dt.month, df[date_col].dt.year))

    # Messaggi di benvenuto
    print("#########################################################################")
    print("streamflow data has been imported successfully.")
    print(f"data starts from {m_cal[0]} and ends on {m_cal[-1]}.")
    print("#########################################################################")
    print("Run the following class methods to access key functionalities:\n")
    print(" >>> ._plot_scan(): to plot the sqiset heatmap and D_{SPI} \n ")
    print("*************** Alternatively, you can access to: \n >>> streamflow.ts (Q timeseries), \n >>> streamflow.spi_like_set (SQI (1:K) timeseries) \n >>> streamflow.SIDI (D_{SQI}) \n to visualize the data your way or proceed with further analyses!")

    return ts, m_cal


def era_snowfall_to_mm(DSO):
    """
    Convert monthly snowfall rate from ERA5 (in m/s) to mm/month using fixed month lengths.

    Parameters
    ----------
    snowfall_rate : np.ndarray
        Monthly mean snowfall rate (1D array) in m/s.
    m_cal : np.ndarray
        Calendar array of shape (N, 2), with month in column 0.

    Returns
    -------
    np.ndarray
        Total monthly snowfall in mm (same shape as input).
    """

    # Number of days in each month (non-leap year)
    days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31]
    months = np.arange(1,13)
    mlen = np.zeros(len(DSO.ts))*np.nan
    for i,m in enumerate(months):
        ii = np.where(DSO.m_cal[:,0]==m)[0]
        mlen[ii]=days_in_month[i]

    # Convert m/s to mm/month: m/s × 1000
    snowfall_mm = DSO.ts * mlen * 1000

    return snowfall_mm


