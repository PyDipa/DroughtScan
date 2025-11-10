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
import matplotlib.font_manager as fm
available = {f.name for f in fm.fontManager.ttflist}
if 'Helvetica' in available:
    mpl.rcParams['font.family'] = 'Helvetica'
else:
    mpl.rcParams['font.family'] = 'Arial'
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

def import_netcdf_for_cumulative_variable(file_path, possible_names,shape,verbose,cumulate=True):
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
            Pgrid = Pgrid.astype(float)


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
                            if cumulate:
                                monthly_sum = np.nansum(Pgrid[month_indices, :, :], axis=0)
                                Pgrid_m[yr_idx * 12 + month - 1, :, :] = monthly_sum
                            else:
                                monthly_mean = np.nanmean(Pgrid[month_indices, :, :], axis=0)
                                Pgrid_m[yr_idx * 12 + month - 1, :, :] = monthly_mean

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
                plt.show(block=False)

            # Aggregate precipitation timeseries over the basin
            if len(np.shape(Pgrid))>3: #forecast/multi members dataset
                ts = np.array(
                    [[np.nanmean(Pgrid[t, m, mask >= 0]) for m in range(Pgrid.shape[1])] for t in range(Pgrid.shape[0])])
            elif len(np.shape(Pgrid))==3:
                # ts = np.array([np.nanmean(Pgrid[i, mask >= 0]) for i in range(Pgrid.shape[0])])
                ts = np.nanmean(Pgrid[:, mask >= 0], axis=1)
                for t, p in enumerate(Pgrid):
                    p[np.isnan(mask)] = np.nan
                    Pgrid[t, :, :] = p

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

# ---------------------------
# Helpers for streamflow data
# ---------------------------


_DEF_NA = ['-9999', '-999.000', '@', '-', '- ', '', ' ', '--', 'NA', 'NaN', 'nan']

format_map = {
    "DD/MM/YY": "%d/%m/%y",
    "DD/MM/YYYY": "%d/%m/%Y",
    "D/M/YYYY":"%d/%m/%Y",
    "YYYY-MM-DD": "%Y-%m-%d",
    "YYYY/MM/DD": "%Y/%m/%d",
    "DD-MM-YYYY": "%d-%m-%Y",
    "YYYY.MM.DD": "%Y.%m.%d",
    "DD.MM.YYYY": "%d.%m.%Y",
    "YYYYMMDD": "%Y%m%d"
}

format_to_regex = {
    "YYYY-MM-DD": r'^\d{4}-\d{2}-\d{2}$',
    "YYYY/MM/DD": r'^\d{4}/\d{2}/\d{2}$',
    "DD-MM-YYYY": r'^\d{2}-\d{2}-\d{4}$',
    "DD/MM/YYYY": r'^\d{2}/\d{2}/\d{4}$',
    "D/M/YYYY": r'^\d{1,2}/\d{1,2}/\d{4}$',
    "DD/MM/YY": r'^\d{2}/\d{2}/\d{2}$',
    "YYYYMMDD": r'^\d{4}\d{2}\d{2}$',
    "DD MMM YYYY": r'^\d{2}\s\w{3}\s\d{4}$',  # e.g., 01 Dec 2023
    "MMM DD, YYYY": r'^\w{3}\s\d{2},\s\d{4}$',  # e.g., Dec 01, 2023
    "YYYY-DOY": r'^\d{4}-\d{3}$',  # Julian day, e.g., 2023-365
    "YYYY.MM.DD": r'^\d{4}\.\d{2}\.\d{2}$',
    "DD.MM.YYYY": r'^\d{2}\.\d{2}\.\d{4}$',
    "YYYY/MM": r'^\d{4}/\d{2}$',  # Year and month
    "HH:MM:SS": r'^\d{2}:\d{2}:\d{2}$',
    "ISO8601": r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$',  # ISO 8601
    "YYYY-MM-DD": r'^\d{4}-\d{2}-\d{2}$'  # e.g., 1960-12-31
}

def _detect_date_format(date_str):
    """
    Automatically detects the format of a text date
    by comparing it to a set of predefined regexes.
    Returns the format string compatible with strftime/pandas.
    """
    for fmt_key, pattern in format_to_regex.items():
        if re.match(pattern, date_str.strip()):
            return format_map.get(fmt_key)
    return None

def _check_datetime(text):
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


def _detect_delimiter(line):
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

def _normalize_colnames(df: pd.DataFrame):
    """
    Create a lowercase-to-original mapping of DataFrame column names.
    """
    return {str(c).strip().lower(): c for c in df.columns}

def _pick_col(df, candidates, contains=False):
    """
       Select the first column from a DataFrame that matches any candidate name.

       Args:
           df (pd.DataFrame): Input DataFrame.
           candidates (list): List of possible column name candidates.
           contains (bool): If True, match by substring instead of exact name.

       Returns:
           str or None: The matching column name from the DataFrame, or None if not found.
       """
    lower2orig = _normalize_colnames(df)
    if not contains:
        for k in candidates:
            k = str(k).strip().lower()
            if k in lower2orig:
                return lower2orig[k]
    else:
        for k in candidates:
            k = str(k).strip().lower()
            for lc, orig in lower2orig.items():
                if k in lc:
                    return orig
    return None

def _pick_date_col(df: pd.DataFrame) -> str:
    """
    Identify the name of the column containing date information.

    Logic:
        1. Check for common Italian/English date column names.
        2. Apply a first-row heuristic using `check_datetime()` or the presence of ':'.
        3. If split date columns (day/month/year) exist, create '_date' and return its name.
        4. Fallback: choose the column with at least 50% valid datetime parses.

    Returns:
        str: The name of the detected date column.

    Raises:
        ValueError: If no suitable date column can be identified.
    """
    lower2orig = _normalize_colnames(df)



    # 2) Singola colonna data (IT/EN)
    single_date_candidates = ['data', 'date', 'timestamp', 'datetime', 'time', 'data/ora', 'data ora']
    col = _pick_col(df, single_date_candidates, contains=False)
    if col is not None:
        return col

    # 3) Euristica "prima riga" (come nel tuo script originale)
    #    Esamina i primi valori della prima riga: se uno "sembra" una data, usa quella colonna.
    if len(df) > 0:
        row0 = df.iloc[0].to_numpy()
        n_try = min(30, len(row0))
        for i in range(n_try):
            try:
                v = str(row0[i]).strip()
                # taglia per evitare noise oltre la parte data
                v10 = v[:10]
                if _check_datetime(v10) or (":" in v):  # il tuo criterio ":" incluso
                    return df.columns[i]
            except Exception:
                pass

    # 4) Split columns (giorno/mese/anno) – crea '_date' e restituisci quel nome
    day_col   = _pick_col(df, ['giorno', 'gg', 'day', 'd'])
    month_col = _pick_col(df, ['mese', 'mm', 'month', 'm'])
    year_col  = _pick_col(df, ['anno', 'aaaa', 'yy', 'yyyy', 'year', 'y'])

    if year_col is not None and month_col is not None and day_col is not None:
        yy = pd.to_numeric(df[year_col], errors='coerce')
        mm = pd.to_numeric(df[month_col], errors='coerce')
        dd = pd.to_numeric(df[day_col], errors='coerce')
        df['_date'] = pd.to_datetime({'year': yy, 'month': mm, 'day': dd}, errors='coerce')
        return '_date'

    if year_col is not None and month_col is not None:
        yy = pd.to_numeric(df[year_col], errors='coerce')
        mm = pd.to_numeric(df[month_col], errors='coerce')
        per = pd.PeriodIndex.from_fields(year=yy, month=mm, freq='M')
        df['_date'] = per.to_timestamp(how='end')
        return '_date'

    # 5) Fallback brute-force: scegli la colonna con >=50% di datetime validi
    best_col, best_valid = None, 0.0
    for c in df.columns:
        s = pd.to_datetime(df[c], errors='coerce', dayfirst=True, utc=False)
        valid = s.notna().mean()
        if valid > best_valid and valid > 0.5:
            best_col, best_valid = c, valid
    if best_col is not None:
        return best_col

    raise ValueError("Impossibile determinare una colonna data (singola, split o euristica).")
def _date_related_cols(df: pd.DataFrame, date_col: str) -> set:
    """
    Return the set of columns related to date information that should be excluded
    when selecting the numeric value column.

    Includes:
        - The chosen date column (chosen_date_col)
        - Any '_date' column created by _pick_date_col
        - Split date components such as day/month/year in Italian or English
        - Common aliases for datetime fields

    Args:
        df (pd.DataFrame): The input dataframe.
        chosen_date_col (str or None): The name of the selected date column.

    Returns:
        set: Names of columns to exclude from numeric value detection.
    """
    rel = set()

    # Add the detected main date column
    if date_col and date_col in df.columns:
        rel.add(date_col)

    # Add the possible synthetic column created by _pick_date_col
    if '_date' in df.columns:
        rel.add('_date')

    # Typical split column names for day, month, year (IT/EN variants)
    day_keys   = ['giorno', 'gg', 'day', 'd']
    month_keys = ['mese', 'mm', 'month', 'm']
    year_keys  = ['anno', 'aaaa', 'yy', 'yyyy', 'year', 'y']

    # Map lowercase names to original columns for matching
    cols_lower = {str(c).strip().lower(): c for c in df.columns}

    def add_matches(keys):
        """Add any column whose name equals or clearly contains a keyword (word boundaries)."""
        for k in keys:
            pattern = r'\b' + re.escape(k) + r'\b'  # match whole word
            for low, orig in cols_lower.items():
                if re.search(pattern, low):
                    rel.add(orig)

    # Add split date components
    add_matches(day_keys)
    add_matches(month_keys)
    add_matches(year_keys)

    # Also include typical single-column aliases for datetime fields
    single_date_alias = ['data', 'date', 'timestamp', 'datetime', 'time', 'data/ora', 'data ora']
    add_matches(single_date_alias)

    return rel

def _pick_value_col(df: pd.DataFrame, exclude_cols=None) -> str:
    """
    Identify the name of the column containing numeric streamflow values.

    Logic:
        0. Exclude candidate columns passed via `exclude_cols` (es. data, giorno/mese/anno, '_date').
        1. Use the provided hint if valid (and not excluded).
        2. Match common IT/EN keywords for discharge/flow (and not excluded).
        3. Fallback: choose the column with the highest ratio of numeric (non-NaN) values (and not excluded).
        4. Last resort: first non-datetime column not excluded.

    Returns:
      str: The name of the detected value column.
    """
    lower2orig = _normalize_colnames(df)

    excl_set = set()
    if exclude_cols:
        # Normalizza contro i nomi originali del DF
        for ec in exclude_cols:
            if ec is None:
                continue
            ec_norm = str(ec).strip().lower()
            orig = lower2orig.get(ec_norm, ec)
            if orig in df.columns:
                excl_set.add(orig)

    # Heuristics su nomi IT/EN
    name_keys = [
        'portata', 'q', 'discharge', 'flow', 'flows', 'streamflow', 'm3/s', 'mc/s',
        'valore', 'value', 'values', 'qt', 'q_mean', 'qmean', 'portata_media'
    ]
    matches = []
    for key in name_keys:
        matches += [c for c in df.columns if key in str(c).lower() and c not in excl_set]
    if matches:
        return matches[0]

    # Fallback: colonna con massima densità numerica
    # --- 3) Fallback: colonna con massima densità numerica (skip escluse) ---
    candidates = []
    for c in df.columns:
        if c in excl_set:
            continue
        s = pd.to_numeric(df[c], errors='coerce')
        if s.notna().mean() > 0.5: #at least 50% of data mush be numeric and finite!
            candidates.append((c, s.notna().mean()))
    if candidates:
        return sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]

    return df.columns[0]

def _coerce_to_monthly(df, date_col, value_col, min_days=20):
    """
    Coerce daily streamflow data to monthly means, applying a minimum valid-days rule.

    If the time series is already monthly, it just pass over.
    Otherwise, it resamples daily data to monthly means, keeping months with at least
    `min_days` of valid (non-NaN) values.

    Args:
        df (pd.DataFrame): Input dataframe with date and value columns.
        date_col (str): Name of the datetime column.
        value_col (str): Name of the numeric value column.
        min_days (int): Minimum number of valid daily entries to accept a monthly mean.

    Returns:
        monthly_df (pd.DataFrame): Dataframe with '_date' and '_value' (monthly data).
        ts (np.ndarray): Numeric array of monthly values.
        m_cal (np.ndarray): Calendar array [[month, year], ...].
    """

    df = df.copy()
    try: #works for stings, not for timestamp
        format = _detect_date_format(df[date_col][0])
        df[date_col] = pd.to_datetime(df[date_col], format=format, errors="coerce")
    except AttributeError:
       df[date_col] = pd.to_datetime(df[date_col],errors="coerce")
    df = df.dropna(subset=[date_col, value_col])
    # df = df.sort_values(date_col)

    # Check frequency: daily if multiple distinct days per month
    is_daily = df[date_col].dt.day.nunique() > 4

    if is_daily:
        print(f"Daily resolution detected: aggregating to monthly means (min_valid_days={min_days}).")
        monthly = (
            df.resample('ME', on=date_col)[value_col]
            .agg(['mean', 'count'])  # mean = nanmean, count = non-NaN
            .rename(columns={'mean': value_col, 'count': 'valid_days'})
            .reset_index()
        )

        monthly.loc[monthly['valid_days'] < min_days, value_col] = pd.NA

    else:
        monthly = df.copy()
    # Build m_cal
    ts = monthly[value_col].values
    m_cal = np.column_stack((monthly[date_col].dt.month, monthly[date_col].dt.year))

    return monthly, ts, m_cal

# ---------------------------
# specific ingestion for CSV
# ---------------------------
def _read_csv_smart(file_path: str) -> pd.DataFrame:
    """
      Read a CSV file with automatic detection of header, footer, and delimiter.

      Behavior:
          - Scans the file to estimate the number of header and footer lines to skip.
          - Detects the most likely column delimiter (tab, comma, semicolon, etc.).
          - Handles comment lines and irregular structures.
          - Automatically retries parsing if decimal commas are used.
          - Cleans up empty columns and returns a valid DataFrame.

      Args:
          file_path (str): Path to the CSV file.

      Returns:
          pd.DataFrame: Parsed and cleaned DataFrame ready for further processing.

      Raises:
          FileNotFoundError: If the file path does not exist.
      """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File non trovato: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # stima skip_rows cercando la prima "riga tabellare"


    skip_rows = 0  # Contatore righe da saltare
    while skip_rows < len(lines):

        # explore data getting the first 5 rows starting from skip_rows
        test_block = lines[skip_rows:skip_rows + 5]

        # Try separating columns using a space, comma or semicolon.
        split_lines = [line.strip().replace(",", ".").split() for line in test_block]

        # Convert the first two columns to numeric arrays
        split_lines = np.array(split_lines, dtype=object)
        # print(f'skip_rows = {skip_rows}')
        if split_lines.ndim > 1:
            if (skip_rows == 0) | (skip_rows == 1):
                skip_rows = 0
            delimiter = _detect_delimiter(lines[skip_rows + 1])
            break
        else:
            skip_rows += 1

    # !tail -n 20 file_path
        sospetti = ('#', '@', '--', '//')  # aggiungi qui altri prefissi sospetti
        if lines[skip_rows].lstrip().startswith(sospetti):
            skip_rows=skip_rows+1

    end_row = 1 if skip_rows == 0 else skip_rows
    while end_row < len(lines):

        # explore data getting the first 5 rows starting from skip_rows
        test_block = lines[end_row:end_row + 5]

        # Try separating columns using a space, comma or semicolon.
        split_lines = [line.strip().replace(",", ".").split() for line in test_block]

        # Convert the first two columns to numeric arrays
        split_lines = np.array(split_lines, dtype=object)
        # print(f'skip_rows = {skip_rows}')
        if split_lines.ndim == 1:
            break
        else:
            end_row += 1

    if len(lines) - end_row == 0:  # no skipfooter
        skip_footer = 0
    elif len(lines) - end_row > 0:
        skip_footer = len(lines) - end_row - skip_rows
    else:
        print('check the footer of the csv file')

    # primo tentativo
    try:
        df = pd.read_csv(
            file_path,
            encoding="utf-8",
            encoding_errors="ignore",
            delimiter=delimiter,
            skiprows=skip_rows,
            skipfooter=skip_footer,
            na_values=_DEF_NA,
            engine='python',
            index_col=False,
            header=0
        )
    except Exception:
        # fallback minimale
        df = pd.read_csv(
            file_path,
            encoding="utf-8",
            encoding_errors="ignore",
            delimiter=delimiter,
            skiprows=skip_rows,
            na_values=_DEF_NA,
            engine='python'
        )



    # test the value column
    test_val_col = _pick_value_col(df)
    try:
        _ = pd.to_numeric(df[test_val_col], errors='raise')

    except Exception: # if the selected value column is not numeric due to commas, retry with decimal=','
        try:
            df = pd.read_csv(
                file_path,
                encoding="utf-8",
                encoding_errors="ignore",
                delimiter=delimiter,
                skiprows=skip_rows,
                na_values=_DEF_NA,
                engine='python',
                decimal=','
            )
        except Exception:
            pass

    # drop colonne completamente vuote
    df = df.dropna(axis=1, how='all')
    return df


def _pipeline_common(df, origin_label=""):
    """
      Standard pipeline for cleaning, detecting, and converting streamflow data to monthly format.

      Steps:
          1. Detect the date and value columns automatically (unless specified).
          2. Convert sub-monthly or irregular monthly data into a consistent monthly series.
          3. Print summary information about the imported dataset.

      Args:
          df (pd.DataFrame): Input DataFrame containing raw streamflow data.
          date_col (str, optional): Name of the date column (auto-detected if None).
          value_col (str, optional): Name of the value column (auto-detected if None).
          origin_label (str): Optional label for identifying the data source in the log output.

      Returns:
          tuple:
              - ts (np.ndarray): Monthly streamflow time series.
              - m_cal (np.ndarray): Corresponding calendar array [month, year].
      """
    date_col_name = _pick_date_col(df)
    exclude = _date_related_cols(df, date_col_name)
    value_col_name = _pick_value_col(df, exclude_cols=exclude)


    monthly_df, ts, m_cal = _coerce_to_monthly( df, date_col_name, value_col_name)



    # 4) banner
    print("#########################################################################")
    print(f"streamflow data has been imported successfully ({origin_label}).")
    print(f"data starts from [{m_cal[0,0]} {m_cal[0,1]}] and ends on [{m_cal[-1,0]} {m_cal[-1,1]}].")
    print("#########################################################################")

    return ts, m_cal

# ---------------------------
# Unified API  + wrapper
# ---------------------------
def load_streamflow(file_path):
    """
    Robust loader for streamflow data from CSV or Excel files.

    Automatically detects the file type and applies a common cleaning and
    conversion pipeline to produce a monthly time series.

    Args:
        file_path (str): Path to the input file (.csv, .txt, .xlsx, .xls).

    Returns:
        tuple:
            - ts (np.ndarray): Monthly streamflow time series.
            - m_cal (np.ndarray): Corresponding calendar array [month, year].

    Raises:
        ValueError: If the file extension is not recognized.
        FileNotFoundError: If the specified file does not exist.
    """

    ext = os.path.splitext(file_path)[1].lower()
    if ext in ('.csv', '.txt'):
        df = _read_csv_smart(file_path)
        return _pipeline_common(df, origin_label="csv")
    elif ext in ('.xlsx', '.xls'):
        df = pd.read_excel(file_path)
        df = df.dropna(axis=1, how='all')
        return _pipeline_common(df, origin_label="Excel")
    else:
        raise ValueError(f"Estensione non riconosciuta: {ext}")

#--------------------------------------
# Teleindex
# ----------------------------------------

def get_teleindex_info(data_path):
    data = nc.Dataset(data_path)
    var_name  = []
    for v in data.variables:
        v.append(var_name)
        var = data.variables[var_name]

        unit = getattr(var, "units", "N/A")
        shape = var.shape

        if var_name == "time":
            time_units = getattr(var, "units", "N/A")  # Es: "months since 1900-01-01"
            if "months since" in time_units:
                #get info about the timespan
                base_date_str = time_units.split("since")[1].strip()  # "1900-01-01"
                base_date = datetime.strptime(base_date_str[0:10], "%Y-%m-%d")

                time_values = [base_date + relativedelta(months=int(t)) for t in var[:]]
                starting_date, ending_date = time_values[0], time_values[-1]
    return var_name,starting_date, ending_date
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




# OLD WORKING SCRIPT TO INGEST CVS and EXCEL
# def load_streamflow_from_csv(file_path, date_col=None, value_col=None):
#         """
#         Load and process streamflow data from a csv file.
#
#         This method automatically detects the file format, delimiter, and column structure,
#         then processes the data to create a usable streamflow time series. If the data is
#         daily, it aggregates it to monthly averages.
#
#         Args:
#             file_path (str): Path to the streamflow data file.
#             date_col (str, optional): Name of the column containing date values. If None, it is auto-detected.
#             value_col (str, optional): Name of the column containing streamflow values. If None, it is auto-detected.
#
#         Returns:
#             None: Updates `self.ts` and `self.m_cal` attributes, and recalculates derived attributes.
#
#         """
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"The file '{file_path}' does not exist.")
#
#         with open(file_path, "r", encoding="utf-8", errors="replace") as file:
#             lines = file.readlines()  # Legge tutte le righe del file
#
#         skip_rows = 0  # Contatore righe da saltare
#         while skip_rows < len(lines):
#
#             # explore data getting the first 5 rows starting from skip_rows
#             test_block = lines[skip_rows:skip_rows + 5]
#
#             # Try separating columns using a space, comma or semicolon.
#             split_lines = [line.strip().replace(",", ".").split() for line in test_block]
#
#             # Convert the first two columns to numeric arrays
#             split_lines = np.array(split_lines,dtype=object)
#             # print(f'skip_rows = {skip_rows}')
#             if split_lines.ndim>1:
#                 if (skip_rows==0) | (skip_rows==1):
#                     skip_rows=0
#                 delimiter = detect_delimiter(lines[skip_rows + 1])
#                 break
#             else:
#                 skip_rows += 1
#
#         # !tail -n 20 file_path
#
#         end_row = 1 if skip_rows==0 else skip_rows
#         while end_row < len(lines):
#
#             # explore data getting the first 5 rows starting from skip_rows
#             test_block = lines[end_row:end_row + 5]
#
#             # Try separating columns using a space, comma or semicolon.
#             split_lines = [line.strip().replace(",", ".").split() for line in test_block]
#
#             # Convert the first two columns to numeric arrays
#             split_lines = np.array(split_lines,dtype=object)
#             # print(f'skip_rows = {skip_rows}')
#             if split_lines.ndim==1:
#                 break
#             else:
#                 end_row += 1
#
#         if len(lines)-end_row==0: #no skipfooter
#             skip_footer = 0
#         elif len(lines)-end_row>0:
#             skip_footer = len(lines)-end_row-skip_rows
#         else:
#             print('check the footer of the csv file')
#
#         sospetti = ('#', '@', '--', '//')  # aggiungi qui altri prefissi sospetti
#         if lines[skip_rows].lstrip().startswith(sospetti):
#             skip_rows = skip_rows+1
#
#         df = pd.read_csv(
#             file_path,encoding="utf-8",
#             encoding_errors="ignore",
#             delimiter=delimiter,
#             skiprows=skip_rows,
#             skipfooter=skip_footer,
#             na_values=['-9999', '-999.000', '@'],
#             engine='python',
#             index_col=False,
#             header=0
#         )
#
#         # Remove extra spaces from column names
#         # df.columns = df.columns.str.strip()
#         # remove any potential colums of only nan
#         df = df.dropna(axis=1, how='all')
#
#         df_example =  df.iloc[0].to_numpy()
#
#         # Auto-detect date and value columns if not provided
#
#         for col in range(30):
#             try:
#                 check = check_datetime(df_example[col][0:10])
#                 if date_col is None and (check == True or ":" in df_example[col]):
#                     date_column = df_example[col]
#                     date_col = col
#
#             except IndexError:
#                 pass
#
#         # for col in range(5):
#         #     try:
#         #         check = check_datetime(df_example[col][0:10])
#         #         if value_col is None and (check ==False and ":" not in df_example[col]):
#         #             value_column = df_example[col]
#         #     except IndexError:
#         #         pass
#         for col in range(30):
#             try:
#                 check = check_datetime(df_example[col][0:10])
#                 pass
#             except IndexError:
#                 try:
#                     df_example[col]
#                     if value_col is None:
#                         value_column = df_example[col]
#                         value_col = col
#                 except IndexError:
#                     pass
#
#         value_col, date_col = None,None
#         if value_col is None:
#             value_col = [col for col in df.columns if df[col].eq(float(value_column)).any()][0]
#         if date_col is None:
#             date_col = [col for col in df.columns if df[col].eq(date_column).any()][0]
#
#         try:
#             df[value_col].mean()
#         except TypeError:
#             decimal = ','
#             df = pd.read_csv(
#                 file_path, encoding="utf-8", encoding_errors="ignore",
#                 delimiter=delimiter,
#                 skiprows=skip_rows,
#                 na_values=['-9999', '-999.000', '@'],
#                 engine='python',
#                 decimal=decimal,
#             )
#
#         # remove any potential colums of only nan
#         df = df.dropna(axis=1, how='all')
#
#         # Convert date column to datetime
#         df[date_col] = pd.to_datetime(df[date_col])
#         df = df.dropna(subset=[date_col, value_col])
#
#         # Aggregate daily data to monthly averages if needed
#         if df[date_col].dt.day.unique().size > 4:
#             print("Risoluzione giornaliera rilevata: aggrego a medie mensili.")
#             # df = df.resample('ME', on=date_col)[value_col].mean().reset_index()
#
#             min_days = 20
#             monthly = (
#                 df.resample('ME', on=date_col)[value_col]
#                 .agg(['mean', 'count'])  # mean = nanmean, count = non-NaN
#                 .rename(columns={'mean': value_col, 'count': 'valid_days'})
#                 .reset_index()
#             )
#
#             monthly.loc[monthly['valid_days'] < min_days, value_col] = pd.NA
#
#             df = monthly.copy()
#
#         # Update class attributes
#
#         ts = df[value_col].values
#         m_cal = np.column_stack((df[date_col].dt.month, df[date_col].dt.year))
#
#         # Welcome and guidance messages
#         print("#########################################################################")
#         print("streamflow data has been imported successfully.")
#         print(f"data starts from {m_cal[0]} and ends on {m_cal[-1]}.")
#         print("#########################################################################")
#         print("Run the following class methods to access key functionalities:\n")
#         print(" >>> ._plot_scan(): to plot the sqiset heatmap and D_{SPI} \n ")
#         print("*************** Alternatively, you can access to: \n >>> streamflow.ts (Q timeseries), \n >>> streamflow.spi_like_set (SQI (1:K) timeseries) \n >>> streamflow.SIDI (D_{SQI}) \n to visualize the data your way or proceed with further analyses!")
#
#         return ts,m_cal
#
# def load_streamflow_from_excel(file_path, date_col=None, value_col= None):
#     """
#     Load and process streamflow data from an Excel file in a robust way.
#     If data have daily resolition the average monthly mean is computed only if  a minum number of 20 days are finite real values
#
#     - Detects dates either in a single column (including datetime types) or in split columns (day/month/year).
#     - Accepts Italian/English column names, case-insensitive.
#     - Cleans values (decimal commas, sentinel values, negatives) and aggregates daily data to monthly means.
#     - Applies a minimum valid-days rule for monthly means (hard minimum = 20 days).
#
#     Args:
#         file_path: Path to the .xlsx/.xls file. Data must be in the first sheet
#         date_col: Name of the date column; if None, it will be auto-detected.
#         value_col: Name of the value (discharge/flow) column; if None, it will be auto-detected.
#
#
#     Returns:
#         (ts, m_cal)
#         ts: numpy array of monthly values (float, NaN allowed).
#         m_cal: numpy array with shape (n, 2) containing [month, year] per row.
#     """
#     df = pd.read_excel(file_path)
#
#     # Normalize column names for robust matching
#     lower2orig = {str(c).strip().lower(): c for c in df.columns}
#
#     def pick_col(candidates):
#         """Return the first existing column (original name) matching any lower-cased candidate."""
#         for k in candidates:
#             if k in lower2orig:
#                 return lower2orig[k]
#         return None
#
#     # --- Date detection ---
#     # If date_col is provided, try case-insensitive matching; else try single or split columns.
#     date_col_orig = None
#     if date_col is not None:
#         date_col_orig = lower2orig.get(str(date_col).strip().lower(), date_col if date_col in df.columns else None)
#
#     # Split columns candidates (IT/EN, short forms included)
#     day_col   = pick_col(['giorno', 'gg', 'day', 'd'])
#     month_col = pick_col(['mese', 'mm', 'month', 'm'])
#     year_col  = pick_col(['anno', 'aaaa', 'yy', 'yyyy', 'year', 'y'])
#
#     # Single date column candidates (IT/EN)
#     single_date_candidates = ['data', 'date', 'timestamp', 'datetime', 'time', 'data/ora', 'data ora']
#     if date_col_orig is None:
#         date_col_orig = pick_col(single_date_candidates)
#
#     # Build the _date column
#     if date_col_orig is not None:
#         # Direct parsing from a single date column
#         df['_date'] = pd.to_datetime(df[date_col_orig], errors='coerce', utc=False, dayfirst=True)
#         if df['_date'].isna().mean() > 0.5:
#             # Retry with dayfirst=False if too many NaT
#             df['_date'] = pd.to_datetime(df[date_col_orig], errors='coerce', utc=False, dayfirst=False)
#     elif all(c is not None for c in (day_col, month_col, year_col)):
#         # Combine split columns into a proper datetime
#         dd = pd.to_numeric(df[day_col], errors='coerce')
#         mm = pd.to_numeric(df[month_col], errors='coerce')
#         yy = pd.to_numeric(df[year_col], errors='coerce')
#
#         df['_date'] = pd.to_datetime({'year': yy, 'month': mm, 'day': dd}, errors='coerce')
#         # If the day is missing but month and year exist, default to day=1
#         mask_day_missing = dd.isna() & mm.notna() & yy.notna()
#         if mask_day_missing.any():
#             df.loc[mask_day_missing, '_date'] = pd.to_datetime(
#                 {'year': yy[mask_day_missing], 'month': mm[mask_day_missing], 'day': 1},
#                 errors='coerce'
#             )
#     else:
#         # Fallback: try to_datetime on every column and pick the best candidate
#         best_col, best_valid = None, 0.0
#         for c in df.columns:
#             s = pd.to_datetime(df[c], errors='coerce', utc=False, dayfirst=True)
#             valid = s.notna().mean()
#             if valid > best_valid and valid > 0.5:
#                 best_col, best_valid = c, valid
#                 df['_date'] = s
#         if best_col is None:
#             raise ValueError("Unable to determine a date column (single or split).")
#
#     # --- Value detection ---
#     value_col_orig = None
#     if value_col is not None:
#         value_col_orig = lower2orig.get(str(value_col).strip().lower(), value_col if value_col in df.columns else None)
#
#     if value_col_orig is None:
#         # Name-based heuristics (IT/EN)
#         value_name_candidates = [
#             'portata', 'q', 'discharge', 'flow', 'flows', 'streamflow', 'm3/s', 'mc/s',
#             'valore', 'value', 'values', 'qt', 'q_mean', 'qmean', 'portata_media'
#         ]
#         # Match by inclusion (e.g., "Q (m3/s)" contains "m3/s")
#         matches = []
#         for key in value_name_candidates:
#             matches += [c for c in df.columns if key in str(c).lower()]
#         if matches:
#             value_col_orig = matches[0]
#
#     if value_col_orig is None:
#         # Type-based fallback: pick the column that converts best to numeric
#         candidates = []
#         for c in df.columns:
#             if c == '_date' or c in {day_col, month_col, year_col}:
#                 continue
#             s = pd.to_numeric(df[c], errors='coerce')
#             if s.notna().mean() > 0.8:
#                 candidates.append((c, s.notna().mean()))
#         if not candidates:
#             # As a last resort, take the first non-date column
#             candidates = [(c, 0.0) for c in df.columns if c != '_date']
#         value_col_orig = sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]
#
#     # --- Value cleaning ---
#     # Convert to string for normalization, handle decimal commas and sentinels
#     na_values = ['-9999', '-999.000', '@', '-', '- ', '', ' ', '--', 'NA', 'NaN', 'nan']
#     s = df[value_col_orig].astype(str).str.strip()
#     s = s.replace(na_values, np.nan)
#     s = s.str.replace(',', '.', regex=False)
#     s = pd.to_numeric(s, errors='coerce')
#     # Disallow negative values for streamflow
#     s = s.mask(s < 0, np.nan)
#     df['_value'] = s
#
#     # Drop colums of only nan
#     df = df.dropna(axis=1, how='all')
#     # Remove timezone if present
#     if hasattr(df['_date'].dt, 'tz'):
#         try:
#             df['_date'] = df['_date'].dt.tz_localize(None)
#         except Exception:
#             pass
#
#     # --- Monthly aggregation ---
#     # If multiple distinct days exist, we consider it daily and aggregate to monthly means
#     is_daily = df['_date'].dt.day.nunique() > 1
#
#     if is_daily:
#         min_valid_days = 20
#         print(f"Daily resolution detected: aggregating to monthly means (min_valid_days={min_valid_days}).")
#         grouped = df.set_index('_date')['_value']
#         monthly_mean = grouped.resample('ME').mean()
#         valid_days = grouped.resample('ME').count()
#         monthly = pd.DataFrame({'_value': monthly_mean, 'valid_days': valid_days}).reset_index()
#         monthly.loc[monthly['valid_days'] < min_valid_days, '_value'] = pd.NA
#         out_dates = monthly['_date']
#         out_values = monthly['_value'].astype(float).to_numpy()
#     else:
#         # Already monthly (or coarser): align to month-end timestamps for consistency
#         out_dates = df['_date'].dt.to_period('M').dt.to_timestamp('M')
#         out_values = df['_value'].astype(float).to_numpy()
#
#     # --- Build m_cal and return
#     m_month = out_dates.dt.month.to_numpy()
#     m_year = out_dates.dt.year.to_numpy()
#     m_cal = np.column_stack((m_month, m_year))
#
#     print("#########################################################################")
#     print("streamflow data has been imported successfully (Excel).")
#     print(f"data starts from [{m_cal[0,0]:02d} {m_cal[0,1]}] and ends on [{m_cal[-1,0]:02d} {m_cal[-1,1]}].")
#     print("#########################################################################")
#     print(" >>> ._plot_scan(): to plot the sqiset heatmap and D_{SPI}")
#     print(" >>> streamflow.ts (Q timeseries), streamflow.spi_like_set, streamflow.SIDI")
#
#     return out_values, m_cal




