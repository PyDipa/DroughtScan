"""
author: PyDipa

Drought Indices Computation.

This module implements calculations for various **drought indices**, including:
- **SPI (Standardized Precipitation Index)**.
- **SIDI (Standardized Integrated Drought Index)**.
- **CDN (Cumulative Deviation from Normal)**.

Main functions:
- `_calculate_spi_like_set()`: Computes SPI-based indices from precipitation data.
- `_calculate_SIDI()`: Computes SIDI using weighted drought indicators.
- `_calculate_CDN()`: Computes CDN for hydrological drought monitoring.

Used by: `core.py`, `ESM_scenarios.py`, `hydrology.py`.
"""



import numpy as np
from scipy.stats import gamma, pearson3, norm
import warnings


# FOR THE STANDARDIEZED INTEGRATED PRECIPITATION INDEX (SIDI)
def weighted_metrics(values, weights):
    """
    Compute the weighted average and weighted standard deviation of a dataset.

    Args:
        values (numpy.ndarray): 1D array of data values for which the weighted metrics are to be calculated.
        weights (numpy.ndarray): 1D array of weights corresponding to the values. Must have the same shape as `values`.

    Returns:
        tuple: (weighted_average, weighted_std)
    """
    if values.shape != weights.shape:
        raise ValueError("Values and weights must have the same shape.")

    if np.sum(weights) == 0:
        raise ValueError("The sum of the weights must be greater than zero.")

    # Weighted average
    weighted_average = np.average(values, weights=weights)

    # Weighted variance (squared standard deviation)
    weighted_variance = np.average((values - weighted_average) ** 2, weights=weights)

    # Return the weighted average and the weighted standard deviation
    return weighted_average, np.sqrt(weighted_variance)

def generate_weights(k):
    """
    Generate weight matrices for SIDI computation.

    Args:
        k (int, optional): Number of temporal scales to consider. Defaults to self.K.

    Returns:
        ndarray: Weight matrix of shape (k, 5), where each column represents a different weight distribution:
            - Column 0: Uniform weights
            - Column 1: Inverted linear weights
            - Column 2: Inverted geometric weights
            - Column 3: Linear weights
            - Column 4: Geometric weights
        """

    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError("`k` must be a positive integer.")

    # Generate weights
    geom_weights = np.geomspace(1, k, k)
    linear_weights = np.linspace(1, k, k)

    return np.vstack([
        np.tile(1 / k, k),  # Uniform weights
        np.flipud(linear_weights) / np.sum(linear_weights),  # Inverted linear weights
        np.flipud(geom_weights) / np.sum(geom_weights),  # Inverted geometric weights
        linear_weights / np.sum(linear_weights),  # Linear weights
        geom_weights / np.sum(geom_weights)  # Geometric weights
    ]).T
# # QUI STORO LE CALIBRAZIONI DI GAMMA E PEARSON3 per lA ricostruzione dello SPI e SPEI

def baseline_indices(m_cal,start_baseline_year,end_baseline_year):
    """
    Get indices for the baseline period based on start and end years.

    Returns:
        tuple: Indices for the start and end of the baseline period.
    """
    # Find indices for the start and end years
    start_indices = np.where(m_cal[:, 1] == start_baseline_year)[0]
    end_indices = np.where(m_cal[:, 1] ==end_baseline_year)[0]

    if len(start_indices) == 0:
        raise ValueError(f"Start baseline year {start_baseline_year} not found in `m_cal`.")
    if len(end_indices) == 0:
        raise ValueError(f"End baseline year {end_baseline_year} not found in `m_cal`.")

    tb1_id = start_indices[0]
    tb2_id = end_indices[-1]

    if tb1_id > tb2_id:
        raise ValueError("Inconsistent baseline indices: start index is after end index.")

    return tb1_id, tb2_id

def get_month_indices(month, start_year, end_year, m_cal):
    """
    Returns the indices of m_cal where the specified month is present for the given start-year - end-year .

    Args:
        month (int): The month to search for (1 = January, ..., 12 = December).
        start_year (int): The starting year for the search.
        end_year (int): The ending year for the search.
        m_cal (numpy.ndarray): The calendar array with two columns:
            - Column 0: Month (1-12)
            - Column 1: Year.

    Returns:
        numpy.ndarray: Indices of the months matching the specified criteria.
    """

    try:
        idx = np.array([
            np.where((m_cal[:, 0] == month) & (m_cal[:, 1] == year))[0][0]
            for year in range(start_year, end_year + 1)
        ])
    except IndexError:
        raise ValueError(f"Inconsistent baseline years: define a period within the data temporal domain: {m_cal[0]} - {m_cal[-1]}.")
        # print(f"************ BASELINE WARNING **********')")
        # print(f"Inconsistent baseline years: baseline must be whitin the  {m_cal[0]} - {m_cal[-1]} domain.")
    return idx


# FOR ONLY POSITIVE & RIGHT-SKEWED DATA: (using a Gamma Function)
# ===================================================================
#  SPI Computation
# ===================================================================
def f_spi(prec,stride,m,m_cal, tb1,tb2,gamma_params=None):
    """
    Calculate the Standardized Precipitation Index (SPI) or other SPI-like indices
    (e.g., Standardized Streamflow Index (SQI)) for a time series of monthly data.

    Args:
        prec (numpy.ndarray):
            A one-dimensional array of monthly data (e.g., precipitation or streamflow) with size n (number of months).
        stride (int):
            The length of the SPI accumulation period (temporal frame), e.g., 18 for SPI18.
        m (int):
            The reference month (1-12), where 1 = January, 2 = February, etc.
        m_cal (numpy.ndarray):
            A calendar array corresponding to the time series, with two columns:
            - Column 0: Month (1-12)
            - Column 1: Year.
        tb1 (int):
            The starting year for the baseline period.
        tb2 (int):
            The ending year for the baseline period.

    Returns:
        numpy.ndarray: Indices of the months considered for the SPI calculation.
        numpy.ndarray: The calculated SPI values for each month.
        numpy.ndarray: Coefficients of the polynomial fit used for reversing SPI to precipitation.
        list: Estimated precipitation values corresponding to SPI values of -1.0, -1.5, and -2.0.


    Notes:
    ------
    - The function computes the baseline SPI values using the specified reference month and
      accumulation period.
    - If the accumulation period (stride) is set to 1, the calculation is straightforward,
      utilizing all available data for the specified month.
    - If the accumulation period is greater than 1, the function aggregates precipitation values
      over the defined periods.
    - The SPI is computed based on a fitted gamma distribution to the baseline precipitation data,
      with results transformed into a normal distribution.
    - The output includes coefficients for reversing the SPI to estimate corresponding precipitation values.
    """


    # Extract start and end years from the calendar
    t1 = int(m_cal[0, 1])  # Start year
    t2 = int(m_cal[-1, 1])  # End year
    anni = np.unique(m_cal[:, 1]).astype(int)  # Unique years in the calendar
    try:
        tb1_id = np.where(anni == tb1)[0][0]  # Index of the baseline start year
        tb2_id = np.where(anni == tb2)[0][0]  # Index of the baseline end year
    except IndexError:
        print('some inconsistency arise:  ')
        print('maybe nconsistent baseline years: define a period within the data temporal domain...')
        print('maybe inconsistent calendar: check whether some years or months miss in the calendar...')
    # 1) Define the baseline period (e.g., 30 or 40 years)
    # Function to obtain the indices of specific months within a time range


    # -------- BASELINE ------------------------------------------------
    if stride == 1:
        # SPI at 1: directly use all data
        idmesi_base = get_month_indices(m, tb1, tb2, m_cal)
        xbase = prec[idmesi_base]  # Precipitation for the baseline period

        # Get all months for the entire time period
        try:
            idmesi_all = get_month_indices(m, t1, t2, m_cal)
        except (IndexError, ValueError):
            try:
                idmesi_all = get_month_indices(m, t1, t2 - 1, m_cal)
            except (IndexError, ValueError):
                try:
                    idmesi_all = get_month_indices(m, t1+1, t2, m_cal)
                except (IndexError, ValueError):
                    idmesi_all = get_month_indices(m, t1 + 1, t2-1, m_cal)


        x = prec[idmesi_all]  # Precipitation for the entire period

    else:
        # SPI with accumulation period greater than 1
        # Calculate the baseline
        idmesi_base = get_month_indices(m, tb1, tb2, m_cal)
        a = np.array(
            [idmesi_base - j for j in np.flip(np.arange(0, stride))]).T  # Create the matrix of months to select
        xbase = np.array([np.sum(prec[row]) if np.all(row >= 0) else np.nan for row in a])

        # WHOLE PERIOD ----------------------------------------------
        # Get all months for the entire time period
        try:
            idmesi_all = get_month_indices(m, t1, t2, m_cal)
        except (IndexError, ValueError):
            try:
                idmesi_all = get_month_indices(m, t1, t2 - 1, m_cal)
            except (IndexError, ValueError):
                try:
                    idmesi_all = get_month_indices(m, t1+1, t2, m_cal)
                except (IndexError, ValueError):
                    idmesi_all = get_month_indices(m, t1 + 1, t2-1, m_cal)

        a = np.array([idmesi_all - j for j in np.flip(np.arange(0, stride))]).T  # Create the matrix of months to select
        x = np.array(
            [np.sum(prec[row]) if np.all(row >= 0) else np.nan for row in a])  # Precipitation for the entire period

    # --------------------------- SPI -----------------------------------------
    # Calculate the monthly balance for the baseline and the entire period
    # Start with SPI: pre-allocate and calculate gamma distribution parameters
    spi = np.empty(np.shape(x))
    spi[:] = np.nan
    if gamma_params is None:
        alpha, loc, beta = gamma.fit(xbase[xbase > 0], floc=0)
    else:
        alpha, loc, beta = gamma_params

    # Use the gamma distribution
    Gx = gamma.cdf(x, a=alpha, loc=loc, scale=beta)
    # Calculate the proportion of zero values
    qq = len(np.where(x == 0)[0]) / len(x)
    Hx = qq + (1 - qq) * Gx
    Hx = np.clip(Hx, 3.17e-5, 1 - 3.17e-5) #+-4 allowed

    spi  = np.round(norm.ppf(Hx),4)

    # *********** REVERSE FROM SPI TO PRECIPITATION *************************
    spibase = spi[tb1_id:tb2_id + 1]
    coef = np.polyfit(spibase[np.isfinite(xbase)], xbase[np.isfinite(xbase)], deg=3)
    # Coefficients from a 5th-degree polynomial fit can be used to reverse SPI:
    # Example:
    # values = np.array([np.polyval(coef, spi[i]) for i in range(len(spi))])
    # ***************************************************************
    # After calculating SPI for month m, insert the processed values into the full SPI array
    # Spi[idmesi_all] = spi

    return idmesi_all, spi, coef,(alpha, loc, beta) if gamma_params is None else None

# FOR REAL VALUES & RIGHT-SKEWED (using Pearson III function)
# ===================================================================
#  SPEI Computation
# ===================================================================
def f_spei(balance, stride, m, m_cal, tb1, tb2,gamma_params= None):
    """
    Calculate the Standardized Precipitation Evapotranspiration Index (SPEI)
    using precipitation and PET time series.

    Args:
        balance(np.ndarray):
            Monthly precipitation-evapotrampiration time series, dimension (n,).

        stride (int):
            Accumulation period for SPEI (e.g., 18 for SPEI18).
        m (int):
            Reference month (1-12), where 1 = January, ..., 12 = December.
        m_cal (np.ndarray):
            Calendar array corresponding to the time series, shape (n, 2):
            - Column 0: Month (1-12)
            - Column 1: Year.
        tb1 (int):
            Starting year for the baseline period.
        tb2 (int):
            Ending year for the baseline period.

    Returns:
        tuple:
            - idmesi_all (np.ndarray): Indices of the months for the entire period.
            - spei (np.ndarray): Calculated SPEI values.
            - coef (np.ndarray): Polynomial coefficients to reverse SPEI to D.

    Raises:
        ValueError: If the input data or indices are inconsistent.
    """


    # t1, t2 == start and end year of the time period
    # Extract start and end years from the calendar
    t1 = int(m_cal[0, 1])  # Start year
    t2 = int(m_cal[-1, 1])  # End year
    anni = np.unique(m_cal[:, 1]).astype(int)  # Unique years in the calendar
    try:
        tb1_id = np.where(anni == tb1)[0][0]  # Index of the baseline start year
        tb2_id = np.where(anni == tb2)[0][0]  # Index of the baseline end year
    except IndexError:
        print('Inconsistent baseline years: define a period within the data temporal domain')


    # -------- BASELINE ------------------------------------------------
    if stride == 1:
        # SPI at 1: directly use all data
        idmesi_base = get_month_indices(m, tb1, tb2, m_cal)
        xbase = balance[idmesi_base]  # Precipitation for the baseline period

        # Get all months for the entire time period
        try:
            idmesi_all = get_month_indices(m, t1, t2, m_cal)
        except (IndexError, ValueError):
            try:
                idmesi_all = get_month_indices(m, t1, t2 - 1, m_cal)
            except (IndexError, ValueError):
                try:
                    idmesi_all = get_month_indices(m, t1+1, t2, m_cal)
                except (IndexError, ValueError):
                    idmesi_all = get_month_indices(m, t1 + 1, t2-1, m_cal)


        x = balance[idmesi_all]  # Precipitation for the entire period

    else:
        # SPI with accumulation period greater than 1
        # Calculate the baseline
        idmesi_base = get_month_indices(m, tb1, tb2, m_cal)
        a = np.array(
            [idmesi_base - j for j in np.flip(np.arange(0, stride))]).T  # Create the matrix of months to select
        xbase = np.array([np.sum(balance[row]) if np.all(row >= 0) else np.nan for row in a])

        # WHOLE PERIOD ----------------------------------------------
        # Get all months for the entire time period
        try:
            idmesi_all = get_month_indices(m, t1, t2, m_cal)
        except (IndexError, ValueError):
            try:
                idmesi_all = get_month_indices(m, t1, t2 - 1, m_cal)
            except (IndexError, ValueError):
                try:
                    idmesi_all = get_month_indices(m, t1+1, t2, m_cal)
                except (IndexError, ValueError):
                    idmesi_all = get_month_indices(m, t1 + 1, t2-1, m_cal)

        a = np.array([idmesi_all - j for j in np.flip(np.arange(0, stride))]).T  # Create the matrix of months to select
        x = np.array(
            [np.sum(balance[row]) if np.all(row >= 0) else np.nan for row in a])  # Precipitation for the entire period

    # ------------------------------ SPEI Calculation --------------------------------------
    # Fit Pearson distribution for Dbase
    if gamma_params is None:
        c, loc, scale = pearson3.fit(xbase[np.isfinite(xbase)])  # Fit Pearson III distribution
        # c, loc, scale = fisk.fit(x[np.isfinite(x)])  # Fit Pearson III distribution
    else:
        c, loc, scale = gamma_params

    # Cumulative distribution function for D
    fx = pearson3.cdf(x, skew=c, loc=loc, scale=scale)
    # fx = fisk.cdf(x,c=c, loc=loc, scale=scale)
    # Avoid "divide by zero" errors in calculations
    fx = np.clip(fx, 3.17e-5, 1 - 3.17e-5) #+-4 allowed

    spei = norm.ppf(fx,loc=0,scale=1)

    # plt.plot(spei, x, 'o')
    # *********** REVERSE FROM SPEI TO D *************************
    speibase = spei[tb1_id:tb2_id + 1]
    coef = np.polyfit(speibase[np.isfinite(xbase)], xbase[np.isfinite(xbase)], deg=3)
    # Coefficients from a 3th-degree polynomial fit can be used to reverse SPI:
    # Example:
    # val = np.polyval(coef, spei)
    # np.polyval(coef, 0)
    # ***************************************************************
    # After calculating SPI for month m, insert the processed values into the full SPI array
    # Spi[idmesi_all] = spi

    return idmesi_all, spei, coef,(c, loc, scale) if gamma_params is None else None

# FOR NORMAL DISTRIBUTED DATA
# ===================================================================
#  Z-Score Computation
# ===================================================================

def f_zscore(data, stride, m, m_cal, tb1, tb2):
    """
    Calculate the Z-Score for a time series of monthly data using a specified baseline period.

    Args:
        data (numpy.ndarray):
            A one-dimensional array of monthly data (e.g., temperature, precipitation, etc.) with size n.
        stride (int):
            The length of the accumulation period for z-score (e.g., 18 for 18-month accumulation).
        m (int):
            The reference month (1-12), where 1 = January, 2 = February, etc.
        m_cal (numpy.ndarray):
            A calendar array corresponding to the data time series, with two columns:
            - Column 0: Month (1-12)
            - Column 1: Year.
        tb1 (int):
            The starting year for the baseline period.
        tb2 (int):
            The ending year for the baseline period.

    Returns:
        numpy.ndarray: Indices of the months considered for the z-score calculation.
        numpy.ndarray: The calculated z-score values for each month.
        numpy.ndarray: Coefficients of the polynomial fit used for reversing z-score to original data.

    Raises:
        ValueError: If the baseline years are inconsistent with the provided calendar.

    Notes:
    ------
    - The function computes the baseline z-score values using the specified reference month and
      accumulation period.
    - If the accumulation period (stride) is set to 1, the calculation uses individual monthly data points.
    - If the accumulation period is greater than 1, the function aggregates data values over the defined periods.
    - The z-score is calculated as (value - mean) / standard deviation for the baseline period.
    - The output includes coefficients for reversing the z-score to estimate the original data values.
    """

    # Extract start and end years from the calendar
    t1 = int(m_cal[0, 1])  # Start year
    t2 = int(m_cal[-1, 1])  # End year
    anni = np.unique(m_cal[:, 1]).astype(int)  # Unique years in the calendar

    try:
        tb1_id = np.where(anni == tb1)[0][0]  # Index of the baseline start year
        tb2_id = np.where(anni == tb2)[0][0]  # Index of the baseline end year
    except IndexError:
        raise ValueError("Inconsistent baseline years: define a period within the data temporal domain.")

    # Function to obtain the indices of specific months within a time range


    # -------- BASELINE ------------------------------------------------
    if stride == 1:
        # Z-score for single month: directly use all data
        idmesi_base = get_month_indices(m, tb1, tb2, m_cal)
        xbase = data[idmesi_base]  # Data for the baseline period

        # Get all months for the entire time period
        try:
            idmesi_all = get_month_indices(m, t1, t2, m_cal)
        except (IndexError, ValueError):
            try:
                idmesi_all = get_month_indices(m, t1, t2 - 1, m_cal)
            except (IndexError, ValueError):
                try:
                    idmesi_all = get_month_indices(m, t1 + 1, t2, m_cal)
                except (IndexError, ValueError):
                    idmesi_all = get_month_indices(m, t1 + 1, t2 - 1, m_cal)

        x = data[idmesi_all]  # Data for the entire period

    else:
        # Z-score with accumulation period greater than 1
        # Calculate the baseline
        idmesi_base = get_month_indices(m, tb1, tb2, m_cal)
        a = np.array(
            [idmesi_base - j for j in np.flip(np.arange(0, stride))]).T  # Create the matrix of months to select
        xbase = np.array([np.sum(data[row]) if np.all(row >= 0) else np.nan for row in a])

        # WHOLE PERIOD ----------------------------------------------
        # Get all months for the entire time period
        try:
            idmesi_all = get_month_indices(m, t1, t2, m_cal)
        except (IndexError, ValueError):
            try:
                idmesi_all = get_month_indices(m, t1, t2 - 1, m_cal)
            except (IndexError, ValueError):
                try:
                    idmesi_all = get_month_indices(m, t1+1, t2, m_cal)
                except (IndexError, ValueError):
                    idmesi_all = get_month_indices(m, t1 + 1, t2-1, m_cal)

        a = np.array([idmesi_all - j for j in np.flip(np.arange(0, stride))]).T  # Create the matrix of months to select
        x = np.array([np.sum(data[row]) if np.all(row >= 0) else np.nan for row in a])  # Data for the entire period

    # --------------------------- Z-Score -----------------------------------------
    # Calculate the monthly balance for the baseline and the entire period
    baseline_mean = np.nanmean(xbase)
    baseline_std = np.nanstd(xbase)

    if baseline_std !=0:
        zscore = (x - baseline_mean) / baseline_std
    else:
        zscore = np.zeros(np.shape(x))


    # *********** REVERSE FROM Z-Score TO ORIGINAL DATA *************************
    z_params =[baseline_mean,baseline_std]

    return idmesi_all, zscore, z_params

