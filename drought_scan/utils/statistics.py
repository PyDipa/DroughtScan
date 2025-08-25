"""
author: PyDipa
# © 2025 Arianna Di Paola
# License: GNU General Public License v3.0 (GPLv3)

Statistical functions for drought analysis.

Provides helper functions for:
- **Time series analysis** (e.g., moving averages, trends).
- **Probability distributions** (Gamma fitting, percentiles).
- **Monte Carlo simulations** for uncertainty quantification.

Used by  and 'core.py'
"""

from scipy import stats
import numpy as np
from datetime import datetime,timedelta


# ===================================================================
#  Temporal Overlap and Concatenation Functions
# ===================================================================
def find_overlap(m_cal1, m_cal2):
    """
    Find temporal overlap between two calendar arrays (month, year).

    Args:
        m_cal1, m_cal2 (np.ndarray): calendar arrays (N, 2) with columns [month, year].

    Returns:
        tuple: indices of overlapping periods in m_cal1 and m_cal2.
    """
    # Convert (year, month) to numpy datetime64
    dates1 = np.array([np.datetime64(f'{int(y)}-{int(m):02d}') for m, y in m_cal1])
    dates2 = np.array([np.datetime64(f'{int(y)}-{int(m):02d}') for m, y in m_cal2])

    # Find overlapping dates
    overlap_dates = np.intersect1d(dates1, dates2)

    if overlap_dates.size == 0:
        raise ValueError("No overlapping periods found between the two calendars.")

    # Find indices of overlapping dates
    indices1 = np.where(np.isin(dates1, overlap_dates))[0]
    indices2 = np.where(np.isin(dates2, overlap_dates))[0]

    return indices1, indices2

from dateutil.relativedelta import relativedelta
from datetime import date
def concatenate_m_cal(m_cal1,m_cal2):

    """
    Generates a new m_cal vector based on the relationship between self.m_cal and self.forecast_m_cal.

    - If self.forecast_m_cal is fully contained within self.m_cal, it returns self.m_cal.
    - If self.forecast_m_cal partially overlaps with self.m_cal, it returns their intersection.
    - If self.forecast_m_cal is contiguous with self.m_cal, it returns their union.
    - If there is a gap between the two time ranges, it raises an error.

    Returns:
        np.ndarray: The modified m_cal based on the above conditions.
    Raises:
        ValueError: If there is a time gap between the two time ranges.
    """
    m_cal1 = m_cal1.astype(int)
    m_cal2 = m_cal2.astype(int)
    last_dt = date(m_cal1[-1,1], m_cal1[-1,0], 1)  # (anno, mese, giorno 1)
    first_new_dt = date(m_cal2[0,1], m_cal2[0,0], 1)  # (anno, mese, giorno 1)

    # Calcola il mese successivo
    last_month_plus_1 = last_dt + relativedelta(months=1)

    # date comparison
    if first_new_dt <= last_dt:
        case = 1 # total or partial overlap
    elif first_new_dt == last_month_plus_1:
        case = 2 #  continuity
    else:
        raise ValueError(" There is a time gap between self.m_cal and self.forecast_m_cal!")


    cal1_tuples = {tuple(row): i for i, row in enumerate(m_cal1)}

    # Per ogni elemento di `m_cal2`, troviamo il suo indice in `m_cal1`
    indices = [cal1_tuples.get(tuple(row), np.nan) for row in m_cal2]

    if case == 1:
        if sum(np.isnan(indices)) == 0:
            print('case 1 - total overlap')
            unique_m_cal = m_cal1
        else:
            print('case 1 - partial overlap')
            cells = sum(np.isnan(indices))
            unique_m_cal = np.vstack((m_cal1, m_cal2[-cells:]))
    elif case == 2:
        print('case 2 - continuity')
        unique_m_cal = np.vstack((m_cal1, m_cal2))
    else:
        print('---------')
        print(m_cal1)
        print('--------')
        print(m_cal2)
        raise ValueError('rivedi blocco')
    # # total overlap
    # if (case ==1) & (sum(np.isnan(indices)) == 0):
    #     print('case 1')
    #     unique_m_cal = m_cal1
    # # Partial overlap
    # elif (case==1) & (sum(np.isnan(indices)) >0 ):
    #     cells = sum(np.isnan(indices))
    #     unique_m_cal = np.vstack((m_cal1, m_cal2[-cells::]))
    #     print('case 1 partial')
    # # continuitu
    # elif sum(np.isnan(indices)) == 0: #total ovelap
    #     unique_m_cal = np.vstack((m_cal1, m_cal2))
    #     print('case 2')
    # else:
    #
    #     print('---------')
    #     print(m_cal1)
    #     print('--------')
    #     print(m_cal2)
    #     raise ValueError('rivedi blocco')

    return unique_m_cal
    # combined_m_cal = np.vstack((m_cal1, m_cal2))
    # combined_m_cal = np.sort(combined_m_cal, axis=1)
    # unique_m_cal = np.unique(combined_m_cal, axis=1)

    # #Check for continuity
    # date_list = [datetime(year, month, 1) for month, year in unique_m_cal]
    #
    # # Check for continuity
    # for i in range(len(date_list) - 1):
    #     expected_next_month = date_list[i] + timedelta(days=32)  # Approximate to ensure next month
    #     expected_next_month = datetime(expected_next_month.year, expected_next_month.month, 1)  # Normalize
    #     if expected_next_month != date_list[i + 1]:
    #         raise ValueError(" There is a time gap between self.m_cal and self.forecast_m_cal!")

# ===================================================================
#  Standardization Test
# ===================================================================
def test_standardization(data):
    """
    Determine the most appropriate standardization method for a dataset.

    This function analyzes the input data to decide whether to use:
    - Gamma function: for datasets with strong left-skew and only positive values.
    - Pearson III function: for asymmetric datasets with both positive and negative values.
    - Gaussian (z-score): for datasets following a normal distribution.

    Parameters:
    ----------
    data : array-like
        The input dataset to analyze.

    Returns:
    -------
    dict
        A dictionary containing:
        - Skewness: The calculated skewness of the data.
        - Normality p-value: The p-value from the normality test.
        - Recommendation: The suggested standardization method based on the analysis.

    Example:
    --------
    data = [102.44, 103.45, 104.46, ..., 132.73]
    result = test_standardization(data)
    print(result)
    """
    # Calculate the skewness of the data.
    # A high skewness (>1 or <-1) indicates significant asymmetry.
    skewness = stats.skew(data)

    # Perform a normality test (D'Agostino and Pearson's test).
    # Null hypothesis: the data comes from a normal distribution.
    _, p_value = stats.normaltest(data)
    if p_value < 0.05:
        print("The null hypothesis can be rejected: data is not normally distributed.")
    else:
        print("The null hypothesis cannot be rejected: data may follow a normal distribution.")

    # Determine the appropriate standardization method.
    if np.min(data) > 0 and skewness > 1:
        recommendation = "Gamma (strong left-skew and only positive values)"
    elif skewness > 1 or skewness < -1:
        recommendation = "Pearson III (significant asymmetry with both positive and negative values)"
    elif p_value > 0.05:  # Data likely follows a normal distribution
        recommendation = "Gaussian (z-score, data follows a normal distribution)"
    else:
        recommendation = "Unclear. Further exploration of the data is required."

# statistics.py in utils/

# ===================================================================
#  Rolling Trend Analysis
# ===================================================================
def rolling_trend_analysis(Y, window=60, significance=0.05):
    """
    Perform rolling trend analysis on a given time series.
    Args:
        Y (ndarray): Input time series array.
        window (int): Window size in months for rolling regression.
        significance (float): p-value threshold for trend significance.

    Returns:
        dict: Dictionary containing arrays of trend direction, slopes, p-values, and deltas.
    """

    n = len(Y)

    # Arrays for storing results
    trends = np.zeros(n, dtype=int)
    slopes = np.full(n, np.nan, dtype=float)
    p_values = np.full(n, np.nan, dtype=float)
    deltas = np.full(n, np.nan, dtype=float)

    for i in range(n - window + 1):
        y_window = Y[i:i + window]
        x = np.arange(window)

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y_window)

        if p_value < significance:
            if slope > 0:
                trends[i + window - 1] = 1
            elif slope < 0:
                trends[i + window - 1] = -1
        else:
            trends[i + window - 1] = 0

        slopes[i + window - 1] = slope
        p_values[i + window - 1] = p_value
        deltas[i + window - 1] = slope * window

    return {
        'trend': trends,
        'slope': slopes,
        'p_value': p_values,
        'delta': deltas
    }
