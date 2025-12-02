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
def test_standardization(data, groups=None):
    """
    Determine the most appropriate standardization method for a dataset,
    optionally performing the analysis separately per group.

    The function computes skewness, a normality test, a recommended
    distribution family (Gamma / Pearson III / Gaussian), and a KS-based
    goodness-of-fit metric. If `groups` is provided, the analysis is run
    independently for each group.

    Parameters
    ----------
    data : array-like
        Input data to analyze.
    groups : array-like or None, optional
        Optional grouping labels, must have the same length as `data`.
        If provided, a separate analysis is performed for each unique
        group value.

    Returns
    -------
    dict
        If `groups` is None:
            A dictionary with keys:
                - "skewness"
                - "normality_p_value"
                - "recommendation"
                - "KS_statistic"
                - "KS_p_value"
                - "error_percent"
                - "goodness_percent"
        If `groups` is provided:
            A nested dictionary of the form:
            {group_value: stats_dict} with the same fields as above.
    """

    data = np.asarray(data)

    # -------------------------
    # INTERNAL ANALYSIS FUNCTION
    # -------------------------
    def analyze_single(dataset):
        dataset = np.asarray(dataset)
        dataset = dataset[np.isfinite(dataset)]

        # --- basic stats ---
        skewness = stats.skew(dataset)
        _, p_normal = stats.normaltest(dataset)

        # --- recommendation logic ---
        if np.min(dataset) > 0 and skewness > 1:
            recommendation = "Gamma (strong right-skew and only positive values)"
            dist = "gamma"
        elif skewness > 1 or skewness < -1:
            recommendation = "Pearson III (significant asymmetry)"
            dist = "pearson3"
        elif p_normal > 0.05:
            recommendation = "Gaussian (z-score, likely normal)"
            dist = "gaussian"
        else:
            recommendation = "Unclear — further analysis needed"
            dist = "gamma"  # fallback

        # -------------------------
        # KS TEST SECTION
        # -------------------------
        # gamma: need positive data
        if dist == "gamma":
            # shift to avoid zero
            shifted = dataset + 1
            alpha, loc, scale = stats.gamma.fit(shifted, floc=0)
            D, p_ks = stats.kstest(shifted, 'gamma', args=(alpha, loc, scale))

        # Pearson III: SciPy implementation is stats.pearson3
        elif dist == "pearson3":
            # pearson3 fit: shape, loc, scale
            shape, loc, scale = stats.pearson3.fit(dataset)
            D, p_ks = stats.kstest(dataset, 'pearson3', args=(shape, loc, scale))

        # Gaussian
        else:
            mu, sigma = np.mean(dataset), np.std(dataset)
            D, p_ks = stats.kstest(dataset, 'norm', args=(mu, sigma))

        # Convert KS statistic to intuitive metrics
        error_pct = 100 * D
        goodness_pct = 100 * (1 - D)

        return {
            "skewness": skewness,
            "normality_p_value": p_normal,
            "recommendation": recommendation,
            "KS_statistic": D,
            "KS_p_value": p_ks,
            "error_percent": error_pct,
            "goodness_percent": goodness_pct,
        }

    # -------------------------
    # CASE 1: single dataset
    # -------------------------
    if groups is None:
        return analyze_single(data)

    # -------------------------
    # CASE 2: grouped dataset
    # -------------------------
    groups = np.asarray(groups)
    if len(groups) != len(data):
        raise ValueError("`groups` must have the same length as `data`.")

    results = {}
    for g in np.unique(groups):
        subset = data[groups == g]
        results[g] = analyze_single(subset)

    return results

def fit_distribution_stats(data, dist="gamma", groups=None, shift_for_gamma=True):
    """
    Compute goodness-of-fit statistics for a chosen distribution
    (gamma, pearson3, gaussian), optionally per group.

    The function performs:
        - Skewness estimation
        - Normality test (D'Agostino & Pearson)
        - MLE fitting of the selected distribution
        - KS test between empirical data and fitted distribution
        - Log-likelihood and AIC computation
        - Error/goodness percentages from KS statistic
        - Optional grouped analysis

    Parameters
    ----------
    data : array-like
        Input dataset.
    dist : {"gamma", "pearson3", "gaussian"}, default="gamma"
        Target distribution for fitting and KS testing.
    groups : array-like or None, optional
        If provided, must have same length as `data`. The analysis is
        performed independently for each unique group value.
    shift_for_gamma : bool, optional
        If True, and the chosen distribution is Gamma, a shift
        (data + 1) is applied to avoid issues with zeros or negative values.

    Returns
    -------
    dict
        If groups is None:
            A dictionary containing all fit statistics.
        If groups is provided:
            A nested dictionary {group_value: stats_dict}.

        Each stats_dict includes:
            - "distribution": selected distribution
            - "skewness"
            - "normality_p_value"
            - "params": fitted parameters
            - "KS_statistic"
            - "KS_p_value"
            - "error_percent"
            - "goodness_percent"
            - "log_likelihood"
            - "AIC"

    Notes
    -----
    - Gamma and Pearson III fits use 3 parameters (shape, loc, scale),
      Gaussian uses 2 (mu, sigma).
    - Data are automatically filtered for non-finite values.
    """

    data = np.asarray(data)
    dist = dist.lower()

    allowed = {"gamma", "pearson3", "gaussian"}
    if dist not in allowed:
        raise ValueError(f"`dist` must be one of {allowed}, got: {dist}")

    # --------------------------------------------------
    # Internal helper: analyze a single dataset
    # --------------------------------------------------
    def analyze_single(dataset):
        dataset = np.asarray(dataset)
        dataset = dataset[np.isfinite(dataset)]  # ensure clean data

        # Basic statistics
        skewness = stats.skew(dataset)
        _, p_normal = stats.normaltest(dataset)

        # Fit and KS test according to selected distribution
        if dist == "gamma":
            # Apply optional shift to avoid zero or negative values
            if shift_for_gamma:
                data_used = dataset + 1
            else:
                data_used = dataset

            # Fit Gamma: shape, loc, scale
            shape, loc, scale = stats.gamma.fit(data_used, floc=0)
            D, p_ks = stats.kstest(data_used, "gamma", args=(shape, loc, scale))

            params = {
                "shape": shape,
                "loc": loc,
                "scale": scale,
                "shift_applied": bool(shift_for_gamma),
            }

            logpdf_vals = stats.gamma.logpdf(data_used, shape, loc=loc, scale=scale)

        elif dist == "pearson3":
            data_used = dataset

            # Fit Pearson type III: shape, loc, scale
            shape, loc, scale = stats.pearson3.fit(data_used)
            D, p_ks = stats.kstest(data_used, "pearson3", args=(shape, loc, scale))

            params = {
                "shape": shape,
                "loc": loc,
                "scale": scale,
            }

            logpdf_vals = stats.pearson3.logpdf(data_used, shape, loc=loc, scale=scale)

        else:  # Gaussian
            data_used = dataset

            mu = np.mean(data_used)
            sigma = np.std(data_used, ddof=0)

            D, p_ks = stats.kstest(data_used, "norm", args=(mu, sigma))

            params = {
                "mu": mu,
                "sigma": sigma,
            }

            logpdf_vals = stats.norm.logpdf(data_used, loc=mu, scale=sigma)

        # Log-likelihood
        log_likelihood = np.sum(logpdf_vals)

        # Number of parameters for AIC:
        #   Gamma / Pearson III: 3 parameters (shape, loc, scale)
        #   Gaussian: 2 parameters (mu, sigma)
        k = 3 if dist in {"gamma", "pearson3"} else 2
        aic = 2 * k - 2 * log_likelihood

        # KS metric mapping
        error_pct = 100 * D
        goodness_pct = 100 * (1 - D)

        return {
            "distribution": dist,
            "skewness": skewness,
            "normality_p_value": p_normal,
            "params": params,
            "KS_statistic": D,
            "KS_p_value": p_ks,
            "error_percent": error_pct,
            "goodness_percent": goodness_pct,
            "log_likelihood": log_likelihood,
            "AIC": aic,
        }

    # --------------------------------------------------
    # Case 1: no groups
    # --------------------------------------------------
    if groups is None:
        return analyze_single(data)

    # --------------------------------------------------
    # Case 2: grouped analysis
    # --------------------------------------------------
    groups = np.asarray(groups)
    if len(groups) != len(data):
        raise ValueError("`groups` must have the same length as `data`.")

    results = {}
    for g in np.unique(groups):
        subset = data[groups == g]
        results[g] = analyze_single(subset)

    return results


# ===================================================================
#  Rolling Trend Analysis
# ===================================================================
def rolling_trend_analysis(var, window=60, significance=0.05):
    """
    Perform rolling trend analysis on a given time series.
    Args:
        Y (ndarray): Input time series array.
        window (int): Window size in months for rolling regression.
        significance (float): p-value threshold for trend significance.

    Returns:
        dict: Dictionary containing arrays of trend direction, slopes, p-values, and deltas.
    """

    n = len(var)

    # Arrays for storing results
    trends = np.zeros(n, dtype=int)
    slopes = np.full(n, np.nan, dtype=float)
    p_values = np.full(n, np.nan, dtype=float)
    deltas = np.full(n, np.nan, dtype=float)

    for i in range(n - window + 1):
        y_window = var[i:i + window]
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
