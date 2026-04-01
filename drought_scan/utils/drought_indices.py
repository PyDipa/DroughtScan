"""
author: PyDipa
# © 2026 Arianna Di Paola
# License: GNU General Public License v3.0 (GPLv3)

Drought Indices Computation.

This module implements the core standardization functions used to compute
drought indices from monthly time series:

- ``f_spi``    — Gamma-based standardization (SPI).
- ``f_spei``   — Pearson III-based standardization (SPEI).
- ``f_kde``    — Non-parametric KDE-based standardization.
- ``f_zscore`` — Gaussian z-score standardization.

Supporting utilities:

- ``weighted_metrics``  — weighted average and standard deviation.
- ``generate_weights``  — weight matrices for SIDI computation.
- ``baseline_indices``  — resolve baseline start/end indices from calendar.
- ``get_month_indices`` — cached lookup of month positions in calendar.

Used by: ``core.py``.
"""

import numpy as np
from scipy.stats import gamma, pearson3, norm, gaussian_kde
import warnings


# ===================================================================
#  SIDI weights
# ===================================================================
def weighted_metrics(values, weights):
    """
    Compute the weighted average and weighted standard deviation of a dataset.

    Args:
        values (numpy.ndarray): 1D array of data values.
        weights (numpy.ndarray): 1D array of weights (same shape as `values`).

    Returns:
        tuple: (weighted_average, weighted_std)
    """
    if values.shape != weights.shape:
        raise ValueError("Values and weights must have the same shape.")

    if np.sum(weights) == 0:
        raise ValueError("The sum of the weights must be greater than zero.")

    weighted_average = np.average(values, weights=weights)
    weighted_variance = np.average((values - weighted_average) ** 2, weights=weights)

    return weighted_average, np.sqrt(weighted_variance)


def generate_weights(k):
    """
    Generate weight matrices for SIDI computation.

    Args:
        k (int): Number of temporal scales to consider.

    Returns:
        ndarray: Weight matrix of shape (k, 5), where each column represents:
            - Column 0: Uniform weights
            - Column 1: Inverted linear weights
            - Column 2: Inverted geometric weights
            - Column 3: Linear weights
            - Column 4: Geometric weights
    """
    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError("`k` must be a positive integer.")

    geom_weights = np.geomspace(1, k, k)
    linear_weights = np.linspace(1, k, k)

    return np.vstack([
        np.tile(1 / k, k),                                  # Uniform
        np.flipud(linear_weights) / np.sum(linear_weights),  # Inverted linear
        np.flipud(geom_weights) / np.sum(geom_weights),      # Inverted geometric
        linear_weights / np.sum(linear_weights),              # Linear
        geom_weights / np.sum(geom_weights),                  # Geometric
    ]).T


# ===================================================================
#  Calendar utilities
# ===================================================================
def baseline_indices(m_cal, start_baseline_year, end_baseline_year):
    """
    Get indices for the baseline period based on start and end years.

    Returns:
        tuple: (start_index, end_index) into m_cal.
    """
    start_indices = np.where(m_cal[:, 1] == start_baseline_year)[0]
    end_indices = np.where(m_cal[:, 1] == end_baseline_year)[0]

    if len(start_indices) == 0:
        raise ValueError(f"Start baseline year {start_baseline_year} not found in `m_cal`.")
    if len(end_indices) == 0:
        raise ValueError(f"End baseline year {end_baseline_year} not found in `m_cal`.")

    tb1_id = start_indices[0]
    tb2_id = end_indices[-1]

    if tb1_id > tb2_id:
        raise ValueError("Inconsistent baseline indices: start index is after end index.")

    return tb1_id, tb2_id


_month_indices_cache = {}

def get_month_indices(month, start_year, end_year, m_cal):
    """
    Return indices of ``m_cal`` where the specified month appears for years
    ``start_year`` through ``end_year``.

    Results are cached by (month, start_year, end_year, id(m_cal)) so that
    repeated calls (e.g., across grid points sharing the same calendar) are
    effectively free.

    Args:
        month (int): Calendar month (1–12).
        start_year (int): First year.
        end_year (int): Last year.
        m_cal (numpy.ndarray): Calendar array (N, 2) with [month, year].

    Returns:
        numpy.ndarray: Indices into m_cal.
    """
    key = (month, start_year, end_year, id(m_cal))
    if key in _month_indices_cache:
        return _month_indices_cache[key]

    try:
        idx = np.array([
            np.where((m_cal[:, 0] == month) & (m_cal[:, 1] == year))[0][0]
            for year in range(start_year, end_year + 1)
        ])
    except IndexError:
        raise ValueError(
            f"Month {month} not found for all years in range "
            f"{start_year}–{end_year}. Check m_cal for gaps."
        )

    _month_indices_cache[key] = idx
    return idx


# ===================================================================
#  Shared helper: baseline & whole-period index resolution
# ===================================================================
def _resolve_indices(ts, stride, m, m_cal, tb1, tb2):
    """
    Compute baseline values (xbase), full-period values (x), and full-period
    month indices (idmesi_all) for a given accumulation stride and reference month.

    This logic is shared by f_spi, f_spei, f_kde, and f_zscore.

    Parameters
    ----------
    ts : ndarray
        Monthly time series (precipitation, balance, temperature, etc.).
    stride : int
        Accumulation period (1 = no accumulation).
    m : int
        Reference month (1–12).
    m_cal : ndarray
        Calendar array (N, 2) with [month, year].
    tb1, tb2 : int
        Baseline start and end years.

    Returns
    -------
    xbase : ndarray
        Values for the baseline period.
    x : ndarray
        Values for the full period.
    idmesi_all : ndarray
        Indices into m_cal for the full period.
    """
    t1 = int(m_cal[0, 1])
    t2 = int(m_cal[-1, 1])

    # --- Baseline indices ---
    idmesi_base = get_month_indices(m, tb1, tb2, m_cal)

    # --- Full-period indices (with fallback for edge years) ---
    idmesi_all = _get_idmesi_all_with_fallback(m, t1, t2, m_cal)

    if stride == 1:
        xbase = ts[idmesi_base]
        x = ts[idmesi_all]
    else:
        # Accumulate over `stride` consecutive months
        a_base = np.array([idmesi_base - j for j in np.flip(np.arange(stride))]).T
        valid_base = np.all(a_base >= 0, axis=1)
        xbase = np.where(valid_base, np.sum(ts[a_base.clip(0)], axis=1), np.nan)

        a_all = np.array([idmesi_all - j for j in np.flip(np.arange(stride))]).T
        valid_all = np.all(a_all >= 0, axis=1)
        x = np.where(valid_all, np.sum(ts[a_all.clip(0)], axis=1), np.nan)

    return xbase, x, idmesi_all


def _get_idmesi_all_with_fallback(m, t1, t2, m_cal):
    """
    Resolve full-period month indices with graceful fallback if the first or
    last year is incomplete.
    """
    attempts = [
        (t1, t2),
        (t1, t2 - 1),
        (t1 + 1, t2),
        (t1 + 1, t2 - 1),
    ]
    for i, (y1, y2) in enumerate(attempts):
        try:
            idx = get_month_indices(m, y1, y2, m_cal)
            if i > 0:
                dropped = f"t1={t1},t2={t2}" if i == 0 else f"using t1={y1},t2={y2}"
                warnings.warn(
                    f"Month {m}: incomplete edge year(s) — {dropped}. "
                    f"First/last year of data may be excluded for this month-scale.",
                    RuntimeWarning, stacklevel=3
                )
            return idx
        except (IndexError, ValueError):
            continue

    raise ValueError(
        f"Cannot resolve indices for month {m} in any year range "
        f"near [{t1}, {t2}]. Check m_cal for large gaps."
    )


# ===================================================================
#  Shared helper: reverse polyfit (index → raw values)
# ===================================================================
def _reverse_polyfit(spi, xbase, m_cal, idmesi_all, tb1, tb2, stride, m):
    """
    Fit a degree-3 polynomial from standardized baseline values to raw baseline
    values, for the inverse mapping (index → raw variable).

    Returns
    -------
    coef : ndarray, shape (4,)
        Polynomial coefficients. Set to NaN if fit fails.
    """
    years_all = m_cal[idmesi_all, 1]
    baseline_mask = (years_all >= tb1) & (years_all <= tb2)
    spibase = spi[baseline_mask]

    # Guard: length mismatch between spibase and xbase
    if len(spibase) != len(xbase):
        warnings.warn(
            f"Length mismatch: spibase={len(spibase)}, xbase={len(xbase)} "
            f"(scale={stride}, month={m}). Reverse coefficients set to NaN.",
            RuntimeWarning, stacklevel=2
        )
        return np.full(4, np.nan)

    valid = np.isfinite(spibase) & np.isfinite(xbase)
    if np.sum(valid) < 4:
        warnings.warn(
            f"Too few valid points for polyfit (scale={stride}, month={m}, "
            f"n_valid={np.sum(valid)}). Reverse coefficients set to NaN.",
            RuntimeWarning, stacklevel=2
        )
        return np.full(4, np.nan)

    try:
        coef = np.polyfit(spibase[valid], xbase[valid], deg=3)
    except Exception as e:
        warnings.warn(
            f"polyfit failed (scale={stride}, month={m}): {e}. "
            f"Reverse coefficients set to NaN.",
            RuntimeWarning, stacklevel=2
        )
        coef = np.full(4, np.nan)

    return coef


# ===================================================================
#  f_spi — Gamma-based (for positive, right-skewed data)
# ===================================================================
def f_spi(prec, stride, m, m_cal, tb1, tb2, gamma_params=None):
    """
    Calculate the Standardized Precipitation Index (SPI) using a Gamma distribution.

    Args:
        prec (numpy.ndarray): Monthly precipitation (or streamflow) series, shape (n,).
        stride (int): Accumulation period (e.g., 18 for SPI-18).
        m (int): Reference month (1–12).
        m_cal (numpy.ndarray): Calendar array (n, 2) with [month, year].
        tb1 (int): Baseline start year.
        tb2 (int): Baseline end year.
        gamma_params (tuple, optional): Pre-computed (alpha, loc, beta).

    Returns:
        tuple:
            - idmesi_all (ndarray): month indices for the full period.
            - spi (ndarray): standardized SPI values.
            - coef (ndarray): degree-3 polyfit coefficients for inverse mapping.
            - params (tuple or None): (alpha, loc, beta) if fitted, else None.
    """
    xbase, x, idmesi_all = _resolve_indices(prec, stride, m, m_cal, tb1, tb2)

    if xbase.size < 10:
        warnings.warn(
            f"SPI baseline sample size is very small (n={xbase.size}). "
            f"Fit may be unreliable.",
            RuntimeWarning, stacklevel=2
        )

    # --- Gamma fit ---
    if gamma_params is None:
        alpha, loc, beta = gamma.fit(xbase[xbase > 0], floc=0)
    else:
        alpha, loc, beta = gamma_params

    # --- CDF with zero-inflation ---
    Gx = gamma.cdf(x, a=alpha, loc=loc, scale=beta)
    qq = np.sum(x == 0) / len(x)
    Hx = qq + (1 - qq) * Gx
    Hx = np.clip(Hx, 3.17e-5, 1 - 3.17e-5)

    spi = np.round(norm.ppf(Hx), 4)

    # --- Reverse mapping ---
    coef = _reverse_polyfit(spi, xbase, m_cal, idmesi_all, tb1, tb2, stride, m)

    return idmesi_all, spi, coef, (alpha, loc, beta) if gamma_params is None else None


# ===================================================================
#  f_spei — Pearson III (for real-valued, possibly skewed data)
# ===================================================================
def f_spei(balance, stride, m, m_cal, tb1, tb2, gamma_params=None):
    """
    Calculate the Standardized Precipitation Evapotranspiration Index (SPEI)
    using a Pearson III distribution.

    Args:
        balance (numpy.ndarray): Monthly P–PET (or any real-valued) series, shape (n,).
        stride (int): Accumulation period.
        m (int): Reference month (1–12).
        m_cal (numpy.ndarray): Calendar array (n, 2) with [month, year].
        tb1 (int): Baseline start year.
        tb2 (int): Baseline end year.
        gamma_params (tuple, optional): Pre-computed (c, loc, scale).

    Returns:
        tuple:
            - idmesi_all (ndarray): month indices for the full period.
            - spei (ndarray): standardized SPEI values.
            - coef (ndarray): degree-3 polyfit coefficients for inverse mapping.
            - params (tuple or None): (c, loc, scale) if fitted, else None.
    """
    xbase, x, idmesi_all = _resolve_indices(balance, stride, m, m_cal, tb1, tb2)

    if xbase.size < 10:
        warnings.warn(
            f"SPEI baseline sample size is very small (n={xbase.size}). "
            f"Fit may be unreliable.",
            RuntimeWarning, stacklevel=2
        )

    # --- Pearson III fit ---
    if gamma_params is None:
        c, loc, scale = pearson3.fit(xbase[np.isfinite(xbase)])
    else:
        c, loc, scale = gamma_params

    # --- CDF → normal quantile ---
    fx = pearson3.cdf(x, skew=c, loc=loc, scale=scale)
    fx = np.clip(fx, 3.17e-5, 1 - 3.17e-5)
    spei = norm.ppf(fx, loc=0, scale=1)

    # --- Reverse mapping ---
    coef = _reverse_polyfit(spei, xbase, m_cal, idmesi_all, tb1, tb2, stride, m)

    return idmesi_all, spei, coef, (c, loc, scale) if gamma_params is None else None


# ===================================================================
#  f_kde — Non-parametric KDE (for any data)
# ===================================================================
def f_kde(prec, stride, m, m_cal, tb1, tb2, log_transform=True, kde_params=None):
    """
    SPI-like standardization using Gaussian KDE (Silverman bandwidth).

    Args:
        prec (np.ndarray): Monthly series (n,).
        stride (int): Accumulation period.
        m (int): Reference month (1–12).
        m_cal (np.ndarray): Calendar (n, 2) with [month, year].
        tb1 (int): Baseline start year.
        tb2 (int): Baseline end year.
        log_transform (bool): If True, apply log-transform before KDE fitting.
        kde_params (dict or None): Optional pre-computed KDE info.

    Returns:
        tuple:
            - idmesi_all (ndarray): month indices for the full period.
            - spi (ndarray): standardized index values.
            - coef (ndarray): degree-3 polyfit coefficients for inverse mapping.
            - out_params (dict or None): KDE metadata if fitted, else None.
    """
    xbase, x, idmesi_all = _resolve_indices(prec, stride, m, m_cal, tb1, tb2)

    # --- Pre-allocate output ---
    spi = np.full_like(x, np.nan, dtype=float)
    finite_x = np.isfinite(x)
    finite_xbase = np.isfinite(xbase)

    # --- Optional log-transform ---
    if log_transform:
        with np.errstate(divide='ignore', invalid='ignore'):
            x_log = np.log(x)
            xbase_log = np.log(xbase)
        finite_x = np.isfinite(x_log)
        finite_xbase = np.isfinite(xbase_log)

    # Early exit if no valid data
    if not np.any(finite_x):
        return idmesi_all, spi, np.full(4, np.nan), (
            None if kde_params is None else None
        )

    # --- KDE fit (baseline) ---
    if kde_params is None:
        xb = xbase_log[finite_xbase] if log_transform else xbase[finite_xbase]

        if xb.size < 10:
            warnings.warn(
                f"KDE baseline sample size is very small (n={xb.size}). "
                f"Fit may be unreliable.",
                RuntimeWarning, stacklevel=2
            )

        kde = gaussian_kde(xb, bw_method="silverman")
        out_params = {
            "kde": kde,
            "bw_factor": float(kde.factor),
            "n_fit": int(xb.size),
            "fit_domain": "xbase (finite, full R)",
        }
    else:
        kde = kde_params.get("kde", None)
        if kde is None:
            raise ValueError("kde_params provided but missing key 'kde'.")
        xb = xbase_log[finite_xbase] if log_transform else xbase[finite_xbase]
        out_params = None

    # --- CDF evaluation via broadcasting ---
    Gx = np.full_like(x, np.nan, dtype=float)
    Hx = np.full_like(x, np.nan, dtype=float)

    xfull = x_log if log_transform else x
    finite_mask = np.isfinite(xfull)
    zero_mask = (x == 0)

    h = 0.9 * np.std(xb, ddof=1) * xb.size ** (-1 / 5)
    qq = np.sum(zero_mask) / len(x)

    # Broadcasting: (n_eval, n_baseline) → mean over axis=1
    Gx[finite_mask] = norm.cdf(
        (xfull[finite_mask, None] - xb[None, :]) / h
    ).mean(axis=1)

    # --- Mixed distribution for zeros (same approach as f_spi) ---
    Hx[finite_mask] = qq + (1 - qq) * Gx[finite_mask]
    Hx[zero_mask] = qq

    # --- Normal-score transform ---
    Hx = np.clip(Hx, 3.17e-5, 1 - 3.17e-5)
    spi = np.round(norm.ppf(Hx), 4)

    # --- Reverse mapping ---
    coef = _reverse_polyfit(spi, xbase, m_cal, idmesi_all, tb1, tb2, stride, m)

    return idmesi_all, spi, coef, out_params


# ===================================================================
#  f_zscore — Gaussian z-score (for normally distributed data)
# ===================================================================
def f_zscore(data, stride, m, m_cal, tb1, tb2):
    """
    Calculate the z-score for a time series of monthly data.

    Args:
        data (numpy.ndarray): Monthly series, shape (n,).
        stride (int): Accumulation period.
        m (int): Reference month (1–12).
        m_cal (numpy.ndarray): Calendar (n, 2) with [month, year].
        tb1 (int): Baseline start year.
        tb2 (int): Baseline end year.

    Returns:
        tuple:
            - idmesi_all (ndarray): month indices for the full period.
            - zscore (ndarray): standardized z-score values.
            - z_params (list): [baseline_mean, baseline_std].
    """
    xbase, x, idmesi_all = _resolve_indices(data, stride, m, m_cal, tb1, tb2)

    baseline_mean = np.nanmean(xbase)
    baseline_std = np.nanstd(xbase)

    if baseline_std != 0:
        zscore = (x - baseline_mean) / baseline_std
    else:
        zscore = np.zeros(np.shape(x))

    z_params = [baseline_mean, baseline_std]
    return idmesi_all, zscore, z_params