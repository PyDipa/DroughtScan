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

# Tail clip applied to cumulative probabilities before the normal-score transform.
# Caps the index at about +/-4 sigma, so a value beyond anything the baseline
# observed saturates instead of diverging. Used by the forward standardization
# (f_spi/f_spei/f_kde) and, for symmetry, by the reverse transforms in core.py
# (native_to_spi/spi_to_native) — otherwise the two directions disagree at the
# tails, the forward saturating at 4 while the reverse returns +/-inf.
PROB_CLIP = 3.17e-5


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
#  Season grouping from spi_sqi_corr's R2(month, k)
# ===================================================================
# MAX_K_GAP_MONTHS: two adjacent calendar months merge into the same season
# iff their optimal accumulation scale k* differs by strictly less than this
# many months; a gap of `MAX_K_GAP_MONTHS` or more is always a season
# boundary (e.g. August next to a July/September block with a much larger
# optimal k). Since k* is always integer-valued, "< MAX_K_GAP_MONTHS merges"
# and ">= MAX_K_GAP_MONTHS cuts" are complementary and exhaustive — every
# possible integer gap falls on exactly one side, so this single threshold
# fully determines each boundary; no separate "closer of the two neighbours"
# comparison is needed (an earlier draft used one, but it could split two
# months as similar as e.g. Jan/Feb (gap 1) apart just because each of them
# individually happened to be even closer to its other neighbour — this
# threshold rule doesn't have that artifact). Value agreed with Arianna on
# 2026-08-13.
MAX_K_GAP_MONTHS = 2


def optimal_k_per_month(R2):
    """
    Per calendar month (row of R2, Jan=0..Dec=11), the 1-indexed scale k
    that maximises R^2. NaN where the whole row is 0, i.e. no scale reaches
    significance (p < 0.05) for that month — spi_sqi_corr already zeroes
    non-significant cells, so `row.max() == 0` is unambiguous.

    Args:
        R2 (ndarray): shape (12, K), as returned by spi_sqi_corr.

    Returns:
        ndarray, shape (12,): optimal k per calendar month (NaN if none significant).
    """
    k_star = np.full(R2.shape[0], np.nan)
    for m in range(R2.shape[0]):
        row = R2[m]
        if np.any(row > 0):
            k_star[m] = np.argmax(row) + 1
    return k_star


def seasons_from_r2(R2, max_k_gap=MAX_K_GAP_MONTHS):
    """
    Derive a `seasons` dict (contiguous calendar-month groups) from
    spi_sqi_corr's R2(month, k), in the exact shape
    `analyze_correlation_seasonal(seasons=...)` expects: {label: [months]}.

    Merge rule (agreed with Arianna, 2026-08-13, revised same day to drop
    the "closer neighbour" comparison — see MAX_K_GAP_MONTHS), walking the
    12 calendar months circularly (the Dec-Jan boundary can merge, like any
    other):
      - a month merges with the PREVIOUS month iff their optimal scales k*
        differ by less than `max_k_gap`; otherwise it starts a new season;
      - a month with no significant k at any scale (k* is NaN) never merges
        with a significant neighbour. Adjacent non-significant months DO
        merge with each other, forming one contiguous "broken signal" season.

    Args:
        R2 (ndarray): shape (12, K), as returned by spi_sqi_corr.
        max_k_gap (int): see MAX_K_GAP_MONTHS.

    Returns:
        dict[str, list[int]]: season label -> 1-indexed calendar months.
    """
    k_star = optimal_k_per_month(R2)
    n = 12

    # cut_before[m] = True -> a season boundary immediately before calendar
    # month m (0-indexed, 0=Jan), decided solely from the gap to the
    # PREVIOUS month (circular) — so all 12 circular boundaries are covered
    # exactly once, no conflicts.
    cut_before = [False] * n
    for m in range(n):
        prev_i = (m - 1) % n
        if np.isnan(k_star[m]):
            # start a new "broken signal" block unless the previous month is
            # also non-significant, in which case this month extends it.
            cut_before[m] = not np.isnan(k_star[prev_i])
            continue
        if np.isnan(k_star[prev_i]):
            cut_before[m] = True  # can't merge backward into a NaN block
            continue
        d_prev = abs(k_star[m] - k_star[prev_i])
        cut_before[m] = d_prev >= max_k_gap

    starts = [m for m in range(n) if cut_before[m]]
    if not starts:
        starts = [0]  # degenerate case: whole year merges into one season

    month_initials = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    seasons = {}
    for idx, s in enumerate(starts):
        e = starts[(idx + 1) % len(starts)]
        months0 = list(range(s, n)) + list(range(0, e)) if e <= s else list(range(s, e))
        months = [m + 1 for m in months0]  # 1-indexed calendar months
        label = "".join(month_initials[m - 1] for m in months)
        # Two non-adjacent groups can produce the same label (e.g. two
        # isolated single-month seasons sharing a letter); disambiguate
        # rather than silently overwrite one in the dict.
        if label in seasons:
            label = f"{label}_{idx}"
        seasons[label] = months

    return seasons


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

    Results are cached by (month, start_year, end_year, m_cal.tobytes()) so that
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
    key = (month, start_year, end_year, m_cal.tobytes())
    # # try the follwing to b faster:
    # key = (month, start_year, end_year, hash(m_cal.tobytes()))
    if key in _month_indices_cache:
        return _month_indices_cache[key]

    idx_list = []
    for year in range(start_year, end_year + 1):
        matches = np.where((m_cal[:, 0] == month) & (m_cal[:, 1] == year))[0]
        if matches.size == 0:
            raise ValueError(
                f"Month {month} not found for year {year} \n"
                f"m_cal spans {int(m_cal[:, 1].min())}–{int(m_cal[:, 1].max())}.\n "
                f"! Adjust baseline boundaries or check for gaps in m_cal."
            )
        idx_list.append(matches[0])
    idx = np.array(idx_list)


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
#  Zero-inflated Gamma forward/reverse — shared exact CDF/PPF for f_spi
# ===================================================================
def gamma_cdf_zi(x, alpha, loc, beta, qq, shift=0.0):
    """
    Zero-inflated Gamma cumulative probability Hx at native value(s) x.

    ``Hx = qq + (1 - qq) * Gamma.cdf(x + shift)``, i.e. a point mass ``qq`` at
    exactly zero mixed with the continuous Gamma fitted on the strictly-positive
    part. This is THE definition of the f_spi forward transform, factored out —
    exactly as ``kde_cdf`` is for f_kde — so that f_spi itself, the reverse
    conversions in core.py (native_to_spi/spi_to_native), the pixel-level
    deficit in ``_process_grid_point_trends`` and the diagnostic fits in
    ``statistics._mixture_cdf`` all evaluate one formula instead of four.

    Before this existed the reverse direction used a bare ``gamma.cdf``/
    ``gamma.ppf`` with no ``qq`` term, so it was NOT the inverse of the forward
    transform on zero-inflated data: with 32% exact zeros in the baseline,
    ``native_to_spi(20 mm)`` returned 0.09 where f_spi had produced 0.49, and
    the SPI=0 "normal value" came out at 17.5 mm instead of 6.1 mm — an error
    that propagated into normal_values/deficit_from_spi/volume_anomaly_rolling
    and every native-unit product downstream.

    Args:
        x (float or ndarray): native-unit value(s) to evaluate.
        alpha, loc, beta (float): fitted Gamma shape, location and scale.
        qq (float): fraction of exact zeros in the baseline at fit time.
        shift (float): constant added to x before the Gamma is evaluated, for
            fits calibrated on shifted data (``statistics``' legacy
            ``shift_for_gamma``). f_spi itself never shifts, so this is 0.0.

    Returns:
        float or ndarray: cumulative probability Hx (not clipped — the caller
        applies PROB_CLIP, as f_spi/f_kde do). NaN propagates from NaN input.
    """
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    is_scalar = np.ndim(x) == 0

    Gx = gamma.cdf(x_arr + shift, a=alpha, loc=loc, scale=beta)
    Hx = qq + (1 - qq) * Gx
    # Everything at or below zero is the point mass, whatever `loc`/`shift`
    # would make the continuous part say there; NaN stays NaN.
    Hx = np.where(x_arr <= 0, qq, Hx)
    Hx = np.where(np.isnan(x_arr), np.nan, Hx)

    return float(Hx[0]) if is_scalar else Hx


def gamma_ppf_zi(Hx, alpha, loc, beta, qq, shift=0.0):
    """
    Invert ``gamma_cdf_zi``: cumulative probability Hx -> native-unit value(s).

    Values of Hx at or below the zero-inflation fraction ``qq`` map to exactly
    0.0, matching the point mass at zero in the forward transform; above it the
    conditional probability ``(Hx - qq) / (1 - qq)`` is fed to the Gamma ppf.
    Same contract as ``kde_ppf`` for f_kde.

    Args:
        Hx (float or ndarray): cumulative probability value(s) to invert.
        alpha, loc, beta, qq, shift: same fitted parameters as ``gamma_cdf_zi``.

    Returns:
        float or ndarray: native-unit value(s). NaN propagates from NaN input.
    """
    Hx_arr = np.atleast_1d(np.asarray(Hx, dtype=float))
    is_scalar = np.ndim(Hx) == 0

    out = np.zeros(Hx_arr.shape, dtype=float)
    above = np.isfinite(Hx_arr) & (Hx_arr > qq)

    if np.any(above):
        if qq >= 1.0:
            # Degenerate baseline: all mass at zero, nothing above it to invert.
            out[above] = 0.0
        else:
            Gx = np.clip((Hx_arr[above] - qq) / (1 - qq), 0.0, 1.0)
            out[above] = gamma.ppf(Gx, a=alpha, loc=loc, scale=beta) - shift

    out = np.where(np.isnan(Hx_arr), np.nan, out)

    return float(out[0]) if is_scalar else out


# ===================================================================
#  f_spi — Gamma-based (for positive, right-skewed data)
# ===================================================================
def f_spi(prec, stride, m, m_cal, tb1, tb2, fit_params=None):
    """
    Calculate the Standardized Precipitation Index (SPI) using a Gamma distribution.

    Args:
        prec (numpy.ndarray): Monthly precipitation (or streamflow) series, shape (n,).
        stride (int): Accumulation period (e.g., 18 for SPI-18).
        m (int): Reference month (1–12).
        m_cal (numpy.ndarray): Calendar array (n, 2) with [month, year].
        tb1 (int): Baseline start year.
        tb2 (int): Baseline end year.
        fit_params (tuple, optional): Pre-computed (alpha, loc, beta, qq) — the
            complete calibration, zero-inflation fraction included. A legacy
            3-tuple (alpha, loc, beta) is still accepted, but then `qq` has to be
            re-estimated from the supplied data and the result is NOT in the
            reference frame of the fit that was passed in (a warning says so).

    Returns:
        tuple:
            - idmesi_all (ndarray): month indices for the full period.
            - spi (ndarray): standardized SPI values.
            - coef (ndarray): degree-3 polyfit coefficients for inverse mapping
              (approximate; retained for backward compatibility only — the exact
              inverse is `gamma_ppf_zi`, used by spi_to_native).
            - params (tuple or None): (alpha, loc, beta, qq) if fitted, else None.
              `qq` is part of the fit, not a derived quantity: it is what makes
              `gamma_cdf_zi`/`gamma_ppf_zi` an exact inverse pair (see
              native_to_spi/spi_to_native).
    """
    xbase, x, idmesi_all = _resolve_indices(prec, stride, m, m_cal, tb1, tb2)

    if xbase.size < 10:
        warnings.warn(
            f"SPI baseline sample size is very small (n={xbase.size}). "
            f"Fit may be unreliable.",
            RuntimeWarning, stacklevel=2
        )

    # --- Zero-inflation fraction, estimated on the BASELINE ---
    # qq is P(X=0) of the fitted mixture, so it must come from the same sample the
    # Gamma below is fitted on (xbase[xbase > 0]). Taking it from the full period
    # would mix two samples and let appended data shift past SPI values.
    xbase_valid = xbase[np.isfinite(xbase)]
    qq_baseline = float(np.sum(xbase_valid == 0) / xbase_valid.size) if xbase_valid.size else 0.0

    # --- Gamma fit ---
    if fit_params is None:
        alpha, loc, beta = gamma.fit(xbase[xbase > 0], floc=0)
        qq = qq_baseline
    else:
        # Honour the stored calibration: (alpha, loc, beta, qq) ARE the fitted
        # distribution, so reusing qq as-is is what keeps new values in the
        # calibration's reference frame (the whole point of passing fit_params —
        # e.g. scenarios/forecast, where `prec` is modified but SPI must stay
        # comparable to the original fit). Re-deriving qq from the supplied data
        # would silently recalibrate the point mass at zero.
        fit_params = tuple(np.ravel(fit_params))
        if len(fit_params) == 4:
            alpha, loc, beta, qq = (float(v) for v in fit_params)
        elif len(fit_params) == 3:
            alpha, loc, beta = (float(v) for v in fit_params)
            qq = qq_baseline
            if qq != 0.0:
                warnings.warn(
                    f"f_spi: fit_params provided without the zero-inflation "
                    f"fraction qq (legacy 3-tuple) — re-estimated on the supplied "
                    f"data as qq={qq:.4f} (month {m}, month-scale {stride}). The "
                    f"result is NOT in the reference frame of the fit that was "
                    f"passed in.",
                    RuntimeWarning, stacklevel=2
                )
        else:
            raise ValueError(
                f"f_spi: fit_params must be (alpha, loc, beta, qq) — or a legacy "
                f"(alpha, loc, beta) — got {len(fit_params)} values."
            )

    # --- CDF with zero-inflation (exact; same formula reused by the reverse
    # transforms via gamma_cdf_zi/gamma_ppf_zi) ---
    Hx = gamma_cdf_zi(x, alpha, loc, beta, qq)
    Hx = np.clip(Hx, PROB_CLIP, 1 - PROB_CLIP)

    spi = np.round(norm.ppf(Hx), 4)

    # --- Reverse mapping ---
    coef = _reverse_polyfit(spi, xbase, m_cal, idmesi_all, tb1, tb2, stride, m)

    return idmesi_all, spi, coef, (alpha, loc, beta, qq) if fit_params is None else None


# ===================================================================
#  f_spei — Pearson III (for real-valued, possibly skewed data)
# ===================================================================
def f_spei(balance, stride, m, m_cal, tb1, tb2, fit_params=None):
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
        fit_params (tuple, optional): Pre-computed (c, loc, scale).

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
    if fit_params is None:
        c, loc, scale = pearson3.fit(xbase[np.isfinite(xbase)])
    else:
        c, loc, scale = fit_params

    # --- CDF → normal quantile ---
    fx = pearson3.cdf(x, skew=c, loc=loc, scale=scale)
    fx = np.clip(fx, PROB_CLIP, 1 - PROB_CLIP)
    spei = np.round(norm.ppf(fx, loc=0, scale=1), 4)

    # --- Reverse mapping ---
    coef = _reverse_polyfit(spei, xbase, m_cal, idmesi_all, tb1, tb2, stride, m)

    return idmesi_all, spei, coef, (c, loc, scale) if fit_params is None else None


# ===================================================================
#  KDE forward/reverse — shared exact CDF/PPF for f_kde
# ===================================================================
def kde_fit_sample(values, log_transform=True):
    """
    Decide, from one sample, the three things a KDE fit is made of: whether to work
    in log space, the zero-inflation fraction ``qq``, and the sample the continuous
    part is actually fitted on.

    One implementation, so ``f_kde`` (the operational fit) and
    ``statistics._fit_single_dist`` (the diagnostic that judges it) cannot disagree
    about which points enter the fit — they did: on real-valued data the diagnostic
    fitted the KDE on the strictly-positive subset only, while f_kde fitted it on
    everything, so the page reported how well a KDE described half the sample.

    Rules:

    - **Negative values present** — the variable is real-valued, so it is not bounded
      below and an exact 0.0 is just a value, not a pile-up against a floor. Then
      ``qq = 0`` (no point mass), ``log_transform`` is forced off (log of a negative
      is undefined, not merely inconvenient), and every finite value enters the fit.
      Forcing a point mass at zero here would also make the mixture ill-formed: the
      mass would sit *below* the negative half of the distribution, so the CDF would
      start at ``qq`` instead of 0.
    - **Otherwise** — the variable is bounded below by zero, exact zeros are a genuine
      point mass: ``qq`` is their fraction, they are masked out of the continuous
      part, and ``log_transform`` is honoured. (Under a log transform the masking is
      automatic, since ``log(0) = -inf`` is not finite.)

    Args:
        values (ndarray): the sample the fit is calibrated on — the BASELINE, not the
            full record (see f_kde).
        log_transform (bool): requested transform; may be overridden to False.

    Returns:
        tuple: ``(xb, qq, log_transform)`` — the continuous-part sample (already in
        fit space, i.e. log-transformed when applicable), the zero-inflation fraction
        and the transform actually applied.
    """
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    has_negative = bool(np.any(finite & (values < 0)))

    if has_negative:
        return values[finite], 0.0, False

    finite_values = values[finite]
    qq = float(np.sum(finite_values == 0) / finite_values.size) if finite_values.size else 0.0

    nonzero = values[finite & (values != 0)]
    if log_transform:
        with np.errstate(divide='ignore', invalid='ignore'):
            xb = np.log(nonzero)
        xb = xb[np.isfinite(xb)]
    else:
        xb = nonzero
    return xb, qq, log_transform


def kde_cdf(x, xb, h, qq, log_transform):
    """
    Evaluate the fitted KDE-based cumulative probability Hx at native value(s) x,
    reusing an already-fitted baseline (xb, h) and zero-inflation fraction (qq).

    Factors out the exact formula used internally by f_kde (mean of Gaussian
    kernel CDFs centered on each baseline point, mixed with a point mass `qq`
    at zero), so it can be reused both by f_kde itself and by
    native_to_spi/_process_grid_point_trends to standardize a *new* value
    against an already-fitted baseline, without refitting.

    Args:
        x (float or ndarray): native-unit value(s) to evaluate.
        xb (ndarray): baseline sample used for the KDE fit (already
            log-transformed if log_transform=True), as in out_params['xb'].
        h (float): kernel bandwidth used for the fit, as in out_params['h'].
        qq (float): zero-inflation fraction of the fit, as in out_params['qq'];
            0 when the fitted variable is real-valued (see kde_fit_sample).
        log_transform (bool): whether x must be log-transformed before
            evaluation, as in out_params['log_transform'].

    Returns:
        float or ndarray: cumulative probability Hx (not clipped — matches
        native_to_spi's f_spi/f_spei branches, which don't clip either).
    """
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    is_scalar = np.ndim(x) == 0

    zero_mask = (x_arr == 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        xfull = np.log(x_arr) if log_transform else x_arr
    finite_mask = np.isfinite(xfull)

    Gx = np.full(x_arr.shape, np.nan, dtype=float)
    Gx[finite_mask] = norm.cdf(
        (xfull[finite_mask, None] - xb[None, :]) / h
    ).mean(axis=1)

    Hx = np.full(x_arr.shape, np.nan, dtype=float)
    Hx[finite_mask] = qq + (1 - qq) * Gx[finite_mask]
    # Pin x == 0 to the point mass only when there IS one. With qq == 0 the fit is a
    # plain KDE over the reals (see kde_fit_sample), and zero is an ordinary interior
    # point whose cumulative probability is whatever the kernels say — forcing it to
    # 0.0 there put H(0) below H(-1).
    if qq > 0:
        Hx[zero_mask] = qq

    return float(Hx[0]) if is_scalar else Hx


def kde_ppf(Hx, xb, h, qq, log_transform, n_grid=2000):
    """
    Invert kde_cdf: cumulative probability Hx -> native-unit value(s).

    There is no closed form for the inverse of a Gaussian-kernel CDF mixture,
    so this builds a monotonic grid of native values spanning the baseline
    (+/- 6 bandwidths — norm.cdf(6) is already within 1e-9 of 1, well beyond
    the 3.17e-5 tail clip used throughout the library), evaluates kde_cdf on
    the whole grid at once, and interpolates with np.interp (which clamps to
    the grid's edge values outside its range, rather than extrapolating).
    Values of Hx at or below the zero-inflation fraction `qq` map to exactly
    0.0, matching the point mass at zero in the forward transform.

    Args:
        Hx (float or ndarray): cumulative probability value(s) to invert.
        xb, h, qq, log_transform: same fitted-baseline parameters as kde_cdf.
        n_grid (int): number of grid points used for the interpolation.

    Returns:
        float or ndarray: native-unit value(s).

    Raises:
        ValueError: if the baseline bandwidth h is not finite or not
            positive (degenerate baseline — e.g. a single point, or
            all-identical values).
    """
    if not np.isfinite(h) or h <= 0:
        raise ValueError(
            f"kde_ppf: invalid bandwidth h={h!r} (baseline too small or "
            f"degenerate); cannot invert the KDE-based CDF."
        )

    Hx_arr = np.atleast_1d(np.asarray(Hx, dtype=float))
    is_scalar = np.ndim(Hx) == 0

    out = np.zeros(Hx_arr.shape, dtype=float)
    above = Hx_arr > qq

    if np.any(above):
        lo = np.min(xb) - 6 * h
        hi = np.max(xb) + 6 * h
        grid = np.linspace(lo, hi, n_grid)
        grid_Gx = norm.cdf((grid[:, None] - xb[None, :]) / h).mean(axis=1)
        grid_Hx = qq + (1 - qq) * grid_Gx
        grid_Hx = np.maximum.accumulate(grid_Hx)  # guard vs float non-monotonicity

        vals = np.interp(Hx_arr[above], grid_Hx, grid)
        out[above] = np.exp(vals) if log_transform else vals

    return float(out[0]) if is_scalar else out


# ===================================================================
#  f_kde — Non-parametric KDE (for any data)
# ===================================================================
def f_kde(prec, stride, m, m_cal, tb1, tb2, log_transform=True, fit_params=None):
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
        fit_params (dict or None): Optional pre-computed KDE info.

    Returns:
        tuple:
            - idmesi_all (ndarray): month indices for the full period.
            - spi (ndarray): standardized index values.
            - coef (ndarray): degree-3 polyfit coefficients for inverse mapping
              (approximate; retained for backward compatibility only).
            - out_params (dict or None): KDE metadata if fitted, else None.
              Keys: 'kde', 'bw_factor', 'n_fit', 'fit_domain' (fitted object
              and diagnostics, not used for computation), plus 'xb', 'h',
              'qq', 'log_transform' — the exact ingredients actually used by
              kde_cdf/kde_ppf for exact forward/reverse conversion of new
              values against this fit (see native_to_spi/spi_to_native).
    """
    xbase, x, idmesi_all = _resolve_indices(prec, stride, m, m_cal, tb1, tb2)



    # --- Pre-allocate output ---
    spi = np.full_like(x, np.nan, dtype=float)
    finite_x = np.isfinite(x)
    finite_xbase = np.isfinite(xbase)

    # --- The fit's three ingredients, decided on the BASELINE ---
    # xb / qq / log_transform ARE the fitted distribution, so all three must come
    # from the same sample. Deciding any of them on the full period would let data
    # appended later change the calibration — and hence already-published SPI values.
    # The rules live in kde_fit_sample so the diagnostic fits in statistics.py apply
    # exactly the same ones (see that function).
    xb_baseline, qq, log_transform_resolved = kde_fit_sample(xbase, log_transform)
    if log_transform and not log_transform_resolved:
        warnings.warn(
            f"f_kde: negative data detected in month {m} - month-scale {stride} → "
            f"log_transform disabled, and the exact-zero point mass dropped "
            f"(qq=0): a real-valued variable has no floor for values to pile up on.",
            RuntimeWarning, stacklevel=2
        )
    log_transform = log_transform_resolved

    finite_xbase = np.isfinite(xbase)

    if log_transform:
        with np.errstate(divide='ignore', invalid='ignore'):
            x_log = np.log(x)
            xbase_log = np.log(xbase)
        finite_x = np.isfinite(x_log)
        finite_xbase = np.isfinite(xbase_log)
        # Negative values are outside the domain the model was calibrated on:
        # they yield NaN rather than silently switching to a different model.
        if np.any(np.isfinite(x) & (x < 0)):
            warnings.warn(
                f"f_kde: negative values outside the log-calibrated domain "
                f"(month {m}, month-scale {stride}) → SPI is NaN for those months.",
                RuntimeWarning, stacklevel=2
            )

    # Early exit if no valid data. out_params is None whichever branch we were on:
    # with fit_params given there is nothing new to report, and without it no fit was
    # possible. (This used to read `None if fit_params is None else None`.) Callers
    # that cache the fit therefore store None for this (scale, month) — see
    # _get_kde_fit, which tells the two cases apart.
    if not np.any(finite_x):
        return idmesi_all, spi, np.full(4, np.nan), None

    # --- KDE fit (baseline) ---
    if fit_params is None:
        xb = xb_baseline

        if xb.size < 10:
            warnings.warn(
                f"KDE baseline sample size is very small (n={xb.size}). "
                f"Fit may be unreliable.",
                RuntimeWarning, stacklevel=2
            )

        kde = gaussian_kde(xb, bw_method="silverman")
        # Silverman's rule of thumb: ddof=1 on purpose — the rule is defined on the
        # unbiased sample sigma. Do not "align" it with the ddof=0 used to
        # standardize against the baseline (see BaseDroughtAnalysis._zscore_baseline):
        # they are different conventions for different purposes.
        h = 0.9 * np.std(xb, ddof=1) * xb.size ** (-1 / 5)
        out_params = {
            "kde": kde,
            "bw_factor": float(kde.factor),
            "n_fit": int(xb.size),
            "fit_domain": "xbase (finite, full R)",
            # Exact-inverse ingredients (see kde_cdf/kde_ppf): xb/h/qq/log_transform
            # are what the CDF below actually uses — NOT `kde`/`bw_factor`, which
            # are fitted independently and never consulted for the computation.
            "xb": xb,
            "h": h,
            "qq": qq,
            "log_transform": log_transform,
        }
    else:
        kde = fit_params.get("kde", None)
        if kde is None:
            raise ValueError("kde_params provided but missing key 'kde'.")

        # Honour the stored calibration: xb/h/qq/log_transform ARE the fitted
        # distribution, so reusing them is what keeps the new values in the
        # calibration's reference frame (the whole point of passing fit_params —
        # e.g. scenarios/forecast, where ts is modified but SPI must stay
        # comparable to the original fit). Recomputing them here would silently
        # recalibrate on the modified data instead.
        if all(key in fit_params for key in ("xb", "h", "qq", "log_transform")):
            xb = fit_params["xb"]
            h = fit_params["h"]
            qq = fit_params["qq"]
            log_transform = fit_params["log_transform"]
        else:
            warnings.warn(
                f"f_kde: fit_params provided without the calibration keys "
                f"('xb', 'h', 'qq', 'log_transform') — recalibrating on the supplied "
                f"data (month {m}, month-scale {stride}). The result is NOT in the "
                f"reference frame of the fit that was passed in.",
                RuntimeWarning, stacklevel=2
            )
            xb, qq, log_transform = kde_fit_sample(xbase, log_transform)
            h = 0.9 * np.std(xb, ddof=1) * xb.size ** (-1 / 5)
        out_params = None

    # --- CDF evaluation (exact; same formula reused by native_to_spi/spi_to_native) ---
    Hx = kde_cdf(x, xb, h, qq, log_transform)

    # --- Normal-score transform ---
    Hx = np.clip(Hx, PROB_CLIP, 1 - PROB_CLIP)
    spi = np.round(norm.ppf(Hx), 4)

    # --- Reverse mapping (polynomial approximation, retained for backward
    # compatibility only — the exact inverse is kde_ppf, used by
    # native_to_spi/spi_to_native/spatial_trends) ---
    coef = _reverse_polyfit(spi, xbase, m_cal, idmesi_all, tb1, tb2, stride, m)

    return idmesi_all, spi, coef, out_params


# ===================================================================
#  f_zscore — Gaussian z-score (for normally distributed data)
# ===================================================================
def f_zscore(data, stride, m, m_cal, tb1, tb2,fit_params=None):
    """
    Calculate the Z-Score for a time series of monthly data using a specified baseline period.

    Args:
        data (numpy.ndarray): Monthly series, shape (n,).
        stride (int): Accumulation period.
        m (int): Reference month (1–12).
        m_cal (numpy.ndarray): Calendar (n, 2) with [month, year].
        tb1 (int): Baseline start year.
        tb2 (int): Baseline end year.
        fit_params (list, optional): Precomputed [baseline_mean, baseline_std].
            If None, statistics are estimated from the baseline period.

    Returns:
        tuple:
            - idmesi_all (ndarray): month indices for the full period.
            - zscore (ndarray): standardized z-score values.
            - z_params (list): [baseline_mean, baseline_std] used for standardization.
            - out_params (list or None): [baseline_mean, baseline_std] if fitted
              from scratch, else None. Consistent with the 4-tuple interface
              of f_spi, f_spei, and f_kde.
    """
    xbase, x, idmesi_all = _resolve_indices(data, stride, m, m_cal, tb1, tb2)


    if fit_params is None:
        baseline_mean = np.nanmean(xbase)
        baseline_std = np.nanstd(xbase)
        out_params = [baseline_mean, baseline_std]
    else:
        baseline_mean, baseline_std = fit_params  # riuso parametri salvati
        out_params = None

    zscore = (x - baseline_mean) / baseline_std if baseline_std != 0 else np.zeros_like(x)

    return idmesi_all, zscore, [baseline_mean, baseline_std], out_params


# ===================================================================
#  c2r_eval — correct "reverse" evaluation of c2r coefficients
# ===================================================================
def c2r_eval(coeff, x, calculation_method):
    """
    Evaluate the c2r "reverse" coefficients at x (standardized index -> native units).

    For f_spi/f_spei/f_kde, `coeff` holds degree-3 polynomial coefficients
    (from `_reverse_polyfit`) and must be evaluated with `np.polyval`. For
    f_zscore, `coeff` instead holds `[baseline_mean, baseline_std]` (not
    polynomial coefficients) and must be evaluated as `x * std + mean`:
    calling `np.polyval([mean, std], x)` on it computes `mean * x + std`,
    which only coincides with the correct result by coincidence at x=1 and
    is wrong everywhere else (including x=0, the "normal value" reference
    used throughout the library).

    Args:
        coeff (ndarray): c2r coefficients for one (scale, month) slice —
            shape (4,) for f_spi/f_spei/f_kde, shape (2,) for f_zscore.
        x (float or ndarray): standardized index value(s) to convert back.
        calculation_method (callable): the object's `calculation_method`
            (f_spi, f_spei, f_kde, f_zscore, or a `functools.partial` thereof).

    Returns:
        float or ndarray: native-unit value(s).
    """
    from functools import partial
    base_func = calculation_method.func if isinstance(calculation_method, partial) else calculation_method
    if base_func is f_zscore:
        mean, std = coeff
        return x * std + mean
    return np.polyval(coeff, x)


#  Domain requirements  (dominio della distribuzione, non della classe)
# ===================================================================
f_spi.requires_positive    = True    # Gamma → (0, +∞)
f_spei.requires_positive   = False   # Pearson III → ℝ
f_kde.requires_positive    = False   # KDE → ℝ

# ===================================================================
#  Numeric fit-parameter width, per method
# ===================================================================
# How many floats each method's `params` (4th return value) carries, so that
# core.py sizes `self.fit_params` from the method itself instead of hard-coding
# a width per branch — the widths stopped agreeing when f_spi gained its
# zero-inflation fraction `qq` as a fourth parameter.
# f_kde's real calibration is a dict living in `fit_params_kde`; its slot in the
# numeric array is never written, and the value here only keeps the array
# allocatable.
f_spi.n_fit_params     = 4   # (alpha, loc, beta, qq)
f_spei.n_fit_params    = 3   # (c, loc, scale)
f_kde.n_fit_params     = 3   # unused placeholder — see fit_params_kde
f_zscore.n_fit_params  = 2   # (baseline_mean, baseline_std)
f_zscore.requires_positive = False   # z-score → ℝ