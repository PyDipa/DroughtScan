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
import matplotlib.pyplot as plt

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


def effective_month_weights(K=36, plot=True, ax=None):
    """
    Real per-calendar-month weight of D(SPI) for a fixed K, once SPI_1..SPI_K
    are expanded back into their own raw K-month accumulation windows.

    D(SPI)(t) = sum_k w_k * SPI_k(t), and SPI_k(t) is itself built from a
    k-month window P(t), P(t-1), ..., P(t-k+1). So month-lag j (0 = current
    month, j=1 = last month, ..., j=K-1 = the oldest one in the window)
    enters SPI_k for every k >= j+1, weighted there by w_k. Summing over
    those k gives the "real"/effective weight of lag j across the whole
    D(SPI) computation:

        W_eff(j) = sum_{k=j+1}^{K} w_k          j = 0, ..., K-1

    i.e. the reverse cumulative sum of generate_weights(K). W_eff(0) =
    sum_k w_k = 1 for every scheme (the current month always belongs to
    every window, from SPI_1 to SPI_K); what differs between schemes is how
    fast W_eff decays for j>0 - DW schemes (weight concentrated on short
    scales) decay fast, IW schemes (weight concentrated on long scales) stay
    high for most of the window and only fall off near j=K-1.

    Note: W_eff can never INCREASE with lag, for any non-negative weight
    vector - W_eff(j) - W_eff(j+1) = w_{j+1} >= 0 always, because a month
    further back is only ever a subset of the scales that include a more
    recent one. The flattest possible profile (W_eff(j)=1 for every j) is
    the limiting case of putting all the weight on the single longest scale
    (w_K=1); there is no way to get a genuinely increasing-with-lag kernel
    out of a weighted sum of nested SPI_k windows - that needs a kernel
    built directly on raw months instead (see benchmark_nash /
    benchmark_ihacres / benchmark_convolution's h(j)).

    Args:
        K (int, optional): number of scales / months of memory. Default 36
            (the library's usual scale for whole-basin figures).
        plot (bool, optional): if True (default), draw W_eff(lag) for the 5
            weighting schemes on `ax` (or a new figure/axes if `ax` is None).
        ax (matplotlib.axes.Axes, optional): axes to draw into. Only used
            when plot=True.

    Returns:
        ndarray, shape (K, 5): W_eff per lag (rows, 0 = current month) and
        weighting scheme (columns, generate_weights/WEIGHT_LABELS order).
    """
    # Lazy import: statistics.py imports FROM this module at the top level,
    # so importing it back here would be circular - see generate_weights'
    # callers in statistics.py for the same pattern.
    from drought_scan.utils.statistics import WEIGHT_LABELS, WEIGHT_COLORS

    W = generate_weights(K)                     # (K, 5): w_k per scale k=1..K
    W_eff = np.cumsum(W[::-1], axis=0)[::-1]     # (K, 5): reverse cumsum

    if plot:
        lag = np.arange(K)
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4.5))
        else:
            fig = ax.figure
        for i, lab in enumerate(WEIGHT_LABELS):
            ax.plot(lag, W_eff[:, i],  lw=2,
                    color=WEIGHT_COLORS[i], label=lab)
        ax.set_xlabel('lag (months)', fontsize=12)
        ax.set_ylabel('effective weight', fontsize=12)
        idname = rf"$\mathrm{{D}}_{{\mathrm{{(SPI)}}}}$"
        ax.set_title(f'whole single-month weight in {idname}, K={K}', fontsize=14)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=12)
        fig.tight_layout()

    return W_eff

def effective_weights_at_optimum(best_k, plot=True, ax=None, title=None, n_lags=None):
    """
    Effective per-month weight W_eff(j) of each weighting scheme, each one
    evaluated at ITS OWN optimal K, plus the mean age of that weight.

    Companion of ``effective_month_weights``. That function compares the five
    schemes at the SAME K (e.g. K=36, Fig. A1): there decreasing schemes look
    steep and increasing ones gentle. This one compares them at the K each
    scheme actually selects on the data, which is what the R2(w, K) optimum
    describes. The two pictures can differ: an increasing scheme peaking at a
    short K can retain LESS memory than a decreasing scheme peaking at a long K.

    Mean age (months before t, 0 = current month):
        age(w) = sum_j j * W_eff(j) / sum_j W_eff(j)
    i.e. the centre of mass of the W_eff curve. Unlike K, it is comparable
    across schemes. Same linear reading of D(SPI) as effective_month_weights
    (the per-scale standardization of SPI_k is not included).

    Args:
        best_k (sequence of 5 int, or dict {label: K}): optimal K per scheme,
            in generate_weights / WEIGHT_LABELS order if a sequence - e.g.
            ``analyze_correlation(...)["best_k_per_weight"]`` or
            ``analyze_correlation_seasonal(...)[season]["best_k_per_weight"]``.
        plot (bool): draw the curves (default True).
        ax (matplotlib Axes, optional): axes to draw into.
        title (str, optional): panel title.
        n_lags (int, optional): x-axis extent in months (default max K + 1).

    Returns:
        dict {label: {"K": int, "W_eff": ndarray (K,), "mean_age": float,
                      "median_lag": int}}
        median_lag = first lag at which half of the total effective weight
        has been accumulated.
    """
    # Lazy import, same pattern as effective_month_weights (avoids a circular import).
    from drought_scan.utils.statistics import WEIGHT_LABELS, WEIGHT_COLORS

    if isinstance(best_k, dict):
        ks = [int(best_k[lab]) for lab in WEIGHT_LABELS]
    else:
        ks = [int(k) for k in np.ravel(best_k)]
    if len(ks) != len(WEIGHT_LABELS):
        raise ValueError(f"best_k must give one K per scheme "
                         f"({len(WEIGHT_LABELS)}), got {len(ks)}.")
    if min(ks) < 1:
        raise ValueError("every K must be >= 1.")

    out = {}
    for w, (lab, K) in enumerate(zip(WEIGHT_LABELS, ks)):
        we = np.cumsum(generate_weights(K)[::-1, w])[::-1]     # W_eff(j), j = 0..K-1
        j = np.arange(K)
        out[lab] = {
            "K": K,
            "W_eff": we,
            "mean_age": float(np.sum(j * we) / np.sum(we)),
            "median_lag": int(np.searchsorted(np.cumsum(we) / np.sum(we), 0.5)),
        }

    if plot:
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4.5))
        n = int(n_lags) if n_lags is not None else max(ks) + 1
        lags = np.arange(n)
        for w, lab in enumerate(WEIGHT_LABELS):
            d = out[lab]
            y = np.zeros(n)
            m = min(d["K"], n)
            y[:m] = d["W_eff"][:m]
            col = WEIGHT_COLORS[w]
            ax.plot(lags, y, "-o", color=col, ms=3, lw=1.8,
                    label=f"{lab}  K={d['K']}, mean age={d['mean_age']:.1f}")
            # mean age as a small triangle on the x-axis, tip pointing down
            ax.plot(d["mean_age"], 0.0, marker="v", color=col, ms=8,
                    clip_on=False, zorder=5)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("lag (months, 0 = current month)")
        ax.set_ylabel("effective weight")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title(title if title else "Effective weight at each scheme's optimal K")
    return out


def plot_weight_families(K=36, plot=True, ax=None):
    """
    Plot the 5 weighting families themselves: w_k vs scale k, exactly as
    returned by generate_weights(K).

    Companion to effective_month_weights: that one shows the REAL weight a
    raw calendar month ends up carrying once SPI_1..SPI_K are expanded back
    into their own accumulation windows (the reverse cumulative sum of this
    one); this one shows the weight generate_weights assigns to each SCALE k
    directly, before any such expansion - the five curves that go into the
    D(SPI) weighted sum sum_k w_k * SPI_k(t).

    Args:
        K (int, optional): number of scales. Default 36 (the library's usual
            scale for whole-basin figures).
        plot (bool, optional): if True (default), draw w_k vs k for the 5
            weighting schemes on `ax` (or a new figure/axes if `ax` is None).
        ax (matplotlib.axes.Axes, optional): axes to draw into. Only used
            when plot=True.

    Returns:
        ndarray, shape (K, 5): same as generate_weights(K) - w_k per scale
        (rows, k=1..K) and weighting scheme (columns, WEIGHT_LABELS order).
    """
    # Lazy import: statistics.py imports FROM this module at the top level,
    # so importing it back here would be circular - same pattern as
    # effective_month_weights above.
    from drought_scan.utils.statistics import WEIGHT_LABELS, WEIGHT_COLORS

    W = generate_weights(K)

    if plot:
        k_range = np.arange(1, K + 1)
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4.5))
        else:
            fig = ax.figure
        for i, lab in enumerate(WEIGHT_LABELS):
            ax.plot(k_range, W[:, i], lw=2, color=WEIGHT_COLORS[i], label=lab)
        ax.set_xlabel('scale k', fontsize=12)
        ax.set_ylabel(r'weight $w_k$', fontsize=12)
        ax.set_title(f'The 5 weighting families, K={K}', fontsize=14)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=12)
        fig.tight_layout()

    return W


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
def _kde_bandwidth_and_centres(xb, variance_correction=True):
    """
    Silverman bandwidth for ``xb``, and the kernel centres the KDE should actually
    use - the two halves of one decision, so they cannot be computed from
    different samples (they used to be: ``kde_fit_sample`` returned the sample and
    both ``f_kde`` and ``statistics._fit_single_dist`` then re-derived ``h`` from it
    with their own copy of the formula).

    A Gaussian-kernel mixture centred on the sample points has variance
    ``s0**2 + h**2`` (s0 = population sd of ``xb``), not ``s0**2``: the kernels add
    their own spread on top of the data's. Evaluating that over-dispersed CDF back
    at the very points it was fitted on therefore compresses the normal scores by
    ``1/sqrt(1 + h**2/s0**2)`` - 0.907 at n=30, and it depends on n alone, not on
    the variable - so an index that is supposed to be N(0,1) over its baseline came
    out at sd ~0.87 instead. Measured on the Po (1981-2010): P(SPI <= -1) was 12.5%
    of baseline months against the nominal 15.87%, while f_spi gave 15.83%.

    With ``variance_correction`` the centres are shrunk toward their mean by
    ``c = sqrt(1 - h**2/s0**2)``, which makes the mixture variance exactly
    ``h**2 + c**2*s0**2 == s0**2``. The bandwidth is deliberately NOT rescaled: it
    stays Silverman's rule on the ORIGINAL sample, because the rule is about how
    much to smooth, not about how wide the result should be. The fit is therefore
    as smooth as before, only correctly scaled - and every consumer that reads
    ``xb``/``h`` out of the stored fit (``kde_cdf``, ``kde_ppf``, ``_kde_pdf``,
    ``statistics._mixture_cdf``/``_mixture_pdf``, ``native_to_spi``,
    ``spi_to_native``, ``_process_grid_point_spi``) inherits the correction with no
    change of its own.

    References
    ----------
    The variance inflation is a standard property of non-negative kernel density
    estimates, treated at length in the density-estimation literature but not in
    the drought-index one - which is where a KDE-based SPI falls between two
    chairs, since the mainstream non-parametric standardized index (Farahmand &
    AghaKouchak 2015) uses empirical plotting positions rather than a kernel CDF
    and so never meets this bias at all.

    - Jones, M.C. (1991). On correcting for variance inflation in kernel density
      estimation. Computational Statistics & Data Analysis 11(1), 3-15.
      doi:10.1016/0167-9473(91)90049-8  <- the dedicated treatment of exactly
      this correction.
    - Silverman, B.W. & Young, G.A. (1987). The bootstrap: to smooth or not to
      smooth? Biometrika 74(3), 469-479. Introduces the shrinkage as the way to
      preserve the variance structure; they apply it to smoothed-bootstrap draws,
      here it is folded into the fit itself.
    - Silverman, B.W. (1986). Density Estimation for Statistics and Data
      Analysis. Chapman & Hall. Source of the 0.9*sigma*n**(-1/5) rule used above.
    - Farahmand, A. & AghaKouchak, A. (2015). A generalized framework for deriving
      nonparametric standardized drought indicators. Advances in Water Resources
      76, 140-145. The plotting-position route, for contrast.

    Cheapest way to catch a regression here without reading any of the above:
    a standardized index must have sd ~1 over its own baseline, and
    P(SPI <= -1) must come out near 0.1587. Both are one line of numpy on
    `spi_like_set[0]` restricted to the baseline.

    What this does NOT remove: a non-parametric CDF evaluated in-sample still puts
    its own points at plotting positions ~(i-0.5)/n, whose normal scores have sd
    slightly below 1 (0.979 at n=30, 0.995 at n=120). That residual is a property
    of in-sample evaluation, not a bias of the fit, and it vanishes as n grows -
    which is why the corrected baseline sd lands at 0.967, not exactly 1.

    Args:
        xb (ndarray): the continuous-part sample, already in fit space
            (log-transformed upstream when applicable).
        variance_correction (bool): ``False`` restores the uncorrected behaviour,
            for reproducing numbers published before this correction existed.

    Returns:
        tuple: ``(xb_centres, h)`` - the centres to store in the fit, and the
        bandwidth. ``h`` is NaN when the sample is too small to define one.
    """
    xb = np.asarray(xb, dtype=float)
    if xb.size < 2:
        return xb, np.nan
    # Silverman's rule of thumb: ddof=1 on purpose - the rule is defined on the
    # unbiased sample sigma. Do not "align" it with the ddof=0 used to standardize
    # against the baseline (see BaseDroughtAnalysis._zscore_baseline): they are
    # different conventions for different purposes. Note this is NOT scipy's own
    # bw_method="silverman" factor either, whose constant differs from this 0.9.
    h = 0.9 * np.std(xb, ddof=1) * xb.size ** (-1 / 5)
    if not variance_correction or not np.isfinite(h) or h <= 0:
        return xb, h
    s0 = float(np.std(xb, ddof=0))
    ratio = (h / s0) ** 2 if s0 > 0 else np.inf
    if not np.isfinite(ratio) or ratio >= 1.0:
        # Degenerate: the kernel is as wide as the data itself. Only reachable at
        # n=2 (where h/s0 = 1.11) or on a constant sample; there is nothing to
        # correct without inverting the density, so leave the centres alone.
        return xb, h
    mean = float(np.mean(xb))
    return mean + (xb - mean) * np.sqrt(1.0 - ratio), h


def kde_fit_sample(values, log_transform=True, variance_correction=True):
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
        return _kde_bandwidth_and_centres(values[finite], variance_correction) + (0.0, False)

    finite_values = values[finite]
    qq = float(np.sum(finite_values == 0) / finite_values.size) if finite_values.size else 0.0

    nonzero = values[finite & (values != 0)]
    if log_transform:
        with np.errstate(divide='ignore', invalid='ignore'):
            xb = np.log(nonzero)
        xb = xb[np.isfinite(xb)]
    else:
        xb = nonzero
    return _kde_bandwidth_and_centres(xb, variance_correction) + (qq, log_transform)


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
    xb_baseline, h_baseline, qq, log_transform_resolved = kde_fit_sample(xbase, log_transform)
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
        # Bandwidth AND centres are one decision, taken in kde_fit_sample - see
        # _kde_bandwidth_and_centres for why they cannot be settled separately.
        h = h_baseline
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
            xb, h, qq, log_transform = kde_fit_sample(xbase, log_transform)
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

# ===================================================================
#  Standardization-method registry
# ===================================================================
# Everything the library needs to know about a standardization method, in ONE
# place: how to go native -> index and index -> native from the fitted params,
# how wide its numeric params / c2r rows are, whether its params are numeric
# (stored in the `fit_params` array) or an opaque object (stored in
# `fit_params_kde`), and its domain. core.py asks the registry instead of
# branching on `f_spi / f_spei / f_kde / f_zscore` in every method, so adding or
# fixing a method touches one entry, and a user-defined method works as soon as
# it is registered with `register_standardization`.

def _clip_prob(p):
    return np.clip(p, PROB_CLIP, 1 - PROB_CLIP)


def _spi_to_index(x, params):
    # zero-inflated Gamma mixture: the exact forward transform f_spi applies
    a, loc, beta, qq = params
    return norm.ppf(_clip_prob(gamma_cdf_zi(x, a, loc, beta, qq)))


def _spi_to_native(z, params):
    a, loc, beta, qq = params
    return gamma_ppf_zi(_clip_prob(norm.cdf(z)), a, loc, beta, qq)


def _spei_to_index(x, params):
    c, loc, scale = params
    return norm.ppf(_clip_prob(pearson3.cdf(x, c, loc=loc, scale=scale)))


def _spei_to_native(z, params):
    c, loc, scale = params
    return pearson3.ppf(_clip_prob(norm.cdf(z)), c, loc=loc, scale=scale)


def _kde_to_index(x, fit):
    return norm.ppf(_clip_prob(kde_cdf(x, fit['xb'], fit['h'], fit['qq'], fit['log_transform'])))


def _kde_to_native(z, fit):
    return kde_ppf(norm.cdf(z), fit['xb'], fit['h'], fit['qq'], fit['log_transform'])


def _zscore_to_index(x, params):
    mean, std = params
    return (x - mean) / std


def _zscore_to_native(z, params):
    mean, std = params
    return z * std + mean


class StandardizationMethod:
    """
    Registry entry for one standardization method.

    Attributes:
        func (callable): the method itself, with the f_* signature
            ``func(ts, stride, m, m_cal, tb1, tb2, fit_params=None)
            -> (indices, index_values, c2r_coeff, params)``.
        name (str): display name.
        to_index (callable): ``to_index(native_values, params) -> index values``
            for one (scale, month) fit.
        to_native (callable): ``to_native(index_values, params) -> native values``,
            the exact inverse of ``to_index``.
        n_fit_params (int): width of ``params`` when numeric (size of the
            ``fit_params`` array's last axis).
        n_c2r (int): width of the (deprecated) c2r coefficient row.
        numeric_params (bool): True if ``params`` is a flat numeric vector
            (stored in ``fit_params``); False for an opaque object (stored in
            ``fit_params_kde``, e.g. f_kde's dict).
        requires_positive (bool): the method is defined on positive data only.
    """
    __slots__ = ("func", "name", "to_index", "to_native", "n_fit_params",
                 "n_c2r", "numeric_params", "requires_positive")

    def __init__(self, func, name, to_index, to_native, n_fit_params, n_c2r,
                 numeric_params, requires_positive):
        self.func, self.name = func, name
        self.to_index, self.to_native = to_index, to_native
        self.n_fit_params, self.n_c2r = int(n_fit_params), int(n_c2r)
        self.numeric_params = bool(numeric_params)
        self.requires_positive = bool(requires_positive)

    def __repr__(self):
        return f"StandardizationMethod({self.name})"


_STANDARDIZATION_REGISTRY = {}


def register_standardization(func, *, to_index, to_native, n_fit_params, n_c2r=4,
                             numeric_params=True, requires_positive=False, name=None):
    """
    Register a standardization method so every DroughtScan class can use it as
    ``calculation_method`` (forward fit, reverse transforms, spatial maps).

    Args:
        func (callable): the method, with the f_* signature (see
            ``StandardizationMethod``).
        to_index, to_native (callable): exact forward / inverse transforms for one
            (scale, month) fit, ``(values, params) -> values``.
        n_fit_params (int): number of floats in ``params`` (if numeric).
        n_c2r (int): width of the c2r coefficient row the method returns. Default 4.
        numeric_params (bool): False if ``params`` is an opaque object. Default True.
        requires_positive (bool): reject series with negative values. Default False.
        name (str, optional): display name; defaults to ``func.__name__``.

    Returns:
        StandardizationMethod
    """
    spec = StandardizationMethod(func, name or func.__name__, to_index, to_native,
                                 n_fit_params, n_c2r, numeric_params, requires_positive)
    _STANDARDIZATION_REGISTRY[func] = spec
    # mirrored on the function for code that still reads the attributes directly
    func.requires_positive = spec.requires_positive
    func.n_fit_params = spec.n_fit_params
    return spec


def get_standardization(calculation_method):
    """
    The registry entry of ``calculation_method`` (a registered function, or a
    ``functools.partial`` of one). Raises ValueError for an unregistered method.
    """
    from functools import partial
    base = calculation_method.func if isinstance(calculation_method, partial) else calculation_method
    try:
        return _STANDARDIZATION_REGISTRY[base]
    except (KeyError, TypeError):
        raise ValueError(
            f"Unknown calculation_method {base!r}: expected one of "
            f"{', '.join(spec.name for spec in _STANDARDIZATION_REGISTRY.values())} "
            f"(optionally wrapped in functools.partial), or a method registered with "
            f"drought_scan.utils.drought_indices.register_standardization."
        ) from None


# f_kde's real calibration is a dict living in `fit_params_kde`; its n_fit_params
# only keeps the (unused) numeric slot allocatable.
register_standardization(f_spi, to_index=_spi_to_index, to_native=_spi_to_native,
                         n_fit_params=4, n_c2r=4, requires_positive=True,
                         name="f_spi (zero-inflated Gamma)")
register_standardization(f_spei, to_index=_spei_to_index, to_native=_spei_to_native,
                         n_fit_params=3, n_c2r=4, name="f_spei (Pearson III)")
register_standardization(f_kde, to_index=_kde_to_index, to_native=_kde_to_native,
                         n_fit_params=3, n_c2r=4, numeric_params=False,
                         name="f_kde (Gaussian KDE)")
register_standardization(f_zscore, to_index=_zscore_to_index, to_native=_zscore_to_native,
                         n_fit_params=2, n_c2r=2, name="f_zscore (z-score)")
