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
import warnings
from typing import Optional
import matplotlib.pyplot as plt
from datetime import date,datetime,timedelta

from drought_scan.utils.drought_indices import (
    gamma_cdf_zi,
    gamma_ppf_zi,
    kde_cdf,
    kde_fit_sample,
    kde_ppf,
)


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
    Generates a new m_cal vector based on the relationship between m_cal1 and m_cal2.

    - If m_cal2 is fully contained within m_cal1, it returns m_cal1.
    - If m_cal2 partially overlaps with m_cal1, it returns their intersection.
    - If m_cal2 is contiguous with m_cal1, it returns their union.
    - If there is a gap between the two time ranges, it raises an error.

    Args:
        m_cal1 (np.ndarray): First calendar array [[month, year], ...].
        m_cal2 (np.ndarray): Second calendar array [[month, year], ...].

    Returns:
        np.ndarray: The combined/overlapping m_cal based on the above conditions.
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
#  Standardization Test | Parametric and non parametric fits
# ===================================================================
# Four distribution families are always evaluated:
#     Gaussian  |  Gamma  |  Pearson type III  |  Gaussian KDE (non-parametric)
#
# Model selection criterion
# -------------------------
# - KS statistic D  : primary, comparable across ALL four families (lower = better).
# - AIC             : secondary, parametric families only (Gaussian / Gamma / Pearson III).
#                     KDE is excluded from AIC ranking because no canonical k is defined.

# KS p-value caveat
# -----------------
# When distribution parameters are estimated by MLE from the same sample, the
# classical KS p-values are anti-conservative (Lilliefors 1967; Stephens 1974).
# They are reported here as indicative; for formal inference use the bootstrap-
# corrected p-values via `n_bootstrap > 0` (≥ 999 recommended).

_ALL_DISTS = ("gaussian", "gamma", "pearson3", "kde")


# ===================================================================
#  Zero-inflation mixture helpers, shared by every fitting/plotting
#  function below — the single place that decides how "gamma" and "kde"
#  handle exact zeros, so it can never again drift out of sync with
#  f_spi/f_kde (drought_indices.py) or between the functions in this file.
#
#  Convention (matches f_spi/f_kde exactly): qq = P(X=0), estimated as the
#  empirical fraction of exact zeros in the sample. The continuous family
#  (Gamma or KDE) is fitted on the strictly-positive subset ONLY — zeros are
#  masked out of the fit, never shifted or left in. Hx(x) = qq for x<=0,
#  Hx(x) = qq + (1-qq)*continuous_CDF(x) for x>0. "gaussian"/"pearson3" have
#  no zero-inflation handling, matching f_zscore/f_spei (real-valued data,
#  not naturally zero-inflated).
# ===================================================================
def _mixture_cdf(x, dist, params):
    """Zero-inflation-aware CDF, shared by KS tests and CDF plots."""
    x = np.asarray(x, dtype=float)
    if dist == "gaussian":
        return stats.norm.cdf(x, loc=params["mu"], scale=params["sigma"])
    if dist == "pearson3":
        return stats.pearson3.cdf(x, params["shape"], loc=params["loc"], scale=params["scale"])
    qq = params.get("qq", 0.0)
    if dist == "gamma":
        # Same implementation f_spi's forward transform uses, so a diagnostic
        # Gamma fit and the operational one cannot drift apart.
        shift = 1.0 if params["shift_applied"] else 0.0
        with np.errstate(invalid="ignore"):
            return gamma_cdf_zi(x, params["shape"], params["loc"], params["scale"],
                                qq, shift=shift)
    if dist == "kde":
        return kde_cdf(x, params["xb"], params["h"], qq, params["log_transform"])
    raise ValueError(f"Unknown distribution: '{dist}'")


def _mixture_ppf(Hx, dist, params):
    """
    Inverse of `_mixture_cdf`: cumulative probability -> native value.
    Used to draw a calibration curve (index domain -> native units) for
    whichever family actually won a given month, e.g. in a diagnostics
    report — NOT tied to any particular DSO's own fixed calculation_method.
    """
    Hx = np.asarray(Hx, dtype=float)
    if dist == "gaussian":
        return stats.norm.ppf(Hx, loc=params["mu"], scale=params["sigma"])
    if dist == "pearson3":
        return stats.pearson3.ppf(Hx, params["shape"], loc=params["loc"], scale=params["scale"])
    qq = params.get("qq", 0.0)
    if dist == "gamma":
        # Shared with spi_to_native's f_spi branch — see _mixture_cdf.
        shift = 1.0 if params["shift_applied"] else 0.0
        return gamma_ppf_zi(Hx, params["shape"], params["loc"], params["scale"],
                            qq, shift=shift)
    if dist == "kde":
        return kde_ppf(Hx, params["xb"], params["h"], qq, params["log_transform"])
    raise ValueError(f"Unknown distribution: '{dist}'")


def _kde_pdf(x, xb, h):
    """
    Native-space KDE density, evaluated with the SAME bandwidth `h` and
    baseline sample `xb` used by `kde_cdf` (the exact derivative of its
    CDF formula) — not scipy's own `gaussian_kde(...).evaluate()`, whose
    default bandwidth constant differs from f_kde's (0.9*std*n^-1/5).
    `xb`/`x` are already in fit space (log-transformed upstream if needed);
    callers apply the Jacobian correction for the log case themselves.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    return stats.norm.pdf((x[:, None] - xb[None, :]) / h).mean(axis=1) / h


def _mixture_pdf(x, dist, params):
    """
    Companion density to `_mixture_cdf`, for PDF plots only — the qq point
    mass at exactly 0 is not a density and is omitted; only the continuous
    part for x>0, scaled by (1-qq), is returned (0 elsewhere).
    """
    x = np.asarray(x, dtype=float)
    if dist == "gaussian":
        return stats.norm.pdf(x, loc=params["mu"], scale=params["sigma"])
    if dist == "pearson3":
        return stats.pearson3.pdf(x, params["shape"], loc=params["loc"], scale=params["scale"])
    qq = params.get("qq", 0.0)
    if dist == "gamma":
        shift = 1.0 if params["shift_applied"] else 0.0
        with np.errstate(invalid="ignore"):
            gpdf = stats.gamma.pdf(x + shift, params["shape"], loc=params["loc"], scale=params["scale"])
        return np.where(x > 0, (1 - qq) * gpdf, 0.0)
    if dist == "kde":
        xb, h, log_transform = params["xb"], params["h"], params["log_transform"]
        with np.errstate(divide="ignore", invalid="ignore"):
            x_fit = np.log(x) if log_transform else x
            dens_fit = _kde_pdf(np.where(np.isfinite(x_fit), x_fit, 0.0), xb, h)
            jac = (1.0 / x) if log_transform else 1.0
            pdf = (1 - qq) * dens_fit * jac
        # Below zero the density is 0 only when the variable is bounded there, i.e.
        # when a point mass at zero was fitted. A real-valued KDE (qq == 0, see
        # kde_fit_sample) has a perfectly good density at negative x, and clamping it
        # to 0 drew the left half of every P-PET / temperature fit as a flat line.
        in_support = (x > 0) if qq > 0 else np.isfinite(x)
        return np.where(in_support & np.isfinite(pdf), pdf, 0.0)
    raise ValueError(f"Unknown distribution: '{dist}'")


# helpers
# PRIVATE CORE FITTER  (single distribution, single clean array)
def _fit_single_dist(dataset,dist=None,shift_for_gamma = False):
    """
    Fit *one* distribution to a clean (finite, 1-D) array and return stats.

    Parameters
    ----------
    dataset : np.ndarray
        Finite-only values (caller must filter).
    dist : {"gaussian", "gamma", "pearson3", "kde"}
    shift_for_gamma : bool, default False
        If True, apply (positive part) + 1 before Gamma fitting. Defaults to
        False because `f_spi` does not shift: it fits `gamma.fit(x[x > 0],
        floc=0)` on the raw positive part, with the exact zeros carried by
        `qq` instead. Leaving this True made every diagnostic here score a
        Gamma fitted on `x + 1` — a different model from the one the library
        would actually apply, which is precisely what these functions exist to
        judge. Kept as an opt-in for otherwise-pathological positive-only
        samples only.

    Returns
    -------
    dict
        distribution, params, KS_statistic, KS_p_value, log_likelihood,
        AIC, k_params, error_percent, goodness_percent.
        `params` includes `"qq"` (P(X=0)) for "gamma" and "kde".
        error_percent/goodness_percent are a point-by-point mean CDF
        deviation, NOT derived from KS_statistic (which is the worst-case/
        max deviation instead) — see module note above `_mixture_cdf`.

    Raises
    ------
    ValueError
        If Gamma/KDE is requested and the strictly-positive subset of
        `dataset` (after masking exact zeros) is empty or still contains
        non-positive values.
    """
    if dist is None:
        dist = 'gaussian'
    dist = dist.lower()
    dataset = np.asarray(dataset, dtype=float)

    # ── Gamma ──────────────────────────────────────────────────────────────
    if dist == "gamma":
        # Zero-inflation mixture, matching f_spi exactly: qq = P(X=0) from
        # the full sample, Gamma fitted on the strictly-positive part only.
        qq = float(np.mean(dataset == 0)) if dataset.size else 0.0
        positive = dataset[dataset > 0]
        data_used = positive + 1.0 if shift_for_gamma else positive
        if data_used.size == 0 or np.any(data_used <= 0):
            raise ValueError(
                "Gamma distribution requires strictly positive values after "
                "masking exact zeros. Set shift_for_gamma=True or pre-process the data."
            )
        shape, loc, scale = stats.gamma.fit(data_used, floc=0)
        params = {"shape": shape, "loc": loc, "scale": scale,
                  "shift_applied": shift_for_gamma, "qq": qq}

        D, p_ks = stats.kstest(dataset, lambda x: _mixture_cdf(x, "gamma", params))

        logpdf_pos = stats.gamma.logpdf(data_used, shape, loc=loc, scale=scale)
        n_zero = dataset.size - positive.size
        ll_zero = n_zero * np.log(qq) if qq > 0 else 0.0
        ll_pos = float(np.sum(logpdf_pos)) + (positive.size * np.log(1 - qq) if qq < 1 else 0.0)
        log_likelihood = float(ll_zero + ll_pos)
        # qq is a plug-in empirical fraction, not an MLE-optimised free
        # parameter (same convention as f_spi), so it is not counted in k.
        k = 3

    # ── Pearson type III ───────────────────────────────────────────────────
    elif dist == "pearson3":
        data_used = dataset
        shape, loc, scale = stats.pearson3.fit(data_used)
        params = {"shape": shape, "loc": loc, "scale": scale}
        D, p_ks = stats.kstest(data_used, "pearson3", args=(shape, loc, scale))
        logpdf = stats.pearson3.logpdf(data_used, shape, loc=loc, scale=scale)
        log_likelihood = float(np.sum(logpdf))
        k = 3

    # ── Gaussian ────────────────────────────────────────────────────────────
    elif dist == "gaussian":
        data_used = dataset
        mu    = np.mean(data_used)
        sigma = np.std(data_used, ddof=0)
        if sigma == 0:
            raise ValueError("Gaussian fit requires non-zero variance.")
        params = {"mu": mu, "sigma": sigma}
        D, p_ks = stats.kstest(data_used, "norm", args=(mu, sigma))
        logpdf = stats.norm.logpdf(data_used, loc=mu, scale=sigma)
        log_likelihood = float(np.sum(logpdf))
        k = 2

    # ── Gaussian KDE (non-parametric) ───────────────────────────────────────
    elif dist == "kde":
        # The fit's three ingredients come from drought_indices.kde_fit_sample — the
        # same call f_kde makes — so this diagnostic judges exactly the sample the
        # library would fit. It used to compute them here: `dataset[dataset > 0]`,
        # which on a real-valued series (P-PET balance, temperature) discarded the
        # whole negative half, while f_kde kept it. The page then reported how well a
        # KDE described half the data.
        xb, qq, log_transform = kde_fit_sample(dataset)
        positive = dataset[dataset > 0]

        if xb.size < 2:
            raise ValueError(
                "KDE requires at least 2 finite values in the continuous part "
                "(exact zeros are masked out when the variable is bounded below)."
            )
        # Same bandwidth formula as f_kde (Silverman's rule of thumb,
        # ddof=1) — NOT scipy's own bw_method="silverman" factor, whose
        # constant differs from f_kde's 0.9.
        h = 0.9 * np.std(xb, ddof=1) * xb.size ** (-1 / 5)
        if not np.isfinite(h) or h <= 0:
            raise ValueError(
                "KDE bandwidth is degenerate (data may be constant after masking zeros)."
            )
        params = {
            "bw_method": "silverman",
            "h": h,
            "xb": xb,
            "qq": qq,
            "log_transform": log_transform,
            "n_fit": int(xb.size),
        }

        D, p_ks = stats.kstest(dataset, lambda x: _mixture_cdf(x, "kde", params))

        # Likelihood over the points the continuous part actually covers: the
        # non-zero ones when there is a point mass, every finite one when there
        # isn't (a real-valued fit — see kde_fit_sample).
        continuous = dataset[np.isfinite(dataset) & (dataset != 0)] if qq > 0 \
            else dataset[np.isfinite(dataset)]
        pdf_vals = _mixture_pdf(continuous, "kde", params) if continuous.size else np.array([])
        pdf_vals = np.clip(pdf_vals, 1e-300, None)
        n_zero = dataset.size - continuous.size
        ll_zero = n_zero * np.log(qq) if qq > 0 else 0.0
        ll_cont = float(np.sum(np.log(pdf_vals))) if pdf_vals.size else 0.0
        log_likelihood = float(ll_zero + ll_cont)
        k = None  # AIC undefined for KDE

    else:
        raise ValueError(
            f"Unknown distribution '{dist}'. "
            "Choose from: 'gaussian', 'gamma', 'pearson3', 'kde'."
        )

    # ── Derived metrics ─────────────────────────────────────────────────────
    aic = (2 * k - 2 * log_likelihood) if k is not None else np.nan

    # error_percent/goodness_percent are a POINT-BY-POINT MEAN absolute
    # deviation between the fitted and empirical CDF (evaluated at every
    # observed value), not the KS statistic D (which is only the worst-case/
    # max deviation, at a single point) — deliberately a different, milder
    # summary than KS_statistic, kept alongside it rather than derived from it.
    x_sorted = np.sort(dataset)
    F_emp = np.arange(1, x_sorted.size + 1) / x_sorted.size
    F_theo = _mixture_cdf(x_sorted, dist, params)
    mean_abs_error = float(np.mean(np.abs(F_theo - F_emp)))

    return {
        "distribution":    dist,
        "params":          params,
        "KS_statistic":    float(D),
        "KS_p_value":      float(p_ks),
        "log_likelihood":  log_likelihood,
        "AIC":             float(aic) if not np.isnan(aic) else np.nan,
        "k_params":        k,
        "error_percent":   100.0 * mean_abs_error,
        "goodness_percent": 100.0 * (1.0 - mean_abs_error),
    }

#  BOOTSTRAP KS p-VALUE CORRECTION  (optional)
def _bootstrap_ks_pvalue(dataset:np.ndarray,fit_result: dict,
    n_bootstrap: int = 999,rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Parametric bootstrap to correct KS p-values for the Lilliefors bias.

    Resample n_bootstrap synthetic datasets from the fitted distribution,
    refit, compute D* each time → empirical null distribution of D.
    Bootstrap p-value = fraction of D* ≥ D_observed.

    Parameters
    ----------
    dataset   : original data (filtered).
    fit_result: dict returned by _fit_single_dist.
    n_bootstrap: number of bootstrap replicates.
    rng       : optional numpy Generator for reproducibility.

    Returns
    -------
    float : bootstrap-corrected p-value.
    """
    if rng is None:
        rng = np.random.default_rng()

    dist   = fit_result["distribution"]
    params = fit_result["params"]
    n      = len(dataset)
    D_obs  = fit_result["KS_statistic"]
    D_boot = np.empty(n_bootstrap)

    qq = params.get("qq", 0.0)  # 0.0 for gaussian/pearson3 (no zero-inflation)

    for i in range(n_bootstrap):
        # Draw synthetic sample from the fitted distribution. For gamma/kde,
        # first decide exact-zero vs positive per point via `qq` (matching
        # the mixture the fit itself uses), then draw the positive part from
        # the continuous family.
        is_zero = rng.random(n) < qq if qq > 0 else np.zeros(n, dtype=bool)
        n_pos = int(np.sum(~is_zero))
        sample = np.zeros(n, dtype=float)

        if dist == "gaussian":
            sample = rng.normal(params["mu"], params["sigma"], size=n)
        elif dist == "gamma":
            pos_sample = stats.gamma.rvs(
                params["shape"], loc=params["loc"], scale=params["scale"],
                size=n_pos, random_state=rng.integers(1 << 31)
            )
            if params["shift_applied"]:
                pos_sample = pos_sample - 1.0          # undo shift for consistent comparison
            sample[~is_zero] = pos_sample
        elif dist == "pearson3":
            sample = stats.pearson3.rvs(
                params["shape"], loc=params["loc"], scale=params["scale"],
                size=n, random_state=rng.integers(1 << 31)
            )
        else:  # kde — smoothed bootstrap in fit space, same h/xb as the fit
            xb, h, log_transform = params["xb"], params["h"], params["log_transform"]
            idx = rng.integers(0, xb.size, size=n_pos)
            pos_sample_fit = xb[idx] + rng.normal(0, h, size=n_pos)
            sample[~is_zero] = np.exp(pos_sample_fit) if log_transform else pos_sample_fit

        # Refit and compute D*
        try:
            # Refit under the SAME convention as the fit under test: a shifted
            # null compared against an unshifted D_obs (or vice versa) is not the
            # null distribution of that statistic.
            res = _fit_single_dist(
                sample, dist,
                shift_for_gamma=bool(params.get("shift_applied", False)),
            )
            D_boot[i] = res["KS_statistic"]
        except Exception:
            D_boot[i] = np.nan

    valid = D_boot[np.isfinite(D_boot)]
    return float(np.mean(valid >= D_obs))

# Fit all four families to `dataset` and return a comparison dict.
def _analyze_all(dataset, shift_for_gamma = False,  n_bootstrap=0, seed=None):
    """
    Fit all four families to `dataset` and return a comparison dict.

    Selection rule
    --------------
    1. Lowest mean point-by-point CDF deviation (all 4) → primary recommendation.
    2. Lowest KS statistic D (all 4) → secondary note.
    3. Lowest AIC (parametric 3 only) → secondary note.

    Why the MEAN deviation and not KS: D is the single worst point of
    disagreement. A zero-inflated fit (gamma/kde on a month with exact zeros)
    has a genuine vertical jump of height `qq` at x=0, and D latches onto that
    jump — it returns ~qq no matter how well the curve tracks the data
    everywhere else. Measured on a synthetic gamma sample: at qq=0 the four
    families score 0.165/0.047/0.043/0.029 and kde correctly wins; at qq=0.31
    gamma, pearson3 and kde all collapse to exactly 0.315 and the Gaussian
    "wins" with 0.221 — solely because it has no jump to be penalised for.
    The mean deviation weights that single point in proportion to the others,
    so it ranks the families on how well they actually fit. On samples without
    exact zeros (e.g. Po basin-average rainfall, qq=0) the two criteria agree.
    """
    dataset = dataset[np.isfinite(dataset)]

    skewness = float(stats.skew(dataset))
    _, p_normal = stats.normaltest(dataset)

    fits: dict[str, dict] = {}
    rng = np.random.default_rng(seed)
    for d in _ALL_DISTS:
        try:
            res = _fit_single_dist(dataset, d, shift_for_gamma=shift_for_gamma)
            if n_bootstrap > 0:
                res["KS_p_value_bootstrap"] = _bootstrap_ks_pvalue(
                    dataset, res, n_bootstrap=n_bootstrap, rng=rng
                )
            fits[d] = res
        except Exception as exc:
            warnings.warn(f"Fitting '{d}' failed: {exc}", RuntimeWarning)
            fits[d] = None

    # ── Rank by mean point-by-point CDF deviation (lower = better) ────────
    valid_fits = {d: v for d, v in fits.items() if v is not None}
    best_error = min(valid_fits, key=lambda d: valid_fits[d]["error_percent"])
    best_ks    = min(valid_fits, key=lambda d: valid_fits[d]["KS_statistic"])

    # ── Best parametric by AIC ─────────────────────────────────────────────
    parametric = {d: v for d, v in valid_fits.items()
                  if d != "kde" and not np.isnan(v["AIC"])}
    best_aic = min(parametric, key=lambda d: parametric[d]["AIC"]) if parametric else None

    return {
        "skewness":         skewness,
        "normality_p_value": float(p_normal),
        "fits":             fits,
        "best_by_mean_error": best_error,  # primary — see the selection rule above
        "best_by_KS":       best_ks,       # secondary, kept for comparison
        "best_by_AIC":      best_aic,      # secondary, parametric families only
        "recommendation":   best_error,    # primary
    }


# CDF COMPARISON PLOT
def _plot_cdf_comparison(dataset,analysis,title = ""):
    """
    Empirical CDF (ECDF) vs theoretical CDFs for all fitted families.

    This is the most direct visual diagnostic: how closely does each
    fitted distribution track the empirical CDF?
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax_cdf, ax_pdf = axes

    dataset = dataset[np.isfinite(dataset)]
    # ECDF
    x_ecdf = np.sort(dataset)
    y_ecdf = np.arange(1, len(x_ecdf) + 1) / len(x_ecdf)

    colors = {
        "gaussian": "#2166ac",
        "gamma":    "#d6604d",
        "pearson3": "#4dac26",
        "kde":      "#9970ab",
    }
    labels = {
        "gaussian": "Gaussian",
        "gamma":    "Gamma",
        "pearson3": "Pearson III",
        "kde":      "KDE (Silverman)",
    }

    x_plot = np.linspace(dataset.min(), dataset.max(), 1000)

    fits = analysis["fits"]
    best = analysis["recommendation"]

    for d, res in fits.items():
        if res is None:
            continue
        params = res["params"]
        lw  = 2.5 if d == best else 1.2
        ls  = "-"  if d == best else "--"
        col = colors[d]
        D   = res["KS_statistic"]
        lbl = f"{labels[d]}  (D={D:.3f})"
        if d == best:
            lbl += "  *"

        # CDF/PDF curves — zero-inflation-aware for gamma/kde (see
        # _mixture_cdf/_mixture_pdf); plain theoretical curves otherwise.
        cdf = _mixture_cdf(x_plot, d, params)
        pdf = _mixture_pdf(x_plot, d, params)

        ax_cdf.plot(x_plot, cdf, color=col, lw=lw, ls=ls, label=lbl)
        ax_pdf.plot(x_plot, pdf, color=col, lw=lw, ls=ls, label=lbl)

    # Empirical CDF
    ax_cdf.step(x_ecdf, y_ecdf, where="post", color="black",
                lw=1.5, alpha=0.8, label="Empirical CDF")
    ax_pdf.hist(dataset, bins="auto", density=True,
                color="black", alpha=0.2, label="Empirical density")
    bar_ymax = max(p.get_height() for p in ax_pdf.patches)
    ax_pdf.set_ylim(0, bar_ymax * 1.2)  # 20% di margine sopra
    # -----------

    ax_pdf.set(xlabel="Value", ylabel="Density", title=f"PDF — {lbl}")

    for ax, ylabel, ttl in zip(
        [ax_cdf, ax_pdf],
        ["Cumulative probability", "Density"],
        [f"CDF comparison — {title}".strip(" —"),
         f"PDF comparison — {title}".strip(" —")],
    ):
        ax.set_xlabel("Value")
        ax.set_ylabel(ylabel)
        ax.set_title(ttl)
        ax.legend(fontsize=8)

    fig.tight_layout()
    return fig

# PUBLIC API

def test_standardization(data, groups=None, shift_for_gamma = False,
    plot = True, n_bootstrap = 0, seed = None):
    """
    Fit all four distribution families and recommend the best one.

    Always evaluates: Gaussian | Gamma | Pearson III | Gaussian KDE.

    Primary selection criterion : lowest KS statistic D (scale-free, valid
                                  for all four families including KDE).
    Secondary criterion         : lowest AIC (parametric families only).

    Parameters
    ----------
    data : array-like
        Input data. Temporal aggregation (e.g., for SPI-3/6/12) must be
        performed BEFORE calling this function — results reflect the
        input scale as-is.
    groups : array-like or None
        Optional grouping vector (same length as `data`). Analysis is run
        independently per group.
    shift_for_gamma : bool, default False
        Add 1 to data before Gamma fitting. False by default so that the Gamma
        judged here is the same one `f_spi` fits (raw positive part, `floc=0`,
        exact zeros carried by the zero-inflation fraction `qq`). Only turn it
        on to reproduce a legacy shifted fit.

        Note that this function judges whatever sample it is handed: to decide
        a `calculation_method` for a DroughtScan object, pass the BASELINE
        slice of the series, since that is the period `f_spi`/`f_kde` calibrate
        on (see `diagnostics.methodology`).
    plot : bool, default True
        Generate empirical vs theoretical CDF/PDF comparison figures.
    n_bootstrap : int, default 0
        If > 0, compute Lilliefors-corrected KS p-values via parametric
        bootstrap. Recommended ≥ 999 for stable estimates.
    seed : int or None
        Random seed for bootstrap reproducibility.

    Returns
    -------
    dict
        If groups is None:
            {skewness, normality_p_value, fits, best_by_mean_error,
             best_by_KS, best_by_AIC, recommendation}
        If groups is provided:
            {group_label: <same dict>}

    Notes
    -----
    KS p-values without bootstrap correction are anti-conservative when
    parameters are estimated from the same sample (Lilliefors 1967).
    The KS statistic D itself remains a valid distance metric regardless.
    """
    data   = np.asarray(data, dtype=float)


    if groups is None:
        result = _analyze_all(data, shift_for_gamma, n_bootstrap, seed=seed)
        if plot:
            fig = _plot_cdf_comparison(data, result)
            plt.show()
        return result

    groups = np.asarray(groups)
    if len(groups) != len(data):
        raise ValueError("`groups` must have the same length as `data`.")

    results = {}
    for g in np.unique(groups):
        subset = data[groups == g]
        res    = _analyze_all(subset, shift_for_gamma, n_bootstrap, seed=seed)
        if plot:
            fig = _plot_cdf_comparison(subset, res, title=str(g))
            plt.show()
        results[g] = res

    return results


def fit_distribution_stats(data, dist= "gamma", groups=None,
    shift_for_gamma= False, plot = True, n_bootstrap: int = 0,
    seed = None):
    """
    Fit a *single* specified distribution and return goodness-of-fit stats.

    Use `test_standardization` first to identify the best family, then
    call this function to obtain the full fit statistics for that family.

    Parameters
    ----------
    data : array-like
        Input dataset. Results reflect the input time scale as-is.
    dist : {"gaussian", "gamma", "pearson3", "kde"}, default "gamma"
    groups : array-like or None
    shift_for_gamma : bool, default False
        See `test_standardization` — False matches what `f_spi` actually fits.
    plot : bool, default True
        Show empirical vs theoretical CDF/PDF for the chosen distribution.
    n_bootstrap : int, default 0
        If > 0, compute Lilliefors-corrected KS p-value.
    seed : int or None

    Returns
    -------
    dict
        If groups is None:
            {distribution, skewness, normality_p_value, params,
             KS_statistic, KS_p_value [, KS_p_value_bootstrap],
             log_likelihood, AIC, k_params, error_percent, goodness_percent}
        If groups is provided:
            {group_label: <same dict>}
    """
    data  = np.asarray(data, dtype=float)
    dist  = dist.lower()
    rng   = np.random.default_rng(seed)

    def _single(dataset: np.ndarray) -> dict:
        dataset = dataset[np.isfinite(dataset)]
        skewness  = float(stats.skew(dataset))
        _, p_norm = stats.normaltest(dataset)
        res = _fit_single_dist(dataset, dist, shift_for_gamma=shift_for_gamma)
        if n_bootstrap > 0:
            res["KS_p_value_bootstrap"] = _bootstrap_ks_pvalue(
                dataset, res, n_bootstrap=n_bootstrap, rng=rng
            )
        res["skewness"]          = skewness
        res["normality_p_value"] = float(p_norm)
        return res

    if groups is None:
        dataset_clean = data[np.isfinite(data)]
        res = _single(dataset_clean)

        if plot:
            # Minimal plot: empirical ECDF + fitted CDF
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            ax_cdf, ax_pdf = axes

            x_ecdf = np.sort(dataset_clean)
            y_ecdf = np.arange(1, len(x_ecdf) + 1) / len(x_ecdf)
            x_plot = np.linspace(x_ecdf[0], x_ecdf[-1], 1000)
            params = res["params"]

            # Zero-inflation-aware CDF/PDF for gamma/kde (see _mixture_cdf/
            # _mixture_pdf); plain theoretical curves otherwise.
            cdf = _mixture_cdf(x_plot, dist, params)
            pdf = _mixture_pdf(x_plot, dist, params)
            lbl = {"gaussian": "Gaussian fit", "gamma": "Gamma fit",
                   "pearson3": "Pearson III fit", "kde": "KDE (Silverman)"}[dist]

            ax_cdf.step(x_ecdf, y_ecdf, where="post", color="black",
                        lw=1.5, alpha=0.8, label="Empirical CDF")
            ax_cdf.plot(x_plot, cdf, color="#d6604d", lw=2, label=lbl)
            ax_cdf.set(xlabel="Value", ylabel="Cumulative probability",
                       title=f"CDF — {lbl}  (D={res['KS_statistic']:.3f})")
            ax_cdf.legend()

            ax_pdf.hist(dataset_clean, bins="auto", density=True,
                        color="black", alpha=0.25, label="Empirical density")
            ax_pdf.plot(x_plot, pdf, color="#d6604d", lw=2, label=lbl)

            bar_ymax = max(p.get_height() for p in ax_pdf.patches)
            ax_pdf.set_ylim(0, bar_ymax * 1.2)  # 20% di margine sopra
            # -----------

            ax_pdf.set(xlabel="Value", ylabel="Density", title=f"PDF — {lbl}")
            ax_pdf.set(xlabel="Value", ylabel="Density", title=f"PDF — {lbl}")
            ax_pdf.legend()

            fig.tight_layout()
            plt.show()

        return res

    groups = np.asarray(groups)
    if len(groups) != len(data):
        raise ValueError("`groups` must have the same length as `data`.")

    results = {}
    for g in np.unique(groups):
        results[g] = _single(data[groups == g])
    return results

def standardize_data(data,analysis_result,groups=None,plot = True):
    """
    Transform data to standard normal scores using the distribution recommended
    by `test_standardization()`.

    Method — Probability Integral Transform (PIT):
    -----------------------------------------------
        1.  Fit the recommended distribution to the data
            (parameters are re-estimated here, consistent with _fit_single_dist).
        2.  Map each observation x → p = F(x)  (CDF value ∈ (0, 1)).
        3.  Map p → z = Φ⁻¹(p)               (standard-normal quantile).

    The result is a zero-mean, unit-variance series that can be directly
    compared across sites, variables, and time scales — the same logic
    underlying SPI (McKee 1993) and SPEI (Vicente-Serrano 2010).

    Why re-estimate parameters instead of reusing stored params?
    ------------------------------------------------------------
    When `groups` are present, each group needs its own fit.  For the
    ungrouped case the re-estimation cost is negligible and guarantees
    that the standardized scores are always internally consistent with
    the data passed in (e.g. if the user subsets or filters before calling).

    Parameters
    ----------
    data : array-like
        Original (non-standardized) values.  Must be the same series, or
        a compatible subset, of what was passed to `test_standardization()`.
        Temporal aggregation (SPI-3, SPI-6, …) must be done beforehand.
    analysis_result : dict
        Output of `test_standardization()`.
        - If `groups` is None → flat dict with key "recommendation".
        - If `groups` is provided → nested dict {group: {..., "recommendation"}}.
    groups : array-like or None
        Grouping vector (e.g. month labels for seasonal standardization).
        Must have the same length as `data`.
        Each group is standardized independently using the recommendation
        found for that group in `analysis_result`.
    plot : bool, default True
        Scatter plot of original vs standardized values, plus histogram of
        z-scores with a N(0,1) reference curve.

        CDF values are clipped to [cdf_clip, 1 - cdf_clip] before the
        normal-quantile transform to avoid ±∞ at the tails.

    Returns
    -------
    dict with keys:
        "z_scores"       : np.ndarray, standardized values (NaN preserved).
        "distribution"   : str, distribution used.
        "params"         : dict (or {group: dict} when grouped), fitted params.
        "cdf_values"     : np.ndarray, intermediate CDF values F(x).
        "recommendation" : str (or {group: str} when grouped).

    Raises
    ------
    KeyError
        If `analysis_result` does not contain a "recommendation" key (wrong input).
    ValueError
        If `groups` is provided but not present as keys in `analysis_result`.

    Notes
    -----
    - NaN / ±Inf in `data` are preserved as NaN in `z_scores`.
    - For KDE the CDF is approximated via trapezoidal integration on a fine
      grid (4096 points) — same approach as in `_fit_single_dist`.
    - The Gaussian case degenerates to the classic z-score: z = (x − μ) / σ,
      but is routed through the PIT for uniformity.

    References
    ----------
    McKee T.B. et al. (1993).  J. Am. Meteorol. Soc.
    Vicente-Serrano S.M. et al. (2010).  J. Clim. 23, 1696–1718.
    """

    data   = np.asarray(data, dtype=float)
    valid_mask = np.isfinite(data)
    cdf_clip = 1e-6

    # ── Internal PIT engine ────────────────────────────────────────────────
    def _pit(dataset_clean: np.ndarray, dist: str, params: dict) -> np.ndarray:
        """
        Probability Integral Transform for one (dist, params) pair.
        Returns CDF values in (cdf_clip, 1-cdf_clip). Zero-inflation-aware
        for "gamma"/"kde" via `_mixture_cdf` (params already carry `qq`,
        and for "kde" the exact `xb`/`h`/`log_transform` fit ingredients —
        no need to refit anything here).
        """
        p = _mixture_cdf(dataset_clean, dist, params)
        return np.clip(p, cdf_clip, 1.0 - cdf_clip)

    # ── Helper: standardize one group ──────────────────────────────────────
    def _standardize_group(
        data_full: np.ndarray,         # full-length array (with NaN)
        mask: np.ndarray,              # boolean, finite positions
        group_result: dict,            # analysis dict for this group
    ) -> tuple[np.ndarray, dict, np.ndarray, str]:
        """
        Returns (z_scores_full, params, cdf_values_full, dist_used).
        NaN positions in data_full remain NaN in z_scores_full.
        """
        rec   = group_result["recommendation"]
        dist  = rec
        # Refit to get fresh params (guarantees consistency)
        fit   = _fit_single_dist(data_full[mask], dist)
        params = fit["params"]

        # PIT on clean data
        p_clean = _pit(data_full[mask], dist, params)
        z_clean = stats.norm.ppf(p_clean)

        # Reconstruct full-length arrays preserving NaN positions
        z_full   = np.full(len(data_full), np.nan)
        cdf_full = np.full(len(data_full), np.nan)
        z_full[mask]   = z_clean
        cdf_full[mask] = p_clean

        return z_full, params, cdf_full, dist

    # ══════════════════════════════════════════════════════════════════════
    # CASE 1: no groups
    # ══════════════════════════════════════════════════════════════════════
    if groups is None:
        if "recommendation" not in analysis_result:
            raise KeyError(
                "'recommendation' key not found in analysis_result. "
                "Pass the direct output of test_standardization() with groups=None."
            )
        z_full, params, cdf_full, dist_used = _standardize_group(
            data, valid_mask, analysis_result
        )

        if plot:
            _plot_standardization(data[valid_mask], z_full[valid_mask], dist_used)

        return {
            "z_scores":      z_full,
            "distribution":  dist_used,
            "params":        params,
            "cdf_values":    cdf_full,
            "recommendation": dist_used,
        }

    # ══════════════════════════════════════════════════════════════════════
    # CASE 2: grouped
    # ══════════════════════════════════════════════════════════════════════
    groups = np.asarray(groups)
    if len(groups) != len(data):
        raise ValueError("`groups` must have the same length as `data`.")

    unique_groups = np.unique(groups)
    missing = [g for g in unique_groups if g not in analysis_result]
    if missing:
        raise ValueError(
            f"Groups {missing} are present in `data` but not in `analysis_result`. "
            "Run test_standardization() with the same `groups` vector."
        )

    z_full   = np.full(len(data), np.nan)
    cdf_full = np.full(len(data), np.nan)
    params_by_group: dict  = {}
    dist_by_group:   dict  = {}

    for g in unique_groups:
        g_mask = (groups == g) & valid_mask
        group_result = analysis_result[g]

        z_g, params_g, cdf_g, dist_g = _standardize_group(
            data, g_mask, group_result
        )
        z_full[g_mask]   = z_g[g_mask]
        cdf_full[g_mask] = cdf_g[g_mask]
        params_by_group[g] = params_g
        dist_by_group[g]   = dist_g

    if plot:
        _plot_standardization(
            data[valid_mask], z_full[valid_mask],
            dist_label="grouped (" + ", ".join(
                f"{g}→{d}" for g, d in dist_by_group.items()
            ) + ")",
        )

    return {
        "z_scores":      z_full,
        "distribution":  dist_by_group,
        "params":        params_by_group,
        "cdf_values":    cdf_full,
        "recommendation": dist_by_group,
    }


def _plot_standardization(original,z_scores,dist_label= ""):
    """
    Two-panel diagnostic:
        Left  — scatter: original values vs z-scores (shows monotonic mapping).
        Right — histogram of z-scores vs N(0,1) reference.

    A well-standardized series should yield:
        - monotonically increasing scatter (left),
        - histogram hugging the N(0,1) bell curve (right).
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax_sc, ax_hz = axes

    # ── Scatter: original vs z ─────────────────────────────────────────────
    ax_sc.scatter(original, z_scores, s=10, alpha=0.5, color="#2166ac")
    ax_sc.axhline(0,  color="grey", lw=0.8, ls="--")
    ax_sc.axhline( 1.96, color="#d6604d", lw=0.8, ls=":", label="±1.96 σ")
    ax_sc.axhline(-1.96, color="#d6604d", lw=0.8, ls=":")
    ax_sc.set_xlabel("Original value")
    ax_sc.set_ylabel("z-score")
    ax_sc.set_title(f"Original → z-score  [{dist_label}]", fontsize=9)
    ax_sc.legend(fontsize=8)

    # ── Histogram of z-scores vs N(0,1) ───────────────────────────────────
    ax_hz.hist(z_scores, bins="auto", density=True,
               color="#2166ac", alpha=0.4, label="Standardized data")
    x_ref = np.linspace(-4, 4, 400)
    ax_hz.plot(x_ref, stats.norm.pdf(x_ref), color="black",
               lw=1.8, label="N(0, 1)")
    ax_hz.set_xlabel("z-score")
    ax_hz.set_ylabel("Density")
    ax_hz.set_title("Distribution of z-scores vs N(0,1)", fontsize=9)
    ax_hz.legend(fontsize=8)

    fig.tight_layout()
    plt.show()
    return fig

def plot_cdf_comparison(data, dist="gamma", params=None, shift_for_gamma=False,unit=None):
    """
    Plot empirical CDF vs theoretical CDF for a fitted distribution.
    Useful to inspect where the KS error is concentrated (low tail, central body, high tail).

    Parameters
    ----------
    data : array-like
        Input dataset used for fitting.
    dist : {"gamma", "pearson3", "gaussian"}
        Distribution family used for the theoretical CDF.
    params : dict or None
        Dictionary of fitted parameters from fit_distribution_stats().
        If None, parameters are fitted internally.
    shift_for_gamma : bool, default False
        If True and dist == "gamma", data are shifted by +1 as in
        fit_distribution_stats. False by default, matching `f_spi`.
    unit : str or None
        Label for the data axis (x-axis of both panels). Defaults to "value".

    Returns
    -------
    None
        Displays the figure (histogram + PDF, empirical vs theoretical CDF).
    """
    if unit==None:
        unit = "value"
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    dist = dist.lower()

    # Fit params if not provided. Gamma is zero-inflation aware, matching
    # f_spi/_fit_single_dist: fitted on the strictly-positive subset only,
    # with qq = P(X=0) from the full sample (never shifted-and-kept).
    if params is None:
        fit = _fit_single_dist(data, dist, shift_for_gamma=shift_for_gamma)
        params = fit["params"]

    if dist == "gamma":
        qq = params.get("qq", 0.0)
        label = ["Gamma fit", "Gamma CDF"]
    elif dist == "pearson3":
        qq = 0.0
        label = ["Pearson III fit", "Pearson III CDF"]
    else:
        qq = 0.0
        label = ["Gaussian fit", "Gaussian CDF"]

    # For the histogram/ECDF panels only: gamma's fitted curve describes the
    # positive part, so exact zeros (if any) are excluded from the display
    # sample the same way they're excluded from the fit.
    data_used = data[data > 0] if (dist == "gamma" and qq > 0) else data

    # --------------------------
    # Empirical CDF
    # --------------------------
    sorted_data = np.sort(data_used)
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    # --------------------------
    # Theoretical CDF/PDF (zero-inflation-aware for gamma via _mixture_*)
    # --------------------------
    x = np.linspace(sorted_data.min(), sorted_data.max(), 400)
    cdf = _mixture_cdf(x, dist, params)
    pdf = _mixture_pdf(x, dist, params)

    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(10, 4))
    ax = ax.ravel()

    # --------------------------
    # Histogramp + Theoreticsl PDF
    # --------------------------
    # Empirical density
    ax[0].hist(data_used, bins=15, density=True, alpha=0.4, label="Empirical")
    ax[0].plot(x, pdf, label=label[0])
    ax[0].set_xlabel(unit)
    ax[0].set_ylabel("Density")
    ax[0].legend()



    ax[1].plot(sorted_data, ecdf, label="Empirical CDF", color="black")
    ax[1].plot(x, cdf, label=label[1], linewidth=2)

    ax[1].set_xlabel(unit)
    ax[1].set_ylabel("CDF")
    ax[1].set_title("Empirical vs Theoretical CDF")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    plt.tight_layout()


# ===================================================================
#  Rolling Trend Analysis
# ===================================================================
def _rolling_trend_analysis(var, window=60, significance=0.05):
    """
    Perform rolling trend analysis on a given time series.
    Args:
        var (ndarray): Input time series array.
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

def _rolling_phase_test(DSO, window=60, alpha=0.05, min_valid=None):
    """
    Rolling significance test for wetting/drying phases on the 1-month SPI series.

    A phase is a sustained departure of monthly anomalies from zero (a LEVEL),
    not a slope. Over each W-month window the mean of SPI1 is tested against zero
    with a one-sample t-test. The test is applied to SPI1 (approximately serially
    independent), NOT to the CDN (a cumulative/integrated series on which trend
    tests are spurious). Results are aligned to the LAST month of each window,
    matching the CDN time axis.

    Args:
        DSO (object): DroughtScan object; the 1-month SPI-like series is read
            internally from `DSO.spi_like_set[0]`.
        window (int): window length in months (= accumulation scale W).
        alpha (float): two-sided significance level.
        min_valid (int, optional): min number of non-NaN months to run the test.
            Defaults to window // 2.

    Returns:
        dict aligned to the SPI-1 series:
            'phase'   : int  (+1 significant wetting, -1 significant drying, 0 otherwise)
            'mean'    : float (mean SPI1 over the window, NaN where undefined)
            'p_value' : float (two-sided p-value, NaN where undefined)
    """
    from scipy.stats import ttest_1samp
    spi1 = DSO.spi_like_set[0]
    n = len(spi1)
    if min_valid is None:
        min_valid = window // 2

    phase = np.zeros(n, dtype=int)
    mean_w = np.full(n, np.nan)
    pval = np.full(n, np.nan)

    for i in range(n - window + 1):
        w = spi1[i:i + window]
        w = w[~np.isnan(w)]  # scarta i gap
        j = i + window - 1  # ultimo mese della finestra
        if len(w) < min_valid:
            continue
        m = w.mean()
        _, p = ttest_1samp(w, 0.0)  # H0: media SPI1 = 0 (nessuna tendenza netta)
        mean_w[j] = m
        pval[j] = p
        if p < alpha:
            phase[j] = 1 if m > 0 else -1

    return {'phase': phase, 'mean': mean_w, 'p_value': pval}



# =====================================================================
# Generic year-aligned block-bootstrap primitives
# ---------------------------------------------------------------------
# Paired moving-block resampling of a raw monthly (P, Q) overlap: whole
# years are drawn in contiguous blocks and laid end-to-end under a fresh
# January-anchored synthetic calendar, so f_spi/f_kde still see a gap-free
# monotonic m_cal. The k-month accumulation is meaningless for the first
# (k-1) months after every block join, so those cells are NaN-masked per
# scale; the fraction lost grows with k and is reported by
# _print_contamination_table. Ported from Drought-Scan.
# =====================================================================

_BOOT_BASE_YEAR = 2000   # arbitrary anchor for the synthetic calendar


def _round_to_year(L):
    """Nearest whole number of years, at least 2 (24 months)."""
    return 12 * max(2, int(round(L / 12)))


def _year_block_index(n_years, block_years, rng, circular):
    """Row indices (into a year-major layout) for a bootstrap replica:
    ceil(n_years / block_years) contiguous runs of whole years."""
    n_draw = int(np.ceil(n_years / block_years))
    if circular:
        starts = rng.integers(0, n_years, size=n_draw)
        yr = np.concatenate([(np.arange(s, s + block_years) % n_years) for s in starts])
    else:
        hi = max(1, n_years - block_years + 1)
        starts = rng.integers(0, hi, size=n_draw)
        yr = np.concatenate([np.arange(s, s + block_years) for s in starts])
    return yr[:n_years]


def _junction_distance(T, L):
    """Months since the last block start (blocks tile [0, T) at 0, L, 2L, ...)."""
    pos = np.arange(T)
    return pos - (pos // L) * L


def _contamination_fraction(T, L, K):
    """Fraction of timesteps whose scale-k rolling accumulation crosses a
    block join (hence NaN-masked), for k = 1..K. Independent of the draw."""
    dist = _junction_distance(T, L)
    return np.array([float(np.mean(dist < (k - 1))) for k in range(1, K + 1)])


def _spi_set_from_series(ts, m_cal, calc_method, tb1, tb2, K):
    """spi_like_set (K, T) for an arbitrary (ts, m_cal), mirroring
    BaseDroughtAnalysis._compute_spi's inner loop with no side effects on
    any instance. Used only by the bootstrap workers."""
    T = len(ts)
    sset = np.full((K, T), np.nan, dtype=float)
    for k in range(1, K + 1):
        for ref_month in range(1, 13):
            out = calc_method(ts, k, ref_month, m_cal, tb1, tb2)
            indices, spi_values = out[0], out[1]
            if indices is None or spi_values is None:
                continue
            sset[k - 1, indices] = spi_values
    return sset


def _print_contamination_table(meta, K_range):
    """The 'dati persi' table: per scale, the share of timesteps whose rolling
    accumulation crosses a block join and the effective N that survives."""
    cf = meta["contaminated_fraction"]
    en = meta["eff_n_per_scale"]
    print(f"  junction contamination (L={meta['block_length']} months / "
          f"{meta['block_years']}y, {meta['n_blocks']} blocks over "
          f"T={meta['overlap_length']} months):")
    print("    K    masked%   eff_N")
    for k, c, n in zip(K_range, cf, en):
        mark = "  <-- >50%" if c > 0.5 else ""
        print(f"   {int(k):3d}   {c * 100:5.1f}    {int(n):5d}{mark}")


# =====================================================================
# Block-bootstrap CI for the parametric rainfall-runoff benchmarks
# ---------------------------------------------------------------------
# Same paired year-aligned moving-block resampling of the raw (P, Q)
# overlap as the primitives above (fresh synthetic calendar, per-scale
# junction masking), but each replica re-runs ONE benchmark's own fit on
# the synthetic series and returns a flat dict of scalar estimates (fitted
# parameters + skill). Percentiles of those give the CI. Kernel ordinates
# h(j) and the OLS rescaling coefficients are deliberately NOT bootstrapped.
# =====================================================================


class _BenchmarkReplica:
    """The synthetic contiguous series for one resample, handed to the
    per-benchmark fit dispatch. Junction-aware: ``spi_P`` is already
    NaN-masked at every block join per scale; ``junction_dist`` lets the
    lag-convolution benchmarks mask their own windows."""
    __slots__ = ("P", "Q", "spi_P", "sqi1", "months", "junction_dist", "K")

    def __init__(self, P, Q, spi_P, sqi1, months, junction_dist, K):
        self.P, self.Q = P, Q
        self.spi_P, self.sqi1 = spi_P, sqi1
        self.months = months
        self.junction_dist = junction_dist
        self.K = K


def _benchmark_fit(kind, rep, season_mask, bench_K, l_step):
    """Re-run one benchmark's fit on a bootstrap replica; return only scalars.

    kind : 'conv_P' | 'conv_SPI' | 'dspi_free' | 'nash' | 'ihacres'
    """
    from drought_scan.core import BaseDroughtAnalysis as _B

    jd = rep.junction_dist
    K_range = np.arange(1, bench_K + 1)

    def _num(v):
        return np.nan if v is None else float(v)

    if kind == "nash":
        r = _B._nash_fit(rep.P, rep.sqi1, bench_K, season_mask=season_mask,
                         junction_dist=jd, l_step=l_step, verbose=False)
        return {"optimal_n": _num(r["optimal_n"]), "optimal_k": _num(r["optimal_k"]),
                "optimal_K": _num(r["optimal_K"]), "R2": _num(r["R2"]),
                "RMSE": _num(r["metrics"]["RMSE"]), "KGE": _num(r["metrics"]["KGE"])}

    if kind == "ihacres":
        r = _B._ihacres_fit(rep.P, rep.sqi1, bench_K, season_mask=season_mask,
                            junction_dist=jd, l_step=l_step, verbose=False)
        return {"optimal_tau_f": _num(r["optimal_tau_f"]),
                "optimal_tau_s": _num(r["optimal_tau_s"]),
                "optimal_alpha": _num(r["optimal_alpha"]),
                "optimal_K": _num(r["optimal_K"]), "R2": _num(r["R2"]),
                "RMSE": _num(r["metrics"]["RMSE"]), "KGE": _num(r["metrics"]["KGE"])}

    if kind == "dspi_free":
        r = _B._dspi_free_fit(rep.spi_P, rep.sqi1, K_range,
                              season_mask=season_mask, verbose=False)
    elif kind == "conv_P":
        r = _B._ols_convolution_fit(rep.P, rep.Q, K_range, season_mask=season_mask,
                                    junction_dist=jd, verbose=False)
    elif kind == "conv_SPI":
        r = _B._ols_convolution_fit(rep.spi_P[0], rep.sqi1, K_range,
                                    season_mask=season_mask, junction_dist=jd,
                                    verbose=False)
    else:
        raise ValueError(f"unknown benchmark kind {kind!r}")

    if r is None:
        return {"optimal_K": np.nan, "R2_adj_opt": np.nan, "R2": np.nan,
                "RMSE": np.nan, "KGE": np.nan}
    ok = int(r["optimal_K"])
    return {"optimal_K": float(ok),
            "R2_adj_opt": float(r["R2_adj"][ok - 1]),
            "R2": _num(r["metrics"]["R2"]), "RMSE": _num(r["metrics"]["RMSE"]),
            "KGE": _num(r["metrics"]["KGE"])}


def _one_benchmark_replica(P_yr, Q_yr, calcP, n_base_years, calcQ, n_base_years_Q,
                           spi_K, block_years, circular, seed, kind, bench_K, l_step,
                           season_months=None):
    """One paired year-aligned block-bootstrap replica for a benchmark: build
    the synthetic (P, Q) series + its junction-masked SPI/SQI set, then call
    ``_benchmark_fit``. Returns its scalar dict (non-seasonal) or
    ``{season: scalar dict}`` (seasonal)."""
    rng = np.random.default_rng(seed)
    n_years = P_yr.shape[0]
    rows = _year_block_index(n_years, block_years, rng, circular)

    P_b = P_yr[rows].reshape(-1).astype(float)
    Q_b = Q_yr[rows].reshape(-1).astype(float)
    T = P_b.size
    months = (np.arange(T) % 12) + 1
    years = _BOOT_BASE_YEAR + np.arange(T) // 12
    m_cal_b = np.column_stack([months, years])

    L = block_years * 12
    dist = _junction_distance(T, L)

    # The full K-scale SPI set of the synthetic P (spi_K per-calendar-month fits,
    # the dominant cost of a replica) is only needed by the two SPI-driven
    # benchmarks. 'conv_P' / 'nash' / 'ihacres' drive on raw P, so skip it there
    # entirely - this is what takes benchmark_nash's bootstrap from hours to
    # minutes.
    need_spi = kind in ("conv_SPI", "dspi_free")
    tb2P = _BOOT_BASE_YEAR + min(n_base_years, n_years) - 1
    tb2Q = _BOOT_BASE_YEAR + min(n_base_years_Q, n_years) - 1
    if need_spi:
        spi_P = _spi_set_from_series(P_b, m_cal_b, calcP, _BOOT_BASE_YEAR, tb2P, spi_K)
        for k in range(2, spi_K + 1):
            spi_P[k - 1, dist < (k - 1)] = np.nan
    else:
        spi_P = None
    sqi1 = _spi_set_from_series(Q_b, m_cal_b, calcQ, _BOOT_BASE_YEAR, tb2Q, 1)[0]

    rep = _BenchmarkReplica(P_b, Q_b, spi_P, sqi1, months, dist, spi_K)
    if season_months is None:
        return _benchmark_fit(kind, rep, None, bench_K, l_step)
    return {name: _benchmark_fit(kind, rep, np.isin(months, mlist), bench_K, l_step)
            for name, mlist in season_months.items()}


def _bootstrap_benchmark(self_obj, streamflow, self_indices, streamflow_indices,
                         kind, bench_K, n_boot, block_length, ci, circular,
                         random_state, seasons=None, l_step=1):
    """Paired year-aligned block-bootstrap CI for one parametric benchmark.

    Returns
    -------
    non-seasonal : (boot {param: (B,)}, ci {param: (lo, hi)}, meta)
    seasons given : ({season: boot}, {season: ci}, meta)
    """
    from joblib import Parallel, delayed, cpu_count

    m_cal_ov = self_obj.m_cal[self_indices]
    P_ov = np.asarray(self_obj.ts, float)[self_indices]
    Q_ov = np.asarray(streamflow.ts, float)[streamflow_indices]

    jan = np.where(m_cal_ov[:, 0].astype(int) == 1)[0]
    if jan.size == 0:
        raise ValueError("block bootstrap needs at least one January in the overlap.")
    j0 = int(jan[0])
    n_years = (len(P_ov) - j0) // 12
    if n_years < 4:
        raise ValueError(f"block bootstrap needs >= 4 whole overlap years, got {n_years}.")
    end = j0 + 12 * n_years
    P_yr = P_ov[j0:end].reshape(n_years, 12)
    Q_yr = Q_ov[j0:end].reshape(n_years, 12)
    T = n_years * 12

    L = int(block_length) if block_length else max(24, 2 * self_obj.K)
    L = min(_round_to_year(L), 12 * n_years)
    block_years = L // 12

    n_base_years = self_obj.end_baseline_year - self_obj.start_baseline_year + 1
    n_base_years_Q = streamflow.end_baseline_year - streamflow.start_baseline_year + 1

    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, 2 ** 32 - 1, size=int(n_boot))
    season_months = dict(seasons) if seasons is not None else None

    n_jobs = max(1, min(4, cpu_count() // 2))
    step_note = f", L-search stride {l_step}" if l_step > 1 and kind in ("nash", "ihacres") else ""
    spi_note = ("rebuilds the full SPI set" if kind in ("conv_SPI", "dspi_free")
                else "SQI1 only, raw-P driven")
    print(f"  benchmark block bootstrap [{kind}]: B={n_boot}, L={L} months "
          f"({block_years}y) {'circular' if circular else 'moving'} blocks over "
          f"{n_years} overlap years{step_note} ({spi_note}; refits {n_boot}x on "
          f"{n_jobs} workers)...")

    import matplotlib.pyplot as plt
    plt.close("all")

    with Parallel(n_jobs=n_jobs) as parallel:
        reps = parallel(
            delayed(_one_benchmark_replica)(
                P_yr, Q_yr, self_obj.calculation_method, n_base_years,
                streamflow.calculation_method, n_base_years_Q,
                self_obj.K, block_years, circular, int(s), kind, bench_K, l_step,
                season_months)
            for s in seeds
        )

    contam = _contamination_fraction(T, L, self_obj.K)
    meta = {
        "n_boot": int(n_boot), "block_length": int(L), "block_years": int(block_years),
        "circular": bool(circular), "ci": tuple(ci),
        "n_blocks": int(np.ceil(n_years / block_years)),
        "overlap_years": int(n_years), "overlap_length": int(T),
        "l_step": int(l_step),
        "contaminated_fraction": contam,
        "eff_n_per_scale": np.rint(T * (1.0 - contam)).astype(int),
    }

    def _reduce(dicts):
        keys = list(dicts[0])
        boot = {k: np.array([d[k] for d in dicts], float) for k in keys}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")          # all-NaN param -> NaN CI
            cis = {k: tuple(np.nanpercentile(v, list(ci))) for k, v in boot.items()}
        return boot, cis

    if season_months is None:
        boot, cis = _reduce(reps)
        return boot, cis, meta
    boot_by_season, ci_by_season = {}, {}
    for name in season_months:
        boot_by_season[name], ci_by_season[name] = _reduce([r[name] for r in reps])
    return boot_by_season, ci_by_season, meta


# ===================================================================
#  SIDI calibration: R2-surface block bootstrap + scale clusters
# ===================================================================
# Ported from Drought-Scan. _bootstrap_r2 resamples the raw paired
# (P, Q) overlap in whole-year blocks and rebuilds the entire
# SPI/SQI/SIDI/R2(K, weight) surface per replica (reusing the generic
# primitives above); _peak_summary turns the observed surface + its
# bootstrap into per-scheme peaks and scale clusters; the two
# bootstrap_summary_table / _print_summary_table render that as the
# per-cluster table analyze_correlation[_seasonal] attach as 'summary'.
# ===================================================================

WEIGHT_LABELS = ("EW", "Lin. DW", "Log. DW", "Lin. IW", "Log. IW")

# Canonical per-scheme colours (matplotlib tab10, in WEIGHT_LABELS order). Kept
# here so every renderer - the library figures and the diagnostic site - tints a
# cluster box with the SAME hue as the scheme that leads that cluster, instead of
# a single fixed colour. Drawn very transparent, these read as pastels: EW ->
# pale blue, Lin. DW -> straw, Log. DW -> pale green, Lin. IW -> pink, Log. IW ->
# lavender.
WEIGHT_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd")

# Short lowercase handles, WEIGHT_LABELS order, for the cluster-reference print
# only (_print_cluster_refs) - everywhere else (families column, legends,
# plots) keeps the full WEIGHT_LABELS names. "Log. DW"/"Log. IW" are
# geometric weighting under the hood (see generate_weights), hence
# "geodw"/"geoiw" here rather than "logdw"/"logiw".
_WEIGHT_ABBR = dict(zip(WEIGHT_LABELS, ("ew", "ldw", "geodw", "liw", "geoiw")))


def _peak_summary(M, r2_boot, ci=(2.5, 97.5), weight_labels=WEIGHT_LABELS, k_tol=0):
    """One season's row of ``bootstrap_summary_table`` — see that function.

    Per weighting scheme ``w`` (observed surface ``M`` (K, W), bootstrap
    ``r2_boot`` (B, K, W)):

      - ``peak_R2(w)``     = max over K of ``M[:, w]``
      - ``argmax_K(w)``    = the K where that max sits (1-indexed)
      - ``peak_CI(w)``     = the ``ci`` percentiles of ``max_K r2_boot[b, :, w]``
      - ``K_CI(w)``        = the ``ci`` percentiles of the bootstrap ``argmax_K``
        of that family — an integer interval (drawn as the horizontal error bar).

    **Scale clusters** ``clusters``: families are grouped by the K they peak at,
    NOT by R². Two families are linked iff each one's observed peak K falls
    inside the OTHER's bootstrap K CI (symmetric mutual inclusion, on K); the
    clusters are the connected components of that graph. So a family whose peak
    K lies clearly outside the others' K CIs stays on its own — a response
    mechanism in its own right — while a flat/unresolved season (every K CI
    spans most of the axis) collapses to one cluster with a wide box. ``k_tol``
    (default 0) pads every K CI by that many months before the test, for callers
    who want to loosen the linkage.

    Per cluster: ``K_cluster`` / ``R2_cluster`` = median over its families of
    ``argmax_K`` / ``peak_R2`` (observed); the CIs are the ``ci`` percentiles of
    the per-replica median of the same quantity over the cluster's families.
    Clusters are returned sorted by ``K_cluster``, with no interpretive label —
    the ``K_cluster_CI`` x ``R2_cluster_CI`` box is meant to speak for itself.
    Each cluster also carries ``color`` — the hue (see ``WEIGHT_COLORS``) of its
    leading scheme, i.e. the member with the highest observed peak R2 — so a
    renderer can tint the box with that scheme's colour instead of a fixed one.
    For a one-name handle on the cluster, ``ref_scheme`` / ``ref_K`` /
    ``ref_K_CI`` / ``ref_R2`` / ``ref_R2_CI`` give a display reference: the
    member whose observed peak K is closest to ``K_cluster`` (ties -> smaller
    K), reported with THAT scheme's own peak K and R2. Purely cosmetic — the
    SIDI uses each scheme's own K regardless.

    **Level 2 - response sub-clusters** ``cluster["subclusters"]``: within a K
    cluster that holds more than one family, the SAME mutual-inclusion rule is
    re-applied on peak R2 (each family's observed peak R2 must sit inside the
    other's bootstrap peak-R2 CI). So schemes that share a scale but reach
    clearly different R2 split apart; schemes whose R2 CIs overlap stay merged.
    Each entry is ``{"families", "R2", "R2_CI", "color"}``, sorted by decreasing
    R2 (sub-cluster 1 = strongest). A single-family K cluster has one sub-cluster
    equal to that family's own peak-R2 CI.
    """
    M = np.asarray(M, float)
    boot = np.asarray(r2_boot, float)
    n_w = M.shape[1] if M.ndim == 2 else len(weight_labels)
    lo_p, hi_p = ci
    empty = {"R2_peak": np.nan, "peak_scheme": None, "peak_CI": (np.nan, np.nan),
             "w_best": None, "peak_by_family": {}, "clusters": []}
    if boot.ndim != 3 or M.ndim != 2 or not np.isfinite(M).any():
        return empty

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # observed per-family peak R2 and the K it sits on (1-indexed)
        fin_M = np.isfinite(M)
        peak_R2 = np.nanmax(np.where(fin_M, M, -np.inf), axis=0)      # (W,)
        peak_R2[~fin_M.any(axis=0)] = np.nan
        argK = np.array([
            (np.nanargmax(np.where(fin_M[:, w], M[:, w], -np.inf)) + 1
             if fin_M[:, w].any() else np.nan)
            for w in range(n_w)], float)

        # bootstrap per-replica per-family peak R2 and its K
        fin_b = np.isfinite(boot)
        filled = np.where(fin_b, boot, -np.inf)
        peak_boot = np.nanmax(np.where(fin_b, boot, np.nan), axis=1)  # (B, W)
        B = boot.shape[0]
        argK_boot = np.full((B, n_w), np.nan)
        for w in range(n_w):
            ok = fin_b[:, :, w].any(axis=1)
            if ok.any():
                argK_boot[ok, w] = np.nanargmax(filled[ok, :, w], axis=1) + 1

    peak_CI = np.full((n_w, 2), np.nan)
    for w in range(n_w):
        col = peak_boot[np.isfinite(peak_boot[:, w]), w]
        if col.size:
            peak_CI[w] = np.percentile(col, [lo_p, hi_p])

    if not np.isfinite(peak_R2).any():
        return empty
    w_best = int(np.nanargmax(peak_R2))

    # per-family CI of the peak K: percentiles of the bootstrap argmax_K
    K_CI = np.full((n_w, 2), np.nan)
    for w in range(n_w):
        col = argK_boot[np.isfinite(argK_boot[:, w]), w]
        if col.size:
            K_CI[w] = np.percentile(col, [lo_p, hi_p])

    fam_ok = [w for w in range(n_w)
              if np.isfinite(argK[w]) and np.isfinite(peak_R2[w])
              and np.all(np.isfinite(K_CI[w]))]

    # --- scale clusters: group families that peak at the SAME K ------------
    # Two families share a cluster iff each one's observed peak K falls inside
    # the OTHER's bootstrap K CI (symmetric mutual inclusion, on K - not R2).
    # The clusters are the connected components of that graph. A family whose
    # peak K sits clearly outside the others' K CIs stays on its own (a
    # response mechanism in its own right); a flat/unresolved season, where
    # every K CI spans most of the axis, gives one big cluster with a wide box.
    # `k_tol` is a floor on that CI half-width, so two near-identical peaks with
    # degenerate CIs still link.

    def _linked(a, b):
        (la, ha), (lb, hb) = K_CI[a], K_CI[b]
        la, ha = la - k_tol, ha + k_tol
        lb, hb = lb - k_tol, hb + k_tol
        return (lb <= argK[a] <= hb) and (la <= argK[b] <= ha)

    def _cluster_stats(members):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mk = np.nanmedian(argK_boot[:, members], axis=1)
            mr = np.nanmedian(peak_boot[:, members], axis=1)
        mk, mr = mk[np.isfinite(mk)], mr[np.isfinite(mr)]
        k_ci = ((int(np.floor(np.percentile(mk, lo_p))),
                 int(np.ceil(np.percentile(mk, hi_p)))) if mk.size else (np.nan, np.nan))
        r2_ci = (tuple(float(v) for v in np.percentile(mr, [lo_p, hi_p]))
                 if mr.size else (np.nan, np.nan))
        return k_ci, r2_ci

    def _r2_ci(members):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mr = np.nanmedian(peak_boot[:, members], axis=1)
        mr = mr[np.isfinite(mr)]
        return (tuple(float(v) for v in np.percentile(mr, [lo_p, hi_p]))
                if mr.size else (np.nan, np.nan))

    def _components(items, linked):
        """Connected components of the graph on ``items`` with edge = ``linked``."""
        out, remaining = [], set(items)
        while remaining:
            comp = {remaining.pop()}
            frontier = list(comp)
            while frontier:
                a = frontier.pop()
                for b in list(remaining):
                    if linked(a, b):
                        remaining.discard(b)
                        comp.add(b)
                        frontier.append(b)
            out.append(sorted(comp))
        return out

    def _hue(members):
        rep = members[int(np.argmax([peak_R2[w] for w in members]))]
        return WEIGHT_COLORS[rep] if rep < len(WEIGHT_COLORS) else "#efe08c"

    def _cluster_ref(members):
        """Representative scheme for a K cluster, for display only. No scheme is
        privileged - not even EW. The SIDI itself still uses each scheme's own K;
        this is only a label.
          - <= 2 members: the one with the SHORTER peak K (nearest-to-median is
            a tie with only two, and the shorter memory is the conservative
            reading - it already captures the shared signal).
          - > 2 members: the member whose peak K is closest to the cluster's
            MEDIAN peak K (ties -> shorter K)."""
        if len(members) <= 2:
            return min(members, key=lambda w: argK[w])
        k_med = float(np.nanmedian([argK[w] for w in members]))
        return min(members, key=lambda w: (abs(argK[w] - k_med), argK[w]))

    def _r2_linked(a, b):
        """Level 2: same mutual-inclusion rule, on peak R2 instead of peak K.
        Two families in one K cluster stay together iff each one's observed peak
        R2 falls inside the OTHER's bootstrap peak-R2 CI."""
        (la, ha), (lb, hb) = peak_CI[a], peak_CI[b]
        if not np.all(np.isfinite([la, ha, lb, hb])):
            return False
        return (lb <= peak_R2[a] <= hb) and (la <= peak_R2[b] <= ha)

    def _subclusters(members):
        """Split one K cluster into response (R2) sub-clusters. A single-family
        K cluster yields one sub-cluster (its own peak-R2 CI). Sub-clusters are
        sorted by decreasing R2, so sub-cluster 1 is always the strongest."""
        comps = ([[m] for m in members] if len(members) == 1
                 else _components(members, _r2_linked))
        subs = []
        for sub in comps:
            subs.append({
                "families": [weight_labels[w] for w in sub],
                "R2": float(np.nanmedian([peak_R2[w] for w in sub])),
                "R2_CI": _r2_ci(sub),
                "color": _hue(sub),
            })
        subs.sort(key=lambda s: (-s["R2"] if np.isfinite(s["R2"]) else np.inf))
        return subs

    clusters = []
    if fam_ok:
        for members in _components(fam_ok, _linked):
            k_ci, r2_ci = _cluster_stats(members)
            rep = _cluster_ref(members)
            clusters.append({
                "families": [weight_labels[w] for w in members],
                "K_cluster": float(np.nanmedian([argK[w] for w in members])),
                "K_cluster_CI": k_ci,
                "R2_cluster": float(np.nanmedian([peak_R2[w] for w in members])),
                "R2_cluster_CI": r2_ci,
                # display reference (see _cluster_ref for the rule). Its OWN peak
                # K / R2 (+ CIs), not the cluster medians.
                "ref_scheme": weight_labels[rep],
                "ref_K": int(argK[rep]),
                "ref_K_CI": ((int(K_CI[rep, 0]), int(K_CI[rep, 1]))
                             if np.all(np.isfinite(K_CI[rep])) else (np.nan, np.nan)),
                "ref_R2": float(peak_R2[rep]),
                "ref_R2_CI": (float(peak_CI[rep, 0]), float(peak_CI[rep, 1])),
                # tint the box with the hue of the cluster's leading scheme (the
                # member with the highest observed peak R2 - the curve that tops
                # out highest in this K band)
                "color": _hue(members),
                # level-2 split: schemes that share this K but differ in R2
                "subclusters": _subclusters(members),
            })
        clusters.sort(key=lambda c: c["K_cluster"])

    peak_by_family = {
        weight_labels[w]: {
            "peak_R2": float(peak_R2[w]) if np.isfinite(peak_R2[w]) else np.nan,
            "argmax_K": int(argK[w]) if np.isfinite(argK[w]) else None,
            "peak_CI": (float(peak_CI[w, 0]), float(peak_CI[w, 1])),
            "K_CI": ((int(K_CI[w, 0]), int(K_CI[w, 1]))
                     if np.all(np.isfinite(K_CI[w])) else (np.nan, np.nan)),
        } for w in range(n_w)}

    return {
        "R2_peak": float(peak_R2[w_best]),
        "peak_scheme": weight_labels[w_best],
        "peak_CI": (float(peak_CI[w_best, 0]), float(peak_CI[w_best, 1])),
        "w_best": weight_labels[w_best],
        "peak_by_family": peak_by_family,
        "clusters": clusters,
    }


def bootstrap_summary_table(result, ci=(2.5, 97.5), weight_labels=WEIGHT_LABELS,
                            as_frame=True):
    """
    One row per (season, K cluster, R² sub-cluster), from
    ``analyze_correlation_seasonal(..., n_boot>0)`` (or a ``{"whole period":
    {...}}`` wrapper for the non-seasonal case) — the paper table. Columns:

    - ``season``
    - ``cluster`` : ordinal (1, 2, ...) of the K cluster, by increasing ``K``.
    - ``K`` [``K_CI``] : median of the K cluster's families' peak K, with its
      bootstrap CI (integer interval) — the K-extent of the box drawn for this
      cluster in the response-surface figure (Fig. 3). Repeated on each
      sub-cluster row.
    - ``ref_scheme`` [``ref_K`` / ``ref_K_CI``] [``ref_R2`` / ``ref_R2_CI``] :
      a one-name handle on the K cluster — for a 2-scheme cluster the member
      with the shorter peak K, for a larger cluster the member whose peak K is
      nearest the cluster median (ties -> shorter K) — with THAT scheme's own
      peak K and R², each with its own bootstrap CI (not the cluster/box
      values above). Cosmetic; the SIDI still uses each scheme's own K.
      Repeated per row.
    - ``sub-cluster`` : ordinal (1, 2, ...) of the R² sub-cluster within that K
      cluster, by decreasing ``R2`` (so 1 is the strongest). A K cluster whose
      families do not differ in R² has a single sub-cluster row.
    - ``families`` : the weighting schemes in that R² sub-cluster.
    - ``R2`` [``R2_CI``] : median of the sub-cluster's families' peak R², with
      its bootstrap CI — the R²-extent of that sub-cluster's box in Fig. 3.
      Median, not max → no winner's curse.

    ``as_frame=True`` returns a pandas DataFrame (falls back to a list of dicts).
    """
    def _fmt_ci(t, ints=False):
        if not (np.isfinite(t[0]) and np.isfinite(t[1])):
            return "-"
        return f"[{int(t[0])}, {int(t[1])}]" if ints else f"[{t[0]:.3f}, {t[1]:.3f}]"

    rows = []
    for name, d in result.items():
        if name == "summary" or not isinstance(d, dict) or "R2_matrix" not in d:
            continue
        boot = d.get("R2_boot")
        if boot is None:
            continue
        s = d.get("summary") or _peak_summary(d["R2_matrix"], boot, ci, weight_labels)
        for j, c in enumerate(s.get("clusters", []), start=1):
            k_val = round(c["K_cluster"], 1) if np.isfinite(c["K_cluster"]) else np.nan
            k_ci = _fmt_ci(c["K_cluster_CI"], ints=True)
            subs = c.get("subclusters") or [{
                "families": c["families"], "R2": c["R2_cluster"],
                "R2_CI": c["R2_cluster_CI"]}]
            ref_r2 = c.get("ref_R2", np.nan)
            ref_k_ci = _fmt_ci(c.get("ref_K_CI", (np.nan, np.nan)), ints=True)
            ref_r2_ci = _fmt_ci(c.get("ref_R2_CI", (np.nan, np.nan)))
            for si, sc in enumerate(subs, start=1):
                rows.append({
                    "season": name,
                    "cluster": j,
                    "K": k_val,
                    "K_CI": k_ci,
                    "ref_scheme": c.get("ref_scheme"),
                    "ref_K": c.get("ref_K"),
                    "ref_K_CI": ref_k_ci,
                    "ref_R2": round(ref_r2, 3) if np.isfinite(ref_r2) else np.nan,
                    "ref_R2_CI": ref_r2_ci,
                    "sub-cluster": si,
                    "families": ", ".join(sc["families"]),
                    "R2": round(sc["R2"], 3) if np.isfinite(sc["R2"]) else np.nan,
                    "R2_CI": _fmt_ci(sc["R2_CI"]),
                })
    if as_frame:
        try:
            import pandas as pd
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        except Exception:
            pass
    return rows


def _one_bootstrap_replica(P_yr, Q_yr, calcP, n_base_years, calcQ, n_base_years_Q,
                           K, block_years, circular, seed, season_months=None):
    """One year-aligned block-bootstrap replica of the R2(K, weight) surface.

    `P_yr`, `Q_yr` are (n_years, 12): the overlap trimmed to whole years and
    reshaped year-major, so a block is a set of rows. Returns (K, 5) when
    `season_months` is None, else {name: (K, 5)}.
    """
    from drought_scan.core import BaseDroughtAnalysis   # lazy: avoid import cycle

    rng = np.random.default_rng(seed)
    n_years = P_yr.shape[0]
    rows = _year_block_index(n_years, block_years, rng, circular)

    P_b = P_yr[rows].reshape(-1)
    Q_b = Q_yr[rows].reshape(-1)
    T = P_b.size
    months = (np.arange(T) % 12) + 1
    years = _BOOT_BASE_YEAR + np.arange(T) // 12
    m_cal_b = np.column_stack([months, years])

    tb2P = _BOOT_BASE_YEAR + min(n_base_years, n_years) - 1
    tb2Q = _BOOT_BASE_YEAR + min(n_base_years_Q, n_years) - 1
    spi_P = _spi_set_from_series(P_b, m_cal_b, calcP, _BOOT_BASE_YEAR, tb2P, K)
    sqi1 = _spi_set_from_series(Q_b, m_cal_b, calcQ, _BOOT_BASE_YEAR, tb2Q, 1)[0]

    # NaN-mask the first (k-1) timesteps after every block join, per scale:
    # SIDI(k, t) then survives only where scales 1..k are all clean at t.
    L = block_years * 12
    dist = _junction_distance(T, L)
    for k in range(2, K + 1):
        spi_P[k - 1, dist < (k - 1)] = np.nan

    K_range = np.arange(1, K + 1)
    if season_months is None:
        return BaseDroughtAnalysis._r2_surface(spi_P, sqi1, K_range, min_valid=1)
    out = {}
    for name, mlist in season_months.items():
        m = np.isin(months, mlist)
        out[name] = BaseDroughtAnalysis._r2_surface(
            spi_P[:, m], sqi1[m], K_range, min_valid=10)
    return out


def _bootstrap_r2(self_obj, streamflow, self_indices, streamflow_indices,
                  n_boot, block_length, ci, circular, random_state, seasons=None):
    """Run n_boot year-aligned block-bootstrap replicas of the full pipeline
    and reduce to per-cell percentile bands.

    Non-seasonal  -> (boot (B,K,5), ci (2,K,5), meta).
    seasons given -> ({season: boot}, {season: ci}, meta).
    """
    from joblib import Parallel, delayed, cpu_count

    m_cal_ov = self_obj.m_cal[self_indices]
    P_ov = np.asarray(self_obj.ts, float)[self_indices]
    Q_ov = np.asarray(streamflow.ts, float)[streamflow_indices]

    # trim the overlap to whole years starting in January
    jan = np.where(m_cal_ov[:, 0].astype(int) == 1)[0]
    if jan.size == 0:
        raise ValueError("block bootstrap needs at least one January in the overlap.")
    j0 = int(jan[0])
    n_years = (len(P_ov) - j0) // 12
    if n_years < 4:
        raise ValueError(f"block bootstrap needs >= 4 whole overlap years, got {n_years}.")
    end = j0 + 12 * n_years
    P_yr = P_ov[j0:end].reshape(n_years, 12)
    Q_yr = Q_ov[j0:end].reshape(n_years, 12)
    T = n_years * 12

    L = int(block_length) if block_length else max(24, 2 * self_obj.K)
    L = min(_round_to_year(L), 12 * n_years)
    block_years = L // 12

    n_base_years = self_obj.end_baseline_year - self_obj.start_baseline_year + 1
    n_base_years_Q = streamflow.end_baseline_year - streamflow.start_baseline_year + 1

    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, 2 ** 32 - 1, size=int(n_boot))
    season_months = dict(seasons) if seasons is not None else None

    # Cap the pool well below n_jobs=-1: every loky worker re-imports the whole
    # drought_scan.core stack (geopandas/xarray/netCDF4/matplotlib), ~0.3-0.5 GB
    # resident each, so one-process-per-core saturates RAM before any replica
    # runs. Half the cores, at most 4, is enough for a job this short-lived.
    n_jobs = max(1, min(4, cpu_count() // 2))

    print(f"  block bootstrap: B={n_boot}, L={L} months ({block_years}y) "
          f"{'circular' if circular else 'moving'} blocks over {n_years} overlap years "
          f"(rebuilds the full SPI pipeline {n_boot}x on {n_jobs} workers)...")

    # Drop any open pyplot figures before forking: each worker would otherwise
    # inherit a copy of the figure manager (mirrors compute_spatial_sidi /
    # spatial_spi in core.py).
    import matplotlib.pyplot as plt
    plt.close("all")

    # Context manager so the loky pool is torn down on exit instead of lingering
    # for its default 5-minute idle timeout (which is what makes a second call
    # spike memory again).
    with Parallel(n_jobs=n_jobs) as parallel:
        reps = parallel(
            delayed(_one_bootstrap_replica)(
                P_yr, Q_yr, self_obj.calculation_method, n_base_years,
                streamflow.calculation_method, n_base_years_Q,
                self_obj.K, block_years, circular, int(s), season_months)
            for s in seeds
        )

    contam = _contamination_fraction(T, L, self_obj.K)
    meta = {
        "n_boot": int(n_boot), "block_length": int(L), "block_years": int(block_years),
        "circular": bool(circular), "ci": tuple(ci),
        "n_blocks": int(np.ceil(n_years / block_years)),
        "overlap_years": int(n_years), "overlap_length": int(T),
        "contaminated_fraction": contam,
        "eff_n_per_scale": np.rint(T * (1.0 - contam)).astype(int),
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # All-NaN slices -> NaN CI, expected
        if seasons is None:
            boot = np.stack(reps)
            return boot, np.nanpercentile(boot, list(ci), axis=0), meta
        boot_by_season, ci_by_season = {}, {}
        for name in seasons:
            stack = np.stack([r[name] for r in reps])
            boot_by_season[name] = stack
            ci_by_season[name] = np.nanpercentile(stack, list(ci), axis=0)
        return boot_by_season, ci_by_season, meta


def _print_summary_table(summary):
    """Pretty-print the cluster table (DataFrame or list of dicts)."""
    print("\n  --- bootstrap clusters (one row per season x K cluster x R2 sub-cluster) ---")
    rows = summary if not hasattr(summary, "to_string") else None
    if rows is None:
        print(summary.to_string())
        rows = summary.to_dict("records")
    else:
        for row in rows:
            print("   " + "  ".join(f"{k}={v}" for k, v in row.items()))
    _print_cluster_refs(rows)


def _print_cluster_refs(rows):
    """One line per K cluster: its display reference scheme (2 schemes -> the
    shorter peak K; more -> the member whose peak K is nearest the cluster
    median, ties -> shorter K), with that scheme's own K and peak R2 - each
    with its own bootstrap CI - and the cluster median K in parentheses as a
    reminder of how it was picked. Scheme names are abbreviated (ew, ldw,
    geodw, liw, geoiw) for this recap only; the table itself (and everything
    else) keeps the full WEIGHT_LABELS names."""
    if not rows or "ref_scheme" not in rows[0]:
        return
    print("\n  cluster reference (2 schemes -> shorter K; more -> nearest the cluster median K):")
    seen = set()
    for r in rows:
        key = (r.get("season"), r.get("cluster"))
        if key in seen:
            continue
        seen.add(key)
        name = _WEIGHT_ABBR.get(r.get("ref_scheme"), r.get("ref_scheme"))
        ref_r2 = r.get("ref_R2")
        r2_txt = (f"{ref_r2:.3f}" if isinstance(ref_r2, (int, float)) and np.isfinite(ref_r2)
                  else "-")
        k_ci = r.get("ref_K_CI", "-")
        r2_ci = r.get("ref_R2_CI", "-")
        print(f"    {str(r.get('season')):<10} cluster {r.get('cluster')}:  "
              f"ref = {name}  K={r.get('ref_K')} {k_ci}  R2={r2_txt} {r2_ci}"
              f"   (cluster median K = {r.get('K')})")
