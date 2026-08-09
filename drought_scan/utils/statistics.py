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

# helpers
# PRIVATE CORE FITTER  (single distribution, single clean array)
def _fit_single_dist(dataset,dist=None,shift_for_gamma = True):
    """
    Fit *one* distribution to a clean (finite, 1-D) array and return stats.

    Parameters
    ----------
    dataset : np.ndarray
        Finite-only values (caller must filter).
    dist : {"gaussian", "gamma", "pearson3", "kde"}
    shift_for_gamma : bool
        If True, apply dataset + 1 before Gamma fitting to avoid zeros.

    Returns
    -------
    dict
        distribution, params, KS_statistic, KS_p_value, log_likelihood,
        AIC, k_params, error_percent, goodness_percent.

    Raises
    ------
    ValueError
        If Gamma is requested and the data used for fitting (dataset, or
        dataset + 1 when shift_for_gamma=True) still contains non-positive values.
    """
    if dist is None:
        dist = 'gaussian'
    dist = dist.lower()

    # ── Gamma ──────────────────────────────────────────────────────────────
    if dist == "gamma":
        data_used = dataset + 1.0 if shift_for_gamma else dataset
        if np.any(data_used <= 0):
            raise ValueError(
                "Gamma distribution requires strictly positive values. "
                "Set shift_for_gamma=True or pre-process the data."
            )
        shape, loc, scale = stats.gamma.fit(data_used, floc=0)
        D, p_ks = stats.kstest(data_used, "gamma", args=(shape, loc, scale))
        logpdf = stats.gamma.logpdf(data_used, shape, loc=loc, scale=scale)
        params = {"shape": shape, "loc": loc, "scale": scale,
                  "shift_applied": shift_for_gamma}
        k = 3

    # ── Pearson type III ───────────────────────────────────────────────────
    elif dist == "pearson3":
        data_used = dataset
        shape, loc, scale = stats.pearson3.fit(data_used)
        D, p_ks = stats.kstest(data_used, "pearson3", args=(shape, loc, scale))
        logpdf = stats.pearson3.logpdf(data_used, shape, loc=loc, scale=scale)
        params = {"shape": shape, "loc": loc, "scale": scale}
        k = 3

    # ── Gaussian ────────────────────────────────────────────────────────────
    elif dist == "gaussian":
        data_used = dataset
        mu    = np.mean(data_used)
        sigma = np.std(data_used, ddof=0)
        if sigma == 0:
            raise ValueError("Gaussian fit requires non-zero variance.")
        D, p_ks = stats.kstest(data_used, "norm", args=(mu, sigma))
        logpdf = stats.norm.logpdf(data_used, loc=mu, scale=sigma)
        params = {"mu": mu, "sigma": sigma}
        k = 2

    # ── Gaussian KDE (non-parametric) ───────────────────────────────────────
    elif dist == "kde":
        data_used = dataset
        kde = stats.gaussian_kde(data_used, bw_method="silverman")

        # Vectorised CDF via trapezoidal integration on a fine grid
        std   = np.std(data_used)
        bw    = kde.factor * std
        x_min = data_used.min() - 4 * bw
        x_max = data_used.max() + 4 * bw
        x_grid   = np.linspace(x_min, x_max, 4096)
        pdf_grid = kde.evaluate(x_grid)
        cdf_grid = np.cumsum(pdf_grid) * (x_grid[1] - x_grid[0])
        cdf_grid /= cdf_grid[-1]                       # ensure [0, 1]

        def kde_cdf(x: np.ndarray) -> np.ndarray:
            return np.interp(x, x_grid, cdf_grid)

        D, p_ks = stats.kstest(data_used, kde_cdf)

        pdf_vals = kde.evaluate(data_used)
        pdf_vals = np.clip(pdf_vals, 1e-300, None)
        logpdf   = np.log(pdf_vals)

        # Store only serialisable primitives; expose CDF callable separately
        params = {
            "bw_method":  "silverman",
            "bw_factor":  float(kde.factor),
            "n_fit":      int(len(data_used)),
            "_kde_cdf":   kde_cdf,      # callable, not JSON-serialisable
        }
        k = None  # AIC undefined for KDE

    else:
        raise ValueError(
            f"Unknown distribution '{dist}'. "
            "Choose from: 'gaussian', 'gamma', 'pearson3', 'kde'."
        )

    # ── Derived metrics ─────────────────────────────────────────────────────
    log_likelihood = float(np.sum(logpdf))
    aic = (2 * k - 2 * log_likelihood) if k is not None else np.nan

    return {
        "distribution":    dist,
        "params":          params,
        "KS_statistic":    float(D),
        "KS_p_value":      float(p_ks),
        "log_likelihood":  log_likelihood,
        "AIC":             float(aic) if not np.isnan(aic) else np.nan,
        "k_params":        k,
        "error_percent":   float(100.0 * D),
        "goodness_percent": float(100.0 * (1.0 - D)),
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

    for i in range(n_bootstrap):
        # Draw synthetic sample from fitted distribution
        if dist == "gaussian":
            sample = rng.normal(params["mu"], params["sigma"], size=n)
        elif dist == "gamma":
            sample = stats.gamma.rvs(
                params["shape"], loc=params["loc"], scale=params["scale"],
                size=n, random_state=rng.integers(1 << 31)
            )
            if params["shift_applied"]:
                sample -= 1.0          # undo shift for consistent comparison
        elif dist == "pearson3":
            sample = stats.pearson3.rvs(
                params["shape"], loc=params["loc"], scale=params["scale"],
                size=n, random_state=rng.integers(1 << 31)
            )
        else:  # kde — resample from original data (smoothed bootstrap)
            std = np.std(dataset)
            bw  = params["bw_factor"] * std
            idx    = rng.integers(0, n, size=n)
            sample = dataset[idx] + rng.normal(0, bw, size=n)

        # Refit and compute D*
        try:
            res = _fit_single_dist(sample, dist, shift_for_gamma=(dist == "gamma"))
            D_boot[i] = res["KS_statistic"]
        except Exception:
            D_boot[i] = np.nan

    valid = D_boot[np.isfinite(D_boot)]
    return float(np.mean(valid >= D_obs))

# Fit all four families to `dataset` and return a comparison dict.
def _analyze_all(dataset, shift_for_gamma = True,  n_bootstrap=0, seed=None):
    """
    Fit all four families to `dataset` and return a comparison dict.

    Selection rule
    --------------
    1. Best KS (all 4) → primary recommendation.
    2. Best AIC (parametric 3 only) → secondary note.
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

    # ── Rank by KS statistic (lower = better), None last ──────────────────
    valid_fits = {d: v for d, v in fits.items() if v is not None}
    best_ks  = min(valid_fits, key=lambda d: valid_fits[d]["KS_statistic"])

    # ── Best parametric by AIC ─────────────────────────────────────────────
    parametric = {d: v for d, v in valid_fits.items()
                  if d != "kde" and not np.isnan(v["AIC"])}
    best_aic = min(parametric, key=lambda d: parametric[d]["AIC"]) if parametric else None

    return {
        "skewness":         skewness,
        "normality_p_value": float(p_normal),
        "fits":             fits,
        "best_by_KS":       best_ks,
        "best_by_AIC":      best_aic,
        "recommendation":   best_ks,       # primary
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
    best = analysis["best_by_KS"]

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

        # CDF curve
        if d == "gaussian":
            cdf = stats.norm.cdf(x_plot, loc=params["mu"], scale=params["sigma"])
            pdf = stats.norm.pdf(x_plot, loc=params["mu"], scale=params["sigma"])
        elif d == "gamma":
            shift = 1.0 if params["shift_applied"] else 0.0
            cdf = stats.gamma.cdf(
                x_plot + shift, params["shape"],
                loc=params["loc"], scale=params["scale"]
            )
            pdf = stats.gamma.pdf(
                x_plot + shift, params["shape"],
                loc=params["loc"], scale=params["scale"]
            )
        elif d == "pearson3":
            cdf = stats.pearson3.cdf(
                x_plot, params["shape"],
                loc=params["loc"], scale=params["scale"]
            )
            pdf = stats.pearson3.pdf(
                x_plot, params["shape"],
                loc=params["loc"], scale=params["scale"]
            )
        else:  # kde
            cdf = params["_kde_cdf"](x_plot)
            pdf = stats.gaussian_kde(dataset, bw_method="silverman").evaluate(x_plot)

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

def test_standardization(data, groups=None, shift_for_gamma = True,
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
    shift_for_gamma : bool, default True
        Add 1 to data before Gamma fitting to handle zeros.
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
            {skewness, normality_p_value, fits, best_by_KS, best_by_AIC,
             recommendation}
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
    shift_for_gamma= True, plot = True, n_bootstrap: int = 0,
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
    shift_for_gamma : bool, default True
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

            if dist == "gaussian":
                cdf = stats.norm.cdf(x_plot, loc=params["mu"], scale=params["sigma"])
                pdf = stats.norm.pdf(x_plot, loc=params["mu"], scale=params["sigma"])
                lbl = "Gaussian fit"
            elif dist == "gamma":
                shift = 1.0 if params["shift_applied"] else 0.0
                cdf = stats.gamma.cdf(
                    x_plot + shift, params["shape"],
                    loc=params["loc"], scale=params["scale"]
                )
                pdf = stats.gamma.pdf(
                    x_plot + shift, params["shape"],
                    loc=params["loc"], scale=params["scale"]
                )
                lbl = "Gamma fit"
            elif dist == "pearson3":
                cdf = stats.pearson3.cdf(
                    x_plot, params["shape"],
                    loc=params["loc"], scale=params["scale"]
                )
                pdf = stats.pearson3.pdf(
                    x_plot, params["shape"],
                    loc=params["loc"], scale=params["scale"]
                )
                lbl = "Pearson III fit"
            else:  # kde
                cdf = params["_kde_cdf"](x_plot)
                pdf = stats.gaussian_kde(dataset_clean, bw_method="silverman").evaluate(x_plot)
                lbl = "KDE (Silverman)"

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
        Returns CDF values ∈ (cdf_clip, 1-cdf_clip).
        """
        x = dataset_clean

        if dist == "gaussian":
            p = stats.norm.cdf(x, loc=params["mu"], scale=params["sigma"])

        elif dist == "gamma":
            shift = 1.0 if params["shift_applied"] else 0.0
            p = stats.gamma.cdf(
                x + shift, params["shape"],
                loc=params["loc"], scale=params["scale"],
            )

        elif dist == "pearson3":
            p = stats.pearson3.cdf(
                x, params["shape"],
                loc=params["loc"], scale=params["scale"],
            )

        elif dist == "kde":
            # Rebuild the CDF from scratch (kde_object not guaranteed serialisable)
            kde = stats.gaussian_kde(dataset_clean, bw_method="silverman")
            std  = np.std(dataset_clean)
            bw   = kde.factor * std
            x_min = dataset_clean.min() - 4 * bw
            x_max = dataset_clean.max() + 4 * bw
            grid     = np.linspace(x_min, x_max, 4096)
            pdf_grid = kde.evaluate(grid)
            cdf_grid = np.cumsum(pdf_grid) * (grid[1] - grid[0])
            cdf_grid /= cdf_grid[-1]
            p = np.interp(x, grid, cdf_grid)

        else:
            raise ValueError(f"Unknown distribution: '{dist}'")

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

def plot_cdf_comparison(data, dist="gamma", params=None, shift_for_gamma=True,unit=None):
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
    shift_for_gamma : bool
        If True and dist == "gamma", data are shifted by +1 as in fit_distribution_stats.
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

    # Apply gamma shift if the user wants to be consistent with fitting step
    if dist == "gamma":
        data_used = data + 1 if shift_for_gamma else data
    else:
        data_used = data

    # Fit params if not provided
    if params is None:
        if dist == "gamma":
            shape, loc, scale = stats.gamma.fit(data_used, floc=0)
        elif dist == "pearson3":
            shape, loc, scale = stats.pearson3.fit(data_used)
        else:  # gaussian
            mu = np.mean(data_used)
            sigma = np.std(data_used, ddof=0)
    else:
        # read precomputed params from fit_distribution_stats()
        if dist in {"gamma", "pearson3"}:
            shape = params["shape"]
            loc   = params["loc"]
            scale = params["scale"]
        else:
            mu    = params["mu"]
            sigma = params["sigma"]

    # --------------------------
    # Empirical CDF
    # --------------------------
    sorted_data = np.sort(data_used)
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    # --------------------------
    # Theoretical CDF
    # --------------------------
    if dist == "gamma":
        x = np.linspace(sorted_data.min(), sorted_data.max(), 400)
        cdf = stats.gamma.cdf(x, shape, loc=loc, scale=scale)
        pdf = stats.gamma.pdf(x, shape, loc=loc, scale=scale)
        label = ["Gamma fit","Gamma CDF"]
    elif dist == "pearson3":
        x = np.linspace(sorted_data.min(), sorted_data.max(), 400)
        cdf = stats.pearson3.cdf(x, shape, loc=loc, scale=scale)
        pdf = stats.pearson3.pdf(x, shape, loc=loc, scale=scale)
        label =["Pearson III fit", "Pearson III CDF"]
    else:
        x = np.linspace(sorted_data.min(), sorted_data.max(), 400)
        cdf = stats.norm.cdf(x, loc=mu, scale=sigma)
        pdf = stats.norm.pdf(x, loc=mu, scale=sigma)
        label = ["Gaussian fit", "Gaussian CDF"]

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


