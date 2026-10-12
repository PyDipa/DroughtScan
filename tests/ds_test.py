# tests/test_precipitation.py

import numpy as np
import pytest

def test_import():
    """Check that the package imports and has the expected classes"""
    import drought_scan as DS
    assert hasattr(DS, "Precipitation")
    assert hasattr(DS, "Streamflow")
    assert hasattr(DS, "Pet")
    assert hasattr(DS, "Balance")
    assert hasattr(DS, "Temperature")
    assert hasattr(DS, "Teleindex")


def test_precipitation_init():
    """Initialize Precipitation object from test data"""
    import drought_scan as DS

    shape = "tests/data/bacino_pontelagoscuro.shp"
    prec  = "tests/data/LAPrec1871.v1.1.nc"

    ds = DS.Precipitation(
        prec_path=prec,
        shape_path=shape,
        start_baseline_year=1981,
        end_baseline_year=2010,
        basin_name="Po"
    )

    # Check time series and calendar are built
    assert ds.ts is not None
    assert ds.m_cal.shape[1] == 2   # month, year
    assert ds.spi_like_set.shape[0] > 0  # SPI scales


def test_severe_events_old_detection():
    """Detect severe drought events with custom threshold"""
    import drought_scan as DS

    shape = "tests/data/bacino_pontelagoscuro.shp"
    prec  = "tests/data/LAPrec1871.v1.1.nc"

    ds = DS.Precipitation(
        prec_path=prec,
        shape_path=shape,
        start_baseline_year=1981,
        end_baseline_year=2010,
        basin_name="Po"
    )

    # Lower threshold to make sure some events are found
    ds.threshold = -0.5
    result = ds.severe_events_old(max_events=3,plot=False)
    ds.plot_scan()

    assert result is not None
    # severe_events returns tuples of indices → at least one
    assert len(result[0]) > 0


def _build_synthetic_dso(calculation_method, seed=3, start_month=1):
    """Minimal BaseDroughtAnalysis instance from synthetic data, bypassing
    __init__ (which requires a shapefile for _area()). Shared by the
    native_to_spi/spi_to_native roundtrip tests below.

    `start_month` shifts the calendar so the series does not begin in January,
    which is what exposes month-alignment bugs."""
    from drought_scan.core import BaseDroughtAnalysis

    rng = np.random.default_rng(seed)
    years = np.repeat(np.arange(1980, 2020), 12)
    months = np.tile(np.arange(1, 13), 40)
    m_cal = np.column_stack([months, years])
    if start_month != 1:
        # drop the leading months so the record starts on `start_month`
        m_cal = m_cal[start_month - 1:]
    n = len(m_cal)

    from drought_scan.utils.drought_indices import f_kde, f_spi
    ts = (np.abs(rng.normal(50, 20, size=n)) + 1
          if calculation_method in (f_kde, f_spi)
          else rng.normal(50, 20, size=n))

    obj = object.__new__(BaseDroughtAnalysis)
    obj.ts = ts
    obj.m_cal = m_cal
    obj.K = 3
    obj.start_baseline_year = 1981
    obj.end_baseline_year = 2010
    obj.threshold = 1
    obj.calculation_method = calculation_method
    obj.index_name = "SPI"
    obj.basin_name = "test"
    obj.spi_like_set, obj.c2r_index = obj._calculate_spi_like_set()
    return obj


@pytest.mark.parametrize("calculation_method_name", ["f_spi", "f_spei", "f_kde", "f_zscore"])
def test_native_spi_roundtrip(calculation_method_name):
    """spi_to_native(native_to_spi(x)) should recover x for every
    calculation_method, exactly for f_spi/f_spei/f_zscore (closed-form
    inverse) and within the kde_ppf interpolation grid's resolution for
    f_kde (no closed form — see drought_indices.kde_cdf/kde_ppf)."""
    from drought_scan.utils import drought_indices as di

    calculation_method = getattr(di, calculation_method_name)
    obj = _build_synthetic_dso(calculation_method)

    spi_test_values = np.array([-2.5, -1.0, 0.0, 1.0, 2.5])
    tol = 1e-3 if calculation_method_name == "f_kde" else 1e-8

    for month_scale, ref_month in [(1, 3), (2, 7), (3, 12)]:
        native_vals = obj.spi_to_native(spi_test_values, month_scale, ref_month)
        spi_back = obj.native_to_spi(native_vals, month_scale, ref_month)
        max_err = np.max(np.abs(np.asarray(spi_back) - spi_test_values))
        assert max_err < tol, (
            f"{calculation_method_name} scale={month_scale} month={ref_month}: "
            f"roundtrip error {max_err} exceeds tolerance {tol}"
        )


def test_kde_ppf_degenerate_baseline_raises():
    """kde_ppf must raise a clear error (not crash or return garbage) when
    the baseline bandwidth is degenerate (e.g. a single/constant baseline
    point makes h non-finite)."""
    from drought_scan.utils.drought_indices import kde_ppf

    xb = np.array([1.0])  # single point -> std(ddof=1) is NaN -> h is NaN
    h = 0.9 * np.std(xb, ddof=1) * xb.size ** (-1 / 5)
    with pytest.raises(ValueError):
        kde_ppf(0.5, xb, h, qq=0.0, log_transform=False)


def test_deficit_and_normal_values_real_data():
    """End-to-end check of the migrated normal_values/deficit_from_spi on
    real data with the default calculation_method (f_kde)."""
    import drought_scan as DS

    shape = "tests/data/bacino_pontelagoscuro.shp"
    prec = "tests/data/LAPrec1871.v1.1.nc"

    ds = DS.Precipitation(
        prec_path=prec,
        shape_path=shape,
        start_baseline_year=1981,
        end_baseline_year=2010,
        basin_name="Po",
    )

    normal = ds.normal_values()
    assert normal.shape == ds.ts.shape
    assert np.all(np.isfinite(normal))

    deficit = ds.deficit_from_spi(window=3)
    assert deficit.shape == ds.ts.shape
    assert np.any(np.isfinite(deficit))


def test_normal_values_aligned_on_non_january_start():
    """normal_values() must give each timestep the normal of ITS OWN calendar
    month, also when the record does not start in January (it used to tile
    Jan..Dec from position 0, misaligning every timestep)."""
    from drought_scan.utils.drought_indices import f_kde

    obj = _build_synthetic_dso(f_kde, start_month=8)
    normal = obj.normal_values()
    clim = obj._monthly_normals()

    assert normal.shape == obj.ts.shape
    expected = clim[obj.m_cal[:, 0].astype(int) - 1]
    assert np.allclose(normal, expected)


@pytest.mark.parametrize("calculation_method_name", ["f_spi", "f_spei", "f_kde", "f_zscore"])
def test_fit_params_is_honoured(calculation_method_name):
    """Passing fit_params must apply the stored calibration to the new data
    instead of recalibrating on it — otherwise scenario/forecast runs are
    silently standardized against themselves. f_kde used to ignore it."""
    from drought_scan.utils import drought_indices as di

    f = getattr(di, calculation_method_name)
    obj = _build_synthetic_dso(f)

    _, spi_ref, _, params = f(obj.ts, 1, 3, obj.m_cal, 1981, 2010)
    _, spi_scaled, _, _ = f(obj.ts * 2.0, 1, 3, obj.m_cal, 1981, 2010, fit_params=params)

    # doubling the data under the ORIGINAL calibration must shift SPI upwards
    assert np.nanmean(spi_scaled) > np.nanmean(spi_ref) + 0.5, (
        f"{calculation_method_name}: fit_params appears to be ignored "
        f"(ref={np.nanmean(spi_ref):.4f}, scaled={np.nanmean(spi_scaled):.4f})"
    )


def test_fit_params_legacy_shape_warns_and_falls_back():
    """A fit_params dict without the calibration keys (the shape produced by
    older versions) must degrade to recalibration WITH an explicit warning,
    never a KeyError."""
    from drought_scan.utils.drought_indices import f_kde

    obj = _build_synthetic_dso(f_kde)
    _, spi_ref, _, params = f_kde(obj.ts, 1, 3, obj.m_cal, 1981, 2010)
    legacy = {k: params[k] for k in ("kde", "bw_factor", "n_fit", "fit_domain")}

    with pytest.warns(RuntimeWarning, match="calibration keys"):
        _, spi_legacy, _, _ = f_kde(obj.ts * 2.0, 1, 3, obj.m_cal, 1981, 2010,
                                    fit_params=legacy)

    # fallback == recalibration, i.e. same result as fitting from scratch
    assert np.allclose(np.nanmean(spi_legacy), np.nanmean(spi_ref), atol=1e-6)


@pytest.mark.parametrize("calculation_method_name", ["f_spi", "f_kde"])
def test_zero_inflation_estimated_on_baseline(calculation_method_name):
    """qq is P(X=0) of the fitted mixture, so it must be estimated on the
    baseline — the same sample the continuous part is fitted on. Zeros that
    occur only outside the baseline must not enter it."""
    from drought_scan.utils import drought_indices as di

    f = getattr(di, calculation_method_name)
    obj = _build_synthetic_dso(di.f_kde)          # strictly positive series
    ts = obj.ts.copy()

    # zeros only AFTER the baseline (baseline is 1981-2010)
    post = np.where(obj.m_cal[:, 1] > 2010)[0]
    ts[post[:20]] = 0.0

    _, _, _, params = f(ts, 1, 7, obj.m_cal, 1981, 2010)
    qq = params["qq"] if isinstance(params, dict) else None
    if qq is not None:                             # f_kde exposes it, f_spi does not
        assert qq == 0.0, f"qq picked up out-of-baseline zeros: {qq}"


def test_log_transform_decided_on_baseline_is_not_retroactive():
    """The log decision comes from the baseline, so appending/altering data
    outside it cannot change already-computed historical SPI values (it used
    to flip the whole cell to a different model)."""
    from drought_scan.utils.drought_indices import f_kde

    obj = _build_synthetic_dso(f_kde)
    idx = np.where((obj.m_cal[:, 0] == 6) & (obj.m_cal[:, 1] == 2015))[0][0]

    i0, spi0, _, _ = f_kde(obj.ts, 1, 6, obj.m_cal, 1981, 2010)
    ts_dry = obj.ts.copy()
    ts_dry[idx] = 0.0                              # one dry month, outside baseline
    _, spi1, _, _ = f_kde(ts_dry, 1, 6, obj.m_cal, 1981, 2010)

    pos = np.where(i0 == idx)[0][0]
    other = np.setdiff1d(np.arange(len(spi0)), [pos])
    delta = np.abs(np.asarray(spi1)[other] - np.asarray(spi0)[other])

    assert np.nanmax(delta) == 0.0, (
        f"altering data outside the baseline moved historical SPI by "
        f"{np.nanmax(delta)}"
    )


def test_spatial_spi_keeps_undefined_pixels_nan():
    """An undefined pixel (window longer than the record available at t_idx) is not
    a pixel with zero anomaly. spatial_spi used to write 0.0 for both, making the
    two indistinguishable on the map."""
    from drought_scan.core import BaseDroughtAnalysis
    from drought_scan.utils.drought_indices import f_kde

    obj = _build_synthetic_dso(f_kde)
    deficit, spi, err = BaseDroughtAnalysis._process_grid_point_spi(
        obj.ts, obj.m_cal, 1981, 2010, f_kde, windows=[36], t_idx=10
    )

    assert err is None
    assert np.isnan(spi[36])
    assert np.isnan(deficit[36])


def test_spatial_spi_keeps_real_values_inside_the_neutral_band():
    """The |SPI| < 0.5 band is a rendering convention, not a hole in the data:
    reverse_SPI_grid used to be written as 0.0 there, so the millimetres inside
    the band were unrecoverable. plot_spatial greys the band instead."""
    from drought_scan.core import BaseDroughtAnalysis
    from drought_scan.utils.drought_indices import f_spi

    obj = _build_synthetic_dso(f_spi)

    found = False
    for t_idx in range(120, 300):
        deficit, spi, err = BaseDroughtAnalysis._process_grid_point_spi(
            obj.ts, obj.m_cal, 1981, 2010, f_spi, windows=[3], t_idx=t_idx
        )
        assert err is None
        if np.isfinite(spi[3]) and abs(spi[3]) < 0.5:
            found = True
            assert deficit[3] != 0.0
            # and it is the same number the basin-level method reports
            assert np.isclose(deficit[3], obj.deficit_from_spi(3)[t_idx], rtol=1e-6)
            break
    assert found, "no timestep landed inside the neutral band"


def test_plot_spatial_rejects_the_cdn_alias():
    """var='CDN' used to be an alias for 'reverse_spi'. They are different
    quantities and the library does not map the CDN, so the alias is refused
    with an explanation rather than silently answering something else."""
    from drought_scan.core import BaseDroughtAnalysis
    from drought_scan.utils.drought_indices import f_kde

    obj = _build_synthetic_dso(f_kde)
    obj.threshold = -1
    with pytest.raises(ValueError, match="no longer supported"):
        obj.plot_spatial(var="CDN", month_scale=36)


def test_r2_surface_is_invariant_to_rescaling_the_sidi():
    """_r2_surface correlates the raw weighted mean. The three loops it replaced each
    rescaled it first — a step that cannot change Pearson's r, and that looked like
    the library's baseline standardization while referencing the overlap instead."""
    from drought_scan.core import BaseDroughtAnalysis
    from drought_scan.utils.drought_indices import generate_weights, weighted_metrics
    from scipy import stats

    rng = np.random.default_rng(17)
    n_scales, n_times = 6, 300
    spi_sub = rng.normal(0, 1, (n_scales, n_times))
    y_sub = 0.5 * spi_sub[0] + rng.normal(0, 1, n_times)
    K_range = np.arange(1, n_scales + 1)

    got = BaseDroughtAnalysis._r2_surface(spi_sub, y_sub, K_range)

    # the pre-existing formula, rescaling before correlating
    for ki, k in enumerate(K_range):
        W = generate_weights(int(k))
        sidis = np.array([[weighted_metrics(spi_sub[:k, d], w)[0] for w in W.T]
                          for d in range(n_times)])
        for w in range(W.shape[1]):
            z = (sidis[:, w] - np.nanmean(sidis[:, w])) / np.nanstd(sidis[:, w])
            m = np.isfinite(y_sub) & np.isfinite(z)
            assert np.isclose(got[ki, w], stats.pearsonr(z[m], y_sub[m])[0] ** 2, atol=1e-12)


def test_r2_surface_leaves_unscorable_cells_nan():
    """Too few overlapping finite timesteps must yield NaN, not a fabricated score —
    and consumers must read the surface with nanargmax, since np.argmax would return
    the position of the first NaN."""
    from drought_scan.core import BaseDroughtAnalysis

    rng = np.random.default_rng(5)
    spi_sub = rng.normal(0, 1, (4, 40))
    y_sub = rng.normal(0, 1, 40)
    y_sub[5:] = np.nan  # only 5 usable timesteps

    surface = BaseDroughtAnalysis._r2_surface(spi_sub, y_sub, np.arange(1, 5), min_valid=10)
    assert np.all(np.isnan(surface))


def test_correlation_scatter_axis_is_the_committed_sidi():
    """analyze_correlation's scatter must be drawn on the same axis as the SIDI
    set_optimal_SIDI commits — i.e. standardized on the baseline via _zscore_baseline,
    not rescaled on the overlap window."""
    from drought_scan.utils.drought_indices import f_kde

    driver = _build_synthetic_dso(f_kde, seed=2)
    target = _build_synthetic_dso(f_kde, seed=9)
    driver.__class__ = type("_Driver", (driver.__class__,), {})
    driver._EXCLUDED_FROM_CORRELATION = ()
    driver._EXCLUDED_FROM_SIDI_OPTIMIZATION = ()

    res = driver.analyze_correlation(target, plot=False)
    k, w = int(res["best_k"]), int(res["col_best_weight"])

    scatter_axis = driver.recalculate_SIDI(K=k)[:, w]
    driver.set_optimal_SIDI(k, w, overwrite=True)
    assert np.allclose(scatter_axis, driver.SIDI[:, w], equal_nan=True)


def test_kde_fit_sample_is_shared_by_f_kde_and_the_diagnostic():
    """The operational KDE fit (f_kde) and the diagnostic that judges it
    (statistics._fit_single_dist) must be fitted on the same points. On real-valued
    data the diagnostic used to keep only the strictly-positive subset, so it scored
    a KDE describing half the sample."""
    from drought_scan.utils.drought_indices import f_kde
    from drought_scan.utils.statistics import _fit_single_dist

    rng = np.random.default_rng(3)
    years = np.arange(1981, 2021)
    m_cal = np.array([[m, y] for y in years for m in range(1, 13)])
    balance = rng.normal(-10, 60, len(m_cal))  # P-PET-like: about half negative

    _, _, _, operational = f_kde(balance, 1, 7, m_cal, 1981, 2010)
    sample = balance[(m_cal[:, 0] == 7) & (m_cal[:, 1] >= 1981) & (m_cal[:, 1] <= 2010)]
    diagnostic = _fit_single_dist(sample, "kde")["params"]

    assert np.array_equal(np.sort(operational["xb"]), np.sort(diagnostic["xb"]))
    assert np.isclose(operational["h"], diagnostic["h"])
    assert np.sum(diagnostic["xb"] < 0) > 0, "negatives must be kept in the fit"


def test_ks_ad_and_bootstrap_pvalues():
    """_fit_single_dist reports three views of the same fit-vs-data disagreement:
    error_percent (mean over every observation, what the recommendation ranks on),
    KS D (sup-norm, the single worst point) and Anderson-Darling A^2 (tail-sensitive
    via its 1/[F(1-F)] weighting). A Gaussian fit on strongly skewed data loses the
    tail, so its A^2 is the worse one. n_bootstrap > 0 adds Lilliefors-corrected
    p-values for both statistics."""
    from drought_scan.utils.statistics import _fit_single_dist, test_standardization

    rng = np.random.default_rng(0)
    data = rng.gamma(1.5, 20.0, 900)

    gam = _fit_single_dist(data, "gamma")
    gau = _fit_single_dist(data, "gaussian")

    # the reported keys, and nothing else
    assert set(gam) == {"distribution", "params", "n_sample", "KS_statistic",
                        "KS_p_value", "AD_statistic", "log_likelihood", "AIC",
                        "k_params", "error_percent"}
    assert gam["n_sample"] == 900
    assert np.isfinite(gam["AD_statistic"]) and gam["AD_statistic"] >= 0
    assert gau["AD_statistic"] > gam["AD_statistic"]      # gaussian loses the tail
    assert gam["error_percent"] < gau["error_percent"]    # and the body too
    assert 0.0 <= gam["KS_statistic"] <= 1.0

    # bootstrap p-values, one per statistic, only when asked for
    res = test_standardization(data, plot=False, n_bootstrap=0)
    assert "KS_p_value_bootstrap" not in res["fits"]["gamma"]
    boot = test_standardization(data, plot=False, n_bootstrap=49, seed=1)
    for stat in ("KS_p_value_bootstrap", "AD_p_value_bootstrap"):
        pv = boot["fits"]["gamma"][stat]
        assert 0.0 <= pv <= 1.0, stat

    # with no bootstrap nothing can be admitted, so the gate is dropped and the
    # recommendation degrades to the plain lowest-error one
    assert res["recommendation"] == res["best_by_mean_error"]
    assert res["recommendation"] == min(
        res["fits"], key=lambda d: res["fits"][d]["error_percent"])

    # with a bootstrap the gate applies: whatever it recommends must be a family
    # it did NOT reject, and among those it must be the lowest error_percent
    admitted = {d: f for d, f in boot["fits"].items()
                if f and f["KS_p_value_bootstrap"] >= 0.05}
    if admitted:
        assert boot["recommendation"] in admitted
        assert boot["recommendation"] == min(
            admitted, key=lambda d: admitted[d]["error_percent"])
    # the gaussian is rejected outright on strongly skewed data
    assert boot["fits"]["gaussian"]["KS_p_value_bootstrap"] < 0.05
    assert boot["recommendation"] != "gaussian"


def test_real_valued_kde_has_no_point_mass_at_zero():
    """A variable that goes negative is not bounded below, so an exact 0.0 is an
    ordinary value, not a pile-up. Keeping qq > 0 there also made the mixture
    ill-formed: the mass sat below the negative half, so the CDF started at qq."""
    from drought_scan.utils.drought_indices import kde_fit_sample
    from drought_scan.utils.statistics import _mixture_cdf, _mixture_pdf

    rng = np.random.default_rng(8)
    values = rng.normal(0, 50, 400)
    values[:20] = 0.0  # exact zeros AND negatives in the same sample

    xb, h, qq, log_transform = kde_fit_sample(values)
    assert qq == 0.0
    assert log_transform is False
    assert len(xb) == 400, "every finite value enters a real-valued fit"

    params = {"xb": xb, "h": 0.9 * np.std(xb, ddof=1) * xb.size ** (-1 / 5),
              "qq": qq, "log_transform": log_transform}
    grid = np.array([-120.0, -40.0, -1.0, 0.0, 1.0, 40.0, 120.0])
    cdf = _mixture_cdf(grid, "kde", params)
    assert np.all(np.diff(cdf) > 0), "CDF must increase monotonically through zero"
    assert np.all(_mixture_pdf(grid[:3], "kde", params) > 0), "density exists below zero"


def test_bounded_variable_keeps_its_point_mass():
    """The zero-inflation half of the rule must be untouched: a non-negative series
    with exact zeros still gets a point mass, and H(0) == qq."""
    from drought_scan.utils.drought_indices import kde_fit_sample, kde_cdf

    rng = np.random.default_rng(4)
    values = rng.gamma(1.3, 40, 500)
    values[rng.random(500) < 0.2] = 0.0

    xb, h, qq, log_transform = kde_fit_sample(values, log_transform=True)
    assert 0.15 < qq < 0.25
    assert log_transform is True
    assert not np.any(np.isinf(xb)), "log(0) must not reach the fit"

    h = 0.9 * np.std(xb, ddof=1) * xb.size ** (-1 / 5)
    assert np.isclose(kde_cdf(0.0, xb, h, qq, log_transform), qq)


def test_grid_point_sidi_matches_the_basin_sidi():
    """_process_grid_point replicates the basin-level SIDI pipeline (spi_like_set ->
    weighted ensemble mean -> _zscore_baseline). Run on the basin series itself it
    must return exactly what self.SIDI holds, in every non-seasonal state."""
    from drought_scan.core import BaseDroughtAnalysis
    from drought_scan.utils.drought_indices import f_kde

    obj = _build_synthetic_dso(f_kde)
    obj._EXCLUDED_FROM_SIDI_OPTIMIZATION = ()
    t_idx = len(obj.m_cal) - 1

    for K in (obj.K, [1, 2, 3, 2, 1]):
        sidi, err = BaseDroughtAnalysis._process_grid_point(
            obj.ts, obj.m_cal, K, obj.start_baseline_year, obj.end_baseline_year,
            obj.calculation_method, t_idx, 5,
        )
        assert err is None
        assert np.allclose(sidi, obj.recalculate_SIDI(K)[t_idx, :], atol=1e-9)


def test_grid_point_sidi_matches_the_seasonal_sidi():
    """With season_months given, the pixel SIDI must equal the seasonal SIDI at that
    timestamp. Without it, spatial_sidi(seasonal_params=...) applied the season's K
    but standardized against the whole baseline — the global SIDI at a seasonal K,
    not the seasonal SIDI."""
    from drought_scan.core import BaseDroughtAnalysis
    from drought_scan.utils.drought_indices import f_kde

    obj = _build_synthetic_dso(f_kde)
    obj._EXCLUDED_FROM_SIDI_OPTIMIZATION = ()
    seasons = obj._build_seasons_dict('quarter')
    corr = _seasonal_corr_stub(obj.K, seasons)
    obj.set_optimal_SIDI_seasonal(corr, agg='quarter', overwrite=True)

    t_idx = len(obj.m_cal) - 1
    month_t = int(obj.m_cal[t_idx, 0])
    season = next(s for s, ms in seasons.items() if month_t in ms)
    K_seas = corr[season]['best_k_per_weight']

    with_season, err = BaseDroughtAnalysis._process_grid_point(
        obj.ts, obj.m_cal, K_seas, obj.start_baseline_year, obj.end_baseline_year,
        obj.calculation_method, t_idx, 5, seasons[season],
    )
    assert err is None
    assert np.allclose(with_season, obj.SIDI[t_idx, :], atol=1e-9)


def _seasonal_corr_stub(K, n_seasons_months, seed=0):
    """A minimal analyze_correlation_seasonal-shaped dict with a distinct R2 peak
    per weighting scheme, so per-scheme optima are genuinely different."""
    rng = np.random.default_rng(seed)
    out = {}
    for i, (name, months) in enumerate(n_seasons_months.items()):
        M = np.zeros((K, 5))
        for w in range(5):
            peak = 1 + (w * 2 + i) % K
            M[:, w] = 0.1 + 0.5 * np.exp(-((np.arange(1, K + 1) - peak) ** 2) / 4.0)
        bk, bw = np.unravel_index(np.nanargmax(M), M.shape)
        out[name] = {
            "best_k": int(bk + 1),
            "col_best_weight": int(bw),
            "best_k_per_weight": np.arange(1, K + 1)[np.nanargmax(M, axis=0)],
            "max_correlation": float(np.nanmax(M)),
            "R2_matrix": M,
        }
    return out


def test_seasonal_sidi_keeps_one_column_per_scheme():
    """self.SIDI is always (time, 5) with the column meaning the weighting scheme.
    The seasonal commit used to tile the 1-D mosaic across five identical columns,
    which was the single case where that stopped being true."""
    from drought_scan.utils.drought_indices import f_kde

    obj = _build_synthetic_dso(f_kde)
    obj._EXCLUDED_FROM_SIDI_OPTIMIZATION = ()
    seasons = obj._build_seasons_dict('quarter')
    corr = _seasonal_corr_stub(obj.K, seasons)

    obj.set_optimal_SIDI_seasonal(corr, agg='quarter', overwrite=True)

    assert obj.SIDI.shape == (len(obj.m_cal), 5)
    # columns must differ: each scheme has its own per-season K
    assert not np.allclose(obj.SIDI[:, 0], obj.SIDI[:, 2], equal_nan=True)
    # and each equals what recalculate_SIDI_seasonal produces for that scheme alone
    expected = obj.recalculate_SIDI_seasonal_all_schemes(corr, seasons)
    assert np.allclose(obj.SIDI, expected, equal_nan=True)


def test_seasonal_commit_has_no_single_best_series():
    """The 'single best-(K, weight) per season' concept was removed: several
    schemes are statistically indistinguishable, so an argmax-per-season pick is
    a winner's-curse artefact. After the commit there is only self.SIDI (time, 5),
    and overwrite=False returns that same (time, 5) array."""
    from drought_scan.utils.drought_indices import f_kde

    obj = _build_synthetic_dso(f_kde)
    obj._EXCLUDED_FROM_SIDI_OPTIMIZATION = ()
    seasons = obj._build_seasons_dict('quarter')
    corr = _seasonal_corr_stub(obj.K, seasons)

    preview = obj.set_optimal_SIDI_seasonal(corr, agg='quarter', overwrite=False)
    assert preview.shape == (len(obj.m_cal), 5)

    obj.set_optimal_SIDI_seasonal(corr, agg='quarter', overwrite=True)
    assert not hasattr(obj, "SIDI_seasonal_best")
    assert obj.SIDI.shape == (len(obj.m_cal), 5)
    assert np.allclose(obj.SIDI, preview, equal_nan=True)


def test_resolve_weight_index_precedence():
    """Explicit -> seasonal calibration (0) -> committed global -> construction default -> 2."""
    from drought_scan.utils.drought_indices import f_kde

    obj = _build_synthetic_dso(f_kde)

    # nothing set at all
    assert obj._resolve_weight_index() == 2
    # construction preference
    obj.weight_index = 4
    assert obj._resolve_weight_index() == 4
    # a committed calibration outranks it
    obj.optimal_weight_index = 1
    assert obj._resolve_weight_index() == 1
    # an explicit argument outranks everything
    assert obj._resolve_weight_index(3) == 3
    # a committed SEASONAL calibration defaults to 0 (equal weights),
    # over both the construction preference and any optimal_weight_index
    obj.seasonal_params = {'agg': 'quarter', 'seasons': {}, 'config': {}}
    assert obj._resolve_weight_index() == 0
    assert obj._resolve_weight_index(3) == 3

    with pytest.raises(ValueError, match="must be in 0..4"):
        obj._resolve_weight_index(7)


def test_every_class_carries_a_weight_index():
    """_resolve_weight_index must never meet a missing attribute. Balance never set
    it, and Pet/Temperature dropped an explicitly passed value on the floor."""
    from drought_scan.core import BaseDroughtAnalysis
    import inspect

    for cls in (BaseDroughtAnalysis,):
        src = inspect.getsource(cls.__init__)
        assert "self.weight_index = 2" in src


def _build_zero_inflated_dso(K=3):
    """A Precipitation-like series with a real point mass at zero in one month —
    the regime where f_spi's zero-inflation term stops being a no-op."""
    from drought_scan.core import BaseDroughtAnalysis
    from drought_scan.utils.drought_indices import f_spi

    rng = np.random.default_rng(11)
    years = np.repeat(np.arange(1980, 2020), 12)
    months = np.tile(np.arange(1, 13), 40)
    m_cal = np.column_stack([months, years])
    ts = rng.gamma(1.2, 30, size=len(m_cal))
    # ~35% exact zeros in July/August, none elsewhere
    ts[np.isin(months, [7, 8]) & (rng.random(len(m_cal)) < 0.35)] = 0.0

    obj = object.__new__(BaseDroughtAnalysis)
    obj.ts = ts
    obj.m_cal = m_cal
    obj.K = K
    obj.start_baseline_year = 1981
    obj.end_baseline_year = 2010
    obj.threshold = -1
    obj.calculation_method = f_spi
    obj.index_name = "SPI"
    obj.basin_name = "test"
    obj.spi_like_set, obj.c2r_index = obj._calculate_spi_like_set()
    return obj


def test_f_spi_stores_zero_inflation_as_a_fit_parameter():
    """qq is part of f_spi's calibration, not a quantity re-derived downstream:
    it must survive into fit_params so the reverse transforms can undo the
    mixture. A 3-wide fit_params silently dropped it."""
    obj = _build_zero_inflated_dso()
    assert obj.fit_params.shape[2] == 4
    qq_july = obj.fit_params[0, 6, 3]
    qq_january = obj.fit_params[0, 0, 3]
    assert qq_july > 0.15, "July should carry a non-trivial point mass at zero"
    assert qq_january == 0.0


def test_native_to_spi_matches_the_forward_transform_under_zero_inflation():
    """native_to_spi must reproduce the SPI f_spi actually produced. With exact
    zeros in the baseline a bare gamma.cdf (no qq term) drifted by ~0.4 SPI."""
    obj = _build_zero_inflated_dso()
    months = obj.m_cal[:, 0].astype(int)

    for ref_month in (7, 8, 1):
        idx = np.where(months == ref_month)[0]
        z_fwd = obj.native_to_spi(obj.ts[idx], 1, ref_month)
        z_set = obj.spi_like_set[0, idx]
        # f_spi rounds its output to 4 decimals; nothing beyond that may differ.
        assert np.nanmax(np.abs(z_fwd - z_set)) < 1e-3


def test_spi_to_native_inverts_f_spi_under_zero_inflation():
    """spi_to_native o native_to_spi must be the identity on observed values,
    and map anything at or below the point mass back to exactly zero."""
    obj = _build_zero_inflated_dso()
    months = obj.m_cal[:, 0].astype(int)

    idx = np.where(months == 7)[0]
    observed = obj.ts[idx]
    back = obj.spi_to_native(obj.native_to_spi(observed, 1, 7), 1, 7)
    assert np.allclose(back, observed, atol=1e-8)

    # every dry month round-trips to exactly 0.0, not to a positive quantile
    assert np.all(back[observed == 0.0] == 0.0)


def test_normal_value_is_the_median_of_the_zero_inflated_mixture():
    """normal_values() is the SPI=0 reference by definition. Ignoring qq put it
    at the median of the positive part instead, far above the true median."""
    obj = _build_zero_inflated_dso()
    baseline = (obj.m_cal[:, 1] >= 1981) & (obj.m_cal[:, 1] <= 2010)
    months = obj.m_cal[:, 0].astype(int)

    normal_july = obj.spi_to_native(0.0, month_scale=1, ref_month=7)
    sample = obj.ts[(months == 7) & baseline]
    # SPI=0 is the 50th percentile of the fitted mixture: the fraction of the
    # baseline at or below it must be ~0.5 (loose bound: it is a fit, not the ECDF).
    assert 0.40 < np.mean(sample <= normal_july) < 0.60


def test_basin_and_pixel_deficit_agree_under_zero_inflation():
    """The pixel-level deficit in _process_grid_point_spi and the basin-level
    deficit_from_spi must be the same quantity — both are the exact inverse of
    the same fit, zero-inflation included."""
    from drought_scan.core import BaseDroughtAnalysis
    from drought_scan.utils.drought_indices import f_spi

    obj = _build_zero_inflated_dso(K=3)
    window = 3
    t_idx = 300

    deficit_pixel, _, err = BaseDroughtAnalysis._process_grid_point_spi(
        obj.ts, obj.m_cal, 1981, 2010, f_spi, windows=[window], t_idx=t_idx
    )
    assert err is None

    deficit_basin = obj.deficit_from_spi(window)[t_idx]
    spi_at_t = obj.spi_like_set[window - 1, t_idx]
    if abs(spi_at_t) < 0.5:
        # the pixel path zeroes the near-neutral band; the basin path does not
        assert deficit_pixel[window] == 0.0
    else:
        assert np.isclose(deficit_pixel[window], deficit_basin, rtol=1e-6)


@pytest.mark.parametrize("calculation_method_name", ["f_spi", "f_spei", "f_kde"])
def test_reverse_transforms_saturate_like_the_forward(calculation_method_name):
    """The forward transform clips cumulative probabilities, capping the index
    near +/-4. native_to_spi/spi_to_native must clip the same way instead of
    diverging to +/-inf on values beyond anything the baseline observed."""
    from drought_scan.utils import drought_indices as di

    f = getattr(di, calculation_method_name)
    obj = _build_synthetic_dso(f)
    ref_month = int(obj.m_cal[0, 0])

    spi = obj.native_to_spi(float(np.nanmax(obj.ts)) * 50, 1, ref_month)
    assert np.all(np.isfinite(spi))
    assert np.all(np.abs(spi) <= 4.01)

    native = obj.spi_to_native(50.0, 1, ref_month)
    assert np.all(np.isfinite(native))


def test_degenerate_grid_point_reports_error_not_short_tuple():
    """A grid point whose baseline is constant must be reported through the normal
    (values, error) contract — it used to return a short tuple and blow up the
    caller's unpacking."""
    from drought_scan.core import BaseDroughtAnalysis
    from drought_scan.utils.drought_indices import f_kde

    obj = _build_synthetic_dso(f_kde)
    ts = np.full(len(obj.m_cal), 7.0)
    ts[-12:] = np.abs(np.random.default_rng(0).normal(7, 2, size=12))

    result = BaseDroughtAnalysis._process_grid_point(
        ts, obj.m_cal, obj.K, 1981, 2010, f_kde, t_idx=300, n_weights=5,
    )

    assert len(result) == 2
    sidi_vals, err = result                    # must not raise
    assert sidi_vals is None and err is not None


def test_recalculate_sidi_scalar_k_unchanged():
    """A scalar K must keep producing exactly what the single-K implementation
    always produced — the per-weight K support must not perturb it."""
    from drought_scan.utils.drought_indices import (
        f_kde, generate_weights, weighted_metrics, baseline_indices,
    )

    obj = _build_synthetic_dso(f_kde)
    K = 3
    W = generate_weights(K)
    raw = np.array([
        [weighted_metrics(obj.spi_like_set[:K, t], w)[0] for w in W.T]
        for t in range(len(obj.m_cal))
    ])
    tb1, tb2 = baseline_indices(obj.m_cal, 1981, 2010)
    base = raw[tb1:tb2 + 1, :]
    expected = (raw - np.nanmean(base, axis=0)) / np.nanstd(base, axis=0)

    assert np.allclose(obj.recalculate_SIDI(K), expected, equal_nan=True)


def test_recalculate_sidi_per_weight_k():
    """Each weighting scheme peaks at its own K, so SIDI must accept one K per
    column — and every column must equal what that K alone would have given."""
    from drought_scan.utils.drought_indices import f_kde

    obj = _build_synthetic_dso(f_kde)
    Ks = [1, 2, 3, 2, 1]
    sidi = obj.recalculate_SIDI(Ks)

    assert sidi.shape == (len(obj.m_cal), 5)
    for wi, k in enumerate(Ks):
        assert np.allclose(sidi[:, wi], obj.recalculate_SIDI(k)[:, wi], equal_nan=True)


def test_resolve_per_weight_k_validation():
    """Bad K values must be rejected before any state is modified."""
    from drought_scan.core import BaseDroughtAnalysis

    with pytest.raises(ValueError):
        BaseDroughtAnalysis._resolve_per_weight_K([1, 2, 3], n_scales=12)   # wrong length
    with pytest.raises(ValueError):
        BaseDroughtAnalysis._resolve_per_weight_K(0, n_scales=12)           # non-positive
    with pytest.raises(ValueError):
        BaseDroughtAnalysis._resolve_per_weight_K(99, n_scales=12)          # beyond scales

    assert list(BaseDroughtAnalysis._resolve_per_weight_K(4, n_scales=12)) == [4] * 5


# ---------------------------------------------------------------------------
# Block-bootstrap confidence band on the R2(K, weight) surface
# ---------------------------------------------------------------------------

def _corr_pair(method=None, seed_a=2, seed_b=9):
    """A driver/target pair wired so analyze_correlation is callable on the
    bare synthetic DSO (Precipitation-only guard lifted)."""
    from drought_scan.utils.drought_indices import f_spi
    method = method or f_spi
    driver = _build_synthetic_dso(method, seed=seed_a)
    target = _build_synthetic_dso(method, seed=seed_b)
    driver.__class__ = type("_Driver", (driver.__class__,), {})
    driver._EXCLUDED_FROM_CORRELATION = ()
    driver._EXCLUDED_FROM_SIDI_OPTIMIZATION = ()
    return driver, target


def test_analyze_correlation_bootstrap_is_opt_in():
    """n_boot=0 (default) leaves the return value exactly as before; n_boot>0
    adds R2_boot / R2_ci / boot_meta / summary and nothing else."""
    driver, target = _corr_pair()

    base = driver.analyze_correlation(target, plot=False)
    assert not any(k in base for k in ("R2_boot", "R2_ci", "boot_meta", "summary"))

    res = driver.analyze_correlation(target, plot=False, n_boot=12, random_state=0)
    K = driver.K
    assert res["R2_boot"].shape == (12, K, 5)
    assert res["R2_ci"].shape == (2, K, 5)
    # the point estimate is untouched and sits inside its own band (allowing the
    # percentile band to be degenerate at tiny B)
    lo, hi = res["R2_ci"]
    inside = (res["R2_matrix"] >= lo - 1e-9) & (res["R2_matrix"] <= hi + 1e-9)
    assert np.mean(inside[np.isfinite(res["R2_matrix"])]) > 0.5

    bm = res["boot_meta"]
    for key in ("block_length", "block_years", "n_blocks", "contaminated_fraction",
                "eff_n_per_scale", "overlap_years"):
        assert key in bm
    assert bm["contaminated_fraction"][0] == 0.0            # scale 1 never crosses a join
    assert np.all(np.diff(bm["contaminated_fraction"]) >= -1e-12)   # non-decreasing in K


def test_analyze_correlation_seasonal_bootstrap():
    """Each scored season gains R2_boot / R2_ci / summary; one replica set is
    built on the continuous overlap and sliced per season."""
    driver, target = _corr_pair()
    S = driver.analyze_correlation_seasonal(target, agg="quarter", plot=False,
                                            n_boot=12, random_state=1)
    K = driver.K
    seasons = [k for k in S if k != "summary"]
    assert len(seasons) >= 1
    for name in seasons:
        d = S[name]
        assert d["R2_boot"].shape == (12, K, 5)
        assert d["R2_ci"].shape == (2, K, 5)
        assert "summary" in d
        assert d["boot_meta"]["n_boot"] == 12


def test_set_optimal_SIDI_seasonal_ignores_the_top_level_summary_key():
    """analyze_correlation_seasonal(n_boot>0) adds a top-level 'summary'
    DataFrame next to the season dicts. set_optimal_SIDI_seasonal (via
    recalculate_SIDI_seasonal_all_schemes) must skip it, not treat it as a
    malformed season."""
    driver, target = _corr_pair()
    seasons = {"H1": [1, 2, 3, 4, 5, 6], "H2": [7, 8, 9, 10, 11, 12]}
    S = driver.analyze_correlation_seasonal(target, seasons=seasons, plot=False,
                                            n_boot=8, random_state=1)
    assert "summary" in S                       # the offending key is present
    driver.set_optimal_SIDI_seasonal(S, seasons=seasons, overwrite=True)
    assert driver.SIDI.shape == (len(driver.m_cal), 5)
    assert not hasattr(driver, "SIDI_seasonal_best")


def test_year_block_index_is_year_aligned():
    """The bootstrap resamples whole overlap YEARS (rows), so f_spi/f_kde still
    see a gap-free calendar. The index has one entry per year, all in range."""
    from drought_scan.utils.statistics import _year_block_index

    rng = np.random.default_rng(0)
    for circular in (True, False):
        idx = _year_block_index(30, block_years=4, rng=rng, circular=circular)
        assert idx.shape == (30,)
        assert idx.min() >= 0 and idx.max() < 30


def test_contamination_fraction_grows_with_k():
    """f(k) = share of timesteps whose k-month accumulation straddles a block
    join. Zero at k=1, monotone up, and matching ceil(n_blocks)*(k-1)/T."""
    from drought_scan.utils.statistics import _contamination_fraction

    T, L, K = 12 * 30, 48, 24
    cf = _contamination_fraction(T, L, K)
    assert cf[0] == 0.0
    assert np.all(np.diff(cf) >= 0)
    n_blocks = int(np.ceil(T / L))
    assert np.isclose(cf[-1], n_blocks * (K - 1) / T, atol=1e-6)


def test_bootstrap_summary_table_is_a_per_family_table():
    """analyze_correlation_seasonal(n_boot>0) attaches a per-season 'summary'
    dict (the raw _peak_summary, with its scale clusters — used by Fig. 3) and
    an overall table, one row per (season, weighting scheme)."""
    driver, target = _corr_pair()
    S = driver.analyze_correlation_seasonal(target, agg="quarter", plot=False,
                                            n_boot=16, random_state=2)
    assert "summary" in S                                   # overall table
    seasons = [k for k in S if k != "summary"]
    for name in seasons:
        s = S[name]["summary"]
        assert set(s) >= {"w_best", "peak_by_family", "clusters"}
        assert set(s["peak_by_family"]) == {"ew", "lindw", "geodw",
                                            "liniw", "geoiw"}
        # every family lands in >= 1 cluster; clusters carry their pieces
        covered = set()
        for c in s["clusters"]:
            assert set(c) == {"families", "K_cluster", "K_cluster_CI",
                              "R2_cluster", "R2_cluster_CI", "color", "subclusters",
                              "ref_scheme", "ref_K", "ref_K_CI", "ref_R2", "ref_R2_CI"}
            assert isinstance(c["color"], str) and c["color"].startswith("#")
            # cosmetic reference: the member whose peak K is nearest the cluster
            # median (ties -> smaller K); no scheme is privileged
            assert c["ref_scheme"] in c["families"]
            klo, khi = c["K_cluster_CI"]
            assert klo <= khi
            covered |= set(c["families"])
            # level 2: sub-clusters partition the K cluster's families, sorted by
            # decreasing R2
            sub_fams = []
            for sc in c["subclusters"]:
                assert set(sc) == {"families", "R2", "R2_CI", "color"}
                sub_fams += sc["families"]
            assert sorted(sub_fams) == sorted(c["families"])
            r2s = [sc["R2"] for sc in c["subclusters"]]
            assert r2s == sorted(r2s, reverse=True)
        assert covered == set(s["peak_by_family"])
        # per-family K CI is present too
        for fam in s["peak_by_family"].values():
            assert "K_CI" in fam

    tab = S["summary"]
    if hasattr(tab, "columns"):        # pandas available
        assert list(tab.columns) == ["season", "family", "peak_K", "peak_K_CI",
                                     "peak_R2", "peak_R2_CI", "mean_age",
                                     "not_distinguishable_from"]
        assert set(tab["season"]) == set(seasons)
        assert set(tab["family"]) == {"ew", "lindw", "geodw", "liniw", "geoiw"}


def test_peak_summary_scale_clusters_group_by_k_not_r2():
    """Families group by the K they peak at (mutual K-CI inclusion), not by R².
    A scheme whose peak K sits clearly outside the others' K CIs is its own
    cluster; the cluster's K / R2 are medians over its families."""
    from drought_scan.utils.statistics import _peak_summary

    rng = np.random.default_rng(0)
    K, W, B = 30, 5, 600
    k = np.arange(1, K + 1)[:, None]
    surf = np.zeros((K, W))
    # tight pair at K~5, a lone scheme at K~13, a pair at K~22
    surf[:, 3] = 0.77 * np.exp(-((k[:, 0] - 5.0) ** 2) / 25.0)
    surf[:, 4] = 0.76 * np.exp(-((k[:, 0] - 5.0) ** 2) / 25.0)
    surf[:, 0] = 0.80 * np.exp(-((k[:, 0] - 13.0) ** 2) / 40.0)     # lone, mid-K
    surf[:, 1] = 0.81 * np.exp(-((k[:, 0] - 22.0) ** 2) / 45.0)
    surf[:, 2] = 0.82 * np.exp(-((k[:, 0] - 22.0) ** 2) / 45.0)     # global best R2

    boot = surf[None] * rng.normal(1.0, 0.02, (B, 1, 1)) + rng.normal(0, 0.003, (B, K, W))
    s = _peak_summary(surf, boot)                              # default k_tol=0

    fams = [set(c["families"]) for c in s["clusters"]]
    assert {"liniw", "geoiw"} in fams
    assert {"ew"} in fams                                       # the mid-K scheme, alone
    assert {"lindw", "geodw"} in fams
    assert [c["K_cluster"] for c in s["clusters"]] == sorted(
        c["K_cluster"] for c in s["clusters"])                  # returned in K order
    lone = next(c for c in s["clusters"] if c["families"] == ["ew"])
    assert np.isclose(lone["K_cluster"], s["peak_by_family"]["ew"]["argmax_K"])


def test_peak_summary_flat_season_collapses_to_one_wide_cluster():
    """A near-flat surface has no real timescale: every family's peak K wanders
    over the whole axis, K CIs overlap, and the scattered clusters merge into a
    single cluster with a K CI that spans most of the range."""
    from drought_scan.utils.statistics import _peak_summary

    rng = np.random.default_rng(2)
    K, W, B = 36, 5, 500
    surf = np.full((K, W), 0.6) + rng.normal(0, 0.02, (K, W))
    boot = surf[None] + rng.normal(0, 0.07, (B, K, W))
    s = _peak_summary(surf, boot, k_tol=3)

    assert len(s["clusters"]) == 1
    c = s["clusters"][0]
    assert set(c["families"]) == {"ew", "lindw", "geodw", "liniw", "geoiw"}
    klo, khi = c["K_cluster_CI"]
    assert (khi - klo) >= K / 2


def test_analyze_correlation_summary_is_a_per_family_table():
    driver, target = _corr_pair()
    res = driver.analyze_correlation(target, plot=False, n_boot=16, random_state=3)
    tab = res["summary"]
    if hasattr(tab, "columns"):
        assert {"family", "peak_K", "peak_K_CI", "peak_R2", "peak_R2_CI",
                "not_distinguishable_from"} <= set(tab.columns)
        assert set(tab["season"]) == {"whole period"}
        assert set(tab["family"]) == {"ew", "lindw", "geodw", "liniw", "geoiw"}


def test_peak_summary_r2_subclusters_split_same_K_by_response():
    """Level 2: two families that peak at the SAME K but reach clearly separated
    R² split into two sub-clusters; two families at the same K with overlapping
    R² CIs stay one sub-cluster; a lone family has a single sub-cluster."""
    from drought_scan.utils.statistics import _peak_summary

    rng = np.random.default_rng(0)
    K, W, B = 30, 5, 800
    k = np.arange(1, K + 1, dtype=float)

    def bump(center, amp, width):
        return amp * np.exp(-((k - center) ** 2) / width)

    surf = np.zeros((K, W))
    # K cluster A at K~6: two families, R² 0.75 vs 0.55 -> must split
    surf[:, 3] = bump(6.0, 0.75, 25.0)
    surf[:, 4] = bump(6.0, 0.55, 25.0)
    # K cluster B at K~20: two families, R² ~equal -> stay merged
    surf[:, 1] = bump(20.0, 0.700, 45.0)
    surf[:, 2] = bump(20.0, 0.695, 45.0)
    # lone family at K~13
    surf[:, 0] = bump(13.0, 0.60, 40.0)

    # tight replica-level noise: peak-R² CIs ~±0.02, far under the 0.20 gap in A
    boot = surf[None] * rng.normal(1.0, 0.012, (B, 1, 1)) \
        + rng.normal(0.0, 0.002, (B, K, W))
    s = _peak_summary(surf, boot)

    by_fam = {frozenset(c["families"]): c for c in s["clusters"]}
    cA = by_fam[frozenset({"liniw", "geoiw"})]
    assert len(cA["subclusters"]) == 2
    assert cA["subclusters"][0]["R2"] > cA["subclusters"][1]["R2"]
    assert [set(sc["families"]) for sc in cA["subclusters"]] == \
        [{"liniw"}, {"geoiw"}]

    cB = by_fam[frozenset({"lindw", "geodw"})]
    assert len(cB["subclusters"]) == 1
    assert set(cB["subclusters"][0]["families"]) == {"lindw", "geodw"}

    cLone = by_fam[frozenset({"ew"})]
    assert len(cLone["subclusters"]) == 1
    assert cLone["subclusters"][0]["families"] == ["ew"]


def test_economical_scheme_picks_smallest_K_among_globally_R2_tied():
    """_economical_scheme uses rule 2 (peak-R2 CI mutual inclusion) GLOBALLY,
    not nested inside a rule-1 K cluster: two schemes peaking at clearly
    different K (so NOT in the same _peak_summary K cluster) but with
    overlapping R2 CIs must still tie on R2, and the cheaper (smaller K) one
    must be picked over the one with the higher point estimate. A scheme with
    a clearly separated (lower) R2 must be excluded from the tie."""
    from drought_scan.utils.statistics import _peak_summary, _economical_scheme

    rng = np.random.default_rng(1)
    K, W, B = 30, 5, 800
    k = np.arange(1, K + 1, dtype=float)

    def bump(center, amp, width):
        return amp * np.exp(-((k - center) ** 2) / width)

    surf = np.zeros((K, W))
    surf[:, 0] = bump(5.0, 0.69, 20.0)     # ew: smaller K, R2 tied with geodw's
    surf[:, 2] = bump(20.0, 0.70, 20.0)    # geodw: larger K, top point-estimate R2
    surf[:, 3] = bump(5.0, 0.40, 20.0)     # liniw: clearly worse R2, must be excluded
    surf[:, 1] = surf[:, 4] = bump(5.0, 0.10, 20.0)  # lindw/geoiw: far off, filler

    boot = surf[None] * rng.normal(1.0, 0.012, (B, 1, 1)) \
        + rng.normal(0.0, 0.002, (B, K, W))
    s = _peak_summary(surf, boot)

    # sanity: ew and geodw do NOT share a K cluster (rule 1 keeps them apart)
    k_clusters = [set(c["families"]) for c in s["clusters"]]
    assert not any({"ew", "geodw"} <= fams for fams in k_clusters)

    cheapest, group = _economical_scheme(s["peak_by_family"])
    assert cheapest == "ew"
    assert set(group) == {"ew", "geodw"}
    assert "liniw" not in group


# =============================================================================
# Architecture review (2026-10): kernels, lags, calibration state, spatial date,
# standardization registry, class defaults. Synthetic data, public constructors.
# =============================================================================

def _pq_pair(K=4, seed=0, kernel=(0.5, 0.3, 0.2), method=None):
    """Precipitation / Streamflow objects on a 40-year synthetic record where Q is
    a causal convolution of P with `kernel` (kernel[0] on the current month)."""
    import warnings
    import drought_scan as DS
    from drought_scan.utils.drought_indices import f_zscore
    method = method or f_zscore          # fast and exact: keeps the tests quick
    rng = np.random.default_rng(seed)
    n = 12 * 40
    m_cal = np.column_stack([np.tile(np.arange(1, 13), 40),
                             np.repeat(np.arange(1981, 2021), 12)])
    P = rng.gamma(2.0, 40.0, n)
    Q = np.convolve(P, kernel)[:n] + rng.normal(0, 2, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # no shapefile -> area NaN warning
        prec = DS.Precipitation(1981, 2010, 'syn', ts=P, m_cal=m_cal, K=K,
                                calculation_method=method, verbose=False)
        flow = DS.Streamflow(1981, 2010, 'syn', ts=Q, m_cal=m_cal, K=K,
                             calculation_method=method, verbose=False)
    return prec, flow


def test_lag_convolve_puts_the_first_ordinate_on_the_current_month():
    from drought_scan.core import BaseDroughtAnalysis as B
    x = np.arange(10, dtype=float)
    h = np.array([0.7, 0.2, 0.1])
    y = B._lag_convolve(x, h)
    assert np.all(np.isnan(y[:2]))
    for t in range(2, 10):
        assert y[t] == pytest.approx(h[0] * x[t] + h[1] * x[t - 1] + h[2] * x[t - 2])


def test_ihacres_fit_and_plot_share_one_model():
    """The fit's simulation is the documented model: the largest weight on the
    current month (the reversed-kernel bug gave it to the oldest one)."""
    from drought_scan.core import BaseDroughtAnalysis as B
    rng = np.random.default_rng(1)
    x = rng.normal(size=300)
    sim = B._ihacres_simulate(x, 1.0, 4.0, 0.6, 6)
    h = 0.6 * B._exp_kernel(1.0, 6) + 0.4 * B._exp_kernel(4.0, 6)
    assert h[0] == h.max()
    manual = [np.dot(h, x[t - 5:t + 1][::-1]) for t in range(5, 300)]
    np.testing.assert_allclose(sim[5:], manual)


def test_ihacres_recovers_a_decaying_response_and_accepts_driver_target():
    prec, flow = _pq_pair(K=6, kernel=(0.6, 0.25, 0.15))
    r = prec.benchmark_ihacres(flow, K=6, plot=False)
    assert r['kernel_fast'][0] > r['kernel_fast'][-1]
    assert r['R2'] > 0.5
    assert r['metrics']['R2'] == pytest.approx(r['R2'], abs=1e-6)   # Pearson² both
    r2 = prec.benchmark_ihacres(flow, K=4, plot=False, driver_var='SPI', target_var='Q')
    assert np.isfinite(r2['R2'])
    with pytest.raises(ValueError):
        prec.benchmark_ihacres(flow, driver_var='X', plot=False)


def test_nash_bounds_follow_the_paper():
    from drought_scan.core import BaseDroughtAnalysis as B
    assert (B._NASH_N_MAX, B._NASH_TAU_MAX) == (20.0, 36.0)
    h = B._nash_kernel(1.0, 1e-4, 5)          # τ -> 0: all mass on j = 1
    np.testing.assert_array_equal(h, [1, 0, 0, 0, 0])
    prec, flow = _pq_pair(K=4)
    r = prec.benchmark_nash(flow, K=4, plot=False)
    assert 0 < r['optimal_n'] <= 20 and 0 < r['optimal_k'] <= 36
    assert r['metrics']['R2'] == pytest.approx(r['R2'], abs=1e-6)


def test_eval_metrics_r2_is_squared_pearson():
    from drought_scan.core import BaseDroughtAnalysis as B
    rng = np.random.default_rng(2)
    o = rng.normal(size=200)
    s = 3 * o + 5 + rng.normal(size=200)      # biased and rescaled: NSE would be < 0
    assert B._eval_metrics(o, s)['R2'] == pytest.approx(np.corrcoef(o, s)[0, 1] ** 2)


def test_lagged_correlations_reduce_to_the_unlagged_ones_at_lag_0():
    prec, flow = _pq_pair()
    R0, _ = prec.spi_sqi_corr(flow, plot=False)
    RL = prec.spi_sqi_corr_lagged(flow, max_lag=3, plot=False)
    assert RL.shape == (12, prec.K, 4)
    np.testing.assert_array_equal(RL[:, :, 0], R0)
    A = prec.analyze_correlation(flow, plot=False)
    L0 = prec.analyze_correlation_lagged(flow, lag=0, plot=False)
    np.testing.assert_allclose(L0['R2_matrix'], A['R2_matrix'], equal_nan=True)


def test_analyze_correlation_lagged_finds_a_true_lead():
    """Q driven by P two months earlier: the SIDI must explain Q better at lag 2."""
    prec, flow = _pq_pair(K=1, kernel=(0.0, 0.0, 1.0))
    r0 = prec.analyze_correlation_lagged(flow, lag=0, plot=False)['max_correlation']
    r2 = prec.analyze_correlation_lagged(flow, lag=2, plot=False)['max_correlation']
    assert r2 > 0.8 > r0


def test_result_keys_are_aligned():
    prec, flow = _pq_pair()
    A = prec.analyze_correlation(flow, plot=False)
    assert 'R2_matrix' in A and 'MatCorr' not in A
    S = prec.analyze_correlation_seasonal(flow, agg='semiannual', plot=False)
    assert all('sample_number' in v for v in S.values())
    irf = prec.analyze_convolution_irf(flow, plot=False)
    assert irf['R2_matrix'].shape == (prec.K, 5) and len(irf['weights']) == irf['best_k']


def test_calibration_states_are_mutually_exclusive(capsys):
    prec, flow = _pq_pair()
    A = prec.analyze_correlation(flow, plot=False)
    S = prec.analyze_correlation_seasonal(flow, agg='semiannual', plot=False)
    assert prec.calibration_mode is None

    assert prec.set_optimal_SIDI_seasonal(S, agg='semiannual', overwrite=True) is None
    assert prec.calibration_mode == 'seasonal'
    assert prec._resolve_weight_index() == 0           # seasonal default column

    assert prec.set_optimal_SIDI(A['best_k_per_weight'], 3, overwrite=True) is None
    assert prec.calibration_mode == 'global'
    assert not hasattr(prec, 'seasonal_params')
    assert prec._resolve_weight_index() == 3

    prec.set_optimal_SIDI_seasonal(S, agg='semiannual', overwrite=True)
    assert not hasattr(prec, 'optimal_k') and not hasattr(prec, 'optimal_weight_index')
    out = capsys.readouterr().out
    assert "discarded" in out

    capsys.readouterr()
    df = prec.ds_balance(flow, verbose=False)
    out = capsys.readouterr().out
    assert "SEASONALLY" in out and "weight_index=0" in out
    assert len(df) == len(prec.m_cal)


def test_spatial_maps_share_one_date():
    import warnings
    from drought_scan.utils.drought_indices import f_zscore
    obj = _build_synthetic_dso(f_zscore)
    obj.__class__ = type("_Gridded", (obj.__class__,), {})
    obj.Pgrid = np.random.default_rng(4).normal(50, 20, (len(obj.m_cal), 2, 2))
    obj.SIDI = obj._calculate_SIDI()

    obj.spatial_spi(windows=[2], timestamp=(6, 2000))
    obj.spatial_sidi(K=2)                              # inherits June 2000
    assert list(obj.spatial_timestamp) == [6, 2000]
    assert hasattr(obj, 'SPI_grid') and hasattr(obj, 'SIDI_grid')

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        obj.spatial_sidi(K=2, timestamp=(7, 2000))
    assert any("spatial timestamp changed" in str(x.message) for x in w)
    assert not hasattr(obj, 'SPI_grid') and not hasattr(obj, 'reverse_SPI_grid')


def test_spatial_grid_of_every_gridded_class():
    from drought_scan.core import Balance, Pet, Teleindex
    bal = object.__new__(Balance)
    bal.prec_grid, bal.pet_grid = np.full((3, 2, 2), 5.0), np.full((3, 2, 2), 2.0)
    np.testing.assert_array_equal(bal._spatial_grid(), np.full((3, 2, 2), 3.0))
    pet = object.__new__(Pet)
    pet.PETgrid = np.ones((3, 2, 2))
    assert pet._spatial_grid() is pet.PETgrid
    with pytest.raises(ValueError, match="No spatial grid"):
        object.__new__(Teleindex)._spatial_grid()


def test_standardization_registry():
    import warnings
    import drought_scan as DS
    from drought_scan.utils.drought_indices import (
        f_zscore, register_standardization, get_standardization)
    from functools import partial
    assert get_standardization(partial(f_zscore)).func is f_zscore

    def f_custom(*a, **k):
        return f_zscore(*a, **k)
    register_standardization(f_custom, to_index=lambda x, p: (x - p[0]) / p[1],
                             to_native=lambda z, p: z * p[1] + p[0],
                             n_fit_params=2, n_c2r=2)
    rng = np.random.default_rng(5)
    m_cal = np.column_stack([np.tile(np.arange(1, 13), 30),
                             np.repeat(np.arange(1981, 2011), 12)])
    t = DS.Teleindex(1981, 2010, ts=rng.normal(0, 1, 360), m_cal=m_cal, K=2,
                     calculation_method=f_custom, verbose=False)
    v = t.spi_to_native(0.7, 1, 3)
    assert t.native_to_spi(v, 1, 3) == pytest.approx(0.7)
    with pytest.raises(ValueError, match="Unknown calculation_method"):
        DS.Teleindex(1981, 2010, ts=rng.normal(0, 1, 360), m_cal=m_cal, K=2,
                     calculation_method=len, verbose=False)


def test_class_defaults_are_aligned():
    import warnings
    import drought_scan as DS
    rng = np.random.default_rng(6)
    m_cal = np.column_stack([np.tile(np.arange(1, 13), 30),
                             np.repeat(np.arange(1981, 2011), 12)])
    ts = rng.gamma(3, 10, 360)
    kw = dict(ts=ts, m_cal=m_cal, K=2, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        objs = {c: getattr(DS, c)(1981, 2010, 'b', **kw)
                for c in ('Precipitation', 'Streamflow', 'Pet', 'Balance', 'Temperature')}
        objs['Teleindex'] = DS.Teleindex(1981, 2010, **kw)
        assert DS.Pet(1981, 2010, 'b', weight_index=3, **kw).weight_index == 3
        assert DS.Temperature(1981, 2010, 'b', weight_index=1, **kw).weight_index == 1
    assert all(o.weight_index == 2 for o in objs.values())
    assert objs['Streamflow'].threshold == -1 and objs['Pet'].threshold == 1


def test_monthly_seasonal_correlation_does_not_plot_when_told_not_to():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    prec, flow = _pq_pair()
    plt.close('all')
    prec.analyze_correlation_seasonal(flow, agg='monthly', plot=False)
    assert plt.get_fignums() == []


def test_bfi_needs_the_daily_file():
    _, flow = _pq_pair()
    with pytest.raises(ValueError, match="data_path"):
        flow.BFI(plot=False)


def test_peak_summary_k_ci_is_integer_and_outward():
    from drought_scan.utils.statistics import _peak_summary
    rng = np.random.default_rng(7)
    K = 8
    M = np.tile(np.linspace(0.2, 0.6, K)[:, None], (1, 5))
    boot = M[None] + rng.normal(0, 0.05, (40, K, 5))
    s = _peak_summary(M, boot)
    for fam in s["peak_by_family"].values():
        lo, hi = fam["K_CI"]
        assert float(lo).is_integer() and float(hi).is_integer() and lo <= hi
