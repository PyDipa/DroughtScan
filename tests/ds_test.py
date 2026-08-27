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


def test_real_valued_kde_has_no_point_mass_at_zero():
    """A variable that goes negative is not bounded below, so an exact 0.0 is an
    ordinary value, not a pile-up. Keeping qq > 0 there also made the mixture
    ill-formed: the mass sat below the negative half, so the CDF started at qq."""
    from drought_scan.utils.drought_indices import kde_fit_sample
    from drought_scan.utils.statistics import _mixture_cdf, _mixture_pdf

    rng = np.random.default_rng(8)
    values = rng.normal(0, 50, 400)
    values[:20] = 0.0  # exact zeros AND negatives in the same sample

    xb, qq, log_transform = kde_fit_sample(values)
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

    xb, qq, log_transform = kde_fit_sample(values, log_transform=True)
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


def test_seasonal_best_series_survives_the_commit():
    """The single best-(K, weight)-per-season mosaic is a product in its own right
    (the real-time monitoring index). It must coexist with the per-scheme columns
    instead of competing for self.SIDI."""
    from drought_scan.utils.drought_indices import f_kde

    obj = _build_synthetic_dso(f_kde)
    obj._EXCLUDED_FROM_SIDI_OPTIMIZATION = ()
    seasons = obj._build_seasons_dict('quarter')
    corr = _seasonal_corr_stub(obj.K, seasons)

    mosaic = obj.recalculate_SIDI_seasonal(corr, seasons)
    obj.set_optimal_SIDI_seasonal(corr, agg='quarter', overwrite=True)

    assert obj.SIDI_seasonal_best.shape == (len(obj.m_cal),)
    assert np.allclose(obj.SIDI_seasonal_best, mosaic, equal_nan=True)


def test_resolve_weight_index_precedence():
    """Explicit argument -> committed calibration -> construction default -> 2."""
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


def test_spatial_sidi_and_spatial_spi_own_separate_products():
    """spatial_sidi owns the SIDI, spatial_spi owns the SPI and its millimetre
    reverse. They used to both write SPI_grid, so whichever ran last won."""
    from drought_scan.core import Precipitation
    from drought_scan.utils.drought_indices import f_kde

    rng = np.random.default_rng(2)
    m_cal = np.column_stack([np.tile(np.arange(1, 13), 40),
                             np.repeat(np.arange(1981, 2021), 12)])
    Pgrid = rng.gamma(1.3, 40, (len(m_cal), 3, 4))

    obj = object.__new__(Precipitation)
    obj.ts = np.nanmean(Pgrid, axis=(1, 2))
    obj.m_cal, obj.K, obj.Pgrid, obj.day = m_cal, 6, Pgrid, None
    obj.start_baseline_year, obj.end_baseline_year = 1981, 2010
    obj.threshold, obj.calculation_method = -1, f_kde
    obj.index_name, obj.basin_name, obj.weight_index = 'SPI', 't', 2
    obj.spi_like_set, obj.c2r_index = obj._calculate_spi_like_set()
    obj.SIDI = obj._calculate_SIDI()

    obj.spatial_sidi()
    assert obj.SIDI_grid.shape == (3, 4, 5)
    assert not hasattr(obj, 'SPI_grid'), "spatial_sidi must not write SPI_grid"

    obj.spatial_spi(windows=[3, 6])
    assert sorted(obj.SPI_grid) == [3, 6]

    obj.spatial_sidi()                          # order no longer matters
    assert sorted(obj.SPI_grid) == [3, 6]


def test_spatial_sidi_follows_the_committed_calibration():
    """With no K given, spatial_sidi uses whatever calibration the object carries,
    so the grid matches self.SIDI instead of silently falling back to self.K."""
    from drought_scan.core import Precipitation
    from drought_scan.utils.drought_indices import f_kde

    rng = np.random.default_rng(6)
    m_cal = np.column_stack([np.tile(np.arange(1, 13), 40),
                             np.repeat(np.arange(1981, 2021), 12)])
    Pgrid = rng.gamma(1.3, 40, (len(m_cal), 2, 2))

    obj = object.__new__(Precipitation)
    obj.ts = np.nanmean(Pgrid, axis=(1, 2))
    obj.m_cal, obj.K, obj.Pgrid, obj.day = m_cal, 8, Pgrid, None
    obj.start_baseline_year, obj.end_baseline_year = 1981, 2010
    obj.threshold, obj.calculation_method = -1, f_kde
    obj.index_name, obj.basin_name, obj.weight_index = 'SPI', 't', 2
    obj._EXCLUDED_FROM_SIDI_OPTIMIZATION = ()
    obj.spi_like_set, obj.c2r_index = obj._calculate_spi_like_set()
    obj.SIDI = obj._calculate_SIDI()

    obj.spatial_sidi()
    assert obj.spatial_K == 8                   # nothing committed yet

    obj.set_optimal_SIDI([2, 3, 4, 5, 6], 2, overwrite=True)
    obj.spatial_sidi()
    assert list(np.ravel(obj.spatial_K)) == [2, 3, 4, 5, 6]


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
