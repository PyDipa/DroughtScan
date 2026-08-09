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


def test_severe_events_detection():
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
    result = ds.severe_events(max_events=3,plot=False)
    ds.plot_scan()

    assert result is not None
    # severe_events returns tuples of indices → at least one
    assert len(result[0]) > 0


def _build_synthetic_dso(calculation_method, seed=3):
    """Minimal BaseDroughtAnalysis instance from synthetic data, bypassing
    __init__ (which requires a shapefile for _area()). Shared by the
    native_to_spi/spi_to_native roundtrip tests below."""
    from drought_scan.core import BaseDroughtAnalysis

    rng = np.random.default_rng(seed)
    years = np.repeat(np.arange(1980, 2020), 12)
    months = np.tile(np.arange(1, 13), 40)
    m_cal = np.column_stack([months, years])
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
