# tests/test_precipitation.py

import pytest

def test_import():
    """Check that the package imports and has the expected classes"""
    import drought_scan as DS
    assert hasattr(DS, "Precipitation")
    assert hasattr(DS, "Streamflow")
    assert hasattr(DS, "Pet")
    assert hasattr(DS, "Balance")
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
