"""
author: PyDipa
# © 2025 Arianna Di Paola
# License: GNU General Public License v3.0 (GPLv3)

Function decorators for logging and performance monitoring.

This module provides decorators to:
- **Measure execution time** of functions.
- **Cache results** for optimized performance.
- **Handle exceptions gracefully**.
- **Enable debugging/logging** for selected operations.

Main decorators:
- `@requires_forecast_data`: Ensures forecast data are available before executing a method.

Used by: `core.py`, `statistics.py`.
"""


import warnings
from functools import wraps

def requires_forecast_data(method):
    """a simple function to check whether forecast data have been imported."""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self.DSO, 'forecast_ts') or self.DSO.forecast_ts is None:
            warnings.warn("Forecast data not loaded! Please run '_import_forecast()' on the Precipitation instance first.", UserWarning)
            return
        return method(self, *args, **kwargs)
    return wrapper
