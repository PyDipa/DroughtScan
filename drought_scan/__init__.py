"""
drought_scan
============

Open-source toolkit for drought monitoring and seasonal prediction.

Public API:
    - BaseDroughtAnalysis
    - Precipitation
    - Streamflow
    - Pet
    - Balance
    - Temperature
    - Teleindex
    - utils        (submodule, lazy)
"""

from typing import TYPE_CHECKING

# --- Version (PEP 621 / pyproject.toml) --------------------------------------
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("droughtscan")
except PackageNotFoundError:
    __version__ = "3.2.1"

# --- Public API ---------------------------------------------------------------
from .core import (
    BaseDroughtAnalysis,
    Precipitation,
    Streamflow,
    Pet,
    Balance,
    Temperature,
    Teleindex,
)

__all__ = [
    "BaseDroughtAnalysis",
    "Precipitation",
    "Streamflow",
    "Pet",
    "Balance",
    "Temperature",
    "Teleindex",
    "utils",
    "__version__",
]

# --- Typing-only imports (avoid costs at runtime) -----------------------------
if TYPE_CHECKING:
    from . import utils

# --- Lazy submodules ----------------------------------------------------------
def __getattr__(name: str):
    if name == "utils":
        from . import utils as _utils
        return _utils
    raise AttributeError(name)

def __dir__():
    return sorted(list(globals().keys()) + ["utils"])
