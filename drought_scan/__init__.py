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
    - utils        (submodule, lazy)
    - scenarios    (submodule, lazy)
"""

from typing import TYPE_CHECKING

# --- Version (PEP 621 / pyproject.toml) --------------------------------------
try:
    from importlib.metadata import version, PackageNotFoundError  # Py>=3.8
except Exception:  # pragma: no cover
    from importlib_metadata import version, PackageNotFoundError  # type: ignore

if version is not None:
    try:
        #  "project name" as that used in pyproject.toml 
        __version__ = version("droughtscan")
    except PackageNotFoundError:  # type: ignore
        __version__ = "3.0.4"
else:
    __version__ = "3.0.4"

# --- Public API (leggera) ----------------------------------------------------
# Import diretti delle classi core: devono essere leggeri e senza side-effect.
from .core import BaseDroughtAnalysis, Precipitation, Streamflow, Pet, Balance, Temperature,Teleindex

__all__ = [
    "BaseDroughtAnalysis",
    "Precipitation",
    "Streamflow",
    "Pet",
    "Balance",
    "Temperature",
    "Teleindex",
    "utils",        # lazy
    "__version__",
]

# --- Typing-only imports (avoid costs at runtime) -----------------------------
if TYPE_CHECKING:
    from . import utils


# --- Lazy submodules ---------------------------------------------------------
def __getattr__(name: str):
    if name == "utils":
        from . import utils as _utils
        return _utils
    raise AttributeError(name)


def __dir__():
    return sorted(list(globals().keys()) + ["utils"])
