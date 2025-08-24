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
    version = None  # type: ignore
    PackageNotFoundError = Exception  # type: ignore

if version is not None:
    try:
        #  "project name" as that used in pyproject.toml 
        __version__ = version("drought_scan")
    except PackageNotFoundError:  # type: ignore
        __version__ = "0.0.0"
else:
    __version__ = "0.0.0"

# --- Public API (leggera) ----------------------------------------------------
# Import diretti delle classi core: devono essere leggeri e senza side-effect.
from .core import BaseDroughtAnalysis, Precipitation, Streamflow, Pet, Balance, Teleindex

__all__ = [
    "BaseDroughtAnalysis",
    "Precipitation",
    "Streamflow",
    "Pet",
    "Balance",
   "Teleindex",
    "utils",        # lazy
    "__version__",
]

# --- Typing-only imports (evita costi a runtime) -----------------------------
if TYPE_CHECKING:
    from . import utils, scenarios  # noqa: F401


# --- Lazy submodules ---------------------------------------------------------
def __getattr__(name: str):
    if name == "utils":
        from . import utils as _utils
        return _utils
    if name == "scenarios":
        from . import scenarios as _scenarios
        return _scenarios
    raise AttributeError(name)


def __dir__():
    return sorted(list(globals().keys()) + ["utils", "scenarios"])
