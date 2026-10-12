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
    # Fallback for a source tree that was never pip-installed; keep in step
    # with pyproject.toml's `version`.
    __version__ = "4.0.0"


# --- Public API (leggera) ----------------------------------------------------
# Import diretti delle classi core: devono essere leggeri e senza side-effect.
from .core import BaseDroughtAnalysis, Precipitation, Streamflow, Pet, Balance, Temperature, Teleindex

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

# --- Typing-only imports (evita costi a runtime) -----------------------------
if TYPE_CHECKING:
    from . import utils  # noqa: F401


# --- Lazy submodules ---------------------------------------------------------
def __getattr__(name: str):
    if name == "utils":
        from . import utils as _utils
        return _utils
    raise AttributeError(name)


def __dir__():
    return sorted(list(globals().keys()) + ["utils"])
