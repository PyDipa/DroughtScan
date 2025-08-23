# drought_scan/utils/__init__.py
"""Subpackage `utils`: supporting functions for drought-scan (I/O, statistics, visualization, etc.)."""

from typing import TYPE_CHECKING


from .statistics import find_overlap
from .drought_indices import f_spi, f_spei, f_zscore
from .hydrology import Qcs2Qmm, Qmm2Qcs

__all__ = [
    "find_overlap",
    "f_spi",
    "f_spei",
    "f_zscore",
    "Qcs2Qmm",
    "Qmm2Qcs",
    "savefig",     
]

# to avoid to import matplotlib as runtime
if TYPE_CHECKING:
    from .visualization import savefig  #  F401

# --- Import lazy (matplotlib) ---
def __getattr__(name: str):
    if name == "savefig":
        from .visualization import savefig  
        return savefig
    raise AttributeError(name)

def __dir__():
    return sorted(list(globals().keys()) + ["savefig"])
