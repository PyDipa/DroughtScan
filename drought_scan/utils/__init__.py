# drought_scan/utils/__init__.py
"""Subpackage `utils`: supporting functions for drought-scan (I/O, statistics, visualization, etc.)."""

from typing import TYPE_CHECKING

# --- Lightweight imports (no matplotlib dependency) ---------------------------
from .statistics import find_overlap
from .drought_indices import f_spi, f_spei, f_zscore, f_kde
from .hydrology import Qcs2Qmm, Qmm2Qcs
from .statistics import test_standardization

__all__ = [
    "find_overlap",
    "f_spi",
    "f_spei",
    "f_kde",
    "f_zscore",
    "Qcs2Qmm",
    "Qmm2Qcs",
    "test_standardization",
    "fit_distribution_stats",
    "standardize_data",
    "plot_cdf_comparison",
    "savefig",
]

# --- Typing-only imports (avoid matplotlib at runtime) ------------------------
if TYPE_CHECKING:
    from .statistics import fit_distribution_stats, standardize_data, plot_cdf_comparison
    from .visualization import savefig

# --- Lazy imports (deferred until first access) -------------------------------
_LAZY_STATISTICS = {"fit_distribution_stats", "standardize_data", "plot_cdf_comparison"}

def __getattr__(name: str):
    if name in _LAZY_STATISTICS:
        from . import statistics as _stats
        return getattr(_stats, name)
    if name == "savefig":
        from .visualization import savefig
        return savefig
    raise AttributeError(name)

def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_STATISTICS) + ["savefig"])