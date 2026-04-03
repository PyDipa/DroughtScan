# Changelog
## [3.2.2] - ....in progress
-add sec 8.1.2
- bug fix in defining threshold within plot_covariates()
- New method _reapply_optimization() encapsulates the logic to restore
  SIDI optimization state after spi_like_set recalculation
- Handles three cases: seasonal (is_seasonal_sidi), global (optimal_k),
  and no optimization (no-op)
- Replaces scattered hasattr/optimal_k/optimal_weight_index checks
  in ESM scenarios and any future code that recalculates indices

## [3.2.1] - 2026-04-02
- Minor bug fixes (various).

## [3.2.0] - 2026-04-01

### Added
- `spatial_trends()` method: pixel-wise CDN trend maps with mm-equivalent conversion.
- `plot_spatial(var='CDN')`: visualize trend maps from `spatial_trends()`.
- `plot_boundary()`: basin shapefile visualization on equal-area projection.
- `export_scan_plot_csv()`: export data for replicating `plot_scan` in external tools.
- `analyze_correlation_seasonal()`: season-specific SIDI–SQI₁ optimization.
- `set_optimal_SIDI_seasonal()`: apply seasonal optimization to SIDI.
- `BFI()` method on `Streamflow`: Baseflow Index computation (daily data).
- `gui` optional dependency group (`pip install droughtscan[gui]`) for PyQt5.
- `joblib` added to dependencies (required by spatial methods).
- Zero-inflation mask in `f_kde` (mixed distribution for zeros, matching `f_spi`).
- Rolling daily aggregation option (`rolling=True`) in data I/O functions.
- Length-mismatch guard and NaN fallback in reverse polyfit (all standardization functions).
- fill missing monthly timestamps in `_coerce_to_monthly()`

When input data is already at monthly resolution, missing months are now
detected and filled with NaN (matching the existing behavior for daily data).
A visible warning prints which months are missing.
- Python 3.12 compatibility confirmed.

### Changed
- **`analyze_correlation()`** and **`analyze_correlation_seasonal()`** moved from
  `Precipitation`/`Pet`/`Balance` to `BaseDroughtAnalysis` (single implementation).
  Now callable on any meteorological driver; guarded against use on `Streamflow`/`Temperature`/`Teleindex`.
- **`spi_sqi_corr()`** unified: replaces `speti_sqi_corr` (Pet) and `spei_sqi_corr` (Balance).
  Single method name in `BaseDroughtAnalysis` with dynamic labels.
- **`f_kde`** is now the default `calculation_method` for all classes (documented;
  was already the code default since v3.1.0 but docs said `f_spi`/`f_zscore`).
- `drought_indices.py` refactored: baseline/whole-period logic extracted into
  `_resolve_indices()` helper; reverse polyfit extracted into `_reverse_polyfit()`.
  ~236 lines removed (759 → 523).
- Wildcard imports in `core.py` replaced with explicit named imports.
- Matplotlib removed from `data_io.py` module-level imports (moved to local import
  inside `if verbose` block).
- `numpy` dependency floor raised from `>=1.2` to `>=1.24`.

### Fixed
- **`export_scan_plot_csv()`**: `DSO.m_cal` and `DSO.area_kmq` → `self.m_cal` and
  `self.area_kmq` (was `NameError` on every call).
- **`plot_scan(saveplot=True)`**: called non-existent `self._saveplot()` →
  corrected to `self._savedsplot()`.
- **`f_spi` polyfit handler**: bare `coef` expression in except block replaced with
  `np.full(4, np.nan)` + `warnings.warn` (was silent data corruption if polyfit failed).
- `f_spi` warning message said "SPEI" instead of "SPI".
- `utils/__init__.py`: missing comma between `"plot_cdf_comparison"` and `"savefig"` in
  `__all__` (Python silently concatenated them); added lazy imports for
  `fit_distribution_stats`, `standardize_data`, `plot_cdf_comparison`.
- Dead code removed: `tb1_id`/`tb2_id` in `f_spi`/`f_spei`/`f_kde` (computed but unused),
  `os.environ['USE_PYGEOS']` (no-op since geopandas 1.0), `ListedColormap` import.

### Documentation
- `statistics_tools.md`: rewritten to match current API (4-family comparison,
  `standardize_data`, `plot_cdf_comparison`, `find_overlap` with examples).
- `user_guide.md`: fixed all default `calculation_method` references, fixed
  `gap_filling` example (was passing wrong arguments), added `analyze_correlation_seasonal`
  section, added Qt5Agg setup note, added links to all guides.
- `visualization_guide.md`: added sections for `plot_boundary`, `plot_covariates`,
  `plot_annual_ts`, `BFI`, `export_scan_plot_csv`, spatial cross-reference.
- `spatial_guide.md`: fixed method name `compute_spatial_sidi` → `spatial_maps`,
  added `spatial_trends` section, updated `plot_spatial` for CDN support.
- `common_errors.md`: fixed default from `f_spi` to `f_kde`, fixed import typo
  `z_score` → `f_zscore`, added link to `statistics_tools.md`.
- All docstrings in `core.py` fixed: `Temperature.__init__` (was "Pet class"),
  `Teleindex.__init__` (was "Precipitation class"), `_calculate_CDN` (was SIDI docstring),
  `calculation_method` defaults corrected to `f_kde`, duplicate parameter docs removed.
- Welcome messages: fixed private method references, wrong variable names.

### Breaking Changes
- `Pet.speti_sqi_corr()` → use `Pet.spi_sqi_corr()`.
- `Balance.spei_sqi_corr()` → use `Balance.spi_sqi_corr()`.
- `Precipitation.analyze_correlation()` signature unchanged but method now lives
  in `BaseDroughtAnalysis` (transparent to callers unless using `hasattr` checks).

## [3.1.0] - 2026-02-13
### added
- KDE-based distribution fitting in `fit_distribution_stats`, including KS goodness-of-fit and AIC evaluation.
- New correlation tools:
  - `Precipitation.analyze_correlation_seasonal()`
  - `Pet.analyze_correlation()`
  - `Balance.analyze_correlation()`
- New month-wise SPIₖ–SQI₁ correlation tool (`spi_sqi_corr`) documented in `user_guide.md`.
- New advanced statistics documentation page: `statistics_tools.md`.

### Changed
- `f_kde` is now the default method to standardize data.
- Added optional log-transform support in `f_kde` (via `x_log = np.log(x)`).

### Fixed
- Minor bug fixes (various).
- Added warnings for missing timestamps in `import_netcdf_for_cumulative_variable()`.

### Documentation
- Added documentation describing KDE as an alternative method to standardize data.

# Changelog
## [3.0.0] - 2025-10-10
### Added
- **plot_spi_fit()** method in `BaseDroughtAnalysis` for visualizing SPI–raw value relationships.
- **plot_covariates()** in `Precipitation` to explore covariate correlations; user guide updated accordingly.
- **plot_annual_ts()** function for annual balance-style visualization starting from custom months.
- **_area()** method and `self.area_kmq` attribute to report basin area in km².
- **test_standardization** exposed in `utils.__init__.py` for quick validation.
- Flexible plot order and spacing in `plot_overview`:
  - Custom `plot_order` permutations ('H', 'S', 'C', e.g. 'SCH').
  - Automatic height ratio adjustment and better suptitle positioning.
- Added new English docstrings across IO utilities for better clarity.
- README, `user_guide.md`, `visualization_guide.md`, and JOSS paper updated.

### Changed
- **Precipitation.analyze_correlation()**: simplified and refactored with new signature  
  (`plot_mode={'all','seasonal','monthly'}`) replacing legacy `yellow` and `seasonal` arguments.
- **BaseDroughtAnalysis**: added `SIDI_name` for dynamic plot labeling of drought indices.
- **Data IO**: major refactor of streamflow ingestion pipeline  
  (`load_streamflow()` unifies CSV/Excel/TXT workflows with auto-detection of headers, date/value columns, and monthly resampling).
- **Hydrology**: moved `era_snowfall_to_mm()` to `hydrology.py`.
- **Improved labeling and aesthetics**:
  - `plot_cdn_trends()` now uses consistent x-axis labeling.
  - `_save_dsplot()` now supports `split_plot=True`.
- **Refactored core methods**:
  - `recalculate_SIDI(K)` simplified, redundant arguments removed, validation improved.
  - New `Precipitation.set_optimal_SIDI(optimal_k, optimal_weight_index, overwrite=False)` method.
  - `Streamflow.gap_filling()` rewritten using NumPy OLS (removed sklearn dependency).
- Improved data IO to automatically clip gridded data to the basin shape.

### Renamed
- **`export_for_r_plot()` → `export_scan_plot_csv()`** (breaking name change).

### Fixed
- Title bug in `plot_severe_events()` fixed.
- Minor alignment issues in `plot_overview` and SPI correlation plots.

### Breaking Changes
- Refactored method signatures in `recalculate_SIDI`, `Streamflow.gap_filling`, and `Precipitation.set_optimal_SIDI`.
- Renamed function `export_for_r_plot()` → `export_scan_plot_csv()`.
- Removed sklearn dependency from `Streamflow` gap-filling operations.

## [2.0.1] - 2025-09-18
### Fixed
- Visualization: x-axis limits now use `DSO.K` instead of the hard-coded `36` in `visualization.py`.
- Matplotlib calls adjusted to avoid blocking behavior in non-interactive environments.

### Docs
- README and `user_guide.md`: improved installation steps and troubleshooting to avoid common errors.
- `common_errors.md`: added two new sections —
  - **Inconsistent baseline years or timestamp gaps**  
  - **Too few data points for fitting distributions**

### Chore
- Updated `pyproject.toml` with project URLs and refined dependencies.


## [2.0.0] - 2025-09-03
### Added
- New `Temperature` class and public export in `__init__`.
- `plot_scan()` now supports drought highlighting bands via `utils.visualization.highlight_drought`.
- New SIDI-vs-SQI1 scatter plot, color-coded by month/season (auto-fallback from Crameri to a cyclic pyplot colormap).
- `Precipitation.analyze_correlation` improved: color mapping by 2 seasons or by month.
- `import_netcdf_for_cumuative_variable(cumulate=True)` optional parameter (defaults to cumulative behavior).
- Documentation: common errors and solutions; improved user guide; clarified class differences in `core.py`.

### Changed
- `plot_scan` argument renamed: `xlim` → `year_ext` (more explicit).
- `monthly_profile`: statistics computed only on baseline years; introduced `season_shift` to center winter months; behavior refined and documentation updated.
- `plot_trends()` labels translated to English; basin name added; y-ticks regularized.

### Removed
- `assign_streamflow_data` method from `Streamflow` and `Teleindex`.
- `reverse_color` argument in `plot_scan()` and in `visualization.plot_overview()`; color reversal is now determined automatically from `DSO.threshold`.

### Fixed
- Various minor bugs throughout; updated `user_guide.md` and `visualization_guide.md`.

### Breaking Changes
- API break from `xlim` to `year_ext` in `plot_scan`.
- Removal of `reverse_color` in `plot_scan` and dependent functions.
- Behavior change and signature changes in `monthly_profile` (removed `two_years`, introduced `season_shift`).
- Removal of `assign_streamflow_data` methods.

