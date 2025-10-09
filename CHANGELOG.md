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

