# Changelog

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
