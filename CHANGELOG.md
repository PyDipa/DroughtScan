## [3.6.2] - Unreleased

### Added
- `spatial_maps`: new `seasonal_params` argument. When passed the output of
  `set_optimal_SIDI_seasonal`'s `self.seasonal_params`, `K` is automatically
  resolved to the season-specific `best_k` matching the requested `timestamp`,
  and a warning reports the corresponding `weight_index` to use for consistent
  plotting. This allows the spatial SIDI grid to reflect a seasonal optimization
  that a single global `K` cannot represent.
- `spatial_maps`: emits a warning when `K` is left unspecified and `self.optimal_k`
  is present on the instance (i.e. `set_optimal_SIDI` was called), signalling that
  the spatial grid is being computed with `self.K` rather than the optimized value,
  and pointing to `K=self.optimal_k` as the fix.

### Fixed
- `spatial_trends` (`_process_grid_point_trends`): per-window deficit/surplus at
  `t_idx` (feeding `trend_grid`, plotted via `plot_spatial(var='CDN', ...)`) is now
  computed via the exact inverse of the fitted distribution (`norm.cdf(spi) ->
  distribution.ppf`, same approach as `spi_to_native`), instead of the degree-3
  polynomial (`c2r`) approximation. The polynomial fit extrapolates poorly in the
  distribution tails, which was flattening/understating deficit magnitude at pixels
  with strongly negative SPI at the requested `month_scale`. Dispatches to
  `gamma.ppf` for `f_spi`, `pearson3.ppf` for `f_spei`, and the linear transform
  for `f_zscore`; `f_kde` raises `NotImplementedError` (consistent with
  `spi_to_native`). The `0.0` clip for `|SPI| < 0.5` is unchanged.
- `native_to_spi` / `spi_to_native`: `f_spei` was incorrectly routed through
  `gamma.cdf`/`gamma.ppf` using its Pearson III fit parameters (`c, loc, scale`
  from `pearson3.fit`), mismatching distribution family and parameterization.
  Now correctly uses `pearson3.cdf`/`pearson3.ppf`. `f_spi` (gamma) and `f_zscore`
  were unaffected.

### Documentation
- `spatial_guide.md` §2.6 (new): "Using an optimized SIDI" — documents the three
  supported workflows for aligning `spatial_maps` with a point-scale SIDI
  optimization: explicit manual `K`, `set_optimal_SIDI` (with the new warning),
  and `set_optimal_SIDI_seasonal` (via `seasonal_params`).
- `spatial_guide.md` §3.1: updated to describe `spatial_trends`'s deficit/surplus
  computation via the reverse-gamma transform (`deficit_from_spi`-consistent),
  replacing the previous description based on the linear `std_to_mm`
  approximation and rolling-trend significance gating.

# Changelog
## [3.6.1] - 2026 - 06 - 24

- 
### Added
- `fit_params`: new attribute storing the fitted-distribution parameters
  per (scale, reference_month), populated during calibration in
  `_compute_spi`. Shape is (K, 12, 3) for gamma-based methods
  (f_spi, f_spei) and (K, 12, 2) for f_zscore. Auto-expanded to larger
  scales when `_compute_spi` is invoked on the fly (e.g. from
  `deficit_from_spi` or `plot_trends` for windows > K).
- `spi_to_native(spi_value, month_scale, ref_month)` and
  `native_to_spi(value, month_scale, ref_month)`: analytical SPI/native
  conversions via the fitted distribution stored in `fit_params`
  (gamma for f_spi/f_spei, mean+std for f_zscore). Numerically equivalent
  to the existing `c2r_index` polynomial approximation (max error ≈ 0.03%
  over SPI ∈ [-3, +3]), provided as a cleaner API for future use.
  `f_kde` not yet supported (raises NotImplementedError).
- `_rolling_phase_test(window, alpha, min_valid)` in statistics: rolling
  one-sample t-test on the 1-month SPI series to detect significant
  wetting/drying phases. Complements `find_trends` (slope-based on CDN)
  by testing the LEVEL of the anomaly on a non-integrated series.
  Returns 'phase' (+1/-1/0), 'mean', and 'p_value' aligned to the last
  month of each window. P-values are approximate (assumes serial
  independence of SPI1).

### Changed
- `native_to_spi` / `spi_to_native` report a clearer error when invoked with
  `calculation_method=f_kde`, directing users to `f_spi` / `f_spei`.
- `plot_trends`: deficit bars are now masked by SPI magnitude (|SPI_w| < 0.5)
  rather than by `find_trends` on CDN. The previous mask was statistically
  fragile because trend tests on cumulative series (CDN) behave like trend
  tests on random walks. The new mask operates on the standardized SPI
  series at the window scale, non-integrated, where magnitude thresholds
  carry direct climatological meaning. For rigorous level-based phase
  detection, use `_rolling_phase_test` instead.

### Fixed
- `f_kde` no longer raises `TypeError: ... not 'dict'` during DroughtScan/Balance
  initialization, nor on the `plot_trends(show_spi=True)` and deficit code paths.
  Its fitted parameters (a dict holding a live `gaussian_kde` object) are now kept
  in a dedicated `fit_params_kde` store instead of being forced into the numeric
  `fit_params` array.
- `spatial_trends`: il calcolo del deficit/surplus per pixel ora usa la
  trasformazione inversa gamma mese-specifica (`polyval` su `c2r`),
  in luogo dell'approssimazione lineare scalare `std_to_mm`. Il risultato
  è coerente con `deficit_from_spi` e con `plot_cdn_trends`.
- `spatial_trends`: aggiunta soglia `|SPI| < 0.5` per azzerare anomalie
  meteorologicamente trascurabili, allineata al comportamento di
  `plot_cdn_trends`.
- `spatial_trends`: invalidazione esplicita di `trend_grid` a ogni chiamata;
  rilanci con `windows` custom non lasciano più stato residuo.
- `spatial_trends`: aggiunto `plt.close('all')` prima di `Parallel` per
  evitare `RuntimeError: main thread is not in main loop` in sessioni
  interattive con matplotlib attivo.
- `spatial_maps`: corretto import mancante di `generate_weights` nello
  scope del metodo (era importata solo nel worker `_process_grid_point`).
- `spatial_maps`: corretta docstring che indicava erroneamente il calcolo
  dei trend CDN tra gli output del metodo.


### Deprecated (planned)
- `c2r_index`: polynomial approximation of the SPI inverse transform.
  Future releases will gradually migrate internal usage to the analytical
  methods `spi_to_native` and `native_to_spi` based on `fit_params`.
  `c2r_index` remains the default in this release with no breaking change.
  The polynomial approximation is numerically accurate (< 0.05% error
  throughout the [-3, +3] SPI range for both gamma-fit and z-score
  methods); the planned change is an architectural simplification, not
  an accuracy improvement.

### Documentation
- `visualization_guide.md` §5: rewritten "Plot trends" subsection
  reflecting the new behaviour (deficit bars in physical units, auto
  unit inference, returned dict). New subsection "Cumulative
  deficit/surplus in physical units" documenting `deficit_from_spi`
  and `volume_anomaly_rolling`.
- `user_guide.md` §7 (new): "Trends and deficit quantification on the
  CDN". Workflow-oriented introduction framing the two deficit methods
  as complementary perspectives (statistical rarity vs physical water
  balance). Subsequent sections renumbered (§7→§8 ... §11→§12).


## [3.5.0] - 2026 - 05 - 28

### Added
- `deficit_from_spi(window)`: new method estimating the cumulative water
  deficit/surplus over a moving window by mapping the SPI-like anomaly at the
  matching accumulation scale back to native units via `c2r_index`, relative to
  the SPI=0 reference (which equals `normal_values()`). Reflects the statistical
  rarity of the accumulated anomaly. SPI is taken from `spi_like_set` when
  `window <= K`, otherwise computed on the fly via `_compute_spi`. For
  `Streamflow`, the cumulative monthly-mean discharge is converted to total
  volume (m^3) via average seconds/month. Accepts optional pre-computed `spi`
  and `coeff` to avoid recomputation.
- `volume_anomaly_rolling(window)`: new method computing the observed cumulative
  water balance by direct summation of monthly `(obs - normal)` anomalies over
  the window. Direct, observation-based counterpart to `deficit_from_spi`; serves
  as a sanity check and complementary metric. Units: mm (`Precipitation`),
  total m^3 (`Streamflow`, each monthly anomaly weighted by its calendar-month
  seconds).

### Changed
- `plot_trends()`: bars now report the cumulative water deficit/surplus in
  physical units (mm for `Precipitation`, m^3 for `Streamflow`) computed via
  `deficit_from_spi`, instead of standardized CDN deltas rescaled by a mean
  `std_to_mm` factor. The unit is inferred automatically from the object type;
  bars are zeroed where `find_trends` detects no significant monotonic trend.
- `plot_trends()`: `show_spi=True` now overlays the SPI-like series at any window
  scale, including scales exceeding `K` (computed on the fly). Previously limited
  to `window <= K`.
- `plot_trends()`: now returns a dict (`Changes`) mapping each window to its array
  of deficit values (previously returned `None`).

### Deprecated
- `plot_trends()`: the `unit` argument is retained for backward compatibility but
  no longer used; the unit is inferred from the object type.


## [3.4.0] - 2026 - 05 - 19

### Fixed
- `plot_sptatial()`:fixed custum cmpa option
- `plot_trends()`:fixed custum figsize option
- `benchmark_convolution()`: `optimal_K` in the returned dict was based on
  `np.nanargmax(MatCorr)` while the diagnostic plot used `_monotonic_plateau`,
  causing a silent inconsistency between plotted K and downstream
  `beta` / `condition_number` / `metrics`. Both sites now use the plateau
  criterion. `MatCorr` grid unchanged; only the selection index changes.

### Changed
- `benchmark_convolution()`: `_monotonic_plateau` defined once at method top,
  removed duplicate inline definitions in plot blocks.
- `Dspi_free()`: `optimal_K` now plateau-based (was argmax), consistent with
  `benchmark_convolution()`. Stabilises `beta` against multicollinearity
  at large K.

### Added
- `Dspi_free()`: seasonal split API (`agg`, `seasons` arguments), aligned
  with `benchmark_convolution()`, `benchmark_nash()` and `benchmark_ihacres()`.
  No-split behaviour preserved (`agg=None` default).
  Seasonal mode returns a dict-of-dicts keyed by season name with an
  additional `sample_number` field.


## [3.3.0]  

### Added
- Benchmarking suite for D(SPI)/SIDI predictive skill evaluation:
  - `benchmark_convolution()`: OLS linear convolution (Benchmark A: raw P; Benchmark B: SPI_1 anomalies), with optional seasonal split
  - `Dspi_free()`: OLS regression on multi-scale SPI predictors [SPI_1..SPI_K]
  - `benchmark_nash()`: parametric Nash IUH (Gamma kernel, params n and k) via Nelder-Mead
  - `benchmark_ihacres()`: two-component IHACRES routing (fast + slow exponential kernels) via L-BFGS-B
  - `_eval_metrics()`: shared helper returning R², RMSE, MAE, KGE, bias, POD and FAR on drought events (threshold SPI ≤ −1)
  
fix(mask): restore (reinstate) centroid-based nearest-pixel fallback in create_mask():
when regionmask returns all-NaN (shape smaller than grid resolution),
the mask defaults to the closest grid cell to the shape centroid.
 
improve the colors in plot_covariate, now it depends on the weight_index

- plot_covariates (method): resolves weight_index from self,
  handles seasonal SIDI (None sentinel, red color) and
  optimal_weight_index; raises TypeError if not optimized
- plot__covariates (func): accepts weight_index=None for
  seasonal case (_plot_index=0, _color='red'); replaces
  magic number 5 sentinel with None for semantic clarity
- add Figure 2: timeseries + square scatter panel with
  _color consistent across both figures"

## [3.2.2] - 2026 -04 - 09
refactor: unify fit_params interface across all calculation methods

### Refactored
- Rename gamma_params → fit_params throughout the codebase (f_spi,
  f_spei, f_kde, _compute_spi, _calculate_spi_like_set, _get_fit_params)
  to reflect the method-agnostic nature of the parameter container
- _compute_spi now acts as a transparent dispatcher: type interpretation
  of fit_params is fully delegated to each calculation method
- Add dispatcher comment in _compute_spi else-branch documenting
  the delegation pattern for type interpretation

### Added
- fit_params support to f_zscore, aligning its interface with f_spi,
  f_spei and f_kde (4-tuple return: indices, values, coeff, params)
- `_reapply_optimization()` method in `BaseDroughtAnalysis`: encapsulates
  SIDI re-optimization after `spi_like_set` recalculation (seasonal, global,
  or no-op). Used internally by ESM scenarios.
- User Guide Section 8.1.2: documents the distinction between global
  (`optimal_k` / `optimal_weight_index`) and seasonal (`is_seasonal_sidi`)
  SIDI optimization states.

### Fixed
- Silent failure in _compute_spi when f_zscore was used with precomputed
  fit_params (was falling into f_spi branch, wrong unpacking)
- `plot_covariates()`: threshold was not correctly defined when called
  without prior optimization.
- Minor bug fixes.

### Docs
- Update docstrings: _compute_spi, _calculate_spi_like_set,
  _get_fit_params, f_zscore

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

