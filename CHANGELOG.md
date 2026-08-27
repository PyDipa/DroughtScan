## [4.0.0] - Unreleased

Major version: the SIDI data model, the spatial API and the forward/reverse
transform pair all changed in ways that break existing code. Migration notes are
at the end of this section.

The [3.6.2] and [3.7.0] blocks below were never released; their contents ship
here.


### Added
- `analyze_correlation`: now also reports the optimum of **each** weighting scheme,
  not just the single best (K, weight) pair. The full R² surface over every
  (K, weight) combination was already being computed and plotted, but discarded;
  it is now returned. New keys: `best_k_per_weight` (5,), `max_correlation_per_weight`
  (5,) and `MatCorr` (K, 5). The per-scheme optima are printed alongside the global
  best.
- `recalculate_SIDI` / `set_optimal_SIDI` / `_spi_like_set_ensemble_mean`: `K` may
  now be a sequence with one value per weighting scheme, in addition to a single
  integer. Each weighting scheme peaks at a different K, so applying one scalar K
  left the four columns other than `optimal_weight_index` calibrated at a scale
  chosen for a different scheme — values with no interpretation, yet
  indistinguishable from valid ones. Passing `best_k_per_weight` gives every column
  its own optimum. A scalar keeps the previous behaviour, bit-for-bit.
- `spatial_sidi(seasonal_params=...)` now also accepts the **raw output of
  `analyze_correlation_seasonal()`** directly, not only the `set_optimal_SIDI_seasonal`
  wrapper (`self.seasonal_params`). Pass `agg=` (or `seasons=`) alongside it so months
  map to seasons; it falls back to a committed `self.seasonal_params['agg']`. This
  lets you map a seasonal calibration without committing it to `self.SIDI`. New
  `agg` / `seasons` parameters on `spatial_sidi`.

### Changed
- `spatial_maps()` is renamed **`spatial_sidi()`**, so the gridded-index pair reads
  `spatial_sidi()` (SIDI) / `spatial_spi()` (SPI). No behaviour change from the rename.
- `spatial_sidi` no longer computes SPI maps: it owns the SIDI grid, `spatial_spi`
  owns `SPI_grid` and `reverse_SPI_grid`. Both used to write `SPI_grid` — one capped
  at K, one uncapped — so whichever ran last won and a warning told you to mind the
  order. The `month_scales` argument is gone with it.
- `spatial_sidi` with no `K` now follows whatever calibration the object carries:
  `self.seasonal_params` if a seasonal SIDI was committed, else `self.optimal_k`,
  else `self.K`. It used to always fall back to `self.K` and merely warn that a
  committed `optimal_k` was being ignored, so the map silently showed a different
  index from `self.SIDI`.
- Removed the dead spatial code: `spatial_maps_old`, `_process_grid_point_trends_old`,
  the deprecated `spatial_trends()` wrapper and the deprecated `trend_grid`
  property/setter/deleter. Use `spatial_spi()` and `reverse_SPI_grid`.
- `plot_spi_fit(K, month)` filled the whole `(K, 31, 12)` calibration grid to draw a
  single curve — `K*12` `spi_to_native` calls, each of which rebuilds a 2000-point
  interpolation grid under `f_kde`. It now computes only the curve being plotted
  unless `return_data=True` (measured 8x faster at K=24).
- **`self.SIDI` now always means one thing: `(time, 5)`, column = weighting scheme.**
  It was that in four of the five SIDI states; the fifth —
  `set_optimal_SIDI_seasonal(overwrite=True)` — tiled the 1-D best-(K, weight)-per-season
  mosaic across five identical columns, so the column axis silently stopped meaning
  "scheme". That single exception is what made `weight_index` unanswerable downstream,
  and what forced `ds_balance_all_schemes` and `diagnostics.basin_response` to rebuild
  the four discarded schemes from the raw R² surface, each with its own copy of the
  loop. The commit now stores the per-scheme columns in `self.SIDI` and the 1-D mosaic
  alongside in **`self.SIDI_seasonal_best`** — the single absolute-best series is a
  product in its own right (the real-time monitoring index) and coexists rather than
  competing.
- New `recalculate_SIDI_seasonal_all_schemes(seasonal_corr, seasons)` → `(time, 5)`:
  each column that scheme's own seasonal series at its own per-season K. This is the
  loop that used to live outside the class in two places; verified bit-identical to it
  on the Po record. `recalculate_SIDI_seasonal` is unchanged and still returns the 1-D
  mosaic.
- `ds_balance_all_schemes` reduced from ~170 lines re-implementing the whole balance to
  a loop over the new shared `_ds_balance_core`, which `ds_balance` also calls.
  `seasonal_corr`/`seasons` are now optional: pass them to score without committing,
  omit them to read the calibration already on the object. Verified that
  `ds_balance(weight_index=w)` and `ds_balance_all_schemes()[label_w]` now agree
  exactly — they could not before, since `ds_balance` read the tiled mosaic while
  pairing it with the season's overall-best K.
- New `BaseDroughtAnalysis._resolve_weight_index()`: one answer to "which column of
  SIDI", used by `plot_scan`/`plot_overview`, `plot_covariates`, `_savedsplot` and
  `ds_balance`. Precedence: explicit argument → `optimal_weight_index` (a committed
  calibration) → `weight_index` (construction default) → 2. Previously each site
  resolved it differently — `plot_overview` and `severe_events` hardcoded 2 and ignored
  a committed calibration entirely, while `ds_balance` fell back to 0 — so one object
  could be plotted, exported and balanced on three different schemes.
  `export_scan_plot_csv` is deliberately left on its own explicit default.
- The R² surface over every (K, weighting scheme) pair now comes from one shared
  `BaseDroughtAnalysis._r2_surface`. `analyze_correlation`,
  `analyze_correlation_seasonal` and `analyze_correlation_seasonal_rolling` each
  carried their own copy of that loop — the last two byte-for-byte identical, the
  first differing only in lacking the sample-size guard. Verified identical to the
  previous implementation to 4e-16 on the Po record (same argmax, same
  `best_k_per_weight`).
- Those three copies each rescaled the weighted mean with
  `(sidi - nanmean(sidi)) / nanstd(sidi)` before correlating. That looked like the
  library's standardization but referenced the OVERLAP window, while the real one
  (`_zscore_baseline`, used by `_calculate_SIDI`/`recalculate_SIDI`) references the
  BASELINE. Since Pearson's r is invariant under positive affine rescaling, the step
  could not affect any result — so it is dropped rather than repaired in triplicate,
  leaving `_zscore_baseline` as the only place that decides what is standardized
  against what.
- `spatial_cdn` renamed to **`spatial_spi`**, and its per-pixel CDN pass removed.
  The method computed a full CDN series (cumulative sum of SPI-1 over the record)
  for every grid point and then discarded it — the caller only tested it for
  `None`. Mapping the CDN is not a product of this library; the spatial products
  are the SPI and its millimetre-equivalent reverse. Dropping the pass saves one
  12-month scale-1 fit per grid point (measured: 2.1 ms/pixel with `f_spi`,
  2.8 ms/pixel with `f_kde`, on a 360-month record). `spatial_cdn` was introduced
  earlier the same day and is removed outright rather than deprecated; the older
  `spatial_trends` stays as a deprecated wrapper.
- The near-neutral `|SPI| < 0.5` band is now a rendering convention on the map, as
  it already was on the time-series plot. `spatial_spi` writes the real millimetres
  into `reverse_SPI_grid` everywhere the SPI is defined, and `plot_spatial` greys
  the band out at draw time (new `neutral_band` argument, default 0.5, set to 0 to
  disable). Previously the zeroing happened inside the returned value, so the
  millimetres inside the band were unrecoverable from the stored grid — the same
  band on `plot_cdn_trends` only ever touched a local copy of the bars.
- `plot_spatial`: `var='CDN'` no longer accepted. It was an alias for
  `'reverse_spi'`, but the two are different quantities and the alias suggested
  the library produced a CDN map. It now raises with an explanation pointing at
  `'reverse_spi'` (millimetre-equivalent deficit/surplus) and `'SPI'`.
- Distribution-family selection (`test_standardization` / `_analyze_all` and the
  methodology report) is now decided by the **mean point-by-point CDF deviation**
  instead of the KS statistic D. D is the single worst point of disagreement, and
  a zero-inflated fit has a genuine vertical jump of height `qq` at x=0 that D
  latches onto: it returns ~`qq` however well the curve tracks the data elsewhere,
  handing the win to whichever family has no jump to be penalised for. On a
  synthetic gamma sample, at qq=0 the families score 0.165/0.047/0.043/0.029 and
  KDE correctly wins; at qq=0.31 gamma, pearson3 and KDE all collapse to exactly
  0.315 while the Gaussian "wins" with 0.221. Both numbers stay in the report
  table, and on samples without exact zeros (e.g. Po basin-average rainfall,
  qq=0) the two criteria agree. New key `best_by_mean_error`; `best_by_KS` is
  kept and `recommendation` now points at the new criterion.
- The SIDI weighting schemes built with `np.geomspace` were labelled
  "logarithmic" everywhere user-facing, including plot labels and the guides; they
  are geometric and are now named so (`gdw`/`giw` replace `lgdw`/`lgiw`).

### Fixed
- `spatial_sidi` passed `self.K` to the grid workers instead of the resolved `K`, so
  an explicit `K=` (or a seasonal/committed calibration) was accepted, reported in
  `spatial_K`, and then ignored by the actual computation — every pixel was built at
  `self.K`.
- `_process_grid_point_spi` took a `K` argument it never used.
- `spatial_sidi(seasonal_params=...)` produced the GLOBAL SIDI evaluated at a seasonal
  K, not the seasonal SIDI: it resolved the season's K but then standardized every
  pixel against the whole baseline, while `recalculate_SIDI_seasonal` standardizes
  per season. On the Po the two differed by up to 0.16 index units (JJA); the map was
  labelled seasonal and was not. `_process_grid_point` takes a `season_months`
  argument and `spatial_sidi` forwards the season's months alongside its K. Agreement
  with the basin-level seasonal SIDI is now ~1e-15.
- `spatial_sidi` assigned `self.K = K` for the duration of the computation and
  restored it at the end, so any exception in between (a failing grid point, an
  interrupt) left the object permanently carrying the spatial K — and a per-scheme
  sequence left `self.K` as a list. It no longer touches `self.K` at all.
- `ds_balance`: the weighted rain deficit used `np.nansum` over the k accumulation
  scales, treating a scale that is not yet defined as a zero contribution. Over the
  first k-1 months of the record this silently under-reported the deficit instead of
  reporting it as undefined; it is now a plain sum, so those months are NaN.
- `_compute_spi` left `c2rspi` unbound for a `calculation_method` outside the four
  known ones, failing with `NameError` instead of saying what was wrong. It now
  raises a `ValueError` naming the expected methods. (The unused `way` flag is gone.)
- `_get_kde_fit` reported "the SPI-like set must be computed first" when a fit had
  been attempted but no usable baseline value existed at that scale. The two cases
  now get different messages.
- `f_kde`'s no-valid-data early exit returned `None if fit_params is None else None`.
- `f_kde` and `statistics._fit_single_dist` disagreed about which points enter a KDE
  fit. With the log transform off (i.e. on real-valued data — the P-PET balance,
  temperature), the operational fit kept the negative half of the sample while the
  diagnostic kept only `dataset[dataset > 0]`. On a synthetic balance series the
  diagnostic fitted 11 points where `f_kde` fitted 30, so the "distributions" page
  reported how well a KDE described half the data. Both now call one shared
  `drought_indices.kde_fit_sample`, which decides the log transform, the
  zero-inflation fraction and the continuous-part sample in one place.
- The zero-inflation point mass was applied to real-valued fits, where it is not
  defined: a variable that goes negative has no floor for values to pile up on, and
  putting a mass `qq` at zero placed it *below* the negative half of the
  distribution — `kde_cdf` then returned `H(0) = qq` regardless of the kernels, so
  the CDF was non-monotone (`H(0) < H(-1)`). `qq` is now 0 whenever negatives are
  present, and `kde_cdf` pins `x == 0` to the point mass only when there is one.
  Behaviour on non-negative data (precipitation, streamflow) is unchanged.
- `statistics._mixture_pdf` clamped the KDE density to 0 for every `x <= 0`, which
  drew the left half of every real-valued fit as a flat line on the diagnostic PDF
  panels. It is now clamped only when a point mass at zero was actually fitted.
- The KDE log-likelihood summed the density over the strictly-positive points only;
  it now covers the points the continuous part actually models.
- `BaseDroughtAnalysis.__init__` now guarantees `self.weight_index` exists on every
  class. `Balance` never set it, and `Pet`/`Temperature` set it only when the argument
  was `None` — so `Pet(..., weight_index=3)` silently left the attribute undefined and
  `_savedsplot` raised `AttributeError` later. Both constructors now honour an
  explicitly passed value.
- `Streamflow.gap_filling` read `precipitation.SIDI[:, 0]` for a seasonally-optimised
  driver. Column 0 is equal weights, not the best scheme — it only looked right while
  all five columns were copies of the mosaic. It now uses `SIDI_seasonal_best`, the
  most predictive series, which is what a gap-filling regression wants.
- `severe_events` renamed to `severe_events_old` pending replacement, with its known
  defects recorded in the docstring (polynomial deficit inconsistent with
  `deficit_from_spi`; `-1` hardcoded instead of `threshold` when choosing run parity;
  drought condition inverted for a positive threshold, as on `Pet`/`Temperature`).
- `analyze_correlation` / `analyze_correlation_seasonal`: the scatter plots were drawn
  on a SIDI rescaled over the overlap window, not the SIDI the corresponding
  `set_optimal_SIDI` / `set_optimal_SIDI_seasonal` would commit. A point read off the
  figure did not correspond to the value the rest of the library reports for that
  month — on the Po record the two axes differed by up to 0.14 index units. Both now
  plot the committed series itself (`recalculate_SIDI` / `recalculate_SIDI_seasonal`).
  The seasonal one additionally standardized every panel over the whole overlap,
  mixing the seasons back together in the very step the seasonal SIDI exists to keep
  apart; it now shows the per-season-standardized mosaic.
- `analyze_correlation`: read its R² surface with `np.max`/`np.argmax`, which return
  the position of the first NaN when any cell is unscorable. Now `np.nanmax`/
  `np.nanargmax`, matching the seasonal variants, with an explicit error when no cell
  could be scored at all.
- `diagnostics.basin_response.compute_seasonal_sidi_calibration`: picked each
  weighting scheme's per-season optimal k with `np.argmax` on a surface that can
  contain NaN, so a single unscorable cell silently produced a wrong k. Now
  `np.nanargmax`, skipping columns with no scored cell.
- `f_spi` / `native_to_spi` / `spi_to_native`: the forward transform applied the
  zero-inflation mixture `Hx = qq + (1-qq)*Gamma.cdf(x)`, but the reverse ones used
  a bare `gamma.cdf`/`gamma.ppf` with no `qq` term, so the two directions were not
  inverse to each other on any month with exact zeros in the baseline. `qq` was
  never stored: `fit_params` carried only `(alpha, loc, beta)`. Measured on a
  July with 32% dry months: `native_to_spi(20 mm)` returned 0.09 where `f_spi`
  had produced 0.49, a 0 mm month round-tripped back to 8.4 mm, and the SPI=0
  "normal value" came out at 17.5 mm instead of 6.1 mm — an error that propagated
  into `normal_values`, `_monthly_normals`, `deficit_from_spi`,
  `volume_anomaly_rolling`, `monthly_profile`, `plot_cdn_trends`, `ds_balance`,
  `plot_spi_fit` and the pixel-level deficit in `spatial_cdn`. `qq` is now the
  fourth element of f_spi's fitted parameters, and both directions go through the
  new shared `gamma_cdf_zi`/`gamma_ppf_zi` (the counterpart of `kde_cdf`/`kde_ppf`
  for f_kde), which `statistics._mixture_cdf`/`_mixture_ppf` and
  `_process_grid_point_trends` also use — one definition instead of four.
  Round-trip error is now ~1e-12 native units. A legacy 3-tuple `fit_params` is
  still accepted, with a `RuntimeWarning` when it forces `qq` to be re-estimated.
- `f_spi`: when `fit_params` was supplied, `qq` was re-derived from the data being
  transformed rather than taken from the calibration — the same defect fixed in
  `f_kde` earlier on this branch. The stored `qq` is now honoured.
- `statistics`: `shift_for_gamma` defaulted to `True`, so every diagnostic fit
  (`test_standardization`, `fit_distribution_stats`, `plot_cdf_comparison` and the
  methodology report) scored a Gamma fitted on `x + 1` — a different model from the
  one `f_spi` applies (`gamma.fit(x[x > 0], floc=0)`, zeros carried by `qq`). The
  default is now `False`, matching `f_spi` exactly; the shift remains available as
  an opt-in. `_bootstrap_ks_pvalue` also refit the bootstrap replicates as shifted
  regardless of the fit under test, comparing `D*` against a `D_obs` from a
  different model; it now follows the fit's own `shift_applied`.
- `diagnostics.methodology`: the test that decides which `calculation_method` the
  whole pipeline uses ran on the full record, while `f_spi`/`f_kde` calibrate the
  fit, `qq` and the log decision on the baseline alone. It now runs on the baseline
  slice, and the report page plots that same slice and states the period. Widening
  the evidence is done by widening the baseline.
- `spatial_spi` (ex `spatial_cdn`): a pixel whose SPI was undefined for a window
  (record shorter than the window at that timestamp, or gaps in the series) was
  written to `reverse_SPI_grid` as `0.0`, i.e. as a pixel with no anomaly. NaN now
  stays NaN, and is visibly distinct from the greyed near-neutral band.
- `refit_gamma_shifted` (`data_io`): returned `(alpha, loc, beta, shift)`, absorbing
  zeros by adding a constant. With `qq` now f_spi's fourth parameter the two tuples
  had the same length, so a shift would have been read as a zero-inflation fraction.
  It now follows f_spi's convention — Gamma on the strictly-positive samples, exact
  zeros carried as `qq` — and raises on negative aggregates. The function has no
  callers in the library.
- `normal_values`: tiled the 12 monthly normals from position 0 and truncated to
  `len(ts)`, silently assuming the record starts in January — on a series starting
  in August every timestep received the wrong month's normal. Each timestep now
  takes the normal of its own calendar month. The 12-value climatology is available
  as `_monthly_normals`.
- `f_kde`: `fit_params` was accepted but ignored — `xb`, `h`, `qq` and
  `log_transform` were recomputed from the supplied data, so a stored calibration
  silently recalibrated on whatever it was applied to. It is the path the
  scenarios/forecast module drives, where the whole point is to keep modified series
  comparable to the original fit. Now honoured when the calibration keys are
  present; when they are not (the dict shape emitted by older versions) the previous
  behaviour is kept but an explicit `RuntimeWarning` states that the result is not in
  the reference frame of the fit passed in.
- `f_spi` / `f_kde`: the zero-inflation fraction `qq` was estimated over the whole
  record while the continuous part of the same mixture was fitted on the baseline
  alone. Both halves now come from the baseline.
- `f_kde`: the log-transform decision was taken from `min(xbase, x)`, so a single
  value anywhere in the record could flip an entire (scale, month) cell to a
  different model family — appending one dry month moved historical SPI values by up
  to 3.2. The decision now comes from the baseline alone; values outside the
  calibrated domain yield NaN with an explicit warning instead of silently switching
  model.
- CDN was computed two different ways under one name: the basin-level version
  cumulated SPI-1 from the start of the baseline, rounded to 3 decimals and zeroed
  the span before it, while the pixel-level version in `_process_grid_point_trends`
  cumulated from index 0 with no rounding and NaN before the baseline. Unified in
  `_cdn_from_spi1`.
- `deficit_from_spi`: converted Streamflow deficits to volume with a fixed average
  month length, while `volume_anomaly_rolling` — its documented counterpart — used
  each month's real length, diverging by up to 7.9%. It now averages the seconds of
  the months actually inside the window.
- `native_to_spi` / `spi_to_native`: did not clip cumulative probabilities, so the
  forward transform saturated near ±4 while the reverse returned ±inf beyond the
  calibrated range. Both now clip through the shared `PROB_CLIP` constant.
- `_process_grid_point`: carried a fourth hand-rolled copy of the baseline z-score
  with a weaker guard, and returned a 2-tuple on a degenerate baseline where the
  caller unpacks three values. It now uses the shared `_zscore_baseline`, whose
  error is caught and reported like any other grid-point failure.
- `Streamflow.gap_filling`: `Q_pred` was referenced outside the block defining it,
  crashing whenever a gap fell outside the precipitation overlap window.
- `_rolling_phase_test`: referenced an undefined `self` instead of its `DSO`
  parameter, failing on every call.
- `test_standardization`: missing `import warnings` broke the handler meant to
  warn-and-skip a failed fit, and the bootstrap RNG was the `Generator` class rather
  than an instance, so `seed` was ignored and bootstrap p-values never computed.
- `c2r_index` is `[mean, std]` for `calculation_method=f_zscore`, but every reverse
  conversion evaluated it with `np.polyval`, computing `mean*x + std` instead of
  `x*std + mean` — wrong everywhere except by coincidence at x=1.
- `f_spei`: now rounds to 4 decimals like `f_spi` and `f_kde`.

### Internal
- Exact numeric inverse for `f_kde` (`kde_cdf` / `kde_ppf`), so `native_to_spi`,
  `spi_to_native` and `spatial_trends` support it instead of raising
  `NotImplementedError`. All live (non-`_old`) reverse conversions now go through
  the exact inverse of the fitted distribution rather than the degree-3 polynomial
  approximation; `c2r_index` is still computed and stored for backward
  compatibility.
- Baseline z-score standardization deduplicated into `_zscore_baseline`
  (was reimplemented independently in four places, with inconsistent guards).
- Docstrings across `core.py` and `utils/` corrected against the implementation
  after a full audit (signatures, return values, defaults, described behaviour).

### Migration from 3.x

- `self.SIDI` after `set_optimal_SIDI_seasonal(..., overwrite=True)` is no longer
  five identical columns. Code reading `SIDI[:, 0]` to get "the" seasonal series
  must read `self.SIDI_seasonal_best` instead; code reading `SIDI[:, w]` now gets
  scheme *w*'s own seasonal series, which is the intended meaning.
- `spatial_cdn()` is removed — use `spatial_spi()`. `spatial_trends()` still works
  but is deprecated.
- `plot_spatial(var='CDN')` is removed — use `var='reverse_spi'`.
- `severe_events()` is renamed `severe_events_old()` pending replacement.
- `f_spi` returns four fitted parameters `(alpha, loc, beta, qq)` instead of three;
  `self.fit_params` is correspondingly `(K, 12, 4)` for `f_spi`. A stored 3-tuple is
  still accepted with a warning. `native_to_spi`/`spi_to_native` now invert the
  zero-inflation mixture, so native-unit results (`normal_values`,
  `deficit_from_spi`, `ds_balance`, …) change on any series with exact zeros.
- `test_standardization` and friends default to `shift_for_gamma=False`, and their
  `recommendation` is now the lowest mean CDF deviation rather than the lowest KS.
- `ds_balance_all_schemes(streamflow, seasonal_corr, seasons)` — the last two
  arguments are now optional.
- `spatial_maps()` is renamed **`spatial_sidi()`** (pairs with `spatial_spi()`).
  Its `month_scales` argument is gone; SPI maps come from `spatial_spi()`. With no
  `K`, `spatial_sidi` now follows the committed calibration instead of `self.K`.
- `spatial_maps_old`, `_process_grid_point_trends_old`, `spatial_trends()` and the
  `trend_grid` property are removed.

## [3.6.2] - Unreleased

### Added
- `spatial_sidi`: new `seasonal_params` argument. When passed the output of
  `set_optimal_SIDI_seasonal`'s `self.seasonal_params`, `K` is automatically
  resolved to the season-specific `best_k` matching the requested `timestamp`,
  and a warning reports the corresponding `weight_index` to use for consistent
  plotting. This allows the spatial SIDI grid to reflect a seasonal optimization
  that a single global `K` cannot represent.
- `spatial_sidi`: emits a warning when `K` is left unspecified and `self.optimal_k`
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
- `spatial_trends`: `windows` now also accepts a single `int` (e.g. `windows=10`)
  in addition to a list. Previously a bare int raised
  `TypeError: 'int' object is not iterable` when building `trend_grid`.
- `plot_spatial(var='CDN', ...)`: raises a clear `ValueError` when the requested
  `trend_grid[month_scale]` is entirely NaN, instead of letting the error
  propagate opaquely from matplotlib's colorbar (`pcolormesh` rejecting
  non-finite `vmin`/`vmax`). This happens when `spatial_trends` was run with
  `calculation_method=f_kde`, for which the exact deficit/native conversion is
  not supported (see above) — the new message points to `f_spi`/`f_spei`/
  `f_zscore` as the fix.
- `plot_spatial(var='SIDI', ...)`: title now shows the temporal scale (`K=...`)
  and the weighting scheme spelled out (e.g. "logarithmically decreasing")
  instead of the bare `weight_index` int, matching the naming used in
  `user_guide.md`. `spatial_sidi` now stores the `K` it actually ran with in
  `self.spatial_K`, since it resets `self.K` to its pre-call value on exit —
  reading `self.K` alone at plot time could report a stale/wrong scale.

### Documentation
- `spatial_guide.md` §2.6 (new): "Using an optimized SIDI" — documents the
  supported workflows for aligning `spatial_sidi` with a point-scale SIDI
  optimization: explicit manual `K`; `set_optimal_SIDI` (picked up automatically);
  and seasonal, either by passing the raw `analyze_correlation_seasonal` output
  (with `agg=`) to score it without committing, or by `set_optimal_SIDI_seasonal
  (overwrite=True)` then a bare `spatial_sidi()`.
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

