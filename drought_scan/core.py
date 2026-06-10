"""
author: PyDipa
# © 2026 Arianna Di Paola


Core module for Drought Scan.

This file defines the **main base classes** for drought analysis:
- `BaseDroughtAnalysis`: Parent class with core drought analysis functions.
- `Precipitation`: Handles precipitation-related calculations.
- `Streamflow`: Manages streamflow data.
- `PET`: Computes potential evapotranspiration (PET).
- `Balance`: Integrates water balance computations.
- 'Teleindex': a general purpose class base on BaseDroughtAnalysis to handles timeseries of Teleconnections (i.e. timeseries
not linked to any hydrografic basin (no shapefile required)

These classes serve as the **foundation** for the entire library.

# License: GNU General Public License v3.0 (GPLv3)
"""

try:
    import cmcrameri.cm as cmc
except Exception:
    cmc = None

import numpy as np
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from functools import partial
from scipy import stats






# --- drought_indices --------------------------------------------------------
from drought_scan.utils.drought_indices import (
    baseline_indices,
    f_kde,
    f_spei,
    f_spi,
    f_zscore,
    generate_weights,
    weighted_metrics,
)

# --- data_io ----------------------------------------------------------------
from drought_scan.utils.data_io import (
    import_netcdf_for_cumulative_variable,
    import_timeseries,
    load_shape,
    load_streamflow,
)

# --- hydrology --------------------------------------------------------------
from drought_scan.utils.hydrology import (
    severe_events_deficits_computation,
)

# --- visualization ----------------------------------------------------------
from drought_scan.utils.visualization import (
    monthly_profile,
    plot__covariates,
    plot_cdn_trends,
    plot_overview,
    plot_severe_events,
    spi_cmap,
)

# --- statistics -------------------------------------------------------------
from drought_scan.utils.statistics import (
    find_overlap,
    # _rolling_trend_analysis,
)


class BaseDroughtAnalysis:
    def __init__(self, ts, m_cal, K, start_baseline_year, end_baseline_year,basin_name,
                 calculation_method,threshold,index_name='SPI',day=None):
        """
        Base class for drought analysis.

        Args:
            ts (ndarray): Time series data (e.g., precipitation or streamflow).
            m_cal (ndarray): Calendar array (month, year) matching `ts`.
            K (int): Maximum temporal scale for SPI calculations.
            start_baseline_year (int): Starting year for baseline period.
            end_baseline_year (int): Ending year for baseline period.
            calculation_method (callable, optional): Function for index calculation. Defaults to f_kde.
            Available methods (in utils.py) are:
                f_spi:   FOR  POSITIVE & RIGHT-SKEWED DATA (uses a Gamma Function) but works fine also for positive normal distribuited sample
                f_spei:  FOR REAL VALUES & RIGHT-SKEWED (uses a Pearson III function)
                f_zscore FOR REAL VALUES NORMAL DISTRIBUTED
            threshold (float, optional): Threshold for severe events. Defaults to -1.
        """
        if len(ts) != len(m_cal):
            raise ValueError("The time series `ts` and calendar `m_cal` must have the same length.")
        if start_baseline_year > end_baseline_year:
            raise ValueError("`start_baseline_year` must be less than or equal to `end_baseline_year`.")
        if K <= 0:
            raise ValueError("`K` must be a positive integer.")

        if getattr(calculation_method, "requires_positive", False):
            finite = ts[np.isfinite(ts)]
            if finite.size and np.any(finite < 0):
                frac_neg = np.mean(finite < 0)
                raise ValueError(
                    f"Negative distribution detected ({frac_neg:.0%} of values < 0). "
                    f"{calculation_method.__name__} (Gamma) is defined only on positive, "
                    f"right-skewed data. Re-run with calculation_method=f_spei "
                    f"(Pearson III) or calculation_method=f_kde for real-valued "
                    f"variables such as the P-PET balance."
                )


        self.ts = ts
        self.m_cal = m_cal
        self.K = K
        self.start_baseline_year = start_baseline_year
        self.end_baseline_year = end_baseline_year
        self.threshold = threshold
        self.calculation_method = calculation_method
        self.index_name = index_name
        self.basin_name = basin_name
        self.SIDI_name = rf"$\mathrm{{D}}_{{\mathrm{{({self.index_name})}}}}$"

        # SPI-related attributes
        self.spi_like_set, self.c2r_index = self._calculate_spi_like_set()
        self.SIDI = self._calculate_SIDI()
        self.CDN = self._calculate_CDN()
        self.area_kmq = self._area()

    def _area(self):
        if isinstance(self, Teleindex):
            return -1
        if self.shape.crs is None:
            self.shape = self.shape.set_crs('epsg:4326')
        elif self.shape.crs.to_string() != 'EPSG:4326':
            self.shape = self.shape.to_crs('epsg:4326')

        area_proj = self.shape.to_crs(epsg=32632)

        #estiamte the area in square meters
        area_kmq = area_proj.geometry.area.iloc[0]/1e6
        return area_kmq

    def plot_boundary(self, ax=None, figsize=(7, 7), buffer_deg=0.15,
                      facecolor='#e8e8e8', edgecolor='k', linewidth=1.2):
        """
        Plot the basin boundary on an equal-area projection with
        coordinate frame and north arrow.

        Args:
            ax (GeoAxes, optional): Existing cartopy axes. If None, creates new figure.
            figsize (tuple): Figure size. Default (7, 7).
            buffer_deg (float): Padding around the basin in degrees. Default 0.15.
            facecolor (str): Basin fill color.
            edgecolor (str): Boundary line color.
            linewidth (float): Boundary line width.

        Returns:
            matplotlib.axes.Axes (GeoAxes)
        """
        import cartopy.crs as ccrs
        from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
        import matplotlib.pyplot as plt

        if not hasattr(self, 'shape') or self.shape is None:
            raise ValueError("No shapefile associated with this object.")

        shape_4326 = (self.shape.to_crs(epsg=4326)
                      if self.shape.crs is not None and self.shape.crs.to_epsg() != 4326
                      else self.shape.set_crs(epsg=4326) if self.shape.crs is None
        else self.shape)

        # --- bounding box & center ---
        b = shape_4326.geometry.total_bounds  # [minx, miny, maxx, maxy]
        lon0 = (b[0] + b[2]) / 2
        lat0 = (b[1] + b[3]) / 2

        # --- equal-area projection centered on basin ---
        projection = ccrs.AlbersEqualArea(
            central_longitude=lon0, central_latitude=lat0,
            standard_parallels=(b[1], b[3])
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': projection})

        # --- basin fill + boundary ---
        shape_4326.plot(ax=ax, transform=ccrs.PlateCarree(),
                        facecolor=facecolor, edgecolor=edgecolor,
                        linewidth=linewidth, zorder=2)

        # --- extent ---
        ax.set_extent([b[0] - buffer_deg, b[2] + buffer_deg,
                       b[1] - buffer_deg, b[3] + buffer_deg],
                      crs=ccrs.PlateCarree())

        # --- coordinate frame (graticule) ---
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='grey',
                          alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}

        # --- north arrow ---
        arrow_x, arrow_y = 0.93, 0.92  # axes fraction
        ax.annotate('N', xy=(arrow_x, arrow_y), xycoords='axes fraction',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.annotate('', xy=(arrow_x, arrow_y), xycoords='axes fraction',
                    xytext=(arrow_x, arrow_y - 0.08), textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', lw=1.8, color='k'))

        # --- title ---
        name = self.basin_name if self.basin_name else ''
        ax.set_title(name, fontweight='bold', fontsize=14)

        plt.tight_layout()
        return ax
    # =====================================================================
    # ▸ STANDARD Methods
    # =====================================================================
    def _compute_spi_old(self, month_scale,fit_params=None):
        """
        Calculate SPI for a specific temporal scale, optionally using precomputed gamma parameters.

        Args:
            month_scale (int): Temporal scale for SPI (e.g., SPI-3, SPI-6).
            fit_params (dict, optional): Precomputed parameters for the reference month,
                keyed by month (1-12). Each value is a dict with keys:
                - 'kde': fitted scipy gaussian_kde object
                - 'bw_factor': bandwidth factor used in the KDE fit
                - 'n_fit': number of samples used for fitting
                - 'fit_domain': string describing the domain of the fit
                If None, parameters are estimated from scratch.

        Returns:
            tuple:
                - ndarray: SPI time series for the given scale, with NaN for undefined values.
                - ndarray: Coefficients for SPI calculation (12 months x 6 columns).
        """
        Spi_ts = np.full_like(self.ts, np.nan, dtype=float)

        method = self.calculation_method
        base_func = method.func if isinstance(method, partial) else method

        if  base_func in [f_spi, f_spei,f_kde]:
            c2rspi = np.zeros((12, 4), dtype=float)
            way = 1
        elif  base_func == f_zscore:
            c2rspi = np.zeros((12, 2), dtype=float)
            way = 2

        for ref_month in range(1, 13):
            if fit_params is None:
                indices, spi_values, coeff, params = self.calculation_method(
                    self.ts, month_scale, ref_month, self.m_cal,
                    self.start_baseline_year, self.end_baseline_year
                )
                # self.fit_params[month_scale - 1, ref_month - 1, :] = params  # (a, loc, scale)

                # Auto-expand fit_params if the requested scale exceeds the original
                # allocation (K). This happens when _compute_spi is called on-the-fly
                # for windows > K (e.g. from deficit_from_spi or plot_trends).
                if month_scale > self.fit_params.shape[0]:
                    n_extra = month_scale - self.fit_params.shape[0]
                    n_params = self.fit_params.shape[2]
                    extra = np.full((n_extra, 12, n_params), np.nan)
                    self.fit_params = np.concatenate([self.fit_params, extra], axis=0)
                self.fit_params[month_scale - 1, ref_month - 1, :] = params


            else:
                # USE SAVED PARAMS
                # _compute_spi acts as a transparent dispatcher: it extracts the
                # pre-fitted parameters for the current reference month and forwards
                # them to the calculation_method without inspecting their internal
                # structure. Type interpretation is fully delegated to each method:
                #   - f_spi / f_spei : expects a tuple (param1, loc, param2)
                #   - f_kde          : expects a dict  {'kde': ..., 'bw_factor': ..., ...}
                #   - f_zscore       : expects a list  [baseline_mean, baseline_std]
                # This keeps routing logic separate from statistical logic.
                params = fit_params[ref_month]
                indices, spi_values, coeff, _ = self.calculation_method(
                    self.ts, month_scale, ref_month, self.m_cal,
                    self.start_baseline_year, self.end_baseline_year,
                    fit_params=params
                )


            if indices is None or spi_values is None or coeff is None:
                raise ValueError(f"`f_spi` returned invalid results for ref_month={ref_month}.")

            Spi_ts[indices] = spi_values.copy()
            c2rspi[ref_month - 1, :] = coeff.copy()
        return Spi_ts,c2rspi

    def _compute_spi(self, month_scale, fit_params=None):
        """
        Calculate SPI for a specific temporal scale, optionally using precomputed parameters.

        Args:
            month_scale (int): Temporal scale for SPI (e.g., SPI-3, SPI-6).
            fit_params (dict, optional): Precomputed parameters keyed by month (1-12).
                Structure depends on the calculation_method (tuple for f_spi/f_spei,
                dict for f_kde, list for f_zscore). If None, fitted from scratch.

        Returns:
            tuple:
                - ndarray: SPI time series for the given scale, with NaN for undefined values.
                - ndarray: c2r coefficients (12 months x {4|2} columns).

        Note:
            f_spi/f_spei/f_zscore parameters (numeric) are stored in the numeric
            array `self.fit_params`. f_kde parameters (a dict holding a live
            gaussian_kde object) are stored separately in `self.fit_params_kde`,
            keyed by (month_scale, ref_month), to keep the numeric array — and its
            np.isnan-based guards in native_to_spi/spi_to_native — intact.
        """
        Spi_ts = np.full_like(self.ts, np.nan, dtype=float)

        method = self.calculation_method
        base_func = method.func if isinstance(method, partial) else method

        if base_func in [f_spi, f_spei, f_kde]:
            c2rspi = np.zeros((12, 4), dtype=float)
            way = 1
        elif base_func == f_zscore:
            c2rspi = np.zeros((12, 2), dtype=float)
            way = 2

        for ref_month in range(1, 13):
            if fit_params is None:
                indices, spi_values, coeff, params = self.calculation_method(
                    self.ts, month_scale, ref_month, self.m_cal,
                    self.start_baseline_year, self.end_baseline_year
                )

                # --- Store fitted parameters ---
                if base_func == f_kde:
                    # KDE returns a dict with a live gaussian_kde object: keep it OUT
                    # of the numeric fit_params array (np.isnan/float dtype would break).
                    self.fit_params_kde[(month_scale, ref_month)] = params
                else:
                    # Numeric params (f_spi/f_spei: 3 floats; f_zscore: 2 floats).
                    # Auto-expand if the requested scale exceeds the original K
                    # allocation (e.g. from deficit_from_spi or plot_trends).
                    if month_scale > self.fit_params.shape[0]:
                        n_extra = month_scale - self.fit_params.shape[0]
                        n_params = self.fit_params.shape[2]
                        extra = np.full((n_extra, 12, n_params), np.nan)
                        self.fit_params = np.concatenate([self.fit_params, extra], axis=0)
                    self.fit_params[month_scale - 1, ref_month - 1, :] = params
            else:
                # USE SAVED PARAMS — transparent dispatcher: forwards the per-month
                # params object as-is, type interpretation delegated to each method:
                #   f_spi / f_spei : tuple (param1, loc, param2)
                #   f_kde          : dict  {'kde': ..., 'bw_factor': ..., ...}
                #   f_zscore       : list  [baseline_mean, baseline_std]
                params = fit_params[ref_month]
                indices, spi_values, coeff, _ = self.calculation_method(
                    self.ts, month_scale, ref_month, self.m_cal,
                    self.start_baseline_year, self.end_baseline_year,
                    fit_params=params
                )

            if indices is None or spi_values is None or coeff is None:
                raise ValueError(f"calculation_method returned invalid results for ref_month={ref_month}.")

            Spi_ts[indices] = spi_values.copy()
            c2rspi[ref_month - 1, :] = coeff.copy()
        return Spi_ts, c2rspi

    def _calculate_spi_like_set_old(self,fit_params=None):
        """
       Compute SPI values for all temporal scales up to K, optionally using precomputed gamma parameters.

       Args:
        fit_params (dict, optional): Nested dictionary of precomputed fit parameters,
        structured as {k: {m: params_dict}} where k is the temporal scale (1..K),
        m is the reference month (1-12), and params_dict contains the KDE fit
        (see _compute_spi for the expected structure of params_dict).
        If None, parameters are estimated from scratch at each scale.

       Returns:
           tuple:
               - ndarray: SPI values arranged in a 2D array (scale, time).
               - ndarray: 6 coefficients for each scale and month (K, 12, 6).
        """
        # Initialize SPI set and coefficients
        spiset = np.full((self.K, len(self.ts)), np.nan, dtype=float)
        method = self.calculation_method
        base_func = method.func if isinstance(method, partial) else method

        if  base_func in [f_spi, f_spei,f_kde]:
            c2rspi = np.zeros((self.K, 12, 4), dtype=float)
            self.fit_params = np.full((self.K, 12, 3), np.nan)  # (shape, loc, scale)

        elif  base_func == f_zscore:
            c2rspi = np.zeros((self.K, 12, 2), dtype=float)
            self.fit_params = np.full((self.K, 12, 2), np.nan)  # (shape, loc, scale)

        self.fit_params_kde = {}  #specific store for f_kde: {(scale, month): params_dict}

        # Calculate SPI for each temporal scale
        for k in range(1, self.K + 1):
            if fit_params is None:
                Spi_ts, coeff = self._compute_spi(k)
            else:
                params = fit_params[k]
                Spi_ts, coeff = self._compute_spi(k,fit_params=params)
            spiset[k - 1, :] = Spi_ts.copy()
            c2rspi[k - 1, :, :] = coeff.copy()
        return spiset, c2rspi

    def _calculate_spi_like_set(self, fit_params=None):
        """
       Compute SPI values for all temporal scales up to K, optionally using precomputed parameters.

       Args:
        fit_params (dict, optional): Nested dict {k: {m: params}} where k is the
        temporal scale (1..K) and m the reference month (1-12). If None, parameters
        are estimated from scratch at each scale.

       Returns:
           tuple:
               - ndarray: SPI values arranged as (scale, time).
               - ndarray: c2r coefficients (K, 12, {4|2}).
        """
        # Initialize SPI set and coefficients
        spiset = np.full((self.K, len(self.ts)), np.nan, dtype=float)
        method = self.calculation_method
        base_func = method.func if isinstance(method, partial) else method

        if base_func in [f_spi, f_spei, f_kde]:
            c2rspi = np.zeros((self.K, 12, 4), dtype=float)
            self.fit_params = np.full((self.K, 12, 3), np.nan)  # (shape, loc, scale)
        elif base_func == f_zscore:
            c2rspi = np.zeros((self.K, 12, 2), dtype=float)
            self.fit_params = np.full((self.K, 12, 2), np.nan)  # (mean, std)

        # Opaque store for non-numeric fit params (currently f_kde only).
        # Always allocated → exists for every method (empty when unused), so no
        # attribute appears/disappears depending on calculation_method.
        self.fit_params_kde = {}

        # Calculate SPI for each temporal scale
        for k in range(1, self.K + 1):
            if fit_params is None:
                Spi_ts, coeff = self._compute_spi(k)
            else:
                params = fit_params[k]
                Spi_ts, coeff = self._compute_spi(k, fit_params=params)
            spiset[k - 1, :] = Spi_ts.copy()
            c2rspi[k - 1, :, :] = coeff.copy()
        return spiset, c2rspi

    def native_to_spi(self, value, month_scale, ref_month):
        """
        Convert native-unit value(s) to SPI/SQI/SPEI etc via the fitted distribution stored
        at calibration time.

        Analytical counterpart to numerically inverting
        `np.polyval(c2r_index[scale-1, month-1, :], ...)`: uses the exact parameters
        of the fitted distribution (gamma for f_spi/f_spei, mean+std for f_zscore),
        avoiding the polynomial approximation error of `c2r_index`, which is
        non-trivial in the distribution tails.

        Parameters
        ----------
        value : float or array-like
            Value(s) in native units (e.g. mm for Precipitation, m^3/s for
            Streamflow) to convert to SPI/SQI.
        month_scale : int
            Accumulation scale (1..K) at which the conversion is performed.
        ref_month : int
            Reference calendar month (1..12) whose calibrated distribution is used.

        Returns
        -------
        spi : float or ndarray
            SPI/SQI value(s) corresponding to `value` at the given scale and
            reference month.

        Raises
        ------
        NotImplementedError
            If the active `calculation_method` is `f_kde`, whose fitted KDE object
            is not currently stored as numerical parameters in `fit_params`.
        ValueError
            If `calculation_method` is unrecognised.

        See Also
        --------
        spi_to_native : Inverse transform (SPI -> native units).
        c2r_index : Polynomial approximation, retained for backward compatibility
            (deprecated).
        """
        from functools import partial
        from scipy.stats import norm, gamma
        from drought_scan.utils.drought_indices import f_spi, f_spei, f_zscore, f_kde

        method = self.calculation_method
        base_func = method.func if isinstance(method, partial) else method

        # Guard: ensure fit_params has been computed for the requested scale
        if month_scale > self.fit_params.shape[0] or \
                np.any(np.isnan(self.fit_params[month_scale - 1, ref_month - 1, :])):
            raise ValueError(
                f"fit_params not available for scale={month_scale}, month={ref_month}. "
                f"Note: this method is not supported with f_kde. "
                f"Initialize the Drought Scan object with calculation_method=f_spi (or f_spei)."
            )

        params = self.fit_params[month_scale - 1, ref_month - 1, :]

        if base_func in [f_spi, f_spei]:
            a, loc, scale_param = params
            p = gamma.cdf(value, a, loc=loc, scale=scale_param)
            return norm.ppf(p)

        elif base_func == f_zscore:
            mean, std = params
            return (value - mean) / std

        elif base_func == f_kde:
            raise NotImplementedError(
                "native_to_spi is not yet implemented for f_kde. "
                "KDE requires storing the fitted gaussian_kde object, "
                "which is not currently saved as numerical parameters in fit_params."
            )

        else:
            raise ValueError(f"Unknown calculation_method: {base_func}")

    def spi_to_native(self, spi_value, month_scale, ref_month):
        """
        Convert SPI/SQI/SPEI etc to  native units via the fitted distribution stored at calibration time.

        Analytical counterpart to np.polyval(c2r_index[scale-1, month-1, :], spi_value):
        uses the exact parameters of the fitted distribution (gamma for f_spi/f_spei,
        mean+std for f_zscore), avoiding the polynomial approximation error of
        c2r_index in the distribution tails.

        Parameters
        ----------
        spi_value : float or array-like
            SPI/SQI value(s) to convert.
        month_scale : int
            Accumulation scale (1..K).
        ref_month : int
            Reference month (1..12).

        Returns
        -------
        float or ndarray
            Native-unit value(s) corresponding to `spi_value` at the given scale
            and reference month.

        See Also
        --------
        native_to_spi : Inverse transform (native → SPI).
        c2r_index : Polynomial approximation, retained for backward compatibility.
        """
        from functools import partial
        from scipy.stats import norm, gamma
        from drought_scan.utils.drought_indices import f_spi, f_spei, f_zscore, f_kde

        method = self.calculation_method
        base_func = method.func if isinstance(method, partial) else method

        # Guard: ensure fit_params has been computed for the requested scale
        if month_scale > self.fit_params.shape[0] or \
                np.any(np.isnan(self.fit_params[month_scale - 1, ref_month - 1, :])):
            raise ValueError(
                f"fit_params not available for scale={month_scale}, month={ref_month}. "
                f"Note: this method is not supported with f_kde. "
                f"Initialize the Drought Scan object with calculation_method=f_spi (or f_spei)."
            )
        params = self.fit_params[month_scale - 1, ref_month - 1, :]

        if base_func in [f_spi, f_spei]:
            a, loc, scale_param = params
            p = norm.cdf(spi_value)
            return gamma.ppf(p, a, loc=loc, scale=scale_param)

        elif base_func == f_zscore:
            mean, std = params
            return spi_value * std + mean

        elif base_func == f_kde:
            raise NotImplementedError(
                "spi_to_native not yet implemented for f_kde. "
                "KDE requires storing the fitted gaussian_kde object, "
                "which is not currently saved as numerical parameters."
            )

        else:
            raise ValueError(f"Unknown calculation_method: {base_func}")

    def _spi_like_set_ensemble_mean(self):
        """
    Compute the weighted SIDI values using predefined weighting functions.

    Returns:
        ndarray: Weighted SIDI values (time steps x number of implemented weighting function).

    """
        K = self.K if not hasattr(self, 'optimal_k') or self.optimal_k is None else self.optimal_k
        # print(f'************************************')
        # print(f'spiset ensamble mean up to SPI-{K}')
        weights = generate_weights(K)
        # weights = generate_weights(self.K)
        sidi = []
        for j in range(len(self.m_cal)):
            vec = self.spi_like_set[:K, j]
            sidi_w = [weighted_metrics(vec, w)[0] for w in weights.T]
            sidi.append(sidi_w)
        return np.array(sidi, dtype=float)

    def _calculate_SIDI(self):
        """
        Compute the Standardized Integrated Drought Index (SIDI).

        Returns:
            ndarray: SIDI values (time steps x number of implemented weighting function) standardized to zero mean and unit variance.

        """
        # Get baseline indices and ensemble mean
        tb1_id, tb2_id = baseline_indices(self.m_cal,self.start_baseline_year,self.end_baseline_year)
        sidi = self._spi_like_set_ensemble_mean()

        # Validate baseline indices
        if tb1_id >= tb2_id:
            raise ValueError("Invalid baseline indices: start index must be less than or equal to end index.")

        # Standardize the SIDI values
        baseline_values = sidi[tb1_id:tb2_id + 1, :]
        baseline_mean = np.nanmean(baseline_values, axis=0)
        baseline_std = np.nanstd(baseline_values, axis=0)

        if np.any(baseline_std == 0):
            raise ValueError("Baseline standard deviation contains zero values, cannot standardize.")

        SIDI = (sidi - baseline_mean) / baseline_std
        return SIDI

    @property
    def is_seasonal_sidi(self):
        """True if SIDI was overwritten by seasonal optimization."""
        return hasattr(self, 'seasonal_params')

    def recalculate_SIDI(self, K):
        """
        Recalculate SIDI using a custom K (top-K SPI-like scales) for each weighting
        without altering the original SPI-like set.

        Parameters
        ----------
        K : int
          Number of SPI-like scales to use for SIDI recalculation (top-K).

        Returns
        -------
        np.ndarray
          SIDI array with shape (time, n_weightings).
        """

        if K is None or K <= 0:
            raise ValueError("K must be a positive integer.")

        n_scales, T = self.spi_like_set.shape
        if K > n_scales:
            raise ValueError(f"K={K} exceeds available scales ({n_scales}).")

        # weights: shape (K, n_weightings)
        weights = generate_weights(K)
        if weights.shape[0] != K:
            raise RuntimeError("generate_weights(K) returned unexpected shape.")

        # Build SIDI (time x n_weightings)
        # self.spi_like_set has shape (scales, time). We use the first K rows.
        sidi_matrix = []
        for j in range(T):
            vec = self.spi_like_set[:K, j]
            sidi_w = [weighted_metrics(vec, w)[0] for w in weights.T]  # one value per weight_index
            sidi_matrix.append(sidi_w)

        sidi_matrix = np.array(sidi_matrix)  # (time, n_weightings)

        # Standardize on the original baseline
        tb1_id, tb2_id = baseline_indices(self.m_cal, self.start_baseline_year, self.end_baseline_year)
        baseline = sidi_matrix[tb1_id:tb2_id + 1, :]
        mean = np.nanmean(baseline, axis=0)
        std = np.nanstd(baseline, axis=0)
        if np.any(~np.isfinite(std)) or np.any(std == 0):
            raise ValueError("Zero or non-finite std in baseline; cannot standardize SIDI.")
        SIDI_new = (sidi_matrix - mean) / std

        return SIDI_new

    # =====================================================================
    # ▸ Seasonal SIDI helpers
    # =====================================================================
    @staticmethod
    def _build_seasons_dict(agg, seasons=None):
        """
        Build the month→season mapping from an aggregation keyword or a custom dict.

        Parameters
        ----------
        agg : str
            Aggregation scheme: 'quarter', 'semiannual' (alias 'biannual'),
            'four-monthly', 'monthly', or 'custom'.
        seasons : dict, optional
            Required when ``agg='custom'``. Keys are season labels, values
            are lists of month numbers (1-12).

        Returns
        -------
        dict
            ``{season_name: [month_ints]}``.
        """
        if agg == 'quarter':
            return {"DJF": [12, 1, 2], "MAM": [3, 4, 5],
                    "JJA": [6, 7, 8], "SON": [9, 10, 11]}
        elif agg in ('semiannual', 'biannual'):
            return {"autwin": [9, 10, 11, 12, 1, 2],
                    "springsum": [3, 4, 5, 6, 7, 8]}
        elif agg == 'four-monthly':
            return {"ONDG": [10, 11, 12, 1], "FMAM": [2, 3, 4, 5],
                    "JJAS": [6, 7, 8, 9]}
        elif agg == 'monthly':
            import calendar as _cal
            return {_cal.month_abbr[m]: [m] for m in range(1, 13)}
        elif agg == 'custom':
            if seasons is None:
                raise ValueError(
                    "When agg='custom', a `seasons` dictionary must be provided.")
            return seasons
        else:
            raise ValueError(
                f"Unrecognized aggregation: '{agg}'. "
                "Use 'quarter', 'semiannual', 'four-monthly', 'monthly', or 'custom'.")

    def recalculate_SIDI_seasonal(self, seasonal_params, seasons):
        """
        Recalculate SIDI using season-specific K and weight_index.

        For each timestep the method identifies the corresponding season and
        applies the season-specific K and weighting function.
        Standardisation is performed **per-season** on the baseline period
        (mixing seasons would dilute the signal — like averaging January and
        July temperatures).

        Parameters
        ----------
        seasonal_params : dict
            Output of ``analyze_correlation_seasonal()``, or any dict whose
            keys are season names and whose values contain at least
            ``'best_k'`` (int) and ``'col_best_weight'`` (int).
        seasons : dict
            Month mapping, e.g. ``{'DJF': [12,1,2], 'MAM': [3,4,5], …}``.

        Returns
        -------
        np.ndarray
            SIDI array of shape ``(time,)``.
        """
        T = len(self.m_cal)
        months = self.m_cal[:, 0].astype(int)
        tb1_id, tb2_id = baseline_indices(
            self.m_cal, self.start_baseline_year, self.end_baseline_year)

        # --- reverse map:  month → (best_k, best_weight) ----------------
        month_to_params = {}
        for season_name, month_list in seasons.items():
            if season_name not in seasonal_params:
                continue
            sp = seasonal_params[season_name]
            for m in month_list:
                month_to_params[m] = (sp['best_k'], sp['col_best_weight'])

        # --- raw (un-standardised) SIDI per timestep ---------------------
        SIDI_raw = np.full(T, np.nan, dtype=float)
        for j in range(T):
            m = months[j]
            if m not in month_to_params:
                continue
            k, wi = month_to_params[m]
            W = generate_weights(k)
            vec = self.spi_like_set[:k, j]
            SIDI_raw[j] = weighted_metrics(vec, W[:, wi])[0]

        # --- per-season standardisation on the baseline ------------------
        SIDI = np.full(T, np.nan, dtype=float)
        baseline_mask = np.zeros(T, dtype=bool)
        baseline_mask[tb1_id:tb2_id + 1] = True

        for season_name, month_list in seasons.items():
            if season_name not in seasonal_params:
                continue
            season_mask = np.isin(months, month_list)
            bl = season_mask & baseline_mask & np.isfinite(SIDI_raw)

            if np.sum(bl) < 2:
                print(f"Warning: season '{season_name}' has fewer than 2 "
                      f"valid baseline points; skipping standardisation.")
                continue

            mean = np.nanmean(SIDI_raw[bl])
            std = np.nanstd(SIDI_raw[bl])
            if std == 0 or not np.isfinite(std):
                raise ValueError(
                    f"Zero or non-finite std in baseline for season "
                    f"'{season_name}'; cannot standardize SIDI.")

            SIDI[season_mask] = (SIDI_raw[season_mask] - mean) / std

        return SIDI

    def _reapply_optimization(self):
        """Re-apply SIDI optimization after spi_like_set recalculation."""
        if self.is_seasonal_sidi:
            SIDI_1d = self.recalculate_SIDI_seasonal(
                self.seasonal_params['config'],
                self.seasonal_params['seasons']
            )
            self.SIDI = np.tile(SIDI_1d[:, np.newaxis], (1, 5))
        elif hasattr(self, 'optimal_k'):
            self.set_optimal_SIDI(
                self.optimal_k, self.optimal_weight_index, overwrite=True
            )

    # --- Classes that should NOT call analyze_correlation -----------------
    _EXCLUDED_FROM_CORRELATION = ()  # populated after Streamflow/Temperature/Teleindex are defined

    def _check_correlation_eligible(self):
        """Guard: prevent calling correlation methods on response variables."""
        if isinstance(self, self._EXCLUDED_FROM_CORRELATION):
            raise TypeError(
                f"{type(self).__name__} cannot call analyze_correlation(). "
                f"This method is for meteorological drivers (Precipitation, Pet, Balance) "
                f"to be correlated against a Streamflow target."
            )

    def analyze_correlation(self, streamflow, plot=True, plot_mode="all"):
        """
        Analyze correlations between this object's SIDI and a streamflow target
        (SQI₁) for different weighting schemes and K (month-scale) values.

        Applicable to: Precipitation, Pet, Balance.

        Parameters
        ----------
        streamflow : Streamflow (or any BaseDroughtAnalysis)
            Instance carrying the streamflow-based SPI-like indices.
        plot : bool, default True
            Whether to generate diagnostic plots.
        plot_mode : {'all', 'seasonal', 'monthly'}, default 'all'
            Scatter-plot coloring mode.

        Returns
        -------
        dict
            - "best_k" (int): Optimal month-scale (K).
            - "col_best_weight" (int): Index of the best weighting function.
            - "max_correlation" (float): Maximum R² value achieved.
            - "spi_corr" (ndarray): Sorted single-scale SPI vs SQI₁ R² values.
        """
        self._check_correlation_eligible()

        wlabel = ['equal weights (ew)', 'linearly decreasing weights (ldw)',
                  'logarithmically decreasing weights (lgdw)', 'linearly increasing weights (liw)',
                  'logarithmically increasing weights (lgiw)']

        if not isinstance(streamflow, BaseDroughtAnalysis):
            raise TypeError("The input must be an instance of Streamflow or BaseDroughtAnalysis.")

        # find the temporal overlap
        self_indices, streamflow_indices = find_overlap(self.m_cal, streamflow.m_cal)
        if len(self_indices) == 0 or len(streamflow_indices) == 0:
            raise ValueError("No overlapping data found between the two objects.")

        y = streamflow.spi_like_set[0, streamflow_indices]  # SQI-1
        spi_like_set = self.spi_like_set[:, self_indices]

        K_range = np.arange(1, self.K + 1)
        MatCorr = []

        print("Starting correlation analysis...")
        for k in K_range:
            W = generate_weights(k)
            sidis = []
            for doy in range(len(spi_like_set[0])):
                vec = spi_like_set[:k, doy]
                sidis.append([weighted_metrics(vec, w)[0] for w in W.T])
            sidis = np.array(sidis)

            rr = []
            for w in range(len(W.T)):
                SIDI = (sidis[:, w] - np.nanmean(sidis[:, w])) / np.nanstd(sidis[:, w])
                valid_mask = np.isfinite(y) & np.isfinite(SIDI)
                r = stats.pearsonr(SIDI[valid_mask], y[valid_mask])[0]
                rr.append(r ** 2)
            MatCorr.append(rr)

        # single SPI-k vs SQI-1 correlation
        rr_spi = []
        for j, spi in enumerate(spi_like_set):
            valid_mask = np.isfinite(y) & np.isfinite(spi)
            r = stats.pearsonr(spi[valid_mask], y[valid_mask])[0]
            rr_spi.append(r ** 2)
        rr_spi = np.array(rr_spi)
        ii = np.argsort(rr_spi)[::-1]
        R2_spi = np.array([np.arange(1, self.K + 1)[ii], rr_spi[ii]]).T

        MatCorr = np.array(MatCorr)
        max_corr = np.max(MatCorr)
        best_k, best_weight = np.unravel_index(np.argmax(MatCorr), MatCorr.shape)

        print(f"Best correlation: R2  = {max_corr:.3f} (K={K_range[best_k]}, Weight={wlabel[best_weight]})")

        W = generate_weights(K_range[best_k])
        sidi = []
        for doy in range(len(spi_like_set[0])):
            vec = spi_like_set[:K_range[best_k], doy]
            sidi.append(weighted_metrics(vec, W[:, best_weight])[0])
        sidi = np.array(sidi)
        SIDI = (sidi - np.nanmean(sidi)) / np.nanstd(sidi)

        # --- plots ----------------------------------------------------------------
        if plot:
            plt.figure(figsize=(10, 5))
            for w in range(len(W.T)):
                plt.plot(MatCorr[:, w], label=wlabel[w], linewidth=2)
            plt.grid()
            plt.legend(loc=3)
            plt.xticks(np.arange(len(K_range)), K_range)
            plt.ylabel(r"$R^2$", fontweight="bold", fontsize=12)
            plt.xlabel("Month-scale (K)", fontweight="bold", fontsize=12)
            plt.title(f"Correlation Analysis: {self.SIDI_name}  vs.  {streamflow.index_name}1",
                      fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.show(block=False)

            # basic scan plot
            self.plot_scan(optimal_k=K_range[best_k], weight_index=best_weight)

            plt.figure(figsize=(7, 7))
            if plot_mode == "all":
                plt.plot(SIDI, y, 'ok', markerfacecolor='yellow', linewidth=2)
            elif plot_mode == "seasonal":
                g1 = [4, 5, 6, 7, 8, 9]
                g2 = [10, 11, 12, 1, 2, 3]
                # summer
                m1summer = np.isin(self.m_cal[self_indices, 0], g1)
                m2summer = np.isin(streamflow.m_cal[streamflow_indices, 0], g1)
                f = np.isfinite(SIDI[m1summer]) & np.isfinite(y[m2summer])
                rho, pval = stats.pearsonr(SIDI[m1summer][f], y[m2summer][f])
                rho = 0 if pval > 0.5 else rho
                plt.plot(SIDI[m1summer], y[m2summer], 'o', color='tab:olive', alpha=0.4,
                         label=f'Apr-Oct; $R^2$ = {np.round(rho ** 2, 2)}')
                # winter
                m1winter = np.isin(self.m_cal[self_indices, 0], g2)
                m2winter = np.isin(streamflow.m_cal[streamflow_indices, 0], g2)
                f = np.isfinite(SIDI[m1winter]) & np.isfinite(y[m2winter])
                rho, pval = stats.pearsonr(SIDI[m1winter][f], y[m2winter][f])
                rho = 0 if pval > 0.5 else rho
                plt.plot(SIDI[m1winter], y[m2winter], 'o', color='tab:blue', alpha=0.4,
                         label=f'Nov-Mar; $R^2$ = {np.round(rho ** 2, 2)}')
            elif plot_mode == 'monthly':
                if cmc is not None:
                    c = plt.get_cmap(cmc.romaO, 12)
                else:
                    c = plt.get_cmap('twilight_shifted', 12)
                for month in range(1, 13):
                    m1 = np.where(self.m_cal[self_indices, 0] == month)[0]
                    m2 = np.where(streamflow.m_cal[streamflow_indices, 0] == month)[0]
                    plt.plot(SIDI[m1], y[m2], 'o', color=c(month / 12), label=f'month {month}')

            plt.plot(np.arange(-3, 4), np.arange(-3, 4), '--', color='grey')
            plt.grid()
            plt.ylabel(f"{streamflow.index_name}1 ", fontweight="bold", fontsize=12)
            plt.xlabel(f"{self.SIDI_name}", fontweight="bold", fontsize=12)
            plt.title(f"{self.SIDI_name} vs.  {streamflow.index_name}1 . K={best_k} - "
                      f"weighting function n. {best_weight}; $R^2$ = {max_corr:.2f}",
                      fontsize=14, fontweight="bold")
            plt.legend(fontsize=12)
            plt.tight_layout()
            plt.show(block=False)

        return {"best_k": K_range[best_k], "col_best_weight": best_weight,
                "max_correlation": max_corr, 'spi_corr': R2_spi}

    def analyze_correlation_seasonal(self, streamflow, agg='quarter', plot=True, seasons=None):
        """
        Perform seasonal correlation analysis between this object's SIDI and a
        streamflow target (SQI₁) for different weighting schemes and K values.

        Applicable to: Precipitation, Pet, Balance.

        Parameters
        ----------
        streamflow : Streamflow (or any BaseDroughtAnalysis)
            Instance carrying the streamflow-based SPI-like indices.
        agg : {'quarter', 'semiannual', 'four-monthly', 'monthly', 'custom'}, default 'quarter'
            Temporal aggregation scheme.
        plot : bool, default True
            Whether to generate diagnostic plots.
        seasons : dict, optional
            Custom month mapping when ``agg='custom'``.

        Returns
        -------
        dict
            Per-season dictionary with keys:
            ``best_k``, ``col_best_weight``, ``max_correlation``, ``R2_matrix``, ``sample number``.
        """
        self._check_correlation_eligible()

        wlabel = ['EW', 'Lin. DW', 'Log. DW', 'Lin. IW', 'Log. IW']

        if seasons is not None:
            agg = 'custom'

        if not isinstance(streamflow, BaseDroughtAnalysis):
            raise TypeError("The input must be an instance of Streamflow or BaseDroughtAnalysis.")

        # --- Temporal overlap ---
        self_indices, streamflow_indices = find_overlap(self.m_cal, streamflow.m_cal)
        if len(self_indices) == 0 or len(streamflow_indices) == 0:
            raise ValueError("No overlapping data found between the two objects.")

        y = streamflow.spi_like_set[0, streamflow_indices]
        spi_like_set = self.spi_like_set[:, self_indices]
        m_cal_overlap = np.array([self.m_cal[i] for i in self_indices])
        months_overlap = np.array([m[0] for m in m_cal_overlap], dtype=int)

        K_range = np.arange(1, self.K + 1)

        def _compute_corr(y_sub, spi_sub):
            """Compute the R² correlation matrix for all K and weighting functions."""
            MatCorr = []
            for k in K_range:
                W = generate_weights(k)
                sidis = []
                for doy in range(spi_sub.shape[1]):
                    vec = spi_sub[:k, doy]
                    sidis.append([weighted_metrics(vec, w)[0] for w in W.T])
                sidis = np.array(sidis)
                rr = []
                for w in range(W.shape[1]):
                    SIDI = (sidis[:, w] - np.nanmean(sidis[:, w])) / np.nanstd(sidis[:, w])
                    valid_mask = np.isfinite(y_sub) & np.isfinite(SIDI)
                    if np.sum(valid_mask) > 10:
                        r = stats.pearsonr(SIDI[valid_mask], y_sub[valid_mask])[0]
                        rr.append(r ** 2)
                    else:
                        rr.append(np.nan)
                MatCorr.append(rr)
            return np.array(MatCorr)

        # --- Season definitions ---
        seasons = self._build_seasons_dict(agg, seasons)

        # year coverage check
        all_months = [m for lst in seasons.values() for m in lst]
        unique = set(all_months)
        missing = [m for m in range(1, 13) if m not in unique]
        duplicates = [m for m in unique if all_months.count(m) > 1]
        if missing:
            print("Alert: missing month(s):", missing)
        if duplicates:
            print("Alert: duplicate month(s):", duplicates)

        # --- figure layout ---
        n_seasons = len(seasons)
        layout_map = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2), 5: (2, 3), 6: (2, 3),
                      12: (1, 1)}  # monthly handled separately
        nrows, ncols = layout_map.get(n_seasons, (2, 3))
        if agg == 'monthly':
            figsize1, figsize2 = (7, 6), (7, 6)
        elif agg == 'custom':
            wid = np.round(5 * ncols, 1)
            h = np.round(5 * nrows, 1) + 0.5
            figsize1, figsize2 = (wid + 3, h), (wid, h)
        else:
            figsize1 = {2: (14, 5), 3: (16, 6), 4: (14, 10)}.get(n_seasons, (14, 10))
            figsize2 = {2: (10, 6), 3: (15, 6), 4: (10, 11)}.get(n_seasons, (10, 11))

        # --- Compute correlations per season ---
        print("Starting correlation analysis (seasonal)...")
        MatCorr = {}
        for name, mlist in seasons.items():
            idx = np.isin(months_overlap, mlist)
            if np.count_nonzero(idx) <= 10:
                continue
            M = _compute_corr(y[idx], spi_like_set[:, idx])
            max_corr = np.nanmax(M)
            bk, bw = np.unravel_index(np.nanargmax(M), M.shape)
            print(f" Season {name}: best R²={max_corr:.3f} (K={K_range[bk]}, Weight={wlabel[bw]})")
            MatCorr[name] = {
                "best_k": int(K_range[bk]),
                "col_best_weight": int(bw),
                "max_correlation": float(max_corr),
                "R2_matrix": M,
                "sample number": np.count_nonzero(idx),
            }

        # --- Plot 1: R² vs K ---
        if agg == 'monthly':
            import calendar
            months = list(calendar.month_abbr[1:])
            r2max = np.array([MatCorr.get(m, {}).get('max_correlation', np.nan)
                              for m in months], float)
            plt.figure(figsize=(7, 5))
            plt.bar(months, r2max, 0.5, alpha=0.8, facecolor='dimgrey', edgecolor='k')
            plt.ylabel(r"$\max R^2$")
            plt.ylim(0, 1)
            plt.grid(axis='y', alpha=0.3)
            plt.title(f"{self.basin_name} - Monthly max R² "
                      f"{self.SIDI_name} vs. {streamflow.index_name}1")
            plt.axhline(y=0.4, linestyle='--', color='tab:orange', label='Significant limit')
            plt.legend()
            plt.tight_layout()
            plt.show(block=False)
        else:
            if plot:
                fig, ax = plt.subplots(figsize=figsize1, nrows=nrows, ncols=ncols)
                ax = ax.ravel()
                for i, name in enumerate(MatCorr.keys()):
                    mat = MatCorr[name]['R2_matrix']
                    for w in range(mat.shape[1]):
                        ax[i].plot(mat[:, w], label=wlabel[w], linewidth=2)
                    ax[i].grid()
                    ax[i].set_xticks(np.arange(0, len(K_range), 3))
                    ax[i].set_xticklabels(K_range[0:-1:3])
                    ax[i].tick_params(axis='x', labelsize=14)
                    ax[i].tick_params(axis='y', labelsize=14)
                    ax[i].set_ylabel(r"$R^2$", fontweight="bold", fontsize=16)
                    ax[i].set_xlabel("Month-scale (K)", fontweight="bold", fontsize=16)
                    ax[i].set_title(name, fontweight="bold", fontsize=16)
                    if i == 0:
                        ax[i].legend(loc=3)
                fig.suptitle(
                    f"{self.basin_name} - Correlation Analysis: "
                    f"{self.SIDI_name} vs. {streamflow.index_name}1",
                    fontsize=16, fontweight="bold")
                plt.tight_layout()
                plt.show(block=False)

                # --- Plot 2: Scatter plots ---
                if len(seasons) > 5:
                    if cmc is not None:
                        c = plt.get_cmap(cmc.romaO, len(seasons))
                    else:
                        c = plt.get_cmap('twilight_shifted', len(seasons))
                else:
                    c = ['tab:olive', 'tab:brown', 'tab:orange', 'tab:cyan', 'tab:purple']

                fig, ax = plt.subplots(figsize=figsize2, nrows=nrows, ncols=ncols)
                ax = ax.ravel()
                for i, (season, vals) in enumerate(MatCorr.items()):
                    best_k = vals['best_k']
                    best_weight = vals['col_best_weight']
                    W = generate_weights(best_k)
                    sidi = []
                    for doy in range(spi_like_set.shape[1]):
                        vec = spi_like_set[:best_k, doy]
                        sidi.append(weighted_metrics(vec, W[:, best_weight])[0])
                    sidi = np.array(sidi)
                    SIDI = (sidi - np.nanmean(sidi)) / np.nanstd(sidi)

                    idx = np.isin(months_overlap, seasons[season])
                    valid = np.isfinite(SIDI[idx]) & np.isfinite(y[idx])
                    if np.count_nonzero(valid) > 10:
                        r, _ = stats.pearsonr(SIDI[idx][valid], y[idx][valid])
                        r2 = r ** 2
                    else:
                        r2 = np.nan

                    ax[i].plot(SIDI[idx], y[idx], 'o', color=c[i], alpha=0.4,
                               label=f'$R^2$ = {np.round(r2, 2)} \n K ={best_k}; {wlabel[best_weight]}')
                    ax[i].plot(np.arange(-3, 4), np.arange(-3, 4), '--', color='grey')
                    ax[i].tick_params(axis='x', labelsize=14)
                    ax[i].tick_params(axis='y', labelsize=14)
                    ax[i].grid()
                    ax[i].set_title(season, fontweight="bold", fontsize=16)
                    ax[i].set_ylabel(f"{streamflow.index_name}1", fontweight="bold", fontsize=16)
                    ax[i].set_xlabel(f"{self.SIDI_name}", fontweight="bold", fontsize=16)
                    ax[i].legend(fontsize=12)

                fig.suptitle(
                    f"{self.basin_name} - {self.SIDI_name} vs. {streamflow.index_name}1 "
                    f"— Best seasonal configurations",
                    fontsize=14, fontweight="bold")
                if i < ncols * nrows - 1:
                    fig.delaxes(ax[-1])
                plt.tight_layout()
                plt.show(block=False)

        return MatCorr

    def spi_sqi_corr(self, streamflow, plot=True):
        """
        Compute the month-wise correlation between this object's index at
        scales k = 1…K and a streamflow target (SQI₁), returning an R² matrix
        of shape (12 × K).

        Applicable to: Precipitation, Pet, Balance.

        For each calendar month (Jan…Dec), the Pearson correlation between
        the k-scale index and SQI₁ is computed. Non-significant correlations
        (p ≥ 0.05) are zeroed out.

        Parameters
        ----------
        streamflow : BaseDroughtAnalysis
            Object carrying streamflow-based SPI-like indices
            (``spi_like_set[0]`` = SQI₁).
        plot : bool, default True
            If True, produce a contourf heatmap of R²(month, k) with
            month-wrapping for visual continuity.

        Returns
        -------
        R2 : ndarray, shape (12, K)
            Determination coefficients. Rows = calendar months (Jan=0…Dec=11),
            columns = scales (k=1…K).
        """
        self._check_correlation_eligible()

        # --- 1. Temporal overlap ---
        self_indices, streamflow_indices = find_overlap(self.m_cal, streamflow.m_cal)
        if len(self_indices) == 0 or len(streamflow_indices) == 0:
            raise ValueError("No overlapping data found.")

        # --- 2. Extract overlapping indices ---
        sqi1 = streamflow.spi_like_set[0, streamflow_indices]
        spi_like_set = self.spi_like_set[:, self_indices]
        m_cal = self.m_cal[self_indices]

        # --- 3. Compute R²(month, k) ---
        R2 = []
        Ki = np.arange(self.K)

        for m in range(1, 13):
            r2_row = []
            ii = np.where(m_cal[:, 0] == m)[0]
            y = sqi1[ii]

            for K in Ki:
                x = spi_like_set[K][ii]
                mask = np.isfinite(x) & np.isfinite(y)

                if mask.sum() < 3:
                    r2_row.append(0)
                    continue

                rho, pval = stats.pearsonr(x[mask], y[mask])
                r2_val = np.round(rho ** 2, 2) if pval < 0.05 else 0
                r2_row.append(r2_val)

            R2.append(r2_row)

        R2 = np.array(R2)

        # --- 4. Plot with month wrapping ---
        if plot:
            # Extend matrix: pad Nov-Dec before Jan and Jan-Feb after Dec
            extension = np.vstack((R2[-2:, :], R2))
            mat = np.vstack((extension, R2[0:2, :]))
            mat = np.flipud(mat)
            x_months = np.concatenate((np.arange(11, 13), np.arange(1, 13), np.arange(1, 3)))

            plt.figure(figsize=(10, 4))
            levels = np.arange(0.05, 1.15, 0.1)
            centers = (levels[:-1] + levels[1:]) / 2

            y_idx = np.arange(len(x_months))
            X, Y = np.meshgrid(np.arange(1, self.K + 1), y_idx)

            cf = plt.contourf(X, Y, mat, cmap='pink_r', levels=levels)

            plt.xticks(np.arange(1, self.K + 1))
            plt.yticks(y_idx,
                       np.flipud(['Nov', 'Dec', 'Jan.', 'Feb.', 'Mar.', 'Apr.',
                                  'May', 'Jun.', 'Jul.', 'Aug.', 'Sep.', 'Oct.',
                                  'Nov.', 'Dec.', 'Jan.', 'Feb.']))

            # Fade out the padded rows
            plt.axhspan(13.5, 15, facecolor='white', alpha=0.45, zorder=10)
            plt.axhspan(0, 1.5, facecolor='white', alpha=0.45, zorder=10)
            plt.axhline(y=1.5, color='dimgray', linewidth=0.7)
            plt.axhline(y=13.5, color='dimgray', linewidth=0.7)

            plt.xlabel('month scale (K)')
            plt.title(f"{self.basin_name} — {self.index_name}$_k$ vs {streamflow.index_name}$_1$  ($R^2$)",
                      fontsize=12, fontweight='bold')
            # plt.title(f"{self.basin_name} — {self.index_name}ₖ vs {streamflow.index_name}₁  (R²)",
            #           fontsize=12, fontweight='bold')

            cbar = plt.colorbar(cf, ticks=centers)
            cbar.ax.set_yticklabels([f"{c:.2f}" for c in centers])
            cbar.set_label(r"$R^2$", rotation=0, labelpad=12, fontsize=12)

            plt.gca().set_yticks(np.arange(1.5, 14, 1), minor=True)
            plt.grid(axis='y', which='minor', linestyle='--', alpha=0.7)
            plt.gca().tick_params(axis='y', length=0)

            plt.tight_layout()

        return R2

    # =====================================================================
    # =====================================================================
    # BENCHMARKING D(SPI) | SIDI
    def benchmark_convolution(self, streamflow, var=None, Kmax=None, plot=True,
                              agg=None, seasons=None):
        """
        Unified linear convolution benchmark via OLS (Benchmark A and B).

        Estimates the rainfall-runoff relationship by fitting a discrete linear
        convolution model with memory up to K months. Two modes are available,
        selected via the `var` parameter:

        **var='P' — Benchmark A: Unit Hydrograph on raw precipitation**

            Q(t) ≈ Σ_{j=0}^{K-1} h(j) · P(t-j) + intercept

        **var='SPI' — Benchmark B: Impulse response on standardized anomalies**

            SQI_1(t) ≈ Σ_{j=0}^{K-1} h(j) · SPI_1(t-j) + intercept

        CONCEPTUAL POSITIONING
        ----------------------
        The comparison of R²_adj across var='P', var='SPI', and the parametric
        SIDI-based model disentangles three sources of predictive power:
            (i)  distributional non-linearity of SPI at scale 1  (A → B gain)
            (ii) multi-scale aggregation of standardized anomalies (B → SIDI gain)
            (iii) combined effect                                  (A → SIDI gain)

        For each K in [1, ..., Kmax], a separate OLS model is fitted. The optimal
        K is selected at the end of the initial monotonic-increasing plateau of
        R²_adj vs K (via `_monotonic_plateau`), to stop before overfitting or
        numerical noise takes over at large K.

        Parameters
        ----------
        streamflow : BaseDroughtAnalysis
            Streamflow object with .ts, .spi_like_set, and .m_cal attributes.
        var : str or None, optional
            'P'   — Benchmark A: raw precipitation vs streamflow (default).
            'SPI' — Benchmark B: SPI_1 anomalies vs SQI_1.
            None defaults to 'P'.
        Kmax : int, optional
            Maximum lag (months) to test. Defaults to self.K.
        plot : bool, optional
            If True, plots R²_adj vs K and the estimated impulse response h(j).
            Default is True.
        agg : {'quarter', 'semiannual', 'four-monthly', 'monthly', 'custom'} or None
            Temporal aggregation scheme for seasonal analysis.
            If None (default), the analysis is run on all data together.
        seasons : dict, optional
            Custom month mapping when ``agg='custom'`` or when passing a dict
            directly (sets agg='custom' automatically). If None and agg is None,
            no seasonal split is performed.

        Returns
        -------
        If seasons is None and agg is None:
            dict with keys:
                'R2_adj'          : np.ndarray, shape (Kmax,)
                'optimal_K'       : int   (plateau index, 1-based)
                'K_range'         : np.ndarray
                'beta'            : np.ndarray, shape (optimal_K + 1,)
                'condition_number': float
                'metrics'         : dict
        Else:
            dict keyed by season name, each value being the same dict as above,
            with an additional key 'sample_number'.

        Notes
        -----
        - Requires temporal overlap between self.m_cal and streamflow.m_cal.
        - NaN values in either input or output series are excluded row-wise.
        - The guard `len(rows) < K + 2` skips degenerate configurations.
        - `optimal_K` follows the plateau criterion (first end of monotonic-
          increasing sequence in R²_adj), not the global maximum. This stabilises
          beta against multicollinearity at large K.
        - For physically interpretable ordinates when condition_number >> 30,
          consider ridge regression.

        References
        ----------
        Sherman, L.K. (1932). Streamflow from rainfall by the unit-graph method.
            Engineering News-Record, 108, 501-505.
        Chow, V.T., Maidment, D.R., Mays, L.W. (1988). Applied Hydrology.
            McGraw-Hill, New York.
        McKee, T.B., Doesken, N.J., Kleist, J. (1993). The relationship of drought
            frequency and duration to time scales. Proc. 8th Conference on Applied
            Climatology, 179-183.
        Vicente-Serrano, S.M., López-Moreno, J.I., Beguería, S., et al. (2012).
            Accurate computation of a streamflow drought index. Journal of
            Hydrologic Engineering, 17(2), 318-332.
        """

        self._check_correlation_eligible()

        if var is None:
            var = 'P'
        if var not in ('P', 'SPI'):
            raise ValueError("`var` must be 'P' or 'SPI'.")
        if Kmax is None:
            Kmax = self.K

        if seasons is not None:
            agg = 'custom'

        # --- Temporal overlap ---
        self_indices, streamflow_indices = find_overlap(self.m_cal, streamflow.m_cal)
        if len(self_indices) == 0 or len(streamflow_indices) == 0:
            raise ValueError("No overlapping data found between the two objects.")

        if var == 'P':
            driver = self.ts[self_indices]
            target = streamflow.ts[streamflow_indices]  # streamflow.spi_like_set[0, streamflow_indices]#
            x_label = 'P'
            y_label = 'Q'
            title_A = 'Benchmark A — Unit Hydrograph OLS'
            title_B = 'Unit Hydrograph estimated by OLS'
            y_label_beta = 'h(j) — beta OLS'
            x_label_beta = 'lag (months)'
        else:  # 'SPI'
            driver = self.spi_like_set[0, self_indices]
            target = streamflow.spi_like_set[0, streamflow_indices]
            x_label = 'SPI1'
            y_label = 'SQI1'
            title_A = 'Benchmark B — SPI1 lag convolution OLS'
            title_B = 'Impulse response on SPI1 estimated by OLS'
            y_label_beta = 'h(j) — beta OLS'
            x_label_beta = 'lag (months)'

        m_cal_overlap = np.array([self.m_cal[i] for i in self_indices])
        months_overlap = np.array([m[0] for m in m_cal_overlap], dtype=int)

        K_range = np.arange(1, Kmax + 1)
        n = len(driver)

        def _monotonic_plateau(x):
            """
            Returns the index of the last element in the initial monotonic
            increasing sequence of x. Used to identify the plateau of R²_adj
            vs K, stopping before overfitting or numerical noise takes over.
            """
            last = 0
            for i in range(1, len(x)):
                if np.isfinite(x[i]) and x[i] >= x[i - 1]:
                    last = i
                else:
                    break
            return last

        def _run_benchmark(driver_sub, target_sub, season_mask=None):
            """OLS convolution for a given data subset. Returns the full result dict."""
            MatCorr = []
            for K in K_range:
                rows, y_rows = [], []
                for t in range(K - 1, len(driver_sub)):
                    if season_mask is not None and not season_mask[t]:
                        continue
                    if np.isnan(target_sub[t]):
                        continue
                    lag_vec = driver_sub[t - K + 1:t + 1][::-1]
                    if np.any(np.isnan(lag_vec)):
                        continue
                    rows.append(lag_vec)
                    y_rows.append(target_sub[t])

                if len(rows) < K + 2:
                    MatCorr.append(np.nan)
                    continue

                X = np.column_stack([np.array(rows), np.ones(len(rows))])
                y_v = np.array(y_rows)
                beta, _, _, _ = np.linalg.lstsq(X, y_v, rcond=None)
                y_hat = X @ beta
                ss_res = np.sum((y_v - y_hat) ** 2)
                ss_tot = np.sum((y_v - y_v.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                n_v, p = len(y_v), K
                MatCorr.append(1 - (1 - r2) * (n_v - 1) / (n_v - p - 1))

            MatCorr = np.array(MatCorr)

            # === FIX: plateau-based selection instead of global argmax ===
            # Aligns `result['optimal_K']` with the K shown by the plot,
            # and stabilises beta against multicollinearity at large K.
            plateau_idx = _monotonic_plateau(MatCorr)
            best_K = int(K_range[plateau_idx])
            # =============================================================

            rows, y_rows, t_idx = [], [], []
            for t in range(best_K - 1, len(driver_sub)):
                if season_mask is not None and not season_mask[t]:
                    continue
                if np.isnan(target_sub[t]):
                    continue
                lag_vec = driver_sub[t - best_K + 1:t + 1][::-1]
                if np.any(np.isnan(lag_vec)):
                    continue
                rows.append(lag_vec)
                y_rows.append(target_sub[t])
                t_idx.append(t)

            X_opt = np.column_stack([np.array(rows), np.ones(len(rows))])
            y_opt = np.array(y_rows)
            beta_opt, _, _, sv = np.linalg.lstsq(X_opt, y_opt, rcond=None)
            cond_num = float(np.max(sv) / np.min(sv[sv > 0])) if len(sv) > 0 else np.nan

            if cond_num > 30:
                import warnings
                warnings.warn(
                    f"benchmark_convolution (var='{var}'): condition number = {cond_num:.1f} > 30. "
                    f"Multicollinearity at K={best_K} — beta coefficients may be unstable.",
                    UserWarning
                )

            # Compute complementary skill metrics at optimal K
            y_pred_opt = X_opt @ beta_opt
            metrics = self._eval_metrics(y_opt, y_pred_opt)

            return {
                'R2_adj': MatCorr,
                'optimal_K': best_K,
                'K_range': K_range,
                'beta': beta_opt,
                'condition_number': cond_num,
                'metrics': metrics,
            }

        # ---------------------------------------------------------------
        # CASE 1: no seasonal split — original behaviour
        # ---------------------------------------------------------------
        if agg is None:
            print("Starting correlation analysis...")
            result = _run_benchmark(driver, target)

            if plot:
                best_K = result['optimal_K']
                beta_opt = result['beta']
                MatCorr = result['R2_adj']

                fig, axes = plt.subplots(1, 2, figsize=(10, 3))
                axes[0].plot(K_range, MatCorr, 'k-o', lw=2)
                axes[0].axvline(best_K, color='r', ls='--', label=f'optimal K={best_K}')
                axes[0].set_xlabel('K (lag months)')
                axes[0].set_ylabel('R²_adj')
                axes[0].set_title(title_A)
                axes[0].legend()
                axes[0].grid(alpha=0.3)

                axes[1].bar(range(best_K), beta_opt[:-1],
                            color='grey', edgecolor='k', linewidth=0.5)
                axes[1].axhline(0, color='k', lw=0.8, ls='--')
                axes[1].set_xlabel(x_label_beta)
                axes[1].set_ylabel(y_label_beta)
                axes[1].set_title(f'{title_B} (K={best_K})')
                axes[1].grid(alpha=0.3, axis='y')
                plt.tight_layout()
                plt.show()

            return result

        # ---------------------------------------------------------------
        # CASE 2: seasonal split
        # ---------------------------------------------------------------
        seasons_dict = self._build_seasons_dict(agg, seasons)

        # year coverage check (same guard as analyze_correlation_seasonal)
        all_months = [m for lst in seasons_dict.values() for m in lst]
        unique = set(all_months)
        missing = [m for m in range(1, 13) if m not in unique]
        duplicates = [m for m in unique if all_months.count(m) > 1]
        if missing:
            print("Alert: missing month(s):", missing)
        if duplicates:
            print("Alert: duplicate month(s):", duplicates)

        print("Starting correlation analysis (seasonal)...")
        results = {}
        for name, mlist in seasons_dict.items():
            idx = np.isin(months_overlap, mlist)
            if np.count_nonzero(idx) <= Kmax + 2:
                print(f" Season {name}: insufficient data, skipped.")
                continue
            res = _run_benchmark(driver, target, season_mask=idx)
            res['sample_number'] = int(np.count_nonzero(idx))
            print(f" Season {name}: best R²_adj={res['R2_adj'][res['optimal_K'] - 1]:.3f} "
                  f"(K={res['optimal_K']})")
            results[name] = res

        if plot:
            n_seasons = len(results)
            if n_seasons == 0:
                return results

            fig, axes = plt.subplots(nrows=n_seasons, ncols=2,
                                     figsize=(10, 3 * n_seasons))
            if n_seasons == 1:
                axes = axes[np.newaxis, :]  # uniform indexing

            for i, (season, res) in enumerate(results.items()):
                best_K = res['optimal_K']
                beta_opt = res['beta']
                MatCorr = res['R2_adj']

                axes[i, 0].plot(K_range, MatCorr, 'k-o', lw=2)
                axes[i, 0].axvline(best_K, color='r', ls='--', label=f'optimal K={best_K}')
                axes[i, 0].set_xlabel('K (lag months)')
                axes[i, 0].set_ylabel('R²_adj')
                axes[i, 0].set_title(f'{title_A} — {season}')
                axes[i, 0].legend()
                axes[i, 0].grid(alpha=0.3)

                axes[i, 1].bar(range(best_K), beta_opt[:best_K],
                               color='grey', edgecolor='k', linewidth=0.5)
                axes[i, 1].axhline(0, color='k', lw=0.8, ls='--')
                axes[i, 1].set_xlabel(x_label_beta)
                axes[i, 1].set_ylabel(y_label_beta)
                axes[i, 1].set_title(f'{title_B} — {season} (K={best_K})')
                axes[i, 1].grid(alpha=0.3, axis='y')

            plt.tight_layout()
            plt.show()

        return results

    def Dspi_free(self, streamflow, Kmax=None, plot=True,
                  agg=None, seasons=None):
        """
        Benchmark B-multi: Linear convolution of standardized anomalies via OLS.

        This benchmark estimates the relationship between precipitation anomalies at
        increasing temporal scales and standardized streamflow (SQI1) by fitting:

            SQI1(t) ≈ Σ_{k=1}^{K} h(k) · SPI_k(t) + intercept

        where SPI_k(t) is the k-scale SPI-like index at time t (from self.spi_like_set),
        and SQI1(t) = streamflow.spi_like_set[0] is the standardized streamflow index
        at scale 1. Weights h(k) are estimated freely via OLS.

        Unlike Benchmark A, there is no temporal lag: all predictors are evaluated at
        the same time t. The hydrological memory is implicitly encoded in the SPI
        accumulation scales — SPI_3(t) already contains the standardized anomaly of
        the last 3 months of precipitation. Adding SPI_k as a predictor is equivalent
        to asking: "does the k-month accumulated anomaly carry additional information
        about streamflow beyond shorter scales?"

        For each K in [1, ..., Kmax], a separate OLS model is fitted using
        [SPI_1(t), ..., SPI_K(t)] as predictors. The optimal K is selected at the
        end of the initial monotonic-increasing plateau of R²_adj vs K (see
        `_monotonic_plateau`), to avoid distant peaks induced by overfitting or
        numerical noise at large K.

        Note: since SPI_k is a moving average of SPI_1, predictors are structurally
        correlated — multicollinearity increases with K. R²_adj remains valid for
        model selection; individual beta coefficients may be unstable at large K.

        Parameters
        ----------
        streamflow : BaseDroughtAnalysis
            Streamflow object with .spi_like_set and .m_cal attributes.
        Kmax : int, optional
            Maximum number of SPI scales to test. Defaults to self.K.
        plot : bool, optional
            If True, plots R²_adj vs K and the estimated weights h(k).
            Default is True.
        agg : {'quarter', 'semiannual', 'four-monthly', 'monthly', 'custom'} or None
            Temporal aggregation scheme for seasonal analysis.
            If None (default), the analysis is run on all data together.
        seasons : dict, optional
            Custom month mapping when ``agg='custom'`` or when passing a dict
            directly (sets agg='custom' automatically). If None and agg is None,
            no seasonal split is performed.

        Returns
        -------
        If seasons is None and agg is None:
            dict with keys:
                'R2_adj'          : np.ndarray, shape (Kmax,)
                'optimal_K'       : int   (first plateau index, 1-based)
                'K_range'         : np.ndarray
                'beta'            : np.ndarray, shape (optimal_K + 1,)
                'condition_number': float
                'metrics'         : dict
        Else:
            dict keyed by season name, each value being the same dict as above,
            with an additional key 'sample_number'.

        Notes
        -----
        - Requires temporal overlap between self.m_cal and streamflow.m_cal.
        - NaN values in either SQI1 or any SPI_k are excluded row-wise.
        - Kmax is capped at min(self.K, streamflow.spi_like_set.shape[0]).
        - `optimal_K` selection follows the plateau criterion (first end of
          monotonic-increasing sequence in R²_adj), not the global maximum.

        References
        ----------
        McKee, T.B., Doesken, N.J., Kleist, J. (1993). The relationship of drought
            frequency and duration to time scales. Proc. 8th Conference on Applied
            Climatology, 179-183.
        Van Loon, A.F. (2015). Hydrological drought explained.
            WIREs Water, 2(4), 359-392. doi:10.1002/wat2.1085
        """
        self._check_correlation_eligible()

        if Kmax is None:
            Kmax = self.K
        Kmax = min(Kmax, self.spi_like_set.shape[0], streamflow.spi_like_set.shape[0])

        if seasons is not None:
            agg = 'custom'

        # --- Temporal overlap ---
        self_indices, streamflow_indices = find_overlap(self.m_cal, streamflow.m_cal)
        if len(self_indices) == 0 or len(streamflow_indices) == 0:
            raise ValueError("No overlapping data found between the two objects.")

        SQI1 = streamflow.spi_like_set[0, streamflow_indices]
        SPIset = self.spi_like_set[:, self_indices]
        m_cal_overlap = np.array([self.m_cal[i] for i in self_indices])
        months_overlap = np.array([m[0] for m in m_cal_overlap], dtype=int)

        K_range = np.arange(1, Kmax + 1)

        def _monotonic_plateau(x):
            """
            Returns the index of the last element in the initial monotonic
            increasing sequence of x. Used to identify the plateau of R²_adj
            vs K, stopping before overfitting or numerical noise takes over.
            """
            last = 0
            for i in range(1, len(x)):
                if np.isfinite(x[i]) and x[i] >= x[i - 1]:
                    last = i
                else:
                    break
            return last

        def _run_dspi_free(SPIset_sub, SQI1_sub, season_mask=None):
            """OLS multi-scale regression for a given data subset.
            Returns the full result dict. Same logic as the all-data case;
            `season_mask` restricts the rows entering the OLS fit."""
            MatCorr = []

            for K in K_range:
                # predictors: [SPI_1(t), SPI_2(t), ..., SPI_K(t)]
                X_block = SPIset_sub[0:K, :].T  # shape (n, K)

                valid = ~np.isnan(SQI1_sub)
                for k in range(K):
                    valid &= ~np.isnan(SPIset_sub[k, :])
                if season_mask is not None:
                    valid &= season_mask

                if valid.sum() < K + 2:
                    MatCorr.append(np.nan)
                    continue

                X = np.column_stack([X_block[valid], np.ones(valid.sum())])
                y = SQI1_sub[valid]

                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                y_hat = X @ beta
                ss_res = np.sum((y - y_hat) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                n_v, p = valid.sum(), K
                MatCorr.append(1 - (1 - r2) * (n_v - 1) / (n_v - p - 1))

            MatCorr = np.array(MatCorr)
            if np.all(np.isnan(MatCorr)):
                return None

            # === FIX: plateau-based selection instead of global argmax ===
            # Take the end of the initial monotonic-increasing run as optimal K.
            # This avoids distant peaks driven by overfitting/numerical noise
            # at large K, where condition_number explodes and beta becomes
            # unstable but R²_adj may still spuriously creep up.
            plateau_idx = _monotonic_plateau(MatCorr)
            best_K = int(K_range[plateau_idx])
            # =============================================================

            # Recompute at optimal K
            X_block_opt = SPIset_sub[0:best_K, :].T
            valid_opt = ~np.isnan(SQI1_sub)
            for k in range(best_K):
                valid_opt &= ~np.isnan(SPIset_sub[k, :])
            if season_mask is not None:
                valid_opt &= season_mask

            X_opt = np.column_stack([X_block_opt[valid_opt], np.ones(valid_opt.sum())])
            y_opt = SQI1_sub[valid_opt]
            beta_opt, _, _, sv = np.linalg.lstsq(X_opt, y_opt, rcond=None)
            cond_num = float(np.max(sv) / np.min(sv[sv > 0])) if len(sv) > 0 else np.nan

            if cond_num > 30:
                import warnings
                warnings.warn(
                    f"Dspi_free: condition number = {cond_num:.1f} > 30. "
                    f"SPI scales are structurally correlated — "
                    f"beta coefficients unstable at K={best_K}.",
                    UserWarning
                )

            y_pred_opt = X_opt @ beta_opt
            metrics = self._eval_metrics(y_opt, y_pred_opt)

            return {
                'R2_adj': MatCorr,
                'optimal_K': best_K,
                'K_range': K_range,
                'beta': beta_opt,
                'condition_number': cond_num,
                'metrics': metrics,
            }

        # ---------------------------------------------------------------
        # CASE 1: no seasonal split
        # ---------------------------------------------------------------
        if agg is None:
            print("Starting correlation analysis...")
            result = _run_dspi_free(SPIset, SQI1)

            if plot and result is not None:
                MatCorr = result['R2_adj']
                best_K = result['optimal_K']
                beta_opt = result['beta']

                fig, axes = plt.subplots(1, 2, figsize=(10, 3))
                axes[0].plot(K_range, MatCorr, 'k-o', lw=2)
                axes[0].axvline(best_K, color='r', ls='--', label=f'optimal K={best_K}')
                axes[0].set_xlabel('K (number of SPI scales)')
                axes[0].set_ylabel('R²_adj')
                axes[0].set_title('Dspi_free — SPI multi-scale OLS')
                axes[0].legend()
                axes[0].grid(alpha=0.3)

                axes[1].bar(K_range[:best_K], beta_opt[:-1],
                            color='grey', edgecolor='k', linewidth=0.5)
                axes[1].axhline(0, color='k', lw=0.8, ls='--')
                axes[1].set_xlabel('SPI scale k')
                axes[1].set_ylabel('h(k) — beta OLS')
                axes[1].set_title(f'SPI scale weights estimated by OLS (K={best_K})')
                axes[1].grid(alpha=0.3, axis='y')

                plt.tight_layout()
                plt.show()

            return result

        # ---------------------------------------------------------------
        # CASE 2: seasonal split
        # ---------------------------------------------------------------
        seasons_dict = self._build_seasons_dict(agg, seasons)

        # year coverage check (same guard as analyze_correlation_seasonal)
        all_months = [m for lst in seasons_dict.values() for m in lst]
        unique = set(all_months)
        missing = [m for m in range(1, 13) if m not in unique]
        duplicates = [m for m in unique if all_months.count(m) > 1]
        if missing:
            print("Alert: missing month(s):", missing)
        if duplicates:
            print("Alert: duplicate month(s):", duplicates)

        print("Starting correlation analysis (seasonal)...")
        results = {}
        for name, mlist in seasons_dict.items():
            idx = np.isin(months_overlap, mlist)
            if np.count_nonzero(idx) <= Kmax + 2:
                print(f" Season {name}: insufficient data, skipped.")
                continue
            res = _run_dspi_free(SPIset, SQI1, season_mask=idx)
            if res is None:
                print(f" Season {name}: all-NaN R² grid, skipped.")
                continue
            res['sample_number'] = int(np.count_nonzero(idx))
            print(f" Season {name}: best R²_adj={res['R2_adj'][res['optimal_K'] - 1]:.3f} "
                  f"(K={res['optimal_K']})")
            results[name] = res

        if plot:
            n_seasons = len(results)
            if n_seasons == 0:
                return results

            fig, axes = plt.subplots(nrows=n_seasons, ncols=2,
                                     figsize=(10, 3 * n_seasons))
            if n_seasons == 1:
                axes = axes[np.newaxis, :]

            for i, (season, res) in enumerate(results.items()):
                MatCorr = res['R2_adj']
                best_K = res['optimal_K']
                beta_opt = res['beta']

                axes[i, 0].plot(K_range, MatCorr, 'k-o', lw=2)
                axes[i, 0].axvline(best_K, color='r', ls='--', label=f'optimal K={best_K}')
                axes[i, 0].set_xlabel('K (number of SPI scales)')
                axes[i, 0].set_ylabel('R²_adj')
                axes[i, 0].set_title(f'Dspi_free — SPI multi-scale OLS — {season}')
                axes[i, 0].legend()
                axes[i, 0].grid(alpha=0.3)

                axes[i, 1].bar(K_range[:best_K], beta_opt[:-1],
                               color='grey', edgecolor='k', linewidth=0.5)
                axes[i, 1].axhline(0, color='k', lw=0.8, ls='--')
                axes[i, 1].set_xlabel('SPI scale k')
                axes[i, 1].set_ylabel('h(k) — beta OLS')
                axes[i, 1].set_title(f'SPI scale weights — {season} (K={best_K})')
                axes[i, 1].grid(alpha=0.3, axis='y')

            plt.tight_layout()
            plt.show()

        return results

        #--------------------------------------------------------------------------------
    # HELPER: Nash-IUH kernel  (parametric, 2 free params)

    # HELPER: linear convolution R²
    @staticmethod
    def _eval_metrics(y_obs: np.ndarray, y_sim: np.ndarray) -> dict:
        """
        Compute a set of complementary skill metrics on a pair of observed
        and simulated standardized series. NaN-aware (row-wise).

        Returns
        -------
        dict with keys:
            'R2'        : coefficient of determination (Pearson, squared)
            'RMSE'      : root mean square error
            'MAE'       : mean absolute error
            'KGE'       : Kling–Gupta efficiency (Gupta et al., 2009)
            'bias'      : mean(sim) - mean(obs)
            'POD_drought' : Probability of Detection for drought events
                            (obs ≤ -1 AND sim ≤ -1) / (obs ≤ -1)
            'FAR_drought' : False Alarm Ratio for drought events
                            (sim ≤ -1 AND obs > -1) / (sim ≤ -1)
            'n'         : sample size used for the computation
        Notes
        -----
        Drought-event metrics use the WMO moderate-drought threshold (-1)
        on the standardized index. They are reported as complementary
        diagnostics to R², which by construction does not penalize
        misclassification of tail events.
        """
        mask = np.isfinite(y_obs) & np.isfinite(y_sim)
        if mask.sum() < 5:
            return {k: np.nan for k in
                    ['R2', 'RMSE', 'MAE', 'KGE', 'bias',
                     'POD_drought', 'FAR_drought']} | {'n': int(mask.sum())}

        o = y_obs[mask]
        s = y_sim[mask]

        # R² (squared Pearson, == 1 - SS_res/SS_tot for unbiased linear fit)
        ss_res = np.sum((o - s) ** 2)
        ss_tot = np.sum((o - o.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        rmse = np.sqrt(np.mean((o - s) ** 2))
        mae = np.mean(np.abs(o - s))
        bias = float(s.mean() - o.mean())

        # KGE (Gupta et al. 2009): 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)
        if o.std() > 0 and s.std() > 0:
            r = np.corrcoef(o, s)[0, 1]
            alpha = s.std() / o.std()
            beta = s.mean() / o.mean() if abs(o.mean()) > 1e-9 else np.nan
            # for standardized series obs.mean() ~ 0 → beta is unstable;
            # use additive bias formulation instead
            beta_add = (s.mean() - o.mean())  # already standardized scale
            kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + beta_add ** 2)
        else:
            kge = np.nan

        # Drought-event metrics (threshold -1 on standardized scale)
        obs_dr = o <= -1
        sim_dr = s <= -1
        if obs_dr.sum() > 0:
            pod = float(np.sum(obs_dr & sim_dr) / obs_dr.sum())
        else:
            pod = np.nan
        if sim_dr.sum() > 0:
            far = float(np.sum(sim_dr & ~obs_dr) / sim_dr.sum())
        else:
            far = np.nan

        return {
            'R2': float(r2),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'KGE': float(kge),
            'bias': bias,
            'POD_drought': pod,
            'FAR_drought': far,
            'n': int(mask.sum()),
        }

    @staticmethod
    def _convolve_r2(driver: np.ndarray, target: np.ndarray,
                     kernel: np.ndarray, season_mask: np.ndarray = None) -> float:
        """
        Convolve driver with kernel, fit intercept via OLS, return R².
        NaN rows are excluded.
        """
        n = len(driver)
        K = len(kernel)
        y_hat = np.full(n, np.nan)
        for t in range(K - 1, n):
            # the weighted sum of past rainfall:
            # sum ( kernel x last K monhts of precipitation) == scalar product with np.dot
            y_hat[t] = np.dot(kernel, driver[t - K + 1: t + 1][::-1])

        # apply seasonal mask AFTER convolution on full series
        if season_mask is not None:
            y_hat = y_hat[season_mask]
            target = target[season_mask]

        mask = np.isfinite(target) & np.isfinite(y_hat)
        if mask.sum() < 5:
            return np.nan
        # fit a moel where X is the weighted sum of last precipitation + intercept
        X = np.column_stack([y_hat[mask], np.ones(mask.sum())])
        y = target[mask]
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        # y - (X @ beta) == observed-simulated == residuals!
        # (X * beta).sum(axis=1) == X @ beta  # True
        sim = X @ beta
        ss_res = np.sum((y - sim) ** 2) # sum of squares
        ss_tot = np.sum((y - y.mean()) ** 2)
        rsquare = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        return rsquare

    @staticmethod
    def _nash_kernel(n: float, k: float, K: int) -> np.ndarray:
        """
        Discrete Nash IUH kernel (Gamma pdf sampled at lags 1..L).
        The result is a weightingh scheme woth bell-shape

        Parameters
        ----------
        n : float   shape parameter  (> 0)
        k : float   scale parameter  (months, > 0)
        K : int     kernel length (months)

        Returns
        -------
        h : ndarray, shape (K,), sums to 1
        """
        from scipy.stats import gamma as gamma_dist
        t = np.arange(1, K + 1, dtype=float)
        h = gamma_dist.pdf(t, a=n, scale=k)
        s = h.sum()
        # /s serves to nornalize thw weights!
        return h / s if s > 0 else np.ones(K) / K

    # --------------------------------------------------------------------------------
    # METHOD 1 — benchmark_nash
    def benchmark_nash(self, streamflow, K=None, plot=True,
                       agg=None, seasons=None):
        """
        This method predicts the standardized streamflow anomaly SQI_1(t) from raw precipitation P(t-j)
        via a parametric kernel. Standardization of the target makes R² directly comparable to D(SPI).
        The kernel shape retains its physical interpretation;
        amplitude is absorbed by the OLS intercept and slope during fitting.

        Nash IUH benchmark: parametric convolution with a Gamma kernel.

        Idea: The discharge at time t is the weighted sum of past precipitation, where the weights form a lagged bell-shaped curve—recent
        rainfall matters little (the basin must "fill"), rainfall from a few months ago matters more,
        and rainfall from a long time ago matters little (memory runs out) >> GAMMA SHAPE!
        Nash proposed modeling this curve with a discrete Gamma distribution, characterized by two physical parameters:

        n — bell-shaped (small n = fast and asymmetric response; large n = slow and symmetric response)
        k — time scale in months (larger k = longer memory)

        Streamflow is modelled as:

            SQI1(t) = intercept + Σ_{j=0}^{L-1} h(j; n, k) · P(t-j)

        where h is the discrete Nash IUH (Gamma pdf), and the two shape
        parameters (n, k) plus the kernel length L are jointly optimised
        to maximise R² against observed monthly streamflow.



        Optimisation strategy
        ---------------------
        For each L in [2, ..., K], scipy.optimize.minimize (Nelder-Mead)
        is run from multiple starting points to find the global optimum
        in (n, k). The best (K, n, k) triplet is retained.

        This is the standard implementation of the Nash (1957) IUH for
        monthly data: only P (raw precipitation) and Q are required.

        Parameters
        ----------
        streamflow : BaseDroughtAnalysis
            Streamflow object with .ts and .m_cal attributes.
        K : int, optional
            Maximum kernel length (months) to test. Defaults to self.K.
        plot : bool, default True
            Plot R² surface and best-fit scatter.
        agg : str or None
            Seasonal aggregation scheme (same API as benchmark_convolution).
        seasons : dict, optional
            Custom season dict (sets agg='custom').

        Returns
        -------
        dict (no seasonal split) or dict-of-dicts (seasonal split) with keys:
            'optimal_L'  : int
            'optimal_n'  : float
            'optimal_k'  : float
            'R2'         : float
            'kernel'     : ndarray, shape (optimal_L,)
            'sample_number' : int  (only in seasonal mode)

        References
        ----------
        Nash, J.E. (1957). The form of the instantaneous unit hydrograph.
            IAHS Publ., 45(3), 114-121.
        """
        from scipy.optimize import minimize

        self._check_correlation_eligible()

        if K is None:
            K = self.K
        if seasons is not None:
            agg = 'custom'

        # --- temporal overlap ---
        self_idx, sf_idx = find_overlap(self.m_cal, streamflow.m_cal)
        if len(self_idx) == 0:
            raise ValueError("No overlapping data found.")

        driver = self.ts[self_idx]
        target = streamflow.spi_like_set[0, sf_idx]
        m_cal_overlap = np.array([self.m_cal[i] for i in self_idx])
        months_overlap = np.array([m[0] for m in m_cal_overlap], dtype=int)

        def _optimize_nash(driver, target, season_mask=None):
            """Optimise Nash IUH for a given data subset.
            the challange is figure out the two paramters definish the best gamma-shaped kernel
            testing incresing kernle from 1 to K"""



            best = {'R2': -np.inf, 'L': None, 'n': None, 'k': None}

            for L in range(2, K + 1):
                def neg_r2(params):
                    """Scipy minimize look to the minima!
                    we pass -R² so the minimum corresponds to the maximum of R².
                    """
                    n_, k_ = params
                    if n_ <= 0 or k_ <= 0 or n_ > 20 or k_ > K:
                        return 1.0  # infeasible
                    # given possible n_ and k_, build the bell-shaped kernel of lenght L,
                    h = self._nash_kernel(n_, k_, L)
                    # now fit the convolutional model and estimate the R2
                    r2 = self._convolve_r2(driver, target, h,season_mask=season_mask)
                    return -r2 if np.isfinite(r2) else 1.0

                # grid of starting points: (n, k) ∈ {1,2,4} × {1,2,4}
                # Starting grid (n, k) ∈ {1,2,4} × {1,2,4}: covers the physically plausible
                # space for monthly data. n=1 is a pure exponential (reactive basin), n=4 is
                # a broad bell (slow/inertial basin); k=1..4 months spans typical Mediterranean
                # memory scales. The grid is intentionally coarse — Nelder-Mead refines locally
                # from each starting point, so we only need seeds spread across the domain,
                # not a dense coverage. For basins with very long memory (k>>4, e.g. deep
                # aquifers), consider replacing k_grid with [1.0, Lmax/8, Lmax/4].
                for n0 in [1.0, 2.0, 4.0]:
                    for k0 in [1.0, 2.0, 4.0]:
                        # Minimize moves [n, k] (the params defininf the shape of the bell-kernle
                        # in the parameter space looking for where -R² is lowest
                        res = minimize(neg_r2, x0=[n0, k0],
                                       method='Nelder-Mead',
                                       options={'xatol': 1e-4, 'fatol': 1e-4,
                                                'maxiter': 2000})
                        r2 = -res.fun
                        if r2 > best['R2']:
                            best.update({'R2': r2, 'L': L,
                                         'n': res.x[0], 'k': res.x[1]})

            #
            kernel = self._nash_kernel(best['n'], best['k'], best['L'])
            print(f"  Nash IUH: R²={best['R2']:.3f}  "
                  f"L={best['L']}  n={best['n']:.2f}   τ={best['k']:.2f}")

            # After best is identified, recompute simulated series for metrics
            kernel = self._nash_kernel(best['n'], best['k'], best['L'])

            # Replicate _convolve_r2 logic to obtain sim aligned with target
            y_hat = np.full(len(driver), np.nan)
            for t in range(best['L'] - 1, len(driver)):
                y_hat[t] = np.dot(kernel, driver[t - best['L'] + 1: t + 1][::-1])

            if season_mask is not None:
                y_hat_eval = y_hat[season_mask]
                target_eval = target[season_mask]
            else:
                y_hat_eval = y_hat
                target_eval = target

            # OLS rescaling (same as _convolve_r2) to match target scale
            mask_v = np.isfinite(y_hat_eval) & np.isfinite(target_eval)
            X = np.column_stack([y_hat_eval[mask_v], np.ones(mask_v.sum())])
            b, _, _, _ = np.linalg.lstsq(X, target_eval[mask_v], rcond=None)
            sim_rescaled = np.full_like(y_hat_eval, np.nan, dtype=float)
            sim_rescaled[mask_v] = X @ b

            metrics = self._eval_metrics(target_eval, sim_rescaled)

            return {
                'optimal_K': best['L'],
                'optimal_n': best['n'],
                'optimal_k': best['k'],
                'R2': best['R2'],
                'kernel': kernel,
                'metrics': metrics,
            }

        # ── no seasonal split ──────────────────────────────────────────────────
        if agg is None:
            print("Starting Nash IUH benchmark...")
            # overall best triplet (K*, n*, k*).
            # K* optimal lenght of the kernel (how many months we have to look backwards)
            # n*,k* = optimal two-params defining the gamma-shape of the kernel
            result = _optimize_nash(driver, target)

            if plot:
                K = result['optimal_K']
                h = result['kernel']
                mask = np.isfinite(target)
                y_hat = np.full(len(driver), np.nan)
                for t in range(K - 1, len(driver)):
                    y_hat[t] = np.dot(h, driver[t - K + 1: t + 1][::-1])
                # fit intercept for plot
                vm = np.isfinite(y_hat) & mask
                X = np.column_stack([y_hat[vm], np.ones(vm.sum())])
                beta, _, _, _ = np.linalg.lstsq(X, target[vm], rcond=None)

                fig, axes = plt.subplots(1, 2, figsize=(10, 3))
                lags = np.arange(1, K + 1)
                axes[0].bar(lags, h, color='grey', edgecolor='k', lw=0.5)
                axes[0].set_xlabel('Lag (months)')
                axes[0].set_ylabel('Weight h(j)')
                axes[0].set_title(
                    f'Nash IUH kernel  (n={result["optimal_n"]:.2f}, '
                    f' τ={result["optimal_k"]:.2f}, K={K})')
                axes[0].grid(alpha=0.3, axis='y')

                Q_fit = X @ beta
                axes[1].plot(Q_fit, target[vm], 'o', color='grey',
                             alpha=0.5, label=f'R²={result["R2"]:.3f}')
                axes[1].plot(
                    [target[vm].min(), target[vm].max()],
                    [target[vm].min(), target[vm].max()],
                    '--', color='grey')
                axes[1].set_xlabel('SQI1 simulated')
                axes[1].set_ylabel('SQI1 observed')
                axes[1].set_title('Nash IUH — best fit')
                axes[1].legend()
                axes[1].grid(alpha=0.3)
                plt.tight_layout()
                plt.show()

            return result

        # ── seasonal split ─────────────────────────────────────────────────────
        seasons_dict = self._build_seasons_dict(agg, seasons)
        all_months = [m for lst in seasons_dict.values() for m in lst]
        missing = [m for m in range(1, 13) if m not in all_months]
        if missing:
            print("Alert: missing month(s):", missing)

        print("Starting Nash IUH benchmark (seasonal)...")
        results = {}
        for name, mlist in seasons_dict.items():
            idx = np.isin(months_overlap, mlist)
            if np.count_nonzero(idx) < 10:
                print(f"  Season {name}: insufficient data, skipped.")
                continue
            print(f"  Season {name}:", end=' ')
            res = _optimize_nash(driver, target, idx)
            res['sample_number'] = int(np.count_nonzero(idx))
            results[name] = res

        if plot and results:
            n_seasons = len(results)
            fig, axes = plt.subplots(nrows=n_seasons, ncols=2,
                                     figsize=(10, 3 * n_seasons))
            if n_seasons == 1:
                axes = axes[np.newaxis, :]
            for i, (season, res) in enumerate(results.items()):
                K = res['optimal_K']
                h = res['kernel']
                lags = np.arange(1, K + 1)
                axes[i, 0].bar(lags, h, color='grey',
                               edgecolor='k', lw=0.5)
                axes[i, 0].set_xlabel('Lag (months)')
                axes[i, 0].set_ylabel('Weight h(j)')
                axes[i, 0].set_title(
                    f'{season} — Nash IUH  '
                    f'(n={res["optimal_n"]:.2f}, τ={res["optimal_k"]:.2f}, '
                    f'K={K})')
                axes[i, 0].grid(alpha=0.3, axis='y')

                idx = np.isin(months_overlap, seasons_dict[season])
                # convolve on the full contiguous series, then filter by season
                y_hat_full = np.full(len(driver), np.nan)
                for t in range(K - 1, len(driver)):
                    y_hat_full[t] = np.dot(h, driver[t - K + 1: t + 1][::-1])
                vm = np.isfinite(y_hat_full[idx]) & np.isfinite(target[idx])
                # riapplica OLS per riscalare y_hat come fa _convolve_r2
                X = np.column_stack([y_hat_full[idx][vm], np.ones(vm.sum())])
                beta, _, _, _ = np.linalg.lstsq(X, target[idx][vm], rcond=None)
                Q_fit = X @ beta
                axes[i, 1].plot(Q_fit, target[idx][vm], 'o',
                                color='grey', alpha=0.5,
                                label=f'R²={res["R2"]:.3f}')

                axes[i, 1].plot(
                    [target[idx][vm].min(), target[idx][vm].max()],
                    [target[idx][vm].min(), target[idx][vm].max()],
                    '--', color='grey')
                axes[i,1].set_xlabel('SQI1 simulated')
                axes[i,1].set_ylabel('SQI1 observed')
                axes[i, 1].set_title(f'Nash IUH | {season} — best fit')
                axes[i, 1].legend()
                axes[i, 1].grid(alpha=0.3)

            plt.show()
            plt.tight_layout()


        return results

    # ──────────────────────────────────────────────────────────────────────────────
    # METHOD 2 — benchmark_ihacres
    # ──────────────────────────────────────────────────────────────────────────────
    def benchmark_ihacres(self, streamflow, K=None, plot=True,
                          agg=None, seasons=None):
        """
        This method predicts the standardized streamflow anomaly SQI_1(t) from raw precipitation P(t-j)
        via a parametric kernel. Standardization of the target makes R² directly comparable to D(SPI).
        The kernel shape retains its physical interpretation;
        amplitude is absorbed by the OLS intercept and slope during fitting.

        IHACRES-inspired two-component benchmark (fast + slow routing).
        Idea: The basin response is not a single process but the superposition
        of two components with different velocities:
        IHACRES uses  two decreasing exponentials, mixed in proportion "alpha".

        Thus, monthly streamflow is modelled as the weighted sum of two linear
        reservoir responses convolved with raw precipitation:

            SQI1(t) = alpha * Q_fast(t) + (1 - alpha) * Q_slow(t) + intercept

        e.g.,   alpha = 0.8 → basin dominated by rapid response.
                alpha = 0.2 → basin dominated by baseflow

        where:
            Q_fast(t) = Σ_{j=0}^{K-1} h_fast(j; tau_f) · P(t-j)
            Q_slow(t) = Σ_{j=0}^{K-1} h_slow(j; tau_s) · P(t-j)

        and h(j; tau) = exp(-j/tau) / Σ exp(-i/tau)  is a single-reservoir
        (exponential) kernel — the Nash IUH with n=1.

        Free parameters: tau_f, tau_s, alpha  (3 params total, K fixed by search).
        Constraints: 0 < tau_f < tau_s,  0 < alpha < 1.

        Physical interpretation
        -----------------------
        This is the linear routing module of IHACRES (Jakeman & Hornberger,
        1993) applied directly to monthly P without the non-linear loss
        module (which requires temperature or PET). At monthly scale the
        loss non-linearity is secondary, and R² relative to the free-kernel
        OLS benchmark captures the residual unexplained variance attributable
        to ET and storage non-linearities.

        Parameters
        ----------
        streamflow : BaseDroughtAnalysis
            Streamflow object with .ts and .m_cal attributes.
        K: int, optional
            Maximum kernel length (months). Defaults to self.K.
        plot : bool, default True
        agg : str or None
            Seasonal aggregation (same API as benchmark_convolution).
        seasons : dict, optional

        Returns
        -------
        dict (or dict-of-dicts in seasonal mode) with keys:
            'optimal_K'    : int
            'optimal_tau_f': float   fast time constant (months)
            'optimal_tau_s': float   slow time constant (months)
            'optimal_alpha': float   fast-component weight
            'R2'           : float
            'kernel_fast'  : ndarray
            'kernel_slow'  : ndarray
            'sample_number': int  (seasonal mode only)

        References
        ----------
        Jakeman, A.J., Hornberger, G.M. (1993). How much complexity is
            warranted in a rainfall-runoff model?
            Water Resources Research, 29(8), 2637-2649.
        """
        self._check_correlation_eligible()


        if K is None:
            K = self.K
        if seasons is not None:
            agg = 'custom'

        self_idx, sf_idx = find_overlap(self.m_cal, streamflow.m_cal)
        if len(self_idx) == 0:
            raise ValueError("No overlapping data found.")

        driver = self.ts[self_idx]
        target = streamflow.spi_like_set[0, sf_idx]
        m_cal_overlap = np.array([self.m_cal[i] for i in self_idx])
        months_overlap = np.array([m[0] for m in m_cal_overlap], dtype=int)

        def _exp_kernel(tau: float, K: int) -> np.ndarray:
            """This is the Nash IUH with n=1—a single linear reservoir.
            The response decays exponentially with time constant τ.
            With small τ it decays quickly (fast), with large τ it decays slowly (slow)."""
            j = np.arange(K, dtype=float)
            h = np.exp(-j / tau)
            # the resulting weights must be normalized:
            return h / h.sum()

        def _run_ihacres(drv, tgt, season_mask=None):
            """Optimise IHACRES two-component model for a given data subset.

            Fix 1: vectorized convolution via np.convolve (≈50-100× faster).
            Fix 2: reduced multi-start grid (3 physically-motivated seeds).
            Fix 3: hard bounds on tau_f, tau_s, alpha via L-BFGS-B instead of
                   Nelder-Mead with penalty. Prevents collapse of tau_f → 0 and
                   ensures tau_s > tau_f + delta_min.
            """
            from scipy.optimize import minimize

            best = {'R2': -np.inf, 'K': None,
                    'tau_f': None, 'tau_s': None, 'alpha': None}

            # --- Hard bounds (physical constraints) ---
            TAU_F_MIN = 0.5  # fast reservoir: at least 0.5 months
            TAU_F_MAX = 6.0  # fast reservoir: at most 6 months
            TAU_S_MIN = 2.0  # slow reservoir: at least 2 months
            # TAU_S_MAX set dynamically as 2*K to allow long memory
            ALPHA_MIN = 0.05  # at least 5% fast component
            ALPHA_MAX = 0.95  # at most 95% fast component
            DELTA_MIN = 1.0  # slow reservoir must exceed fast by at least 1 month

            def neg_r2_factory(L_):
                def neg_r2(params):
                    tau_f, tau_s, alpha = params

                    # Soft separation penalty (additional to bounds, keeps tau_s > tau_f)
                    if tau_s - tau_f < DELTA_MIN:
                        return 1.0 + (DELTA_MIN - (tau_s - tau_f))

                    h_f = _exp_kernel(tau_f, L_)
                    h_s = _exp_kernel(tau_s, L_)

                    n = len(drv)
                    Q_fast = np.convolve(drv, h_f[::-1], mode='full')[:n]
                    Q_slow = np.convolve(drv, h_s[::-1], mode='full')[:n]
                    Q_fast[:L_ - 1] = np.nan
                    Q_slow[:L_ - 1] = np.nan
                    Q_sim = alpha * Q_fast + (1 - alpha) * Q_slow

                    if season_mask is not None:
                        Q_sim = Q_sim[season_mask]
                        tgt_loc = tgt[season_mask]
                    else:
                        tgt_loc = tgt

                    mask = np.isfinite(tgt_loc) & np.isfinite(Q_sim)
                    if mask.sum() < 5:
                        return 1.0
                    X = np.column_stack([Q_sim[mask], np.ones(mask.sum())])
                    y = tgt_loc[mask]
                    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    ss_res = np.sum((y - X @ beta) ** 2)
                    ss_tot = np.sum((y - y.mean()) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else -1.0
                    return -r2 if np.isfinite(r2) else 1.0

                return neg_r2

            starting_points = [
                (1.0, 4.0, 0.7),
                (2.0, 8.0, 0.5),
                (2.0, 12.0, 0.3),
            ]

            for L_ in range(2, K + 1):
                neg_r2 = neg_r2_factory(L_)
                # bounds: tau_f ∈ [0.5, 6], tau_s ∈ [2, 2*K], alpha ∈ [0.05, 0.95]
                bounds = [
                    (TAU_F_MIN, TAU_F_MAX),
                    (TAU_S_MIN, 2.0 * K),
                    (ALPHA_MIN, ALPHA_MAX),
                ]
                for tf0, ts0, a0 in starting_points:
                    # Clip starting point within bounds
                    tf0_c = np.clip(tf0, TAU_F_MIN, TAU_F_MAX)
                    ts0_c = np.clip(ts0, max(TAU_S_MIN, tf0_c + DELTA_MIN), 2.0 * K)
                    a0_c = np.clip(a0, ALPHA_MIN, ALPHA_MAX)

                    res = minimize(neg_r2, x0=[tf0_c, ts0_c, a0_c],
                                   method='L-BFGS-B',
                                   bounds=bounds,
                                   options={'ftol': 1e-6, 'maxiter': 300})
                    r2 = -res.fun
                    if r2 > best['R2']:
                        best.update({'R2': r2, 'K': L_,
                                     'tau_f': res.x[0],
                                     'tau_s': res.x[1],
                                     'alpha': res.x[2]})

            h_f = _exp_kernel(best['tau_f'], best['K'])
            h_s = _exp_kernel(best['tau_s'], best['K'])
            print(f"  IHACRES: R²={best['R2']:.3f}  K={best['K']}  "
                  f"τ_fast={best['tau_f']:.2f}  τ_slow={best['tau_s']:.2f}  "
                  f"α={best['alpha']:.2f}")
            return {
                'optimal_K': best['K'],
                'optimal_tau_f': best['tau_f'],
                'optimal_tau_s': best['tau_s'],
                'optimal_alpha': best['alpha'],
                'R2': best['R2'],
                'kernel_fast': h_f,
                'kernel_slow': h_s,
            }
        # ── no seasonal split ──────────────────────────────────────────────────
        if agg is None:
            print("Starting IHACRES benchmark...")
            result = _run_ihacres(driver, target)

            if plot:
                K = result['optimal_K']
                h_f = result['kernel_fast']
                h_s = result['kernel_slow']
                alpha = result['optimal_alpha']
                n = len(driver)
                Q_fast = np.full(n, np.nan)
                Q_slow = np.full(n, np.nan)
                for t in range(K - 1, n):
                    seg = driver[t - K + 1: t + 1][::-1]
                    Q_fast[t] = np.dot(h_f, seg)
                    Q_slow[t] = np.dot(h_s, seg)
                Q_sim = alpha * Q_fast + (1 - alpha) * Q_slow
                mask = np.isfinite(target) & np.isfinite(Q_sim)
                X = np.column_stack([Q_sim[mask], np.ones(mask.sum())])
                beta, _, _, _ = np.linalg.lstsq(X, target[mask], rcond=None)
                Q_fit = X @ beta

                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                lags = np.arange(1, K + 1)
                width = 0.3
                axes[0].bar(lags - width / 2, h_f, width=width, alpha=0.7,
                            color='tab:orange',edgecolor='grey',
                            label=f'fast τ={result["optimal_tau_f"]:.1f}')
                axes[0].bar(lags + width / 2, h_s, width=width, alpha=0.7,
                            color='tab:blue',edgecolor='grey',
                            label=f'slow τ={result["optimal_tau_s"]:.1f}')
                axes[0].set_xlabel('Lag (months)')
                axes[0].set_ylabel('Weight h(j)')
                axes[0].set_title(
                    f'IHACRES kernels  α={result["optimal_alpha"]:.2f}  K={K}')
                axes[0].legend()
                axes[0].grid(alpha=0.3, axis='y')

                axes[1].plot(Q_fit, target[mask], 'o',
                             color='grey', alpha=0.5,
                             label=f'R²={result["R2"]:.3f}')
                axes[1].plot(
                    [target[mask].min(), target[mask].max()],
                    [target[mask].min(), target[mask].max()],
                    '--', color='grey')
                axes[1].set_xlabel('Q simulated')
                axes[1].set_ylabel('Q observed')
                axes[1].set_title('IHACRES — best fit')
                axes[1].legend()
                axes[1].grid(alpha=0.3)
                plt.tight_layout()
                plt.show()

            return result

        # ── seasonal split ─────────────────────────────────────────────────────
        seasons_dict = self._build_seasons_dict(agg, seasons)
        all_months = [m for lst in seasons_dict.values() for m in lst]
        missing = [m for m in range(1, 13) if m not in all_months]
        if missing:
            print("Alert: missing month(s):", missing)

        print("Starting IHACRES benchmark (seasonal)...")
        results = {}
        for name, mlist in seasons_dict.items():
            idx = np.isin(months_overlap, mlist)
            if np.count_nonzero(idx) < 15:
                print(f"  Season {name}: insufficient data, skipped.")
                continue
            print(f"  Season {name}:", end=' ')
            res = _run_ihacres(driver, target, season_mask=idx)
            res['sample_number'] = int(np.count_nonzero(idx))
            results[name] = res

        if plot and results:
            n_seasons = len(results)
            fig, axes = plt.subplots(nrows=n_seasons, ncols=2,
                                     figsize=(14, 5 * n_seasons))
            if n_seasons == 1:
                axes = axes[np.newaxis, :]
            for i, (season, res) in enumerate(results.items()):
                K = res['optimal_K']
                h_f = res['kernel_fast']
                h_s = res['kernel_slow']
                alpha = res['optimal_alpha']
                lags = np.arange(1, K + 1)
                width = 0.3
                axes[i, 0].bar(lags - width / 2, h_f, width=width, alpha=0.7,
                               color='tab:orange', edgecolor='grey',
                               label=f'fast τ={res["optimal_tau_f"]:.1f}')
                axes[i, 0].bar(lags + width / 2, h_s, width=width, alpha=0.7,
                               color='tab:blue', edgecolor='grey',
                               label=f'slow τ={res["optimal_tau_s"]:.1f}')
                axes[i, 0].set_xlabel('Lag (months)')
                axes[i, 0].set_ylabel('Weight h(j)')
                axes[i, 0].set_title(f'{season} — IHACRES  α={alpha:.2f}  K={K}')
                axes[i, 0].legend()
                axes[i, 0].grid(alpha=0.3, axis='y')

                # convolve on full contiguous series, then filter by season
                idx = np.isin(months_overlap, seasons_dict[season])
                Q_fast_full = np.full(len(driver), np.nan)
                Q_slow_full = np.full(len(driver), np.nan)
                for t in range(K - 1, len(driver)):
                    seg = driver[t - K + 1: t + 1][::-1]
                    Q_fast_full[t] = np.dot(h_f, seg)
                    Q_slow_full[t] = np.dot(h_s, seg)
                Q_sim_full = alpha * Q_fast_full + (1 - alpha) * Q_slow_full

                # apply OLS rescaling on seasonal subset
                vm = np.isfinite(Q_sim_full[idx]) & np.isfinite(target[idx])
                X = np.column_stack([Q_sim_full[idx][vm], np.ones(vm.sum())])
                beta, _, _, _ = np.linalg.lstsq(X, target[idx][vm], rcond=None)
                Q_fit = X @ beta

                axes[i, 1].plot(Q_fit, target[idx][vm], 'o',
                                color='grey', alpha=0.5,
                                label=f'R²={res["R2"]:.3f}')
                axes[i, 1].plot(
                    [target[idx][vm].min(), target[idx][vm].max()],
                    [target[idx][vm].min(), target[idx][vm].max()],
                    '--', color='grey')
                axes[i, 1].set_title(f'{season} — best fit')
                axes[i, 1].legend()
                axes[i, 1].grid(alpha=0.3)

            plt.tight_layout()
            plt.show()

        return results

    # =====================================================================
    # =====================================================================
    #  CDN
    def _calculate_CDN(self):
        """
        Compute the Cumulative Deviation from Normal (CDN).

        CDN is the cumulative sum of SPI-1 starting from the baseline period.
        It tracks the long-term memory of standardized anomalies.

        Returns:
            ndarray: CDN values (same length as self.ts). Values before the
            baseline start are set to zero.
        """
        # Get baseline indices and ensemble mean
        tb1_id, tb2_id = baseline_indices(self.m_cal,self.start_baseline_year,self.end_baseline_year)
        spi1 = self.spi_like_set[0].copy()
        # estimate the average to equalize the signal:
        cdn = np.zeros(len(self.ts))
        cdn[tb1_id::] = np.nancumsum(np.round(spi1[tb1_id::],3))#per evitare che si trascina errori
        # base = np.mean(cdn)
        # CDN = cdn-base

        return cdn

    def normal_values(self):
        """
          Compute the "normal" values of the  climatology  using the inverse function of the SPI-like index.

          This method calculates the "normal" values for the variable of interest based on the
          inverse of the SPI-like index at scale 1 (SPI_like_index_1 == 0). It uses the coefficients
          (`self.c2r_index`) from the polynomial fitting of the SPI-like index for each month.
          The normal values are computed for all months and tiled across the entire timeframe.

          Returns
          -------
          numpy.ndarray
              An array of "normal" values corresponding to the timeseries length (`self.ts`).


          """
        Nn = np.zeros(12)
        for m in range(12):
            Nn[m] = np.polyval(self.c2r_index[0,m,:],0)
        Normal = np.tile(np.squeeze(Nn),len(np.unique(self.m_cal[:,1])))
        Normal = Normal[0:len(self.ts)]
        return Normal

    def deficit_from_spi(self, window, spi=None, coeff=None):
        """
        Estimate the cumulative water deficit/surplus over a moving window,
        derived from the SPI-like index at the matching accumulation scale.

        Rather than summing raw monthly anomalies (which, for precipitation, are
        bounded below by zero and unbounded above, producing a systematic positive
        skew), this method maps the anomaly through the SPI-like space. The SPI-like
        index at scale `window` is converted back to native cumulative units via the
        polynomial calibration stored in `c2r_index`, and the deficit/surplus is taken
        relative to the SPI=0 reference (which, by construction, coincides with
        `normal_values()`).

        The resulting deficit is expressed in physical units and reflects the
        statistical rarity of the accumulated anomaly. Note that, because the c2r
        calibration is the non-linear (gamma-based) inverse transform, the deficit in
        native units remains physically asymmetric even though the SPI itself is
        symmetric; this asymmetry is intentional and reflects the real bounded nature
        of the variable.

        Parameters
        ----------
        window : int
            Accumulation scale in months, equal to the moving-window length.
        spi : ndarray, optional
            Pre-computed SPI-like series at scale `window`. If None, it is retrieved
            from `self.spi_like_set` when `window <= self.K`, or computed on the fly
            via `_compute_spi(month_scale=window)` otherwise.
        coeff : ndarray, optional
            Pre-computed c2r calibration coefficients (shape (12, n_coef)) at scale
            `window`. Retrieved or computed together with `spi` when not provided.

        Returns
        -------
        deficit : ndarray, shape (n,)
            Cumulative deficit (negative) or surplus (positive) over the preceding
            `window` months, in native cumulative units: millimetres for
            Precipitation, cubic metres (total volume) for Streamflow. For Streamflow,
            the cumulative sum of monthly mean discharges (m^3/s) returned by the
            calibration is converted to total volume by multiplying by the average
            number of seconds per month. Entries are NaN wherever the SPI-like index
            is undefined (the first `window-1` months, or any gaps propagated from
            missing values in the input series).

        Notes
        -----
        - The SPI=0 reference equals `normal_values()` by construction, so this deficit
          is consistent with the rest of the framework's climatological baseline.
        - For scales exceeding `self.K`, the SPI fit reuses the same baseline period
          (`start_baseline_year`, `end_baseline_year`) as initialisation. At long
          scales the number of independent samples per reference month is small, so
          the gamma fit should be interpreted with care.
        """

        from drought_scan.core import Streamflow

        n = len(self.ts)
        months = self.m_cal[:, 0].astype(int)

        if spi is None or coeff is None:
            if window <= self.K:
                spi = self.spi_like_set[window - 1, :]
                coeff = self.c2r_index[window - 1]
            else:
                spi, coeff = self._compute_spi(month_scale=window)

        deficit = np.full(n, np.nan)
        for x in np.where(~np.isnan(spi))[0]:
            m = months[x] - 1
            deficit[x] = np.polyval(coeff[m], spi[x]) - np.polyval(coeff[m], 0.0)

        # Streamflow: deficit è in m³/s cumulati (somma di window medie mensili).
        # Converti a volume totale m³ moltiplicando per i secondi medi al mese.
        if isinstance(self, Streamflow):
            avg_seconds_per_month = np.mean(
                [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) * 86400
            deficit = deficit * avg_seconds_per_month

        return deficit

    def find_trends(self, var=None, window=None):
        """
        Analyze trends in self.CDN using rolling windows and linear regression.

        Args:
            window  (list of int, optional):   window size  in months.
                Defaults to [60].

        Returns:
            dict: Dictionary containing results for each window size.
                Each entry contains:
                - 'trend': Array with -1 (negative trend), 0 (no trend), 1 (positive trend).
                - 'slope': Array with slope coefficients.
                - 'p_value': Array with p-values.
                - 'delta': Array with the cumulative change (slope * window size).
        """
        from drought_scan.utils.statistics import _rolling_trend_analysis

        # Default to a window size of 60 if none is provided
        if window is None:
            window = 60
        if var is None:
            var = self.CDN
        results = _rolling_trend_analysis(var=var, window=window, significance=0.05)
        return results

    def volume_anomaly_rolling(self, window):
        """
        Compute the observed cumulative water deficit/surplus over a moving window
        by direct summation of monthly anomalies.

        For each time step, the anomaly of every month within the preceding `window`
        months is summed, where the monthly anomaly is the difference between the
        observed value and its climatological normal (`normal_values()`, i.e. the
        SPI=0 reference). This yields the actual physical water balance over the
        window, without any statistical transformation.

        This is the direct, observation-based counterpart to `deficit_from_spi`:
        `volume_anomaly_rolling` measures how many real cubic metres / millimetres
        were missing or in excess (the quantity relevant for water-balance and
        reservoir-management purposes), whereas `deficit_from_spi` measures the
        statistical rarity of the accumulated anomaly. The two are strongly
        correlated but diverge at extremes; for precipitation the raw sum is
        physically asymmetric (bounded below by zero, unbounded above). This method
        is useful as a sanity check and as a complementary metric to the SPI-based
        estimate.

        Parameters
        ----------
        window : int
            Moving-window length in months.

        Returns
        -------
        anomaly : ndarray, shape (n,)
            Cumulative deficit (negative) or surplus (positive) over the preceding
            `window` months. Units are millimetres for Precipitation. For Streamflow,
            each monthly mean discharge anomaly (m^3/s) is multiplied by the number of
            seconds in its calendar month before summation, so the result is a total
            volume in cubic metres. Entries are NaN for the first `window-1` months;
            gaps from missing values in the input series propagate into the sum.

        Notes
        -----
        - The reference (`normal_values()`) equals the SPI=0 native value by
          construction, ensuring consistency with `deficit_from_spi` and the rest of
          the framework's baseline.
        - Unlike `deficit_from_spi`, no SPI fit is involved, so this method is robust
          at any window length but does not correct the bounded-variable asymmetry.
        """
        n = len(self.ts)
        months = self.m_cal[:, 0].astype(int)
        normal = self.normal_values()

        anomaly = np.full(n, np.nan)

        if isinstance(self, Streamflow):
            DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            seconds = np.array([DAYS_IN_MONTH[m - 1] * 86400 for m in months])
            delta_per_month = (self.ts - normal[months - 1]) * seconds  # m³/mese
        else:  # Precipitation
            delta_per_month = self.ts - normal[months - 1]  # mm/mese

        for i in range(window - 1, n):
            anomaly[i] = np.sum(delta_per_month[i - window + 1: i + 1])

        return anomaly
    # =====================================================================
    # other
    # =====================================================================
    def export_scan_plot_csv(self, weight_index=2, optimal_k=None, name=None, out_dir="exports"):
        """
        Exports the minimum data needed in CSV format to replicate the plot_scan in another workspace.

        Args:
            weight_index (int): weight index for the SIDI (default: decreasing log = 2)
            optimal_k (int, optional): if specified, calculates a new SIDI with optimal K
            name (str, optional): name of the basin for filename
            out_dir (str): directory to save the exported data
        """
        os.makedirs(out_dir, exist_ok=True)

        # time series needed
        df_mcal = pd.DataFrame(self.m_cal, columns=["month", "year"])
        df_cdn = pd.DataFrame({"CDN": self.CDN})
        spi_df = pd.DataFrame(self.spi_like_set)

        # SIDI:
        if optimal_k is not None:
            from drought_scan.utils import generate_weights, weighted_metrics  # o path corretto
            weights = generate_weights(k=optimal_k)
            sidis = []
            for j in range(len(self.m_cal)):
                vec = self.spi_like_set[0:optimal_k, j]
                sidis.append([weighted_metrics(vec, weights[:, weight_index])[0]])
            sidi_vec = np.squeeze(np.array(sidis))
        else:
            if self.is_seasonal_sidi:
                print(f"Note: SIDI is seasonally optimized — "
                      f"weight_index={weight_index} is ignored (all columns identical).")
            sidi_vec = self.SIDI[:, weight_index]

        df_sidi = pd.DataFrame({"SIDI": sidi_vec})

        # general metadata
        metadata = {
            "index_name": self.index_name,
            "K": self.K,
            "threshold": float(self.threshold),
            "weight_index": int(weight_index),
            "optimal_k": int(optimal_k) if optimal_k is not None else self.K,
            "start_baseline_year": int(self.start_baseline_year),
            "end_baseline_year": int(self.end_baseline_year)
        }

        if hasattr(self, "shape"):
            metadata["area_kmq"] = self.area_kmq
        else:
            metadata["area_kmq"] = np.nan
        # Salvataggio
        prefix = name.replace(" ", "_") if name else "DS_export"
        df_mcal.to_csv(os.path.join(out_dir, f"{prefix}_m_cal.csv"), index=False)
        df_cdn.to_csv(os.path.join(out_dir, f"{prefix}_cdn.csv"), index=False)
        spi_df.to_csv(os.path.join(out_dir, f"{prefix}_spi.csv"), index=False, header=False)
        df_sidi.to_csv(os.path.join(out_dir, f"{prefix}_sidi.csv"), index=False)

        with open(os.path.join(out_dir, f"{prefix}_meta.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Data exported successfully in {out_dir}/ with prefix '{prefix}'")

    def _savedsplot(self):

        k = self.K if not hasattr(self, 'optimal_k') or self.optimal_k is None else self.optimal_k
        w = self.weight_index if not hasattr(self,
                                             'optimal_weight_index') or self.optimal_weight_index is None else self.optimal_weight_index
        baseline = self.start_baseline_year, self.end_baseline_year
        print(f"saving plot in {os.getcwd()}")

        # check the number of plots:
        figs = plt.get_fignums()
        if len(figs) == 1:
            # --- singola figura ---
            fname = f"DS_{self.basin_name}_k{k}_w{w}_baseline{baseline}.png"
            plt.figure(figs[0]).savefig(
                fname, dpi=300, facecolor='w', edgecolor='w',
                bbox_inches='tight', pad_inches=0.1, metadata=None
            )
            print(f"  -> saved {fname}")

        elif len(figs) == 3:
            # --- three single figures ---
            fnames = [
                f"CDN_{self.basin_name}_baseline{baseline}.png",
                f"HeatMap_{self.basin_name}_baseline{baseline}.png",
                f"SIDI_{self.basin_name}_k{k}_w{w}_baseline{baseline}.png"
            ]
            for fig_num, fname in zip(figs, fnames):
                plt.figure(fig_num).savefig(
                    fname, dpi=300, facecolor='w', edgecolor='w',
                    bbox_inches='tight', pad_inches=0.1, metadata=None
                )
                print(f"  -> saved {fname}")

        else:
            print(f"Warning: unexpected number of plots ({len(figs)}). No files saved.")

    # =====================================================================
    # ▸ STANDARD Methods GRIDDED-LEVEL
    # =====================================================================
    @staticmethod
    def _process_grid_point(ts_ij, m_cal, K, start_baseline_year, end_baseline_year,
                            calculation_method, M_valid, t_idx, n_weights):
        """Standalone computation for a single grid point — picklable by joblib.
         results = Parallel(n_jobs=-1)(
            delayed(self._process_grid_point)(
                self.Pgrid[:, i, j],
                self.m_cal, self.K,
                self.start_baseline_year, self.end_baseline_year,
                self.calculation_method,
                M_valid, t_idx, n_weights
            )
            for (i, j) in valid_ij
        )
        """
        from drought_scan.utils.drought_indices import (
            generate_weights, weighted_metrics, baseline_indices
        )
        try:
            # --- replicate _calculate_spi_like_set ---
            spiset = np.full((K, len(ts_ij)), np.nan, dtype=float)
            for k in range(1, K + 1):
                Spi_ts = np.full_like(ts_ij, np.nan, dtype=float)
                for ref_month in range(1, 13):
                    indices, spi_values, coeff, _ = calculation_method(
                        ts_ij, k, ref_month, m_cal, start_baseline_year, end_baseline_year
                    )
                    if indices is not None and spi_values is not None:
                        Spi_ts[indices] = spi_values.copy()
                spiset[k - 1, :] = Spi_ts

            # --- replicate _spi_like_set_ensemble_mean ---
            weights = generate_weights(K)
            sidi_raw = np.array([
                [weighted_metrics(spiset[:K, j], w)[0] for w in weights.T]
                for j in range(len(m_cal))
            ], dtype=float)

            # --- replicate _calculate_SIDI ---
            tb1_id, tb2_id = baseline_indices(m_cal, start_baseline_year, end_baseline_year)
            baseline_mean = np.nanmean(sidi_raw[tb1_id:tb2_id + 1, :], axis=0)
            baseline_std = np.nanstd(sidi_raw[tb1_id:tb2_id + 1, :], axis=0)
            if np.any(baseline_std == 0):
                return None, None
            SIDI_ij = (sidi_raw - baseline_mean) / baseline_std

            spi_out = {m: spiset[m - 1, t_idx] for m in M_valid}

            return SIDI_ij[t_idx, :], spi_out, None

        except Exception as e:
            return None, None, str(e)  # aggiungi messaggio errore


    def spatial_maps(self, month_scales =None, timestamp=None, K=None):
        """
        Compute gridded SPI at selected temporal scales,SIDI and Trends in CDN for each grid point in Pgrid.

        Args:
            month_scales (list of int, optional): Temporal scales for SPI maps. Defaults to [1, 3, 6, 12, 18, 24].
            timestamp (tuple, optional): (month, year) of target slice. Defaults to last timestamp.
            K (int, optional): Max temporal scale. Overrides self.K for this computation only.

        Stores in self:
            SIDI_grid        : ndarray (n_rows, n_cols, n_weights) — all weighting schemes.
            SPI_grid         : dict {scale: ndarray (n_rows, n_cols)}.
            spatial_timestamp: ndarray (2,) — [month, year] of the stored snapshot.

        Notes:
            Default weight for display: weight_index=2 (see generate_weights convention).
            Access a specific weight slice with: ds.SIDI_grid[:, :, weight_index]
        """
        from joblib import Parallel, delayed, cpu_count

        if month_scales is None:
            month_scales = [1, 3, 6, 12, 18, 24]

        _K_orig = self.K
        if K is None:
            K = self.K
        else:
            self.K = K

        if timestamp is None:
            t_idx = len(self.m_cal) - 1
        else:
            month_t, year_t = timestamp
            matches = np.where((self.m_cal[:, 0] == month_t) & (self.m_cal[:, 1] == year_t))[0]
            if len(matches) == 0:
                raise ValueError(f"Timestamp {timestamp} not found in m_cal.")
            t_idx = int(matches[0])

        M_valid = [m for m in month_scales if m <= self.K]
        if len(M_valid) < len(month_scales):
            import warnings
            warnings.warn(f"Scales {set(month_scales) - set(M_valid)} exceed K={self.K} and will be skipped.")

        n_time, n_rows, n_cols = self.Pgrid.shape
        n_weights = generate_weights(self.K).shape[1]

        # --- pre-mask valid grid points ---
        valid_mask = ~(np.all(np.isnan(self.Pgrid), axis=0) | (np.nanstd(self.Pgrid, axis=0) == 0))
        valid_ij = np.argwhere(valid_mask)
        n_valid = len(valid_ij)

        n_cores = cpu_count()
        t_per_point = 1.4  # secondi stimati dal benchmark
        estimated_min = (n_valid * t_per_point / n_cores) / 60
        print(f"compute_spatial_sidi: {n_valid}/{n_rows * n_cols} valid grid points.")
        print(f"Running on {n_cores} cores — may take up to {estimated_min:.0f} min.")

        # --- parallel computation ---
        results = Parallel(n_jobs=-1)(
            delayed(self._process_grid_point)(
                self.Pgrid[:, i, j],
                self.m_cal, self.K,
                self.start_baseline_year, self.end_baseline_year,
                self.calculation_method,
                M_valid, t_idx, n_weights
            )
            for (i, j) in valid_ij
        )

        # --- reconstruct grids ---
        SIDI_grid = np.full((n_rows, n_cols, n_weights), np.nan)
        SPI_grid = {m: np.full((n_rows, n_cols), np.nan) for m in M_valid}

        # for idx, (i, j) in enumerate(valid_ij):
        #     sidi_vals, spi_vals = results[idx]
        #     if sidi_vals is None:
        #         continue
        #     SIDI_grid[i, j, :] = sidi_vals
        #     for m in M_valid:
        #         SPI_grid[m][i, j] = spi_vals[m]

        n_none = 0
        errors = {}
        for idx, (i, j) in enumerate(valid_ij):
            sidi_vals, spi_vals, err = results[idx]
            if sidi_vals is None:
                n_none += 1
                errors[(i, j)] = err
                continue
            SIDI_grid[i, j, :] = sidi_vals
            for m in M_valid:
                SPI_grid[m][i, j] = spi_vals[m]

        print(f"compute_spatial_sidi: 100% — done. ({n_none}/{len(valid_ij)} points failed)")
        if errors:
            unique_errors = set(errors.values())
            for e in unique_errors:
                n = sum(1 for v in errors.values() if v == e)
                print(f"  {n} points: {e}")

        self.SIDI_grid = SIDI_grid
        self.SPI_grid = SPI_grid
        new_timestamp = self.m_cal[t_idx].copy()
        if hasattr(self, 'spatial_timestamp') and not np.array_equal(new_timestamp, self.spatial_timestamp):
            import warnings
            warnings.warn(
                f"timestamp changed from {self.spatial_timestamp} to {new_timestamp}. "
                f"trend_grid has been invalidated — rerun compute_spatial_trends."
            )
            if hasattr(self, 'trend_grid'):
                del self.trend_grid
        self.spatial_timestamp = new_timestamp
        # self.spatial_timestamp = self.m_cal[t_idx].copy()
        self.K = _K_orig

    @staticmethod
    def _process_grid_point_trends(ts_ij, m_cal, K, start_baseline_year, end_baseline_year,
                                   calculation_method):
        """
        Standalone computation of CDN and std_to_mm for a single grid point — picklable by joblib.

        Returns:
            CDN_ij    : ndarray (n_months,) — cumulative deviation from normal (cumsum of SPI-1)
            std_to_mm : float — mm equivalent of one standardized unit, derived from pixel-level fit
            error     : str or None
        """
        from drought_scan.utils.drought_indices import get_month_indices, baseline_indices
        try:
            # --- SPI-1 for all 12 months ---
            Spi1 = np.full(len(ts_ij), np.nan, dtype=float)
            c2r = np.full((12, 4), np.nan, dtype=float)

            for ref_month in range(1, 13):
                indices, spi_values, coeff, _ = calculation_method(
                    ts_ij, 1, ref_month, m_cal, start_baseline_year, end_baseline_year
                )
                if indices is not None and spi_values is not None:
                    Spi1[indices] = spi_values.copy()
                    c2r[ref_month - 1, :] = coeff.copy()

            # --- CDN: cumsum of SPI-1 from baseline start ---
            tb1_id, tb2_id = baseline_indices(m_cal, start_baseline_year, end_baseline_year)
            CDN_ij = np.nancumsum(Spi1)
            CDN_ij[:tb1_id] = np.nan  # before baseline: undefined

            # --- std_to_mm: pixel-level equivalent of 1 standardized unit ---
            normal_vals = np.array([
                np.polyval(c2r[m, :], 0) for m in range(12)
            ])
            unit_vals = np.array([
                np.polyval(c2r[m, :], 1) for m in range(12)
            ])
            std_to_mm = float(np.nanmean(unit_vals - normal_vals))

            return CDN_ij, std_to_mm, None

        except Exception as e:
            return None, None, str(e)

    def spatial_trends(self, windows=None, timestamp=None):
        """
        Compute pixel-wise CDN trend maps at a target timestamp.

        For each valid grid point, computes the CDN (cumsum of SPI-1) and applies
        rolling trend analysis over each window. The net change (delta) is converted
        to mm-equivalent using pixel-level calibration coefficients.

        Args:
            windows (list of int, optional): Moving window sizes in months.
                                             Defaults to [24, 36, 60, 120].
            timestamp (tuple, optional): (month, year) of target slice.
                                         Defaults to last available timestamp.

        Stores in self:
            trend_grid       : dict {window: ndarray (n_rows, n_cols)} — mm-equivalent
                               net change over each window at the target timestamp.
            spatial_timestamp: ndarray (2,) — [month, year] of the stored snapshot.

        Notes:
            Default weight for display: weight_index=2 (see generate_weights convention).
            If spatial_timestamp already exists and differs from the requested timestamp,
            SIDI_grid and SPI_grid are invalidated — rerun spatial_maps for consistency.
        """
        from joblib import Parallel, delayed, cpu_count
        from drought_scan.utils.statistics import _rolling_trend_analysis

        if windows is None:
            windows = [24, 36, 60, 120]

        # --- resolve timestamp ---
        if timestamp is None:
            t_idx = len(self.m_cal) - 1
        else:
            month_t, year_t = timestamp
            matches = np.where((self.m_cal[:, 0] == month_t) & (self.m_cal[:, 1] == year_t))[0]
            if len(matches) == 0:
                raise ValueError(f"Timestamp {timestamp} not found in m_cal.")
            t_idx = int(matches[0])

        new_ts = self.m_cal[t_idx].copy()

        # --- cross-check with existing spatial_timestamp ---
        if hasattr(self, 'spatial_timestamp') and not np.array_equal(new_ts, self.spatial_timestamp):
            import warnings
            warnings.warn(
                f"timestamp changed from {self.spatial_timestamp} to {new_ts}. "
                f"SIDI_grid and SPI_grid have been invalidated — rerun spatial_maps."
            )
            for attr in ['SIDI_grid', 'SPI_grid']:
                if hasattr(self, attr):
                    delattr(self, attr)

        self.spatial_timestamp = new_ts

        # --- pre-mask valid grid points ---
        n_time, n_rows, n_cols = self.Pgrid.shape
        valid_mask = ~(np.all(np.isnan(self.Pgrid), axis=0) | (np.nanstd(self.Pgrid, axis=0) == 0))
        valid_ij = np.argwhere(valid_mask)
        n_valid = len(valid_ij)

        n_cores = cpu_count()
        t_per_point = 0.5  # SPI-1 only, lighter than full spatial_maps
        estimated_min = (n_valid * t_per_point / n_cores) / 60
        print(f"compute_spatial_trends: {n_valid}/{n_rows * n_cols} valid grid points.")
        print(f"Running on {n_cores} cores — may take up to {estimated_min:.0f} min.")

        # --- parallel computation ---
        results = Parallel(n_jobs=-1)(
            delayed(self._process_grid_point_trends)(
                self.Pgrid[:, i, j],
                self.m_cal, self.K,
                self.start_baseline_year, self.end_baseline_year,
                self.calculation_method
            )
            for (i, j) in valid_ij
        )

        # --- reconstruct trend grids ---
        trend_grid = {w: np.full((n_rows, n_cols), np.nan) for w in windows}

        n_none = 0
        errors = {}
        for idx, (i, j) in enumerate(valid_ij):
            CDN_ij, std_to_mm, err = results[idx]
            if CDN_ij is None:
                n_none += 1
                errors[(i, j)] = err
                continue

            for w in windows:
                R = _rolling_trend_analysis(CDN_ij, window=w, significance=0.05)
                delta = R['delta'][t_idx]
                trend = R['trend'][t_idx]
                trend_grid[w][i, j] = delta * std_to_mm if trend != 0 else 0.0

        print(f"compute_spatial_trends: 100% — done. ({n_none}/{n_valid} points failed)")
        if errors:
            unique_errors = set(errors.values())
            for e in unique_errors:
                n = sum(1 for v in errors.values() if v == e)
                print(f"  {n} points: {e}")

        self.trend_grid = trend_grid
    # =====================================================================
    # ▸ VISUALIZATION
    # =====================================================================
    def plot_scan(self, optimal_k=None, weight_index=None,year_ext=None,split_plot=None,
                  plot_order=None,saveplot=False,figsize=None):
        """
            Plot the drought scan visualization, including CDN, SPI-like heatmap, and SIDI.

            Args:

                optimal_k (int, optional): Optimal K scale.
                weight_index (int, optional): Weighting scheme index.
                year_ext (tuple, optional): Years defining X-axis limits.
                split_plot :   If True, each panel (CDN, Heatmap, SIDI) is plotted in a separate figure,
                plot_order : str, default='CHS';    Order of the panels when split_plot=False.


            """
        plot_overview(self, optimal_k=optimal_k, weight_index=weight_index,year_ext=year_ext,
                      split_plot=split_plot,plot_order=plot_order,figsize=figsize)
        if saveplot==True:
            self._savedsplot()

    def plot_spi_fit(self,K,month,return_data = False):
        """
        Plot the fitted relationship between the SPI values and the raw variable
        (e.g. precipitation, PET, or balance), for a given accumulation scale (K)
        and calendar month.

        Parameters
        ----------
        K : int
            The SPI timescale (number of months) to be plotted, e.g. 3 for SPI-3.
        month : int
            The calendar month (1 = January, ..., 12 = December).

        Returns
        -------
        mm : ndarray, shape (K, len(domain), 12)
            Matrix of fitted values used for plotting, containing the equivalent
            raw values for each SPI domain point, scale, and month.
            Returned only if the function is assigned to a variable
            (e.g. ``mm = precipitation.plot_spi_fit(K=3, month=3)``).
            If the function is called directly without assignment
            (e.g. ``precipitation.plot_spi_fit(K=3, month=3)``),
            the plot is generated and nothing is printed to the console.
        """

        var = 'mm' if isinstance(self, (Precipitation, Pet, Balance)) else "raw values"
        months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]

        coeff = self.c2r_index
        domain = np.arange(-3,3.2,0.2)
        mm = np.zeros((self.K,len(domain),12))
        for m in range(12):
            for k in range(self.K):
                for i,spi_value in enumerate(domain):
                    mm[k,i,m]  = np.polyval(coeff[k, m, :], spi_value)
        cmap = spi_cmap().reversed() if self.threshold > 0 else spi_cmap()
        plt.figure(figsize=(7,5))
        plt.scatter(mm[K-1, :, month-1], domain, s=80, c=domain, cmap=cmap)
        plt.xlabel(var,fontsize=14)
        plt.ylabel(f"{self.index_name}{K}",fontsize=14)
        plt.title(f"{self.index_name}{K} calibration, {months[month-1]}",fontsize=14)
        plt.grid()
        plt.colorbar(label=f"{self.index_name}")  # opzionale
        plt.tight_layout()
        if return_data:
            return mm

    def severe_events(self, weight_index=None, plot=True, max_events=None, labels=False, unit=None, name=None):

        tstartid, tendid, duration, deficit = severe_events_deficits_computation(self, weight_index=weight_index)
        if plot == True:
            plot_severe_events(self,
                               tstartid=tstartid,
                               duration=duration,
                               deficit=deficit,
                               max_events=max_events,
                               labels=labels,
                               unit=unit,
                               name=name)
        return tstartid, tendid, duration, deficit

    def plot_trends(self, windows=[12, 36, 60, 120],figsize=None,show_spi=False,ax=None,year_ext=None,unit=None):
        """
        Wrapper method to plot trend bars on the CDN time series for a DroughtScan-compatible object.

        Args:
            windows (list of int, optional): List of window lengths (in months) over which to evaluate trends.
                                             Default is [12, 36, 60, 120].

        Returns:
            None. Displays a plot.
        """
        if figsize is None:
            figsize=(14,10)
        plot_cdn_trends(self, windows,figsize=figsize,show_spi=show_spi,ax=ax,year_ext=year_ext,unit=unit)

    def plot_monthly_profile(self, var=None, var_name=None, cumulate=False, ax=None,highlight_years=None,season_shift=False):
        """
        Plot a 24-month profile of a time series, with percentile bands and optional highlighted years.

        Parameters
        ----------
        var : a DSO timeseries to be profiled. If None, `self.ts` will be used as default.
            Must be a 1D array with the same length as `self.m_cal`.

        var_name : str or None, optional
            Optional label to include in the plot title.

        cumulate : bool, default=False
            If True, compute and display the cumulative sum per month for each year.

        highlight_years : list of int or int or None, optional
            One or more years to be highlighted in the plot.

        season_shift : bool, default=False
            If True, display the monthly profile centred on the winter season otherwise on the summer.

        Returns
        -------
        None
            Displays the plot.
        """

        monthly_profile(self, var=var,var_name=var_name, cumulate=cumulate,ax=ax, highlight_years=highlight_years, season_shift=season_shift)

    def plot_spatial(self, var='SIDI', weight_index=2, month_scale = None, ax=None, title=None,cmap=None):
        """
        Plot a spatial map of SIDI or SPI from compute_spatial_sidi output.

        Args:
            var (str): 'SIDI' or 'SPI' or CDN. Default 'SIDI'.
            weight_index (int): Weight slice for SIDI. Default 2.
            ax (matplotlib.axes.Axes, optional): Existing axes. If None, creates new figure.
            cmap (str): Colormap. Default is red-2-green palette using the coulors by Crimeri as used in the heatmap
            title (str, optional): Custom title.

        Returns:
            matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm,ListedColormap
        # from matplotlib.colors import TwoSlopeNorm

        from drought_scan.utils.visualization import spi_cmap
        if cmap is None:
            cmap = spi_cmap().reversed() if self.threshold > 0 else spi_cmap()
        elif isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        bounds = np.array([-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3])
        b_norm = BoundaryNorm(bounds, cmap.N)

        if var =='SIDI' and not hasattr(self, 'SIDI_grid'):
            raise ValueError("No spatial data found. Run compute_spatial_maps() first.")
        if var =='SPI' and not hasattr(self, 'SPI_grid'):
            raise ValueError("No spatial data found. Run spatial_maps() first.")
        if var =='CDN' and not hasattr(self, 'trend_grid'):
            raise ValueError("No spatial data found. Run spatial_trend() first.")

        # --- select data ---
        if var == 'SIDI':
            data = self.SIDI_grid[:, :, weight_index]
            label = f'{self.SIDI_name} (weight {weight_index})'
            norm = b_norm

        elif var == 'CDN':
            if month_scale is None or month_scale not in self.trend_grid:
                raise ValueError(f"Specify a valid month_scale. Available: {list(self.trend_grid.keys())}")
            data = self.trend_grid[month_scale]
            label = f'deficit/surplus on {month_scale} months'
            val = np.nanmax(np.abs(data.flatten()))
            cdnbounds = np.linspace(-val,val, len(bounds))
            norm = BoundaryNorm(cdnbounds, cmap.N)
        elif var == 'SPI':
            if month_scale is None or month_scale not in self.SPI_grid:
                raise ValueError(f"Specify a valid month_scale. Available: {list(self.SPI_grid.keys())}")
            data = self.SPI_grid[month_scale]
            label = f'SPI-{month_scale}'
            norm = b_norm
        else:
            raise ValueError("var must be 'SIDI' or 'SPI'.")

        # --- axes ---
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        # --- extent from lat/lon vectors ---
        lon_min, lon_max = self.Lon.min(), self.Lon.max()
        lat_min, lat_max = self.Lat.min(), self.Lat.max()
        extent = [lon_min, lon_max, lat_min, lat_max]

        # --- colormap centered on zero ---
        vmax = np.nanpercentile(np.abs(data), 95)


        im = ax.imshow(
            data,
            extent=extent,
            origin='upper',
            cmap=cmap,
            norm=norm,
            aspect='auto',
            zorder=1
        )

        # --- shape overlay ---
        self.shape.boundary.plot(ax=ax, color='black', linewidth=0.8, zorder=2)

        # --- colorbar ---
        plt.colorbar(im, ax=ax, label=label, fraction=0.03, pad=0.02)

        bounds = self.shape.geometry.total_bounds  # [minx, miny, maxx, maxy]
        ax.set_xlim(bounds[0] - 0.1, bounds[2] + 0.1)
        ax.set_ylim(bounds[1] - 0.1, bounds[3] + 0.1)

        # --- title ---
        month, year = self.spatial_timestamp
        if self.day:
            default_title = f'{label} | {int(self.day):02d}/{int(month):02d}/{int(year)}'
        else:
            default_title = f'{label} | {int(month):02d}/{int(year)}'
        ax.set_title(title if title else default_title)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        plt.tight_layout()
        return ax



class Precipitation(BaseDroughtAnalysis):
    def __init__(self, start_baseline_year, end_baseline_year,basin_name,ts=None,m_cal=None,prec_path=None,
                 shape_path=None,shape=None, K=None,weight_index=None,
                 calculation_method =f_kde,threshold=None, verbose=True, index_name = 'SPI',rolling=False):

        """
        Initialize the Precipitation class.

        Args:
            start_baseline_year (int): Starting year for baseline period.
            end_baseline_year (int): Ending year for baseline period.
            ts (ndarray, optional): Aggregated basin-level precipitation timeseries.
            m_cal (ndarray, optional): Calendar array (month, year) matching `ts`.
            data_path (str, optional): Path to the NetCDF file containing precipitation data.
            shape_path (str, optional): Path to the shapefile defining the basin.
            shape (object, optional): Shapefile geometry (if already loaded).
            K (int, optional): Maximum temporal scale for SPI calculations. Default is 36.
            calculation_method (callable, optional): Function to compute SPI-like indices. Default is `f_kde`
            weight_index (int, optional): Index of the weighting scheme to use for SIDI calculation.
                - weight_index = 0: Equal weights
                - weight_index = 1: Linear decreasing weights
                - weight_index = 2: Logarithmically decreasing weights (default)
                - weight_index = 3: Linear increasing weights
                - weight_index = 4: Logarithmically increasing weights

            threshold (int,optional) : threshold to define severe events, Default is -1 (i.e. -1 standard deviation of SIDI)
            verbose (bool, optional): Whether to print initialization messages. Default is True.


        """
        # Already checked in BaseDroughtAnalysis
        # if start_baseline_year is None or end_baseline_year is None:
        # 	raise ValueError("`start_baseline_year` and `end_baseline_year` must be provided.")

        self.start_baseline_year = start_baseline_year
        self.end_baseline_year = end_baseline_year
        self.verbose = verbose
        self.basin_name = basin_name

        # Gestione dello shape
        if shape is not None:
            self.shape = shape
        elif shape_path is not None:
            self.shape = load_shape(shape_path)
        elif prec_path is not None and (shape_path is None or shape is None):
            self.shape=None
            raise ValueError("Provide a shapefile (`shape_path` or `shape`) to select gridded precipitation data.")

        if ts is not None and m_cal is not None: # User provided data
            self.ts = ts
            self.m_cal = m_cal
            self.day = None
        elif prec_path is not None and self.shape is not None:
            # Load data from file
            self.prec_path = prec_path
            # self.Pgrid, self.m_cal, self.ts = self._import_data()

            self.ts, self.m_cal, self.Pgrid,self.Lat,self.Lon,self.day = import_netcdf_for_cumulative_variable(prec_path,['tp','rr','precipitation','prec','LAPrec1871',
                                                                                               'pre','swe','SWE','sd','SD','sf','SF',
                                                                                               'sde','smlt'],self.shape,self.verbose,rolling=rolling)
        else:
            raise ValueError("Provide either ts and m_cal directly or specify data_path for a gridded precipitation data in NetCDF format along with the path of the river shapefile.")

        self.K = K if K is not None else 36
        self.threshold = threshold if threshold is not None else -1
        self.weight_index = weight_index if weight_index is not None else 2

        if not callable(calculation_method):
            raise ValueError("`calculation_method` must be a callable function.")
        self.calculation_method = calculation_method
        self.index_name=index_name

        # Initialize the base class
        super().__init__(self.ts, self.m_cal, self.K, self.start_baseline_year, self.end_baseline_year,
                         self.basin_name, self.calculation_method, self.threshold, self.index_name,self.day)

        # Welcome and guidance messages
        if verbose:
            print("#########################################################################")
            print("Welcome to Drought Scan! \n")
            print("The precipitation data has been imported successfully.")
            print(f"Your data starts from {self.m_cal[0]} and ends on {self.m_cal[-1]}.")
            print("#########################################################################")
            print("Run the following class methods to access key functionalities:\n")
            print(" >>> .plot_scan()             — CDN, SPI heatmap, and SIDI overview")
            print(" >>> .analyze_correlation()   — optimal K and weighting (requires Streamflow)")
            print(f" >>> .ts, .spi_like_set, .SIDI, .CDN  — direct attribute access")

    def set_optimal_SIDI(self, optimal_k, optimal_weight_index, overwrite=False):
        """
        Recalculate SIDI using the optimal K (obtained via analyze_correlation with Streamflow).
        Optionally store optimal_k (and optimal_weight_index) on this instance.

        Args:
            optimal_k (int): optimal K determined by analyze_correlation(streamflow).
            optimal_weight_index (int): specific weight index to track/store (0-based).
            overwrite (bool): if True, updates self.SIDI and stores self.optimal_k
                              and self.optimal_weight_index on the instance.

        Returns:
            np.ndarray: SIDI array (time x n_weightings) computed with optimal_k.
        """
        if optimal_k is None or optimal_k < 0:
            raise ValueError("optimal_k must be a positive integer obtained by SIDI vs SQI1 optimitation (use 'analize_correlation' method to estimate optima_k.")

        if optimal_weight_index is None or optimal_weight_index < 0 or  optimal_weight_index >= 5:
            raise ValueError(
                "optimal_weight index must be positive integers obtained via SIDI vs SQI1 optimization")
        # ---- compute SIDI with the requested K (no side effects yet)
        SIDI_new = self.recalculate_SIDI(K=optimal_k)

        if overwrite:
            self.SIDI = SIDI_new
            self.optimal_k = optimal_k
            self.optimal_weight_index = int(optimal_weight_index)

        return SIDI_new

    def set_optimal_SIDI_seasonal(self, seasonal_corr, agg='quarter',
                                  seasons=None, overwrite=False):
        """
        Recalculate SIDI using season-specific optimal K and weight_index
        (obtained via ``analyze_correlation_seasonal``).

        The method builds a "mosaic" SIDI where each month is computed with
        the optimal parameters of its season, then standardised per-season
        on the baseline.

        When ``overwrite=True`` the 1-D seasonal SIDI is tiled to shape
        ``(time, 5)`` so that downstream consumers expecting
        ``self.SIDI[:, weight_index]`` keep working transparently
        (every column holds the same optimised series).

        Args:
            seasonal_corr (dict): Output of ``analyze_correlation_seasonal()``.
                Each key is a season name, each value contains at least
                ``'best_k'`` and ``'col_best_weight'``.
            agg (str): Aggregation scheme — **must match** the one used in
                ``analyze_correlation_seasonal()``.
                Options: 'quarter', 'semiannual', 'four-monthly', 'monthly', 'custom'.
            seasons (dict, optional): Required when ``agg='custom'``.
                ``{season_name: [month_ints]}``.
            overwrite (bool): If True, updates ``self.SIDI`` (tiled to ``(time, 5)``)
                and stores ``self.seasonal_params`` on the instance.

        Returns:
            np.ndarray: SIDI array ``(time,)`` with season-specific optimization.
        """
        if seasons is not None:
            agg = 'custom'

        seasons_dict = self._build_seasons_dict(agg, seasons)

        # Validate coverage
        missing = [s for s in seasons_dict if s not in seasonal_corr]
        if missing:
            print(f"Warning: seasons {missing} not found in seasonal_corr; "
                  f"corresponding months will be NaN in SIDI.")

        SIDI_seasonal = self.recalculate_SIDI_seasonal(seasonal_corr, seasons_dict)

        if overwrite:
            # tile to (time, 5) for backward compatibility with
            # consumers that do self.SIDI[:, weight_index]
            self.SIDI = np.tile(SIDI_seasonal[:, np.newaxis], (1, 5))
            print("Note: SIDI overwritten with seasonal optimization. "
                  "All 5 weight columns are identical (season-specific K and weight already applied).")
            # ... rest of seasonal_params assignment
            self.seasonal_params = {
                'agg': agg,
                'seasons': seasons_dict,
                'config': {
                    name: {'best_k': v['best_k'],
                           'col_best_weight': v['col_best_weight']}
                    for name, v in seasonal_corr.items()
                    if name in seasons_dict
                }
            }

        else:
            return SIDI_seasonal

    def plot_covariates(self, streamflow,year_ext=None, split_plot=False):
        """
        Plot the covariate relationship between the optimized SIDI and a Streamflow-based index.

        Parameters
        ----------
        streamflow : BaseDroughtAnalysis or Streamflow
            Streamflow object providing SQI1 or similar indices.
        year_ext : tuple of int, optional
            (start_year, end_year) to restrict the x-axis.
        split_plot : bool, optional
            If True, produces separate figures for each panel. Default False.

        Raises
        ------
        TypeError
            If streamflow is not a BaseDroughtAnalysis instance, or if the
            Precipitation object has not been optimized.
        """
        if not isinstance(streamflow, BaseDroughtAnalysis):
            raise TypeError("The input must be an instance of Streamflow or BaseDroughtAnalysis.")

        if self.is_seasonal_sidi:
            print(f"Note: SIDI is seasonally optimized — weight_index is ignored (all columns identical).")
            weight_index = None  # colonna 0, colore rosso
        elif hasattr(self, 'optimal_weight_index'):
            weight_index = self.optimal_weight_index
        else:
            raise TypeError(
                "The Precipitation object must be optimized with "
                "'set_optimal_SIDI()' or 'set_optimal_SIDI_seasonal()' OVERWRITE=True "
                "before calling this function."
            )

        plot__covariates(self, streamflow=streamflow, weight_index=weight_index,
                         year_ext=year_ext, split_plot=split_plot)

class Streamflow(BaseDroughtAnalysis):
    def __init__(self, start_baseline_year, end_baseline_year,basin_name,
                 ts=None, m_cal=None, shape=None, shape_path=None,
                 data_path=None, K=None, weight_index=2,
                 calculation_method=f_kde, threshold=-1, index_name='SQI',rolling = False):
        """
        Initialize the Streamflow class for drought analysis using streamflow data (e.g., river discharge).

        This class is fully independent from the Precipitation class.

        You must provide either:
        - `ts` and `m_cal`, or
        - a valid `data_path` to a CSV file from which to load the streamflow time series.

        Args:
            start_baseline_year (int): Start year of the reference baseline period.
            end_baseline_year (int): End year of the reference baseline period.
            ts (ndarray, optional): Streamflow time series (e.g., monthly means).
            m_cal (ndarray, optional): Calendar array (month, year) matching `ts`.
            shape (object, optional): Preloaded shapefile geometry.
            shape_path (str, optional): Path to the shapefile defining the basin.
            data_path (str, optional): CSV file path containing streamflow data.
            K (int, optional): Maximum aggregation scale for drought index calculation. Default is 36.
            weight_index (int, optional): Weighting scheme index for the SIDI/SQI index. Default is 2.
            calculation_method (callable, optional): Function to compute SPI-like indices. Default is `f_kde`.
            threshold (float, optional): Threshold (in standard deviations) to define severe drought events. Default is -1.
            index_name (str, optional): Name of the drought index. Default is 'SQI'.

        Raises:
            ValueError: If neither streamflow data nor a path to load it are provided.
        """
        # Parametri principali
        self.start_baseline_year = start_baseline_year
        self.end_baseline_year = end_baseline_year
        self.K = K if K is not None else 36
        self.threshold = threshold
        self.weight_index = weight_index
        self.basin_name = basin_name
        self.data_path = data_path



        # Metodo di calcolo e nome indice
        if not callable(calculation_method):
            raise ValueError("`calculation_method` must be a callable function.")
        self.calculation_method = calculation_method
        self.index_name = index_name

        # Gestione shapefile
        if shape is not None:
            self.shape = shape
        elif shape_path is not None:
            self.shape = load_shape(shape_path)
        else:
            self.shape = None

        # Gestione dati: ts e m_cal oppure data_path
        if ts is not None and m_cal is not None:
            self.ts = ts
            self.m_cal = m_cal
            self.is_placeholder = False
            self.day = None
        elif data_path is not None:
            # All'interno della tua classe (es. Streamflow, BaseDroughtAnalysis, ecc.)
            if data_path.endswith(('.csv', '.txt','.xls', '.xlsx')):
                print("Loading streamflow data from text/excel file...")
                self.ts, self.m_cal,self.day = load_streamflow(data_path,rolling=rolling)
            else:
                raise ValueError("Unsupported file format. Use .csv, .txt, .xls, or .xlsx")
        else:
            raise ValueError("You must provide either (`ts` and `m_cal`) or a valid `data_path`.")

        # Inizializzazione della superclasse
        super().__init__(self.ts, self.m_cal, self.K,
                         self.start_baseline_year, self.end_baseline_year,self.basin_name,
                         self.calculation_method, self.threshold, self.index_name,self.day)

    def gap_filling(self, precipitation):
        """
        Fill missing values in the streamflow time series by regressing SQI1
        on precipitation-based SIDI, then back-transforming predictions to
        discharge.

        The precipitation object **must** carry an optimized SIDI before
        calling this method.  Two paths are supported:

        - ``precipitation.set_optimal_SIDI(k, w, overwrite=True)``
        - ``precipitation.set_optimal_SIDI_seasonal(corr, agg, overwrite=True)``

        If you need to keep the original SIDI intact, call the setter with
        ``overwrite=False``, store the returned array, and optimize a copy.

        Parameters
        ----------
        precipitation : BaseDroughtAnalysis
            Precipitation (or Balance / Pet) instance whose ``SIDI`` has
            already been overwritten with optimal parameters.

        Returns
        -------
        None
            Updates ``self.ts`` in place. If any gaps are filled, also
            recomputes ``self.spi_like_set``, ``self.SIDI``, and ``self.CDN``.
        """
        if not isinstance(precipitation, BaseDroughtAnalysis):
            raise TypeError("The input must be an instance of Precipitation.")

        # --- require an optimized SIDI on the precipitation object --------
        if precipitation.is_seasonal_sidi:
            SIDI_full = precipitation.SIDI[:, 0]
        elif hasattr(precipitation, 'optimal_weight_index'):
            SIDI_full = precipitation.SIDI[:, precipitation.optimal_weight_index]
        else:
            raise ValueError(
                "precipitation.SIDI has not been optimized yet.\n"
                "Run one of the following before calling gap_filling():\n"
                "  precipitation.set_optimal_SIDI(k, w, overwrite=True)\n"
                "  precipitation.set_optimal_SIDI_seasonal(corr, agg, overwrite=True)")

        # --- checks for missing values otherwise exit ---------------------
        mask_nan = np.isnan(self.ts)
        if not np.any(mask_nan):
            print("No gaps detected in streamflow timeseries. Nothing to fill.")
            return

        # identify the gaps
        gaps_idx = np.where(np.isnan(self.ts))
        self.gap_flag = np.zeros_like(self.ts, dtype=int)
        self.gap_flag[gaps_idx] = 1

        # ==================================================================
        # Find overlap between calendars and train a model for sqi1 regression
        self_idx, prec_idx = find_overlap(self.m_cal, precipitation.m_cal)
        if len(self_idx) == 0:
            raise ValueError("No overlapping data between Precipitation and Streamflow.")

        # specify the variable of interest
        sqi1 = self.spi_like_set[0][self_idx]
        m_cal = self.m_cal[self_idx]  # (N_overlap, 2) atteso [month, year]
        ts = self.ts[self_idx].copy()  # (N_overlap,)

        SIDI = SIDI_full[prec_idx]

        valid_mask = np.isfinite(sqi1) & np.isfinite(SIDI)
        # ------------------------------------------------------------------
        # OLS fit with intercept using NumPy (no scikit-learn needed)
        X = SIDI[valid_mask]
        y = sqi1[valid_mask]

        if X.size < 2:
            raise ValueError("Not enough overlapping finite points to fit regression.")

        if np.nanstd(X) == 0:
            raise ValueError("Predictor SIDI has zero variance over valid points.")

        # Design matrix [x, 1] to estimate slope (a) and intercept (b)
        A = np.column_stack([X, np.ones_like(X)])
        coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
        a, b = coeffs  # sqi1 ≈ a * SIDI + b

        # ==================================================================
        # prediction
        prediction_mask = np.isnan(sqi1) & np.isfinite(SIDI)
        n_to_fill = int(np.sum(prediction_mask))

        if n_to_fill > 0:
            sqi1_pred = sqi1.copy()
            sqi1_pred[prediction_mask] = a * SIDI[prediction_mask] + b

            # back-transform SQI1 -> portata con polinomi mensili
            Q_pred = ts.copy()
            mc_pred = m_cal[prediction_mask]
            s_pred = sqi1_pred[prediction_mask]

            Q_pred[prediction_mask] = [
                np.polyval(self.c2r_index[0, mc_pred[i, 0] - 1, :], s_pred[i])
                for i in range(s_pred.shape[0])
            ]

        # ==================================================================
        # UPDATE
        self.ts[self_idx] = Q_pred

        # Recalculate SPI-like set, SIDI and CDN
        self.spi_like_set, self.c2r_index = self._calculate_spi_like_set()
        self.SIDI = self._calculate_SIDI()
        self.CDN = self._calculate_CDN()

        print(f"Gap filling completed. {np.sum(prediction_mask)} values updated.")

    def plot_annual_ts(self, DSO, starting_month=8, values='abs'):
        """
        Plot annual (12-month) aggregates starting from a custom month, comparing
        streamflow (Q) with an external driver (e.g., P, PET, or P-PET).

        Workflow
        --------
        1) Align monthly series on the common calendar (intersection of timestamps).
        2) Build 12-month windows starting at `starting_month` (1..12). Each window is
           included only if all 12 consecutive months exist (no gaps).
        3) Aggregate by sum within each window for both series.
        4) Plot external driver (blue, left Y-axis) and streamflow (black, right Y-axis),
           and annotate R² between the annual aggregates.

        Parameters
        ----------
        DSO : object
            Object with `.ts` (monthly series) and `.m_cal` (calendar).
            If `values='std'`, it should also expose `.spi_like_set[0]`.
        starting_month : int, default 8
            Month (1..12) at which each 12-month window starts.
        values : {'abs', 'std'}, default 'abs'
            - 'abs': use raw series (`self.ts` and `DSO.ts`).
            - 'std': use standardized SPI-like series (`self.spi_like_set[0]` and
                     `DSO.spi_like_set[0]`).

        Notes
        -----
        - External variable label is inferred from the class name:
            Precipitation -> 'P', Pet -> 'PET', Balance -> 'P-PET',
            otherwise `DSO.__class__.__name__`.
        - If your goal is to start from the climatologically driest month of Q,
          first inspect `self.plot_monthly_profile()` and pass that month via
          `starting_month`.
        - Requires an existing static method `find_overlap(cal_a, cal_b)` that returns
          aligned indices into the two calendars.

        Raises
        ------
        ValueError
            If no overlapping timestamps are found or no complete 12-month window exists.
        """
        # ---------- overlap ----------
        self_idx, dso_idx = find_overlap(self.m_cal, DSO.m_cal)
        if self_idx.size == 0 or dso_idx.size == 0:
            raise ValueError("No overlapping data found between Streamflow and the Independent variable.")

        if values not in {'abs', 'std'}:
            raise ValueError("values must be either 'abs' or 'std'.")

        if values == "abs":
            Q = np.asarray(self.ts, dtype=float)[self_idx]
            X = np.asarray(DSO.ts, dtype=float)[dso_idx]
        else:  # 'std'
            Q = np.asarray(self.spi_like_set[0], dtype=float)[self_idx]
            X = np.asarray(DSO.spi_like_set[0], dtype=float)[dso_idx]

        cal = np.asarray(self.m_cal, dtype=object)[self_idx]

        # ---------- infer label for external variable ----------
        cls = DSO.__class__.__name__.lower()
        if "precip" in cls:
            x_name = "P"
        elif cls == "pet" or "pet" in cls:
            x_name = "PET"
        elif "balance" in cls:
            x_name = "P-PET"
        else:
            x_name = DSO.__class__.__name__

        cal = np.asarray(self.m_cal, dtype=object)[self_idx]
        m0 = starting_month - 1
        years = np.unique(cal[:, 1])

        # annual aggregation:
        annual_q = []
        annual_x = []
        for yr in range(len(years)):
            try:
                win = np.arange(yr * 12 + m0, yr * 12 + 12 + m0)
                annual_x.append(np.sum(X[win]))
                annual_q.append(np.sum(Q[win]))
            except IndexError:
                last_t = yr * 12 + m0
                annual_x.append(np.sum(X[last_t::]))
                annual_q.append(np.sum(Q[last_t::]))

        annual_x = np.array(annual_x)
        annual_q = np.array(annual_q)

        # ---------- plotting ----------
        plt.figure(figsize=(11, 4))

        # Left axis: external (blue)
        ax = plt.gca()
        ax.plot(annual_x, label=f"{x_name}, yearly", color="tab:blue")
        ax.set_ylabel(f"{x_name} (yearly sum)")
        ax.grid(alpha=0.25)

        # x ticks as years
        ax.set_xlim(0, len(annual_x))
        ax.set_xticks(np.arange(len(annual_x)))
        ax.set_xticklabels(years.astype(int), rotation=90, fontweight="bold")

        # Right axis: Q (black)
        ax2 = ax.twinx()
        ax2.plot(annual_q, "-k", label="Q, yearly")
        ax2.set_ylabel("Q (yearly sum)")

        # R^2 annotation (finite pairs only)

        r = np.corrcoef(annual_q, annual_x)[0, 1]
        r2 = float(r * r)
        # place near top-left of the left axis
        y_top = np.nanmax(annual_x)
        ax.text(0.03, 0.92, f"R² = {r2:.2f}", transform=ax.transAxes,
                fontsize=12, fontweight="bold")

        # Legends
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")

        # Title
        title = f"Annual balance (start at month = {starting_month})"
        plt.title(title)

        plt.tight_layout()
        plt.show(block=False)
    # prima di usare BFI bisogna impostare la possibità che i dati di portata siano giornaliri

    def BFI(self, block_size=5, plot=True, figsize=(14, 5)):
        """
        Compute the Baseflow Index (BFI) using the Institute of Hydrology method
        (Gustard et al., 1992). Applicable to daily streamflow series only.

        The method works in three steps:
          1. Split the series into non-overlapping blocks of `block_size` days;
             extract the minimum discharge in each block.
          2. Identify turning points: a block minimum is a valid turning point if
             0.9 * centre_min < min(left_min, right_min).
          3. Linearly interpolate between turning points to obtain the baseflow
             hydrograph; clip to observed discharge so baseflow <= Q at all times.

        BFI = sum(baseflow) / sum(Q)

        Parameters
        ----------
        block_size : int
            Block length in days (default 5, as per IH standard method).
        plot : bool
            If True, plot the observed hydrograph overlaid with the baseflow.
        figsize : tuple
            Figure size passed to matplotlib (width, height) in inches.

        Returns
        -------
        bfi : float
            Baseflow Index in [0, 1].
        baseflow : np.ndarray
            Daily baseflow series, same length as self.ts.

        Raises
        ------
        AttributeError
            If self.day does not exist, indicating the series is not daily.
        ValueError
            If the series is too short or contains only NaN.

        References
        ----------
        Gustard, A., Bullock, A., Dixon, J.M. (1992). Low flow estimation in the
        United Kingdom. Institute of Hydrology Report No. 108, Wallingford, UK.
        """

        # ── Guard: daily series only ──────────────────────────────────────────────
        import matplotlib.dates as mdates
        import os
        import pandas as pd
        from drought_scan.utils.data_io import _read_csv_smart,_pick_date_col,_pick_value_col,_date_related_cols

        # re-read raw file to get the daily series (self.ts is already monthly)
        ext = os.path.splitext(self.data_path)[1].lower()
        if ext in ('.csv', '.txt'):
            df = _read_csv_smart(self.data_path)
        elif ext in ('.xlsx', '.xls'):
            df = pd.read_excel(self.data_path)
            df = df.dropna(axis=1, how='all')
        else:
            raise ValueError(f"Unsupported format: {ext}")

        date_col = _pick_date_col(df)
        exclude = _date_related_cols(df, date_col)
        value_col = _pick_value_col(df, exclude_cols=exclude)

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col, value_col])

        # ── Gap filling on daily index ────────────────────────────────────────────
        dates = pd.to_datetime(df[date_col])
        values = df[value_col].to_numpy(dtype=float)

        full_index = pd.date_range(start=dates.iloc[0], end=dates.iloc[-1], freq='D')
        missing_dates = full_index.difference(dates)

        if len(missing_dates):
            print(f"Found {len(missing_dates)} missing dates — filling with NaN")
            fill_df = pd.DataFrame({date_col: missing_dates, value_col: np.nan})
            df = (
                pd.concat([df[[date_col, value_col]], fill_df])
                .sort_values(date_col)
                .reset_index(drop=True)
            )
            dates = pd.to_datetime(df[date_col])
            values = df[value_col].to_numpy(dtype=float)

        Q = values.astype(float)
        n = len(Q)


        if n < block_size * 3:
            raise ValueError(
                f"Series too short: at least {block_size * 3} days required."
            )

        # ── Step 1: block minima ──────────────────────────────────────────────────
        n_blocks = n // block_size
        Q_trimmed = Q[:n_blocks * block_size]  # drop trailing incomplete block
        blocks = Q_trimmed.reshape(n_blocks, block_size)

        block_min = np.nanmin(blocks, axis=1)  # minimum per block
        block_days = np.arange(n_blocks) * block_size + block_size // 2  # centre day index

        # ── Step 2: turning points (IH criterion) ────────────────────────────────
        tp_days = []
        tp_vals = []

        for i in range(1, n_blocks - 1):
            centre = block_min[i]
            left = block_min[i - 1]
            right = block_min[i + 1]
            if centre * 0.9 < min(left, right):  # Gustard et al. (1992)
                tp_days.append(block_days[i])
                tp_vals.append(centre)

        if len(tp_vals) < 2:
            raise ValueError(
                "Insufficient turning points. Check series quality or reduce block_size."
            )

        tp_days = np.array(tp_days)
        tp_vals = np.array(tp_vals)

        # ── Step 3: linear interpolation → daily baseflow ────────────────────────
        all_days = np.arange(n)
        baseflow = np.minimum(
            np.interp(all_days, tp_days, tp_vals),  # interpolated baseflow envelope
            Q  # cannot exceed observed Q
        )

        # Propagate NaN where discharge is missing
        baseflow[np.isnan(Q)] = np.nan

        # ── BFI ──────────────────────────────────────────────────────────────────
        valid = ~np.isnan(Q) & ~np.isnan(baseflow)
        bfi = float(np.sum(baseflow[valid]) / np.sum(Q[valid]))

        # ── Optional plot ─────────────────────────────────────────────────────────
        if plot:

            time_axis = all_days
            use_dates = False

            fig, ax = plt.subplots(figsize=figsize)

            # Filled areas: baseflow (blue) and quickflow (grey above baseflow)
            ax.fill_between(time_axis, baseflow, alpha=0.45,
                            color='steelblue', label='Baseflow')
            ax.fill_between(time_axis, baseflow, Q, alpha=0.30,
                            color='slategray', label='Quickflow')
            ax.plot(time_axis, Q, color='navy', linewidth=0.8,
                    label='Observed Q')

            # Overlay turning points (only those within series bounds)
            mask = tp_days < n
            ax.plot(time_axis[tp_days[mask]], tp_vals[mask],
                    'o', color='firebrick', markersize=3,
                    label='Turning points', zorder=5)

            if use_dates:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                fig.autofmt_xdate()

            ax.set_xlabel('Date' if use_dates else 'Day index')
            ax.set_ylabel('Discharge  [units of self.ts]')
            ax.set_title(
                f'Baseflow separation — IH method (Gustard et al., 1992)\n'
                f'BFI = {bfi:.3f}   |   block size = {block_size} days'
            )
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.4)
            plt.tight_layout()
            plt.show(block=False)

        return bfi, baseflow

class Pet(BaseDroughtAnalysis):
    def __init__(self, start_baseline_year, end_baseline_year, basin_name, ts=None, m_cal=None, data_path=None,
                 shape_path=None, shape=None, K=None, weight_index=None,
                 calculation_method =f_kde,threshold=None, index_name = 'SPETI',verbose=True,rolling=False):
        """
        Initialize the Pet class.

        Args:
            start_baseline_year (int): Starting year for baseline period.
            end_baseline_year (int): Ending year for baseline period.
            ts (ndarray, optional): Aggregated basin-level PET timeseries.
            m_cal (ndarray, optional): Calendar array (month, year) matching `ts`.
            data_path (str, optional): Path to the NetCDF file containing PET data.
            shape_path (str, optional): Path to the shapefile defining the basin.
            shape (object, optional): Shapefile geometry (if already loaded).
            K (int, optional): Maximum temporal scale for calculations. Default is 36.
            calculation_method (callable, optional): Method to use for drought calculations. Default is f_kde.
                Available methods (in utils.py) are:
                f_spi:   FOR  POSITIVE & RIGHT-SKEWED DATA (uses a Gamma Function) but works fine also for positive normal distribuited sample
                f_spei:  FOR REAL VALUES & RIGHT-SKEWED (uses a Pearson III function)
                f_zscore FOR REAL VALUES NORMAL DISTRIBUTED
            threshold (float, optional): Threshold for severe events. Defaults to -1.
            verbose (bool, optional): Whether to print initialization messages. Default is True.
        """
        self.start_baseline_year = start_baseline_year
        self.end_baseline_year = end_baseline_year
        self.verbose = verbose
        self.basin_name=basin_name

        if shape is not None:
            self.shape = shape
        elif shape_path is not None:
            self.shape = load_shape(shape_path)
        elif data_path is not None and (shape_path is None or shape is None):
            self.shape = None
            raise ValueError("Provide a shapefile (`shape_path` or `shape`) to select gridded PET data.")

        if ts is not None and m_cal is not None:  # User provided data
            self.ts = ts
            self.m_cal = m_cal
            self.day = None
        elif data_path is not None and self.shape is not None:
            self.data_path = data_path
            self.ts, self.m_cal, self.PETgrid,self.Lat,self.Lon , self.day= import_netcdf_for_cumulative_variable(data_path,
                                                ['e', 'E','ET','PET','pet','et','evaporation',
                                                 'evapotranspiration','potential evapotranspiration',
                                                 'reference evapotranspiration','swe','pev','Ep',
                                                 ],
                                                self.shape,self.verbose,rolling=rolling)
        else:
            raise ValueError("Provide either ts and m_cal directly or specify data_path for gridded PET data in NetCDF format along with the path of the river shapefile.")

        self.K = K if K is not None else 36

        self.threshold = 1 if threshold is None else threshold

        if weight_index is None:
            self.weight_index = 2

        if not callable(calculation_method):
            raise ValueError("`calculation_method` must be a callable function.")
        self.calculation_method = calculation_method
        self.index_name = index_name

        super().__init__(self.ts, self.m_cal, self.K, self.start_baseline_year, self.end_baseline_year,
                         self.basin_name, self.calculation_method, self.threshold, self.index_name,self.day)

        if verbose:
            print("#########################################################################")
            print("Welcome to Drought Scan! \n")
            print("The PET data has been imported successfully.")
            print(f"Data range: {self.m_cal[0]} to {self.m_cal[-1]}.")
            print(" >>> .plot_scan()  — CDN, index heatmap, and SIDI overview")
            print("#########################################################################")

class Balance(BaseDroughtAnalysis):
    def __init__(self, start_baseline_year, end_baseline_year, basin_name, prec_path=None, pet_path=None,
                 shape_path=None, shape=None, ts=None, m_cal=None, K=None,
                 calculation_method=f_kde, threshold=None, index_name = 'SPEI',verbose=True,rolling=False):
        """
        Initialize the Balance class for calculating water balance (precipitation - PET).

        Args:
            start_baseline_year (int): Starting year for baseline period.
            end_baseline_year (int): Ending year for baseline period.
            prec_path (str, optional): Path to the NetCDF file containing precipitation data.
            pet_path (str, optional): Path to the NetCDF file containing PET data.
            shape_path (str, optional): Path to the shapefile defining the basin.
            shape (object, optional): Shapefile geometry (if already loaded).
            ts (ndarray, optional): Pre-computed water balance timeseries (precipitation - PET).
            m_cal (ndarray, optional): Pre-computed calendar array (month, year).
            K (int, optional): Maximum temporal scale for calculations. Default is 36.
            threshold (int, optional): Threshold to define severe events. Default is -1.
            calculation_method (callable, optional): Method to use for drought calculations. Default is f_kde.
                Available methods (in utils.py) are:
                f_spi:   FOR  POSITIVE & RIGHT-SKEWED DATA (uses a Gamma Function) but works fine also for positive normal distribuited sample
                f_spei:  FOR REAL VALUES & RIGHT-SKEWED (uses a Pearson III function)
                f_zscore FOR REAL VALUES NORMAL DISTRIBUTED
            verbose (bool, optional): Whether to print initialization messages. Default is True.
        """
        self.start_baseline_year = start_baseline_year
        self.end_baseline_year = end_baseline_year
        self.verbose = verbose
        self.basin_name=basin_name
        self.rolling = rolling

        # Load shapefile if provided
        if shape is not None:
            self.shape = shape
        elif shape_path is not None:
            self.shape = load_shape(shape_path)
        elif ts is None or m_cal is None:
            raise ValueError("Provide a shapefile (`shape_path` or `shape`) if NetCDF files are used.")

        # If ts and m_cal are provided, skip data import
        if ts is not None and m_cal is not None:
            self.ts = ts
            self.m_cal = m_cal
            self.day = None
        elif prec_path is not None and pet_path is not None and self.shape is not None:
            self.prec_path = prec_path
            self.pet_path = pet_path
            self.prec_grid, self.pet_grid, self.m_cal, self.ts, self.Lat, self.Lon, self.day= self._import_data() #only over common timeframe
        else:
            raise ValueError("Provide either `ts` and `m_cal` or specify `prec_path`, `pet_path`, and `shape_path`.")

        # Set optional arguments
        self.K = 36 if K is None else K
        self.threshold = -1 if threshold is None else threshold

        if not callable(calculation_method):
            raise ValueError("`calculation_method` must be a callable function.")
        self.calculation_method = calculation_method
        self.index_name = index_name

        # Initialize the base class
        super().__init__(self.ts, self.m_cal, self.K, self.start_baseline_year, self.end_baseline_year,
                         self.basin_name,self.calculation_method, self.threshold, self.index_name,self.day)

        if verbose:
            print("#########################################################################")
            print("Welcome to Drought Scan! \n")
            print("The P-PET balance has been imported successfully.")
            print(f"Data range: {self.m_cal[0]} to {self.m_cal[-1]}.")
            print(" >>> .plot_scan()  — CDN, index heatmap, and SIDI overview")
            print("#########################################################################")

    def _import_data(self):
        """
        Import precipitation and PET data, align them on a common calendar,
        and crop them to their common spatial extent.

        Returns:
            tuple: (Pgrid, ETgrid, m_cal, ts, Lat_common, Lon_common)
                - Pgrid: precipitation grid on common domain [time, lat, lon]
                - ETgrid: PET grid on common domain [time, lat, lon]
                - m_cal: common calendar
                - ts: precipitation minus PET time series
                - Lat_common: common latitude vector
                - Lon_common: common longitude vector
        """

        # import preciptiation data
        prec_ts, prec_cal, Pgrid, Lat_p, Lon_p,day1= import_netcdf_for_cumulative_variable(
            self.prec_path,
            possible_names=['tp','rr','precipitation','prec','LAPrec1871','pre','swe','SWE','sd','SD','sf','SF'],  # Possibili nomi della variabile
            shape=self.shape,
            verbose=self.verbose, rolling = self.rolling)


        # import PET data
        pet_ts, pet_cal, ETgrid,Lat_et,Lon_et,day2 = import_netcdf_for_cumulative_variable(
            self.pet_path,
            possible_names=['e', 'ET','PET','pet','et','evaporation',
                                                 'evapotranspiration','potential evapotranspiration',
                                                 'reference evapotranspiration','swe','pev'],
            shape=self.shape,
            verbose=self.verbose,rolling = self.rolling
        )
        # align the timestamp
        if day1 is not None and day2 is not None and day1 != day2:
            raise ValueError(f"Prec anchor day ({day1}) ≠ PET anchor day ({day2}): rolling windows not aligned.")

        p_id, pet_id = find_overlap(prec_cal, pet_cal)
        if not p_id.size:
            raise ValueError("No common dates found between precipitation and PET datasets.")

            # Allinea i dati secondo le date comuni
        m_cal = prec_cal[p_id]
        Pgrid = Pgrid[p_id, :, :]
        ETgrid = ETgrid[pet_id, :, :]
        # Calcola la differenza tra precipitazione e PET
        ts = prec_ts[p_id] - pet_ts[pet_id]


        # -----------------------------
        # convert coordinates to 1D arrays
        # -----------------------------
        Lat_p = np.asarray(Lat_p).squeeze()
        Lon_p = np.asarray(Lon_p).squeeze()
        Lat_et = np.asarray(Lat_et).squeeze()
        Lon_et = np.asarray(Lon_et).squeeze()

        if Lat_p.ndim != 1 or Lon_p.ndim != 1 or Lat_et.ndim != 1 or Lon_et.ndim != 1:
            raise ValueError("Lat/Lon must be 1D arrays.")

        # -----------------------------
        # check same resolution
        # -----------------------------
        dlat_p = np.median(np.abs(np.diff(Lat_p)))
        dlon_p = np.median(np.abs(np.diff(Lon_p)))
        dlat_et = np.median(np.abs(np.diff(Lat_et)))
        dlon_et = np.median(np.abs(np.diff(Lon_et)))

        tol = 1e-8
        if not (np.isclose(dlat_p, dlat_et, atol=tol) and np.isclose(dlon_p, dlon_et, atol=tol)):
            raise ValueError("P and ET grids do not have the same spatial resolution.")

        # -----------------------------
        # define common spatial extent
        # -----------------------------
        lat_min = max(np.min(Lat_p), np.min(Lat_et))
        lat_max = min(np.max(Lat_p), np.max(Lat_et))
        lon_min = max(np.min(Lon_p), np.min(Lon_et))
        lon_max = min(np.max(Lon_p), np.max(Lon_et))

        if (lat_min >= lat_max) or (lon_min >= lon_max):
            raise ValueError("No common spatial extent found between precipitation and PET grids.")

        # -----------------------------
        # select coordinates inside common extent
        # -----------------------------
        lat_idx_p = np.where((Lat_p >= lat_min - tol) & (Lat_p <= lat_max + tol))[0]
        lon_idx_p = np.where((Lon_p >= lon_min - tol) & (Lon_p <= lon_max + tol))[0]

        lat_idx_et = np.where((Lat_et >= lat_min - tol) & (Lat_et <= lat_max + tol))[0]
        lon_idx_et = np.where((Lon_et >= lon_min - tol) & (Lon_et <= lon_max + tol))[0]

        if lat_idx_p.size == 0 or lon_idx_p.size == 0 or lat_idx_et.size == 0 or lon_idx_et.size == 0:
            raise ValueError("Common extent found, but no valid grid cells were selected.")

        # -----------------------------
        # crop the 3D grids
        # -----------------------------
        Pgrid = Pgrid[:, lat_idx_p[0]:lat_idx_p[-1] + 1, lon_idx_p[0]:lon_idx_p[-1] + 1]
        ETgrid = ETgrid[:, lat_idx_et[0]:lat_idx_et[-1] + 1, lon_idx_et[0]:lon_idx_et[-1] + 1]

        Lat_common_p = Lat_p[lat_idx_p[0]:lat_idx_p[-1] + 1]
        Lon_common_p = Lon_p[lon_idx_p[0]:lon_idx_p[-1] + 1]

        Lat_common_et = Lat_et[lat_idx_et[0]:lat_idx_et[-1] + 1]
        Lon_common_et = Lon_et[lon_idx_et[0]:lon_idx_et[-1] + 1]

        # -----------------------------
        # final consistency check
        # -----------------------------
        if len(Lat_common_p) != len(Lat_common_et) or len(Lon_common_p) != len(Lon_common_et):
            raise ValueError("After cropping, P and ET grids still have inconsistent dimensions.")

        if not np.allclose(Lat_common_p, Lat_common_et, atol=tol):
            raise ValueError("Latitude coordinates do not match after cropping.")

        if not np.allclose(Lon_common_p, Lon_common_et, atol=tol):
            raise ValueError("Longitude coordinates do not match after cropping.")

        Lat_common = Lat_common_p
        Lon_common = Lon_common_p

        # -----------------------------
        # compute spatial average time series difference
        # -----------------------------
        # ts = np.nanmean(Pgrid, axis=(1, 2)) - np.nanmean(ETgrid, axis=(1, 2))


        return Pgrid, ETgrid, m_cal, ts,Lat_common, Lon_common,day1

    def set_optimal_SIDI(self, optimal_k, optimal_weight_index, overwrite=False):
        """
        Recalculate SIDI using the optimal K (obtained via analyze_correlation with Streamflow).
        Optionally store optimal_k (and optimal_weight_index) on this instance.

        Args:
            optimal_k (int): optimal K determined by analyze_correlation(streamflow).
            optimal_weight_index (int): specific weight index to track/store (0-based).
            overwrite (bool): if True, updates self.SIDI and stores self.optimal_k
                              and self.optimal_weight_index on the instance.

        Returns:
            np.ndarray: SIDI array (time x n_weightings) computed with optimal_k.
        """
        if optimal_k is None or optimal_k < 0:
            raise ValueError("optimal_k must be a positive integer obtained by SIDI vs SQI1 optimitation (use 'analize_correlation' method to estimate optima_k.")

        if optimal_weight_index is None or optimal_weight_index < 0 or  optimal_weight_index >= 5:
            raise ValueError(
                "optimal_weight index must be positive integers obtained via SIDI vs SQI1 optimization")
        # ---- compute SIDI with the requested K (no side effects yet)
        SIDI_new = self.recalculate_SIDI(K=optimal_k)

        if overwrite:
            self.SIDI = SIDI_new
            self.optimal_k = optimal_k
            self.optimal_weight_index = int(optimal_weight_index)

        return SIDI_new

    def set_optimal_SIDI_seasonal(self, seasonal_corr, agg='quarter',
                                  seasons=None, overwrite=False):
        """
        Recalculate SIDI using season-specific optimal K and weight_index
        (obtained via ``analyze_correlation_seasonal``).

        The method builds a "mosaic" SIDI where each month is computed with
        the optimal parameters of its season, then standardised per-season
        on the baseline.

        When ``overwrite=True`` the 1-D seasonal SIDI is tiled to shape
        ``(time, 5)`` so that downstream consumers expecting
        ``self.SIDI[:, weight_index]`` keep working transparently
        (every column holds the same optimised series).

        Args:
            seasonal_corr (dict): Output of ``analyze_correlation_seasonal()``.
                Each key is a season name, each value contains at least
                ``'best_k'`` and ``'col_best_weight'``.
            agg (str): Aggregation scheme — **must match** the one used in
                ``analyze_correlation_seasonal()``.
                Options: 'quarter', 'semiannual', 'four-monthly', 'monthly', 'custom'.
            seasons (dict, optional): Required when ``agg='custom'``.
                ``{season_name: [month_ints]}``.
            overwrite (bool): If True, updates ``self.SIDI`` (tiled to ``(time, 5)``)
                and stores ``self.seasonal_params`` on the instance.

        Returns:
            np.ndarray: SIDI array ``(time,)`` with season-specific optimization.
        """
        if seasons is not None:
            agg = 'custom'

        seasons_dict = self._build_seasons_dict(agg, seasons)

        # Validate coverage
        missing = [s for s in seasons_dict if s not in seasonal_corr]
        if missing:
            print(f"Warning: seasons {missing} not found in seasonal_corr; "
                  f"corresponding months will be NaN in SIDI.")

        SIDI_seasonal = self.recalculate_SIDI_seasonal(seasonal_corr, seasons_dict)

        if overwrite:
            # tile to (time, 5) for backward compatibility with
            # consumers that do self.SIDI[:, weight_index]
            self.SIDI = np.tile(SIDI_seasonal[:, np.newaxis], (1, 5))
            print("Note: SIDI overwritten with seasonal optimization. "
                  "All 5 weight columns are identical (season-specific K and weight already applied).")
            # ... rest of seasonal_params assignment
            self.seasonal_params = {
                'agg': agg,
                'seasons': seasons_dict,
                'config': {
                    name: {'best_k': v['best_k'],
                           'col_best_weight': v['col_best_weight']}
                    for name, v in seasonal_corr.items()
                    if name in seasons_dict
                }
            }

        else:
            return SIDI_seasonal

    def plot_covariates(self, streamflow,year_ext=None, split_plot=False):
        """
        Plot the covariate relationship between the optimized SIDI and a Streamflow-based index.

        Parameters
        ----------
        streamflow : BaseDroughtAnalysis or Streamflow
            Streamflow object providing SQI1 or similar indices.
        year_ext : tuple of int, optional
            (start_year, end_year) to restrict the x-axis.
        split_plot : bool, optional
            If True, produces separate figures for each panel. Default False.

        Raises
        ------
        TypeError
            If streamflow is not a BaseDroughtAnalysis instance, or if the
            Precipitation object has not been optimized.
        """
        if not isinstance(streamflow, BaseDroughtAnalysis):
            raise TypeError("The input must be an instance of Streamflow or BaseDroughtAnalysis.")

        if self.is_seasonal_sidi:
            print(f"Note: SIDI is seasonally optimized — weight_index is ignored (all columns identical).")
            weight_index = None  # colonna 0, colore rosso
        elif hasattr(self, 'optimal_weight_index'):
            weight_index = self.optimal_weight_index
        else:
            raise TypeError(
                "The Precipitation object must be optimized with "
                "'set_optimal_SIDI()' or 'set_optimal_SIDI_seasonal()' OVERWRITE=True "
                "before calling this function."
            )

        plot__covariates(self, streamflow=streamflow, weight_index=weight_index,
                         year_ext=year_ext, split_plot=split_plot)


class Temperature(BaseDroughtAnalysis):
    def __init__(self, start_baseline_year, end_baseline_year,
                 basin_name, data_path=None, shape_path=None, ts=None, m_cal=None,
                 shape=None, K=None, weight_index=None,
                 calculation_method =f_kde,threshold=None, index_name = 'STI',verbose=True,rolling=False):
        """
        Initialize the Temperature class.

        Args:
            start_baseline_year (int): Starting year for baseline period.
            end_baseline_year (int): Ending year for baseline period.
            ts (ndarray, optional): Aggregated basin-level Temperature timeseries.
            m_cal (ndarray, optional): Calendar array (month, year) matching `ts`.
            data_path (str, optional): Path to the NetCDF file containing Temperature data.
            shape_path (str, optional): Path to the shapefile defining the basin.
            shape (object, optional): Shapefile geometry (if already loaded).
            K (int, optional): Maximum temporal scale for calculations. Default is 36.
            weight_index (int, optional): Index of the weighting scheme to use for calculations.
                - weight_index = 0: Equal weights
                - weight_index = 1: Linear decreasing weights
                - weight_index = 2: Logarithmically decreasing weights (default)
                - weight_index = 3: Linear increasing weights
                - weight_index = 4: Logarithmically increasing weights

            threshold (int, optional): Threshold to define severe events, Default is -1.
            calculation_method (callable, optional): Method to use for drought calculations. Default is f_kde.
                Available methods (in utils.py) are:
                f_spi:   FOR  POSITIVE & RIGHT-SKEWED DATA (uses a Gamma Function) but works fine also for positive normal distribuited sample
                f_spei:  FOR REAL VALUES & RIGHT-SKEWED (uses a Pearson III function)
                f_zscore FOR REAL VALUES NORMAL DISTRIBUTED
            verbose (bool, optional): Whether to print initialization messages. Default is True.
        """
        self.start_baseline_year = start_baseline_year
        self.end_baseline_year = end_baseline_year
        self.verbose = verbose
        self.basin_name=basin_name

        if shape is not None:
            self.shape = shape
        elif shape_path is not None:
            self.shape = load_shape(shape_path)
        elif data_path is not None and (shape_path is None or shape is None):
            self.shape = None
            raise ValueError("Provide a shapefile (`shape_path` or `shape`) to select gridded Temperature data.")

        if ts is not None and m_cal is not None:  # User provided data
            self.ts = ts
            self.m_cal = m_cal
            self.day = None
        elif data_path is not None and self.shape is not None:
            self.data_path = data_path
            self.ts, self.m_cal, self.Tgrid,self.Lat,self.Lon,self.day = import_netcdf_for_cumulative_variable(data_path,
                                                ['deg0l','tg','tm','tx','t2m','d2m','mn2t'],
                                                self.shape,self.verbose,cumulative=False, rolling=rolling)
        else:
            raise ValueError("Provide either ts and m_cal directly or specify data_path for gridded Temperature data in NetCDF format along with the path of the river shapefile.")

        self.K = K if K is not None else 36

        self.threshold = 1 if threshold is None else threshold

        if weight_index is None:
            self.weight_index = 2

        if not callable(calculation_method):
            raise ValueError("`calculation_method` must be a callable function.")
        self.calculation_method = calculation_method
        self.index_name = index_name

        super().__init__(self.ts, self.m_cal, self.K, self.start_baseline_year, self.end_baseline_year,
                         self.basin_name, self.calculation_method, self.threshold, self.index_name,self.day)

        if verbose:
            print("#########################################################################")
            print("Welcome to Drought Scan! \n")
            print("The Temperature data has been imported successfully.")
            print(f"Data range: {self.m_cal[0]} to {self.m_cal[-1]}.")
            print(" >>> .plot_scan()  — CDN, index heatmap, and SIDI overview")
            print("#########################################################################")

class Teleindex(BaseDroughtAnalysis):
    def __init__(self, start_baseline_year, end_baseline_year, basin_name=None,ts=None, m_cal=None, data_path=None,
                 K=None, weight_index=None,
                 calculation_method=f_kde, threshold=None, verbose=True, index_name=''):

        """
        Initialize the Teleindex class.

        A general-purpose class for standardizing scalar time series that are
        not linked to any hydrographic basin (e.g., teleconnection indices
        like NAO, ENSO, AMO). No shapefile is required.

        Args:
            start_baseline_year (int): Starting year for baseline period.
            end_baseline_year (int): Ending year for baseline period.
            basin_name (str, optional): Label for plots. Default is None.
            ts (ndarray, optional): Time series values.
            m_cal (ndarray, optional): Calendar array (month, year) matching `ts`.
            data_path (str, optional): Path to a CSV/Excel file containing the time series.
            K (int, optional): Maximum temporal scale. Default is 36.
            weight_index (int, optional): Weighting scheme index. Default is 2.
            calculation_method (callable, optional): Standardization function. Default is f_kde.
            threshold (float, optional): Threshold for severe events. Default is -1.
            verbose (bool, optional): Print initialization messages. Default is True.
            index_name (str, optional): Label for the index. Default is ''.

		"""
        # Already checked in BaseDroughtAnalysis
        # if start_baseline_year is None or end_baseline_year is None:
        # 	raise ValueError("`start_baseline_year` and `end_baseline_year` must be provided.")

        self.start_baseline_year = start_baseline_year
        self.end_baseline_year = end_baseline_year
        self.verbose = verbose
        self.basin_name=basin_name



        if ts is not None and m_cal is not None:  # User provided data
            self.ts = ts
            self.m_cal = m_cal
        elif data_path is not None:
            # Load data from file
            self.data_path = data_path
            # self.Pgrid, self.m_cal, self.ts = self._import_data()
            self.ts, self.m_cal = import_timeseries(data_path)
        else:
            raise ValueError(
                "Provide either (`ts` and `m_cal`) or a valid `data_path` to a CSV/Excel file")

        self.K = K if K is not None else 36
        self.threshold = threshold if threshold is not None else -1
        self.weight_index = weight_index if weight_index is not None else 2

        if not callable(calculation_method):
            raise ValueError("`calculation_method` must be a callable function.")
        self.calculation_method = calculation_method
        self.index_name = index_name

        # Inizializza forecast come None
        self.forecast_ts = None
        self.forecast_m_cal = None

        # Initialize the base class
        super().__init__(self.ts, self.m_cal, self.K, self.start_baseline_year, self.end_baseline_year,
                         self.basin_name,self.calculation_method, self.threshold, self.index_name,self.day)

        # Welcome and guidance messages
        if verbose:
            print("#########################################################################")
            print("Welcome to Drought Scan! \n")
            print("The teleconnection data has been imported successfully.")
            print(f"Your data starts from {self.m_cal[0]} and ends on {self.m_cal[-1]}.")
            print("#########################################################################")
            print("Run the following class methods to access key functionalities:\n")
            print(" >>> ._plot_scan(): to plot the CDN, spiset heatmap, and D_{SPI} \n ")
            print(
                " >>> ._analyze_correlation(): to estimate the best K and weighting function (only if streamflow data are available) \n")
            print(
                "*************** Alternatively, you can access to: \n >>> precipitation.ts (P timeseries), \n >>> precipitation.spi_like_set (SPI (1:K) timeseries) \n >>> precipitation.SIDI (D_{SPI}) \n to visualize the data your way or proceed with further analyses!")


# This populates the exclusion tuple now that the classes exist:
BaseDroughtAnalysis._EXCLUDED_FROM_CORRELATION = (Streamflow, Temperature, Teleindex)

if __name__ == "__main__":
    print("This module contains the main classes for computing SPI, SIDI, and CDN indices.")
    print("Import the classes into an external script to use them in your project.")
