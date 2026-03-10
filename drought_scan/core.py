"""
author: PyDipa
# © 2025 Arianna Di Paola


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
os.environ['USE_PYGEOS'] = '0'
from matplotlib.colors import ListedColormap
import json
from functools import partial
from drought_scan.utils.drought_indices import *
from drought_scan.utils.data_io import *
from drought_scan.utils.hydrology import *
from drought_scan.utils.visualization import *
from drought_scan.utils.statistics import *
class BaseDroughtAnalysis:
    def __init__(self, ts, m_cal, K, start_baseline_year, end_baseline_year,basin_name,
                 calculation_method,threshold,index_name='SPI'):
        """
        Base class for drought analysis.

        Args:
            ts (ndarray): Time series data (e.g., precipitation or streamflow).
            m_cal (ndarray): Calendar array (month, year) matching `ts`.
            K (int): Maximum temporal scale for SPI calculations.
            start_baseline_year (int): Starting year for baseline period.
            end_baseline_year (int): Ending year for baseline period.
            calculation_method (callable, optional): Function for index calculation. Defaults to f_spi.
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
            # opzionale: None, 0, o raise esplicito
            return -1
        if self.shape.crs is None:
            self.shape = self.shape.set_crs('epsg:4326')
        elif self.shape.crs.to_string() != 'EPSG:4326':
            self.shape = self.shape.to_crs('epsg:4326')

        area_proj = self.shape.to_crs(epsg=32632)

        #estiamte the area in square meters
        area_kmq = area_proj.geometry.area.iloc[0]/1e6
        return area_kmq

    def _compute_spi(self, month_scale,gamma_params=None):
        """
        Calculate SPI for a specific temporal scale, optionally using precomputed gamma parameters.

        Args:
            month_scale (int): Temporal scale for SPI (e.g., SPI-3, SPI-6).
            gamma_params (dict, optional): Dictionary with precomputed gamma parameters {k: {m: (alpha, loc, beta)}}
                where k is the time scale and m is the reference month (1-12).

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
            if gamma_params is None:
                if way==1:
                    indices, spi_values, coeff, _ = self.calculation_method(
                        self.ts, month_scale, ref_month, self.m_cal, self.start_baseline_year, self.end_baseline_year
                    )
                elif way==2:
                    indices, spi_values, coeff = self.calculation_method(
                        self.ts, month_scale, ref_month, self.m_cal, self.start_baseline_year, self.end_baseline_year
                    )

            else:
                alpha, loc, beta = gamma_params[ref_month]
                indices, spi_values, coeff, _ = self.calculation_method(
                    self.ts, month_scale, ref_month, self.m_cal, self.start_baseline_year, self.end_baseline_year,
                    gamma_params=(alpha, loc, beta)  # Passiamo i parametri salvati
                )


            if indices is None or spi_values is None or coeff is None:
                raise ValueError(f"`f_spi` returned invalid results for ref_month={ref_month}.")

            Spi_ts[indices] = spi_values.copy()
            c2rspi[ref_month - 1, :] = coeff.copy()
        return Spi_ts,c2rspi

    def _calculate_spi_like_set(self,gamma_params=None):
        """
           Compute SPI values for all temporal scales up to K, optionally using precomputed gamma parameters.

           Args:
               gamma_params (dict, optional): Dictionary with precomputed gamma parameters {k: {m: (alpha, loc, beta)}}
                   where k is the time scale and m is the reference month (1-12).

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
        elif  base_func == f_zscore:
            c2rspi = np.zeros((self.K, 12, 2), dtype=float)

        # Calculate SPI for each temporal scale
        for k in range(1, self.K + 1):
            if gamma_params is None:
                Spi_ts, coeff = self._compute_spi(k)
            else:
                params = gamma_params[k]
                Spi_ts, coeff = self._compute_spi(k,gamma_params=params)
            spiset[k - 1, :] = Spi_ts.copy()
            c2rspi[k - 1, :, :] = coeff.copy()
        return spiset, c2rspi

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

    def _calculate_CDN(self):
        """
		Compute the Standardized Integrated Drought Index (SIDI).

		Returns:
			ndarray: SIDI values (time steps x number of implemented weighting function) standardized to zero mean and unit variance.

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

    @staticmethod
    def _process_grid_point(ts_ij, m_cal, K, start_baseline_year, end_baseline_year,
                            calculation_method, M_valid, t_idx, n_weights):
        """Standalone computation for a single grid point — picklable by joblib."""
        from drought_scan.utils.drought_indices import (
            generate_weights, weighted_metrics, baseline_indices
        )

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
        return SIDI_ij[t_idx, :], spi_out

    def compute_spatial_sidi(self, M=None, timestamp=None, K=None):
        """
        Compute gridded SPI at selected temporal scales and SIDI for each grid point in Pgrid.

        Args:
            M (list of int, optional): Temporal scales for SPI maps. Defaults to [1, 3, 6, 12, 18, 24].
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

        if M is None:
            M = [1, 3, 6, 12, 18, 24]

        _K_orig = self.K
        if K is not None:
            self.K = K

        if timestamp is None:
            t_idx = len(self.m_cal) - 1
        else:
            month_t, year_t = timestamp
            matches = np.where((self.m_cal[:, 0] == month_t) & (self.m_cal[:, 1] == year_t))[0]
            if len(matches) == 0:
                raise ValueError(f"Timestamp {timestamp} not found in m_cal.")
            t_idx = int(matches[0])

        M_valid = [m for m in M if m <= self.K]
        if len(M_valid) < len(M):
            import warnings
            warnings.warn(f"Scales {set(M) - set(M_valid)} exceed K={self.K} and will be skipped.")

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

        for idx, (i, j) in enumerate(valid_ij):
            sidi_vals, spi_vals = results[idx]
            if sidi_vals is None:
                continue
            SIDI_grid[i, j, :] = sidi_vals
            for m in M_valid:
                SPI_grid[m][i, j] = spi_vals[m]

        print("compute_spatial_sidi: 100% — done.")

        self.SIDI_grid = SIDI_grid
        self.SPI_grid = SPI_grid
        self.spatial_timestamp = self.m_cal[t_idx].copy()
        self.K = _K_orig

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
            self._saveplot()

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

        # Default to a window size of 60 if none is provided
        if window is None:
            window = 60
        if var is None:
            var = self.CDN
        results = rolling_trend_analysis(var=var, window=window, significance=0.05)
        return results

    def plot_trends(self, windows=[12, 36, 60, 120],show_spi=False,ax=None,year_ext=None,unit=None):
        """
        Wrapper method to plot trend bars on the CDN time series for a DroughtScan-compatible object.

        Args:
            windows (list of int, optional): List of window lengths (in months) over which to evaluate trends.
                                             Default is [12, 36, 60, 120].

        Returns:
            None. Displays a plot.
        """
        plot_cdn_trends(self, windows,show_spi=show_spi,ax=ax,year_ext=year_ext,unit=unit)

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


    season_shift : bool, default=False
        If True, display the monthly profile centred on the winter season otherwise on the summer.

        Returns
        -------
        None
            Displays the plot.
        """

        monthly_profile(self, var=var,var_name=var_name, cumulate=cumulate,ax=ax, highlight_years=highlight_years, season_shift=season_shift)

    def export_scan_plot_csv(self, weight_index=2, optimal_k=None, name=None, out_dir="exports"):
        """
        Exports the minimum data needed in CSV format to replicate the plot_scan in another workspace.

        Args:
            DSO: DroughtScan Object (Precipitation or Streamflow)
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
            for j in range(len(DSO.m_cal)):
                vec = self.spi_like_set[0:optimal_k, j]
                sidis.append([weighted_metrics(vec, weights[:, weight_index])[0]])
            sidi_vec = np.squeeze(np.array(sidis))
        else:
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
            metadata["area_kmq"] = DSO.area_kmq
        else:
            metadata["area_kmq"] = np.nan
        # Salvataggio
        prefix = name.replace(" ", "_") if name else "DSO_export"
        df_mcal.to_csv(os.path.join(out_dir, f"{prefix}_m_cal.csv"), index=False)
        df_cdn.to_csv(os.path.join(out_dir, f"{prefix}_cdn.csv"), index=False)
        spi_df.to_csv(os.path.join(out_dir, f"{prefix}_spi.csv"), index=False, header=False)
        df_sidi.to_csv(os.path.join(out_dir, f"{prefix}_sidi.csv"), index=False)

        with open(os.path.join(out_dir, f"{prefix}_meta.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Data exported successfully in {out_dir}/ with prefix '{prefix}'")
    # ----------------------------------------------------------

    def _savedsplot(self):

        k = self.K if not hasattr(self, 'optimal_k') or self.optimal_k is None else self.optimal_k
        w = self.weight_index if not hasattr(self, 'optimal_weight_index') or self.optimal_weight_index is None else self.optimal_weight_index
        baseline =self.start_baseline_year,self.end_baseline_year
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


class Precipitation(BaseDroughtAnalysis):
    def __init__(self, start_baseline_year, end_baseline_year,basin_name,ts=None,m_cal=None,prec_path=None,
                 shape_path=None,shape=None, K=None,weight_index=None,
                 calculation_method =f_kde,threshold=None, verbose=True, index_name = 'SPI'):

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
        elif prec_path is not None and self.shape is not None:
            # Load data from file
            self.prec_path = prec_path
            # self.Pgrid, self.m_cal, self.ts = self._import_data()
            self.ts, self.m_cal, self.Pgrid = import_netcdf_for_cumulative_variable(prec_path,['tp','rr','precipitation','prec','LAPrec1871',
                                                                                               'pre','swe','SWE','sd','SD','sf','SF',
                                                                                               'sde','smlt'],self.shape,self.verbose)
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
                         self.basin_name, self.calculation_method, self.threshold, self.index_name)

        # Welcome and guidance messages
        if verbose:
            print("#########################################################################")
            print("Welcome to Drought Scan! \n")
            print("The precipitation data has been imported successfully.")
            print(f"Your data starts from {self.m_cal[0]} and ends on {self.m_cal[-1]}.")
            print("#########################################################################")
            print("Run the following class methods to access key functionalities:\n")
            print(" >>> ._plot_scan(): to plot the CDN, spiset heatmap, and D_{SPI} \n ")
            print(
                " >>> ._analyze_correlation(): to estimate the best K and weighting function (only if streamflow data are available) \n")
            print(
                "*************** Alternatively, you can access to: \n >>> precipitation.ts (P timeseries), \n >>> precipitation.spi_like_set (SPI (1:K) timeseries) \n >>> precipitation.SIDI (D_{SPI}) \n to visualize the data your way or proceed with further analyses!")

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

    def analyze_correlation(self, streamflow,plot=True,plot_mode="all"):
        """
        Analyze correlations between Precipitation SIDI and Streamflow SPI for different weightings and K values.

        Args:
            streamflow (Streamflow): Instance of the Streamflow class.
            plot (bool, optional): Whether to generate a correlation plot and call `_plot_scan`. Default is True.
            plot_mode (str): 'all' (default), 'seasonal', 'monthly'.

        Returns:
            dict: Contains the best K, weight configuration, and maximum correlation value.
                - "best_k" (int): Optimal month-scale (K).
                - "col_best_weight" (int): Index of the best weight configuration.
                - "max_correlation" (float): Maximum R^2 value achieved.
        """
        wlabel = ['equal weights (ew)', 'linearly decreasing weights (ldw)',
                  ' logarithmically decreasing weights (lgdw)', 'linearly increasing weights (liw)',
                  'logarithmically increasing weights (lgiw)']

        if not isinstance(streamflow, BaseDroughtAnalysis):
            raise TypeError("The input must be an instance of Streamflow or BaseDroughtAnalysis.")

        # find the temporal overlap between Precipitation and Streamflow
        self_indices, streamflow_indices = find_overlap(self.m_cal,streamflow.m_cal)
        if len(self_indices) == 0 or len(streamflow_indices) == 0:
            raise ValueError("No overlapping data found between Precipitation and Streamflow.")

        # Subset di dati per l'overlapping time
        y = streamflow.spi_like_set[0, streamflow_indices]  # SPI-1 dello streamflow
        spi_like_set = self.spi_like_set[:,self_indices]  # Tutte le configurazioni SIDI


        K_range = np.arange(1, self.K + 1)
        MatCorr = []

        print("Starting correlation analysis...")
        for k in K_range:
            W = generate_weights(k)
            # print("Calculating Ensemble Weighted Mean for each weighting function...")
            sidis = []  # SPI ensemble mean for each day
            for doy in range(len(spi_like_set[0])):#in range(self._baseline_indices()[0], self._baseline_indices()[1] + 1):
                vec = spi_like_set[:k, doy]
                sidis.append([weighted_metrics(vec, w)[0] for w in W.T])
            sidis = np.array(sidis)

            rr = []  # Correlations for each weighting function
            for w in range(len(W.T)):
                # Standardize SIDI sull'intero periodo perché l'overlapping è troppo variabile
                SIDI = (sidis[:, w] - np.nanmean(sidis[:, w])) / np.nanstd(sidis[:, w])
                valid_mask = np.isfinite(y) & np.isfinite(SIDI)
                r = stats.pearsonr(SIDI[valid_mask], y[valid_mask])[0]
                rr.append(r ** 2)
                # print(f"K={k}, Weight {w + 1}: R^2 = {np.round(r ** 2, 3)}")
            MatCorr.append(rr)
        # looking to the single SQI - SPI correlation
        rr_spi = []
        for j,spi in enumerate(spi_like_set):
            valid_mask = np.isfinite(y) & np.isfinite(spi)
            r = stats.pearsonr(spi[valid_mask], y[valid_mask])[0]
            rr_spi.append(r ** 2)
        rr_spi = np.array(rr_spi)
        ii = np.argsort(rr_spi)[::-1]
        R2_spi = np.array([np.arange(1,self.K+1)[ii],rr_spi[ii]]).T

        MatCorr = np.array(MatCorr)
        # Find the best K and weight index
        max_corr = np.max(MatCorr)
        best_k, best_weight = np.unravel_index(np.argmax(MatCorr), MatCorr.shape)


        print(f"Best correlation: R2  = {max_corr:.3f} (K={K_range[best_k]}, Weight={wlabel[best_weight]})")
        W = generate_weights(K_range[best_k])
        sidi = []  # SPI ensemble mean for each day
        for doy in range(len(spi_like_set[0])):  # in range(self._baseline_indices()[0], self._baseline_indices()[1] + 1):
            vec = spi_like_set[:K_range[best_k], doy]
            sidi.append(weighted_metrics(vec, W[:,best_weight])[0])
        sidi = np.array(sidi)

        SIDI = (sidi - np.nanmean(sidi)) / np.nanstd(sidi)

        # --- plots ----------------------------------------------------------------

        if plot == True:

            plt.figure(figsize=(10, 5))
            # weight_labels = ["Equal", "Linear Shallow", "Geom Shallow", "Linear Deep", "Geom Deep"]
            for w in range(len(W.T)):
                plt.plot(MatCorr[:, w], label=wlabel[w], linewidth=2)
            plt.grid()
            plt.legend(loc=3)
            plt.xticks(np.arange(len(K_range)), K_range)
            plt.ylabel(r"$R^2$", fontweight="bold", fontsize=12)
            plt.xlabel("Month-scale (K)", fontweight="bold", fontsize=12)
            plt.title(f"Correlation Analysis: {self.SIDI_name}  vs.  {streamflow.index_name}1", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.show(block=False)




            # basic scan plot
            self.plot_scan(optimal_k=K_range[best_k], weight_index=best_weight)

            plt.figure(figsize=(7, 7))
            if plot_mode == "all":
                plt.plot(SIDI, y, 'ok', markerfacecolor='yellow', linewidth=2)
            elif plot_mode == "seasonal":
                g1 = [4, 5, 6, 7, 8, 9]
                g2 = [10,11, 12, 1, 2, 3]
                # summer ------------------------------------------------------
                m1summer = np.isin(self.m_cal[self_indices, 0], g1)
                m2summer = np.isin(streamflow.m_cal[streamflow_indices, 0], g1)
                f  = np.isfinite(SIDI[m1summer]) & np.isfinite(y[m2summer])
                rho,pval = stats.pearsonr(SIDI[m1summer][f], y[m2summer][f])
                rho = 0 if pval>0.5 else rho
                plt.plot(SIDI[m1summer], y[m2summer], 'o', color='tab:olive', alpha=0.4, label=f'Apr-Oct; $R^2$ = {np.round(rho**2,2)}')
                # winter ------------------------------------------------------
                m1winter = np.isin(self.m_cal[self_indices, 0], g2)
                m2winter = np.isin(streamflow.m_cal[streamflow_indices, 0], g2)
                f = np.isfinite(SIDI[m1winter]) & np.isfinite(y[m2winter])
                rho,pval = stats.pearsonr(SIDI[m1winter][f], y[m2winter][f])
                rho = 0 if pval>0.5 else rho
                plt.plot(SIDI[m1winter], y[m2winter], 'o', color='tab:blue', alpha=0.4, label=f'Nov-Mar; $R^2$ = {np.round(rho**2,2)}')
            elif plot_mode == 'monthly':
                if cmc is not None:
                    c = plt.get_cmap(cmc.romaO, 12)
                else:
                    c = plt.get_cmap('twilight_shifted', 12)

                for month in range(1, 13):
                    m1 = np.where(self.m_cal[self_indices, 0] == month)[0]
                    m2 = np.where(streamflow.m_cal[streamflow_indices, 0] == month)[0]
                    plt.plot(SIDI[m1],y[m2], 'o',color=c(month/12),label = f'month {month}')


            plt.plot(np.arange(-3,4),np.arange(-3,4),'--',color='grey')
            plt.grid()
            plt.ylabel(f"{streamflow.index_name}1 ", fontweight="bold", fontsize=12)
            plt.xlabel(f"{self.SIDI_name}", fontweight="bold", fontsize=12)
            plt.title(f"{self.SIDI_name} vs.  {streamflow.index_name}1 . K={best_k} - weighting function n. {best_weight}; $R^2$ = {max_corr:.2f}", fontsize=14, fontweight="bold")

            plt.legend(fontsize=12)
            plt.tight_layout()
            plt.show(block=False)

        return {"best_k": K_range[best_k], "col_best_weight": best_weight, "max_correlation": max_corr,'spi_corr':R2_spi}

    def analyze_correlation_seasonal(self, streamflow, agg='quarter',plot=True,seasons=None):
        """
        Perform seasonal correlation analysis between precipitation-based SIDI and
        streamflow-based SPI indices for different weighting schemes and K (month-scale) values.

        Depending on the selected aggregation mode (`agg`), data are grouped by predefined
        temporal windows (quarterly, semiannual, four-monthly) or by a custom set of months.

        Args:
            streamflow (Streamflow):
                Instance of the Streamflow class to be compared with the precipitation dataset.
            agg (str, optional):
                Defines how data are temporally aggregated. Options are:

                - **'quarter'** *(default)* →
                  Standard climatological seasons:
                  `{'DJF':[12,1,2], 'MAM':[3,4,5], 'JJA':[6,7,8], 'SON':[9,10,11]}`

                - **'semiannual'** →
                  Two six-month periods:
                  `{'autwin':[9,10,11,12,1,2], 'springsum':[3,4,5,6,7,8]}`

                - **'four-monthly'** →
                  Three four-month periods:
                  `{'ONDJ':[10,11,12,1], 'FMAM':[2,3,4,5], 'JJAS':[6,7,8,9]}`

                -**'monthly**

                - **'custom'** →
                  User-defined aggregation; requires passing an additional dictionary
                  through the `seasons` argument, e.g.:
                  `seasons={'period1':[1,2,3,4], 'period2':[5,6,7,8]}`

            plot (bool, optional):
                If True, generate diagnostic plots showing the R²–K relationships
                and the scatter plots for the best seasonal configurations. Default is True.

            seasons (dict, optional):
                Custom month mapping to use when `agg='custom'`.
                The dictionary must have season names as keys and lists of month indices as values.

        Returns:
            dict:
                Dictionary containing seasonal correlation results. Each key corresponds to
                a season or aggregation period, and the value is a dictionary with the fields:

                - `"best_k"` *(int)* → Optimal temporal aggregation window (K)
                - `"col_best_weight"` *(int)* → Index of the best weighting function
                - `"max_correlation"` *(float)* → Maximum R² value obtained
                - `"R2_matrix"` *(np.ndarray)* → Full R² matrix for all K × weighting configurations

                Example:
                {
                    "DJF": {"best_k": 3, "col_best_weight": 1, "max_correlation": 0.72, "R2_matrix": array(...)},
                    "MAM": {...},
                    ...
                }

        Notes:
            The method automatically adjusts subplot layout and figure dimensions according
            to the number of analyzed seasons. Overlapping months between precipitation
            and streamflow datasets are used to ensure temporal consistency.

        """

        wlabel = [
            'EW',
            'Lin. DW',
            'Log. DW',
            'Lin. IW',
            'Log. IW'
        ]

        if not isinstance(streamflow, BaseDroughtAnalysis):
            raise TypeError("The input must be an instance of Streamflow or BaseDroughtAnalysis.")

        # --- Temporal overlap between Precipitation and Streamflow ---
        self_indices, streamflow_indices = find_overlap(self.m_cal, streamflow.m_cal)
        if len(self_indices) == 0 or len(streamflow_indices) == 0:
            raise ValueError("No overlapping data found between Precipitation and Streamflow.")

        # --- Subset to overlapping period ---
        y = streamflow.spi_like_set[0, streamflow_indices]  # SPI-1 (streamflow)
        spi_like_set = self.spi_like_set[:, self_indices]  # All SIDI configurations
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

        # --- Seasonal analysis ---
        if agg == 'quarter':
            seasons = {
                "DJF": [12, 1, 2],
                "MAM": [3, 4, 5],
                "JJA": [6, 7, 8],
                "SON": [9, 10, 11],
            }
            figsize1 = (14,10)
            figsize2 = (10,11)
            ncols,nrows = (2,2)

        elif (agg == 'semiannual') | (agg =='biannual'):
            seasons = {
                "autwin": [9, 10, 11,12, 1, 2],
                "springsum": [3, 4, 5,6, 7, 8],
            }
            figsize1 = (14,5)
            figsize2 = (10,6)
            ncols,nrows = (2,1)

        elif agg == 'four-monthly':
            seasons = {
                "ONDG" : [10,11, 12, 1],
                "FMAM" : [2,3,4,5],
                "JJAS" : [6,7,8,9],
            }
            figsize1 = (16,6)
            figsize2 = (15,6)
            ncols,nrows = (3,1)

        elif agg == 'monthly':
            seasons = {'Jan':[1], 'Feb': [2], 'Mar': [3], 'Apr': [4], 'May': [5], 'Jun': [6],
            'Jul': [7], 'Aug': [8], 'Sep': [9], 'Oct': [10], 'Nov': [11], 'Dec': [12]}
            figsize1 = (7,6)
            figsize2 = (7,6)
            ncols,nrows = (1,1)

        elif agg == 'custom':
            seasons = seasons
            layout_map = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2), 5: (2, 3), 6: (2, 3)}
            nrows, ncols = layout_map.get(len(seasons), (2, 3))
            wid = np.round(5*ncols,1)
            h = np.round(5*nrows,1)+0.5
            figsize1,figsize2 =(wid+3,h),(wid,h)



        # a check on the whole year
        all_months = [m for lst in seasons.values() for m in lst]
        unique = set(all_months)
        missing = [m for m in range(1, 13) if m not in unique]
        duplicates = [m for m in unique if all_months.count(m) > 1]

        if missing:
            print("allert: missing month(s):", missing)
        if duplicates:
            print("allert: duplicates month(s):", duplicates)


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
                "sample number": np.count_nonzero(idx)

            }

        # --- Plot 1: R² vs K for each weighting scheme ---
        if agg =='monthly':
            # ordine mesi: 'Jan'...'Dec'
            months = list(calendar.month_abbr[1:])  # ['Jan','Feb',...,'Dec']

            # vettore con i max R^2 mese per mese (NaN se manca)
            r2max = np.array([MatCorr.get(m, {}).get('max_correlation', np.nan) for m in months], float)

            # plot rapido (bar)
            plt.figure(figsize=(7, 5))
            plt.bar(months, r2max,0.5,alpha=0.8,facecolor='dimgrey',edgecolor='k')
            plt.ylabel(r"$\max R^2$")
            plt.ylim(0, 1)
            plt.grid(axis='y', alpha=0.3)
            plt.title(f"{self.basin_name} - Monthly max R² {self.SIDI_name} vs. {streamflow.index_name}1")
            plt.axhline(y=0.4, linestyle='--',color='tab:orange',label='Significant limit')
            plt.legend()
            # plt.text(0.10,0.45, 'Significant limit', color='tab:orange',fontweight='bold',fontsize=14)

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
                    ax[i].set_xticks(np.arange(0,len(K_range),3))
                    ax[i].set_xticklabels(K_range[0:-1:3])
                    ax[i].tick_params(axis='x', labelsize=14)
                    ax[i].tick_params(axis='y', labelsize=14)
                    ax[i].set_ylabel(r"$R^2$", fontweight="bold", fontsize=16)
                    ax[i].set_xlabel("Month-scale (K)", fontweight="bold", fontsize=16)
                    ax[i].set_title(name, fontweight="bold", fontsize=16)
                    if i==0:
                        ax[i].legend(loc=3)
                fig.suptitle(
                    f"{self.basin_name} - Correlation Analysis: {self.SIDI_name} vs. {streamflow.index_name}1",
                    fontsize=16, fontweight="bold"
                )
                plt.tight_layout()
                plt.show(block=False)

            # --- Plot 2: Scatter plots (best configuration per season) ---
                if len(seasons)>5:
                    if cmc is not None:
                        c = plt.get_cmap(cmc.romaO, len(seasons))
                    else:
                        c = plt.get_cmap('twilight_shifted', len(seasons))
                else:
                    c = ['tab:olive', 'tab:brown', 'tab:orange', 'tab:cyan','tab:purple']
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

                    ax[i].plot(SIDI[idx], y[idx], 'o', color=c[i], alpha=0.4, label=f'$R^2$ = {np.round(r2, 2)} \n K ={best_k}; {wlabel[best_weight]}')
                    ax[i].plot(np.arange(-3, 4), np.arange(-3, 4), '--', color='grey')
                    ax[i].tick_params(axis='x', labelsize=14)
                    ax[i].tick_params(axis='y', labelsize=14)
                    ax[i].grid()
                    ax[i].set_title(season, fontweight="bold", fontsize=16)
                    ax[i].set_ylabel(f"{streamflow.index_name}1", fontweight="bold", fontsize=16)
                    ax[i].set_xlabel(f"{self.SIDI_name}", fontweight="bold", fontsize=16)
                    ax[i].legend(fontsize=12)

                fig.suptitle(
                    f"{self.basin_name} - {self.SIDI_name} vs. {streamflow.index_name}1 — Best seasonal configurations",
                    fontsize=14, fontweight="bold"
                )
                if i < ncols*nrows-1:
                    fig.delaxes(ax[-1])
                plt.tight_layout()
                plt.show(block=False)

        return MatCorr

    def plot_covariates(self, streamflow, year_ext=None,split_plot=False):

        if not isinstance(streamflow, BaseDroughtAnalysis):
            raise TypeError("The input must be an instance of Streamflow or BaseDroughtAnalysis.")

        if not hasattr(self,'optimal_weight_index'):
            raise TypeError(
                "The Precipitation object must be optimized with 'Precipitation.set_optimal_SIDI()' "
                "before calling this function."
            )

        plot__covariates(self,streamflow=streamflow,weight_index=self.optimal_weight_index,year_ext=year_ext,split_plot=split_plot)

    def spi_sqi_corr(self, streamflow, plot=True):
        """
        Compute the month-wise correlation between SPIₖ (k = 1…K) and SQI₁
        and optionally plot the R² heatmap.

        This function quantifies how precipitation-based SPI-like indices
        propagate into hydrological drought (SQI-like) on a month-by-month basis.
        For each calendar month (Jan…Dec), the function computes the
        Pearson correlation between SPIₖ and SQI₁ for all k in [1, K],
        returning an R² matrix of shape (12 × K).

        Parameters
        ----------
        streamflow : object
            A DroughtScan-compatible object containing:
            - `m_cal`: (N, 2) array of [month, year] metadata
            - `spi_like_set`: 2D array of standardized streamflow indices,
               where spi_like_set[0] corresponds to SQI₁.
        plot : bool, optional (default=False)
            If True, produces a contourf heatmap of R²(month, k).

        Returns
        -------
        R2 : ndarray, shape (12, K)
            Matrix of determination coefficients R² for each month (rows)
            and SPI scale k (columns). Only correlations with p < 0.05
            are retained; others are set to 0.

        Raises
        ------
        ValueError
            If no temporal overlap exists between precipitation and streamflow.

        Notes
        -----
        • Overlap between precipitation and streamflow calibration windows
          is enforced to avoid inconsistent correlations.
        • R² values are rounded to 2 decimals for compact visualization.
        • Non-significant correlations (p ≥ 0.05) are set to 0 by construction.

        """

        # ------------------------------------------------------------
        # 1. Find temporal overlap between precipitation and streamflow
        # ------------------------------------------------------------
        self_indices, streamflow_indices = find_overlap(self.m_cal, streamflow.m_cal)

        if len(self_indices) == 0 or len(streamflow_indices) == 0:
            raise ValueError("No overlapping data found between Precipitation and Streamflow.")

        # ------------------------------------------------------------
        # 2. Extract overlapping standardized indices
        # ------------------------------------------------------------
        # SQI₁ (streamflow): first row of streamflow.spi_like_set
        sqi1 = streamflow.spi_like_set[0, streamflow_indices]

        # SPI-like set (precipitation): all scales
        spi_like_set = self.spi_like_set[:, self_indices]

        # Month/year metadata for overlapping period
        m_cal = self.m_cal[self_indices]

        # ------------------------------------------------------------
        # 3. Compute R²(month, k) matrix
        # ------------------------------------------------------------
        R2 = []
        Ki = np.arange(self.K)  # scales 0…K-1

        for m in range(1, 13):  # calendar months
            r2_row = []

            # indices of entries belonging to month m
            ii = np.where(m_cal[:, 0] == m)[0]

            # SQI₁ for month m
            y = sqi1[ii]

            for K in Ki:
                x = spi_like_set[K][ii]

                # filter finite values to avoid NaNs in Pearsonr
                mask = (np.isfinite(x) & np.isfinite(y))

                if mask.sum() < 3:
                    # too few points to compute correlation
                    r2_row.append(0)
                    continue

                rho, pval = stats.pearsonr(x[mask], y[mask])

                # Keep only statistically significant correlations
                r2_val = np.round(rho ** 2, 2) if pval < 0.05 else 0
                r2_row.append(r2_val)

            R2.append(r2_row)

        R2 = np.array(R2)

        # ------------------------------------------------------------
        # 4. Optional plotting
        # ------------------------------------------------------------
        if plot:
            plt.figure(figsize=(10, 4))
            X, Y = np.meshgrid(np.arange(1, self.K + 1), np.arange(1, 13))

            levels = np.arange(0.05, 1.15, 0.1)
            centers = (levels[:-1] + levels[1:]) / 2

            cf = plt.contourf(X, Y, R2, cmap='pink_r', levels=levels)

            plt.xticks(np.arange(1, self.K + 1))
            plt.yticks(
                np.arange(1, 13),
                ['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May', 'Jun.',
                 'Jul.', 'Aug.', 'Sep.', 'Oct.', 'Nov.', 'Dec.']
            )

            plt.xlabel('month scale (K)')
            cbar = plt.colorbar(cf, ticks=centers)
            cbar.ax.set_yticklabels([f"{c:.2f}" for c in centers])
            cbar.set_label(r"$R^2$", rotation=0, labelpad=12, fontsize=12)

            # Minor grid for readability
            plt.gca().set_yticks(np.arange(1.5, 12, 1), minor=True)
            plt.grid(axis='y', which='minor', linestyle='--', alpha=0.7)

            plt.tight_layout()

        return R2


class Streamflow(BaseDroughtAnalysis):
    def __init__(self, start_baseline_year, end_baseline_year,basin_name,
                 ts=None, m_cal=None, shape=None, shape_path=None,
                 data_path=None, K=36, weight_index=2,
                 calculation_method=f_kde, threshold=-1, index_name='SQI'):
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
            calculation_method (callable, optional): Function to compute SPI-like indices. Default is `f_spi`.
            threshold (float, optional): Threshold (in standard deviations) to define severe drought events. Default is -1.
            index_name (str, optional): Name of the drought index. Default is 'SQI'.

        Raises:
            ValueError: If neither streamflow data nor a path to load it are provided.
        """
        # Parametri principali
        self.start_baseline_year = start_baseline_year
        self.end_baseline_year = end_baseline_year
        self.K = K
        self.threshold = threshold
        self.weight_index = weight_index
        self.basin_name = basin_name

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
        elif data_path is not None:
            # All'interno della tua classe (es. Streamflow, BaseDroughtAnalysis, ecc.)
            if data_path.endswith(('.csv', '.txt','.xls', '.xlsx')):
                print("Loading streamflow data from text/excel file...")
                self.ts, self.m_cal = load_streamflow(data_path)
            else:
                raise ValueError("Unsupported file format. Use .csv, .txt, .xls, or .xlsx")
        else:
            raise ValueError("You must provide either (`ts` and `m_cal`) or a valid `data_path`.")

        # Inizializzazione della superclasse
        super().__init__(self.ts, self.m_cal, self.K,
                         self.start_baseline_year, self.end_baseline_year,self.basin_name,
                         self.calculation_method, self.threshold, self.index_name)

    def gap_filling(self, precipitation,optimal_k,optimal_weight_index):
        """

        Fill missing values in the streamflow time series by regressing SQI1 on
        precipitation-based SIDI, then back-transforming predictions to discharge.

        Parameters
        ----------
        precipitation : BaseDroughtAnalysis
            Precipitation instance that provides SPI-like sets and calendar. Typically the same
            object used when running `analyze_correlation(streamflow)` to select `optimal_k`
            and `optimal_weight_index`.
        optimal_k : int
            Number of SPI-like scales (top-K) used to compute SIDI on `precipitation`.
        optimal_weight_index : int
            Index (0-based) of the weighting scheme used when extracting the SIDI column.

        Returns
        -------
        None
            The method updates `self.ts` in place over the overlap with `precipitation`. If any
            gaps are filled, it also recomputes `self.spi_like_set`, `self.SIDI`, and `self.CDN`.
            A message is printed in the form:
            "Gap filling completed. {n_to_fill} values updated." \
        """
        if not isinstance(precipitation, BaseDroughtAnalysis):
            raise TypeError("The input must be an instance of Precipitation.")

        if optimal_k is None or optimal_k < 0:
            raise ValueError(
                "optimal_k must be a positive integer obtained by SIDI vs SQI1 optimitation (use 'analize_correlation' method to estimate optima_k.")

        if optimal_weight_index is None or optimal_weight_index < 0 or optimal_weight_index >= 5:
            raise ValueError(
                "optimal_weight index must be positive integers obtained via SIDI vs SQI1 optimization")



        # checks for missing values otherwise exit
        mask_nan = np.isnan(self.ts)
        if not np.any(mask_nan):
            print("No gaps detected in streamflow timeseries. Nothing to fill.")
            return

        # identify the gaps
        gaps_idx = np.where(np.isnan(self.ts))
        self.gap_flag = np.zeros_like(self.ts, dtype=int)
        self.gap_flag[gaps_idx] = 1

        # ==================================================================================
        # Find overlap between calendars and train a model for sqi1 regression
        self_idx, prec_idx = find_overlap(self.m_cal, precipitation.m_cal)
        if len(self_idx) == 0:
            raise ValueError("No overlapping data between Precipitation and Streamflow.")

        # specify the varibale of interest
        sqi1 = self.spi_like_set[0][self_idx]
        m_cal = self.m_cal[self_idx]  # (N_overlap, 2) atteso [month, year]
        ts = self.ts[self_idx].copy()  # (N_overlap,)

        SIDIs = precipitation.recalculate_SIDI(K=optimal_k)
        SIDI = SIDIs[:, optimal_weight_index][prec_idx]

        valid_mask = np.isfinite(sqi1) & np.isfinite(SIDI)
        # ----------------------------------------------------------------------------------
        # OLS fit with intercept using NumPy (no scikit-learn needed)
        X = SIDI[valid_mask]
        y = sqi1[valid_mask]

        if X.size < 2:
            raise ValueError("Not enough overlapping finite points to fit regression.")

        # Check variance to avoid singular matrix issues
        if np.nanstd(X) == 0:
            raise ValueError("Predictor SIDI has zero variance over valid points.")

        # Design matrix [x, 1] to estimate slope (a) and intercept (b)
        A = np.column_stack([X, np.ones_like(X)])
        coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
        a, b = coeffs  # sqi1 ≈ a * SIDI + b

        # ==================================================================================
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
            # reverse SQI1 (index == 0) to ts

            Q_pred[prediction_mask] = [
                np.polyval(self.c2r_index[0, mc_pred[i, 0] - 1, :], s_pred[i])
                for i in range(s_pred.shape[0])
            ]

        # ==================================================================================
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


    # OLD WORKING SCRIPT
    # def load_streamflow_from_csv(self, file_path, date_col=None, value_col=None, verbose=True):
    #     """
    #     Load and assign streamflow data from a CSV file to this instance.
    #     Wrapper around `load_streamflow_from_csv_file`.
    #
    #     Args:
    #         file_path (str): Path to the CSV file.
    #         date_col (str, optional): Name of the column with dates.
    #         value_col (str, optional): Name of the column with streamflow values.
    #         verbose (bool, optional): Whether to print info messages.
    #
    #     Returns:
    #         None
    #     """
    #     self.ts, self.m_cal = load_streamflow_from_csv(file_path, date_col, value_col)
    #
    #     # Ricomputazione degli indici
    #     self.spi_like_set, self.c2r_index = self._calculate_spi_like_set()
    #     self.SIDI = self._calculate_SIDI()
    #     self.CDN = self._calculate_CDN()
    #     self.is_placeholder = False
    #
    # def load_streamflow_from_excel(self, file_path, date_col=None, value_col=None, verbose=True):
    #     """
    #     Load and assign streamflow data from a CSV file to this instance.
    #     Wrapper around `load_streamflow_from_csv_file`.
    #
    #     Args:
    #         file_path (str): Path to the CSV file.
    #         date_col (str, optional): Name of the column with dates.
    #         value_col (str, optional): Name of the column with streamflow values.
    #         verbose (bool, optional): Whether to print info messages.
    #
    #     Returns:
    #         None
    #     """
    #     self.ts, self.m_cal = load_streamflow_from_excel(file_path, date_col, value_col)
    #
    #     # Ricomputazione degli indici
    #     self.spi_like_set, self.c2r_index = self._calculate_spi_like_set()
    #     self.SIDI = self._calculate_SIDI()
    #     self.CDN = self._calculate_CDN()
    #     self.is_placeholder = False


class Pet(BaseDroughtAnalysis):
    def __init__(self, start_baseline_year, end_baseline_year, basin_name, ts=None, m_cal=None, data_path=None,
                 shape_path=None, shape=None, K=None, weight_index=None,
                 calculation_method =f_kde,threshold=None, index_name = 'SPETI',verbose=True):
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
            weight_index (int, optional): Index of the weighting scheme to use for calculations.
                - weight_index = 0: Equal weights
                - weight_index = 1: Linear decreasing weights
                - weight_index = 2: Logarithmically decreasing weights (default)
                - weight_index = 3: Linear increasing weights
                - weight_index = 4: Logarithmically increasing weights

            threshold (int, optional): Threshold to define severe events, Default is -1.
            calculation_method (callable, optional): Method to use for drought calculations. Default is f_zscore.
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
        elif data_path is not None and self.shape is not None:
            self.data_path = data_path
            self.ts, self.m_cal, self.PETgrid = import_netcdf_for_cumulative_variable(data_path,
                                                ['e', 'E','ET','PET','pet','et','evaporation',
                                                 'evapotranspiration','potential evapotranspiration',
                                                 'reference evapotranspiration','swe','pev','Ep',
                                                 ],
                                                self.shape,self.verbose)
        else:
            raise ValueError("Provide either ts and m_cal directly or specify data_path for gridded PET data in NetCDF format along with the path of the river shapefile.")

        if K is None:
            self.K = 36

        self.threshold = 1 if threshold is None else threshold

        if weight_index is None:
            self.weight_index = 2

        if not callable(calculation_method):
            raise ValueError("`calculation_method` must be a callable function.")
        self.calculation_method = calculation_method
        self.index_name = index_name

        super().__init__(self.ts, self.m_cal, self.K, self.start_baseline_year, self.end_baseline_year,
                         self.basin_name, self.calculation_method, self.threshold, self.index_name)

        if verbose:
            print("#########################################################################")
            print("Welcome to Drought Scan! \n")
            print("The PET data has been imported successfully.")
            print(f"Your data starts from {self.m_cal[0]} and ends on {self.m_cal[-1]}.")
            print("#########################################################################")
            print("Run the following class methods to access key functionalities:\n")
            print(" >>> ._plot_scan(): to plot the CDN, zscore heatmap, and D_{zscore} \n")

    def speti_sqi_corr(self, streamflow, plot=True):
        """
        Compute the month-wise correlation between SPETIₖ (k = 1…K) and SQI₁
        and optionally plot the R² heatmap.

        This function quantifies how precipitation-based SPI-like indices
        propagate into hydrological drought (SQI-like) on a month-by-month basis.
        For each calendar month (Jan…Dec), the function computes the
        Pearson correlation between SPETIₖ and SQI₁ for all k in [1, K],
        returning an R² matrix of shape (12 × K).

        Parameters
        ----------
        streamflow : object
            A DroughtScan-compatible object containing:
            - `m_cal`: (N, 2) array of [month, year] metadata
            - `spi_like_set`: 2D array of standardized streamflow indices,
               where spi_like_set[0] corresponds to SQI₁.
        plot : bool, optional (default=False)
            If True, produces a contourf heatmap of R²(month, k).

        Returns
        -------
        R2 : ndarray, shape (12, K)
            Matrix of determination coefficients R² for each month (rows)
            and SPI scale k (columns). Only correlations with p < 0.05
            are retained; others are set to 0.

        Raises
        ------
        ValueError
            If no temporal overlap exists between precipitation and streamflow.

        Notes
        -----
        • Overlap between precipitation and streamflow calibration windows
          is enforced to avoid inconsistent correlations.
        • R² values are rounded to 2 decimals for compact visualization.
        • Non-significant correlations (p ≥ 0.05) are set to 0 by construction.

        """

        # ------------------------------------------------------------
        # 1. Find temporal overlap between precipitation and streamflow
        # ------------------------------------------------------------
        self_indices, streamflow_indices = find_overlap(self.m_cal, streamflow.m_cal)

        if len(self_indices) == 0 or len(streamflow_indices) == 0:
            raise ValueError("No overlapping data found between Precipitation and Streamflow.")

        # ------------------------------------------------------------
        # 2. Extract overlapping standardized indices
        # ------------------------------------------------------------
        # SQI₁ (streamflow): first row of streamflow.spi_like_set
        sqi1 = streamflow.spi_like_set[0, streamflow_indices]

        # SPI-like set (precipitation): all scales
        spi_like_set = self.spi_like_set[:, self_indices]

        # Month/year metadata for overlapping period
        m_cal = self.m_cal[self_indices]

        # ------------------------------------------------------------
        # 3. Compute R²(month, k) matrix
        # ------------------------------------------------------------
        R2 = []
        Ki = np.arange(self.K)  # scales 0…K-1

        for m in range(1, 13):  # calendar months
            r2_row = []

            # indices of entries belonging to month m
            ii = np.where(m_cal[:, 0] == m)[0]

            # SQI₁ for month m
            y = sqi1[ii]

            for K in Ki:
                x = spi_like_set[K][ii]

                # filter finite values to avoid NaNs in Pearsonr
                mask = (np.isfinite(x) & np.isfinite(y))

                if mask.sum() < 3:
                    # too few points to compute correlation
                    r2_row.append(0)
                    continue

                rho, pval = stats.pearsonr(x[mask], y[mask])

                # Keep only statistically significant correlations
                r2_val = np.round(rho ** 2, 2) if pval < 0.05 else 0
                r2_row.append(r2_val)

            R2.append(r2_row)

        R2 = np.array(R2)

        # ------------------------------------------------------------
        # 4. Optional plotting
        # ------------------------------------------------------------
        if plot:
            plt.figure(figsize=(10, 4))
            X, Y = np.meshgrid(np.arange(1, self.K + 1), np.arange(1, 13))

            levels = np.arange(0.05, 1.15, 0.1)
            centers = (levels[:-1] + levels[1:]) / 2

            cf = plt.contourf(X, Y, R2, cmap='pink_r', levels=levels)

            plt.xticks(np.arange(1, self.K + 1))
            plt.yticks(
                np.arange(1, 13),
                ['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May', 'Jun.',
                 'Jul.', 'Aug.', 'Sep.', 'Oct.', 'Nov.', 'Dec.']
            )

            plt.xlabel('month scale (K)')
            cbar = plt.colorbar(cf, ticks=centers)
            cbar.ax.set_yticklabels([f"{c:.2f}" for c in centers])
            cbar.set_label(r"$R^2$", rotation=0, labelpad=12, fontsize=12)

            # Minor grid for readability
            plt.gca().set_yticks(np.arange(1.5, 12, 1), minor=True)
            plt.grid(axis='y', which='minor', linestyle='--', alpha=0.7)

            plt.tight_layout()

        return R2

    def analyze_correlation(self, streamflow,plot=True,plot_mode="all"):
        """
        Analyze correlations between Precipitation SIDI and Streamflow SPI for different weightings and K values.

        Args:
            streamflow (Streamflow): Instance of the Streamflow class.
            plot (bool, optional): Whether to generate a correlation plot and call `_plot_scan`. Default is True.
            plot_mode (str): 'all' (default), 'seasonal', 'monthly'.

        Returns:
            dict: Contains the best K, weight configuration, and maximum correlation value.
                - "best_k" (int): Optimal month-scale (K).
                - "col_best_weight" (int): Index of the best weight configuration.
                - "max_correlation" (float): Maximum R^2 value achieved.
        """
        wlabel = ['equal weights (ew)', 'linearly decreasing weights (ldw)',
                  ' logarithmically decreasing weights (lgdw)', 'linearly increasing weights (liw)',
                  'logarithmically increasing weights (lgiw)']

        if not isinstance(streamflow, BaseDroughtAnalysis):
            raise TypeError("The input must be an instance of Streamflow or BaseDroughtAnalysis.")

        # find the temporal overlap between Precipitation and Streamflow
        self_indices, streamflow_indices = find_overlap(self.m_cal,streamflow.m_cal)
        if len(self_indices) == 0 or len(streamflow_indices) == 0:
            raise ValueError("No overlapping data found between Precipitation and Streamflow.")

        # Subset di dati per l'overlapping time
        y = streamflow.spi_like_set[0, streamflow_indices]  # SPI-1 dello streamflow
        spi_like_set = self.spi_like_set[:,self_indices]  # Tutte le configurazioni SIDI


        K_range = np.arange(1, self.K + 1)
        MatCorr = []

        print("Starting correlation analysis...")
        for k in K_range:
            W = generate_weights(k)
            # print("Calculating Ensemble Weighted Mean for each weighting function...")
            sidis = []  # SPI ensemble mean for each day
            for doy in range(len(spi_like_set[0])):#in range(self._baseline_indices()[0], self._baseline_indices()[1] + 1):
                vec = spi_like_set[:k, doy]
                sidis.append([weighted_metrics(vec, w)[0] for w in W.T])
            sidis = np.array(sidis)

            rr = []  # Correlations for each weighting function
            for w in range(len(W.T)):
                # Standardize SIDI sull'intero periodo perché l'overlapping è troppo variabile
                SIDI = (sidis[:, w] - np.nanmean(sidis[:, w])) / np.nanstd(sidis[:, w])
                valid_mask = np.isfinite(y) & np.isfinite(SIDI)
                r = stats.pearsonr(SIDI[valid_mask], y[valid_mask])[0]
                rr.append(r ** 2)
                # print(f"K={k}, Weight {w + 1}: R^2 = {np.round(r ** 2, 3)}")
            MatCorr.append(rr)
        # looking to the single SQI - SPI correlation
        rr_spi = []
        for j,spi in enumerate(spi_like_set):
            valid_mask = np.isfinite(y) & np.isfinite(spi)
            r = stats.pearsonr(spi[valid_mask], y[valid_mask])[0]
            rr_spi.append(r ** 2)
        rr_spi = np.array(rr_spi)
        ii = np.argsort(rr_spi)[::-1]
        R2_spi = np.array([np.arange(1,self.K+1)[ii],rr_spi[ii]]).T

        MatCorr = np.array(MatCorr)
        # Find the best K and weight index
        max_corr = np.max(MatCorr)
        best_k, best_weight = np.unravel_index(np.argmax(MatCorr), MatCorr.shape)


        print(f"Best correlation: R2  = {max_corr:.3f} (K={K_range[best_k]}, Weight={wlabel[best_weight]})")
        W = generate_weights(K_range[best_k])
        sidi = []  # SPI ensemble mean for each day
        for doy in range(len(spi_like_set[0])):  # in range(self._baseline_indices()[0], self._baseline_indices()[1] + 1):
            vec = spi_like_set[:K_range[best_k], doy]
            sidi.append(weighted_metrics(vec, W[:,best_weight])[0])
        sidi = np.array(sidi)

        SIDI = (sidi - np.nanmean(sidi)) / np.nanstd(sidi)

        # --- plots ----------------------------------------------------------------

        if plot == True:

            plt.figure(figsize=(10, 5))
            # weight_labels = ["Equal", "Linear Shallow", "Geom Shallow", "Linear Deep", "Geom Deep"]
            for w in range(len(W.T)):
                plt.plot(MatCorr[:, w], label=wlabel[w], linewidth=2)
            plt.grid()
            plt.legend(loc=3)
            plt.xticks(np.arange(len(K_range)), K_range)
            plt.ylabel(r"$R^2$", fontweight="bold", fontsize=12)
            plt.xlabel("Month-scale (K)", fontweight="bold", fontsize=12)
            plt.title(f"Correlation Analysis: {self.SIDI_name}  vs.  {streamflow.index_name}1", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.show(block=False)




            # basic scan plot
            self.plot_scan(optimal_k=K_range[best_k], weight_index=best_weight)

            plt.figure(figsize=(7, 7))
            if plot_mode == "all":
                plt.plot(SIDI, y, 'ok', markerfacecolor='yellow', linewidth=2)
            elif plot_mode == "seasonal":
                g1 = [4, 5, 6, 7, 8, 9]
                g2 = [10,11, 12, 1, 2, 3]
                # summer ------------------------------------------------------
                m1summer = np.isin(self.m_cal[self_indices, 0], g1)
                m2summer = np.isin(streamflow.m_cal[streamflow_indices, 0], g1)
                f  = np.isfinite(SIDI[m1summer]) & np.isfinite(y[m2summer])
                rho,pval = stats.pearsonr(SIDI[m1summer][f], y[m2summer][f])
                rho = 0 if pval>0.5 else rho
                plt.plot(SIDI[m1summer], y[m2summer], 'o', color='tab:olive', alpha=0.4, label=f'Apr-Oct; $R^2$ = {np.round(rho**2,2)}')
                # winter ------------------------------------------------------
                m1winter = np.isin(self.m_cal[self_indices, 0], g2)
                m2winter = np.isin(streamflow.m_cal[streamflow_indices, 0], g2)
                f = np.isfinite(SIDI[m1winter]) & np.isfinite(y[m2winter])
                rho,pval = stats.pearsonr(SIDI[m1winter][f], y[m2winter][f])
                rho = 0 if pval>0.5 else rho
                plt.plot(SIDI[m1winter], y[m2winter], 'o', color='tab:blue', alpha=0.4, label=f'Nov-Mar; $R^2$ = {np.round(rho**2,2)}')
            elif plot_mode == 'monthly':
                if cmc is not None:
                    c = plt.get_cmap(cmc.romaO, 12)
                else:
                    c = plt.get_cmap('twilight_shifted', 12)

                for month in range(1, 13):
                    m1 = np.where(self.m_cal[self_indices, 0] == month)[0]
                    m2 = np.where(streamflow.m_cal[streamflow_indices, 0] == month)[0]
                    plt.plot(SIDI[m1],y[m2], 'o',color=c(month/12),label = f'month {month}')


            plt.plot(np.arange(-3,4),np.arange(-3,4),'--',color='grey')
            plt.grid()
            plt.ylabel(f"{streamflow.index_name}1 ", fontweight="bold", fontsize=12)
            plt.xlabel(f"{self.SIDI_name}", fontweight="bold", fontsize=12)
            plt.title(f"{self.SIDI_name} vs.  {streamflow.index_name}1 . K={best_k} - weighting function n. {best_weight}; $R^2$ = {max_corr:.2f}", fontsize=14, fontweight="bold")

            plt.legend(fontsize=12)
            plt.tight_layout()
            plt.show(block=False)

        return {"best_k": K_range[best_k], "col_best_weight": best_weight, "max_correlation": max_corr,'spi_corr':R2_spi}

    def analyze_correlation_seasonal(self, streamflow, agg='quarter',plot=True,seasons=None):
        """
        Perform seasonal correlation analysis between precipitation-based SIDI and
        streamflow-based SPI indices for different weighting schemes and K (month-scale) values.

        Depending on the selected aggregation mode (`agg`), data are grouped by predefined
        temporal windows (quarterly, semiannual, four-monthly) or by a custom set of months.

        Args:
            streamflow (Streamflow):
                Instance of the Streamflow class to be compared with the precipitation dataset.
            agg (str, optional):
                Defines how data are temporally aggregated. Options are:

                - **'quarter'** *(default)* →
                  Standard climatological seasons:
                  `{'DJF':[12,1,2], 'MAM':[3,4,5], 'JJA':[6,7,8], 'SON':[9,10,11]}`

                - **'semiannual'** →
                  Two six-month periods:
                  `{'autwin':[9,10,11,12,1,2], 'springsum':[3,4,5,6,7,8]}`

                - **'four-monthly'** →
                  Three four-month periods:
                  `{'ONDJ':[10,11,12,1], 'FMAM':[2,3,4,5], 'JJAS':[6,7,8,9]}`

                -**'monthly**

                - **'custom'** →
                  User-defined aggregation; requires passing an additional dictionary
                  through the `seasons` argument, e.g.:
                  `seasons={'period1':[1,2,3,4], 'period2':[5,6,7,8]}`

            plot (bool, optional):
                If True, generate diagnostic plots showing the R²–K relationships
                and the scatter plots for the best seasonal configurations. Default is True.

            seasons (dict, optional):
                Custom month mapping to use when `agg='custom'`.
                The dictionary must have season names as keys and lists of month indices as values.

        Returns:
            dict:
                Dictionary containing seasonal correlation results. Each key corresponds to
                a season or aggregation period, and the value is a dictionary with the fields:

                - `"best_k"` *(int)* → Optimal temporal aggregation window (K)
                - `"col_best_weight"` *(int)* → Index of the best weighting function
                - `"max_correlation"` *(float)* → Maximum R² value obtained
                - `"R2_matrix"` *(np.ndarray)* → Full R² matrix for all K × weighting configurations

                Example:
                {
                    "DJF": {"best_k": 3, "col_best_weight": 1, "max_correlation": 0.72, "R2_matrix": array(...)},
                    "MAM": {...},
                    ...
                }

        Notes:
            The method automatically adjusts subplot layout and figure dimensions according
            to the number of analyzed seasons. Overlapping months between precipitation
            and streamflow datasets are used to ensure temporal consistency.

        """

        wlabel = [
            'EW',
            'Lin. DW',
            'Log. DW',
            'Lin. IW',
            'Log. IW'
        ]

        if not isinstance(streamflow, BaseDroughtAnalysis):
            raise TypeError("The input must be an instance of Streamflow or BaseDroughtAnalysis.")

        # --- Temporal overlap between Precipitation and Streamflow ---
        self_indices, streamflow_indices = find_overlap(self.m_cal, streamflow.m_cal)
        if len(self_indices) == 0 or len(streamflow_indices) == 0:
            raise ValueError("No overlapping data found between Precipitation and Streamflow.")

        # --- Subset to overlapping period ---
        y = streamflow.spi_like_set[0, streamflow_indices]  # SPI-1 (streamflow)
        spi_like_set = self.spi_like_set[:, self_indices]  # All SIDI configurations
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

        # --- Seasonal analysis ---
        if agg == 'quarter':
            seasons = {
                "DJF": [12, 1, 2],
                "MAM": [3, 4, 5],
                "JJA": [6, 7, 8],
                "SON": [9, 10, 11],
            }
            figsize1 = (14,10)
            figsize2 = (10,11)
            ncols,nrows = (2,2)

        elif (agg == 'semiannual') | (agg =='biannual'):
            seasons = {
                "autwin": [9, 10, 11,12, 1, 2],
                "springsum": [3, 4, 5,6, 7, 8],
            }
            figsize1 = (14,5)
            figsize2 = (10,6)
            ncols,nrows = (2,1)

        elif agg == 'four-monthly':
            seasons = {
                "ONDG" : [10,11, 12, 1],
                "FMAM" : [2,3,4,5],
                "JJAS" : [6,7,8,9],
            }
            figsize1 = (16,6)
            figsize2 = (15,6)
            ncols,nrows = (3,1)

        elif agg == 'monthly':
            seasons = {'Jan':[1], 'Feb': [2], 'Mar': [3], 'Apr': [4], 'May': [5], 'Jun': [6],
            'Jul': [7], 'Aug': [8], 'Sep': [9], 'Oct': [10], 'Nov': [11], 'Dec': [12]}
            figsize1 = (7,6)
            figsize2 = (7,6)
            ncols,nrows = (1,1)

        elif agg == 'custom':
            seasons = seasons
            layout_map = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2), 5: (2, 3), 6: (2, 3)}
            nrows, ncols = layout_map.get(len(seasons), (2, 3))
            wid = np.round(5*ncols,1)
            h = np.round(5*nrows,1)+0.5
            figsize1,figsize2 =(wid+3,h),(wid,h)



        # a check on the whole year
        all_months = [m for lst in seasons.values() for m in lst]
        unique = set(all_months)
        missing = [m for m in range(1, 13) if m not in unique]
        duplicates = [m for m in unique if all_months.count(m) > 1]

        if missing:
            print("allert: missing month(s):", missing)
        if duplicates:
            print("allert: duplicates month(s):", duplicates)


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
                "sample number": np.count_nonzero(idx)

            }

        # --- Plot 1: R² vs K for each weighting scheme ---
        if agg =='monthly':
            # ordine mesi: 'Jan'...'Dec'
            months = list(calendar.month_abbr[1:])  # ['Jan','Feb',...,'Dec']

            # vettore con i max R^2 mese per mese (NaN se manca)
            r2max = np.array([MatCorr.get(m, {}).get('max_correlation', np.nan) for m in months], float)

            # plot rapido (bar)
            plt.figure(figsize=(7, 5))
            plt.bar(months, r2max,0.5,alpha=0.8,facecolor='dimgrey',edgecolor='k')
            plt.ylabel(r"$\max R^2$")
            plt.ylim(0, 1)
            plt.grid(axis='y', alpha=0.3)
            plt.title(f"{self.basin_name} - Monthly max R² {self.SIDI_name} vs. {streamflow.index_name}1")
            plt.axhline(y=0.4, linestyle='--',color='tab:orange',label='Significant limit')
            plt.legend()
            # plt.text(0.10,0.45, 'Significant limit', color='tab:orange',fontweight='bold',fontsize=14)

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
                    ax[i].set_xticks(np.arange(0,len(K_range),3))
                    ax[i].set_xticklabels(K_range[0:-1:3])
                    ax[i].tick_params(axis='x', labelsize=14)
                    ax[i].tick_params(axis='y', labelsize=14)
                    ax[i].set_ylabel(r"$R^2$", fontweight="bold", fontsize=16)
                    ax[i].set_xlabel("Month-scale (K)", fontweight="bold", fontsize=16)
                    ax[i].set_title(name, fontweight="bold", fontsize=16)
                    if i==0:
                        ax[i].legend(loc=3)
                fig.suptitle(
                    f"{self.basin_name} - Correlation Analysis: {self.SIDI_name} vs. {streamflow.index_name}1",
                    fontsize=16, fontweight="bold"
                )
                plt.tight_layout()
                plt.show(block=False)

            # --- Plot 2: Scatter plots (best configuration per season) ---
                if len(seasons)>5:
                    if cmc is not None:
                        c = plt.get_cmap(cmc.romaO, len(seasons))
                    else:
                        c = plt.get_cmap('twilight_shifted', len(seasons))
                else:
                    c = ['tab:olive', 'tab:brown', 'tab:orange', 'tab:cyan','tab:purple']
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

                    ax[i].plot(SIDI[idx], y[idx], 'o', color=c[i], alpha=0.4, label=f'$R^2$ = {np.round(r2, 2)} \n K ={best_k}; {wlabel[best_weight]}')
                    ax[i].plot(np.arange(-3, 4), np.arange(-3, 4), '--', color='grey')
                    ax[i].tick_params(axis='x', labelsize=14)
                    ax[i].tick_params(axis='y', labelsize=14)
                    ax[i].grid()
                    ax[i].set_title(season, fontweight="bold", fontsize=16)
                    ax[i].set_ylabel(f"{streamflow.index_name}1", fontweight="bold", fontsize=16)
                    ax[i].set_xlabel(f"{self.SIDI_name}", fontweight="bold", fontsize=16)
                    ax[i].legend(fontsize=12)

                fig.suptitle(
                    f"{self.basin_name} - {self.SIDI_name} vs. {streamflow.index_name}1 — Best seasonal configurations",
                    fontsize=14, fontweight="bold"
                )
                if i < ncols*nrows-1:
                    fig.delaxes(ax[-1])
                plt.tight_layout()
                plt.show(block=False)

        return MatCorr


class Balance(BaseDroughtAnalysis):
    def __init__(self, start_baseline_year, end_baseline_year, basin_name, prec_path=None, pet_path=None,
                 shape_path=None, shape=None, ts=None, m_cal=None, K=None,
                 calculation_method=f_kde, threshold=None, index_name = 'SPEI',verbose=True):
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
            calculation_method (callable, optional): Method to use for drought calculations. Default is f_spei.
            verbose (bool, optional): Whether to print initialization messages. Default is True.
        """
        self.start_baseline_year = start_baseline_year
        self.end_baseline_year = end_baseline_year
        self.verbose = verbose
        self.basin_name=basin_name

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
        elif prec_path is not None and pet_path is not None and self.shape is not None:
            self.prec_path = prec_path
            self.pet_path = pet_path
            self.prec_grid, self.pet_grid, self.m_cal, self.ts= self._import_data() #only over common timeframe
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
                         self.basin_name,self.calculation_method, self.threshold, self.index_name)

        if verbose:
            print("#########################################################################")
            print("Welcome to Drought Scan! \n")
            print("The water balance data (P - PET) has been calculated successfully.")
            print(f"Your data starts from {self.m_cal[0]} and ends on {self.m_cal[-1]}.")
            print("#########################################################################")
            print("Run the following class methods to access key functionalities:\n")
            print(" >>> ._plot_scan(): to plot the CDN, SPEI heatmap, and D_{SPEI} \n")



    def _import_data(self):
        """
        Import precipitation and PET data, aligning them on a common calendar.

        Returns:
            tuple: (prec_grid, pet_grid, m_cal, ts)
                - prec_grid: Gridded precipitation data.
                - pet_grid: Gridded PET data.
                - m_cal: Common calendar.
                - ts: Time series of precipitation minus PET.
        """

        # import preciptiation data
        prec_ts, prec_cal, Pgrid = import_netcdf_for_cumulative_variable(
            self.prec_path,
            possible_names=['tp','rr','precipitation','prec','LAPrec1871','pre','swe','SWE','sd','SD','sf','SF'],  # Possibili nomi della variabile
            shape=self.shape,
            verbose=self.verbose
        )

        # import PET data
        pet_ts, pet_cal, ETgrid = import_netcdf_for_cumulative_variable(
            self.pet_path,
            possible_names=['e', 'ET','PET','pet','et','evaporation',
                                                 'evapotranspiration','potential evapotranspiration',
                                                 'reference evapotranspiration','swe','pev'],
            shape=self.shape,
            verbose=self.verbose
        )
        # align the timestamp
        p_id, pet_id = find_overlap(prec_cal, pet_cal)
        if not p_id.size:
            raise ValueError("No common dates found between precipitation and PET datasets.")

            # Allinea i dati secondo le date comuni
        m_cal = prec_cal[p_id]
        Pgrid = Pgrid[p_id, :, :]
        ETgrid = ETgrid[pet_id, :, :]


        # Calcola la differenza tra precipitazione e PET
        ts = prec_ts[p_id] - pet_ts[pet_id]
        return Pgrid, ETgrid, m_cal, ts

    def spei_sqi_corr(self, streamflow, plot=True):
        """
        Compute the month-wise correlation between SPEIₖ (k = 1…K) and SQI₁
        and optionally plot the R² heatmap.

        This function quantifies how precipitation-based SPI-like indices
        propagate into hydrological drought (SQI-like) on a month-by-month basis.
        For each calendar month (Jan…Dec), the function computes the
        Pearson correlation between SPEIₖ and SQI₁ for all k in [1, K],
        returning an R² matrix of shape (12 × K).

        Parameters
        ----------
        streamflow : object
            A DroughtScan-compatible object containing:
            - `m_cal`: (N, 2) array of [month, year] metadata
            - `spi_like_set`: 2D array of standardized streamflow indices,
               where spi_like_set[0] corresponds to SQI₁.
        plot : bool, optional (default=False)
            If True, produces a contourf heatmap of R²(month, k).

        Returns
        -------
        R2 : ndarray, shape (12, K)
            Matrix of determination coefficients R² for each month (rows)
            and SPI scale k (columns). Only correlations with p < 0.05
            are retained; others are set to 0.

        Raises
        ------
        ValueError
            If no temporal overlap exists between precipitation and streamflow.

        Notes
        -----
        • Overlap between precipitation and streamflow calibration windows
          is enforced to avoid inconsistent correlations.
        • R² values are rounded to 2 decimals for compact visualization.
        • Non-significant correlations (p ≥ 0.05) are set to 0 by construction.

        """

        # ------------------------------------------------------------
        # 1. Find temporal overlap between precipitation and streamflow
        # ------------------------------------------------------------
        self_indices, streamflow_indices = find_overlap(self.m_cal, streamflow.m_cal)

        if len(self_indices) == 0 or len(streamflow_indices) == 0:
            raise ValueError("No overlapping data found between Precipitation and Streamflow.")

        # ------------------------------------------------------------
        # 2. Extract overlapping standardized indices
        # ------------------------------------------------------------
        # SQI₁ (streamflow): first row of streamflow.spi_like_set
        sqi1 = streamflow.spi_like_set[0, streamflow_indices]

        # SPI-like set (precipitation): all scales
        spi_like_set = self.spi_like_set[:, self_indices]

        # Month/year metadata for overlapping period
        m_cal = self.m_cal[self_indices]

        # ------------------------------------------------------------
        # 3. Compute R²(month, k) matrix
        # ------------------------------------------------------------
        R2 = []
        Ki = np.arange(self.K)  # scales 0…K-1

        for m in range(1, 13):  # calendar months
            r2_row = []

            # indices of entries belonging to month m
            ii = np.where(m_cal[:, 0] == m)[0]

            # SQI₁ for month m
            y = sqi1[ii]

            for K in Ki:
                x = spi_like_set[K][ii]

                # filter finite values to avoid NaNs in Pearsonr
                mask = (np.isfinite(x) & np.isfinite(y))

                if mask.sum() < 3:
                    # too few points to compute correlation
                    r2_row.append(0)
                    continue

                rho, pval = stats.pearsonr(x[mask], y[mask])

                # Keep only statistically significant correlations
                r2_val = np.round(rho ** 2, 2) if pval < 0.05 else 0
                r2_row.append(r2_val)

            R2.append(r2_row)

        R2 = np.array(R2)

        # ------------------------------------------------------------
        # 4. Optional plotting
        # ------------------------------------------------------------
        if plot:
            plt.figure(figsize=(10, 4))
            X, Y = np.meshgrid(np.arange(1, self.K + 1), np.arange(1, 13))

            levels = np.arange(0.05, 1.15, 0.1)
            centers = (levels[:-1] + levels[1:]) / 2

            cf = plt.contourf(X, Y, R2, cmap='pink_r', levels=levels)

            plt.xticks(np.arange(1, self.K + 1))
            plt.yticks(
                np.arange(1, 13),
                ['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May', 'Jun.',
                 'Jul.', 'Aug.', 'Sep.', 'Oct.', 'Nov.', 'Dec.']
            )

            plt.xlabel('month scale (K)')
            cbar = plt.colorbar(cf, ticks=centers)
            cbar.ax.set_yticklabels([f"{c:.2f}" for c in centers])
            cbar.set_label(r"$R^2$", rotation=0, labelpad=12, fontsize=12)

            # Minor grid for readability
            plt.gca().set_yticks(np.arange(1.5, 12, 1), minor=True)
            plt.grid(axis='y', which='minor', linestyle='--', alpha=0.7)

            plt.tight_layout()

        return R2

    def analyze_correlation(self, streamflow,plot=True,plot_mode="all"):
        """
        Analyze correlations between Precipitation SIDI and Streamflow SPI for different weightings and K values.

        Args:
            streamflow (Streamflow): Instance of the Streamflow class.
            plot (bool, optional): Whether to generate a correlation plot and call `_plot_scan`. Default is True.
            plot_mode (str): 'all' (default), 'seasonal', 'monthly'.

        Returns:
            dict: Contains the best K, weight configuration, and maximum correlation value.
                - "best_k" (int): Optimal month-scale (K).
                - "col_best_weight" (int): Index of the best weight configuration.
                - "max_correlation" (float): Maximum R^2 value achieved.
        """
        wlabel = ['equal weights (ew)', 'linearly decreasing weights (ldw)',
                  ' logarithmically decreasing weights (lgdw)', 'linearly increasing weights (liw)',
                  'logarithmically increasing weights (lgiw)']

        if not isinstance(streamflow, BaseDroughtAnalysis):
            raise TypeError("The input must be an instance of Streamflow or BaseDroughtAnalysis.")

        # find the temporal overlap between Precipitation and Streamflow
        self_indices, streamflow_indices = find_overlap(self.m_cal,streamflow.m_cal)
        if len(self_indices) == 0 or len(streamflow_indices) == 0:
            raise ValueError("No overlapping data found between Precipitation and Streamflow.")

        # Subset di dati per l'overlapping time
        y = streamflow.spi_like_set[0, streamflow_indices]  # SPI-1 dello streamflow
        spi_like_set = self.spi_like_set[:,self_indices]  # Tutte le configurazioni SIDI


        K_range = np.arange(1, self.K + 1)
        MatCorr = []

        print("Starting correlation analysis...")
        for k in K_range:
            W = generate_weights(k)
            # print("Calculating Ensemble Weighted Mean for each weighting function...")
            sidis = []  # SPI ensemble mean for each day
            for doy in range(len(spi_like_set[0])):#in range(self._baseline_indices()[0], self._baseline_indices()[1] + 1):
                vec = spi_like_set[:k, doy]
                sidis.append([weighted_metrics(vec, w)[0] for w in W.T])
            sidis = np.array(sidis)

            rr = []  # Correlations for each weighting function
            for w in range(len(W.T)):
                # Standardize SIDI sull'intero periodo perché l'overlapping è troppo variabile
                SIDI = (sidis[:, w] - np.nanmean(sidis[:, w])) / np.nanstd(sidis[:, w])
                valid_mask = np.isfinite(y) & np.isfinite(SIDI)
                r = stats.pearsonr(SIDI[valid_mask], y[valid_mask])[0]
                rr.append(r ** 2)
                # print(f"K={k}, Weight {w + 1}: R^2 = {np.round(r ** 2, 3)}")
            MatCorr.append(rr)
        # looking to the single SQI - SPI correlation
        rr_spi = []
        for j,spi in enumerate(spi_like_set):
            valid_mask = np.isfinite(y) & np.isfinite(spi)
            r = stats.pearsonr(spi[valid_mask], y[valid_mask])[0]
            rr_spi.append(r ** 2)
        rr_spi = np.array(rr_spi)
        ii = np.argsort(rr_spi)[::-1]
        R2_spi = np.array([np.arange(1,self.K+1)[ii],rr_spi[ii]]).T

        MatCorr = np.array(MatCorr)
        # Find the best K and weight index
        max_corr = np.max(MatCorr)
        best_k, best_weight = np.unravel_index(np.argmax(MatCorr), MatCorr.shape)


        print(f"Best correlation: R2  = {max_corr:.3f} (K={K_range[best_k]}, Weight={wlabel[best_weight]})")
        W = generate_weights(K_range[best_k])
        sidi = []  # SPI ensemble mean for each day
        for doy in range(len(spi_like_set[0])):  # in range(self._baseline_indices()[0], self._baseline_indices()[1] + 1):
            vec = spi_like_set[:K_range[best_k], doy]
            sidi.append(weighted_metrics(vec, W[:,best_weight])[0])
        sidi = np.array(sidi)

        SIDI = (sidi - np.nanmean(sidi)) / np.nanstd(sidi)

        # --- plots ----------------------------------------------------------------

        if plot == True:

            plt.figure(figsize=(10, 5))
            # weight_labels = ["Equal", "Linear Shallow", "Geom Shallow", "Linear Deep", "Geom Deep"]
            for w in range(len(W.T)):
                plt.plot(MatCorr[:, w], label=wlabel[w], linewidth=2)
            plt.grid()
            plt.legend(loc=3)
            plt.xticks(np.arange(len(K_range)), K_range)
            plt.ylabel(r"$R^2$", fontweight="bold", fontsize=12)
            plt.xlabel("Month-scale (K)", fontweight="bold", fontsize=12)
            plt.title(f"Correlation Analysis: {self.SIDI_name}  vs.  {streamflow.index_name}1", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.show(block=False)




            # basic scan plot
            self.plot_scan(optimal_k=K_range[best_k], weight_index=best_weight)

            plt.figure(figsize=(7, 7))
            if plot_mode == "all":
                plt.plot(SIDI, y, 'ok', markerfacecolor='yellow', linewidth=2)
            elif plot_mode == "seasonal":
                g1 = [4, 5, 6, 7, 8, 9]
                g2 = [10,11, 12, 1, 2, 3]
                # summer ------------------------------------------------------
                m1summer = np.isin(self.m_cal[self_indices, 0], g1)
                m2summer = np.isin(streamflow.m_cal[streamflow_indices, 0], g1)
                f  = np.isfinite(SIDI[m1summer]) & np.isfinite(y[m2summer])
                rho,pval = stats.pearsonr(SIDI[m1summer][f], y[m2summer][f])
                rho = 0 if pval>0.5 else rho
                plt.plot(SIDI[m1summer], y[m2summer], 'o', color='tab:olive', alpha=0.4, label=f'Apr-Oct; $R^2$ = {np.round(rho**2,2)}')
                # winter ------------------------------------------------------
                m1winter = np.isin(self.m_cal[self_indices, 0], g2)
                m2winter = np.isin(streamflow.m_cal[streamflow_indices, 0], g2)
                f = np.isfinite(SIDI[m1winter]) & np.isfinite(y[m2winter])
                rho,pval = stats.pearsonr(SIDI[m1winter][f], y[m2winter][f])
                rho = 0 if pval>0.5 else rho
                plt.plot(SIDI[m1winter], y[m2winter], 'o', color='tab:blue', alpha=0.4, label=f'Nov-Mar; $R^2$ = {np.round(rho**2,2)}')
            elif plot_mode == 'monthly':
                if cmc is not None:
                    c = plt.get_cmap(cmc.romaO, 12)
                else:
                    c = plt.get_cmap('twilight_shifted', 12)

                for month in range(1, 13):
                    m1 = np.where(self.m_cal[self_indices, 0] == month)[0]
                    m2 = np.where(streamflow.m_cal[streamflow_indices, 0] == month)[0]
                    plt.plot(SIDI[m1],y[m2], 'o',color=c(month/12),label = f'month {month}')


            plt.plot(np.arange(-3,4),np.arange(-3,4),'--',color='grey')
            plt.grid()
            plt.ylabel(f"{streamflow.index_name}1 ", fontweight="bold", fontsize=12)
            plt.xlabel(f"{self.SIDI_name}", fontweight="bold", fontsize=12)
            plt.title(f"{self.SIDI_name} vs.  {streamflow.index_name}1 . K={best_k} - weighting function n. {best_weight}; $R^2$ = {max_corr:.2f}", fontsize=14, fontweight="bold")

            plt.legend(fontsize=12)
            plt.tight_layout()
            plt.show(block=False)

        return {"best_k": K_range[best_k], "col_best_weight": best_weight, "max_correlation": max_corr,'spi_corr':R2_spi}

    def analyze_correlation_seasonal(self, streamflow, agg='quarter',plot=True,seasons=None):
        """
        Perform seasonal correlation analysis between precipitation-based SIDI and
        streamflow-based SPI indices for different weighting schemes and K (month-scale) values.

        Depending on the selected aggregation mode (`agg`), data are grouped by predefined
        temporal windows (quarterly, semiannual, four-monthly) or by a custom set of months.

        Args:
            streamflow (Streamflow):
                Instance of the Streamflow class to be compared with the precipitation dataset.
            agg (str, optional):
                Defines how data are temporally aggregated. Options are:

                - **'quarter'** *(default)* →
                  Standard climatological seasons:
                  `{'DJF':[12,1,2], 'MAM':[3,4,5], 'JJA':[6,7,8], 'SON':[9,10,11]}`

                - **'semiannual'** →
                  Two six-month periods:
                  `{'autwin':[9,10,11,12,1,2], 'springsum':[3,4,5,6,7,8]}`

                - **'four-monthly'** →
                  Three four-month periods:
                  `{'ONDJ':[10,11,12,1], 'FMAM':[2,3,4,5], 'JJAS':[6,7,8,9]}`

                -**'monthly**

                - **'custom'** →
                  User-defined aggregation; requires passing an additional dictionary
                  through the `seasons` argument, e.g.:
                  `seasons={'period1':[1,2,3,4], 'period2':[5,6,7,8]}`

            plot (bool, optional):
                If True, generate diagnostic plots showing the R²–K relationships
                and the scatter plots for the best seasonal configurations. Default is True.

            seasons (dict, optional):
                Custom month mapping to use when `agg='custom'`.
                The dictionary must have season names as keys and lists of month indices as values.

        Returns:
            dict:
                Dictionary containing seasonal correlation results. Each key corresponds to
                a season or aggregation period, and the value is a dictionary with the fields:

                - `"best_k"` *(int)* → Optimal temporal aggregation window (K)
                - `"col_best_weight"` *(int)* → Index of the best weighting function
                - `"max_correlation"` *(float)* → Maximum R² value obtained
                - `"R2_matrix"` *(np.ndarray)* → Full R² matrix for all K × weighting configurations

                Example:
                {
                    "DJF": {"best_k": 3, "col_best_weight": 1, "max_correlation": 0.72, "R2_matrix": array(...)},
                    "MAM": {...},
                    ...
                }

        Notes:
            The method automatically adjusts subplot layout and figure dimensions according
            to the number of analyzed seasons. Overlapping months between precipitation
            and streamflow datasets are used to ensure temporal consistency.

        """

        wlabel = [
            'EW',
            'Lin. DW',
            'Log. DW',
            'Lin. IW',
            'Log. IW'
        ]

        if not isinstance(streamflow, BaseDroughtAnalysis):
            raise TypeError("The input must be an instance of Streamflow or BaseDroughtAnalysis.")

        # --- Temporal overlap between Precipitation and Streamflow ---
        self_indices, streamflow_indices = find_overlap(self.m_cal, streamflow.m_cal)
        if len(self_indices) == 0 or len(streamflow_indices) == 0:
            raise ValueError("No overlapping data found between Precipitation and Streamflow.")

        # --- Subset to overlapping period ---
        y = streamflow.spi_like_set[0, streamflow_indices]  # SPI-1 (streamflow)
        spi_like_set = self.spi_like_set[:, self_indices]  # All SIDI configurations
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

        # --- Seasonal analysis ---
        if agg == 'quarter':
            seasons = {
                "DJF": [12, 1, 2],
                "MAM": [3, 4, 5],
                "JJA": [6, 7, 8],
                "SON": [9, 10, 11],
            }
            figsize1 = (14,10)
            figsize2 = (10,11)
            ncols,nrows = (2,2)

        elif (agg == 'semiannual') | (agg =='biannual'):
            seasons = {
                "autwin": [9, 10, 11,12, 1, 2],
                "springsum": [3, 4, 5,6, 7, 8],
            }
            figsize1 = (14,5)
            figsize2 = (10,6)
            ncols,nrows = (2,1)

        elif agg == 'four-monthly':
            seasons = {
                "ONDG" : [10,11, 12, 1],
                "FMAM" : [2,3,4,5],
                "JJAS" : [6,7,8,9],
            }
            figsize1 = (16,6)
            figsize2 = (15,6)
            ncols,nrows = (3,1)

        elif agg == 'monthly':
            seasons = {'Jan':[1], 'Feb': [2], 'Mar': [3], 'Apr': [4], 'May': [5], 'Jun': [6],
            'Jul': [7], 'Aug': [8], 'Sep': [9], 'Oct': [10], 'Nov': [11], 'Dec': [12]}
            figsize1 = (7,6)
            figsize2 = (7,6)
            ncols,nrows = (1,1)

        elif agg == 'custom':
            seasons = seasons
            layout_map = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2), 5: (2, 3), 6: (2, 3)}
            nrows, ncols = layout_map.get(len(seasons), (2, 3))
            wid = np.round(5*ncols,1)
            h = np.round(5*nrows,1)+0.5
            figsize1,figsize2 =(wid+3,h),(wid,h)



        # a check on the whole year
        all_months = [m for lst in seasons.values() for m in lst]
        unique = set(all_months)
        missing = [m for m in range(1, 13) if m not in unique]
        duplicates = [m for m in unique if all_months.count(m) > 1]

        if missing:
            print("allert: missing month(s):", missing)
        if duplicates:
            print("allert: duplicates month(s):", duplicates)


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
                "sample number": np.count_nonzero(idx)

            }

        # --- Plot 1: R² vs K for each weighting scheme ---
        if agg =='monthly':
            # ordine mesi: 'Jan'...'Dec'
            months = list(calendar.month_abbr[1:])  # ['Jan','Feb',...,'Dec']

            # vettore con i max R^2 mese per mese (NaN se manca)
            r2max = np.array([MatCorr.get(m, {}).get('max_correlation', np.nan) for m in months], float)

            # plot rapido (bar)
            plt.figure(figsize=(7, 5))
            plt.bar(months, r2max,0.5,alpha=0.8,facecolor='dimgrey',edgecolor='k')
            plt.ylabel(r"$\max R^2$")
            plt.ylim(0, 1)
            plt.grid(axis='y', alpha=0.3)
            plt.title(f"{self.basin_name} - Monthly max R² {self.SIDI_name} vs. {streamflow.index_name}1")
            plt.axhline(y=0.4, linestyle='--',color='tab:orange',label='Significant limit')
            plt.legend()
            # plt.text(0.10,0.45, 'Significant limit', color='tab:orange',fontweight='bold',fontsize=14)

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
                    ax[i].set_xticks(np.arange(0,len(K_range),3))
                    ax[i].set_xticklabels(K_range[0:-1:3])
                    ax[i].tick_params(axis='x', labelsize=14)
                    ax[i].tick_params(axis='y', labelsize=14)
                    ax[i].set_ylabel(r"$R^2$", fontweight="bold", fontsize=16)
                    ax[i].set_xlabel("Month-scale (K)", fontweight="bold", fontsize=16)
                    ax[i].set_title(name, fontweight="bold", fontsize=16)
                    if i==0:
                        ax[i].legend(loc=3)
                fig.suptitle(
                    f"{self.basin_name} - Correlation Analysis: {self.SIDI_name} vs. {streamflow.index_name}1",
                    fontsize=16, fontweight="bold"
                )
                plt.tight_layout()
                plt.show(block=False)

            # --- Plot 2: Scatter plots (best configuration per season) ---
                if len(seasons)>5:
                    if cmc is not None:
                        c = plt.get_cmap(cmc.romaO, len(seasons))
                    else:
                        c = plt.get_cmap('twilight_shifted', len(seasons))
                else:
                    c = ['tab:olive', 'tab:brown', 'tab:orange', 'tab:cyan','tab:purple']
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

                    ax[i].plot(SIDI[idx], y[idx], 'o', color=c[i], alpha=0.4, label=f'$R^2$ = {np.round(r2, 2)} \n K ={best_k}; {wlabel[best_weight]}')
                    ax[i].plot(np.arange(-3, 4), np.arange(-3, 4), '--', color='grey')
                    ax[i].tick_params(axis='x', labelsize=14)
                    ax[i].tick_params(axis='y', labelsize=14)
                    ax[i].grid()
                    ax[i].set_title(season, fontweight="bold", fontsize=16)
                    ax[i].set_ylabel(f"{streamflow.index_name}1", fontweight="bold", fontsize=16)
                    ax[i].set_xlabel(f"{self.SIDI_name}", fontweight="bold", fontsize=16)
                    ax[i].legend(fontsize=12)

                fig.suptitle(
                    f"{self.basin_name} - {self.SIDI_name} vs. {streamflow.index_name}1 — Best seasonal configurations",
                    fontsize=14, fontweight="bold"
                )
                if i < ncols*nrows-1:
                    fig.delaxes(ax[-1])
                plt.tight_layout()
                plt.show(block=False)

        return MatCorr



class Temperature(BaseDroughtAnalysis):
    def __init__(self, start_baseline_year, end_baseline_year,
                 basin_name, data_path=None, shape_path=None, ts=None, m_cal=None,
                 shape=None, K=None, weight_index=None,
                 calculation_method =f_kde,threshold=None, index_name = 'STI',verbose=True):
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
            weight_index (int, optional): Index of the weighting scheme to use for calculations.
                - weight_index = 0: Equal weights
                - weight_index = 1: Linear decreasing weights
                - weight_index = 2: Logarithmically decreasing weights (default)
                - weight_index = 3: Linear increasing weights
                - weight_index = 4: Logarithmically increasing weights

            threshold (int, optional): Threshold to define severe events, Default is -1.
            calculation_method (callable, optional): Method to use for drought calculations. Default is f_zscore.
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
        elif data_path is not None and self.shape is not None:
            self.data_path = data_path
            self.ts, self.m_cal, self.Tgrid = import_netcdf_for_cumulative_variable(data_path,
                                                ['deg0l','tg','tm','tx','t2m','d2m','mn2t'],
                                                self.shape,self.verbose,cumulate=False)
        else:
            raise ValueError("Provide either ts and m_cal directly or specify data_path for gridded PET data in NetCDF format along with the path of the river shapefile.")

        if K is None:
            self.K = 36

        self.threshold = 1 if threshold is None else threshold

        if weight_index is None:
            self.weight_index = 2

        if not callable(calculation_method):
            raise ValueError("`calculation_method` must be a callable function.")
        self.calculation_method = calculation_method
        self.index_name = index_name

        super().__init__(self.ts, self.m_cal, self.K, self.start_baseline_year, self.end_baseline_year,
                         self.basin_name, self.calculation_method, self.threshold, self.index_name)

        if verbose:
            print("#########################################################################")
            print("Welcome to Drought Scan! \n")
            print("The PET data has been imported successfully.")
            print(f"Your data starts from {self.m_cal[0]} and ends on {self.m_cal[-1]}.")
            print("#########################################################################")
            print("Run the following class methods to access key functionalities:\n")
            print(" >>> ._plot_scan(): to plot the CDN, zscore heatmap, and D_{zscore} \n")


class Teleindex(BaseDroughtAnalysis):
    def __init__(self, start_baseline_year, end_baseline_year, basin_name=None,ts=None, m_cal=None, data_path=None,
                 K=None, weight_index=None,
                 calculation_method=f_kde, threshold=None, verbose=True, index_name=''):

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
                "Provide either ts and m_cal directly or specify data_path for a gridded precipitation data in NetCDF format along with the path of the river shapefile.")

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
                         self.basin_name,self.calculation_method, self.threshold, self.index_name)

        # Welcome and guidance messages
        if verbose:
            print("#########################################################################")
            print("Welcome to Drought Scan! \n")
            print("The precipitation data has been imported successfully.")
            print(f"Your data starts from {self.m_cal[0]} and ends on {self.m_cal[-1]}.")
            print("#########################################################################")
            print("Run the following class methods to access key functionalities:\n")
            print(" >>> ._plot_scan(): to plot the CDN, spiset heatmap, and D_{SPI} \n ")
            print(
                " >>> ._analyze_correlation(): to estimate the best K and weighting function (only if streamflow data are available) \n")
            print(
                "*************** Alternatively, you can access to: \n >>> precipitation.ts (P timeseries), \n >>> precipitation.spi_like_set (SPI (1:K) timeseries) \n >>> precipitation.SIDI (D_{SPI}) \n to visualize the data your way or proceed with further analyses!")



if __name__ == "__main__":
    print("This module contains the main classes for computing SPI, SIDI, and CDN indices.")
    print("Import the classes into an external script to use them in your project.")
