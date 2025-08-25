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


import numpy as np
import os
os.environ['USE_PYGEOS'] = '0'
from matplotlib.colors import ListedColormap
import json

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

        # SPI-related attributes
        self.spi_like_set, self.c2r_index = self._calculate_spi_like_set()
        self.SIDI = self._calculate_SIDI()
        self.CDN = self._calculate_CDN()

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

        if self.calculation_method in [f_spi, f_spei]:
            c2rspi = np.zeros((12, 4), dtype=float)
            way = 1
        elif self.calculation_method == f_zscore:
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

        if self.calculation_method in [f_spi, f_spei]:
            c2rspi = np.zeros((self.K, 12, 4), dtype=float)
        elif self.calculation_method == f_zscore:
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

    def recalculate_SIDI(self, K=None, weight_index = None, overwrite=False):
        """
        Recalculate SIDI using a custom K for each weight_index without altering the original SPI set.

        Args:
            K (int, optional): the optimal number of SPI scales to use for the SIDI recalculation.
                               If None, defaults to self.K.

            weight_index (int,optional): the optimal weight_index according to the SIDI vs SQI1 correlation
                                if None, defaults is self.weight_index

            overwrite (bool): If True, updates self.SIDI with the new values and self will be enriched by
                            self.optimal_k
                            self.optimal_weight_index
                            thus, self.optimal_k and self.weight_index will be the track change for SIDI
                              If False, returns the recalculated SIDI without modifying the object.

        Returns:
            np.ndarray: New SIDI array (time x weightings) if overwrite=False.
        """
        K = self.K if K is None else K
        weight_index = self.weight_index if weight_index is None else weight_index
        weights = generate_weights(K)

        # Pre-allocate SIDI matrix
        sidi_matrix = []
        for j in range(len(self.m_cal)):
            vec = self.spi_like_set[:K, j]  # Use only the top-K SPI values
            sidi_w = [weighted_metrics(vec, w)[0] for w in weights.T]
            sidi_matrix.append(sidi_w)

        sidi_matrix = np.array(sidi_matrix)

        # Standardize using the original baseline period
        tb1_id, tb2_id = baseline_indices(self.m_cal, self.start_baseline_year, self.end_baseline_year)
        baseline = sidi_matrix[tb1_id:tb2_id + 1, :]
        mean = np.nanmean(baseline, axis=0)
        std = np.nanstd(baseline, axis=0)
        if np.any(std == 0):
            raise ValueError("Zero std in baseline; cannot standardize.")
        SIDI_new = (sidi_matrix - mean) / std

        if overwrite:
            self.SIDI = SIDI_new
            print(f"Overwrote self.SIDI with recalculated values using K={K} and weight_index = {weight_index}")
            self.optimal_k = K
            self.optimal_weight_index = weight_index

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

    def plot_scan(self, optimal_k=None, weight_index=None,year_ext=None,reverse_color=False,saveplot=False):
        """
            Plot the drought scan visualization, including CDN, SPI-like heatmap, and SIDI.

            Args:
                year_ext (tuple, optional): Years defining X-axis limits.
                optimal_k (int, optional): Optimal K scale.
                weight_index (int, optional): Weighting scheme index.
                name (str, optional): Name of the basin.
                index_name (str, optional): Name of the drought index (SPI, SPEI, Z-Score).
                reverse_color (bool, optional): If True, reverse color maps and highlight upper anomalies (suggested for PET).

            """
        plot_overview(self, optimal_k=optimal_k, weight_index=weight_index,year_ext=year_ext,reverse_color=reverse_color)
        if saveplot==True:
            self._saveplot()
    def normal_values(self):
        """
          Compute the "normal" values of the variable using the inverse function of the SPI-like index.

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

    def find_trends(self, Y=None, window=None):
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
        if Y is None:
            Y = self.CDN
        results = rolling_trend_analysis(Y, window=window, significance=0.05)
        return results

    def plot_trends(self, windows=[12, 36, 60, 120],ax=None,year_ext=None):
        """
        Wrapper method to plot trend bars on the CDN time series for a DroughtScan-compatible object.

        Args:
            windows (list of int, optional): List of window lengths (in months) over which to evaluate trends.
                                             Default is [12, 36, 60, 120].

        Returns:
            None. Displays a plot.
        """
        plot_cdn_trends(self, windows,ax=ax,year_ext=year_ext)

    def plot_monthly_profile(self, var=None, cumulate=False, highlight_years=None,two_year=False):
        """
        Plot a 24-month profile of a time series, with percentile bands and optional highlighted years.

        Parameters
        ----------
        var : np.ndarray or None, optional
            The time series to analyze. If None, `self.ts` will be used as default.
            Must be a 1D array with the same length as `self.m_cal`.

        cumulate : bool, default=False
            If True, compute and display the cumulative sum per month for each year.

        highlight_years : list of int or int or None, optional
            One or more years to be highlighted in the plot.

        name : str or None, optional
            Optional label to include in the plot title.

        Returns
        -------
        None
            Displays the plot.
        """

        monthly_profile(self, var=var, cumulate=cumulate, highlight_years=highlight_years, two_year=two_year)

    def export_for_r_plot(self, weight_index=2, optimal_k=None, name=None, out_dir="exports"):
        """
        Esporta i dati minimi necessari per replicare il grafico plot_overview in R.

        Args:
            DSO: Oggetto DroughtScan (Precipitation o Streamflow)
            weight_index (int): indice dei pesi per la SIDI (default: log decrescente = 2)
            optimal_k (int, optional): se specificato, calcola nuova SIDI con K ottimale
            name (str, optional): nome del bacino per nome file
            out_dir (str): directory in cui salvare i dati esportati
        """
        os.makedirs(out_dir, exist_ok=True)

        # Serie temporali
        df_mcal = pd.DataFrame(self.m_cal, columns=["month", "year"])
        df_cdn = pd.DataFrame({"CDN": self.CDN})
        spi_df = pd.DataFrame(self.spi_like_set)

        # SIDI: ricalcola se richiesto
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

        # Parametri generali
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
            try:
                area_kmq = float(self.shape.to_crs(epsg=32632).geometry.area.iloc[0]) / 1e6
                metadata["area_kmq"] = area_kmq
            except Exception as e:
                print(f"Impossibile calcolare area: {e}")

        # Salvataggio
        prefix = name.replace(" ", "_") if name else "DSO_export"
        df_mcal.to_csv(os.path.join(out_dir, f"{prefix}_m_cal.csv"), index=False)
        df_cdn.to_csv(os.path.join(out_dir, f"{prefix}_cdn.csv"), index=False)
        spi_df.to_csv(os.path.join(out_dir, f"{prefix}_spi.csv"), index=False, header=False)
        df_sidi.to_csv(os.path.join(out_dir, f"{prefix}_sidi.csv"), index=False)

        with open(os.path.join(out_dir, f"{prefix}_meta.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Dati esportati con successo in {out_dir}/ con prefisso '{prefix}'")
    # ----------------------------------------------------------
    def _saveplot(self):

        k = self.K if not hasattr(self, 'optimal_k') or self.optimal_k is None else self.optimal_k
        w = self.weight_index if not hasattr(self, 'optimal_weight_index') or self.optimal_weight_index is None else self.optimal_weight_index
        baseline =self.start_baseline_year,self.end_baseline_year
        fname=f"DS_{self.basin_name}_k{k}_w{w}_baseline{baseline}.png"
        print(f"saving plot in {os.getcwd()}")
        plt.savefig(fname,
        dpi=300,
        facecolor='w',
        edgecolor='w',
        bbox_inches='tight', #“tight”; None
        pad_inches=0.1, #specifies padding around the image when bbox_inches is “tight”.
        # frameon=None,
        metadata=None)
class Precipitation(BaseDroughtAnalysis):
    def __init__(self, start_baseline_year, end_baseline_year,basin_name,ts=None,m_cal=None,prec_path=None,
                 shape_path=None,shape=None, K=None,weight_index=None,
                 calculation_method =f_spi,threshold=None, verbose=True, index_name = 'SPI'):

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
            self.ts, self.m_cal, self.Pgrid = import_netcdf_for_cumulative_variable(prec_path,['tp','rr','precipitation','prec','LAPrec1871','pre','swe','SWE','sd','SD','sf','SF'],self.shape,self.verbose)
        else:
            raise ValueError("Provide either ts and m_cal directly or specify data_path for a gridded precipitation data in NetCDF format along with the path of the river shapefile.")

        self.K = K if K is not None else 36
        self.threshold = threshold if threshold is not None else -1
        self.weight_index = weight_index if weight_index is not None else 2

        if not callable(calculation_method):
            raise ValueError("`calculation_method` must be a callable function.")
        self.calculation_method = calculation_method
        self.index_name=index_name

        # Inizializza forecast come None
        self.forecast_ts= None
        self.forecast_m_cal = None


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

    def analyze_correlation(self, streamflow,plot=True):
        """
        Analyze correlations between Precipitation SIDI and Streamflow SPI for different weightings and K values.

        Args:
            streamflow (Streamflow): Instance of the Streamflow class.
            plot (bool, optional): Whether to generate a correlation plot and call `_plot_scan`. Default is True.

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


        print(f"Best correlation: R^2 = {max_corr:.3f} (K={K_range[best_k]}, Weight={wlabel[best_weight]})")
        W = generate_weights(K_range[best_k])
        sidi = []  # SPI ensemble mean for each day
        for doy in range(len(spi_like_set[0])):  # in range(self._baseline_indices()[0], self._baseline_indices()[1] + 1):
            vec = spi_like_set[:K_range[best_k], doy]
            sidi.append(weighted_metrics(vec, W[:,best_weight])[0])
        sidi = np.array(sidi)

        SIDI = (sidi - np.nanmean(sidi)) / np.nanstd(sidi)

        # Plot the correlations
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
            plt.title("Correlation Analysis: D{spi} (namely SIDI)  vs. SQI1 ", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.show()
        # ---------------------------------------------------------------
        # Imposta automaticamente parametri ottimali ---------------------
        # self.set_optimal_parameters(K_range[best_k], best_weight)
        # streamflow.set_optimal_parameters(K_range[best_k], best_weight)
        # ---------------------------------------------------------------
        if plot == True:
        # Prompt per rilanciare _plot_scan con best_k
            self.plot_scan(optimal_k=K_range[best_k], weight_index=best_weight)

        if plot == True:
            plt.figure(figsize=(7, 7))
            plt.plot(SIDI,y,'ok',markerfacecolor='yellow', linewidth=2)
            plt.plot(np.arange(-3,4),np.arange(-3,4),'--',color='grey')
            plt.grid()
            plt.ylabel(r"SQI1", fontweight="bold", fontsize=12)
            plt.xlabel("D{spi}", fontweight="bold", fontsize=12)
            plt.title(f"D(spi) vs. SQI1. K={best_k} - weights function n- {best_weight}:", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.show()

        return {"best_k": K_range[best_k], "col_best_weight": best_weight, "max_correlation": max_corr,'spi_corr':R2_spi}

class Streamflow(BaseDroughtAnalysis):
    def __init__(self, start_baseline_year, end_baseline_year,basin_name,
                 ts=None, m_cal=None, shape=None, shape_path=None,
                 data_path=None, K=36, weight_index=2,
                 calculation_method=f_spi, threshold=-1, index_name='SQI'):
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
            if data_path.endswith(('.csv', '.txt')):
                print("Loading streamflow data from CSV/TXT file...")
                self.ts, self.m_cal = load_streamflow_from_csv(data_path)
            elif data_path.endswith(('.xls', '.xlsx')):
                print("Loading streamflow data Excel...")
                self.ts, self.m_cal = load_streamflow_from_excel(data_path)
            else:
                raise ValueError("Formato file non supportato. Usa .csv, .txt, .xls o .xlsx")

            self.is_placeholder = False
        else:
            raise ValueError("You must provide either (`ts` and `m_cal`) or a valid `data_path`.")

        # Inizializzazione della superclasse
        super().__init__(self.ts, self.m_cal, self.K,
                         self.start_baseline_year, self.end_baseline_year,self.basin_name,
                         self.calculation_method, self.threshold, self.index_name)

    def load_streamflow_from_csv(self, file_path, date_col=None, value_col=None, verbose=True):
        """
        Load and assign streamflow data from a CSV file to this instance.
        Wrapper around `load_streamflow_from_csv_file`.

        Args:
            file_path (str): Path to the CSV file.
            date_col (str, optional): Name of the column with dates.
            value_col (str, optional): Name of the column with streamflow values.
            verbose (bool, optional): Whether to print info messages.

        Returns:
            None
        """
        self.ts, self.m_cal = load_streamflow_from_csv(file_path, date_col, value_col)

        # Ricomputazione degli indici
        self.spi_like_set, self.c2r_index = self._calculate_spi_like_set()
        self.SIDI = self._calculate_SIDI()
        self.CDN = self._calculate_CDN()
        self.is_placeholder = False

    def load_streamflow_from_excel(self, file_path, date_col=None, value_col=None, verbose=True):
        """
        Load and assign streamflow data from a CSV file to this instance.
        Wrapper around `load_streamflow_from_csv_file`.

        Args:
            file_path (str): Path to the CSV file.
            date_col (str, optional): Name of the column with dates.
            value_col (str, optional): Name of the column with streamflow values.
            verbose (bool, optional): Whether to print info messages.

        Returns:
            None
        """
        self.ts, self.m_cal = load_streamflow_from_excel(file_path, date_col, value_col)

        # Ricomputazione degli indici
        self.spi_like_set, self.c2r_index = self._calculate_spi_like_set()
        self.SIDI = self._calculate_SIDI()
        self.CDN = self._calculate_CDN()
        self.is_placeholder = False
    def assign_streamflow_data(self, ts, m_cal):
        """
        Assign or update the time series (ts) and calendar (m_cal) if provided by user.
        When the user has a timeseries ready to be assigned, then this method allows to easly assign it
        to the self istance
        Args:
            ts (ndarray): New time series of streamflow values.
            m_cal (ndarray): New calendar array (month, year) matching `ts`.

        Updates:
            - Recomputes SIDI and SPI-related attributes based on the new data.
            - Relies on the BaseDroughtAnalysis class for validation.

        """
        # Update only if new data is provided

        if (m_cal is not None) and (ts is not None):
            # check for temporal resolution:
            years = np.unique(m_cal[:,1])
            if len(m_cal) >  (len(years)-1)*365:
                print("Data appears to have daily resolution. Aggregating to monthly.")
                q = []
                cal =[]
                # potrebbero mancare degli anni per cui enumero da year 0 year end
                for i, year in enumerate(np.arange(years[0],years[-1]+1)):
                    for month in range(1, 13):
                        cal.append([month, year])
                        month_indices = np.where((m_cal[:, 1] == year) & (m_cal[:, 0] == month))[0]
                        if len(month_indices) > 27:
                            monthly_val = np.nanmean(ts[month_indices])
                            # if np.isnan(monthly_val):
                            #     print(month,year)
                            q.append(monthly_val)
                        else:
                            q.append(np.nan)

                q = np.array(q)
                cal = np.array(cal)

                self.m_cal = cal
                self.ts =  q
            else:
                self.m_cal = m_cal
                self.ts = ts

        # Check if both ts and m_cal are set
        if self.ts is None or self.m_cal is None:
            print("Incomplete data provided. Please ensure both `ts` and `m_cal` are set.")
            return

        # Recompute SPI and SIDI
        self.spi_like_set, self.c2r_index = self._calculate_spi_like_set()
        self.SIDI = self._calculate_SIDI()
        # self.CDN = np.cumsum(self.spi_like_set[0])
        self.CDN = self._calculate_CDN()

        # Provide feedback
        print("Streamflow data updated successfully.")
        print(f"Data range: {self.m_cal[0]} to {self.m_cal[-1]}.")

    def gap_filling(self, precipitation, K=None, weight_index=2, alpha=0.1,X2=None):
        """
        Fill missing values (NaN) in streamflow SQI[0] and streamflow time series using SIDI from Precipitation.

        Args:
            precipitation (BaseDroughtAnalysis): Precipitation instance to extract SPI and SIDI.
            K (int, optional): Max temporal scale to recalculate SIDI. If None, uses existing SIDI.
            weight_index (int, optional): Weighting scheme for SIDI calculation.
            alpha (float, optional): Regularization strength for Lasso. Default is 0.1.
        """
        if not isinstance(precipitation, BaseDroughtAnalysis):
            raise TypeError("The input must be an instance of Precipitation.")

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

        sqi1 = self.spi_like_set[0][self_idx]
        spiset = precipitation.spi_like_set[:, prec_idx]
        m_cal = self.m_cal[self_idx]
        ts = self.ts[self_idx]

        # Recalculate SIDI if needed
        if K is None:
            SIDI = precipitation.SIDI[prec_idx, weight_index]
        else:
            W = generate_weights(K)[:, weight_index]
            sidi_vals = np.array([
                weighted_metrics(spiset[:K, t], W)[0]
                for t in range(spiset.shape[1])
            ])
            SIDI = (sidi_vals - np.nanmean(sidi_vals)) / np.nanstd(sidi_vals)

        # finite  mask for regression (where both available)
        valid_mask = np.isfinite(sqi1) & np.isfinite(SIDI)

        # Lasso fit
        # model = Lasso(alpha=alpha)
        model.fit(SIDI[valid_mask].reshape(-1, 1), sqi1[valid_mask])

        # ==================================================================================
        # prediction
        prediction_mask = np.isnan(sqi1) & np.isfinite(SIDI)
        sqi1_pred = sqi1.copy()
        sqi1_pred[prediction_mask] = model.predict(SIDI[prediction_mask].reshape(-1, 1))

        # ==================================================================================
        # reverse SQI1 (index == 0) to ts
        Q_pred = ts.copy()
        Q_pred[prediction_mask] = [np.polyval(self.c2r_index[0,m_cal[prediction_mask][i,0] - 1, :], val) for i,val in enumerate(sqi1_pred[prediction_mask])]

        # ==================================================================================
        # UPDATE
        self.ts[self_idx]=Q_pred

        # Recalculate SPI-like set, SIDI and CDN
        self.spi_like_set, self.c2r_index = self._calculate_spi_like_set()
        self.SIDI = self._calculate_SIDI()
        self.CDN = self._calculate_CDN()

        print(f"Gap filling completed. {np.sum(prediction_mask)} values updated.")

class Pet(BaseDroughtAnalysis):
    def __init__(self, start_baseline_year, end_baseline_year, basin_name, ts=None, m_cal=None, data_path=None,
                 shape_path=None, shape=None, K=None, weight_index=None,
                 calculation_method =f_zscore,threshold=None, index_name = 'SPETI',verbose=True):
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
                                                ['e', 'ET','PET','pet','et','evaporation',
                                                 'evapotranspiration','potential evapotranspiration',
                                                 'reference evapotranspiration','swe','pev'],
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

class Balance(BaseDroughtAnalysis):
    def __init__(self, start_baseline_year, end_baseline_year, basin_name, prec_path=None, pet_path=None,
                 shape_path=None, shape=None, ts=None, m_cal=None, K=None,
                 calculation_method=f_spei, threshold=None, index_name = 'SPEI',verbose=True):
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

class Teleindex(BaseDroughtAnalysis):
    def __init__(self, start_baseline_year, end_baseline_year, basin_name=None,ts=None, m_cal=None, data_path=None,
                 K=None, weight_index=None,
                 calculation_method=f_spei, threshold=None, verbose=True, index_name=''):

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

    def assign_streamflow_data(self, ts, m_cal):
        """
        Assign or update the time series (ts) and calendar (m_cal) if provided by user.
        When the user has a timeseries ready to be assigned, then this method allows to easly assign it
        to the self istance
        Args:
            ts (ndarray): New time series of streamflow values.
            m_cal (ndarray): New calendar array (month, year) matching `ts`.

        Updates:
            - Recomputes SIDI and SPI-related attributes based on the new data.
            - Relies on the BaseDroughtAnalysis class for validation.

        """
        # Update only if new data is provided

        if (m_cal is not None) and (ts is not None):
            # check for temporal resolution:
            years = np.unique(m_cal[:,1])
            if len(m_cal) >  (len(years)-1)*365:
                print("Data appears to have daily resolution. Aggregating to monthly.")
                q = []
                cal =[]
                # potrebbero mancare degli anni per cui enumero da year 0 year end
                for i, year in enumerate(np.arange(years[0],years[-1]+1)):
                    for month in range(1, 13):
                        cal.append([month, year])
                        month_indices = np.where((m_cal[:, 1] == year) & (m_cal[:, 0] == month))[0]
                        if len(month_indices) > 27:
                            monthly_val = np.nanmean(ts[month_indices])
                            # if np.isnan(monthly_val):
                            #     print(month,year)
                            q.append(monthly_val)
                        else:
                            q.append(np.nan)

                q = np.array(q)
                cal = np.array(cal)

                self.m_cal = cal
                self.ts =  q
            else:
                self.m_cal = m_cal
                self.ts = ts

        # Check if both ts and m_cal are set
        if self.ts is None or self.m_cal is None:
            print("Incomplete data provided. Please ensure both `ts` and `m_cal` are set.")
            return

        # Recompute SPI and SIDI
        self.spi_like_set, self.c2r_index = self._calculate_spi_like_set()
        self.SIDI = self._calculate_SIDI()
        # self.CDN = np.cumsum(self.spi_like_set[0])
        self.CDN = self._calculate_CDN()

        # Provide feedback
        print("teleconnection index data updated successfully.")
        print(f"Data range: {self.m_cal[0]} to {self.m_cal[-1]}.")



if __name__ == "__main__":
    print("This module contains the main classes for computing SPI, SIDI, and CDN indices.")
    print("Import the classes into an external script to use them in your project.")
