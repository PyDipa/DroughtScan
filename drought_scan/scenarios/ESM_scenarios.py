"""
auhtor: PyDipa
Scenarios module for drought forecasting.

This module implements the `Scenarios` class, which provides:
- **What-If drought scenarios** based on synthetic precipitation changes.
- **Forecast-based scenarios** using climate model projections.
- **Monte Carlo simulations** for uncertainty estimation.
- **Visualization functions** for comparing different scenarios.

It works in conjunction with the `Precipitation` class.
"""
import os
import warnings
import calendar  # used for month names

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from cmcrameri import cm


from utils.statistics import *
from utils.visualization import *
from utils.decorators import requires_forecast_data

from matplotlib.patches import Rectangle
mpl.rcParams['font.family'] = 'Helvetica'




class Scenarios:
    def __init__(self, drought_scan_object):
        """
        Initialize the Scenarios helper with a Drought Scan object.

        Parameters
        ----------
        drought_scan_object : Precipitation
            A Drought Scan-compatible object providing time series, calendar,
            SPI-like set, CDN, and SIDI attributes.

        Notes
        -----
        This method stores a snapshot of key DSO attributes to allow restoring the
        original state after scenario generation.
        """

        self.DSO = drought_scan_object  # Istanza della classe Precipitation e quindi anche di BaseDroughtAnalysis

        # Snapshots for safe restore
        self._original_ts = drought_scan_object.ts.copy()
        self._original_m_cal = drought_scan_object.m_cal.copy()
        self._original_spi_like_set = drought_scan_object.spi_like_set.copy()
        self._original_CDN = drought_scan_object.CDN.copy()
        self._original_SIDI = drought_scan_object.SIDI.copy()

    def _restore_DSO(self):
        """
        Restore the original state of the Drought Scan object (DSO).

        Restores time series, calendar, SPI-like set, CDN, and SIDI to the values
        saved at initialization, discarding any temporary modifications introduced
        during scenario computation.
        """
        self.DSO.ts = self._original_ts.copy()
        self.DSO.m_cal = self._original_m_cal.copy()
        self.DSO.spi_like_set = self._original_spi_like_set.copy()
        self.DSO.CDN = self._original_CDN.copy()
        self.DSO.SIDI = self._original_SIDI.copy()

    def _get_gamma_params(self):
        """
        Compute and return gamma distribution parameters used for SPI-like indices.

        Returns
        -------
        dict
            Nested dictionary keyed by SPI scale and month, containing the fitted
            gamma parameters required by the SPI computation routine.
        """

        return {
            k: {m: self.DSO.calculation_method(self.DSO.ts, k, m, self.DSO.m_cal, self.DSO.start_baseline_year, self.DSO.end_baseline_year)[-1]
                for m in range(1, 13)}
            for k in range(1, self.DSO.K + 1)
        }

    def _building_WI_scenarios(self,window, monte_carlo_runs=10,year=None, month=None, weight_index=None,use_montecarlo=True):
        """
        Generates and analyzes "What-If" drought scenarios for a specified time window.

        This method simulates hypothetical future drought conditions by applying various
        precipitation scaling factors and running Monte Carlo simulations to capture uncertainty.

        If `weight_index` is None, the method attempts to use `self.DSO.optimal_weight_index` if available;
        otherwise, it falls back to `self.DSO.weight_index`.

        Parameters
        ----------
        year : int
            The year for which the scenario is generated.
        month : int
            The month (1-12) from which the projection starts.
        window : int
            Number of months for the scenario projection.
        monte_carlo_runs : int, optional
            Number of Monte Carlo iterations for uncertainty estimation. Default is 10.
        weight_index : int, optional
            Weighting scheme to apply. If None, it is automatically selected from the DSO object.
        use_montecarlo : bool, optional
            Whether to run Monte Carlo simulations. Default is True.

        Returns
        -------
        dict
            A dictionary with:
            - "SIDIs": Ensemble mean of SIDI for each scenario.
            - "var_SIDIs": Standard deviation of SIDIs across Monte Carlo runs.
            - "CDNs": Ensemble mean of CDN for each scenario.
            - "var_CDNs": Standard deviation of CDNs across Monte Carlo runs.
            - "factors": Precipitation scaling factors used.
            - "Probability": Observed occurrence probability for each scenario.
            - "Calendar": Updated calendar with projected months.
            - "tf1": Index of the last observed month before projection.

        Raises
        ------
        ValueError
            If `year` or `month` is not specified or missing in the dataset.
        """

        print(f'_______________________ SCENARIOS:')
        print(f"building WHAT-IF scenarios for the incoming {window} months from {month}/{year} (last observation) ")
        if use_montecarlo:
            print(f"introducing uncertanities throughtout {monte_carlo_runs} Monte Carlo Repetition")


        # Validate year and month
        if year is None or month is None:
            raise ValueError("Both 'year' and 'month' must be specified.")

        # if weight_index is None:
        #     weight_index = 2  # Default to logarithmically decreasing weights
        if weight_index is None: #wheter the user does not provide a desired weight_index:
            # use self.DSO.optimal_weight_index (available if optimal SIDI has been recalculated!)
            if hasattr(self.DSO, 'optimal_weight_index'):
                weight_index = self.DSO.optimal_weight_index
            else:# use default values stored in self.DSO.weight_index
                weight_index = self.DSO.weight_index

        # Validate year and month
        try:
            # doy input from the user:
            tf1 = np.where(( self.DSO.m_cal[:, 0] == month) & ( self.DSO.m_cal[:, 1] == year))[0][0]
            # last observation available (may coincide or not with tf1
            last_month, last_year =  int(self.DSO.m_cal[-1, 0]),  int(self.DSO.m_cal[-1, 1])
        except IndexError:
            raise ValueError(f"The specified month ({month}) and year ({year}) are not in the dataset.")

        tf2 = tf1 + window

        # CALENDAR EXTENTION: in case of future projections the calendar must be updated
        next_months = []
        mm, yr = last_month, last_year
        for _ in range(window):
            mm += 1
            if mm > 12:
                mm = 1
                yr += 1
            next_months.append([mm, yr])
        # next month is the extentio, regardless that the simulation will be in "forecast" or "hindcast" mode:
        next_months = np.array(next_months)

        # Define scenario factors and expand timeline
        factors = np.array([0.25, 0.50, 0.75, 1, 1.25, 1.50, 2])
        # 2. get the normal values (i.e. SPI1 ===0) for single months
        Nn = np.zeros(12)
        for m in range(12):
            Nn[m] = np.polyval(self.DSO.c2r_index[0, m, :], 0)
        # exrtend to 24 months for cases at the end of the year
        Nn = np.tile(Nn,2)
        norm = np.concatenate([ self.DSO.normal_values(),  Nn[last_month:  last_month+ window]])
        scenarios = np.array([norm * factor for factor in factors])

        # *******************
        # PROBABILTY: observed relative frequency over the last 30 years for single months
        Probability = np.zeros((len(factors), 12)) * np.nan
        # ********************
        for mm in range(1, 13):
            # var = np.array( self.DSO.ts, copy=True)

            try:
                ii = np.where( self.DSO.m_cal[-360::, 0] == mm)[0]  # mm-month index over the last 30 years
                var = np.array( self.DSO.ts[-360::], copy=True)
            except IndexError:
                ii = np.where( self.DSO.m_cal[-240::, 0] == mm)[0]  # mm-month index over the last 20 years
                var = np.array( self.DSO.ts[-240::], copy=True)
                print('relative frequecy of scenarios computed over the last 20 years')

            for s in range(len(factors)):
                threshold_sc = scenarios[s, ii[0]]
                if factors[s] <= 1:
                    Probability[s, mm - 1] = int(len(var[var < threshold_sc]) / len(var) * 100)
                elif factors[s] > 1:
                    Probability[s, mm - 1] = int(len(var[var > threshold_sc]) / len(var) * 100)

        # Prepare results
        SIDIs, eSIDIs, CDNs, eCDNs = [], [], [], []

        # Backup original state
        self._restore_DSO()
        # ---------------------------------------------------------------------------
        # NOTE: Since the time series ( self.DSO.ts) has been modified for scenario testing,
        # we need to ensure that the SPI calculation still uses the original gamma parameters
        # estimated from the original time series (original_ts). This prevents recalibration
        # based on the altered data, ensuring consistency in SPI computation across scenarios.
        gamma_params = self._get_gamma_params()

        try:
            if use_montecarlo:
                for s, scenario in enumerate(scenarios):
                    scenario_tot_water = scenario[tf1 + 1:tf2 + 1]

                    # if s==3:# uncomment for debug
                    #     print(scenario[month:month+window])
                    #     print( scenario[tf1+1 :tf2+1 ])

                    SIDIsc, CDNsc = [], []
                    print(f'proceedig with a Monte Carlo analysis for scenarios {s + 1}')

                    for j in range(monte_carlo_runs):
                        np.random.seed(j)

                        self._restore_DSO()

                        # # Generate random distribution
                        # distribution = np.random.random(window)
                        # distribution /= np.sum(distribution)
                        # modified_scenario = np.sum(scenario_tot_water) * distribution

                        # Preleva i valori normali attesi nei mesi futuri dello scenario
                        norm_weights = Nn[last_month: last_month + window]  # Nn ha 24 mesi, già esteso
                        # Introduci variabilità ma rispettando la stagionalità
                        noise = np.random.random(window)
                        weights = norm_weights * noise  # variabilità "modulata" sulla climatologia

                        # Normalizza
                        weights /= np.sum(weights)

                        # Applica la distribuzione al totale previsto
                        modified_scenario = np.sum(scenario_tot_water) * weights

                        # Update timeseries
                        # manage the timeline for future/partially scenarios:
                        # if the request is over (or partially over) the available data range ('forecast' mode):
                        if (year >  self.DSO.m_cal[-window, 1]) | (
                                year >=  self.DSO.m_cal[-window, 1] and month >=  self.DSO.m_cal[-window, 0]):
                            ts = np.concatenate([ self.DSO.ts, np.tile(0, window)])  # expand the timeline to +12 months
                            ts[tf1 + 1:tf2 + 1] = modified_scenario

                            # update the calendar:
                            m_cal = np.vstack([ self.DSO.m_cal, next_months])
                            # if s == 0:  # set some parameters for plotting
                            #     xlabels = np.array([str(int(c[0])) + ',' + str(int(c[1])) for c in m_cal])
                            #     x = np.arange(len(m_cal))

                            self.DSO.ts = ts[0:tf2 + 1]
                            self.DSO.m_cal = m_cal[0:tf2 + 1]
                        else: #('hindcast' mode)
                            self.DSO.ts[tf1 + 1:tf2 + 1] = modified_scenario


                        # NOTE: Since the time series ( self.DSO.ts) has been modified for scenario testing,
                        # we need to ensure that the SPI calculation still uses the original gamma parameters
                        # estimated from the original time series (original_ts).

                        self.DSO.spi_like_set, _ = self.DSO._calculate_spi_like_set(gamma_params=gamma_params)
                        self.DSO.SIDI = self.DSO._calculate_SIDI()
                        self.DSO.CDN =  self.DSO._calculate_CDN()


                        SIDIsc.append( self.DSO.SIDI)
                        CDNsc.append( self.DSO.CDN)

                    # Ensemble mean and uncertainty
                    SIDIs.append(np.nanmean(SIDIsc, axis=0))
                    eSIDIs.append(np.nanstd(SIDIsc, axis=0))
                    CDNs.append(np.nanmean(CDNsc, axis=0))
                    eCDNs.append(np.nanstd(CDNsc, axis=0))

            else:
                SIDIs, CDNs = [], []
                for s, scenario in enumerate(scenarios):
                    modified_scenario = scenario[tf1 + 1:tf2 + 1]
                    self._restore_DSO()

                    if (year > self.DSO.m_cal[-window, 1]) or (
                            year == self.DSO.m_cal[-window, 1] and month >= self.DSO.m_cal[-window, 0]):
                        ts = np.concatenate([self.DSO.ts, np.tile(0, window)])
                        ts[tf1 + 1:tf2 + 1] = modified_scenario
                        m_cal = np.vstack([self.DSO.m_cal, next_months])
                        self.DSO.ts = ts[0:tf2 + 1]
                        self.DSO.m_cal = m_cal[0:tf2 + 1]
                    else:
                        self.DSO.ts[tf1 + 1:tf2 + 1] = modified_scenario
                    self.DSO.spi_like_set, _ = self.DSO._calculate_spi_like_set(gamma_params=gamma_params)
                    self.DSO.SIDI = self.DSO._calculate_SIDI()
                    self.DSO.CDN = self.DSO._calculate_CDN()


                    # Inserisci singolo risultato come se fosse la media di 1
                    SIDIs.append(self.DSO.SIDI)
                    CDNs.append(self.DSO.CDN)

        finally:
            self._restore_DSO()


        SIDIs = np.array(SIDIs)[:, :, weight_index]
        CDNs = np.array(CDNs)
        if use_montecarlo:
            eCDNs = np.array(eCDNs)
            eSIDIs = np.array(eSIDIs)[:, :, weight_index]
        else:#no varaibility added
            eCDNs = CDNs
            eSIDIs = SIDIs

        Calendar = np.vstack([self.DSO.m_cal, next_months])
        Calendar = Calendar[0:tf2 + 1]

        return {
            "SIDIs": SIDIs,
            "var_SIDIs": eSIDIs,
            "CDNs": CDNs,
            "var_CDNs": eCDNs,
            "factors": factors,
            "Probability": Probability,
            "Calendar": Calendar,
            "tf1":tf1

        }

    def plot_WI_scenarios(self, year=None, month=None, monte_carlo_runs=10, weight_index=None,name=None,window=None,use_montecarlo=True):
        """
        Plots the results of "What-If" drought scenarios for SIDI and CDN.

        This method visualizes the potential drought trajectories under different hypothetical
        precipitation scenarios, with or without Monte Carlo uncertainty, starting from a selected month/year.

        Parameters
        ----------
        year : int, optional
            Year from which the projection starts. Defaults to the last available observation.
        month : int, optional
            Month (1–12) from which the projection starts. Defaults to the last available observation.
        window : int, optional
            Number of months to project. Default is 6. Maximum allowed is 12 (warnings for values >6).
        monte_carlo_runs : int, optional
            Number of Monte Carlo simulations to estimate uncertainty. Default is 10.
        weight_index : int, optional
            Index for the SIDI weighting scheme. Default is 2 (log-decreasing).
        name : str, optional
            Optional name of the region or basin for the plot title.
        use_montecarlo : bool, optional
            Whether to introduce variability through Monte Carlo simulations. Default is True.

        Returns
        -------
        None
            Displays a two-panel plot:
            - Left: simulated SIDI trajectories with uncertainty bands and observed values.
            - Right: CDN trajectories and relative frequency annotations.

        Raises
        ------
        ValueError
            If `year` or `month` is not found in the dataset.
        """
        self._restore_DSO()

        if window > 12:
            warnings.warn("Windows greater than 12 months are not accepted. Setting window to 6 months.", UserWarning)
            window = 6
        elif window > 6:
            warnings.warn("Windows greater than 6 months reduce projection confidence due to uncertainty.", UserWarning)
            # Validate year and month
        elif window is None:
            window = 6


        if year is None or month is None:
            # Determine the start index in m_cal
            month, year = self.DSO.m_cal[-1]  # First new month
            print(f" building scenarios from {month}/{year}")

            # Verify date range
            # last_month, last_year =  self.DSO.m_cal[-1, 0],  self.DSO.m_cal[-1, 1]
            # if (year > last_year) or (year == last_year and month > last_month):
            #     warnings.warn("The selected year and month exceed the available data range.", UserWarning)

        if weight_index is None:  # wheter the user does not provide a desired weight_index:
            # use self.DSO.optimal_weight_index (available if optimal SIDI has been recalculated!)
            if hasattr(self.DSO, 'optimal_weight_index'):
                weight_index = self.DSO.optimal_weight_index
            else:  # use default values stored in self.DSO.weight_index
                weight_index = self.DSO.weight_index

            # Validate year and month
        # try:
        #     tf1 = np.where(( self.DSO.m_cal[:, 0] == month) & ( self.DSO.m_cal[:, 1] == year))[0][0]
        #     last_month, last_year =  self.DSO.m_cal[-1, 0],  self.DSO.m_cal[-1, 1]
        #     tf2 = tf1 + window
        # except IndexError:
        #     raise ValueError(f"The specified month ({month}) and year ({year}) are not in the dataset.")
        # --------------------------------------------------------------------------------------
        # recall a methdo from the mohter class BaseDroguthAnalysis!
        scenario_results = self._building_WI_scenarios(year=year, month=month, window=window,
                                                        monte_carlo_runs=monte_carlo_runs, weight_index=weight_index,
                                                       use_montecarlo=use_montecarlo)
        SIDIs = scenario_results["SIDIs"]
        eSIDIs = scenario_results["var_SIDIs"]
        CDNs = scenario_results["CDNs"]
        eCDNs = scenario_results["var_CDNs"]
        factors = scenario_results["factors"]
        Probability = scenario_results["Probability"]
        m_cal = scenario_results["Calendar"]
        tf1 = scenario_results["tf1"]
        tf2 = tf1 + window

        # --------------------------------------------------------------------------------------
        xlabels = [f"{int(m[0])},{int(m[1])}" for m in m_cal]
        # from month_number\year" to month/year
        xticks_labels = []

        for label in xlabels:
            month_x, year_x = map(int, label.split(','))
            month_name = calendar.month_abbr[month_x]  # abbreviazione (Jan, Feb, etc.)
            xticks_labels.append(f"{month_name}\n{year_x}")


        # xlabels = [f"{int(m[0])},{int(m[1])}" for m in m_cal]
        # # ls = ['--', '-o', '-^', '-s', '--', '-o', '-^']
        # colors = plt.get_cmap('jet', len(factors) * 2).reversed()
        # # colors = mpl.cm.get_cmap('jet', len(factors) * 2).reversed()
        # # se voglio mezza colormap uso 0:0.45 elsev 0:0.95
        # linecolor = ListedColormap(colors(np.linspace(0, 0.95, len(factors))))
        # # SET THE COLORMAP FOR THE HEATMAP
        # cmap = heatmap_cmap()
        # bounds = np.array([-3, -2.5, -2, -1.5, -1, -0.5, 0,0.5, 1, 1.5, 2, 2.5, 3, 3.5])
        # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        #
        # # SET THE COLORMAP FOR CDN (SPI-1 CUMULATIVE DEVIATION)
        # cdnmap =plt.get_cmap('coolwarm').reversed()
        #
        # # using heatmap + Dsqi/Dspi + balance + scenarios of %ile of next month
        # x = np.arange(np.shape(SIDIs)[1])
        # heatmap = np.array( self.DSO.spi_like_set, copy=True)
        # heatmap[:, tf1 + 1::] = np.nan
        #
        # # whenever the SIDI has been recalculated with optimal K:
        # if hasattr(self.DSO, 'optimal_k'):
        #     heatmap = heatmap[0:self.DSO.optimal_k,:].copy()

        title = f'Forecast window: from {calendar.month_abbr[month]}. {year} \n' \
                    f'{self.DSO.basin_name} basin; Baseline: {self.DSO.start_baseline_year} -{self.DSO.end_baseline_year} \n\n'

        x = np.arange(np.shape(SIDIs)[1])

        # ===================
        #  SIDI
        # ==================
        index = r"$\mathit{D}_{\mathit{spi}}$"
        fig, ax = plt.subplots(figsize=(13, 9))

        # Heatmap
        # ax.imshow(heatmap, cmap=cmap, norm=norm, aspect='auto', alpha=0.7,
        #           extent=[-1, heatmap.shape[1] - 1, 0, heatmap.shape[0]])
        # Set the grid ---------------------------------------
        ax.set_yticks(np.arange(-4,5))
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xticks_labels, fontsize=10)
        ax.axhline(y=-1, linestyle='-', c='dimgrey', alpha=1)
        ax.text(tf1 -5 +2, -1.3, "Severe Drought"+r"$\downarrow$", fontsize=12)
        ax.axvline(x=tf2-window,c='dimgrey')
        ax.spines['left'].set_position(( "data",tf2 - window))
        ax.set_ylim(-5, 5)
        ax.set_xlim(tf1 - 5, tf2)
        ax.grid('both',linestyle=':')
        ax.axvline(x=tf1 - 5, color='gray')

        # What-if curves and annotations ----------------------
        add_whatif_curves(ax, x, SIDIs, tf1, tf2,factors, Probability, month)

        # PROJECTED SIDI variability
        # if use_montecarlo:
        #     [ax.fill_between(x, SIDIs[ii] - eSIDIs[ii], SIDIs[ii] + eSIDIs[ii],  color=cmc.vik(ii), linewidth=0,alpha=0.1)
        #      for ii in range(len(factors))]

        # Observed ---------------------------------------
        ax.plot(np.arange(len(self.DSO.ts)), self.DSO.SIDI[:, weight_index], 'k',
                 linewidth=3, alpha=1, label='Observed')

        # Titles and legends ---------------------------------------
        ax.set_title(title, loc='left', linespacing=1.8, fontsize=12)
        fig.suptitle(f'Observed and What-If scenarios for {index}\n', ha='left', x=0.023,
                     fontsize=16, fontweight=14)
        upper_title = f'OBSERVED                                                '
        bottom_title = f'WHAT-IF SCENARIOS\n\nNP = normal precipitation\n\n(%): Occurrence likelihood (historical, last 30 years)              \n\n' \
                       f'Exceedance (≥) or Non-Exceedance (≤)'

        add_legends(fig, ax, upper_title, bottom_title,only_WI=True)

        # graphic embellishments ---------------------------------------
        n_dash = 50
        line_string = "|" + "·" * n_dash + r" $\leftarrow$ " + " Observed "
        ax.text(tf1-5.02, 5.2, line_string, ha='left', va='center', fontsize=12)

        n_dash = 64
        line_string = "Forecasted "+r" $\rightarrow$"  + "·" * n_dash + "|"
        ax.text(tf2 - window+0.2, 5.2, line_string, ha='left', va='center', fontsize=12)
        fig.tight_layout(rect=[0, 0, 0.645, 1])

        # =========================
        # PLOT 2: CDN
        # =========================
        index = 'CDN'
        fig, ax = plt.subplots(figsize=(13, 9))
        # Set the grid ---------------------------------------
        ax.set_yticks(np.arange(-30, 35,5))
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xticks_labels, fontsize=10)
        ax.axhline(y=0, linestyle='-', c='dimgrey', alpha=1)
        ax.axvline(x=tf2 - 6, c='dimgrey')
        ax.spines['left'].set_position(("data", tf2 - 6))

        m = np.round(self.DSO.CDN[-window],-1)
        ax.set_ylim(m-15, m+15)
        ax.set_xlim(tf1 - 5, tf2)
        ax.grid('both', linestyle=':')
        ax.axvline(x=tf1 - 5, color='gray')  # o x=0 se vuoi la cornice a sinistra

        # What-if curves + annotations ---------------------
        add_whatif_curves(ax, x, CDNs, tf1, tf2,factors, Probability, month)
        if use_montecarlo:
            [ax.fill_between(x, CDNs[ii] - eCDNs[ii], CDNs[ii] + eCDNs[ii], color=cmc.vik(ii), linewidth=0,
                                  alpha=0.1)
             for ii in range(len(factors))]
        # Observed CDN ---------------------
        ax.plot(np.arange(len(self.DSO.CDN)), self.DSO.CDN, 'k',
                 linewidth=3, alpha=1, label='Observed')

        # Titles and legends ---------------------
        ax.set_title(title, loc='left', linespacing=1.8, fontsize=12)
        fig.suptitle(f'Observed and What-If scenarios for {index}\n', ha='left', x=0.023,
                     fontsize=16, fontweight=14)
        upper_title = f'OBSERVED                                                '
        bottom_title = f'WHAT-IF SCENARIOS\n\nNP = normal precipitation\n\n(%): Occurrence likelihood (historical, last 30 years)              \n\n' \
                       f'Exceedance (≥) or Non-Exceedance (≤)'

        add_legends(fig, ax, upper_title, bottom_title,only_WI=True)

        # graphic embellishments ---------------------------------------
        n_dash = 50
        line_string = "|" + "·" * n_dash + r" $\leftarrow$ " + " Observed "
        ax.text(tf1-5.02, m+15+1.1, line_string, ha='left', va='center', fontsize=12)
        # ax.text(tf2 - window - 1.8, 5.1, r"$\leftarrow$" + "Observed", ha='left', va='center', fontsize=12)

        n_dash = 64
        line_string = "Forecasted "+r" $\rightarrow$"  + "·" * n_dash + "|"
        ax.text(tf2 - window+0.2, m+15+1.1, line_string, ha='left', va='center', fontsize=12)
         # ax.text(tf2 - window + 0.3, 5.1, "Forecasted"r"$\rightarrow$", ha='left', va='center', fontsize=12)
        fig.tight_layout(rect=[0, 0, 0.645, 1])



        # OLD HEATMAP * JET CURVES
        # PLOTTING -----------------------------------------------------------------------
        # fig, ax = plt.subplots(figsize=(15, 7), nrows=1, ncols=2)
        # ax = ax.ravel()
        # cnt = 0
        # # FIRS SUBPLOT: HEATMPA + D{SPI}---------------------------------------------------
        # # ADDING WHAT-IF Trajectories
        # [ax[cnt].plot(x[tf1::], SIDIs[ii, tf1::], c=linecolor(ii),
        #               linewidth=2, linestyle='--', label=f'{factors[ii]} x Pnorm') for ii in range(len(factors))]
        #
        #
        # ax[cnt].axhline(y=-1, linestyle='-', c='k', alpha=1)
        # ax[cnt].plot(np.arange(len( self.DSO.ts)),  self.DSO.SIDI[:, weight_index], 'k', linewidth=3, label='observed')
        # ax[cnt].set_xticks(np.arange(0, len(xlabels)))
        # ax[cnt].set_xticklabels(xlabels, rotation=90, fontweight='bold')
        # ax[cnt].set_xlim(tf1 - 8, tf2)
        # domain = SIDIs[:, tf1 - 8:tf2 - 1]
        # ax[cnt].set_ylim(-6, 5)
        # # ax1.set_ylim(-6,3)
        # # [ax1.text(tf2,SIDIs[ii][tf2],f'P(occurence)= {int(Probability[ii,m])}%',color=linecolor(ii)) for ii in range(len(fasce))]
        # ax[cnt].text(tf1 - 7.8, 5.55, 'D', style='italic', fontsize=16)
        # ax[cnt].text(tf1 - 7.4, 5.55, '{SPI}:', fontsize=8)
        # ax[cnt].set_title(f'upcoming {window} months  \n from {month},{year}',
        #                   fontsize=12, c='k')
        # ax[cnt].legend(bbox_to_anchor=(-0.05, 1))
        # ax[cnt].grid()
        # # ASSE DESTRO
        # ax1 = ax[cnt].twinx()
        #
        # ax1.imshow(heatmap, cmap=cmap, norm=norm, aspect='auto', alpha=0.7)
        # ax1.set_yticks([])

        # per rinforzare la linea nera
        # ax2 = ax[cnt].twinx()
        # ax2.plot(np.arange(len( self.DSO.ts)),  self.DSO.SIDI[:, weight_index], 'k', linewidth=3, label='observed')
        # ax2.axhline(y=-1, c='grey')
        # ax2.set_ylim(-6, 5)
        # ax2.set_yticks([])

        # # SECOND SUBPLOT: CDN-----------------------------------------------------------
        #
        # xx = np.arange(len( self.DSO.CDN))
        # normalize = mpl.colors.Normalize(vmin=np.nanmin( self.DSO.CDN), vmax=np.nanmax( self.DSO.CDN))
        # cnt = cnt + 1
        # # OBSERVED CDN
        # ax[cnt].plot( self.DSO.CDN, 'k', label='CDN observed', linewidth=3)
        # for i in xx[tf1-8 :tf1]:
        #     ax[cnt].fill_between([xx[i], xx[i + 1]],
        #                          [ self.DSO.CDN[i],  self.DSO.CDN[i + 1]],
        #                          color=cdnmap(normalize( self.DSO.CDN[i]))
        #                          , alpha=0.6)
        # ax[cnt].set_xticks(np.arange(0, len(xlabels)))
        # ax[cnt].set_xticklabels(xlabels, rotation=90, fontweight='bold')
        # ax[cnt].axhline(y=0, linestyle='-', c='k', alpha=1)
        # # PROJECTED CDNs
        # [ax[cnt].plot(x[tf1::], CDNs[ii][tf1::], c=linecolor(ii),
        #               linewidth=2, linestyle='--', label=f'CDN({factors[ii]} x Normal values)') for ii in
        #  range(len(factors))]
        # # PROJECTED CDN variability
        # if use_montecarlo:
        #     [ax[cnt].fill_between(x, CDNs[ii] - eCDNs[ii], CDNs[ii] + eCDNs[ii], color=linecolor(ii), linewidth=0,
        #                           alpha=0.1)
        #      for ii in range(len(factors))]
        # # relative observed frequency of scenarios over the last 30 or 20 years
        # [ax[cnt].text(tf2 + 0.5, CDNs[ii][tf2], f'RF: {int(Probability[ii, month - 1])}%', color=linecolor(ii)) for ii
        #  in
        #  range(len(factors))]
        # # ax[cnt].set_ylim(-35, 5)
        #
        # domain = CDNs[:, tf1-8 :tf2 - 1]
        # ax[cnt].set_ylim((np.min(domain) - 7, np.max(domain) + 7))
        # ax[cnt].set_xlim(tf1 - 8, tf2)
        # ax[cnt].set_title(f'CDN: upcoming {window} months \n from  {month},{year}',
        #                   fontsize=12, c='k')
        # # ax[cnt].legend()
        # # Set the title and layout
        # basin =  self.DSO.shape.to_crs(epsg=32632)
        # # Calcola l'area in metri quadrati
        # area_kmq = basin.geometry.area.iloc[0] / 1e6
        # K = self.DSO.K if not hasattr(self.DSO, 'optimal_k') or self.DSO.optimal_k is None else self.DSO.optimal_k
        # wlabel = ['eq', 'ldw', 'lgdw', 'liw', 'lgiw'][weight_index]



        # plt.tight_layout()

    @requires_forecast_data
    def _forecast_scenarios_abs(self, year=None, month=None, window=None,  weight_index=None):
        """
            Computes absolute forecast-based drought scenarios using external climate projections.

            This method integrates precipitation forecasts into the drought scan framework,
            simulating future conditions based on projected absolute precipitation values.

            Parameters
            ----------
            year : int
                Start year of the projection window.
            month : int
                Start month of the projection window (1–12).
            window : int
                Number of months to project.
            monte_carlo_runs : int, default 10
                Number of Monte Carlo simulations for uncertainty estimation.
            weight_index : int, optional
                Weighting scheme index to compute aggregated indices when applicable.
            use_montecarlo : bool, default True
                Whether to run Monte Carlo sampling in addition to deterministic paths.

            Returns
            -------
            dict
               A dictionary containing:
               - "SIDIs": SIDI forecast ensemble values.
               - "CDNs": CDN forecast ensemble values.
               - "Calendar": Updated time calendar including forecasted months.

            Raises
            ------
            ValueError
               If the requested forecast period is not available in the dataset.
            """

            # months = self.DSO.forecast_m_cal[:, 0]



        if weight_index is None: #wheter the user does not provide a desired weight_index:
            # use self.DSO.optimal_weight_index (available if optimal SIDI has been recalculated!)
            if hasattr(self.DSO, 'optimal_weight_index'):
                weight_index = self.DSO.optimal_weight_index
            else:# use default values stored in self.DSO.weight_index
                weight_index = self.DSO.weight_index

        m_cal = concatenate_m_cal(self.DSO.m_cal, self.DSO.forecast_m_cal)
        n_members, f_window = np.shape( self.DSO.forecast_ts)

        # Backup original state
        original_ts = self.DSO.ts.copy()
        original_m_cal = self.DSO.m_cal.copy()
        original_spi_like_set = self.DSO.spi_like_set.copy()
        original_CDN = self.DSO.CDN.copy()
        original_SIDI = self.DSO.SIDI.copy()

        # ---------------------------------------------------------------------------
        # NOTE: Since the time series (  self.DSO.ts) will be  modified for scenario testing,
        # we need to ensure that the SPI calculation still uses the original gamma parameters
        # estimated from the original time series (original_ts). This prevents recalibration
        # based on the altered data, ensuring consistency in SPI computation across scenarios.
        gamma_params = self._get_gamma_params()

        if year is None or month is None:
            # Determine the start index in m_cal
            month, year = m_cal[len(self.DSO.ts)]  # First new month
        try:
            tf1 = np.where((m_cal[:, 0] == month) & (m_cal[:, 1] == year))[0][0]
        except IndexError:
            raise ValueError(f"The specified month ({month}) and year ({year}) are not in the dataset.")

        tf2 = min(tf1 + window, len(m_cal))  # Evita errori di out-of-bounds
        #     tf1 = np.where((m_cal[:, 0] == month) & (m_cal[:, 1] == year))[0][0]
        #     tf2 = tf1 + f_window
        # else:
        #     try:
        #         tf1 = np.where((m_cal[:, 0] == month) & (m_cal[:, 1] == year))[0][0]
        #         tf2 = tf1 + f_window
        #     except IndexError:
        #         raise ValueError(f"The specified month ({month}) and year ({year}) are not in the dataset.")

        print(f'the forecast starts from {m_cal[tf1]} and arrives to {m_cal[tf2 - 1]} (included))')

        SIDIsc, CDNsc = [], []
        try:
            for s, forecast in enumerate(self.DSO.forecast_ts):
                self.DSO.ts = original_ts  # restore the data
                self.DSO.m_cal = original_m_cal  # restore the caledar
                print(f'precessing member {s + 1} of {n_members}')
                # print(f'processing forecast {s+1} of {np.shape(  self.DSO.forecast_ts)}')
                # if len(forecast) >= window:
                #     forecast_tot_water = forecast[0:window]
                # else:
                #     forecast_tot_water = forecast

                forecast_tot_water = forecast

                # ️ Extend ts if necessary
                if len(self.DSO.ts) < tf2:
                    ts = np.concatenate([self.DSO.ts, np.full((tf2 - len(self.DSO.ts)), np.nan)])
                else:
                    ts =   self.DSO.ts.copy()

                # coerente con il resto del file
                ts[tf1 + 1:tf2 + 1] = forecast_tot_water  # inietta da tf1+1
                self.DSO.ts = ts[:tf2 + 1]  # tieni fino a tf2 incluso
                self.DSO.m_cal = m_cal[:tf2 + 1]
                if s == 0:
                    Calendar = m_cal[:tf2 + 1]

                self.DSO.spi_like_set, _ =   self.DSO._calculate_spi_like_set(gamma_params=gamma_params)
                self.DSO.SIDI =   self.DSO._calculate_SIDI()
                self.DSO.CDN =   self.DSO._calculate_CDN()

                SIDIsc.append(  self.DSO.SIDI[:, weight_index])
                CDNsc.append(  self.DSO.CDN)

        finally:
            # Restore original state
            self.DSO.ts = original_ts
            self.DSO.m_cal = original_m_cal
            self.DSO.spi_like_set = original_spi_like_set
            self.DSO.CDN = original_CDN
            self.DSO.SIDI = original_SIDI

        # Convert results to numpy arrays
        SIDIs = np.array(SIDIsc)
        CDNs = np.array(CDNsc)

        return {
            "SIDIs": SIDIs,
            "CDNs": CDNs,
            "Calendar": Calendar,
        }

    @requires_forecast_data
    def _forecast_scenarios_rel(self, year=None, month=None, weight_index=None,window=None):
        """
        Computes forecast-based drought scenarios using relative precipitation anomalies.

        This method converts relative precipitation anomalies from seasonal forecasts
        into absolute water amounts using observed climatology, then calculates the
        resulting SIDI and CDN indices for each ensemble member.

        Parameters
        ----------
        year : int, optional
            Year in which the forecast starts. If not specified, defaults to the first available forecasted month.
        month : int, optional
            Month (1–12) in which the forecast starts. Defaults to the first available forecasted month.
        window : int, optional
            Number of months to include in the projection. Defaults to the full forecast window available.
        weight_index : int, optional
            Index of the SIDI weighting scheme to use. Default is 2 (log-decreasing).

        Returns
        -------
        dict
            A dictionary containing:
            - "SIDIs": Array of SIDI values from each forecast ensemble member.
            - "CDNs": Array of CDN values from each ensemble member.
            - "Calendar": Combined calendar of observed and forecast periods.
            - "tf1": Index of the last observed month before forecasts start.
            - "window": Number of forecast months used.

        Raises
        ------
        ValueError
            If the selected year/month is not covered by the available forecast data.
            Or if the forecast window is out of bounds.
        """

        if weight_index is None: #wheter the user does not provide a desired weight_index:
            # use self.DSO.optimal_weight_index (available if optimal SIDI has been recalculated!)
            if hasattr(self.DSO, 'optimal_weight_index'):
                weight_index = self.DSO.optimal_weight_index
            else:# use default values stored in self.DSO.weight_index
                weight_index = self.DSO.weight_index

        # common calendar and forecasts convertion to absolute water amounts
        m_cal = concatenate_m_cal(self.DSO.m_cal,self.DSO.forecast_m_cal)
        months = self.DSO.forecast_m_cal[:,0]
        clima = self.DSO.normal_values()[months-1] #climatology OBSERVED not modelled but jan start form 0
        forecast_ts =np.array([clima + ts*clima for ts in self.DSO.forecast_ts])


        # Backup original state
        original_ts = self.DSO.ts.copy()
        original_m_cal = self.DSO.m_cal.copy()
        original_spi_like_set = self.DSO.spi_like_set.copy()
        original_CDN = self.DSO.CDN.copy()
        original_SIDI = self.DSO.SIDI.copy()

        # ---------------------------------------------------------------------------
        # NOTE: Since the time series (  self.DSO.ts) will be  modified for scenario testing,
        # we need to ensure that the SPI calculation still uses the original gamma parameters
        # estimated from the original time series (original_ts). This prevents recalibration
        # based on the altered data, ensuring consistency in SPI computation across scenarios.
        gamma_params = self._get_gamma_params()

        # find the index of the forecast window
        # fix the number of members and the forecast window equals to the forecast window
        n_members = np.shape(self.DSO.forecast_ts)[0]
        max_window = np.shape(self.DSO.forecast_ts)[1]

        if year is None or month is None:
            # Determine the start index as the first in forecast_m_cal
            month, year = self.DSO.forecast_m_cal[0]  # First forecasted month
            tf1 = np.where((m_cal[:, 0] == month) & (m_cal[:, 1] == year))[0][0]
            tf1 = tf1- 1
            if window is None:
                window = max_window
            tf2 = tf1 + window #max_window #Not included
            months_to_skip = 0
        else:
            try:
                # where the forecast starts:
                m, y = self.DSO.forecast_m_cal[0]  # trace the first forecasted month
                tf0 = np.where((m_cal[:, 0] == m) & (m_cal[:, 1] == y))[0][0]
                tf1 = np.where((m_cal[:, 0] == month) & (m_cal[:, 1] == year))[0][0]
                if tf1<tf0-1:
                    raise ValueError(f'chose a month from available forecasts: {self.DSO.forecast_m_cal} ** except for the last! **')

                months_to_skip = tf1 +1 - tf0
                if window is None:
                    window = max_window - months_to_skip
                elif window > (max_window - months_to_skip):
                    window = max_window - months_to_skip
                    print(f"⚠️  'window' exceeds the maximum allowed value: it has been set to {window} months.")

                tf2 = tf1 +  window
                if tf1 >= tf1+window:
                    raise ValueError(f'Chose a month/year between allowed forecasts window: from {self.DSO.forecast_m_cal[0]} up to 1-month before {self.DSO.forecast_m_cal[-1]}')
            except IndexError:
                raise ValueError(f"The specified month ({month}) and year ({year}) are not in the dataset")


        # tf2 = min(tf1 + window, len(m_cal))  # Evita errori di out-of-bounds that may arise (evenf after combingin the clanedars) due to a large window

        print("**************************")
        print(f"The first forecasted month AVAILABLE is {m_cal[tf0]}...")
        print(f"The CHOSEN forecast range starts from {m_cal[tf1 + 1]} and goes up to {m_cal[tf2]} (inclusive).")
        print(f'the window is {window}')

        SIDIsc, CDNsc = [], []
        try:

            for s, forecast_tot_water in enumerate(forecast_ts):
                self.DSO.ts = original_ts  # restore the data
                self.DSO.m_cal = original_m_cal  # restore the caledar
                print(f'precessing member {s + 1} of {n_members}')
                # print(f'processing forecast {s+1} of {np.shape(  self.DSO.forecast_ts)}')
                # if len(forecast) >= window:
                #     forecast_tot_water = forecast[0:window]
                # else:
                #     forecast_tot_water = forecast

                # ️ Extend ts if necessary
                if len(self.DSO.ts) < tf2:
                    ts = np.concatenate([  self.DSO.ts, np.full((window), np.nan)])
                else:
                    ts = self.DSO.ts.copy()

                # Assign forecast values considering only the relevant winder
                # ts[tf1+1:tf2+1] = forecast_tot_water[:window]
                # la prima proiezione la salto!
                ts[tf1+1:tf2+1] = forecast_tot_water[months_to_skip:months_to_skip+window]
                # print(np.round(np.sum(ts[tf1+1:tf2+1])))
                self.DSO.ts = ts[:tf2+1]  # Trim to the required length
                self.DSO.m_cal = m_cal[:tf2+1]

                if s == 0:
                    Calendar = m_cal[:tf2+1]

                self.DSO.spi_like_set, _ =   self.DSO._calculate_spi_like_set(gamma_params=gamma_params)
                # opt_k = self.DSO.K if not hasattr(self.DSO, 'optimal_k') or self.DSO.optimal_k is None else self.DSO.optimal_k
                self.DSO.SIDI =   self.DSO._calculate_SIDI()
                self.DSO.CDN =   self.DSO._calculate_CDN()
                # print(  self.DSO.SIDI[:, weight_index][-3::])

                SIDIsc.append(  self.DSO.SIDI[:, weight_index])
                CDNsc.append(  self.DSO.CDN)
        finally:

            # Restore original state
            self._restore_DSO()
            # self.DSO.ts = original_ts
            # self.DSO.m_cal = original_m_cal
            # self.DSO.spi_like_set = original_spi_like_set
            # self.DSO.CDN = original_CDN
            # self.DSO.SIDI = original_SIDI

        # Convert results to numpy arrays
        SIDIs = np.array(SIDIsc)
        CDNs = np.array(CDNsc)

        return {
            "SIDIs": SIDIs,
            "CDNs": CDNs,
            "Calendar": Calendar,
            "tf1": tf1,
            "window":window
        }

    @staticmethod
    def _ensemble_window_stats(Forecast):
        """
        Compute ensemble mean (over members) on the forecast window and its calendar.

        Expects Forecast to have:
          - 'SIDIs': (n_members, T)
          - 'CDNs' : (n_members, T)
          - 'Calendar': (T, 2) with [month, year]
          - 'tf1': int (last observed index)
          - 'window': int (forecast horizon length)

        Returns a dict with:
          - 'SIDI_mean_win': (window,)
          - 'CDN_mean_win' : (window,)
          - 'Calendar_win' : (window, 2)
          - 'start'/'stop' indices used
        """
        SIDIs = np.asarray(Forecast['SIDIs'])
        CDNs = np.asarray(Forecast['CDNs'])
        Cal = np.asarray(Forecast['Calendar'])
        tf1 = int(Forecast['tf1'])
        win = int(Forecast['window'])

        start = tf1 + 1  # first forecasted month (exclusive tf1)
        stop = tf1 + win + 1  # slice stop (exclusive)

        if stop > SIDIs.shape[1]:
            raise ValueError("Forecast window exceeds available time steps.")

        # mean over ensemble axis=0, keep NaN-safe
        SIDI_mean_win = np.nanmean(SIDIs[:, start:stop], axis=0)
        CDN_mean_win = np.nanmean(CDNs[:, start:stop], axis=0)
        Calendar_win = Cal[start:stop, :]

        return {
            "SIDI_mean_win": SIDI_mean_win,
            "CDN_mean_win": CDN_mean_win,
            "Calendar_win": Calendar_win,
            "start": start,
            "stop": stop
        }

    def export_forecast(self, year=None, month=None,window=None,weight_index=None,export_csv=False):
        """
        """


        if weight_index is None: #wheter the user does not provide a desired weight_index:
            # use self.DSO.optimal_weight_index (available if optimal SIDI has been recalculated!)
            if hasattr(self.DSO, 'optimal_weight_index'):
                weight_index = self.DSO.optimal_weight_index
            else:# use default values stored in self.DSO.weight_index
                weight_index = self.DSO.weight_index

        # find the index of the forecast window
        if year is None or month is None:
            # Determine the start index as the last observed date
            month, year = self.DSO.m_cal[-1]  # First forecsted month

        # ---- PRODUCING SCENARIOS AND FORECASTS -----------------
        print("processing the FORECASTS...")
        if window is None:
            # year and month are the last observed data afer which the simulations starts
            Forecast = self._forecast_scenarios_rel(year = year,month =month,weight_index=weight_index)
        else:
            Forecast = self._forecast_scenarios_rel(year=year, month=month,window=window,weight_index=weight_index)
        window = Forecast['window'] #trim the window after checks in _forecast_scenarios_rel
        tf1 = Forecast['tf1']
        tf2 = tf1+window
        win_forecast = self._ensemble_window_stats(Forecast)
        df = pd.DataFrame(win_forecast)
        if export_csv==True:
            print(f'saving ESM forecast.scv in {os.getcwd()}')
            df.to_csv('ESM_forecast.csv', index=False)
        return win_forecast

    @requires_forecast_data
    def plot_combined_scenarios(self, year=None, month=None, window=6, monte_carlo_runs=10, weight_index=None, use_montecarlo=False,savecast=False):
        """
        Combines "What-If" scenarios and seasonal forecast-based scenarios into a unified drought outlook.

        This method overlays synthetic drought projections (What-If scenarios with scaled precipitation)
        and real ensemble climate forecasts, providing a comprehensive visualization of potential future
        drought conditions. The output includes both SIDI and CDN indicators, with uncertainty bands
        and relative frequencies.

        Parameters
        ----------
        year : int, optional
            Start year of the projection window.
        month : int, optional
            Start month of the projection window (1–12).
        window : int, default 6
            Number of months to project.
        monte_carlo_runs : int, default 10
            Number of Monte Carlo simulations for uncertainty visualization.
        weight_index : int, optional
            Weighting scheme index to compute aggregated indices when applicable.
        use_montecarlo : bool, default False
            Whether to display Monte Carlo spread along with deterministic paths.
        savecast : bool, default False
        If True, persist the generated forecast/scenario data for later reuse.

        Returns
        -------
        None
          Displays a side-by-side plot:
          - Left: SIDI evolution from observed data, What-If scenarios, and forecast ensemble.
          - Right: CDN evolution with uncertainty and relative frequencies of precipitation factors.

        Raises
        ------
        ValueError
          If forecast data is not available or the selected month/year is not within range.
        """

        self._restore_DSO()

        if weight_index is None: #wheter the user does not provide a desired weight_index:
            # use self.DSO.optimal_weight_index (available if optimal SIDI has been recalculated!)
            if hasattr(self.DSO, 'optimal_weight_index'):
                weight_index = self.DSO.optimal_weight_index
            else:# use default values stored in self.DSO.weight_index
                weight_index = self.DSO.weight_index

        # find the index of the forecast window
        if year is None or month is None:
            # Determine the start index as the last observed date
            month, year = self.DSO.m_cal[-1]  # First forecsted month

        # ---- PRODUCING SCENARIOS AND FORECASTS -----------------
        print("processing the FORECASTS...")
        if window is None:
            # year and month are the last observed data afer which the simulations starts
            Forecast = self._forecast_scenarios_rel(year = year,month =month,weight_index=weight_index)

        else:
            Forecast = self._forecast_scenarios_rel(year=year, month=month,window=window,weight_index=weight_index)
        window = Forecast['window'] #trim the window after checks in _forecast_scenarios_rel
        tf1 = Forecast['tf1']
        tf2 = tf1+window
        win_forecast = self._ensemble_window_stats(Forecast)

        # BASIN METADATA to be used in the plots, if desired
        basin = self.DSO.shape.to_crs(epsg=32632)
        area_kmq = basin.geometry.area.iloc[0] / 1e6
        K = self.DSO.K if not hasattr(self.DSO, 'optimal_k') or self.DSO.optimal_k is None else self.DSO.optimal_k
        wlabel = ['eq', 'ldw', 'lgdw', 'liw', 'lgiw'][weight_index]

        print(f"building the WHAT-IF SCENARIOS....using window = {window}")
        scenario_results = self._building_WI_scenarios(year=year, month=month, window=window,
                                                  monte_carlo_runs=monte_carlo_runs, weight_index=weight_index,
                                                       use_montecarlo=use_montecarlo)
        SIDIs = scenario_results["SIDIs"]
        # eSIDIs = scenario_results["var_SIDIs"]
        CDNs = scenario_results["CDNs"]
        # eCDNs = scenario_results["var_CDNs"]
        factors = scenario_results["factors"]
        Probability = scenario_results["Probability"]
        m_cal = scenario_results["Calendar"]

        # --------------------------------------------------------------------------------------

        xlabels = [f"{int(m[0])},{int(m[1])}" for m in m_cal]
        # from month_number\year" to month/year
        xticks_labels = []

        for label in xlabels:
            month_x, year_x = map(int, label.split(','))
            month_name = calendar.month_abbr[month_x]  # abbreviazione (Jan, Feb, etc.)
            xticks_labels.append(f"{month_name}\n{year_x}")

        # colors = cm.grayC
        # e.g. to use half colormap: 0:0.45; or 0:0.95
        # linecolor = ListedColormap(colors(np.linspace(0.25,0.75, len(factors))))

        # # HEATMAP, if desired: ---------------------------------
        # heatmap = np.array(self.DSO.spi_like_set, copy=True)
        # heatmap[:, tf1 + 1::] = np.nan
        # # whenever the SIDI has been recalculated with optimal K:
        # if hasattr(self.DSO, 'optimal_weight_index'):
        #     heatmap = heatmap[0:self.DSO.optimal_k, :].copy()
        # # SET THE COLORMAP FOR THE HEATMAP
        # cmap = heatmap_cmap()
        # bounds = np.array([-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
        # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        title = f'Forecast window: from {calendar.month_abbr[month]}. {year} - Release: {calendar.month_abbr[self.DSO.forecast_m_cal[0, 0]]}. {self.DSO.forecast_m_cal[0, 1]} \n' \
                    f'{self.DSO.basin_name} basin; Baseline: {self.DSO.start_baseline_year} -{self.DSO.end_baseline_year} \n\n'

        x = np.arange(np.shape(SIDIs)[1])

        # ===================
        #  SIDI
        # ==================
        index = r"$\mathit{D}_{\mathit{spi}}$"
        fig1, ax = plt.subplots(figsize=(13, 9))

        # Heatmap
        # ax.imshow(heatmap, cmap=cmap, norm=norm, aspect='auto', alpha=0.7,
        #           extent=[-1, heatmap.shape[1] - 1, 0, heatmap.shape[0]])
        # Set the grid ---------------------------------------
        ax.set_yticks(np.arange(-4,5))
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xticks_labels, fontsize=10)
        ax.axhline(y=-1, linestyle='-', c='dimgrey', alpha=1)
        ax.text(tf1 -5 +2, -1.3, "Severe Drought"+r"$\downarrow$", fontsize=12)
        ax.axvline(x=tf2-window,c='dimgrey')
        ax.spines['left'].set_position(( "data",tf2 - window))
        ax.set_ylim(-5, 5)
        ax.set_xlim(tf1 - 5, tf2)
        ax.grid('both',linestyle=':')
        ax.axvline(x=tf1 - 5, color='gray')

        # What-if curves and annotations ----------------------
        add_whatif_curves(ax, x, SIDIs, tf1, tf2,factors, Probability, month)
        # Observed ---------------------------------------
        ax.plot(np.arange(len(self.DSO.ts)), self.DSO.SIDI[:, weight_index], 'k',
                 linewidth=3, alpha=1, label='Observed')
        # Add forecasts ---------------------------------------
        add_esm_forecast(ax, Forecast['SIDIs'],window=window)


        # Titles and legends ---------------------------------------
        ax.set_title(title, loc='left', linespacing=1.8, fontsize=12)
        fig1.suptitle(f'Observed, Forecast and  What-If scenarios for {index}\n', ha='left', x=0.023,
                     fontsize=16, fontweight=14)
        upper_title = f'OBSERVED & FORECASTED                                                '
        bottom_title = f'WHAT-IF SCENARIOS\n\nNP = normal precipitation\n\n(%): Occurrence likelihood (historical, last 30 years)              \n\n' \
                       f'Exceedance (≥) or Non-Exceedance (≤)'

        add_legends(fig1, ax, upper_title, bottom_title)

        # graphic embellishments ---------------------------------------
        n_dash = 50
        line_string = "|" + "·" * n_dash + r" $\leftarrow$ " + " Observed "
        ax.text(tf1-5.02, 5.2, line_string, ha='left', va='center', fontsize=12)

        n_dash = 64
        line_string = "Forecasted "+r" $\rightarrow$"  + "·" * n_dash + "|"
        ax.text(tf2 - window+0.2, 5.2, line_string, ha='left', va='center', fontsize=12)
        # dove hai tf1, tf2, window definiti
        # add_dot_banner_proportional(ax, tf1=tf1, tf2=tf2, window=window)
        fig1.tight_layout(rect=[0, 0, 0.645, 1])
        if savecast==True:
            safe_index = index.replace('$', '').replace('\\mathit', '').replace('_', '').replace('{', '').replace('}', '')
            self._savecast(index_name=safe_index,month=month,year=year)

        # =========================
        # PLOT 2: CDN
        # =========================
        index = 'CDN'
        fig2, ax = plt.subplots(figsize=(13, 9))
        # Set the grid ---------------------------------------
        ax.set_yticks(np.arange(-30, 35,5))
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xticks_labels, fontsize=10)
        ax.axhline(y=0, linestyle='-', c='dimgrey', alpha=1)
        ax.axvline(x=tf2 - window, c='dimgrey')
        ax.spines['left'].set_position(("data", tf2 - window))

        m = np.round(self.DSO.CDN[-window],-1)
        ax.set_ylim(m-15, m+15)
        ax.set_xlim(tf1 - 5, tf2)
        ax.grid('both', linestyle=':')
        ax.axvline(x=tf1 - 5, color='gray')  # o x=0 se vuoi la cornice a sinistra

        # What-if curves + annotations ---------------------
        add_whatif_curves(ax, x, CDNs, tf1, tf2,factors, Probability, month)
        # Observed CDN ---------------------
        ax.plot(np.arange(len(self.DSO.CDN)), self.DSO.CDN, 'k',
                 linewidth=3, alpha=1, label='Observed')
        # Forecasts ---------------------
        add_esm_forecast(ax, Forecast['CDNs'],window=window)

        # Titles and legends ---------------------
        ax.set_title(title, loc='left', linespacing=1.8, fontsize=12)
        fig2.suptitle(f'Observed, Forecast and  What-If scenarios for {index}\n', ha='left', x=0.023,
                     fontsize=16, fontweight=14)
        upper_title = f'OBSERVED & FORECASTED                                                '
        bottom_title = f'WHAT-IF SCENARIOS\n\nNP = normal precipitation\n\n(%): Occurrence likelihood (historical, last 30 years)              \n\n' \
                       f'Exceedance (≥) or Non-Exceedance (≤)'

        add_legends(fig2, ax, upper_title, bottom_title)

        # graphic embellishments ---------------------------------------
        n_dash = 50
        line_string = "|" + "·" * n_dash + r" $\leftarrow$ " + " Observed "
        ax.text(tf1-5.02, m+15+1.1, line_string, ha='left', va='center', fontsize=12)
        # ax.text(tf2 - window - 1.8, 5.1, r"$\leftarrow$" + "Observed", ha='left', va='center', fontsize=12)

        n_dash = 64
        line_string = "Forecasted "+r" $\rightarrow$"  + "·" * n_dash + "|"
        ax.text(tf2 - window+0.2, m+15+1.1, line_string, ha='left', va='center', fontsize=12)
         # ax.text(tf2 - window + 0.3, 5.1, "Forecasted"r"$\rightarrow$", ha='left', va='center', fontsize=12)
        fig2.tight_layout(rect=[0, 0, 0.645, 1])

        if savecast==True:
            self._savecast(index_name=index,month=month,year=year)

        return  win_forecast

    def _savecast(self,index_name,month,year):
        """
        Persist a cast (forecast or scenario) for later reuse.

        Parameters
        ----------
        index_name : str
            Target index name used in the cast (e.g., "SIDI" or "CDN").
        month : int
            Start month associated with the cast (1–12).
        year : int
            Start year associated with the cast.

        Returns
        -------
        dict
            Minimal metadata describing the saved cast (path, index, start date).
        """

        k = self.DSO.K if not hasattr(self.DSO, 'optimal_k') or self.DSO.optimal_k is None else self.DSO.optimal_k
        w = self.DSO.weight_index if not hasattr(self.DSO, 'optimal_weight_index') or self.DSO.optimal_weight_index is None else self.DSO.optimal_weight_index
        baseline =self.DSO.start_baseline_year,self.DSO.end_baseline_year
        fname=f"ESM_forecast_{index_name}_{self.DSO.basin_name}_{calendar.month_abbr[month]}{year}_k{k}_w{w}_baseline{baseline}.png"
        print(f"saving plot in {os.getcwd()}")
        plt.savefig(fname,
        dpi=300,
        facecolor='w',
        edgecolor='w',
        bbox_inches='tight', #“tight”; None
        pad_inches=0.1, #specifies padding around the image when bbox_inches is “tight”.
        # frameon=None,
        metadata=None)