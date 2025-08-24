import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from drought_scan.utils.drought_indices import *
from matplotlib.colors import Normalize
mpl.rcParams['font.family'] = 'Helvetica'
import os


try:
    import cmcrameri.cm as cmc
except Exception:
    cmc = None


def savefig(fname):
    plt.savefig(fname,
                dpi=300,
                facecolor='w',
                # transparent=True,
                edgecolor='w',
                # orientation='portrait',
                # papertype=None,
                # format=None,
                # transparent=False,
                bbox_inches='tight',  # “tight”; None
                pad_inches=0.1,  # specifies padding around the image when bbox_inches is “tight”.
                # frameon=None,
                metadata=None)
    print(f'fig. saved in {os.getcwd()}')

def heatmap_cmap():
    """
    Creates a custom colormap used for SIDI/CDN plots.
    """
    xmap = plt.get_cmap('RdYlGn', 13)
    cmap = np.array([xmap(i) for i in range(xmap.N)])
    cmap[5, :] = (0.8, 0.8, 0.8, 1)  # Gray for near-neutral SPI
    cmap[6, :] = (0.6, 0.6, 0.6, 1)  # Gray for near-neutral SPI

    return mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmap, xmap.N)


def spi_cmap(n_levels=13):
    if cmc is not None:
        """Create a red-2-green palette using the coulors by Crimeri"""
        # take part of colors by lajolla (red) bamako (green)
        n_half = (n_levels - 2) // 2  # esempio: 5 su 13
        # ROSSI puri: prendiamo solo da index 0.40 a 0.75
        reds = cmc.lajolla(np.linspace(0.3, 0.8, n_half))
        # VERDI saturi: solo da 0.30 a 0.70 e invertiti
        # greens = cmc.bamako(np.linspace(0.30, 0.70, n_half))[::-1]
        # greens = cmc.cork(np.linspace(0.40, 0.75, n_half))[::-1]
        greens = cmc.bam(np.linspace(0.75,1, n_half))
        grays = np.array([[0.9, 0.9, 0.9, 1.0],
                          [0.9, 0.9, 0.9, 1.0]])
        colors = np.vstack([reds, grays, greens])
        return ListedColormap(colors)
    else:
        return heatmap_cmap()

def plot_overview(DSO, optimal_k=None, weight_index=None, year_ext=None,reverse_color=False):
    """
    Plot the drought scan visualization, including CDN, SPI, and SIDI metrics.

    Args:
        DSO: Droguht Scan Obeject: what the user inizialize for istance with DSO = Precipitation(data_path=...)
        otimal_k (int, optional): Optimal number of SPI scales to consider. If provided, the SIDI is recalculated.
        year_ext(tuple, optional): years definining the X-axis limits for the plot. Defaults to None (entire time series).
        weight_index (int, optional): Index of the weighting scheme to use for SIDI calculation.
            - weight_index = 0: Equal weights
            - weight_index = 1: Linear decreasing weights
            - weight_index = 2: Logarithmically decreasing weights (default)
            - weight_index = 3: Linear increasing weights
            - weight_index = 4: Logarithmically increasing weights
        name (string, optional): the name of the basin identified by the shape

    Visualization:
        - Plot 1: Cumulative Deviation of SPI-1 (CDN)
        - Plot 2: Heatmap of SPI scales (1 to K) with transparency control
        - Plot 3: SIDI time series with regions of severe drought highlighted
    """
    if weight_index is None:
        weight_index = 2  # Default to logarithmically decreasing weights

    # Optional recalculation of SIDI with optimal_k
    if optimal_k is not None:
        print(f"Recomputing SIDI with optimal K = {optimal_k}...")
        weights = generate_weights(k=optimal_k)  # Generate weights for the specified K
        sidis = []
        for j in range(len(DSO.m_cal)):
            vec = DSO.spi_like_set[0:optimal_k, j]  # Use only the first optimal_k rows
            sidis.append([weighted_metrics(vec, weights[:, weight_index])[0]])
        SIDI = np.squeeze(np.array(sidis))  # Compute new SIDI
    else:
        SIDI = np.array(DSO.SIDI[:, weight_index], copy=True)  # Use precomputed SIDI

    # ----------------------------------------------------
    # SET THE COLORMAP FOR THE HEATMAP
    # xmap = plt.get_cmap('RdYlGn_r' if reverse_color else 'RdYlGn', 13)
    # cmap = np.array([xmap(i) for i in range(xmap.N)])
    # cmap[5, :] = (0.8, 0.8, 0.8, 1)  # Gray for near-neutral SPI
    # cmap[6, :] = (0.6, 0.6, 0.6, 1)
    # cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmap, xmap.N)
    # USING CRIMERI
    cmap = spi_cmap().reversed() if reverse_color else spi_cmap()


    bounds = np.array([-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3])
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # ----------------------------------------------------
    # CREATE RGBA MATRIX WITH DYNAMIC TRANSPARENCY
    rgba_matrix = cmap(norm(DSO.spi_like_set))  # Convert SPI values to RGBA

    # Replace NaN values with white (RGBA: [1, 1, 1, 1])
    nan_mask = np.isnan(DSO.spi_like_set)
    rgba_matrix[nan_mask] = [1, 1, 1, 1]

    # Adjust transparency for rows below optimal_k
    if optimal_k is not None:
        for i in range(DSO.spi_like_set.shape[0]):
            if i >= optimal_k:
                rgba_matrix[i, :, -1] *= 0.3  # Reduce alpha for rows below optimal_k

    # -------------------------------------------------
    # SET THE COLORMAP FOR CDN (SPI-1 CUMULATIVE DEVIATION)
    # cdnmap = plt.get_cmap('coolwarm' if reverse_color else 'coolwarm_r')
    # USING CRIMERI


    # Generate time labels for the x-axis
    labels = np.array([str(int(c[1])) for c in DSO.m_cal])

    # -------------------------------------------------
    # FIGURE SETTINGS
    # Dynamic figure size based on time series length
    if len(DSO.ts) >= 1200:  # For very long time series (~150 years)
        fig_width = (len(DSO.ts) / 1800) * 20.9
        fig_height = fig_width / 2
    elif (len(DSO.ts) < 1200) & (len(DSO.ts) >= 600):  # For long-to-medium time series (~120 years)
        fig_width = (len(DSO.ts) / 1200) * 20.9
        fig_height = fig_width / 2
    elif (len(DSO.ts) < 600) & (len(DSO.ts) >= 300): #for medium length time series
        fig_width = (len(DSO.ts) / 600) * 20.9
        fig_height = fig_width / 2
    else:  # For shorter time series (<=25 years)
        fig_width = (len(DSO.ts) / 250) * 20.9
        fig_height = fig_width / 1.2

    # Create subplots
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), nrows=3, ncols=1,
                           gridspec_kw={'height_ratios': [1.5, 0.8, 1.5]}, dpi=100)
    fig.subplots_adjust(left=0.07)

    ax = ax.ravel()

    # ----------------------------------------------------
    # PLOT 1: Cumulative Deviation of SPI-1 (CDN)
    # Calcola differenza su una finestra mobile di 6
    window = 36
    xx = np.arange(len(DSO.CDN))
    trend = np.convolve(DSO.CDN, np.ones(window) / window, mode='same')
    diff = np.diff(trend, prepend=trend[0])  # Derivata discreta del trend == trend[1::]-trend[0:-1]
    norm_diff = Normalize(vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))

    # Interpola le dimensioni per allineare con x e y
    # trend_start = window // 2
    # x_plot = xx[trend_start:len(xx) - trend_start]
    # y_plot = DSO.CDN[trend_start:len(xx) - trend_start]
    # diff_plot = diff[trend_start:len(xx) - trend_start]

    ax[0].plot(np.arange(0,len(DSO.CDN)), DSO.CDN, linewidth=1, color='k')
    # Plot colorato - altra versione
    # for i in range(len(xx) - 1):
    #     ax[0].plot(xx[i:i + 2], DSO.CDN[i:i + 2], linewidth=2, color=cdnmap(norm_diff(diff[i])))
    #     # ax[0].plot(x_plot[i:i + 2], y_plot[i:i + 2], linewidth=2,color=cdnmap(norm_diff(diff_plot[i])))


    # # OLD VERSION
    # bal = DSO.CDN  # Compute cumulative deviation
    # xx = np.arange(len(bal))
    # normalize = mpl.colors.Normalize(vmin=-20, vmax=20)
    # ax[0].plot(bal, 'k', label='CDN', linewidth=1, alpha=0.3)
    # for i in range(len(xx) - 1):
    # 	ax[0].fill_between([xx[i], xx[i + 1]],
    # 	                   [bal[i], bal[i + 1]],
    # 	                   color=cdnmap(normalize(bal[i])), alpha=0.6)
    ax[0].axhline(y=0, c='k', linestyle=':', alpha=0.5)
    ax[0].set_xticks([])
    ax[0].set_ylabel('CDN', fontsize=12)
    def round_up(x, base=10):
        return int(-(-x // base) * base)
    ymax = np.max(abs(DSO.CDN))
    ax[0].set_ylim(-round_up(ymax), round_up(ymax))
    plt.setp(ax[0].get_yticklabels(), fontsize=12)

    # ----------------------------------------------------
    # PLOT 2: SPI Heatmap
    xpos = np.round(np.arange(1, DSO.K, DSO.K / 5))

    index_lab = [f"{DSO.index_name}$_{{{int(sub)}}}$" for sub in xpos]
    heatmap = ax[1].imshow(rgba_matrix, aspect='auto', interpolation='none',
                           cmap=cmap)  # Heatmap with dynamic transparency
    ax[1].set_xticks([])
    ax[1].set_yticks(xpos - 1)
    ax[1].set_yticklabels(index_lab, fontsize=12)
    # Add colorbar next to ax[1]
    cbar_ax = fig.add_axes(
        [ax[1].get_position().x1 + 0.01,  # Right edge of ax[1]
         ax[1].get_position().y0,  # Bottom edge of ax[1]
         0.02,  # Width of the colorbar
         ax[1].get_position().height])  # Height of the colorbar
    # cbar = plt.colorbar(heatmap,cax=cbar_ax, orientation='vertical')
    cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
    # cbar.set_ticks(bounds)
    # cbar.set_ticklabels([f"{b:.1f}" for b in bounds])  # Optional: Customize tick labels
    cbar.ax.set_ylabel(f'{DSO.index_name} Value', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # ----------------------------------------------------
    # PLOT 3: SIDI Time Series
    RedArea = np.array(SIDI, copy=True)
    if reverse_color:
        RedArea[RedArea < DSO.threshold] = np.nan  # Highlight anomalies ABOVE threshold
        dot = 0.7
    else:
        RedArea[RedArea > DSO.threshold] = np.nan  # Highlight anomalies BELOW threshold
        dot = 0.3

    ax[2].plot(np.arange(0, len(SIDI)), SIDI, color='k', linewidth=1, label='D', alpha=0.8)
    ax[2].axhline(y=DSO.threshold, c='k', linestyle=':', alpha=0.5)
    ax[2].fill_between(np.arange(0, len(SIDI)), RedArea, DSO.threshold,
                       hatch='xx', color=cmap(dot), linewidth=2, alpha=0.8)
    ax[2].set_xticks(np.arange(0, len(labels[0:-1:12]) * 12, 12))
    ax[2].set_xticklabels(labels[0:-1:12], rotation=90)
    ax[2].set_ylim(-3.5, 3.5)
    ax[2].set_ylabel(rf"$\mathbf{{\mathit{{D}}_{{\{{\mathrm{{{DSO.index_name}}}\}}}}}}$", fontsize=14)
    # ax[2].set_ylabel(r"$\mathbf{\mathit{D}_{\{\mathrm{spi}\}}}$", fontsize=14)
    plt.setp(ax[2].get_yticklabels(), fontsize=12)

    # Set x-axis limits if specified
    for i in range(3):
        if year_ext is None:
            ax[i].set_xlim(36, len(SIDI))
        else:
            try:
                x1 = np.where(DSO.m_cal[:, 1] == year_ext[0])[0][0]
                x2 = np.where(DSO.m_cal[:, 1] == year_ext[1])[0][-1]
            except IndexError:
                raise IndexError(f"provide a tuple of years for xlim within the actual time domain")
            ax[i].set_xlim(x1, x2)

    # Set the title and layout
    basin = DSO.shape.to_crs(epsg=32632)
    # Calcola l'area in metri quadrati
    area_kmq = basin.geometry.area.iloc[0] / 1e6
    K = DSO.K if not hasattr(DSO, 'optimal_k') or DSO.optimal_k is None else DSO.optimal_k
    wlabel = ['eq','ldw','lgdw','liw','lgiw'][weight_index]

    title = f'Drought Scan for {DSO.basin_name}, Area kmq: {int(np.round(area_kmq))}. Baseline: {DSO.start_baseline_year} - {DSO.end_baseline_year}'
    fig.suptitle(title, fontsize=14)
# plt.tight_layout()

def plot_severe_events(DSO, tstartid, duration, deficit, max_events=None, labels=False, unit=None, name=None):
    """
    Generalized plot for severe drought events, ordered by magnitude or duration.

    Args:
        tstartid (ndarray): Indices marking the start of each drought event.
        tendid (ndarray): Indices marking the end of each drought event.
        duration (ndarray): Duration (in time steps) of each drought event.
        deficit (ndarray): Water deficit for each drought event.
        max_events (int, optional): Maximum number of events to plot. Defaults to None (all events).
        unit (str, optional): Unit of measure for the data. Defaults to "mm".
        name (string, optional): the name of the basin identified by the shape
    """
    # Labels for starting dates
    labs_tstart = np.array([f"{int(DSO.m_cal[idx, 0])},{int(DSO.m_cal[idx, 1])}" for idx in tstartid])

    # Sort events by duration or deficit (descending order)
    xi = np.argsort(duration)[::-1]

    # Limit the number of events if specified
    if max_events is None:
        ii = np.where(duration[xi] > 2)
        xi = xi[ii]
    else:
        xi = xi[0:max_events]

    if unit is None:
        unit = 'mm'

    # Create the plot
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 6), dpi=100)
    ax = ax.ravel()

    # Bar plots for water loss
    ax[0].barh(np.arange(len(xi)), deficit[xi] / 10, 0.4, color='orange')
    ax[0].set_yticks(np.arange(len(xi)))
    ax[0].set_yticklabels(labs_tstart[xi])
    ax[0].set_xlabel(f'Water deficit [{unit} *10]')
    ax[0].set_ylabel('Starting date [m,yy]')
    ax[0].grid(axis='x', linestyle='--', alpha=0.6)

    # Bar plots for duration
    ax[1].barh(np.arange(len(xi)), duration[xi], 0.4, color='steelblue')
    ax[1].set_yticks([])
    ax[1].set_xlabel('Duration [months]')
    ax[1].grid(axis='x', linestyle='--', alpha=0.6)

    if labels == True:
        # Annotate values
        for i, v in enumerate(duration[xi]):
            ax[1].text(v + 1, i, f"{v:.0f}", va='center', ha='right')
        # Annotate values
        for i, v in enumerate(deficit[xi] / 10):
            ax[0].text(v - 1, i, f"{v:.1f}", va='center', ha='right')

    # Set the title and layout
    basin = DSO.shape.to_crs(epsg=32632)
    # Calcola l'area in metri quadrati
    area_kmq = basin.geometry.area.iloc[0] / 1e6
    if name is None:
        title = f'Drought Scan, severe events profile. baseline: {DSO.start_baseline_year} - {DSO.end_baseline_year}'
    else:

        title = f'Drought Scan, severe events profile for {name}, Area kmq: {int(np.round(area_kmq))}. Baseline: {DSO.start_baseline_year} - {DSO.end_baseline_year}'
    fig.suptitle(title, fontsize=10)

def plot_cdn_trends(DSO, windows, ax=None,year_ext=None):
    """
    Plot trends in the Cumulative Deviation from Normal (CDN) time series
    over multiple moving window lengths, highlighting the net change
    (translated into mm equivalent) for each period.

    Args:
        DSO: DroughtScan-like object containing the CDN time series,
             method `find_trends(window=...)`, calendar `m_cal`,
             and transformation coefficients `c2r_index`.
        windows (list of int, optional): List of moving window sizes (in months)
             over which to compute and visualize trend magnitudes.

    Notes:
        - For each window, the function calls `DSO.find_trends()` to detect monotonic
          trends and computes the corresponding delta in standardized units.
        - The delta values are rescaled using a climatological coefficient derived
          from the polynomial calibration stored in `DSO.c2r_index`.
        - Bars represent the intensity of the trend (positive or negative), in mm equivalent.
        - The underlying CDN curve is shown as a black line.
        - The dual y-axis allows visualizing both CDN and rescaled trends on the same plot.

    Returns:
        None. Displays a matplotlib figure.
    """
    cmap = plt.get_cmap('Set1')  # o 'Set1', 'Dark2'...
    colors = [cmap(i % cmap.N) for i in range(len(windows))]

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 10), ncols=1, nrows=len(windows))
        ax = ax.ravel()
    else:
        if len(windows) == 1:
            ax = [ax]  # singolo asse -> metti in lista
        else:
            ax = np.asarray(ax).ravel()  # assicura che sia indicizzabile

    normal_values = DSO.normal_values()
    coeff = DSO.c2r_index
    # Coefficiente medio per convertire delta standardizzato in mm
    std_to_mm = np.mean([np.polyval(coeff[0, m, :], 1) - normal_values[m] for m in range(12)])
    anni = np.unique(DSO.m_cal[:, 1])


    for i, window in enumerate(windows):
        R = DSO.find_trends(window=window)
        val = R["delta"] * std_to_mm
        val[R['trend'] == 0] = 0  # annulla dove non c'è trend significativo
        line1,=ax[i].plot(DSO.CDN, '-k',label='CDN')
        ax[i].set_ylabel('CDN', fontsize=12)
        # ax[i].legend(loc=2)
        ax[i].set_xticks(np.arange(0, len(val), 12))
        ax[i].set_xticklabels(anni, rotation=90)

        ax2 = ax[i].twinx()
        line2= ax2.bar(np.arange(len(val)), val, color=colors[i],alpha=0.3, label=f'Trend {window} mesi')

        ax2.axhline(y=0,color='lightgrey')
        ax2.set_ylabel('Change [mm]', fontsize=12)
        ax2.set_xlim(36, len(val))
        # Trova massimo assoluto del valore da rappresentare
        ymax = np.nanmax(np.abs(val))

        # Arrotonda al multiplo di 100 superiore
        ymax_rounded = int(np.ceil(ymax / 100.0)) * 100

        # Espandi se non bastano 7 tick (devono andare da –N a +N)
        if ((2 * ymax_rounded // 100 + 1) < 7):
            ymax_rounded = 300  # minimo range da –300 a 300 → 7 tick

        # Crea i tick: es. da –300 a 300 con passo 100
        if ymax_rounded<1000:
            yticks = np.arange(-ymax_rounded, ymax_rounded + 1,250)
        else:
            yticks = np.arange(-ymax_rounded, ymax_rounded + 1, 300)


        # Applica a ax2
        ax2.set_yticks(yticks)
        ax2.set_ylim(yticks[0], yticks[-1])

        # ax2.legend(loc=2,bbox_to_anchor=(0,0.80))
        # Combina le due curve per una legenda unica
        lines = [line1, line2[0]] #take only a proxy for the barharty
        labels = ['CDN' , f'Trend {window} mesi']
        # Aggiungi la legenda a uno degli assi (es. ax1)
        ax[i].legend(lines, labels, loc='upper left')

        if year_ext is None:
            ax[i].set_xlim(36, len(DSO.CDN))
        else:
            x1 = np.where(DSO.m_cal[:, 1] == year_ext[0])[0]
            x2 = np.where(DSO.m_cal[:, 1] == year_ext[1])[0]

            if len(x1) == 0:
                raise ValueError(f"The first year in year_ext={xlim} is outside the available time domain: "
                                 f"{int(DSO.m_cal[0, 1])}–{int(DSO.m_cal[-1, 1])}")

            if len(x2) == 0:
                x2 = len(DSO.CDN)
                print(f"The domain has been closed at year {int(DSO.m_cal[x2 - 1, 1])}.")
            else:
                x2 = x2[-1]  # include the last instance of the year (e.g., December)

            ax[i].set_xlim(x1[0], x2)

    fig.tight_layout()

def monthly_profile(DSO, var=None, cumulate=False, highlight_years=None,two_year=False):
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
        if var is None:
            x = DSO.ts.copy()
        else:
            x = var

        if len(x) != len(DSO.m_cal):
            raise ValueError("Input variable and m_cal must have the same length.")

        months = DSO.m_cal[:, 0]
        years = DSO.m_cal[:, 1]
        unique_years = np.unique(years)

        monthly_means = np.zeros(12)
        perc_25 = np.zeros(12)
        perc_75 = np.zeros(12)
        perc_10 = np.zeros(12)
        perc_90 = np.zeros(12)

        if not cumulate:
            for month in range(1, 13):
                monthly_data = x[months == month]
                monthly_means[month - 1] = np.mean(monthly_data)
                perc_25[month - 1] = np.percentile(monthly_data, 25)
                perc_75[month - 1] = np.percentile(monthly_data, 75)
                perc_10[month - 1] = np.percentile(monthly_data, 10)
                perc_90[month - 1] = np.percentile(monthly_data, 90)
        else:
            annual_cumsum = {year: np.zeros(12) for year in unique_years}
            for year in unique_years:
                for month in range(1, 13):
                    monthly_data = x[(years == year) & (months == month)]
                    cumulative = np.sum(monthly_data)
                    annual_cumsum[year][month - 1] = (
                        cumulative + annual_cumsum[year][month - 2] if month > 1 else cumulative
                    )

            for month in range(12):
                values = [annual_cumsum[year][month] for year in unique_years]
                monthly_means[month] = np.mean(values)
                perc_25[month] = np.percentile(values, 25)
                perc_75[month] = np.percentile(values, 75)
                perc_10[month] = np.percentile(values, 10)
                perc_90[month] = np.percentile(values, 90)

        # Extend the monthly stats to 24 months by repeating the cycle
        months_n = np.arange(1, 25) if two_year else np.arange(1, 13)
        nyears=2 if two_year else 1
        mean_n = np.tile(monthly_means, nyears)
        x_ticks = np.tile(np.arange(1,13),nyears)
        p25_24 = np.tile(perc_25, nyears)
        p75_24 = np.tile(perc_75, nyears)
        p10_24 = np.tile(perc_10, nyears)
        p90_24 = np.tile(perc_90, nyears)

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(months_n, mean_n, color='darkgray', label='Monthly Mean', linewidth=2)
        plt.fill_between(months_n, p25_24, p75_24, color='gray', alpha=0.5, label='25–75 Percentile')
        plt.fill_between(months_n, p10_24, p90_24, color='lightgray', alpha=0.5, label='10–90 Percentile')

        # Highlight years if specified
        if highlight_years is not None:
            if isinstance(highlight_years, int):
                highlight_years = [highlight_years]
            elif not isinstance(highlight_years, list):
                raise TypeError("highlight_years must be an int, a list of ints, or None.")

            colors = ['orange', 'cyan', 'green']
            for i, year in enumerate(highlight_years[:3]):
                if year in unique_years and (year - 1) in unique_years:
                    # Build 24-month series from year-1 and year
                    if cumulate:
                        data_prev = annual_cumsum[year - 1]
                        data_curr = annual_cumsum[year]
                    else:
                        data_prev = [np.mean(x[(months == month) & (years == year - 1)]) for month in range(1, 13)]
                        data_curr = [np.mean(x[(months == month) & (years == year)]) for month in range(1, 13)]

                    full_series = np.concatenate([data_prev, data_curr]) if two_year else data_curr
                    label = f'{year - 1}-{year}' if two_year else f'{year}'
                    plt.plot(months_n, full_series, color=colors[i], linewidth=2, label=label)
                else:
                    print(f"Warning: Cannot plot N-months profile for year {year} (missing previous year).")

        plt.xlabel('Month')
        plt.ylabel('Cumulative Value' if cumulate else 'Mean Value')
        title = f'DS.ts Monthly Profile - {DSO.basin_name}'
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.xticks(months_n,x_ticks)
        plt.tight_layout()
        plt.show()




#--------------------------------------------------------------------
# function and settings for plotting FORECASTS
# -------------------------------------------------------------------

def add_esm_forecast(ax, forecast_data, window, color_mean="#4A001F"):
    """Plot forecast median line and percentile bands."""

    fmean = np.nanmedian(forecast_data, axis=0)
    p = {k: np.percentile(forecast_data, k, axis=0) for k in [1, 10, 25, 75, 90, 99]}
    x = np.arange(len(fmean))

    ax.plot(x[-window - 1::], fmean[-window - 1::], color_mean, linewidth=4,
            label='Forecast (Ens. Median)', alpha=0.7)
    ax.fill_between(x, p[25], p[75], color=color_mean, linewidth=0, alpha=0.2,
                    label='25°–75° percentile range')
    # 'mediumorchid' 0.2; 'violet' 0.1
    ax.fill_between(x, p[10], p[90], color=color_mean, linewidth=0, alpha=0.1,
                    label='10°–90° percentile range')
    # ax.fill_between(x, p[1], p[99], color="darkslateblue", linewidth=0, alpha=0.1, label='1–99 percentile range')

def add_ml_forecast(ax, prediction,rmse,tf1,window, color_mean="#4A001F"):
    """Plot forecast median line and percentile bands."""
    x = np.arange(len(prediction))
    ax.plot(x[-window - 1::], prediction[-window - 1::], color_mean, linewidth=4,
            label='ML Forecast (F)             ', alpha=0.7)

    # shaodows
    ax.fill_between(np.arange(tf1, tf1 + 1 + window),
                     prediction[tf1:tf1 + 1 + window] - rmse,
                     prediction[tf1:tf1 + 1 + window] + rmse, color=color_mean,
                     label='F ± 1 RMSE                  ', alpha=0.2)
    ax.fill_between(np.arange(tf1, tf1 + 1 + window),
                     prediction[tf1:tf1 + 1 + window] - 2 * rmse,
                     prediction[tf1:tf1 + 1 + window] + 2 * rmse, color=color_mean,
                     label='F ± 2 RMSE                  ', alpha=0.1)


     # ax.fill_between(x, p[1], p[99], color="darkslateblue", linewidth=0, alpha=0.1, label='1–99 percentile range')

def add_whatif_curves(ax, x, curves, tf1,tf2, factors, Probability, month):
    letters = [r'( $\mathit{g}$ )', r'( $\mathit{f}$ )', r'( $\mathit{e}$ )', r'( $\mathit{d}$ )',
               r'( $\mathit{c}$ )', r'( $\mathit{b}$ )', r'( $\mathit{a}$ )']
    for ii, factor in enumerate(factors):
        # print(ii,factors)
        width = ((np.log1p(Probability[ii, month - 1]) ** 2) / 3) ** 1.1
        if width<=0.05:
            width=0.5
        # print(width)
        prob_text = ('≤' if ii < 3 else '≥')
        color = ("darkgoldenrod" if ii < 3 else "lightseagreen")

        if ii == len(factors):
            ax.plot(x[tf1::], np.zeros((len(x[tf1::]),)) * np.nan, color='w', alpha=0, label='    ')

        ax.plot(x[tf1::], curves[ii, tf1::], c=color, linewidth=width,
                linestyle='--', alpha=1,
                label=f'{letters[ii]} {factor} x NP ({prob_text} {int(Probability[ii, month - 1])}%)   ')

        ax.text(x[tf2] - 0.45, curves[ii, tf2] + 0.28, letters[ii], fontsize=11)


def add_legends(fig, ax, upper_title, bottom_title,only_WI=False):
    """Place two legends below the figure."""
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[::-1]
    labels = labels[::-1]
    if only_WI==False:
        group2_handles, group2_labels = handles[4:], labels[4:]
        group1_handles, group1_labels = handles[:4][::-1], labels[:4][::-1]


        fig.legend(group1_handles, group1_labels, loc='upper right', bbox_to_anchor=(0.99, 0.81),
                   title=upper_title, ncol=2, fontsize=11, title_fontsize=12, labelspacing=1.05)
        fig.legend(group2_handles, group2_labels, loc='upper right', bbox_to_anchor=(0.99, 0.69),
                   title=bottom_title, ncol=2, fontsize=11, title_fontsize=12, labelspacing=1.05)

    else:
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.81),
                   title=bottom_title, ncol=2, fontsize=11, title_fontsize=12, labelspacing=1.05)

def compute_n_dash(window, pad_left=5, n_left_ref=50, n_right_ref=64):
    """
    Calibra i puntini '·' per i due segmenti in funzione della larghezza (mesi).
    - pad_left: mesi del segmento observed (costante nel tuo grafico)
    - window:   mesi del segmento forecast (variabile 1..7)
    - n_left_ref / n_right_ref: numero puntini misurati nel caso di riferimento
    - window_ref: finestra forecast nel caso di riferimento (es. 6)

    Ritorna (n_left, n_right) come interi.
    """

    window_ref = 6
    tot_win = window_ref+pad_left
    dens_left_ref  = (n_left_ref+n_right_ref)  * (pad_left/tot_win )     # puntini per mese lato observed
    dens_right_ref = (n_left_ref+n_right_ref)* (window_ref/tot_win )    # puntini per mese lato forecast

    n_left  = int(round(dens_left_ref  * ((tot_win-window)/pad_left)))   # resta ~costante (= n_left_ref)
    n_right = int(round(dens_right_ref  * ((tot_win-pad_left)/window)))     # scala con window

    return n_left, n_right


def add_dot_banner_proportional(ax, tf1, tf2, window, pad_left=5):
    # Calcola puntini proporzionali
    n_left, n_right = compute_n_dash(window)

    y_frac = 1.2

    # Stringhe
    left_str  = "|" + "·"*n_left  + r" $\leftarrow$ " + " Observed "
    right_str = " Forecasted " + r" $\rightarrow$ " + "·"*n_right + "|"

    # Centri dei segmenti (in coordinate x dati), y in frazione asse
    x_left   = tf1 - pad_left
    x_split  = tf2 - window
    x_right  = tf2

    x_center_left  = 0.5 * (x_left  + x_split)
    x_center_right = 0.5 * (x_split + x_right)

    trans = ax.get_xaxis_transform()  # x in dati, y in [0..1]
    ax.text(x_center_left,  y_frac, left_str,  ha="center", va="center", transform=trans, fontsize=12)
    ax.text(x_center_right, y_frac, right_str, ha="center", va="center", transform=trans, fontsize=12)

# ---------------------------------------------------------
# SHAP
# -------------------------------------------------------



def plot_forecast_shap(localshap,month,year,feat_names,basin_name,mode='classic'):
    # A = localshap[:, :, 0]  # (max_lag, n_features)
    # feat_names = np.array(self.feature_labels)
    # xmap =plt.get_cmap('coolwarm',13)
    n_lag, n_feat = localshap.shape
    n_ticks=15
    xmap =cmc.vik_r.resampled(n_ticks)
    cmap = np.array([xmap(i) for i in range(xmap.N)])
    cmap[6, :] = (0.8, 0.8, 0.8, 1)  # Gray for near-neutral SPI
    cmap[7, :] = (0.8, 0.8, 0.8, 1)  # Gray for near-neutral SPI
    cmap[8, :] = (0.8, 0.8, 0.8, 1)
    cmap= ListedColormap(cmap)

    if mode == 'classic':
        A = localshap.astype(float)
        text_label = 'SHAP (CDN, contributo additivo)'
        title_mode = 'Local SHAP (signed)'
    elif mode == 'relative':
        # somma per lead (lungo le feature)
        col_sum = np.sum(localshap, axis=1)  # shape (n_lag,)
        # gestione casi quasi-zero per stabilità
        eps = 1e-9
        safe_den = np.where(np.abs(col_sum) < eps, np.nan, col_sum)  # evita divisione per ~0
        # quota percentuale: per-feature share del totale del lead
        A = (localshap / safe_den[:, None]) * 100.0  # (n_lag, n_feat)
        text_label = 'Contributo relativo (%) sul totale SHAP del lead'
        title_mode = 'Relative contribution per lead (%)'
    else:
        raise ValueError('mode deve essere "classic" oppure "relative"')

    v = np.nanmax(np.abs(A))

    # confini delle bande (14 confini per 13 colori)
    bounds = np.linspace(-v, v, cmap.N + 1)
    norm = BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(10, 0.35 * n_feat))
    im = plt.imshow(A.T, aspect='auto', cmap=cmap, norm=norm)

    # tick al centro di ogni fascia
    tick_vals = (bounds[:-1] + bounds[1:]) / 2

    cbar = plt.colorbar(im, boundaries=bounds, ticks=tick_vals)
    cbar.set_label(text_label)
    cbar.set_ticklabels([f"{t:.2f}" for t in tick_vals])

    plt.xlabel('Lead months',fontsize=12)
    plt.xticks(ticks=np.arange(n_lag), labels=np.arange(1, n_lag + 1))

    K = min(25, n_feat)
    plt.yticks(ticks=np.arange(K), labels=feat_names[:K])
    plt.ylim(K - 0.5, -0.5)

    plt.title(f'{basin_name} — {title_mode}\nCDN Forecast {month}/{year}', fontsize=14)
    plt.tight_layout()


    # # ordina le feature per importanza media assoluta
    # order = np.argsort(np.nanmean(np.abs(A), axis=0))[::-1]
    # A_ord = A[:, order].T                         # (n_features, max_lag) per plot
    # labels_ord = feat_names[order]
    #
    # import matplotlib.pyplot as plt
    # v = np.nanmax(np.abs(A_ord))
    # plt.figure(figsize=(10, 0.35*len(labels_ord)))  # altezza proporzionale alle feature
    # im = plt.imshow(A_ord, aspect='auto', cmap='coolwarm', vmin=-v, vmax=+v)
    # plt.colorbar(im, label='SHAP (CDN, unità modello)')
    # plt.xlabel('Lead (lag)')
    # plt.ylabel('Feature')
    # plt.xticks(ticks=np.arange(A.shape[0]), labels=np.arange(1, A.shape[0]+1))
    # # mostra solo le top-K se sono tante:
    # K = min(25, len(labels_ord))
    # plt.yticks(ticks=np.arange(K), labels=labels_ord[:K])
    # plt.ylim(K-0.5, -0.5)
    # plt.title('Local SHAP su forecast CDN per lag=1..max_lag')
    # plt.tight_layout()
    # plt.show()



