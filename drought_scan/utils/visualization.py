"""
author: PyDipa
# © 2025 Arianna Di Paola
# License: GNU General Public License v3.0 (GPLv3)

Custom functions for visualization
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from drought_scan.utils.drought_indices import *
from drought_scan.utils.statistics import *
from matplotlib.colors import Normalize
import os

import matplotlib as mpl
import matplotlib.font_manager as fm
available = {f.name for f in fm.fontManager.ttflist}
if 'Helvetica' in available:
    mpl.rcParams['font.family'] = 'Helvetica'
else:
    mpl.rcParams['font.family'] = 'Arial'


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


  # -------------------- Dimenion of the full plot --------------------
def _figure_size_for_length(n: object) -> object:
    """
    Compute figure width and height as a continuous function of series length n,
    clamped within predefined min/max limits and rounded to 1 decimal place.
    """
    # coefficiente di scala derivato dal caso n=1200 -> w≈20.9
    k = 20.9 / 1200.0
    w = k * n

    # clamp tra valori minimi e massimi (dai tuoi vecchi casi)
    w = max(20.9 / 250 * n, min(w, 20.9 / 1800 * n * 1800 / 1200))  # opzionale se vuoi stringere
    w = min(max(w, 20.9 / 250 * 250), 20.9 / 1800 * 1800)  # clamp effettivo tra 20.9 e ~20.9 (scalato)

    # oppure più semplice: clamp tra 10 e 21
    w = min(max(w, 10), 21)

    # altezza come proporzione della larghezza
    h = w / 2 if n >= 300 else w / 1.2

    return round(w, 1), round(h, 1)

def spi_cmap(n_levels=13):
    """Create a red-2-green palette using the colors by cmcrameri (falls back
    to heatmap_cmap() if cmcrameri is not installed)."""
    if cmc is not None:
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

def highlight_drought(ax, series, color='brown' ,offset=None, threshold=None):
    """
      Shade vertical bands over contiguous intervals where the series is below a threshold.

      The function detects runs where ``series < threshold`` and draws semi-transparent
      vertical spans on the given Axes to highlight drought periods. Horizontal positions
      are interpreted as integer sample indices (0, 1, 2, ...). If your plotted x-values
      are shifted relative to these indices, use ``offset`` to align the bands.

      Parameters
      ----------
      ax : matplotlib.axes.Axes
          Axes to draw on.
      series : array-like of shape (n,)
          1D numeric sequence (e.g., NumPy array or pandas Series). NaNs are treated
          as non-drought and break contiguous intervals.
      color : str or tuple, default: 'brown'
          Color of the shaded bands.
      offset : int or float, optional
          Horizontal shift added to band positions. Useful when the plot’s x-axis
          does not directly match indices 0..n-1. If None, no shift is applied.
      threshold : float, optional
          Drought threshold; values strictly less than this are highlighted. If None,
          a default of -1 is used.

      Returns
      -------
      None
          Draws on ``ax`` and returns nothing.

      Notes
      -----
      - The Bands are used in the plot_scan() method which recall the plot_overview() functiont to highlight drought in the CDN curve

    """

    if threshold is None:
        threshold = -1

    if threshold <0:
        below = series < threshold
    else: #for Pet and Temperature indices or for classes where the stress condiction are above the threshold
        below = series > threshold

    starts = np.where((~below[:-1]) & below[1:])[0] + 1
    ends = np.where(below[:-1] & (~below[1:]))[0] + 1
    if below[0]:
        starts = np.insert(starts, 0, 0)
    if below[-1]:
        ends = np.append(ends, len(series))
    for start, end in zip(starts, ends):
        if offset is None:
            ax.axvspan(start-0.5, end-0.5, color=color, alpha=0.1)
        else:
            ax.axvspan(start +offset-0.5, end +offset-0.5, color=color, alpha=0.1)

# -------------------- Dimenion of the full plot --------------------
    def _figure_size_for_length(n):
        """
        Compute figure width and height as a continuous function of series length n,
        clamped within predefined min/max limits and rounded to 1 decimal place.
        """
        # coefficiente di scala derivato dal caso n=1200 -> w≈20.9
        k = 20.9 / 1200.0
        w = k * n

        # clamp tra valori minimi e massimi (dai tuoi vecchi casi)
        w = max(20.9 / 250 * n, min(w, 20.9 / 1800 * n * 1800 / 1200))  # opzionale se vuoi stringere
        w = min(max(w, 20.9 / 250 * 250), 20.9 / 1800 * 1800)  # clamp effettivo tra 20.9 e ~20.9 (scalato)

        # oppure più semplice: clamp tra 10 e 21
        w = min(max(w, 10), 21)

        # altezza come proporzione della larghezza
        h = w / 2 if n >= 300 else w / 1.2

        return round(w, 1), round(h, 1)

def plot_overview(DSO, optimal_k=None, weight_index=None, year_ext=None, split_plot=False,
                  plot_order=None,figsize=None):
    """
    Plot the drought scan visualization: Heatmap(H),  SIDI(S) and  CDN (C),

    Parameters
    ----------
    DSO : DroughtScan object
        Instance containing SPI, SIDI, CDN, and metadata (calendar, basin info).
    optimal_k : int, optional
        If provided, SIDI is recomputed using the specified optimal K.
    weight_index : int, optional
        Index of the weighting scheme for SIDI calculation.
        Defaults to 2 (geometrically decreasing weights).
        Options:
            0 = equal weights
            1 = linear decreasing
            2 = geometric decreasing
            3 = linear increasing
            4 = geometric increasing
    year_ext : tuple(int, int), optional
        (start_year, end_year). If provided, limits the x-axis to this period.
    split_plot : bool, default=False
        If True, each panel (CDN, Heatmap, SIDI) is plotted in a separate figure,
        each with its own x-axis labels.
        If False, all panels are combined in a single figure.
    plot_order : str, default='CHS'
        Order of the panels when split_plot=False.
        Must be any permutation of 'H', 'S', 'C':
            'CHS' -> CDN (top), Heatmap (middle), SIDI (bottom)
            'HSC' -> Heatmap (top), SIDI (middle), CDN (bottom)
            'SCH' -> SIDI (top), CDN (middle), Heatmap (bottom)
        The heatmap panel always uses a smaller height ratio, regardless of position.

    Visualization
    -------------
    - CDN panel: cumulative deviation of SPI-1.
    - Heatmap panel: SPI values across multiple timescales with dynamic transparency.
    - SIDI panel: weighted drought index with shaded drought events.

    Notes
    -----
    - When split_plot=False, only the bottom panel shows year labels on the x-axis.
    - When split_plot=True, all figures display year labels independently.
    - The heatmap panel is always thinner than the others to improve readability.
    """


    # -------------------- parametri di default --------------------
    # One resolver for the whole library (BaseDroughtAnalysis._resolve_weight_index):
    # explicit argument -> committed calibration -> construction default -> 2. This
    # used to hardcode 2, so a plot drawn after set_optimal_SIDI(k, w) showed column 2
    # rather than the calibrated column w.
    weight_index = DSO._resolve_weight_index(weight_index)


    # -------------------- SIDI: ricalcolo opzionale --------------------
    if optimal_k is not None:
        SIDI = DSO.recalculate_SIDI(K=optimal_k)[:, weight_index]
        weights = generate_weights(k=optimal_k)
        # sidis = []
        # for j in range(len(DSO.m_cal)):
        #     vec = DSO.spi_like_set[0:optimal_k, j]
        #     sidis.append([weighted_metrics(vec, weights[:, weight_index])[0]])
        # SIDI = np.squeeze(np.array(sidis))
    else:
        SIDI = np.array(DSO.SIDI[:, weight_index], copy=True)

    # -------------------- colormap e norm per heatmap --------------------
    cmap = spi_cmap().reversed() if DSO.threshold > 0 else spi_cmap()
    bounds = np.array([-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3])
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    rgba_matrix = cmap(norm(DSO.spi_like_set))  # (K, T, 4)
    nan_mask = np.isnan(DSO.spi_like_set)
    rgba_matrix[nan_mask] = [1, 1, 1, 1]

    if optimal_k is not None:
        for i in range(DSO.spi_like_set.shape[0]):
            if i >= optimal_k:
                rgba_matrix[i, :, -1] *= 0.3  # trasparenza per scale > Kopt

    # -------------------- xticks labels --------------------
    labels = np.array([str(int(c[1])) for c in DSO.m_cal]).astype(int)  # anni

    # -------------------- graphical helpers  --------------------
    def _apply_xlim(ax):
        if year_ext is None:
            ax.set_xlim(DSO.K, len(SIDI))
        else:
            try:
                x1 = np.where(DSO.m_cal[:, 1] == year_ext[0])[0][0]
                x2 = np.where(DSO.m_cal[:, 1] == year_ext[1])[0][-1]
            except IndexError:
                raise IndexError("provide a tuple of years for xlim within the actual time domain")
            ax.set_xlim(x1, x2)

    def _x_year_ticks(ax, show=True):
        if show:
            ax.set_xticks(np.arange(0, len(labels[0:-1:12]) * 12, 12))
            ax.set_xticklabels(labels[0:-1:12], rotation=90)
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])

    def _round_up(x, base=10):
        return int(-(-x // base) * base)

    def _panel_C(ax, show_xticks):
        # CDN
        ax.plot(np.arange(0, len(DSO.CDN)), DSO.CDN, linewidth=1, color='k')
        highlight_drought(ax, SIDI, threshold=DSO.threshold)
        ax.axhline(y=0, c='k', linestyle=':', alpha=0.5)
        _x_year_ticks(ax, show=show_xticks)
        ax.set_ylabel('CDN', fontsize=12)
        ymax = np.max(abs(DSO.CDN))
        ax.set_ylim(-_round_up(ymax), _round_up(ymax))
        plt.setp(ax.get_yticklabels(), fontsize=12)
        _apply_xlim(ax)

    def _panel_H(ax, add_colorbar=True, show_xticks=False):
        # Heatmap SPI_k
        heat = ax.imshow(rgba_matrix, aspect='auto', interpolation='none', cmap=cmap)
        _x_year_ticks(ax, show=show_xticks)
        xpos = np.round(np.arange(1, DSO.K, max(1, DSO.K / 5)))
        index_lab = [f"{DSO.index_name}$_{{{int(sub)}}}$" for sub in xpos]
        ax.set_yticks(xpos - 1)
        ax.set_yticklabels(index_lab, fontsize=12)
        if add_colorbar:
            cbar_ax = ax.inset_axes([1.01, 0.0, 0.02, 1.0])  # a destra del pannello
            mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
            cbar_ax.set_ylabel(f'{DSO.index_name} value', fontsize=12)
            cbar_ax.tick_params(labelsize=10)
        _apply_xlim(ax)


    def _panel_S(ax, show_xticks):
        # SIDI
        RedArea = np.array(SIDI, copy=True)
        if DSO.threshold >= 0:
            RedArea[RedArea < DSO.threshold] = np.nan
            dot = 0.7
        else:
            RedArea[RedArea > DSO.threshold] = np.nan
            dot = 0.3

        ax.plot(np.arange(0, len(SIDI)), SIDI, color='k', linewidth=1, alpha=0.8, label='D')
        ax.axhline(y=DSO.threshold, c='k', linestyle=':', alpha=0.5)
        ax.fill_between(np.arange(0, len(SIDI)), RedArea, DSO.threshold,
                        hatch='xx', color=cmap(dot), linewidth=2, alpha=0.8)
        _x_year_ticks(ax, show=show_xticks)
        ax.set_ylim(-3.5, 3.5)
        ax.set_ylabel(f"{DSO.SIDI_name}", fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=12)
        _apply_xlim(ax)


    # Map symbols
    PANELS = {
        'C': _panel_C,
        'H': _panel_H,
        'S': _panel_S,
    }

    if plot_order is None:
        plot_order = 'CHS'
    # validazione ordine: deve essere una permutazione di 'HSC'
    plot_order = plot_order.upper()
    if len(plot_order) != 3 or set(plot_order) != set("HSC"):
        raise ValueError("plot_order deve essere una permutazione di 'HSC', es. 'CHS', 'HSC', 'SCH', ...")

    order = list(plot_order)  # es. ['S','C','H']
    # # order check
    # if plot_order not in ('CHS', 'HSC'):
    #     raise ValueError("plot_order deve essere 'CHS' (default) o 'HSC'.")

    order = list(plot_order)

    # -------------------- PLOTTING --------------------
    if not split_plot:
        # unic panel with 3 subplots
        if figsize is None:
            fig_w, fig_h = _figure_size_for_length(len(DSO.ts))
        else:
            fig_w, fig_h = figsize
        ratios = [0.8 if key == 'H' else 1.5 for key in order]
        fig, axes = plt.subplots(
            nrows=3, ncols=1, figsize=(fig_w, fig_h),
            gridspec_kw={'height_ratios': ratios }, dpi=100
        )
        fig.subplots_adjust(left=0.07)
        axes = axes.ravel()

        # top and mid withput xticks; bottom with yearly labels
        for i, key in enumerate(order):
            show_xt = (i == 2)  # solo il pannello in fondo mostra anni
            if key == 'H':
                PANELS[key](axes[i], add_colorbar=True, show_xticks=show_xt)
            else:
                PANELS[key](axes[i], show_xticks=show_xt)

        # plot title

        title = (f"Drought Scan for {DSO.basin_name}, Area kmq: {int(np.round(DSO.area_kmq))}. "
                 f"Baseline: {DSO.start_baseline_year} - {DSO.end_baseline_year}")
        fig.suptitle(title, fontsize=14)
        fig.subplots_adjust(left=0.1)
        plt.show(block=False)

    else:
        # 3 single plots
        for key in order:
            if figsize is None:
                fig_w, fig_h = _figure_size_for_length(len(DSO.ts))
            else:
                fig_w, fig_h = figsize
            fig, ax = plt.subplots(figsize=(fig_w, fig_h / 3), dpi=100)
            if key == 'H':
                PANELS[key](ax, add_colorbar=True, show_xticks=True)
                ax.set_title(f"{DSO.index_name} heatmap", fontsize=14)
            elif key == 'C':
                PANELS[key](ax, show_xticks=True)
                ax.set_title("CDN", fontsize=14)
            else:  # 'S'
                PANELS[key](ax, show_xticks=True)
                ax.set_title(f"{DSO.SIDI_name}", fontsize=14)

            # single titles
            subtitle = (f"{DSO.basin_name} | Baseline {DSO.start_baseline_year}-{DSO.end_baseline_year} | "
                        f"Area {int(np.round(DSO.area_kmq))} km²")
            fig.suptitle(subtitle, fontsize=14, y=0.96)
            fig.subplots_adjust(left=0.1)
            plt.tight_layout()
            plt.show(block=False)

def plot_severe_events(DSO, tstartid, duration, deficit, max_events=None, labels=False, unit=None, name=None):
    """
    Generalized plot for severe drought events, ordered by magnitude or duration.

    Args:
        DSO (object): DroughtScan object providing the calendar (`DSO.m_cal`) for event labels.
        tstartid (ndarray): Indices marking the start of each drought event.
        duration (ndarray): Duration (in time steps) of each drought event.
        deficit (ndarray): Water deficit for each drought event.
        max_events (int, optional): Maximum number of events to plot. Defaults to None (all events).
        labels (bool, optional): If True, annotate bars with event start dates. Defaults to False.
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
    title = f'Drought Scan, severe events profile for {DSO.basin_name}. Baseline: {DSO.start_baseline_year} - {DSO.end_baseline_year}'
    fig.suptitle(title, fontsize=12)
    plt.show(block=False)

def plot_cdn_trends_old(DSO, windows, figsize=(14,10),ax=None,year_ext=None,unit=None,show_spi=False):
    """
    Plot trends in the Cumulative Deviation from Normal (CDN) time series
    over multiple moving window lengths, highlighting the net change
    (translated into mm equivalent) for each period.

    Args:
        DSO: DroughtScan-like object containing the CDN time series,
             method `find_trends(window=...)`, calendar `m_cal`,
             and transformation coefficients `c2r_index`.
        windows (list of int): List of moving window sizes (in months)
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
    from numpy.lib.stride_tricks import sliding_window_view
    from drought_scan.core import Streamflow


    cmap = plt.get_cmap('Set1')  # o 'Set1', 'Dark2'...
    colors = [cmap(i % cmap.N) for i in range(len(windows))]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, ncols=1, nrows=len(windows))
        ax = ax.ravel()
    else:
        if len(windows) == 1:
            ax = [ax]  # singole axis
        else:
            ax = np.asarray(ax).ravel()  # make it iterable
        fig = ax[0].figure

    anni = np.unique(DSO.m_cal[:, 1]).astype(int)
    normal_values = DSO._monthly_normals()   # 12 valori Gen..Dic (era normal_values()[0:12])
    coeff = DSO.c2r_index
    # average std, used to move from delta changes into mm
    # std_to_native_rate = [np.polyval(coeff[0, m, :], 1) - normal_values[m] for m in range(12)]
    # std_to_native_rate = np.mean(np.absolute( std_to_native_rate))
    DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    avg_seconds_per_month = np.mean(DAYS_IN_MONTH) * 86400  # ~2.63e6 s



    # Quanto vale 1σ (z=+1) in unità native (mm per Precipitation, m³/s per Streamflow), mese per mese
    std_to_native_rate_monthly = np.abs(np.array([
        c2r_eval(coeff[0, m, :], 1, DSO.calculation_method) - normal_values[m] for m in range(12)
    ]))

    # Mappo il rate stagionale su ogni step della serie temporale
    cal_months_0 = DSO.m_cal[:, 0].astype(int) - 1  # 0-based
    rate_series = std_to_native_rate_monthly[cal_months_0]  # shape = (n,)


    for i, window in enumerate(windows):
        R = DSO.find_trends(window=window)
        # Rate medio sulla finestra che termina al mese corrente (allineato come delta)
        rate_w = np.full(len(rate_series), np.nan)
        if len(rate_series) >= window:
            rate_w[window - 1:] = sliding_window_view(rate_series, window).mean(axis=1)

        # Conversione delta → unità fisiche
        if isinstance(DSO, Streamflow):
            val = R["delta"] #* rate_w * avg_seconds_per_month
            unit = "m³"
        else:
            val = R["delta"] #* rate_w
            unit = "mm"

        val[R['trend'] == 0] = 0


        # if isinstance(DSO, Streamflow):
        #
        #     val = R["delta"] *  std_to_native_rate * avg_seconds_per_month
        #     unit = "m³"
        # else:
        #     val = R["delta"] *  std_to_native_rate
        #     unit = "mm"
        val[R['trend'] == 0] = 0
        line1,=ax[i].plot(DSO.CDN, '-k',label='CDN')
        ax[i].set_ylabel('CDN', fontsize=12)
        # ax[i].legend(loc=2)
        ax[i].set_xticks(np.arange(0, len(val), 12))
        ax[i].set_xticklabels(anni, rotation=90)

        ax2 = ax[i].twinx()
        line2= ax2.bar(np.arange(len(val)), val, color=colors[i],alpha=0.3, label=f'Trend {window} mesi')

        ax2.axhline(y=0,color='lightgrey')
        ax2.set_ylabel(f'Change [{unit}]', fontsize=12)
        ax2.set_xlim(36, len(val))

        # ----------------------------------------------------
        # ylim domain for ax2:
        ymax = np.nanmax(np.abs(val))
        n_levels = 11
        step = np.ceil(ymax / ((n_levels - 1) // 2))
        base = 10 ** np.floor(np.log10(step))
        for mult in [1, 2, 5, 10]:
            if step <= mult * base:
                step = mult * base
                break
        # make it symmetric
        ymax_rounded = int(np.ceil(ymax / step) * step)
        yticks = np.arange(-ymax_rounded, ymax_rounded + step, step)
        # ----------------------------------------------------

        ax2.set_yticks(yticks)
        ax2.set_ylim(yticks[0], yticks[-1])

        # combine the curves to work with a single label
        lines = [line1, line2[0]] #take only a proxy for the barharty
        labels = ['CDN' , f'trend over {window} months (moving window)']

        if show_spi and window<=DSO.K:
            ax3 = ax[i].twinx()
            ax3.spines["right"].set_position(("axes", 1.12))
            line3, = ax3.plot(DSO.spi_like_set[window - 1], '-',
                              color='dimgrey', alpha=0.7,
                              label=f'SPI{window})')
            ax3.set_ylabel(f'SPI{window}', fontsize=12, color='dimgrey')
            ax3.tick_params(axis='y', labelcolor='dimgrey')
            lines.append(line3)
            labels.append(f'SPI{window}')

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


    fig.suptitle(DSO.basin_name)
    fig.tight_layout()
    plt.show(block=False)



def plot_cdn_trends(DSO, windows, figsize=(14, 10), ax=None,
                    year_ext=None, unit=None, show_spi=False):
    """
      Plot the Cumulative Deviation from Normal (CDN) time series together with the
      cumulative water deficit/surplus estimated over one or more moving windows.

      For each window, the CDN curve is drawn on the left axis, while bars on the
      right axis show the cumulative deficit/surplus in physical units (millimetres
      for Precipitation, cubic metres for Streamflow). The deficit is computed by
      `DSO.deficit_from_spi(window)`, i.e. the SPI-like anomaly at the accumulation
      scale equal to the window length, converted back to native cumulative units
      via the exact inverse of the fitted distribution (`spi_to_native`), relative to
      the SPI=0 reference — not the deprecated `c2r_index` polynomial. Bars are set
      to zero wherever the SPI-like anomaly at the window scale falls within
      [-0.5, 0.5], so that a magnitude is reported only outside this near-neutral band.

      Args:
          DSO: DroughtScan-like object exposing the CDN series, the calendar
              `m_cal`, and the methods `find_trends(window=...)` and
              `deficit_from_spi(window=...)`.
          windows (list of int): Moving-window sizes, in months, over which to
              compute and display the deficit/surplus. One subplot is drawn per
              window.
          figsize (tuple, optional): Figure size, used only when `ax` is not given.
          ax (matplotlib Axes or array of Axes, optional): Target axes. If None, a
              new figure with one row per window is created.
          year_ext (tuple of (int, int), optional): Start and end years to restrict
              the x-axis range. If None, the full record is shown from `window`
              onwards.
          unit (str, optional): Retained for backward compatibility; the unit is now
              inferred automatically ("mm" for Precipitation, "m^3" for Streamflow).
          show_spi (bool, optional): If True, overlay the SPI-like series at the
              window scale on a third axis. The series is taken from
              `DSO.spi_like_set` when `window <= DSO.K`, or computed on the fly for
              larger scales. Gaps (NaN) from missing data in the input series appear
              as breaks in the SPI line.

      Returns:
          None. Displays the figure.
      """

    from drought_scan.core import Streamflow

    cmap = plt.get_cmap('Set1')
    colors = [cmap(i % cmap.N) for i in range(len(windows))]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, ncols=1, nrows=len(windows))
        ax = np.atleast_1d(ax).ravel()
    else:
        ax = [ax] if len(windows) == 1 else np.asarray(ax).ravel()
        fig = ax[0].figure

    anni = np.unique(DSO.m_cal[:, 1]).astype(int)
    if unit is None:
        unit = "m³" if isinstance(DSO, Streamflow) else "mm"



    for i, window in enumerate(windows):

        # R = DSO.find_trends(window=window)

        if window <= DSO.K:
            spi_w = DSO.spi_like_set[window - 1, :]
        else:
            spi_w, _ = DSO._compute_spi(month_scale=window)

        # Deficit/surplus
        anomaly = DSO.deficit_from_spi(window=window, spi=spi_w)

        val = anomaly.copy()
        val[(spi_w>-0.5) & (spi_w<0.5) ] = 0
        # val[R['trend'] == 0] = 0  # azzera dove non c'è trend significativo

        # --- CDN, asse sinistro ---
        line1, = ax[i].plot(DSO.CDN, '-k', label='CDN')
        ax[i].set_ylabel('CDN', fontsize=12)
        ax[i].set_xticks(np.arange(0, len(val), 12))
        ax[i].set_xticklabels(anni, rotation=90)

        # --- Volume anomaly, asse destro (barre) ---
        ax2 = ax[i].twinx()
        line2 = ax2.bar(np.arange(len(val)), val, color=colors[i],
                        alpha=0.3, label=f'water anomaly {window}m')
        ax2.axhline(0, color='lightgrey')
        ax2.set_ylabel(f'Water anomaly [{unit}]', fontsize=12)
        ax2.set_xlim(window, len(val))

        # ylim simmetrico (come tua versione)
        ymax = np.nanmax(np.abs(val))
        if ymax > 0:
            n_levels = 11
            step = np.ceil(ymax / ((n_levels - 1) // 2))
            base = 10 ** np.floor(np.log10(step))
            for mult in [1, 2, 5, 10]:
                if step <= mult * base:
                    step = mult * base
                    break
            ymax_rounded = np.ceil(ymax / step) * step
            yticks = np.arange(-ymax_rounded, ymax_rounded + step, step)
            ax2.set_yticks(yticks)
            ax2.set_ylim(yticks[0], yticks[-1])

        lines = [line1, line2[0]]
        labels = ['CDN', f'water anomaly over {window} months (sig. trend only)']

        if show_spi:
            ax3 = ax[i].twinx()
            ax3.spines["right"].set_position(("axes", 1.12))
            line3, = ax3.plot(spi_w, '-', color='dimgrey', alpha=0.7,
                              label=f'SPI{window}')
            ax3.set_ylabel(f'SPI{window}', fontsize=12, color='dimgrey')
            ax3.tick_params(axis='y', labelcolor='dimgrey')
            lines.append(line3)
            labels.append(f'SPI{window}')

        ax[i].legend(lines, labels, loc='lower left',fontsize=10,frameon=False)

        if year_ext is None:
            ax[i].set_xlim(windows[0], len(DSO.CDN))
        else:
            x1 = np.where(DSO.m_cal[:, 1] == year_ext[0])[0]
            x2 = np.where(DSO.m_cal[:, 1] == year_ext[1])[0]
            if len(x1) == 0:
                raise ValueError(
                    f"year_ext start outside domain: "
                    f"{int(DSO.m_cal[0, 1])}–{int(DSO.m_cal[-1, 1])}")
            x2 = len(DSO.CDN) if len(x2) == 0 else x2[-1]
            ax[i].set_xlim(x1[0], x2)

    fig.suptitle(DSO.basin_name)
    fig.tight_layout()
    plt.show(block=False)


def monthly_profile(DSO,var=None, var_name=None,cumulate=False, ax=None,highlight_years=None, season_shift=False):
    """
    Plot a monthly profile of a time series, with percentile bands and optional highlighted years.

    Parameters
    ----------
    DSO : DroughtScan-like object providing `DSO.ts` and `DSO.m_cal`.

    var : a DSO timeseries to be profiled. If None, `DSO.ts` will be used as default.
        Must be a 1D array with the same length as `DSO.m_cal`.

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
    if ax is None:
        plt.figure(figsize=(8, 6))
        ax=plt.gca()
    else:
        ax = ax
        fig = ax.figure

    if var is None:
        x = DSO.ts.copy()
    else:
        x = var

    if var_name is None:
        var_name = 'input variable'


    if len(x) != len(DSO.m_cal):
        raise ValueError("Input variable and m_cal must have the same length.")

    months = DSO.m_cal[:, 0]
    years = DSO.m_cal[:, 1]
    unique_years = np.unique(years)

    yb0 = int(DSO.start_baseline_year)
    yb1 = int(DSO.end_baseline_year)
    baseline_mask = (years >= yb0) & (years <= yb1)
    baseline_years = np.unique(years[baseline_mask])

    monthly_means = np.zeros(12)
    perc_25 = np.zeros(12)
    perc_75 = np.zeros(12)
    perc_10 = np.zeros(12)
    perc_90 = np.zeros(12)

    # The reference line: normal_values()'s per-month "normal" (the exact
    # inverse of SPI-like=0 at scale 1 via spi_to_native), NOT the raw
    # arithmetic mean — consistent with deficit_from_spi/volume_anomaly_rolling,
    # which both anchor to normal_values() as the SPI=0 reference. Only valid
    # when profiling DSO.ts itself (the series the distribution was fitted
    # to): an arbitrary `var` has no associated fit, so it falls back to the
    # plain baseline mean.
    using_normals = not cumulate and var is None
    mean_label = 'Monthly Normal (SPI=0, baseline)' if using_normals else 'Monthly Mean (baseline)'
    if using_normals:
        monthly_means = DSO._monthly_normals()

    if not cumulate:
        for month in range(1, 13):
            monthly_data = x[(months == month) & baseline_mask]
            if not using_normals:
                monthly_means[month - 1] = np.nanmean(monthly_data)
            perc_25[month - 1] = np.nanpercentile(monthly_data, 25)
            perc_75[month - 1] = np.nanpercentile(monthly_data, 75)
            perc_10[month - 1] = np.nanpercentile(monthly_data, 10)
            perc_90[month - 1] = np.nanpercentile(monthly_data, 90)
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
            values = [annual_cumsum[year][month] for year in baseline_years]
            monthly_means[month] = np.mean(values)
            perc_25[month] = np.percentile(values, 25)
            perc_75[month] = np.percentile(values, 75)
            perc_10[month] = np.percentile(values, 10)
            perc_90[month] = np.percentile(values, 90)

    # Extend the monthly stats to 24 months by repeating the cycle
    months_n = np.arange(1, 25) if season_shift else np.arange(1, 13)
    nyears = 2 if season_shift else 1
    mean_n = np.tile(monthly_means, nyears)
    x_ticks = np.tile(np.arange(1, 13), nyears)
    p25_24 = np.tile(perc_25, nyears)
    p75_24 = np.tile(perc_75, nyears)
    p10_24 = np.tile(perc_10, nyears)
    p90_24 = np.tile(perc_90, nyears)

    # Plotting
    # plt.figure(figsize=(12, 6))

    ax.plot(months_n, mean_n, color='darkgray', label=mean_label, linewidth=3)
    ax.fill_between(months_n, p25_24, p75_24, color='gray', alpha=0.5, label='25–75 Percentile')
    ax.fill_between(months_n, p10_24, p90_24, color='lightgray', alpha=0.5, label='10–90 Percentile')

    # Highlight years if specified
    if highlight_years is not None:
        if isinstance(highlight_years, int):
            highlight_years = [highlight_years]
        elif not isinstance(highlight_years, list):
            raise TypeError("highlight_years must be an int, a list of ints, or None.")

        colors = ['tab:orange', 'tab:cyan', 'tab:purple']
        for i, year in enumerate(highlight_years[:3]):
            if year in unique_years and (year - 1) in unique_years:
                # Build 24-month series from year-1 and year
                if cumulate:
                    data_prev = annual_cumsum[year - 1]
                    data_curr = annual_cumsum[year]
                else:
                    data_prev = [np.mean(x[(months == month) & (years == year - 1)]) for month in range(1, 13)]
                    data_curr = [np.mean(x[(months == month) & (years == year)]) for month in range(1, 13)]

                full_series = np.concatenate([data_prev, data_curr]) if season_shift else data_curr
                label = f'{year - 1}-{year}' if season_shift else f'{year}'
                ax.plot(months_n, full_series, color=colors[i], linewidth=2, label=label)
            else:
                print(f"Warning: Cannot plot N-months profile for year {year} (missing previous year).")


    ax.set_xticks(months_n)
    ax.set_xticklabels(x_ticks)
    ax.set_xlabel('Month')
    ax.set_ylabel(f'{var_name} Cumulative Value' if cumulate else f'{var_name} Mean Value')
    title = f'{DSO.basin_name}'
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    if season_shift:
        ax.set_xlim(8,19)
    else:
        ax.set_xlim(1,12)
    plt.tight_layout()
    plt.show(block=False)


#---------------------------------------------------
# optimal SIDI and SQI1 - diagnostic tool
# -------------------------------------------------
def plot__covariates(DSO, streamflow, weight_index, year_ext=None, split_plot=False):
    """
    Plot the covariate relationship between a Precipitation-based drought index (SIDI)
    and a Streamflow-based index (SQI1), highlighting overlapping drought signals and
    their difference (delta).

    Parameters
    ----------
    DSO : Precipitation
        An instance of Precipitation (subclass of BaseDroughtAnalysis) with an optimized
        SIDI index (requires that `DSO.set_optimal_SIDI()` has been been called before).
    streamflow : BaseDroughtAnalysis or Streamflow
        A Streamflow object (or subclass of BaseDroughtAnalysis) providing SQI1 or similar indices.
    weight_index : int or None
        Index (0–4) of the SIDI series to be compared with streamflow.
        If None, column 0 is used and the SIDI line is rendered in red
        (reserved for seasonally-optimized SIDI where all columns are identical).
    year_ext : tuple of int, optional
        (start_year, end_year) to restrict the x-axis.
    split_plot : bool, optional
        If True, produces two separate figures:
            (1) Optimal SIDI vs SQI1
            (2) Their difference (SQI1 - SIDI).
        If False (default), combines both in a single figure with two stacked panels.
        In both cases, an additional figure with a time series + scatter comparison
        is always rendered.

    Notes
    -----
    - Requires utility functions: `find_overlap`, `spi_cmap`, `highlight_drought`.
    - Assumes DSO and streamflow share at least some overlapping monthly timeline (`m_cal`).
    """

    # --- Resolve plot index and color ---
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:pink', 'tab:purple']
    _plot_index = 0 if weight_index is None else weight_index
    _color      = 'red' if weight_index is None else colors[weight_index]

    # --- Temporal overlap ---
    self_indices, streamflow_indices = find_overlap(DSO.m_cal, streamflow.m_cal)
    if len(self_indices) == 0 or len(streamflow_indices) == 0:
        raise ValueError("No overlapping data found between Precipitation and Streamflow.")

    # --- Subset to overlapping calendar ---
    m_cal = DSO.m_cal[self_indices, :]
    vec   = np.arange(len(m_cal))

    # --- Extract series ---
    x     = DSO.SIDI[:, _plot_index][self_indices]          # SIDI
    y     = streamflow.spi_like_set[0][streamflow_indices]  # SQI1
    delta = y - x                                           # Difference

    # --- Colormap for delta shading ---
    cmap   = spi_cmap().reversed() if DSO.threshold > 0 else spi_cmap()
    bounds = np.array([-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3])
    norm   = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # --- Year tick labels ---
    gen_id      = np.where(m_cal[:, 0] == 1)[0][0]
    year_labels = np.arange(m_cal[gen_id, 1], m_cal[-1, 1] + 1)
    year_ticks  = np.arange(gen_id, len(m_cal), 12)

    def _apply_xlim(ax):
        if year_ext is None:
            ax.set_xlim(0, len(x))
        else:
            try:
                x1 = np.where(m_cal[:, 1] == year_ext[0])[0][0]
                x2 = np.where(m_cal[:, 1] == year_ext[1])[0][-1]
            except IndexError:
                raise IndexError("provide a tuple of years within the actual time domain")
            ax.set_xlim(x1, x2)

    width = _figure_size_for_length(len(x))[0] / 4 * 3
    hight = _figure_size_for_length(len(x))[1] / 3

    # ================================================================
    # Figure 1: timeseries + delta
    # ================================================================
    if split_plot:
        # --- Panel 1: SIDI vs SQI1 ---
        fig, ax = plt.subplots(figsize=(width, hight))
        highlight_drought(ax, x, offset=0, threshold=DSO.threshold)
        ax.plot(vec, x, c=_color,    label=DSO.SIDI_name)
        ax.plot(vec, y, c='dimgrey', label='SQI1')
        ax.set_ylim(-4, 4)
        ax.legend()
        ax.set_title(f"{DSO.basin_name}: optimal {DSO.SIDI_name} and {streamflow.index_name}1 (covariates)")
        ax.set_xticks(year_ticks)
        ax.set_xticklabels(year_labels, rotation=90, fontweight='bold')
        _apply_xlim(ax)
        fig.tight_layout()

        # --- Panel 2: delta ---
        fig, ax = plt.subplots(figsize=(width, hight))
        highlight_drought(ax, x, offset=0, threshold=DSO.threshold)
        ax.plot(vec, delta, 'gray', linewidth=1, alpha=0.3)
        for j in range(len(delta) - 1):
            ax.fill_between([vec[j], vec[j + 1]],
                            [delta[j], delta[j + 1]],
                            color=cmap(norm(delta[j])), alpha=1)
        ax.set_title(f"{DSO.basin_name}: {streamflow.index_name}1 minus optimal {DSO.SIDI_name}")
        ax.set_xticks(year_ticks)
        ax.set_xticklabels(year_labels, rotation=90, fontweight='bold')
        ax.axhline(y=-1, linestyle='--', color='gray', alpha=0.7)
        ax.axhline(y=1,  linestyle='--', color='gray', alpha=0.7)
        _apply_xlim(ax)
        fig.tight_layout()

    else:
        # --- Single figure, two stacked panels ---
        fig, axes = plt.subplots(2, 1, figsize=(width, hight * 2), sharex=True)

        ax = axes[0]
        highlight_drought(ax, x, offset=0, threshold=DSO.threshold)
        ax.plot(vec, x, c=_color,    label=DSO.SIDI_name)
        ax.plot(vec, y, c='dimgrey', label=f'{streamflow.index_name}1')
        ax.set_ylim(-4, 4)
        ax.legend()
        ax.set_title(f"{DSO.basin_name}: optimal {DSO.SIDI_name} and {streamflow.index_name} (covariates)")
        ax.grid(axis='y')
        _apply_xlim(ax)

        ax = axes[1]
        highlight_drought(ax, x, offset=0, threshold=DSO.threshold)
        ax.plot(vec, delta, 'gray', linewidth=1, alpha=0.3)
        for j in range(len(delta) - 1):
            ax.fill_between([vec[j], vec[j + 1]],
                            [delta[j], delta[j + 1]],
                            color=cmap(norm(delta[j])), alpha=1)
        ax.set_ylim(-4, 4)
        ax.set_title(f"{DSO.basin_name}: {streamflow.index_name} minus optimal {DSO.SIDI_name}")
        ax.set_xticks(year_ticks)
        ax.set_xticklabels(year_labels, rotation=90, fontweight='bold')
        ax.grid(axis='y')
        _apply_xlim(ax)
        plt.tight_layout()

    plt.show()

    # ================================================================
    # Figure 2: timeseries + scatter
    # ================================================================
    scatter_size = hight
    assert width > scatter_size, "Figure width too small for scatter panel — increase series length."
    ts_width = width - scatter_size

    fig2, axes2 = plt.subplots(
        1, 2,
        figsize=(width, scatter_size),
        gridspec_kw={'width_ratios': [ts_width / scatter_size, 1]}
    )

    # Left: timeseries
    ax_ts = axes2[0]
    ax_ts.plot(vec, y, c='k',    label=f'{streamflow.index_name}1', linewidth=0.7)
    ax_ts.plot(vec, x, c=_color, label=DSO.SIDI_name,               linewidth=0.7)
    ax_ts.set_ylim(-4, 4)
    ax_ts.legend()
    ax_ts.set_title(f"{DSO.basin_name}: {DSO.SIDI_name} vs {streamflow.index_name}")
    ax_ts.set_xticks(year_ticks)
    ax_ts.set_xticklabels(year_labels, rotation=90, fontweight='bold')
    ax_ts.grid(axis='y')
    _apply_xlim(ax_ts)

    # Right: scatter
    ax_sc = axes2[1]
    ax_sc.scatter(x, y, c=_color, alpha=0.7, edgecolors='none', s=15)
    ax_sc.set_xlabel(DSO.SIDI_name)
    ax_sc.set_ylabel(streamflow.index_name)
    ax_sc.set_xlim(-4, 4)
    ax_sc.set_ylim(-4, 4)
    ax_sc.axhline(0, color='gray', linewidth=0.7, linestyle='--')
    ax_sc.axvline(0, color='gray', linewidth=0.7, linestyle='--')
    ax_sc.set_aspect('equal', adjustable='box')
    ax_sc.set_title("Scatter")
    ax_sc.grid(True, alpha=0.3)

    fig2.tight_layout()
    plt.show()