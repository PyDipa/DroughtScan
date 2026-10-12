import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
# plt.style.use("drought_scan/utils/default.mplstyle")
# mpl.rcParams['font.family'] = ['Helvetica', 'DejaVu Sans', 'Arial']

from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgb
from drought_scan.utils.drought_indices import *
from drought_scan.utils.statistics import find_overlap, WEIGHT_LABELS, WEIGHT_COLORS
from matplotlib.colors import Normalize


import os

try:
    import cmcrameri.cm as cmc
except Exception:
    cmc = None

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

def plot_overview(DSO, optimal_k=None, weight_index=None, year_ext=None,
                  split_plot=False, plot_order=None,figsize=None):
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
    labels = np.array([str(int(c[1])) for c in DSO.m_cal]).astype(int) # anni

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
        raise ValueError("plot_order must be a permutation of 'HSC', e.g. 'CHS', 'HSC', 'SCH', ...")

    order = list(plot_order)  # es. ['S','C','H']
    # # order check
    # if plot_order not in ('CHS', 'HSC'):
    #     raise ValueError("plot_order deve essere 'CHS' (default) o 'HSC'.")

    order = list(plot_order)



    # -------------------- PLOTTING --------------------
    if not split_plot:
        # unica figura con 3 pannelli nell'ordine scelto
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
    # Set the title and layout
    title = f'Drought Scan, severe events profile for {DSO.basin_name}. Baseline: {DSO.start_baseline_year} - {DSO.end_baseline_year}'
    fig.suptitle(title, fontsize=12)
    plt.show(block=False)


# Half-width, in SPI units, of the near-neutral band the bars are blanked in:
# "normal" gets a visible extent instead of a hairline at zero. Same convention as
# plot_spatial's own `neutral_band`, and the default of DSO.find_trends.
NEUTRAL_BAND = 0.5


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
              `m_cal`, and the methods
              `find_trends(window=...)` and `deficit_from_spi(window=...)`.
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
        # "m³"
        unit = "mm-eq" if isinstance(DSO, Streamflow) else "mm"



    for i, window in enumerate(windows):

        # R = DSO.find_trends(window=window)

        if window <= DSO.K:
            spi_w = DSO.spi_like_set[window - 1, :]
        else:
            spi_w, _ = DSO._compute_spi(month_scale=window)

        # Deficit/surplus
        anomaly = DSO.deficit_from_spi(window=window, spi=spi_w)
        if isinstance(DSO, Streamflow):
            anomaly = anomaly / (DSO.area_kmq * 1e6) * 1000

        val = anomaly.copy()
        val[(spi_w > -0.5) & (spi_w < 0.5)] = 0

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
        labels = ['CDN',
                  f'water anomaly over {window} months '
                  f'(|{DSO.index_name}{window}| \u2265 {NEUTRAL_BAND})']

        if show_spi:
            ax3 = ax[i].twinx()
            ax3.spines["right"].set_position(("axes", 1.12))
            line3, = ax3.plot(spi_w, '-', color='dimgrey', alpha=0.7,
                              label=f'{DSO.index_name}{window}')
            ax3.set_ylabel(f'{DSO.index_name}{window}', fontsize=12, color='dimgrey')
            ax3.tick_params(axis='y', labelcolor='dimgrey')
            lines.append(line3)
            labels.append(f'{DSO.index_name}{window}')

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
def plot__covariates(DSO, streamflow, weight_index, year_ext=None, split_plot=False,
                     ax=None):
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
    ax : matplotlib.axes.Axes, optional
        Only valid together with ``weight_index='all'``. When given, the
        5-scheme time-series overlay (normally the first of the three 'all'
        figures) is drawn into this axis instead of its own figure — nothing
        is shown or closed, and the other two 'all' figures (the per-season
        errorbar grid and the per-class median-error line chart) are skipped.
        Use this to build a composite figure — e.g. one row per basin — by
        looping the call over a grid of axes you own; nothing is returned.

    Notes
    -----
    - Requires utility functions: `find_overlap`, `spi_cmap`, `highlight_drought`.
    - Assumes DSO and streamflow share at least some overlapping monthly timeline (`m_cal`).
    - In the ``weight_index='all'`` time-series overlay the shaded bands mark the
      months the observed streamflow was in drought (SQI1 < ``streamflow.threshold``,
      -1 by default). This is only meaningful because the panel is fed a real
      streamflow object and an optimized SIDI (``DSO.set_optimal_SIDI()`` already
      called): the shading documents when the basin actually ran dry, against which
      the five SIDI schemes on the same axis can be read.
    """

    # --- Resolve plot index and color ---
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:pink', 'tab:purple']
    wlabels = list(WEIGHT_LABELS)
    multi = isinstance(weight_index, str) and weight_index.lower() == 'all'
    if ax is not None and not multi:
        raise ValueError(
            "ax is only supported together with weight_index='all' (it draws "
            "the 5-scheme time-series overlay into that axis)."
        )
    _plot_index = 0 if weight_index is None else (0 if multi else weight_index)
    _color      = 'orange' if weight_index is None else ('k' if multi else colors[weight_index])

    # --- Temporal overlap ---
    self_indices, streamflow_indices = find_overlap(DSO.m_cal, streamflow.m_cal)
    if len(self_indices) == 0 or len(streamflow_indices) == 0:
        raise ValueError("No overlapping data found between Precipitation and Streamflow.")

    # --- Subset to overlapping calendar ---
    m_cal = DSO.m_cal[self_indices, :]
    vec   = np.arange(len(m_cal))

    # --- Extract series ---
    x     = DSO.SIDI[:, _plot_index][self_indices]          # SIDI (or column 0 when multi)
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
            ax.set_xlim(0, len(m_cal))
        else:
            if (not isinstance(year_ext, (tuple, list)) or len(year_ext) != 2
                    or not all(isinstance(v, (int, np.integer)) for v in year_ext)):
                # The likely mistake: weight_index='all' passed positionally,
                # landing in year_ext (the actual 2nd positional parameter).
                raise TypeError(
                    "year_ext must be a (start_year, end_year) tuple of int, e.g. "
                    f"(2000, 2010); got {year_ext!r}. If you meant weight_index="
                    f"'all', pass it as a keyword: plot_covariates(streamflow, "
                    f"weight_index='all')."
                )
            try:
                x1 = np.where(m_cal[:, 1] == year_ext[0])[0][0]
                x2 = np.where(m_cal[:, 1] == year_ext[1])[0][-1]
            except IndexError:
                raise IndexError("provide a tuple of years within the actual time domain")
            ax.set_xlim(x1, x2)

    width = _figure_size_for_length(len(m_cal))[0] / 4 * 3
    hight = _figure_size_for_length(len(m_cal))[1] / 3

    # ================================================================
    # weight_index='all': (1) time-series overlay of the 5 SIDI schemes +
    # SQI1; (2) a 2x3 grid of SIDI vs SQI1 (grey monthly points) with
    # per-season median / 5th-95th pct errorbars binned by SQI1 interval;
    # (3) a line chart of the median |SQI1 - SIDI| per SQI1 class, one line
    # per scheme. The delta / colormap panel is single-series, skipped here.
    # ================================================================
    if multi:
        X = DSO.SIDI[self_indices, :]                       # (T_overlap, 5)
        embedded = ax is not None
        if embedded:
            ax_ts = ax
        else:
            fig, ax_ts = plt.subplots(1, 1, figsize=(width, hight))

        # Short displayed span (<= 20 years, after year_ext) -> thicker lines and
        # a larger legend so the overlay stays readable on a narrow x-axis. The
        # legend text is bold regardless of the span.
        shown_months = (len(m_cal) if year_ext is None
                        else (int(year_ext[1]) - int(year_ext[0]) + 1) * 12)
        short_span = shown_months <= 20 * 12
        lw_sqi = 1.8 if short_span else 0.9
        lw_sidi = 1.4 if short_span else 0.7
        leg_fs = 11 if short_span else 8

        # Shade the months the observed streamflow was itself in drought
        # (SQI1 < streamflow.threshold, -1 by default): it documents, on the
        # same axis as the optimized SIDI schemes, when the basin actually
        # ran dry, so the overlay can be read as "did any scheme see it coming".
        highlight_drought(ax_ts, y, offset=0, threshold=streamflow.threshold)

        ax_ts.plot(vec, y, c='k', label=f'{streamflow.index_name}1', linewidth=lw_sqi)
        for w in range(X.shape[1]):
            ax_ts.plot(vec, X[:, w], c=colors[w], label=wlabels[w],
                       linewidth=lw_sidi, alpha=0.9)
        ax_ts.set_ylim(-4, 4)
        ax_ts.legend(ncol=2, prop={'weight': 'bold', 'size': leg_fs})
        ax_ts.set_title(f"{DSO.basin_name}: {DSO.SIDI_name} (5 schemes) vs "
                        f"{streamflow.index_name}1")
        ax_ts.set_xticks(year_ticks)
        ax_ts.set_xticklabels(year_labels, rotation=90, fontweight='bold')
        ax_ts.grid(axis='y')
        _apply_xlim(ax_ts)

        if embedded:
            # Caller owns this axis and its figure — draw into it and stop:
            # figures 2 and 3 ('all' mode's errorbar grid and line chart)
            # only make sense as their own standalone figures.
            return
        fig.tight_layout()
        plt.show()

        # --- month -> season mapping for the per-scheme scatter below -----
        # Prefer the committed seasonal calibration; fall back to the four
        # meteorological seasons when the SIDI was not optimized seasonally.
        months_ov = m_cal[:, 0].astype(int)
        if getattr(DSO, 'is_seasonal_sidi', False):
            season_map = DSO.seasonal_params['seasons']
        else:
            season_map = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5],
                          'JJA': [6, 7, 8], 'SON': [9, 10, 11]}
        s_names = list(season_map)
        # Okabe-Ito qualitative palette (colour-vision-deficiency safe). Colour
        # legibility still needs work (Arianna 2026-08-31: "ci lavoreremo").
        _OKABE_ITO = ['#0072B2', '#E69F00', '#009E73', '#D55E00',
                      '#CC79A7', '#56B4E9', '#F0E442', '#000000']
        s_colors = [_OKABE_ITO[i % len(_OKABE_ITO)] for i in range(len(s_names))]

        # SQI1 classes: open-ended below -2 and above 2, unit-wide in between,
        # except one single central class spanning -1..1 (no split at 0).
        sqi_edges = [-np.inf, -2.0, -1.0, 1.0, 2.0, np.inf]

        # --- Figure: per-scheme SIDI vs SQI1, grey cloud + per-season
        #     5th-95th percentile crosses binned by integer SQI1 class ------
        fig3, axes3 = plt.subplots(2, 3, figsize=(min(width, 13.0), 8.5))
        axes3 = axes3.ravel()
        axis_ticks = np.arange(-4, 5, 1)
        for w in range(X.shape[1]):
            ax = axes3[w]
            xf, yf = X[:, w], y
            fin = np.isfinite(xf) & np.isfinite(yf)
            ax.scatter(xf[fin], yf[fin], s=10, color='0.6', alpha=0.30,
                       edgecolors='none', zorder=1)
            # R2 of this scheme's reconstructed SIDI (its own seasonally-
            # optimized K per season, already baked into X = DSO.SIDI) vs SQI1,
            # over the whole overlap - same finite mask as the scatter/crosses.
            if fin.sum() >= 2 and np.std(xf[fin]) > 0 and np.std(yf[fin]) > 0:
                r2 = np.corrcoef(xf[fin], yf[fin])[0, 1] ** 2
                ax.text(0.03, 0.97, f"R²={r2:.2f}", transform=ax.transAxes,
                        ha='left', va='top', fontsize=10, fontweight='bold',
                        zorder=6, bbox=dict(boxstyle='round', facecolor='white',
                                           edgecolor='none', alpha=0.7))
            for si, sname in enumerate(s_names):
                m_sea = np.isin(months_ov, season_map[sname]) & fin
                for lo, hi in zip(sqi_edges[:-1], sqi_edges[1:]):
                    cell = m_sea & (yf > lo) & (yf <= hi)
                    if not np.any(cell):
                        continue
                    xs, ys = xf[cell], yf[cell]
                    xm, ym = np.median(xs), np.median(ys)
                    # 5th-95th percentiles: trims the single most extreme point
                    # per tail so one outlier month doesn't stretch the whisker.
                    x0, x100 = np.percentile(xs, [5, 95])
                    y0, y100 = np.percentile(ys, [5, 95])
                    ax.errorbar(xm, ym,
                                xerr=[[xm - x0], [x100 - xm]],
                                yerr=[[ym - y0], [y100 - ym]],
                                fmt='o', ms=6, color=s_colors[si],
                                markeredgecolor='black', markeredgewidth=1.0,
                                ecolor=s_colors[si], elinewidth=2,
                                capsize=3, capthick=2, alpha=0.95, zorder=3,
                                label=sname)
            ax.plot([-4, 4], [-4, 4], color='k', ls='--', lw=0.8, alpha=0.7)
            ax.set_xlim(-4.5, 4.5)
            ax.set_ylim(-4.5, 4.5)
            ax.set_xticks(axis_ticks)
            ax.set_yticks(axis_ticks)
            # every panel keeps its own tick labels and axis labels: with an
            # empty 6th panel, shared axes would blank the x labels of the
            # panels above it and the y labels of the middle column.
            ax.tick_params(labelbottom=True, labelleft=True)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(wlabels[w], fontsize=10)
            ax.set_xlabel(DSO.SIDI_name)
            ax.set_ylabel(f"{streamflow.index_name}1")
            ax.grid(True, alpha=0.3)
        axes3[5].axis('off')
        _handles = [plt.Line2D([], [], marker='o', ms=6, ls='none',
                               color='0.6', label='monthly values')]
        _handles += [plt.Line2D([], [], marker='o', ms=6, ls='none',
                                color=s_colors[i], label=n)
                     for i, n in enumerate(s_names)]
        axes3[5].legend(handles=_handles, loc='upper left', frameon=False)
        fig3.suptitle(f"{DSO.basin_name}: {DSO.SIDI_name} vs {streamflow.index_name}1 "
                      f"— errorbars (median, 5th-95th pct) by {streamflow.index_name}1 interval")
        fig3.tight_layout()
        plt.show()

        # --- Figure: one glance at which scheme misses which intensity class -
        # x = 5 SQI1 classes, y = median |SQI1 - SIDI| in that class, one
        # broken line per weighting scheme. A scheme that systematically fails
        # on a class shows a clear bump there.
        cls_edges = [-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf]
        cls_lbl = ["≤−1.5", "−1.5…−0.5",
                   "−0.5…0.5", "0.5…1.5", ">1.5"]
        xc = np.arange(len(cls_lbl))
        fig4, ax4 = plt.subplots(figsize=(min(width, 10.0), 4.6))
        for w in range(X.shape[1]):
            ae = np.abs(y - X[:, w])
            med = []
            for lo, hi in zip(cls_edges[:-1], cls_edges[1:]):
                m = np.isfinite(ae) & (y > lo) & (y <= hi)
                med.append(np.median(ae[m]) if np.any(m) else np.nan)
            ax4.plot(xc, med, marker='o', ms=6, lw=1.8, color=colors[w],
                     markeredgecolor='black', markeredgewidth=0.6,
                     label=wlabels[w])
        ax4.set_xticks(xc)
        ax4.set_xticklabels(cls_lbl)
        ax4.set_xlabel(f"{streamflow.index_name}1 class")
        ax4.set_ylabel(f"median |{streamflow.index_name}1 − {DSO.SIDI_name}|")
        ax4.set_ylim(bottom=0)
        ax4.grid(alpha=0.3)
        ax4.legend(fontsize=9, ncol=2)
        ax4.set_title(f"{DSO.basin_name}: median error by {streamflow.index_name}1 "
                      f"class, per weighting scheme")
        fig4.tight_layout()
        plt.show()

        # NOTE: the per-scheme anomaly (SQI1 - SIDI) histogram + KDE grid that
        # used to be here was replaced by the season-coloured scatter above
        # (Arianna 2026-08-31). It is preserved in git history at commit
        # 9029ba6 if it is ever wanted back.
        return

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
    # highlight_drought(ax_ts, x, offset=0, threshold=DSO.threshold)
    ax_ts.axhline(y=DSO.threshold, linestyle='--', color='red')

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

#--------------------------------------------------------------------
# function and settings for plotting FORECASTS
# -------------------------------------------------------------------

def add_esm_forecast(ax, forecast_data,percentile,t0,color_mean="#4A001F"):
    """Plot forecast median line and percentile bands."""

    fmean = np.percentile(forecast_data,percentile, axis=0)
    p = {k: np.percentile(forecast_data, k, axis=0) for k in [1, 10, 25, 75, 90, 99]}
    x = np.arange(len(fmean))

    # ax.plot(x[-window-1::], fmean[-window-1 ::], color_mean, linewidth=4,
    #         label='Forecast (Ens. Median)', alpha=0.7)

    ax.plot(x[t0::], fmean[t0::], color_mean, linewidth=4,
            label='Forecast (Ens. Median)', alpha=0.7)
    ax.fill_between(x, p[25], p[75], color=color_mean, linewidth=0, alpha=0.2,
                    label='25°–75° percentile range')
    # 'mediumorchid' 0.2; 'violet' 0.1
    ax.fill_between(x, p[10], p[90], color=color_mean, linewidth=0, alpha=0.1,
                    label='10°–90° percentile range')
    # ax.fill_between(x, p[1], p[99], color="darkslateblue", linewidth=0, alpha=0.1, label='1–99 percentile range')

def add_ml_forecast(ax, prediction,rmse,t0,window, color_mean="#4A001F"):
    """Plot forecast median line and percentile bands."""
    x = np.arange(t0, t0 + 1 + window)
    ax.plot(x, prediction, color_mean, linewidth=4,
            label='ML Forecast (F)             ', alpha=0.7)

    # shaodows
    ax.fill_between(x,prediction - rmse,  prediction + rmse, color=color_mean,
                     label='F ± 1 RMSE                  ', alpha=0.2)
    ax.fill_between(x,prediction - 2 * rmse, prediction + 2 * rmse, color=color_mean,
                     label='F ± 2 RMSE                  ', alpha=0.1)


     # ax.fill_between(x, p[1], p[99], color="darkslateblue", linewidth=0, alpha=0.1, label='1–99 percentile range')

def add_whatif_curves(ax, x, curves, t0,t2, factors, Probability, month):
    letters = [r'( $\mathit{g}$ )', r'( $\mathit{f}$ )', r'( $\mathit{e}$ )', r'( $\mathit{d}$ )',
               r'( $\mathit{c}$ )', r'( $\mathit{b}$ )', r'( $\mathit{a}$ )']
    for ii, factor in enumerate(factors):
        # print(ii,factors)
        # width = ((np.log1p(Probability[ii, month - 1]) ** 2) / 3) ** 1.1
        width = ((np.log1p(Probability[ii]) ** 2) / 3) ** 1.1
        if width<=0.05:
            width=0.5
        # print(width)
        prob_text = ('≤' if ii < 3 else '≥')
        color = ("darkgoldenrod" if ii < 3 else "lightseagreen")

        if ii == len(factors):
            ax.plot(x[t0::], np.zeros((len(x[t0::]),)) * np.nan, color='w', alpha=0, label='    ')

        ax.plot(x[t0::], curves[ii, t0::], c=color, linewidth=width,
                linestyle='--', alpha=1,
                label=f'{letters[ii]} {factor} x NP ({prob_text}  {int(Probability[ii])}%)')
                # label=f'{letters[ii]} {factor} x NP ({prob_text} {int(Probability[ii, month - 1])}%)   ')

        ax.text(x[t2] - 0.45, curves[ii, t2] + 0.28, letters[ii], fontsize=11)

def add_legends(fig, ax, upper_title, bottom_title,only_WI=False,index=None):
    """Place two legends below the figure."""
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[::-1]
    labels = labels[::-1]
    if only_WI==False:
        group2_handles, group2_labels = handles[4:], labels[4:]
        group1_handles, group1_labels = handles[:4][::-1], labels[:4][::-1]

        if index == 'CDN':
            fig.legend(group1_handles, group1_labels, loc='upper right', bbox_to_anchor=(0.99, 0.865),
                       title=upper_title, ncol=2, fontsize=11, title_fontsize=12, labelspacing=1.05)
            fig.legend(group2_handles, group2_labels, loc='upper right', bbox_to_anchor=(0.988, 0.735),
                       title=bottom_title, ncol=1, fontsize=11, title_fontsize=12, alignment='left' ,labelspacing=1.05)

        else:
            fig.legend(group1_handles, group1_labels, loc='upper right', bbox_to_anchor=(0.99, 0.85),
                       title=upper_title, ncol=2, fontsize=11, title_fontsize=12, labelspacing=1.05)
            fig.legend(group2_handles, group2_labels, loc='upper right', bbox_to_anchor=(0.988, 0.73),
                       title=bottom_title, ncol=1, fontsize=11, title_fontsize=12, alignment='left' ,labelspacing=1.05)

    else:
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.85),
                   title=bottom_title, ncol=1, fontsize=11, title_fontsize=12, alignment='left' ,labelspacing=1.05)

def compute_n_dash(window, pad_left=5, n_left_ref=50, n_right_ref=64):
    """
    Number of '·' dots for the two banner segments, proportional to their width (months).
    - pad_left: months of the observed segment
    - window:   months of the forecast segment (1..7)
    - n_left_ref / n_right_ref: dots measured in the reference case
    - window_ref: forecast window of the reference case (e.g. 6)

    Returns (n_left, n_right) as integers.
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
        text_label = 'Relative contribution (%) to the total SHAP of the lead'
        title_mode = 'Relative contribution per lead (%)'
    else:
        raise ValueError('mode must be "classic" or "relative"')

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
    plt.show(block=False)

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


# ==============================================================================
# Bootstrap R²(K, weighting scheme) plot helpers
#
# Shared low-level drawing routines for the peak/cluster and benchmark-CI panels
# produced by BaseDroughtAnalysis.analyze_correlation[_seasonal] and the
# benchmark_nash bootstrap (core.py). Not meant to be called directly by users.
# ==============================================================================

def _plot3_peak_clusters(ax, MatCorr, K_range, summary, title="", legend=True):
    """The peak/cluster figure of analyze_correlation[_seasonal]: the per-scheme R²(K) curves
    (dotted, to keep them out of the way of the crosses) and, at each scheme's
    peak, a CROSS (x = peak K with its bootstrap CI, y = peak R² with its CI)
    in a slightly darker shade of that scheme's own curve colour. Every cross
    gets a centre marker; which symbol is incidental - the CI mutual-inclusion
    rule (``_peak_summary``: same K cluster, then same R² sub-cluster) decides
    only whether two schemes end up sharing one (same response group) or each
    keep their own (independent group of one)."""
    # One marker symbol per response sub-cluster - sized 1 or more - cycled so
    # distinct groups are told apart without redrawing the boxes this figure
    # used to carry. Every family belongs to exactly one sub-cluster, so this
    # covers every scheme, not just the ones sharing a group.
    _GROUP_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
    group_marker = {}
    gi = 0
    for c in summary.get("clusters", []):
        for su in c.get("subclusters", []):
            sym = _GROUP_MARKERS[gi % len(_GROUP_MARKERS)]
            gi += 1
            for f in su.get("families", []):
                group_marker[f] = sym

    def _darken(color, factor=1):
        r, g, b = to_rgb(color)
        return (r * factor, g * factor, b * factor)

    pbf = summary.get("peak_by_family", {})
    xk = np.arange(1, len(K_range) + 1)
    for wi, name in enumerate(pbf):
        color = WEIGHT_COLORS[wi % len(WEIGHT_COLORS)]
        ax.plot(xk, MatCorr[:, wi], linewidth=1.6, linestyle="--", label=name, color=color)
        d = pbf[name]
        pr, kk = d.get("peak_R2"), d.get("argmax_K")
        if kk and pr is not None and np.isfinite(pr):
            clo, chi = d["peak_CI"]
            klo, khi = d.get("K_CI", (np.nan, np.nan))
            yerr = [[max(0.0, pr - clo)], [max(0.0, chi - pr)]]
            xerr = ([[max(0.0, kk - klo)], [max(0.0, khi - kk)]]
                    if np.all(np.isfinite([klo, khi])) else None)
            marker = group_marker.get(name, "o")
            ecolor = _darken(color)
            ax.errorbar(kk, pr, yerr=yerr, xerr=xerr, fmt=marker, ms=7,
                        markerfacecolor=ecolor, markeredgecolor="black", markeredgewidth=2,
                        ecolor=ecolor, elinewidth=4, capsize=3, zorder=5)

    ax.set_xlim(1, len(K_range))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Month-scale (K)", fontsize=13)
    ax.set_ylabel(r"$R^2$", fontsize=13)
    ax.grid(alpha=0.3)
    if legend:
        ax.legend(fontsize=9, loc="lower right")
    if title:
        ax.set_title(title, fontsize=13)


def _plot_benchmark_ci_box(ax, ci, k_hat, r2_hat, color="#c1121f"):
    """Overlay the block-bootstrap CI of ``(optimal_K, R²)`` on a benchmark's
    R²_adj(K) panel: a translucent rectangle spanning the K-CI on x (lag months)
    and the R²-CI on y, plus an errorbar cross at the point estimate. Same visual
    language as the peak/cluster boxes of ``analyze_correlation[_seasonal]``
    (see ``_plot3_peak_clusters``). ``ci`` is a benchmark result's ``'ci'`` dict;
    a no-op when it lacks finite ``optimal_K`` / ``R2_adj_opt`` intervals."""
    from matplotlib.patches import Rectangle
    from matplotlib.colors import to_rgba

    klo, khi = ci.get("optimal_K", (np.nan, np.nan))
    rlo, rhi = ci.get("R2_adj_opt", (np.nan, np.nan))
    if np.all(np.isfinite([klo, khi, rlo, rhi])):
        ax.add_patch(Rectangle(
            (klo, rlo), max(khi - klo, 0.4), max(rhi - rlo, 1e-3),
            facecolor=to_rgba(color, 0.12), edgecolor=to_rgba(color, 0.55),
            linewidth=0.8, zorder=0, label="bootstrap 95% CI"))
        for xv in (klo, khi):
            ax.plot([xv, xv], [0, rhi], color="0.8", ls=":", lw=0.8, zorder=0.5)
    xerr = ([[max(0.0, k_hat - klo)], [max(0.0, khi - k_hat)]]
            if np.all(np.isfinite([klo, khi])) else None)
    yerr = ([[max(0.0, r2_hat - rlo)], [max(0.0, rhi - r2_hat)]]
            if np.all(np.isfinite([rlo, rhi])) else None)
    ax.errorbar(k_hat, r2_hat, xerr=xerr, yerr=yerr, fmt='o', ms=5, color=color,
                ecolor=color, elinewidth=1.1, capsize=3, zorder=5)



