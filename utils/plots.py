import math
import numpy as np
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix
from statsmodels.graphics.tsaplots import plot_acf
from cycler import cycler

import matplotlib.dates as mdates
import matplotlib.transforms as mtransforms

colors = [
    '#0A7EF2',
    '#FF6B6B',
    '#FFA94D',
    "#7D52B8",
    "#63BD7E",
    "#53D4E6",
    "#192D86",
    '#FF9F1C' 
]

plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
plt.rcParams["figure.dpi"] = 150

def shade_signal(ax, signal_series):
    """Shade regions of the plot where the signal_series equals 1."""
    in_signal = (signal_series == 1)
    if in_signal.any():
        groups = (in_signal != in_signal.shift()).cumsum()
        for _, group in in_signal.groupby(groups):
            if group.iloc[0]:
                start, end = group.index.min(), group.index.max()
                ax.axvspan(start, end, color='#c7c7c7', alpha=0.3)

def MSM_triple_plot(states_returns, smoothed_probs, state_assignments, returns):
    """Plot the excess returns, bull regime probability, and cumulative return in a single figure."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax1, ax2, ax3 = axes

    # 1. Excess Returns Plot
    for col in states_returns.columns:
        ax1.plot(states_returns.index, states_returns[col], label=col)
    shade_signal(ax1, state_assignments)
    ax1.legend(title="")
    ax1.set_title('Monthly Excess Returns', fontweight='bold')  
    ax1.set_ylabel('Excess Returns')
    ax1.set_ylim(-0.2, 0.2)
    ax1.set_yticks(np.arange(-0.2, 0.21, 0.1))
    ax1.margins(0)
    ax1.tick_params(labelbottom=True)

    # 2. Bull Regime Probability Plot
    ax2.plot(smoothed_probs.index, smoothed_probs[0])
    shade_signal(ax2, state_assignments)
    ax2.set_ylim(0, 1)
    ax2.set_title('Bull Regime probability', fontweight='bold')
    ax2.set_ylabel('$P(S_t=0)$')
    ax2.margins(0)
    ax2.tick_params(labelbottom=True)

    # 3. Cumulative Return (log scale) Plot
    cum_returns = returns.add(1).cumprod()
    ax3.plot(cum_returns.index, cum_returns, linewidth=0.9)
    shade_signal(ax3, state_assignments)
    ax3.set_yscale('log')
    ax3.set_title('Cumulative return (log scale)', fontweight='bold')
    ax3.set_ylabel('Cumulative return (log scale)')
    ax3.margins(0)
    ax3.tick_params(labelbottom=True)

    plt.subplots_adjust(hspace=3)
    plt.xticks(rotation=0)
    plt.tight_layout(pad=3, h_pad=3)
    plt.show()
    

def plot_sharpes(sharpes, reference):
    """
    Plots a list of Pandas Series as bar plots with a horizontal dotted reference line.
    Each subplot is titled using the Series name.

    Parameters:
    sharpes (list of pd.Series): List containing Pandas Series for each subplot.
    reference (float): The reference value for the horizontal dotted line.
    """
    n = len(sharpes)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, 8))
    
    if n == 1:
        axes = [axes]
    
    for ax, series in zip(axes, sharpes):
        series.plot(kind='bar', ax=ax, width=0.6, edgecolor='black', linewidth=0.5)

        if series.name:
            ax.set_title(series.name, fontname='Times New Roman')
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.axhline(
            y = reference,
            color='black',
            linestyle='dotted',
            linewidth=1.25,
            label=f'Buy-and-hold sharpe: {reference:.2f}')
        
        ax.set_ylim(0, 0.69)
        ax.tick_params(axis='both', direction='in', top=True, right=True, left=True, bottom=False, width=0.5)
        ax.set_ylabel("Sharpe ratio", fontname='Times New Roman')
        ax.set_xlabel("Number of months in TSM signal (n)", fontname='Times New Roman')
        
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(left=0.1, right=0.9)
    plt.show()

def plot_msm_ac(returns, msm_ar):
    """
    Plots the autocorrelation of returns alongside the autocorrelation under a Markov Switching Model (MSM).
    Parameters
    ----------
    returns : pandas.Series or array-like
        The time series of returns for which the sample autocorrelation will be computed and plotted.
    msm_ar : pandas.Series or array-like
        The autocorrelation values under the MSM model, indexed by lag.
    Notes
    -----
    - The function plots both the MSM autocorrelation (scaled by 1/10) and the sample autocorrelation of the returns.
    - Confidence bands at ±2/√n are shown, where n is the length of `returns`.
    - The plot is customized for clarity, with labeled axes, legend, and specific tick marks.
    """

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(msm_ar.index, msm_ar/10, marker='s', color='#ffb300', linewidth=0.75, markersize=5,label='Autocorrelation under MSM ($\\rho_k$)')
    plot_acf(returns, lags=20, zero=False, ax=ax, alpha=1, title="", linewidth=0.5)

    n = len(returns)
    conf = 2 / np.sqrt(n)

    ax.axhline(y=conf, color="gray", linestyle="dotted", linewidth=1)
    ax.axhline(y=-conf, color="gray", linestyle="dotted", linewidth=1)
    ax.axhline(y=0, color="black", linewidth=1)
    ylim = 0.11
    ax.set_ylim(-ylim, ylim)
    ax.set_yticks(np.arange(-0.1, 0.11, 0.05))
    ax.set_xticks(range(0, 21, 2))
    ax.set_facecolor("white")
    ax.tick_params( axis='both', which='both', direction='in', top=True, right=True)
    ax.set_ylabel('Autocorrelation', fontsize=12)
    ax.set_xlabel('Number of months ($k$)', fontsize=12)
    ax.legend(["Autocorrelation under MSM ($\\rho_k$)", "Sample autocorrelation of excess returns"])
    plt.tight_layout()
    plt.show()


def plot_misclas(misclassifications: pd.DataFrame):
    """
    Plots the misclassification rate over a range of months using a line plot.
    Parameters
    ----------
    misclassifications : pd.DataFrame
        A DataFrame where the index represents the number of months in the TSM signal (n),
        and the values represent the misclassification rates (between 0 and 1) for each n.
    Notes
    -----
    - The x-axis shows the number of months in the TSM signal.
    - The y-axis shows the misclassification rate.
    - X-ticks are set at every other index for readability.
    - The plot is displayed with customized tick parameters and margins.
    """
    ax = misclassifications.plot(figsize=(14, 5), ylim=(0, 1), fontsize=12)
    ax.tick_params(
        axis='both',
        which='major',
        direction='in',
        length=7,
        width=0.5,
        top=True,
        bottom=True,
        left=True,
        right=True
    )

    ax.margins(x=0)
    ax.set_ylabel('Misclassification rate')
    ax.set_xlabel('Number of months in TSM signal (n)')
    tick_positions = np.arange(1, len(misclassifications.index), 2)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(misclassifications.index[tick_positions], fontsize=10)

    plt.show()

def scatter_plot_misclas(misclassifications):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(misclassifications['False Positive'], misclassifications['False Negative'], s=50)

    for label, x, y in zip(misclassifications.index, misclassifications['False Positive'], misclassifications['False Negative']):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), ha='center')

    ax.tick_params(
        axis='both',
        which='major',
        direction='in',
        length=7,
        width=0.5,
        top=False,
        bottom=True,
        left=True,
        right=False
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('False Negative Rate')
    ax.set_ylim(0, 0.4)
    ax.set_xlim(0.35, 0.65)
    ax.set_yticks(np.arange(0, 0.41, 0.1))
    ax.set_yticklabels([f"{int(y)}" if y.is_integer() else f"{y:.1f}" for y in ax.get_yticks()])

    plt.show()

def msm_cumulative_returns_plot(strats_rets, regime_assignments):
    """
    Plot cumulative returns of strategies with regime shading.
    """
    cumprod_data = strats_rets.add(1).cumprod()

    fig, ax = plt.subplots(figsize=(10, 8))
    cumprod_data.plot(
        ax=ax,
        fontsize=12,
        linewidth=0.75,
        markersize=5,
        logy=True
    )

    ax.set_ylim(1, 1e5)
    ax.tick_params(
        axis='both',
        which='major',
        direction='in',
        length=7,
        width=0.5,
        top=True,
        right=True
    )

    ax.tick_params(
        axis='y',
        which='minor',
        direction='in',
        length=4,
        width=0.4,
        left=True,
        right=True
    )

    ax.xaxis.set_minor_locator(plt.NullLocator())
    min_year = cumprod_data.index.min().year
    max_year = cumprod_data.index.max().year
    start_tick_year = ((min_year + 9) // 10) * 10
    end_tick_year = (max_year // 10) * 10

    tick_years = np.arange(start_tick_year, end_tick_year + 1, 10)
    tick_positions = [pd.Timestamp(f"{year}-01-01") for year in tick_years]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(year) for year in tick_years], fontsize=10)
    ax.set_ylabel("Cumulative return (log scale)")

    zero_bool = (regime_assignments == 1).values
    start = None
    for i, is_one in enumerate(zero_bool):
        if is_one and start is None:
            start = cumprod_data.index[i]
        elif not is_one and start is not None:
            end = cumprod_data.index[i]
            span = ax.axvspan(start, end, facecolor='lightgray', alpha=0.5, zorder=0, edgecolor='none', lw=0)
            span.set_antialiased(False)
            start = None

    ax.legend(loc='lower right', fontsize=12, frameon=True)
    plt.tight_layout()
    plt.show()

def plot_sharpes_df(df, references):
    """
    Plots a Pandas DataFrame with two columns as a grouped bar plot with horizontal reference lines.
    Each horizontal line is drawn for each key-value pair in the `references` dictionary.

    Parameters:
    df (pd.DataFrame): DataFrame with two columns. Each column is plotted as bars.
    references (dict): Dictionary of horizontal reference lines, where keys are labels and values are reference values.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    bar_colors = ['#1f77b4', '#ff7f0e'] 
    
    df.plot(kind='bar', ax=ax, width=0.5, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax.set_ylim(0, 0.69)
    
    ref_colors = ['black', '#1f77b4', '#ff7f0e']
    
    for i, (label, ref_value) in enumerate(references.items()):
        color = ref_colors[i % len(ref_colors)]
        ax.axhline(
            y=ref_value, 
            color=color, 
            linestyle='--',
            dashes=(12, 4), 
            linewidth=0.75,
            label=f'{label}: {ref_value:.2f}'
        )
    
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend().remove()
    
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, length = 6, width = 0.5)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_ylabel("Sharpe ratio", fontname='Times New Roman')
    ax.set_xlabel("Number of months in TSM signal ($n$)", fontname='Times New Roman')
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(left=0.1, right=0.9)
    plt.show()

def plot_acf_custom(returns):
    ci = 1.96 / np.sqrt(len(returns))
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_axisbelow(False)

    plot_acf(
        returns,
        lags=range(1, 21),
        ax=ax,
        alpha=None,
        vlines_kwargs={'linewidth': .75, 'color': 'C0'}, 
        marker='o',
        markersize=4
    )

    ax.hlines(
        [ci, -ci],
        xmin=0,
        xmax=20,
        linestyles='dotted',
        zorder=2
    )

    ax.axhline(0, color='black', linewidth=1.25, zorder=3)
    ax.set_xlabel("Lag (number of days)", fontname="Times New Roman")
    ax.set_ylabel("Autocorrelation",   fontname="Times New Roman")

    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks(np.arange(-0.1, 0.11, 0.02))

    ax.set_xlim(0, 20)
    ax.set_ylim(-0.1, 0.1)

    fig.subplots_adjust(left=0, right=1)
    ax.tick_params(direction="in", top=True, right=True)
    plt.show()

def cumulative_plot(returns):
    fig, ax = plt.subplots(figsize=(11, 9))
    returns.add(1).cumprod().plot(logy=True, ax=ax, linewidth=1)
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(axis='y', which='both', left=True, right=True, direction='in')
    ax.tick_params(axis='x', which='both', top=True, bottom=True, direction='in')
    ax.margins(0)
    ax.set_ylabel('Cumulative return (log scale)')
    ax.set_xlabel((None))
    plt.show()

def cumulative_plot_shaded(returns, shade_indicator):
    """
    Plot cumulative returns (log scale) for one Series or every column of a DataFrame,
    shade background light grey where shade_indicator == 1,
    and show a legend entry for each return series (column name), but not for the shading.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Period returns (floats), indexed by dates (or strings convertible to dates).
        If DataFrame: each column is plotted separately and gets its column name in the legend.
    shade_indicator : pd.Series
        0/1 flags, same datetime index as returns; shades grey where == 1.
    """
    # ensure datetime indexes
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns = returns.copy()
        returns.index = pd.to_datetime(returns.index)
    if not isinstance(shade_indicator.index, pd.DatetimeIndex):
        shade_indicator = shade_indicator.copy()
        shade_indicator.index = pd.to_datetime(shade_indicator.index)

    fig, ax = plt.subplots(figsize=(11, 9))

    # 1) shade background (no legend entry)
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.fill_between(
        returns.index, 0, 1,
        where=shade_indicator.astype(bool),
        transform=trans,
        color='lightgrey',
        alpha=0.5,
        zorder=0,
        label='_nolegend_'
    )

    # 2) plot each return series
    if isinstance(returns, pd.DataFrame):
        for col in returns.columns:
            cum = (returns[col] + 1).cumprod()
            ax.plot(
                cum.index, cum.values,
                linewidth=1,
                zorder=1,
                label=str(col)
            )
    else:  # single Series
        cum = (returns + 1).cumprod()
        label = returns.name if returns.name is not None else ''
        ax.plot(
            cum.index, cum.values,
            linewidth=1,
            zorder=1,
            label=label
        )

    ax.set_yscale('log')

    # 3) ticks on both sides, inward
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(axis='y', which='both', left=True, right=True, direction='in')
    ax.tick_params(axis='x', which='both', top=True, bottom=True, direction='in')

    # 4) compute 5‑year tick positions
    start_year = returns.index.min().year
    end_year   = returns.index.max().year
    first_5    = math.ceil(start_year / 5) * 5
    last_5     = math.floor(end_year   / 5) * 5
    if first_5 > last_5:
        years = [start_year, end_year]
    else:
        years = list(range(first_5, last_5 + 1, 5))

    tick_locs = [pd.Timestamp(y, 1, 1) for y in years]
    ax.set_xticks(tick_locs)
    ax.set_xticklabels([str(y) for y in years], rotation=0)

    # 5) legend for the return lines only
    ax.legend(loc='lower right')

    ax.margins(0)
    ax.set_ylabel('Cumulative return (log scale)')
    ax.set_xlabel(None)
    plt.show()

def plot_return_and_cumulative_by_regime(
    state_assignments: pd.Series,
    returns: pd.Series,
    reg_ids: dict):
    """
    Plot the return series and cumulative return (log scale),
    coloring each contiguous segment by regime using matplotlib's default color cycle.
    Tick marks are drawn inside and on all sides (top, bottom, left, right).
    
    Parameters
    ----------
    state_assignments : pd.Series of strings
        Index = DateTimeIndex, values = regime names
    returns : pd.Series of floats
        Must align with state_assignments.
    reg_ids : dict
        Mapping from regime name to an integer ID (only used for ordering).
    """

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    regimes = list(reg_ids.keys())
    color_map = {
        regime: color_cycle[i % len(color_cycle)]
        for i, regime in enumerate(regimes)
    }

    label_map = {regime: regime for regime in regimes}
    cum_returns = (returns + 1).cumprod()

    run_id = (state_assignments != state_assignments.shift()).cumsum()
    grouped = run_id.groupby(run_id).groups

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    plotted = set()
    for run_int, timestamps in grouped.items():
        regime = state_assignments.loc[timestamps[0]]
        ax1.plot(
            timestamps,
            returns.loc[timestamps],
            color=color_map.get(regime, 'k'),
            linewidth=0.75,
            label=label_map[regime] if regime not in plotted else None
        )
        plotted.add(regime)

    ax1.tick_params(length=4, width=0.5, direction='in', which='both',
                    top=True, right=True, bottom=True, left=True)
    ax1.set_ylim(-0.20, 0.15)
    ax1.set_title("Daily U.S. stock returns", fontweight='bold')
    ax1.set_ylabel("Returns")
    ax1.legend(loc='upper left')
    ax1.margins(0)

    plotted.clear()
    for run_int, timestamps in grouped.items():
        regime = state_assignments.loc[timestamps[0]]
        ax2.plot(
            timestamps,
            cum_returns.loc[timestamps],
            color=color_map.get(regime, 'k'),
            linewidth=0.75,
            label=label_map[regime] if regime not in plotted else None
        )
        plotted.add(regime)

    ax2.tick_params(length=4, width=0.5, direction='in', which='both',
                    top=True, right=True, bottom=True, left=True)
    ax2.set_yscale('log')
    ax2.set_title("Cumulative returns on U.S. stock market (log scale)", fontweight='bold')
    ax2.set_ylabel("Cumulative return (log scale)")
    ax2.legend(loc='upper left')
    ax2.margins(0)

    plt.xticks(rotation=0)
    plt.tight_layout(pad=3)
    plt.show()

def plot_confusion_matrix(ground_truth, forecasted):
    """Plot confusion matrix comparing full sample vs real time regime identification
    in out-of-sample period.
    """

    cm = confusion_matrix(ground_truth, forecasted, labels=['bull', 'bear', 'chop'])
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    labels = ['bull', 'bear', 'chop']
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_norm, interpolation='nearest', cmap='GnBu')
    plt.title("Confusion matrix: full sample vs real time regime identification in out-of-sample period", fontsize=8)
    plt.colorbar()

    tick_marks = range(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    thresh = cm_norm.max() / 2.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            pct = cm_norm[i, j] * 100
            plt.text(
                j, i,
                f"{pct:.1f}%",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black"
            )

    plt.ylabel("True Regime")
    plt.xlabel("Predicted Regime")
    plt.tight_layout()
    plt.show()


def plot_forecasts(
    actual_hist,
    actual_label,
    forecasted_return,
    off_forecasted_returns,
    off_vol,
    figsize=(12, 10)
):
    """
    Plot historical actuals, highlighted actual, forecast daily revisions,
    base forecast, and ±1.96σ bands, with inward ticks and month–year x‑labels.

    Parameters
    ----------
    actual_hist : pd.Series
        Background historical returns (no legend).
    actual_label : pd.Series
        The slice to call out as “Actual” (plotted with legend).
    forecasted_return : pd.Series
        Daily‐revision forecast (legend: 'Forecast daily revision').
    off_forecasted_returns : pd.Series
        Base forecast (legend: 'Forecast').
    off_vol : pd.Series
        Forecasted vol for computing ±1.96σ bands.

    Returns
    -------
    fig, ax : matplotlib objects
    """
    vol_up = off_forecasted_returns + 1.96 * off_vol
    vol_dn = off_forecasted_returns - 1.96 * off_vol

    fig, ax = plt.subplots(figsize=figsize)

    actual_hist.rename(None).plot(ax=ax, color='blue', legend=False, label=None)
    actual_label.plot(ax=ax, linestyle='--', color='blue', label='Actual')

    forecasted_return.plot(
        ax=ax,
        linestyle=':',
        linewidth=2,
        color='magenta',
        label='Forecast daily revision'
    )

    off_forecasted_returns.plot(ax=ax, color='red', label='Forecast')

    vol_up.plot(ax=ax, linestyle='--', color='red', legend=False)
    vol_dn.plot(ax=ax, linestyle='--', color='red', legend=False)

    ax.tick_params(axis='both', which='both', direction='in', rotation=0)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(-0.05, 0.04)
    ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))

    ax.legend()
    plt.margins(x=0.025)
    plt.tight_layout()