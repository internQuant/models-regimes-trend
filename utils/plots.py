import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from statsmodels.graphics.tsaplots import plot_acf
from cycler import cycler

colors = [
    '#0A7EF2',
    '#FF6B6B',
    '#FFA94D',
    '#556270',
    '#C7F464',
    '#6A4C93',
    '#1A535C',
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
import matplotlib.pyplot as plt
import pandas as pd

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

    # 1) Build a list of default colors from matplotlib’s cycle
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # 2) Assign each regime a color based on its position in reg_ids.keys()
    regimes = list(reg_ids.keys())
    color_map = {
        regime: color_cycle[i % len(color_cycle)]
        for i, regime in enumerate(regimes)
    }

    # 3) (Optional) Human‐readable labels for legend
    label_map = {regime: regime for regime in regimes}

    # 4) Compute cumulative returns
    cum_returns = (returns + 1).cumprod()

    # 5) Identify contiguous runs of the same regime
    run_id = (state_assignments != state_assignments.shift()).cumsum()
    grouped = run_id.groupby(run_id).groups

    # 6) Prepare figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: daily returns colored by regime
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

    # Bottom: cumulative returns on log scale
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
