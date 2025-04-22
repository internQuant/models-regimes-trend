import pandas as pd

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from models_regimes_trend.utils.data import *
from models_regimes_trend.utils.misc import *
from models_regimes_trend.utils.plots import *
from models_regimes_trend.utils.transforms import *

###############################################################################
## Data ingestion #############################################################
# Load the Fama-French data from the FFScraper class
# This class provides access to Pofessor Kenneth French's data library
# The data being fetched is in the form of monthly and daily returns

data_source = FFScraper()

monthly_returns =(
    data_source
    .monthly_data['mkt-rf']['1943':'2023-07']
    .copy()
    .squeeze()
    .rename('Buy and Hold')
)

daily_returns = (
    data_source
    .daily_data['mkt-rf']['1942':'2023-07']
    .copy()
    .squeeze()
    .rename('Buy and Hold Daily')
)

daily_returns.plot()

###############################################################################
# PART I ######################################################################
## Compute TSM returns and sharpes using long-only and long-short strategies
## MonthlyTSMEngine is a class that computes the returns and signals for TSM strategies 
## It requires the series of monthly returns as input
## Plot sharpes accordingly
TSME = MonthlyTSMEngine(monthly_returns)
TSME_S = TSME.signals.copy().replace(0, np.nan)

lo_sharpe = m_sharpe(TSME.long_only_returns)
ls_sharpe = m_sharpe(TSME.long_short_returns)
bh_sharpe = m_sharpe(monthly_returns).item() # just need the single value

plot_sharpes([lo_sharpe, ls_sharpe], bh_sharpe)

###############################################################################
# Fit model -> obtain parameters, stats and smoothed probabilities
# Trend constant 'c' includes an intercept and allows switching mean estimation
# switching_variance allows switching variance estimation in a similar manner
model = (
    MarkovRegression(
        monthly_returns,
        k_regimes=2,
        trend='c',
        switching_variance=True
        )
)
result = model.fit()
regimes_stats = identify_regimes(result.params)
pd.DataFrame(regimes_stats)

smoothed_probs = result.smoothed_marginal_probabilities

regime_assignments = smoothed_probs.idxmax(axis=1)

regimes_monthly_returns = pd.concat(
    [
        monthly_returns[regime_assignments==0],
        monthly_returns[regime_assignments==1]
    ],
    axis=1,
    keys = ['Bull', 'Bear']
    )

compute_MSM_metrics(pd.concat([monthly_returns, regimes_monthly_returns], axis=1))

MSM_triple_plot(
    regimes_monthly_returns,
    smoothed_probs,
    regime_assignments,
    monthly_returns
)

###############################################################################
# Plot MSM and sample autocorrelation 
mu1 = regimes_stats['bull']['beta0']
mu2 = regimes_stats['bear']['beta0']
p11 = regimes_stats['bull']['p11']
p22 = regimes_stats['bear']['p22']

k_arr = np.arange(1, 21)
msm_ar = pd.Series(compute_msm_ac(k_arr, mu1, mu2, p11, p22), index = k_arr)
plot_msm_ac(monthly_returns, msm_ar)

###############################################################################
# PART II #####################################################################
## Compute and plot Buy and hold's and MSM's metrics and returns
all_ret = pd.concat([monthly_returns, regimes_monthly_returns], axis=1)
all_ret['LS'] = all_ret['Bull'].fillna(0) - all_ret['Bear'].fillna(0)

strats_rets = all_ret[['Buy and Hold', 'LS', 'Bull',]].fillna(0)
strats_rets.columns = ['Buy and Hold', 'Oracle Long-Short', 'Oracle Long-Only']

compute_MSM_metrics(strats_rets[['Buy and Hold', 'Oracle Long-Only', 'Oracle Long-Short']])

msm_cumulative_returns_plot(strats_rets, regime_assignments)

###############################################################################
## Compute and plot the TSM false positive, false negative and overall error rates
## The TSM signals are compared to the regime assignments from the MSM model
## The TSM signals are then classified into four categories:
## 1. True Positive (TP): TSM Bull and MSM Bull
## 2. True Negative (TN): TSM Bear and MSM Bear
## 3. False Positive (FP): TSM Bull and MSM Bear
## 4. False Negative (FN): TSM Bear and MSM Bull
## The classification is done using the detect_tsm_classifications function
## Here the replication starts to diverge from the original paper a bit
regime_signals = regime_assignments.replace(1,-1).replace(0, 1)
classi = detect_tsm_classifications(TSME_S, regime_signals)

tp = classi[classi == 1].count()
tn = classi[classi == 2].count()
fp = classi[classi == 3].count()
fn = classi[classi == 4].count()

TSME.lookback_returns.head(24)
TSME_S.head(24)

fpr = (fp/(fp+tn))
fnr = (fn/(fn+tp))
oer = (fp+fn)/(classi.count())

errors = pd.concat(
    [fpr, fnr, oer],
    axis=1,
    keys = ['False Positive', 'False Negative', 'Overall error']
)
plot_misclas(errors)
scatter_plot_misclas(errors)

###############################################################################
# Compute the volatility managed returns and metrics for the
# long-only and long-short strategies
# The volatility managed returns are computed using the 
# lovm_target_vol_scalar and lsvm_target_vol_scalars functions
# These functions replicate the formulas described in the paper
# The target volatility is set to 0.1 (10%)
# The volatility managed returns are then scaled to the target volatility 

sigma_1 = all_ret['Bull'].std() * np.sqrt(12)
sigma_2 = all_ret['Bear'].std() * np.sqrt(12)

d = 2 - p11 - p22
pi1 = (1 - p22) / d
pi2 = (1 - p11) / d

target_vol = 0.1
lo_w = lovm_target_vol_scalar(target_vol, sigma_1, pi1, pi2, mu1)
ls_lw, ls_sw = lsvm_target_vol_scalars(target_vol, sigma_1, sigma_2, pi1, pi2, mu1, mu2)

lovm = (all_ret['Bull'].fillna(0) * lo_w)
lovm = lovm * (target_vol / (lovm.std() * np.sqrt(12)))

lsvm = (all_ret['Bull'].fillna(0) * ls_lw + all_ret['Bear'].fillna(0) * ls_sw) 
lsvm = lsvm * (target_vol / (lsvm.std() * np.sqrt(12)))

vm_rets = pd.concat(
    [lovm, lsvm],
    axis=1,
    keys=['Oracle Long-Only (volatility managed)', 'Oracle Long-Short (volatility managed)']
)

compute_MSM_metrics(pd.concat([strats_rets, vm_rets], axis=1)) 

###############################################################################
## Compute de majority vote returns and metrics for the long-only and long-short strategies
## The majority vote returns are computed using the TSME_S signals
## The replication continues to diverge further when dealing with TSM signals
## The majority vote signals are computed by comparing the sum of the TSME_S 
## signals to half of the number of signals
## The signals are then multiplied by the monthly returns to obtain the majority vote returns

positve_tsm_signals = TSME_S.replace(-1, 0).sum(axis=1)
signal_count = TSME_S.count(axis=1)

maj_vote_cond = (positve_tsm_signals >= signal_count/2).astype(int)

lo_maj_vote = monthly_returns * maj_vote_cond
ls_maj_vote = monthly_returns * maj_vote_cond.replace(0, -1)

lo_mv_sharpe = m_sharpe(lo_maj_vote)
ls_mv_sharpe = m_sharpe(ls_maj_vote)

tsm_mv = pd.concat(
    [lo_maj_vote, ls_maj_vote],
    keys = ['TSM Long-Only (majority vote)', 'TSM Long-Short (majority vote)'],
    axis = 1,
)

compute_MSM_metrics(tsm_mv)

###############################################################################
## Compute the volatility managed majority vote returns and metrics
## The volatility managed majority vote returns are computed using the daily returns
## The daily returns are used to compute the rolling 252-day standard deviation
## The rolling standard deviation is then used to compute the volatility scalar
## The volatility managed majority vote returns are then computed by multiplying
## the majority vote returns by the volatility scalar

rolling_252_std = daily_returns.rolling(252).std().dropna()
rolling_252_std = rolling_252_std.groupby(pd.Grouper(freq='ME')).last().dropna().shift(1)
rolling_252_std.index = rolling_252_std.index.map(lambda x: x.replace(day=1))
vol_scalar = 0.1 / (rolling_252_std * np.sqrt(252))

lo_maj_vote_vm = lo_maj_vote * vol_scalar
ls_maj_vote_vm = ls_maj_vote * vol_scalar

lo_mv_vm_sharpe = m_sharpe(lo_maj_vote_vm)
ls_mv_vm_sharpe = m_sharpe(ls_maj_vote_vm)

tsm_lovm_sharpe = m_sharpe(TSME.long_only_returns.mul(vol_scalar, axis=0))
tsm_lsvm_sharpe = m_sharpe(TSME.long_short_returns.mul(vol_scalar, axis=0))

long_only_sharpes = pd.concat([lo_sharpe, tsm_lovm_sharpe],axis=1)
long_short_sharpes = pd.concat([ls_sharpe, tsm_lsvm_sharpe],axis=1)

plot_sharpes_df(
    long_only_sharpes,
    {
        "Buy and Hold": bh_sharpe,
        "Majority Vote": lo_mv_sharpe,
        "Majority Vote Volatility Managed": lo_mv_vm_sharpe
    }
)

plot_sharpes_df(
    long_short_sharpes,
    {
        "Buy and Hold": bh_sharpe,
        "Majority Vote": ls_mv_sharpe,
        "Majority Vote Volatility Managed": ls_mv_vm_sharpe
    }
)