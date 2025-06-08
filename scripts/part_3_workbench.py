import pandas as pd

from statsmodels.tsa.regime_switching.markov_autoregression \
    import MarkovAutoregression

from utils.data import *
from utils.misc import *
from utils.msm import *
from utils.plots import *
from utils.transforms import *

## Load data ###################################################################
# Uncomment this to use latest data snapshot fetched directly from professor ###
# Kenneth French's Data Library instead of the provided snapshot ###############
# data_source = FFScraper()
# returns = data_source.daily_data['mkt-rf'][:'2024-08'].copy().squeeze()
# returns_subsample = returns['1943':].copy().rename('Unconditional')

csv_f = pd.read_csv(
    r"data\historical_snapshots\ff_data_daily_202410_snapshot.csv", 
    skiprows=4, 
    parse_dates=True
).dropna()

csv_f = csv_f.iloc[:,:2]
csv_f.columns = ['date', 'returns']
csv_f['date'] = pd.to_datetime(csv_f['date'])
csv_f = csv_f.set_index('date')
csv_f = csv_f['returns'].rename('Unconditional')
csv_f = csv_f/100

# Comment this if using latest data snapshot as mentioned above
returns = csv_f.copy()[:'2024-08']
returns_subsample = returns['1943':].copy().rename('Unconditional')

## Plot autocorrelation ########################################################
plot_acf_custom(returns_subsample)

## Select best AR lag based on BIC #############################################
bic_series, best_lag = select_ar_lag(series=returns.values, max_lag=21)
best_lag_fit = AutoReg(returns_subsample.values, lags=best_lag, trend='c').fit()
print(best_lag_fit.summary())

## Fit 3 Regimes, AR(2) MSM ####################################################
## This takes a while do run ###################################################
msm_ar_model = MarkovAutoregression(
    returns_subsample,
    k_regimes=3,
    order=2,
    switching_ar=True,
    switching_trend=True,
    switching_variance=True
    )

msm_fit = msm_ar_model.fit(maxiter=300)
print(msm_fit.summary())

## Detect regimes and classify them as bull, bear and chop #####################
msm_classification = MSMParamClassifier(msm_fit.params)
print(msm_classification)

reg_summ = build_3_regimes_summary(
    estimates=msm_fit.params,
    tvalues=msm_fit.tvalues,
    pvalues=msm_fit.pvalues
).round(4)

################################################################################
### Assign the regime with the highest probability as the regime in each day ###
smoothed_probs = msm_fit.smoothed_marginal_probabilities
smoothed_probs = smoothed_probs[msm_classification.order]
smoothed_probs.columns = ['bull', 'bear', 'chop']

active_regime = smoothed_probs.idxmax(axis=1)

## Get returns in bull/bear and calculate their returns accordingly ############
## Build oracle strategies returns #############################################
bull_returns = returns_subsample.iloc[2:].loc[active_regime == 'bull']
bear_returns = returns_subsample.iloc[2:].loc[active_regime == 'bear']

regime_ret = (
    pd.concat([bull_returns, bear_returns], axis=1, keys=["Bull", "Bear"])
    .reindex(active_regime.index)
    .fillna(0)
)

ls = (regime_ret["Bull"] - regime_ret["Bear"])
lo = regime_ret["Bull"]

strat_rets = (
    pd.concat(
        [returns_subsample, ls, lo],
        axis=1,
        keys=["Buy and hold", "Oracle Long-Short", "Oracle Long-Only"]
    )
    .fillna(0)
)

## Plot returns and color by regime ############################################
plot_return_and_cumulative_by_regime(
    state_assignments=active_regime,
    returns=returns_subsample,
    reg_ids=msm_classification.regime_map
)

## Plot cumulative oracle strategy returns #####################################
cumulative_plot(strat_rets)

## Compute regime-specific and oracle strategy metrics ######################### 
cols = []
for k, r in msm_classification.regime_map.items():
    cols.append(returns_subsample.where(active_regime == k).rename(k))
regimes_returns = pd.concat(cols, axis=1)

reg_rets = (
    pd.concat(
        [returns_subsample.iloc[2:], regimes_returns],
        axis=1
    )
    .dropna(how='all')
)

compute_MSM_metrics_daily(reg_rets)
compute_MSM_metrics_daily(strat_rets.iloc[:, [0, 2, 1]])