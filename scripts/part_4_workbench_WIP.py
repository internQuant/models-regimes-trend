import logging
import warnings

import pandas as pd
from tqdm import tqdm
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.regime_switching.markov_autoregression import \
    MarkovAutoregression

from utils.data import *
from utils.misc import *
from utils.plots import *
from utils.transforms import *
from utils.msm import *
import joblib

logger = logging.getLogger("ExpandingMSMARFitter")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    "[%(asctime)s] [%(levelname)s] %(message)s", 
    "%Y-%m-%d %H:%M:%S"
))
logger.addHandler(handler)

warnings.filterwarnings(
    "ignore",
    message=(
        "A date index has been provided, but it has no associated frequency "
        "information and so will be ignored when e.g. forecasting\\."
    ),
    category=ValueWarning,
    module=r"statsmodels\.tsa\.base\.tsa_model"
)

################################################################################
data_source = FFScraper()

returns = (
    data_source
    .daily_data['mkt-rf'][:'2024-08']
    .copy()
    .squeeze()
)

returns_working_sample = returns['1943':].copy().rename('Unconditional')
returns_subsample = returns['1943':'1999'].copy().rename('Unconditional')


################################################################################
first_msm = MarkovAutoregression(
    returns_subsample,
    k_regimes=3,
    order=2,
    switching_ar=True,
    switching_trend=True,
    switching_variance=True
    )
first_msm_fit = first_msm.fit(maxiter=300)

second_msm = MarkovAutoregression(
    returns['1943':'2000-01'],
    k_regimes=3,
    order=2,
    switching_ar=True,
    switching_trend=True,
    switching_variance=True
    )
second_msm_fit = second_msm.fit(maxiter=300)

full_sample_msm = MarkovAutoregression(
    returns_working_sample,
    k_regimes=3,
    order=2,
    switching_ar=True,
    switching_trend=True,
    switching_variance=True
    )
fs_msm_fit = full_sample_msm.fit(maxiter=300)


first_stats = MSMParamClassifier(first_msm_fit.params)
second_stats = MSMParamClassifier(second_msm_fit.params)
fs_stats = MSMParamClassifier(fs_msm_fit.params)

################################################################################
################################################################################
last_probs = (
    first_msm_fit
    .filtered_marginal_probabilities
    .iloc[-1]
    .values[first_stats.order]
)

lagged_returns = pd.concat(
    [returns, returns.shift(1), returns.shift(2)],
    axis=1,
    keys=['l0', 'l1', 'l2']
).dropna()

################################################################################
################################################################################

first_forecast_period = '2000-01'
regrhat = ar2_forecast(
    first_stats.mus,
    first_stats.ars[0],
    first_stats.ars[1],
    lagged_returns.loc[first_forecast_period, 'l1'].values.reshape(-1,1),
    lagged_returns.loc[first_forecast_period, 'l2'].values.reshape(-1,1)
)
################################################################################
################################################################################

sigma_hat = forecast_MSM_vol(phat, first_stats.mus, first_stats.sig)  
sigma_hat = pd.Series(sigma_hat, index=lagged_returns.loc[first_forecast_period].index)
rhat = pd.Series(np.sum(regrhat * phat, axis=1), index=lagged_returns.loc[first_forecast_period].index)

auto_f = ar2_offline_h_forecast(
    const = first_stats.mus,
    ar1 = first_stats.ars[0],
    ar2 = first_stats.ars[1],
    y1 = returns['1999-12-31'],
    y2 = returns['1999-12-30'],
    h=20
)

################################################################################
################################################################################
rahat = pd.Series(np.sum(auto_f * phat, axis=1), index=td.index)

rpstd = rahat + 1.64 * sigma_hat
rmstd = rahat - 1.64 * sigma_hat

returns['1999-07-03':'1999'].plot(figsize=(12,10))
returns['2000-01'].plot(figsize=(12,10), linestyle='--', c='blue')
pred.plot(c = 'magenta', figsize=(12,10), linestyle='--')

rahat.plot(c = 'red', figsize=(12,10))
rpstd.plot(c = 'red', figsize=(12,10), linestyle='--')
rmstd.plot(c = 'red', figsize=(12,10), linestyle='--')



# ##############################################################################
# ##############################################################################
# ## This takes a LOT of time ##################################################

# fitter = ExpandingMSMARFitter(
#     data=returns_working_sample,
#     start_date='1999-12',
#     end_date='2024-08',
#     freq='M',
#     verbose=True,
#     logger=logger
# )

# fitter.run()

# # dump the results to a pickle ###############################################
# joblib.dump(fitter.results, "exp_msm_results_1943-99_2024.joblib")
# ##############################################################################

# load the results from the pickle #############################################
loaded_results = joblib.load("exp_msm_results_1943-99_2024.joblib")
################################################################################

################################################################################
result_series = pd.Series(loaded_results, name='results').to_frame()
result_series.index.name = 'training_cutoff'
result_series.sort_index(inplace=True)
result_series['forecast_period'] = (result_series.index + pd.offsets.MonthEnd(1)).strftime("%Y-%m").unique()

########## Forecasting Loop ####################################################

sigma_hat = []
ret_hat   = []
up_prob_hat  = []
fp_prob_hat = []

for i in tqdm(result_series.index):

    i_res = result_series.loc[i]

    i_clas = MSMParamClassifier(i_res['results'].params)
    i_last_probs = (
        i_res['results']
        .filtered_marginal_probabilities
        .iloc[-1]
        .values[i_clas.order]
    )

    i_lag_rets = lagged_returns.loc[i_res['forecast_period']].copy()
    i_idx = i_lag_rets.index

    i_ar2_forecasts = ar2_forecast(
        i_clas.mus,
        i_clas.ars[0],
        i_clas.ars[1],
        i_lag_rets['l1'].values.reshape(-1,1),
        i_lag_rets['l2'].values.reshape(-1,1)
    )

    i_up_hat, i_fp_hat = compute_probability_forecasts(
        observations=i_lag_rets['l0'].values.reshape(-1,1),
        forecasted_values=i_ar2_forecasts,
        sigmas=i_clas.sig,
        init_probs=i_last_probs,
        P=i_clas.P
    )

    i_sigma_hat = forecast_MSM_vol(i_up_hat, i_clas.mus, i_clas.sig)  
    i_sigma_hat = pd.Series(i_sigma_hat, index=i_idx)
    i_r_hat = pd.Series(np.sum(i_ar2_forecasts * i_up_hat, axis=1), index=i_idx)

    i_up_hat = pd.DataFrame(
        i_up_hat, 
        index=i_idx, 
        columns=['bull', 'bear', 'chop']
    )

    i_fp_hat = pd.DataFrame(
        i_fp_hat, 
        index=i_idx, 
        columns=['bull', 'bear', 'chop']
    )

    sigma_hat.append(i_sigma_hat)
    ret_hat.append(i_r_hat)
    up_prob_hat.append(i_up_hat)
    fp_prob_hat.append(i_fp_hat)


sigma_hat = pd.concat(sigma_hat)

0.15 / (sigma_hat * np.sqrt(252))

ret_hat = pd.concat(ret_hat)
up_prob_hat = pd.concat(up_prob_hat)
fp_prob_hat = pd.concat(fp_prob_hat)



ret_hat.add(1).cumprod().plot(
    figsize=(12,10), 
    title='MSM Forecasted Returns',
    label='Forecasted Returns'
)

returns['2000-01'].plot(kind='bar')

up_prob_hat.plot(figsize=(12,10), title='MSM Regime Probabilities')

ax = up_prob_hat.loc['2000-01'].plot(figsize=(12,10), title='MSM Regime Probabilities')
ax.set_ylim(0, 1)
ax.grid(True)


ax = fp_prob_hat.loc['2000-01'].plot(figsize=(12,10), title='MSM Regime Probabilities')
ax.set_ylim(0, 1)
ax.grid(True)

up_prob_hat['bear'].loc[up_prob_hat['bear']>0.05].reindex(up_prob_hat.index).loc['2008'].plot()

gt = fs_msm_fit.filtered_marginal_probabilities[fs_stats.order]
gt.columns = ['bull', 'bear', 'chop']

gt_s = gt.idxmax(axis=1).loc[up_prob_hat.index]

up_prob_hat_s = up_prob_hat.idxmax(axis=1)



import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Assuming gt_s and up_prob_hat_s are defined (as lists, NumPy arrays, or pandas Series)
cm = confusion_matrix(gt_s, up_prob_hat_s, labels=['bull', 'bear', 'chop'])

# Normalize to percentage by row (true label)
cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

# Optionally, get unique labels for axis ticks
labels = ['bull', 'bear', 'chop']
plt.figure(figsize=(6, 5))
plt.imshow(cm_norm, interpolation='nearest', cmap='GnBu')
plt.title("Confusion matrix: full sample vs real time regime identification in out-of-sample period", fontsize=8)
plt.colorbar()

tick_marks = range(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)

# Annotate percentages in each cell
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

################################################################################

TSME  = TSMEngine(returns_working_sample, [21, 63, 126, 252])

TSM_signals = TSME.signals.replace(-1, 0).dropna()
TSM_signals['tsm_avg'] = (TSM_signals/4).dropna().sum(axis=1)

tsm_returns = TSM_signals.mul(returns_working_sample.loc[TSM_signals.index], axis=0)
tsm_returns = pd.concat([returns_working_sample, tsm_returns], axis=1)


tsm_returns_sub = tsm_returns.loc['2000':'2024-07'].copy()

scalars = 0.15 / (tsm_returns_sub.std() * np.sqrt(252))
vol_m = tsm_returns_sub * scalars
compute_MSM_metrics_daily(vol_m, trading_days=252).T.iloc[:, list(range(3)) + [-1]]



