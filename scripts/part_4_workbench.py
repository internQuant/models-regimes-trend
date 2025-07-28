import joblib
import warnings
import numpy as np
import pandas as pd

from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

from utils.data import *
from utils.misc import *
from utils.msm import *
from utils.plots import *
from utils.transforms import *
from utils.presentation import style_panel
from IPython.display import display

warnings.filterwarnings(
    "ignore",
    message=(
        "A date index has been provided, but it has no associated frequency "
        "information and so will be ignored when e.g. forecasting\\."
    ),
    category=ValueWarning,
    module=r"statsmodels\.tsa\.base\.tsa_model"
)

## Load data ###################################################################
# Uncomment below to use the latest data snapshot fetched directly from ########
# professor Kenneth French's Data Library instead of the provided one ##########

# data_source = FFScraper()
# returns = data_source.daily_data['mkt-rf'][:'2024-08'].copy().squeeze()

# Comment this if using latest data snapshot as mentioned above ################
returns = read_ff_daily_csv()[:'2024-08']

## Prepare data ################################################################
## Compute moving averages and lags ############################################
sma_returns = compute_moving_averages(returns)
ma_lags_rets = sma_returns.shift(1).dropna()

returns_working_sample = returns['1943':].copy().rename('Unconditional')
returns_subsample      = returns['1943':'1999'].copy().rename('Unconditional')

lagged_returns = (
    returns
    .to_frame('l0')
    .assign(l1=returns.shift(1), l2=returns.shift(2))
    .dropna()
)

## Fit the first MSM, set msm parameters #######################################
msm_kwargs = dict(
    k_regimes=3,
    order=2,
    switching_ar=True,
    switching_trend=True,
    switching_variance=True
)

first_msm = MarkovAutoregression(returns_subsample, **msm_kwargs)
first_msm_fit = first_msm.fit(maxiter=300)
msm_stats = MSMParamClassifier(first_msm_fit.params)

## Define the first forecast period ############################################
# Forecast the returns for the first period using the fitted MSM model's
# parameters and the lagged returns data, filtering the probabilities upon new
# observations.
forecast_period = '2000-01'
frcst_idx = lagged_returns.loc[forecast_period].index
first_msm_filtered_probs = first_msm_fit.filtered_marginal_probabilities
last_probs = first_msm_filtered_probs.iloc[-1].values[msm_stats.order]
obs = lagged_returns.loc[forecast_period, 'l0'].values.reshape(-1,1)

# Forecast the returns in each regime for the first period using the msm 
# parameters and observed lagged returns
forecasted_regime_returns = ar2_forecast(
    msm_stats.mus,
    msm_stats.ars[0],
    msm_stats.ars[1],
    lagged_returns.loc[forecast_period, 'l1'].values.reshape(-1,1),
    lagged_returns.loc[forecast_period, 'l2'].values.reshape(-1,1)
)

# Compute the filtered probabilities and update probabilities for the forecasted
# period using the msm parameters, observed lagged returns and last filtered
# probabilities from the first MSM fit.
f_fltd_probs, f_updt_probs = compute_probability_forecasts(
    observations      = obs,
    forecasted_values = forecasted_regime_returns,
    sigmas            = msm_stats.sig,
    init_probs        = last_probs,
    P                 = msm_stats.P
)

# Plot the forecasted probabilities ############################################
forecasted_upd_probs = pd.DataFrame(f_updt_probs, frcst_idx)
forecasted_upd_probs.columns = ['bull', 'bear', 'chop']

ax = forecasted_upd_probs.plot(figsize=(12,10), title='MSM Probabilities')
ax.set_ylim(0, 1)
ax.grid(True)

# Forecast the volatility and returns for the first period #####################
forecasted_vol = forecast_MSM_vol(f_fltd_probs, msm_stats.mus, msm_stats.sig)  
forecasted_vol = pd.Series(forecasted_vol, index=frcst_idx)
forecasted_return = pd.Series(
    np.sum(forecasted_regime_returns*f_fltd_probs, axis=1), frcst_idx
)

# "Offiline" forecast for the first period, refeeds the forecasted returns into
# the AR2 model from the first MSM fit.
y1 = returns['1999-12-31']
y2 = returns['1999-12-30']
off_forecast = ar2_msm_forecast(msm_stats, last_probs, y1, y2, len(frcst_idx))

fltd_probs = off_forecast['filtered_probs']
off_vol = forecast_MSM_vol(fltd_probs, msm_stats.mus, msm_stats.sig)
off_forecasted_returns = pd.Series(off_forecast['mixture_forecast'], frcst_idx)

## Plot the forecasted returns online and offline ##############################
plot_forecasts(
    returns['1999-07-03':'1999'],
    returns['2000-01'],
    forecasted_return,
    off_forecasted_returns,
    off_vol
)

## Run the ExpandingMSMARFitter ################################################
# Uncomment below to run the ExpandingMSMARFitter over the entire sample #######
# This helper does the heavy lifting of fitting an MSM every month from the
# forecast period (2000-01) up until (2024-08), storing results in a dictionary
# with the training cutoff as the key and the MSM results wapper as the value.
# This process took around 10 hours on an AMD Ryzen 7 7800X3D with 64GB of RAM.
# It is recommended to run this with the provided checkpoint first. and then
# Tweak the parameters to your liking.

# fitter = ExpandingMSMARFitter(
#     data=returns_working_sample,
#     start_date='1999-12',
#     end_date='2024-08',
#     freq='M',
#     verbose=True,
#     logger=logger
# )
# fitter.run()

# dump the raw results to a pickle #############################################
# WARNING: this takes around 10.5 GB of disk space. 
#joblib.dump(fitter.results, "raw_msm_results_1943-99_2024_snapshot.joblib")
################################################################################

# load the raw results from the pickle #########################################
# raw_results = joblib.load("raw_msm_results_1943-99_2024_snapshot.joblib")
################################################################################

## This helper saves only the MSM properties that are needed for the
# forecasting and strategy construction, reducing the size of the results to
# around 300 MB, it is recommended to use this instead of the raw results.

# Uncomment below if using the ExpandingMSMARFitter to convert the raw results
# lite_results = convert_msmar_results(fitter.results)##########################
# Dump the lightweight results to a pickle #####################################
# joblib.dump(lite_results, "data/baked/lite_msm_results_1943-99_2024_snapshot.joblib")

# Load the lightweight results from the pickle #################################
lite_results = joblib.load("data/baked/lite_msm_results_1943-99_2024_snapshot.joblib")

## Prepares the results for the forecasting and strategy construction ##########
## Simply atributes the training cutoff period as the index of the DataFrame
## and its forecast period as the immediate next month after the training cutoff.
# This is used to align the forecasted data with the working sample returns.

msm_results_df = pd.Series(lite_results, name='results').to_frame()
msm_results_df.index.name = 'training_cutoff'
msm_results_df.sort_index(inplace=True)

fc_per = (msm_results_df.index + pd.offsets.MonthEnd(1))
msm_results_df['forecast_period'] = fc_per.strftime("%Y-%m").unique()

## Forecast returns, vol and probabilities for each period
## Define the parameters for the Random Forest Classifier ######################
rf_kwargs = dict(
    n_estimators      = 420,
    random_state      = 69,
    n_jobs            = -1,
    max_depth         = 10,
    max_features      = 'sqrt',
    min_samples_split = 5,
    min_samples_leaf  = 5,
)

## Run the main forecasting function
# Takes a while to run, around 6 minutes on the same machine as above.
## For each fitted model (one per month), forecast the returns, vol 
# and probabilities whilst mainting its fitted parameters and updating the 
# probabilities with the new observations.
# At the end of each iteration, trains a Random Forest Classifier on the
# lagged filtered probabilities and SMA returns to predict the regime inferred
# by the smoothed probabilities of the fitted MSM model. 
forecasted_data = forecast_msm_rfc(
    msm_results_df,
    lagged_returns,
    ma_lags_rets,
    rf_kwargs
)

## Forecast the full sample MSM ################################################
# Compute the confusion matrix between the full sample MSM's

full_sample_msm = MarkovAutoregression(returns_working_sample, **msm_kwargs)
full_msm_fit  = full_sample_msm.fit(maxiter=300)
full_stats  = MSMParamClassifier(full_msm_fit.params)

full_fltd_probs = full_msm_fit.filtered_marginal_probabilities[full_stats.order]
full_fltd_probs.columns = ['bull', 'bear', 'chop']
ground_truth = full_fltd_probs.idxmax(axis=1).loc[forecasted_data['filtered_probs'].index]
forecasted_regimes = forecasted_data['filtered_probs'].idxmax(axis=1).dropna()
ground_truth = ground_truth.loc[forecasted_regimes.index]

is_bear_regime = (ground_truth == 'bear').astype(int)
plot_confusion_matrix(ground_truth, forecasted_regimes)

################################################################################\
forecasted_msm_probabilites = forecasted_data['filtered_probs'].copy()
forecasted_rfc_probabilites = forecasted_data['rforest_probs'].copy()

idx = forecasted_msm_probabilites.index

## Compute TSM signals #########################################################
TSME = TSMEngine(returns_working_sample, lookback_list=[21, 63, 126, 252])
# Replace -1 (short) positions with 0 (out of market) positions 
TSM_signals = TSME.signals.replace(-1, 0).dropna()
# Compute the avg of the TSM signals and reindex to the forecasted data index
TSM_signals['tsm_avg'] = TSM_signals.mean(axis=1)
TSM_signals = TSM_signals.reindex(idx)

## Extract signal from the forecasted data #####################################
# forecasted vol, returns, msm and rfc probabilities are used to create signals

# Target our forecasted vol to 15% annualized vol prior to returns scaling
forecast_vol_signals = (0.15/(forecasted_data['vol']*np.sqrt(252))).rename('VM')
forecast_returns_signals = np.sign(forecasted_data['returns']).rename('Pos. Forecast')
forecast_returns_signals.loc[forecast_returns_signals<0] = 0

# This function simply applies a threshold to the probabilities
nb_msm_signals = probability_threshold_signals(forecasted_msm_probabilites)
nb_rfc_signals = probability_threshold_signals(forecasted_rfc_probabilites)
nb_rfc_signals.columns  = "RF_" + nb_rfc_signals.columns

nb_msm_tsm_avg_signals = (nb_msm_signals.mul(TSM_signals['tsm_avg'], axis=0))
nb_msm_tsm_avg_signals.columns = "TSM_avg_" + nb_msm_tsm_avg_signals.columns

nb_rfc_tsm_avg_signals = (nb_rfc_signals.mul(TSM_signals['tsm_avg'], axis=0))
nb_rfc_tsm_avg_signals.columns = "TSM_avg_" + nb_rfc_tsm_avg_signals.columns

## Combine all signals into a single DataFrame #################################
strategies_weights = pd.concat([
    (returns_working_sample*0)+1,
    TSM_signals,
    forecast_vol_signals,
    forecast_returns_signals,
    nb_msm_signals,
    nb_msm_tsm_avg_signals,
    nb_rfc_signals,
    nb_rfc_tsm_avg_signals,
], axis=1).dropna()

## Compute simple returns for each strategy ####################################
# The returns are computed by multiplying the weights with the returns of the 
# working sample and then adjusting the returns to target 15% annualized vol.
strategies_returns = strategies_weights.mul(returns_working_sample.loc[idx], axis=0)
strategies_returns_vol_adj = strategies_returns.mul(0.15 / (strategies_returns.std() * np.sqrt(252)))

## Compute metrics for each strategy ###########################################
all_metrics = part4_format_metrics(strategies_returns_vol_adj, trading_days=252)

panel_titles = {
    'A': 'Panel A: Benchmark strategies',
    'B': 'Panel B: Strategies using MSM forecasts',
    'C': 'Panel C: Strategies combining TSM and MSM',
    'D': 'Panel D: Strategies using random forests trained on MSM regimes'
}

panelA = style_panel(all_metrics.iloc[:6],    panel_titles['A'])
panelB = style_panel(all_metrics.iloc[6:11],  panel_titles['B'])
panelC = style_panel(all_metrics.iloc[11:14], panel_titles['C'])
panelD = style_panel(all_metrics.iloc[14:],   panel_titles['D'])

## Plot cumulative returns of selected strategies ##############################
strats_of_interest = [
    'Unconditional', 
    'tsm_avg',
    'VM',
    'no_bear_20%',
    'TSM_avg_no_bear_20%',
    'TSM_avg_RF_no_bear_5%',
]

select_strats = strategies_returns_vol_adj[strats_of_interest].copy()
cumulative_plot_shaded(select_strats, is_bear_regime)