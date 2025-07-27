import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.regime_switching.markov_autoregression import (
    MarkovAutoregression, MarkovAutoregressionResultsWrapper
)
from tqdm import tqdm

from .transforms import ar2_forecast


def _build_transition_matrix(tp: np.ndarray, k: int) -> np.ndarray:
    """Build a row-stochastic matrix P from flat transition probs tp."""
    upper = tp.reshape((k - 1, k))
    P = np.zeros((k, k), dtype=float)
    P[:-1, :] = upper
    P[-1, :] = 1.0 - upper.sum(axis=0)
    return P.T

@dataclass
class MSMParamClassifier:
    """Parses a k-regimes MarkovAutoregression `MarkovAutoregressionResultsWrapper.params` object into:
        - `mus`: intercepts array (k,)
        - `sig`: standard deviations array (k,)
        - `ars`: AR coefficients array (p, k)
        - `P`: transition matrix (k, k)
        - `regime_map`: {'bull':orig_idx,'bear':orig_idx,'chop':orig_idx}
        - `by_regime`: dict of param per regime
    """
    params: pd.Series
    k: int = 3

    mus: np.ndarray = field(init=False)
    sig: np.ndarray = field(init=False)
    ars: np.ndarray = field(init=False)
    P: np.ndarray = field(init=False)
    regime_map: dict = field(init=False)
    by_regime: dict = field(init=False)
    order: list = field(init=False)

    def __post_init__(self):
        vals = self.params.values
        n_tp = (self.k - 1) * self.k
        n_mu = self.k
        n_sigma = self.k

        tp_vals = vals[:n_tp]
        mu_vals = vals[n_tp:n_tp + n_mu]
        sigma2_vals = vals[n_tp + n_mu : n_tp + n_mu + n_sigma]
        ar_vals = vals[n_tp + n_mu + n_sigma :]

        # Build components
        p = int(ar_vals.size / self.k) if ar_vals.size > 0 else 0
        orig_mus = mu_vals.copy()
        # convert variance to std dev
        orig_sigs = np.sqrt(sigma2_vals.copy())
        orig_ars = ar_vals.reshape((p, self.k)) if p > 0 else np.empty((0, self.k))
        orig_P = _build_transition_matrix(tp_vals, self.k)

        # sort regimes by mu
        sorted_idx = np.argsort(orig_mus)
        bear_idx, chop_idx, bull_idx = sorted_idx.tolist()
        self.regime_map = {'bull': bull_idx, 'bear': bear_idx, 'chop': chop_idx}

        self.order = [bull_idx, bear_idx, chop_idx]
        self.mus = orig_mus[self.order]
        self.sig = orig_sigs[self.order]
        self.ars = orig_ars[:, self.order] if p > 0 else np.empty((0, self.k))
        self.P = orig_P[np.ix_(self.order, self.order)]

        # by_regime dict
        br = {}
        for name, pos in zip(['bull', 'bear', 'chop'], range(self.k)):
            br[f'{name}_mu'] = float(self.mus[pos])
            br[f'{name}_sig'] = float(self.sig[pos])
            for lag in range(1, p + 1):
                br[f'{name}_ar{lag}'] = float(self.ars[lag - 1, pos])
        self.by_regime = br

    def __repr__(self):
        regime_names = ['bull', 'bear', 'chop']
        ar_headers = [f"AR{lag+1:>8}" for lag in range(self.ars.shape[0])]
        header = (
            f"{'Regime':<8} | {'μ':>10} | {'σ':>10} | "
            + " | ".join(ar_headers)
        )
        sep = '-' * len(header)
        rows = []
        for pos, name in enumerate(regime_names):
            mu = f"{self.mus[pos]:10.5f}"
            s = f"{self.sig[pos]:10.5f}"
            ars = ' | '.join(
                f"{self.ars[lag, pos]:10.5f}" for lag in range(self.ars.shape[0])
            )
            row = f"{name:<8} | {mu} | {s}"
            if ars:
                row += " | " + ars
            rows.append(row)
        P_str = '\n'.join(
            ' '.join(f"{v:8.4f}" for v in row) for row in self.P
        )
        return (
            f"{' MSM Param Classifier(k='+str(self.k)+') ':=^{len(header)}}\n"
            f"{header}\n{sep}\n" +
            "\n".join(rows) +
            f"\n\nTransition matrix P (bull, bear, chop):\n{P_str}\n"
            f"\nregime_map: {self.regime_map}\n"
        )

class ExpandingMSMARFitter:
    """Fits Markov-Switching AR models on expanding windows of an univariate series,
    using a ThreadPoolExecutor with a default of n_threads = max(1, cpu_count - 4).
    
    Parameters
    ----------
    data : pd.Series
        Time-indexed target series.
    start_date : str or pd.Timestamp, optional
        End of first training window. Defaults to data.index.min().
    end_date : str or pd.Timestamp, optional
        Final cutoff. Defaults to data.index.max().
    freq : str, default 'M'
        Expansion frequency (e.g. 'M', 'D', 'Y').
    max_workers : int, optional
        Number of threads (default = cpu_count - 4, min 1).
    verbose : bool, default False
        If True, logs each fit.
    logger : logging.Logger, optional
        Custom logger; otherwise uses module logger.
    **model_kwargs
        Passed to MarkovAutoregression(...)
    **fit_kwargs
        Passed to .fit(...)
    """

    def __init__(
        self,
        data: pd.Series,
        start_date: Optional[pd.Timestamp] = None,
        end_date:   Optional[pd.Timestamp] = None,
        freq: str = "M",
        max_workers: Optional[int] = None,
        verbose: bool = False,
        logger: Optional[logging.Logger] = None,
        **model_kwargs
    ):
        self.data = data.sort_index()
        first, last = self.data.index.min(), self.data.index.max()
        self.start_date = pd.to_datetime(start_date) if start_date is not None else first
        self.end_date   = pd.to_datetime(end_date)   if end_date   is not None else last
        if not (first <= self.start_date <= last):
            raise ValueError("start_date outside data range")
        if not (self.start_date <= self.end_date <= last):
            raise ValueError("end_date outside data range")

        self.freq = freq
        defaults = dict(
            k_regimes=3,
            order=2,
            switching_ar=True,
            switching_trend=True,
            switching_variance=True,
        )
        self.model_kwargs = {**defaults, **model_kwargs}
        self.fit_kwargs = {"maxiter":1000, "disp": False}

        cpu = os.cpu_count() or 1
        self.max_workers = max_workers or max(1, cpu - 4)
        self.cancel_event = threading.Event()

        self.results: Dict[pd.Timestamp, Any] = {}
        self.failures: Dict[pd.Timestamp, Exception] = {}
        self.p_count: List[pd.Timestamp] = []
        self._lock = threading.Lock()

        self.verbose = verbose
        self.logger = logger or logging.getLogger(__name__)
        if self.verbose:
            self.logger.setLevel(logging.INFO)

    def _generate_cutoffs(self) -> List[pd.Timestamp]:
        periods = (
            self
            .data
            .index
            .to_period(self.freq)
            .to_timestamp(how="end")
            .normalize()
        )
        unique_ends = sorted(set(periods))
        c = [ts for ts in unique_ends if self.start_date <= ts <= self.end_date]
        return c

    def _fit_and_store(self, cutoff: pd.Timestamp):
        """Worker: fit one model and store results under lock."""
        if self.cancel_event.is_set():
            return
        window = self.data[:cutoff]

        try:
            model  = MarkovAutoregression(window, **self.model_kwargs)
            result = model.fit(**self.fit_kwargs)
        except Exception as ex:
            with self._lock:
                self.failures[cutoff] = ex
            self.logger.error(f"❌ [{cutoff.date()}] failed: {ex}")
            return

        with self._lock:
            self.results[cutoff] = result
            self.p_count.append(cutoff)

        self.logger.info(f"✅ [{cutoff.date()}] fitted ({len(self.p_count)})")

    def run(self):
        """Dispatches one thread per cutoff (up to max_workers), fits in parallel,
        and catches all results in the main thread.
        """
        cutoffs = self._generate_cutoffs()
        if not cutoffs:
            self.logger.warning("No cutoffs to fit.")
            return

        self.logger.info(f"Running {len(cutoffs)} fits on {self.max_workers} threads")
        with ThreadPoolExecutor(max_workers=self.max_workers) as execr:
            futures = {execr.submit(self._fit_and_store, c): c for c in cutoffs}

            for fut in as_completed(futures):
                if self.cancel_event.is_set():
                    break
                fut.result()

        self.logger.info(
            f"Done. ✔️ {len(self.p_count)} succeeded, "
            f"❌ {len(self.failures)} failed."
        )

def forecast_MSM_vol(
    pi: np.ndarray,
    mus: np.ndarray, 
    sig: np.ndarray
) -> np.ndarray:
    """Compute the standard deviation forecasts for a MSM using regime sigmas.

    Parameters
    ----------
    pi : np.ndarray
        Forecasted Regime probabilities, shape (h, k) or (k,).
    mus : np.ndarray
        Regime means, shape (k,).
    sig : np.ndarray
        Regime standard deviations, shape (k,).

    Returns
    -------
    vol_frcst : np.ndarray
        Forecasted standard deviations.
    """
    mean_hat = pi @ mus
    second_mom = pi @ (sig**2 + mus**2)
    var_hat = second_mom - mean_hat**2
    vol_frcst = np.sqrt(var_hat)
    return vol_frcst

def compute_eta(
    observations: np.ndarray,
    forecasts: np.ndarray,
    sigmas: np.ndarray
) -> np.ndarray:
    """Compute eta for a given set of observations, forecasts, and sigmas.

    Parameters
    ----------
    observations : np.ndarray
        The observed values (shape: (T,)).
    forecasts : np.ndarray
        The forecasted values (shape: (T,)).
    sigmas : np.ndarray
        The standard deviations (sigmas) in each regime (shape: (T,)).

    Returns
    -------
    np.ndarray
        The computed eta values (shape: (T,)).
    """
    eta_array = norm.pdf(observations, loc=forecasts, scale=sigmas)
    return eta_array

def update_filter_probs(
    init_probs: np.ndarray,
    eta: np.ndarray,
    P: np.ndarray
) -> tuple:
    """Implements the canonical Hamilton 2-step filter for Markov-switching models.

    Filters and updates probabilities based on initial probabilities, eta, and P.

    Parameters
    ----------
    init_probs : np.ndarray
        Initial probabilities (shape: (k,)).
    eta : np.ndarray
        Computed eta values (shape: (T, k)).
    P : np.ndarray
        Transition matrix (shape: (k, k)).

    Returns
    -------
    tuple
        Filtered probabilities and updated probabilities.
    """
    T = eta.shape[0]
    filtered_probs = np.zeros_like(eta)
    updated_probs = np.zeros_like(eta)
    t_prob = init_probs.copy()

    for t in range(T):
        # filter
        t_prob = t_prob @ P
        filtered_probs[t] = t_prob.copy()
        
        # update
        num = t_prob * eta[t]
        t_prob = num / num.sum()
        updated_probs[t] = t_prob.copy()

    return filtered_probs, updated_probs

def compute_probability_forecasts(
    observations: pd.Series,
    forecasted_values: pd.DataFrame,
    sigmas: pd.Series,
    init_probs: np.ndarray,
    P: np.ndarray
) -> tuple:
    """Compute probability forecasts based on:

    - observations,
    - forecasted_values,
    - sigmas,
    - initial probabilities,
    - transition matrix.

    Parameters
    ----------
    observations : pd.Series
        Actual observed values.
    forecasted_values : pd.DataFrame
        Forecasted values.
    sigmas : pd.Series
        Standard deviations in each regime.
    init_probs : np.ndarray
        Initial probabilities.
    P : np.ndarray
        Transition matrix.

    Returns
    -------
    tuple
        filtered and updated probabilities.
    """
    eta = compute_eta(
        observations=observations,
        forecasts=forecasted_values,
        sigmas=sigmas
    )

    filtered_probs, updated_probs = update_filter_probs(
        init_probs=init_probs,
        eta=eta,
        P=P
    )

    return filtered_probs, updated_probs

def ar2_msm_forecast(
    msm_classification: MSMParamClassifier,
    init_probs: np.ndarray,
    y1: float,
    y2: float,
    n_steps: int
) -> dict:
    """
    Iteratively forecast returns and regime probabilities via an AR(2) in each regime
    and an HMM filter that uses the mixture return as the “observation.”

    Parameters
    ----------
    P : ndarray, shape (k, k)
        Row-stochastic regime transition matrix.
    consts : ndarray, shape (k,)
        Regime-specific intercepts.
    ar1s : ndarray, shape (k,)
        Regime-specific AR(1) coefficients.
    ar2s : ndarray, shape (k,)
        Regime-specific AR(2) coefficients.
    sigmas : ndarray, shape (k,)
        Regime-specific return volatilities (for the likelihood).
    init_probs : ndarray, shape (k,)
        Initial regime probabilities (sum to 1).
    y1 : float
        Most recent observed return (t = 0).
    y2 : float
        Second-most recent observed return (t = -1).
    n_steps : int
        Number of steps ahead to forecast.

    Returns
    -------
    dict with keys
      'filtered_probs'    : ndarray, shape (n_steps, k)
         P( regime at t | data up to t-1 )
      'updated_probs'      : ndarray, shape (n_steps, k)
         P( regime at t | data up to t )
      'regime_forecasts'   : ndarray, shape (n_steps, k)
         AR2 forecasts r̂_{t,j} for each regime j
      'mixture_forecast'   : ndarray, shape (n_steps,)
         Weighted return forecast Σ_j π_{t|t-1,j} · r̂_{t,j}
    """

    # Extract parameters from MSMParamClassifier
    P = msm_classification.P
    consts = msm_classification.mus
    ar1s = msm_classification.ars[0] if msm_classification.ars.shape[0] > 0 else np.zeros(msm_classification.k)
    ar2s = msm_classification.ars[1] if msm_classification.ars.shape[0] > 1 else np.zeros(msm_classification.k)
    sigmas = msm_classification.sig
    
    k = P.shape[0]
    if P.shape != (k, k):
        raise ValueError("P must be square (k×k)")
    for arr, name in [(consts,'consts'), (ar1s,'ar1s'), (ar2s,'ar2s'),
                      (sigmas,'sigmas'), (init_probs,'init_probs')]:
        if arr.shape != (k,):
            raise ValueError(f"{name} must have shape ({k},)")

    pred_probs   = np.zeros((n_steps, k), dtype=float)
    regime_fcast = np.zeros((n_steps, k), dtype=float)
    mix_fcast    = np.zeros(n_steps,     dtype=float)

    prev1 = y1
    prev2 = y2
    prob_t = init_probs.copy()   # P( regime at t=0 | data up to t=0 )

    for t in range(n_steps):
        r_hat = consts + ar1s * prev1 + ar2s * prev2
        regime_fcast[t] = r_hat

        prob_t = prob_t @ P
        pred_probs[t] = prob_t

        mix = prob_t @ r_hat
        mix_fcast[t] = mix

        prev2 = prev1
        prev1 = mix

    return {
        'filtered_probs':    pred_probs,
        'regime_forecasts':  regime_fcast,
        'mixture_forecast':  mix_fcast
    }

class MSMARResultsLite:
    """
    Lightweight container for a MarkovAutoregressionResultsWrapper,
    keeping only copies of filtered_regime_probabilities,
    smoothed_regime_probabilities, and params.
    """
    def __init__(self, result: MarkovAutoregressionResultsWrapper):
        self.filtered_probs = result.filtered_marginal_probabilities.copy()
        self.smoothed_probs = result.smoothed_marginal_probabilities.copy()
        self.params = result.params.copy()

    def __repr__(self):
        return (
            f"<MSMARResultsLite filtered.shape={self.filtered_probs.shape} "
            f"smoothed.shape={self.smoothed_probs.shape} "
            f"n_params={len(self.params)}>"
        )

def convert_msmar_results(
    results: Dict[pd.Timestamp, MarkovAutoregressionResultsWrapper]
) -> Dict[pd.Timestamp, MSMARResultsLite]:
    """
    Given a dict mapping timestamps to MarkovAutoregressionResultsWrapper instances,
    return a new dict mapping the same timestamps to MARResultsLite instances.
    """
    return {ts: MSMARResultsLite(res) for ts, res in results.items()}

def forecast_msm_rfc(
    result_df: pd.DataFrame,
    lagged_returns: pd.DataFrame,
    ma_lags_rets: pd.DataFrame,
    rf_kwargs: dict = None
):
    """Forecast returns, volatility, and regime probabilities using fitted MSM 
    params and a Random Forest classifier.

    Parameters
    ----------
    result_df : pd.DataFrame
        Indexed by forecast date, with columns:
        - 'results': fitted MSM results object containing params, filtered_probs, smoothed_probs
        - 'forecast_period': dates corresponding to out-of-sample forecast periods
    lagged_returns : pd.DataFrame
        Lagged return series with columns ['l0', 'l1', 'l2'], indexed by the same dates.
    ma_lags_rets : pd.DataFrame
        Moving-average of lagged returns to use as additional features for the Random Forest.
    rf_kwargs : dict, optional
        Additional keyword arguments to pass to sklearn.ensemble.RandomForestClassifier.

    Returns
    -------
    dict
        A dictionary containing:
        - 'returns': pd.Series of point forecasts for returns
        - 'vol': pd.Series of volatility forecasts
        - 'filtered_probs': pd.DataFrame of MSM filtered state probabilities
        - 'updated_probs': pd.DataFrame of MSM updated state probabilities
        - 'rforest_probs': pd.DataFrame of state probabilities predicted by the Random Forest
    """
    msm_columns = ['bull', 'bear', 'chop']
    vol_frcst = []
    ret_frcst   = []
    updt_prob_frcst  = []
    fltd_prob_frcst = []
    rforest_prob_frcst = []
    rfc = RandomForestClassifier(**rf_kwargs)

    for i in tqdm(result_df.index, desc="Forecasting loop", unit="month"):

        i_msm             = result_df.loc[i]
        i_results         = i_msm['results']
        i_forecast_period = i_msm['forecast_period']

        i_clas       = MSMParamClassifier(i_results.params)
        i_last_probs = i_results.filtered_probs.iloc[-1].values[i_clas.order]
        i_lag_rets   = lagged_returns.loc[i_forecast_period].copy()
        i_idx        = i_lag_rets.index

        i_ar2_frcst = ar2_forecast(
            const = i_clas.mus,
            ar1   = i_clas.ars[0],
            ar2   = i_clas.ars[1],
            y1    = i_lag_rets['l1'].values.reshape(-1,1),
            y2    = i_lag_rets['l2'].values.reshape(-1,1)
        )

        i_fltd_hat, i_updt_hat = compute_probability_forecasts(
            observations      = i_lag_rets['l0'].values.reshape(-1,1),
            forecasted_values = i_ar2_frcst,
            sigmas            = i_clas.sig,
            init_probs        = i_last_probs,
            P                 = i_clas.P
        )

        i_vol_frcst = forecast_MSM_vol(i_fltd_hat, i_clas.mus, i_clas.sig)
        i_vol_frcst = pd.Series(i_vol_frcst, i_idx)
        i_ret_frcst = pd.Series(np.sum(i_ar2_frcst*i_fltd_hat, axis=1), i_idx)

        i_fltd_hat = pd.DataFrame(i_fltd_hat, i_idx, msm_columns)
        i_updt_hat = pd.DataFrame(i_updt_hat, i_idx, msm_columns)

        # Random Forest Pipeline ###############################################
        i_filtered_probs = i_results.filtered_probs[i_clas.order]
        i_lag_probs = i_filtered_probs.shift(1).dropna()
        i_rf_idx = i_lag_probs.index
        
        i_rf_feats = pd.concat([i_lag_probs, ma_lags_rets], axis=1).dropna()
        i_rf_feats_array = i_rf_feats.loc[i_rf_idx].values

        i_smooth_probs = i_results.smoothed_probs[i_clas.order]
        i_smooth_probs.columns = [0, 1, 2]
        i_rf_targets = i_smooth_probs.idxmax(axis=1).dropna().values[1:]

        rfc.fit(i_rf_feats_array, i_rf_targets)

        i_ma_lags_rets = ma_lags_rets.loc[i_idx].copy()
        i_rf_pred_data = pd.concat([i_fltd_hat, i_ma_lags_rets], axis=1).values

        i_rf_probs = rfc.predict_proba(i_rf_pred_data)
        i_rf_probs = pd.DataFrame(i_rf_probs, i_idx, msm_columns)
        
        ret_frcst.append(i_ret_frcst)
        vol_frcst.append(i_vol_frcst)
        updt_prob_frcst.append(i_updt_hat)
        fltd_prob_frcst.append(i_fltd_hat)
        rforest_prob_frcst.append(i_rf_probs)

    ret_frcst = pd.concat(ret_frcst)
    vol_frcst = pd.concat(vol_frcst)
    updt_prob_frcst = pd.concat(updt_prob_frcst)
    fltd_prob_frcst = pd.concat(fltd_prob_frcst)
    rforest_prob_frcst = pd.concat(rforest_prob_frcst)

    return {
        'returns': ret_frcst,
        'vol': vol_frcst,
        'filtered_probs': fltd_prob_frcst,
        'updated_probs': updt_prob_frcst,
        'rforest_probs': rforest_prob_frcst
    }