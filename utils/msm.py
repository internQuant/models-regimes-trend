import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from scipy.stats import norm

import numpy as np
import pandas as pd

from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

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
        self.fit_kwargs = {"maxiter":300, "disp": False}

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
    sigma_hat : np.ndarray
        Forecasted standard deviations.
    """
    mean_hat = pi @ mus
    second_mom = pi @ (sig**2 + mus**2)
    var_hat = second_mom - mean_hat**2
    sigma_hat = np.sqrt(var_hat)
    return sigma_hat

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

def update_predict_probs(
    init_probs: np.ndarray,
    eta: np.ndarray,
    P: np.ndarray
) -> tuple:
    """Update and predict probabilities based on initial probabilities, eta, and transition matrix P.

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
    predicted_probs = np.zeros_like(eta)
    updated_probs = np.zeros_like(eta)

    prob_t = init_probs.copy()

    for t in range(T):
        # Predict step
        curr_probs = prob_t @ P
        predicted_probs[t] = curr_probs
        
        # Update step
        num = curr_probs * eta[t]
        prob_t = num / num.sum()
        updated_probs[t] = prob_t

    return predicted_probs, updated_probs

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
        Predicted probabilities and updated probabilities.
    """
    eta = compute_eta(
        observations=observations,
        forecasts=forecasted_values,
        sigmas=sigmas
    )

    predicted_probs, updated_probs = update_predict_probs(
        init_probs=init_probs,
        eta=eta,
        P=P
    )

    return predicted_probs, updated_probs