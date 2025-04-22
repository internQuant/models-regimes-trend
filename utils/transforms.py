import numpy as np
import pandas as pd

from numba import njit
from typing import Union, Sequence

class MonthlyTSMEngine:
    """MonthlyTSMEngine computes time series momentum signals and returns for monthly financial data.

    This class implements a time series momentum (TSM) strategy engine for monthly returns data.
    Given a DataFrame or Series of monthly returns, it calculates lookback returns for specified
    lookback periods, generates trading signals based on the sign of these returns, and computes
    the resulting long-short and long-only strategy returns.

    Attributes:
        returns (pd.DataFrame): The input monthly returns as a DataFrame.
        quotes (pd.DataFrame): Cumulative product of returns, representing price quotes.
        lookbacks (Sequence[int]): List of lookback periods to compute momentum.
        lookback_returns (pd.DataFrame): Lookback returns for each period in lookbacks.
        signals (pd.DataFrame): Trading signals (+1 for long, -1 for short) based on lookback returns.
        long_short_returns (pd.DataFrame): Returns from applying long-short TSM strategy.
        long_only_returns (pd.DataFrame): Returns from applying long-only TSM strategy.
    Args:
        monthly_returns (Union[pd.DataFrame, pd.Series]): Monthly returns data.
        lookback_list (Sequence[int], optional): List of lookback periods (in months) to use for momentum calculation.
            Defaults to range(1, 25).
    Methods:
        _compute_lookback_returns(): Computes lookback returns for each lookback period.
        _compute_signals(): Generates trading signals based on the sign of lookback returns.
        _long_short_returns(): Calculates returns for the long-short TSM strategy.
        _long_only_returns(): Calculates returns for the long-only TSM strategy.
    """
    """Monthly Time Series Momentum Engine."""

    def __init__(
        self,
        monthly_returns: Union[pd.DataFrame, pd.Series],
        lookback_list: Sequence[int] = range(1, 25),
        ):
        
        _returns = monthly_returns.to_frame() if isinstance(monthly_returns, pd.Series) else monthly_returns
        self.returns = _returns

        self.quotes = _returns.add(1).cumprod().copy()

        self.lookbacks = lookback_list
        self.lookback_returns = self._compute_lookback_returns()

        self.signals = self._compute_signals()

        self.long_short_returns = self._long_short_returns()
        self.long_only_returns = self._long_only_returns()

    def _compute_lookback_returns(self):
        lookback_rets_list = []

        for lookback in self.lookbacks:
            l_rets = (self.quotes/self.quotes.shift(lookback))-1
            l_rets.columns = [str(lookback)]
            lookback_rets_list.append(l_rets)

        return pd.concat(lookback_rets_list, axis=1)
    
    def _compute_signals(self):
        return np.sign(self.lookback_returns).shift(1).replace(0, 1)
    
    def _long_short_returns(self):
        return self.signals.mul(self.returns.squeeze(), axis=0)
    
    def _long_only_returns(self):
        long_only_signals = self.signals.replace(-1, 0)
        long_only_returns = long_only_signals.mul(self.returns.squeeze(), axis=0).dropna(how='all')
        return long_only_returns

def m_sharpe(returns):
    " Computes the annualized Sharpe ratio of a series of monthly excess returns."
    return (returns.mean()/returns.std()) * np.sqrt(12)

def compute_msm_ac(k, mu0, mu1, p00, p11):
    """Computes lag-k autocorrelation of a 2-state MSM using only transition probs and regime means.
    
    Parameters:
    - k: Lag (integer)
    - mu1, mu2: Regime means
    - p11, p22: Probabilities of remaining in regimes 1 and 2 respectively
    
    Returns:
    - rho_k: Lag-k autocorrelation
    """
    # Compute long-run (stationary) regime probabilities
    d = 2 - p00 - p11
    pi1 = (1 - p11) / d
    pi2 = (1 - p00) / d

    num = pi1 * pi2 * (mu0 - mu1)**2
    den = pi1 * mu0**2 + pi2 * mu1**2 + num
    per = (p00 + p11 - 1)**k
    rho_k = (num / den) * per
    return rho_k

def compute_MSM_metrics(df):
    """Computes key metrics for each column of a DataFrame containing monthly returns.
    
    Parameters:
        df (pd.DataFrame): DataFrame of monthly excess returns.
        
    Returns:
        pd.DataFrame: DataFrame of computed metrics, including annual excess return, 
                      volatility, Sharpe ratio, % positive months, number of months, 
                      and maximum drawdown.
    """
    metrics = {}

    for col in df.columns:
        returns = df[col].dropna()
        n_months = len(returns)
        
        if n_months == 0:
            metrics[col] = {
                'Annual excess return (%)': np.nan,
                'Volatility (%)': np.nan,
                'Sharpe ratio': np.nan,
                '% positive months': np.nan,
                'Number of months': 0,
                'Max Drawdown (%)': np.nan,
            }
            continue
        
        annual_return = returns.mean() * 12
        volatility = returns.std() * np.sqrt(12)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(12) if returns.std() != 0 else np.nan
        pct_positive = (returns > 0).sum() / n_months * 100
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        metrics[col] = {
            'Annual excess return (%)': round(annual_return * 100, 1),
            'Volatility (%)': round(volatility * 100, 1),
            'Sharpe ratio': round(sharpe, 1),
            '% positive months': round(pct_positive, 0),
            'Number of months': n_months,
            'Max Drawdown (%)': round(max_drawdown * 100, 1)
        }

    return pd.DataFrame(metrics)

@njit
# might use in the future
def compute_ewma_vol(
    r: np.ndarray,
    lam: float = 0.94,
    vol_init: float = 0.003
) -> np.ndarray:
    """Compute the exponentially weighted moving average (EWMA) volatility for a matrix of returns.

    args:
    - r: A 2D numpy array of returns (rows represent dates, columns represent different series).
    - lam: The decay factor (lambda) used to weigh past volatility. Typical RiskMetrics value is 0.94.
    - vol_init: The initial volatility value for the first observation.

    returns:
    - A 2D numpy array containing the EWMA volatility values.
    """
    n_obs, n_series = r.shape
    vol = np.empty((n_obs, n_series))
    vol[0, :] = vol_init

    for t in range(1, n_obs):
        vol[t, :] = np.sqrt(lam * vol[t-1, :]**2 + (1-lam) * r[t-1, :]**2)

    return vol

def lovm_target_vol_scalar(
    sigma_target: float, 
    sigma1: float, 
    pi1: float, 
    pi2: float, 
    mu1: float
):
    """
    Computes the scalar adjustment factor to achieve a target level of volatility 
    for the long-only volatility managed MSM model (LOVM).
    Parameters:
        sigma_target (float): The target volatility level.
        sigma1 (float): The volatility of the first regime.
        pi1 (float): The probability of being in the first regime.
        pi2 (float): The probability of being in the second regime.
        mu1 (float): The mean return of the first regime.
    Returns:
        float: The scalar adjustment factor to achieve the target volatility.
    """

    _den = np.sqrt((sigma1**2 * pi1) + (mu1**2 * pi1*pi2))

    return sigma_target/_den

def lsvm_target_vol_scalars(
    sigma_target: float,
    sigma1: float,
    sigma2: float,
    pi1: float,
    pi2: float,
    mu1: float,
    mu2: float,
):
    """
    Calculate the scalar weights for a target volatility in a two-regime 
    long-short volatility managed MSM model (LSVM).
    Parameters:
        sigma_target (float): The target volatility level.
        sigma1 (float): Volatility of the first regime.
        sigma2 (float): Volatility of the second regime.
        pi1 (float): Probability of being in the first regime.
        pi2 (float): Probability of being in the second regime.
        mu1 (float): Mean return of the first regime.
        mu2 (float): Mean return of the second regime.
    Returns:
        tuple: A tuple containing:
            - St1_w (float): Scalar weight for the first regime.
            - St2_w (float): Scalar weight for the second regime.
    """

    _den = np.sqrt(1 + ((mu1/sigma1 + mu2/sigma2)**2 * pi1*pi2))

    St1_w =  (sigma_target/sigma1) / _den
    St2_w = -(sigma_target/sigma2) / _den

    return St1_w, St2_w
