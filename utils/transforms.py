import numpy as np
import pandas as pd

from statsmodels.tsa.ar_model import AutoReg
from numba import njit
from typing import Union, Sequence

class TSMEngine:
    """
    Computes time series momentum (TSM) signals and returns for financial data.

    Given a DataFrame or Series of returns, this class calculates lookback
    returns for specified periods, generates trading signals based on the sign
    of these returns, and computes long-short and long-only strategy returns.

    Attributes:
        returns (pd.DataFrame): Input returns as a DataFrame.
        quotes (pd.DataFrame): Cumulative product of returns.
        lookbacks (Sequence[int]): List of lookback periods.
        lookback_returns (pd.DataFrame): Lookback returns for each period.
        signals (pd.DataFrame): Trading signals (+1 for long, -1 for short).
        long_short_returns (pd.DataFrame): Long-short TSM strategy returns.
        long_only_returns (pd.DataFrame): Long-only TSM strategy returns.

    Args:
        returns (Union[pd.DataFrame, pd.Series]): Returns data.
        lookback_list (Sequence[int], optional): Lookback periods (default: 1-24).
    """

    def __init__(
        self,
        returns: Union[pd.DataFrame, pd.Series],
        lookback_list: Sequence[int] = range(1, 25),
    ):
        _returns = returns.to_frame() if isinstance(returns, pd.Series) else returns
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
            l_rets = (self.quotes / self.quotes.shift(lookback)) - 1
            l_rets.columns = [f"tsm_{lookback}"]
            lookback_rets_list.append(l_rets)

        return pd.concat(lookback_rets_list, axis=1)

    def _compute_signals(self):
        return np.sign(self.lookback_returns).shift(1)#.replace(0, 1)

    def _long_short_returns(self):
        return self.signals.mul(self.returns.squeeze(), axis=0)

    def _long_only_returns(self):
        long_only_signals = self.signals.replace(-1, 0)
        long_only_returns = long_only_signals.mul(self.returns.squeeze(), axis=0).dropna(how='all')
        return long_only_returns

# For testing numerical stability
class LogTSMEngine(TSMEngine):
    """Same API, but momentum is computed in log-price space."""
    def _compute_lookback_returns(self):
        log_prices = np.log1p(self.returns).cumsum()
        rets = []
        for lb in self.lookbacks:
            # Δlog-price over the lookback horizon …
            delta = log_prices - log_prices.shift(lb)
            # … converted back to a simple percentage return
            pct_ret = np.expm1(delta)
            pct_ret.columns = [f"tsm_{lb}"]
            rets.append(pct_ret)
        return pd.concat(rets, axis=1)


def m_sharpe(returns):
    " Computes the annualized Sharpe ratio of a series of monthly excess returns."
    return (returns.mean()/returns.std()) * np.sqrt(12)

def compute_msm_ac(k, mu0, mu1, p00, p11):
    """Computes lag-k autocorrelation of a 2-state MSM using transition probabilitiess and regime means.
    
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

def compute_MSM_metrics_daily(df, trading_days=252):
    """
    Computes key metrics for each column of a DataFrame containing daily excess returns.
    
    Parameters:
        df (pd.DataFrame): DataFrame of daily excess returns.
        trading_days (int): Number of trading days per year for annualization (default=252).
        
    Returns:
        pd.DataFrame: Metrics for each column, including:
            - Annual excess return (%)
            - Volatility (%)
            - Sharpe ratio
            - % positive days
            - Number of days (non-missing)
            - Max Drawdown (%)
    """
    metrics = {}
    nobs = len(df)

    for col in df.columns:
        raw = df[col]
        n_days = raw.notna().sum()
        returns = raw.dropna()

        if n_days == 0:
            metrics[col] = {
                'Annual excess return (%)': np.nan,
                'Volatility (%)': np.nan,
                'Sharpe ratio': np.nan,
                '% positive days': np.nan,
                'Number of days': 0,
                'Max Drawdown (%)': np.nan,
            }
            continue
        
        # Annualize return and volatility
        annual_return = returns.mean() * trading_days
        volatility = returns.std() * np.sqrt(trading_days)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(trading_days) if returns.std() != 0 else np.nan
        
        pct_positive = (returns > 0).sum() / n_days * 100
        
        # Cumulative returns and drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        pct_days = returns.count() / nobs * 100

        metrics[col] = {
            'Annual excess return (%)': round(annual_return * 100, 2),
            'Volatility (%)': round(volatility * 100, 2),
            'Sharpe ratio': round(sharpe, 2),
            '% positive days': round(pct_positive, 2),
            '# of days': int(n_days),
            r'% of days': round(pct_days, 2),
            'Max Drawdown (%)': round(max_drawdown * 100, 2),
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

def select_ar_lag(
    series: pd.Series,
    max_lag: int = 21,
    trend: str = 'c',
    verbose: bool = True
) -> tuple[pd.Series, int]:
    """Fit AutoReg models for lags 1..max_lag, select lag with lowest BIC.
    """
    bic_dict = {}
    for lag in range(1, max_lag + 1):
        model = AutoReg(series, lags=lag, trend=trend)
        result = model.fit()
        bic_dict[lag] = result.bic

    bic_series = pd.Series(bic_dict)
    best_lag = bic_series.idxmin()
    if verbose:
        print(f"Selected lag: {best_lag} (BIC={bic_series[best_lag]:.2f})")
    return bic_series, best_lag

def ar2_forecast(
    const: float,
    ar1: float,
    ar2: float,
    y1: float,
    y2: float
) -> float:
    """AR(2) forecast function."""
    return const + ar1*y1 + ar2*y2

def ar2_offline_h_forecast(
    const: float,
    ar1: float,
    ar2: float,
    y1: float,
    y2: float,
    h: int
) -> np.ndarray:
    """
    Produce h-step ahead forecasts from an AR(2) model by iterative substitution.
    
    Parameters
    ----------
    const : float
        Constant term (intercept).
    ar1 : float
        AR coefficient for lag-1.
    ar2 : float
        AR coefficient for lag-2.
    y1 : float
        Observation at time t-1.
    y2 : float
        Observation at time t-2.
    h : int
        Number of steps ahead to forecast.

    Returns
    -------
    list of floats
        Forecasts [ŷ_{t+1}, ŷ_{t+2}, …, ŷ_{t+h}].
    """
    forecasts = []
    prev, last = y2, y1  # prev = y_{t-2}, last = y_{t-1}
    for step in range(1, h+1):
        y_hat = const + ar1 * last + ar2 * prev
        forecasts.append(y_hat)
        prev, last = last, y_hat
    return np.stack(forecasts)