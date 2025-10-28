"""
TA-Lib-style wrapper implemented using the `ta` library (PyPI).
Provides a pandas-friendly interface and pandas-ta-compatible outputs where needed.
"""
import pandas as pd
import ta
import numpy as np
from typing import Union, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TAWrapper:
    """Wrapper class that mimics TA-Lib interface using the `ta` library.

    Notes:
    - Uppercase methods (SMA, EMA, RSI, MACD, BBANDS, STOCH, ATR, ADX) return numpy arrays
      for compatibility with older TA-Lib-style consumers.
    - Lowercase methods (sma, ema, rsi, macd, bbands, stoch, atr, adx) return pandas Series/DataFrames
      and emulate pandas-ta column naming to minimize refactors elsewhere.
    """
    
    @staticmethod
    def SMA(close: Union[pd.Series, np.ndarray], timeperiod: int = 30) -> np.ndarray:
        """Simple Moving Average"""
        try:
            if isinstance(close, np.ndarray):
                close = pd.Series(close)
            result = ta.trend.sma_indicator(close, window=timeperiod)
            return result.values if result is not None else np.full(len(close), np.nan)
        except Exception as e:
            logger.error(f"SMA calculation error: {e}")
            return np.full(len(close), np.nan)
    
    @staticmethod
    def EMA(close: Union[pd.Series, np.ndarray], timeperiod: int = 30) -> np.ndarray:
        """Exponential Moving Average"""
        try:
            if isinstance(close, np.ndarray):
                close = pd.Series(close)
            result = ta.trend.ema_indicator(close, window=timeperiod)
            return result.values if result is not None else np.full(len(close), np.nan)
        except Exception as e:
            logger.error(f"EMA calculation error: {e}")
            return np.full(len(close), np.nan)
    
    @staticmethod
    def RSI(close: Union[pd.Series, np.ndarray], timeperiod: int = 14) -> np.ndarray:
        """Relative Strength Index"""
        try:
            if isinstance(close, np.ndarray):
                close = pd.Series(close)
            result = ta.momentum.rsi(close, window=timeperiod)
            return result.values if result is not None else np.full(len(close), np.nan)
        except Exception as e:
            logger.error(f"RSI calculation error: {e}")
            return np.full(len(close), np.nan)
    
    @staticmethod
    def MACD(close: Union[pd.Series, np.ndarray], 
             fastperiod: int = 12, 
             slowperiod: int = 26, 
             signalperiod: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD - Moving Average Convergence Divergence"""
        try:
            if isinstance(close, np.ndarray):
                close = pd.Series(close)
            macd_line = ta.trend.macd(close, window_fast=fastperiod, window_slow=slowperiod)
            macd_signal = ta.trend.macd_signal(close, window_fast=fastperiod, window_slow=slowperiod, window_sign=signalperiod)
            macd_hist = ta.trend.macd_diff(close, window_fast=fastperiod, window_slow=slowperiod, window_sign=signalperiod)
            if macd_line is not None and macd_signal is not None and macd_hist is not None:
                return macd_line.values, macd_signal.values, macd_hist.values
            nan_array = np.full(len(close), np.nan)
            return nan_array, nan_array, nan_array
        except Exception as e:
            logger.error(f"MACD calculation error: {e}")
            nan_array = np.full(len(close), np.nan)
            return nan_array, nan_array, nan_array
    
    @staticmethod
    def BBANDS(close: Union[pd.Series, np.ndarray], 
               timeperiod: int = 20, 
               nbdevup: float = 2, 
               nbdevdn: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands"""
        try:
            if isinstance(close, np.ndarray):
                close = pd.Series(close)
            upper = ta.volatility.bollinger_hband(close, window=timeperiod, window_dev=nbdevup)
            middle = ta.volatility.bollinger_mavg(close, window=timeperiod)
            lower = ta.volatility.bollinger_lband(close, window=timeperiod, window_dev=nbdevdn)
            if upper is not None and middle is not None and lower is not None:
                return upper.values, middle.values, lower.values
            nan_array = np.full(len(close), np.nan)
            return nan_array, nan_array, nan_array
        except Exception as e:
            logger.error(f"BBANDS calculation error: {e}")
            nan_array = np.full(len(close), np.nan)
            return nan_array, nan_array, nan_array
    
    @staticmethod
    def STOCH(high: Union[pd.Series, np.ndarray], 
              low: Union[pd.Series, np.ndarray], 
              close: Union[pd.Series, np.ndarray],
              fastk_period: int = 5,
              slowk_period: int = 3,
              slowd_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic Oscillator"""
        try:
            if isinstance(high, np.ndarray):
                high = pd.Series(high)
            if isinstance(low, np.ndarray):
                low = pd.Series(low)
            if isinstance(close, np.ndarray):
                close = pd.Series(close)
            slowk = ta.momentum.stoch(high, low, close, window=fastk_period, smooth_window=slowd_period)
            slowd = ta.momentum.stoch_signal(high, low, close, window=fastk_period, smooth_window=slowd_period)
            if slowk is not None and slowd is not None:
                return slowk.values, slowd.values
            nan_array = np.full(len(close), np.nan)
            return nan_array, nan_array
        except Exception as e:
            logger.error(f"STOCH calculation error: {e}")
            nan_array = np.full(len(close), np.nan)
            return nan_array, nan_array
    
    @staticmethod
    def ATR(high: Union[pd.Series, np.ndarray], 
            low: Union[pd.Series, np.ndarray], 
            close: Union[pd.Series, np.ndarray],
            timeperiod: int = 14) -> np.ndarray:
        """Average True Range"""
        try:
            if isinstance(high, np.ndarray):
                high = pd.Series(high)
            if isinstance(low, np.ndarray):
                low = pd.Series(low)
            if isinstance(close, np.ndarray):
                close = pd.Series(close)
            result = ta.volatility.average_true_range(high, low, close, window=timeperiod)
            return result.values if result is not None else np.full(len(close), np.nan)
        except Exception as e:
            logger.error(f"ATR calculation error: {e}")
            return np.full(len(close), np.nan)
    
    @staticmethod
    def ADX(high: Union[pd.Series, np.ndarray], 
            low: Union[pd.Series, np.ndarray], 
            close: Union[pd.Series, np.ndarray],
            timeperiod: int = 14) -> np.ndarray:
        """Average Directional Index"""
        try:
            if isinstance(high, np.ndarray):
                high = pd.Series(high)
            if isinstance(low, np.ndarray):
                low = pd.Series(low)
            if isinstance(close, np.ndarray):
                close = pd.Series(close)
            result = ta.trend.adx(high, low, close, window=timeperiod)
            return result.values if result is not None else np.full(len(close), np.nan)
        except Exception as e:
            logger.error(f"ADX calculation error: {e}")
            return np.full(len(close), np.nan)

    # pandas-ta style, pandas-returning helpers used by ChartAnalysisAgent
    def sma(self, close: Union[pd.Series, np.ndarray], length: int = 30) -> pd.Series:
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
        return ta.trend.sma_indicator(close, window=length)

    def ema(self, close: Union[pd.Series, np.ndarray], length: int = 30) -> pd.Series:
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
        return ta.trend.ema_indicator(close, window=length)

    def rsi(self, close: Union[pd.Series, np.ndarray], length: int = 14) -> pd.Series:
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
        return ta.momentum.rsi(close, window=length)

    def macd(self, close: Union[pd.Series, np.ndarray], fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
        macd_line = ta.trend.macd(close, window_fast=fast, window_slow=slow)
        macd_sig = ta.trend.macd_signal(close, window_fast=fast, window_slow=slow, window_sign=signal)
        macd_diff = ta.trend.macd_diff(close, window_fast=fast, window_slow=slow, window_sign=signal)
        df = pd.DataFrame({
            f"MACD_{fast}_{slow}_{signal}": macd_line,
            f"MACDs_{fast}_{slow}_{signal}": macd_sig,
            f"MACDh_{fast}_{slow}_{signal}": macd_diff,
        })
        return df

    def bbands(self, close: Union[pd.Series, np.ndarray], length: int = 20, std: float = 2.0) -> pd.DataFrame:
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
        upper = ta.volatility.bollinger_hband(close, window=length, window_dev=std)
        middle = ta.volatility.bollinger_mavg(close, window=length)
        lower = ta.volatility.bollinger_lband(close, window=length, window_dev=std)
        df = pd.DataFrame({
            f"BBU_{length}_{std}": upper,
            f"BBM_{length}_{std}": middle,
            f"BBL_{length}_{std}": lower,
        })
        return df

    def stoch(self, high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray], close: Union[pd.Series, np.ndarray], window: int = 14, smooth_window: int = 3) -> pd.DataFrame:
        if isinstance(high, np.ndarray):
            high = pd.Series(high)
        if isinstance(low, np.ndarray):
            low = pd.Series(low)
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
        k = ta.momentum.stoch(high, low, close, window=window, smooth_window=smooth_window)
        d = ta.momentum.stoch_signal(high, low, close, window=window, smooth_window=smooth_window)
        # Emulate pandas-ta default column names: STOCHk_14_3_3, STOCHd_14_3_3
        df = pd.DataFrame({
            f"STOCHk_{window}_{smooth_window}_{smooth_window}": k,
            f"STOCHd_{window}_{smooth_window}_{smooth_window}": d,
        })
        return df

    def atr(self, high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray], close: Union[pd.Series, np.ndarray], length: int = 14) -> pd.Series:
        if isinstance(high, np.ndarray):
            high = pd.Series(high)
        if isinstance(low, np.ndarray):
            low = pd.Series(low)
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
        return ta.volatility.average_true_range(high, low, close, window=length)

    def adx(self, high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray], close: Union[pd.Series, np.ndarray], length: int = 14) -> pd.Series:
        if isinstance(high, np.ndarray):
            high = pd.Series(high)
        if isinstance(low, np.ndarray):
            low = pd.Series(low)
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
        return ta.trend.adx(high, low, close, window=length)

# Create a module-level instance for easy importing
talib = TAWrapper()

# Export commonly used functions at module level for compatibility
SMA = talib.SMA
EMA = talib.EMA
RSI = talib.RSI
MACD = talib.MACD
BBANDS = talib.BBANDS
STOCH = talib.STOCH
ATR = talib.ATR
ADX = talib.ADX