"""
TA-Lib Wrapper using pandas-ta for cross-platform compatibility
Provides the same interface as TA-Lib but uses pandas-ta under the hood
"""
import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Union, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TAWrapper:
    """Wrapper class that mimics TA-Lib interface using pandas-ta"""
    
    @staticmethod
    def SMA(close: Union[pd.Series, np.ndarray], timeperiod: int = 30) -> np.ndarray:
        """Simple Moving Average"""
        try:
            if isinstance(close, np.ndarray):
                close = pd.Series(close)
            result = ta.sma(close, length=timeperiod)
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
            result = ta.ema(close, length=timeperiod)
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
            result = ta.rsi(close, length=timeperiod)
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
            
            macd_data = ta.macd(close, fast=fastperiod, slow=slowperiod, signal=signalperiod)
            
            if macd_data is not None and len(macd_data.columns) >= 3:
                macd_line = macd_data.iloc[:, 0].values
                macd_signal = macd_data.iloc[:, 1].values
                macd_hist = macd_data.iloc[:, 2].values
                return macd_line, macd_signal, macd_hist
            else:
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
            
            bb_data = ta.bbands(close, length=timeperiod, std=nbdevup)
            
            if bb_data is not None and len(bb_data.columns) >= 3:
                upper = bb_data.iloc[:, 0].values  # Upper band
                middle = bb_data.iloc[:, 1].values  # Middle band (SMA)
                lower = bb_data.iloc[:, 2].values   # Lower band
                return upper, middle, lower
            else:
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
            
            stoch_data = ta.stoch(high, low, close, k=fastk_period, d=slowk_period)
            
            if stoch_data is not None and len(stoch_data.columns) >= 2:
                slowk = stoch_data.iloc[:, 0].values  # %K
                slowd = stoch_data.iloc[:, 1].values  # %D
                return slowk, slowd
            else:
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
            
            result = ta.atr(high, low, close, length=timeperiod)
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
            
            result = ta.adx(high, low, close, length=timeperiod)
            return result.values if result is not None else np.full(len(close), np.nan)
        except Exception as e:
            logger.error(f"ADX calculation error: {e}")
            return np.full(len(close), np.nan)

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