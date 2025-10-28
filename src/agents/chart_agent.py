"""
Chart Analysis Agent with technical indicators and pattern detection
Migrated to use pandas-ta for cross-platform compatibility
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ..utils.ta_wrapper import TAWrapper
from ..core.trae_orchestrator import TraeOrchestrator, Event, EventType

logger = logging.getLogger(__name__)


class ChartAnalysisAgent:
    """Chart analysis agent with technical indicators using pandas-ta"""
    
    def __init__(self, orchestrator: TraeOrchestrator):
        self.orchestrator = orchestrator
        self.ta = TAWrapper()
        self.running = False
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_expiry: Dict[str, datetime] = {}
        self.cache_duration = timedelta(minutes=5)
        
        # Configuration
        self.config = {
            'ema_fast': 12,
            'ema_slow': 26,
            'rsi_period': 14,
            'atr_period': 14,
            'bb_period': 20,
            'bb_std': 2
        }
    
    async def analyze_symbol(self, symbol: str, timeframe: str = '1h') -> Optional[Dict[str, Any]]:
        """Perform comprehensive chart analysis for a symbol"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe}"
            if self._is_cached(cache_key):
                return self.analysis_cache[cache_key]
            
            # Get OHLCV data (placeholder - would connect to data source)
            df = await self._get_ohlcv_data(symbol, timeframe)
            if df is None or len(df) < 50:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # Perform analysis
            analysis = await self._perform_analysis(df, symbol, timeframe)
            
            # Cache the result
            self.analysis_cache[cache_key] = analysis
            self.cache_expiry[cache_key] = datetime.now() + self.cache_duration
            
            # Publish analysis result via orchestrator
            await self.orchestrator.publish_event(Event(
                type=EventType.CHART_ANALYSIS,
                data={
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'analysis': analysis,
                    'confidence': analysis.get('overall_confidence', 0.5),
                    'timestamp': datetime.now().isoformat()
                },
                source='ChartAgent'
            ))
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze {symbol}: {e}")
            return None
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if analysis is cached and not expired"""
        if cache_key not in self.analysis_cache:
            return False
        
        if cache_key not in self.cache_expiry:
            return False
        
        return datetime.now() < self.cache_expiry[cache_key]
    
    async def _get_ohlcv_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get OHLCV data - placeholder for actual data connector"""
        try:
            # This would be replaced with actual data connector
            # For now, return a sample DataFrame structure
            logger.info(f"Getting OHLCV data for {symbol} {timeframe}")
            
            # Placeholder - in real implementation, this would fetch from broker or data provider
            dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
            np.random.seed(42)  # For consistent test data
            
            # Generate sample OHLCV data
            close_prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
            high_prices = close_prices + np.random.rand(200) * 2
            low_prices = close_prices - np.random.rand(200) * 2
            open_prices = np.roll(close_prices, 1)
            open_prices[0] = close_prices[0]
            volumes = np.random.randint(1000, 10000, 200)
            
            df = pd.DataFrame({
                'timestamp': dates,
                'open': open_prices,
                'high': high_prices,
                'low': low_prices,
                'close': close_prices,
                'volume': volumes
            })
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get OHLCV data for {symbol}: {e}")
            return None
    
    async def _perform_analysis(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Perform comprehensive technical analysis"""
        try:
            # Calculate indicators
            indicators = await self._calculate_indicators(df)
            
            # Detect patterns
            patterns = await self._detect_patterns(df, indicators)
            
            # Calculate support/resistance levels
            levels = await self._calculate_levels(df)
            
            # Determine trend
            trend = await self._determine_trend(indicators)
            
            # Calculate volatility
            volatility = await self._calculate_volatility(df)
            
            # Generate signals
            signals = await self._generate_signals(indicators, patterns, trend)
            
            # Calculate overall confidence
            confidence = await self._calculate_confidence(indicators, patterns, trend, signals)
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'indicators': indicators,
                'patterns': patterns,
                'levels': levels,
                'trend': trend,
                'volatility': volatility,
                'signals': signals,
                'overall_confidence': confidence,
                'recommendation': self._get_recommendation(signals, confidence)
            }
            
        except Exception as e:
            logger.error(f"Failed to perform analysis: {e}")
            return {}
    
    async def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators using pandas-ta wrapper"""
        try:
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            
            # Moving Averages
            ema_fast = self.ta.ema(close, length=self.config['ema_fast'])
            ema_slow = self.ta.ema(close, length=self.config['ema_slow'])
            sma_20 = self.ta.sma(close, length=20)
            sma_50 = self.ta.sma(close, length=50)
            sma_200 = self.ta.sma(close, length=200)
            
            # Momentum Indicators
            rsi = self.ta.rsi(close, length=self.config['rsi_period'])
            macd_data = self.ta.macd(close)
            stoch_data = self.ta.stoch(high, low, close)
            
            # Volatility Indicators
            atr = self.ta.atr(high, low, close, length=self.config['atr_period'])
            bb_data = self.ta.bbands(close, length=self.config['bb_period'], std=self.config['bb_std'])
            
            # Trend Indicators
            adx = self.ta.adx(high, low, close)
            
            return {
                'ema_fast': float(ema_fast.iloc[-1]) if not pd.isna(ema_fast.iloc[-1]) else None,
                'ema_slow': float(ema_slow.iloc[-1]) if not pd.isna(ema_slow.iloc[-1]) else None,
                'sma_20': float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else None,
                'sma_50': float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else None,
                'sma_200': float(sma_200.iloc[-1]) if not pd.isna(sma_200.iloc[-1]) else None,
                'rsi': float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None,
                'macd': float(macd_data['MACD_12_26_9'].iloc[-1]) if 'MACD_12_26_9' in macd_data.columns and not pd.isna(macd_data['MACD_12_26_9'].iloc[-1]) else None,
                'macd_signal': float(macd_data['MACDs_12_26_9'].iloc[-1]) if 'MACDs_12_26_9' in macd_data.columns and not pd.isna(macd_data['MACDs_12_26_9'].iloc[-1]) else None,
                'macd_histogram': float(macd_data['MACDh_12_26_9'].iloc[-1]) if 'MACDh_12_26_9' in macd_data.columns and not pd.isna(macd_data['MACDh_12_26_9'].iloc[-1]) else None,
                'stoch_k': float(stoch_data['STOCHk_14_3_3'].iloc[-1]) if 'STOCHk_14_3_3' in stoch_data.columns and not pd.isna(stoch_data['STOCHk_14_3_3'].iloc[-1]) else None,
                'stoch_d': float(stoch_data['STOCHd_14_3_3'].iloc[-1]) if 'STOCHd_14_3_3' in stoch_data.columns and not pd.isna(stoch_data['STOCHd_14_3_3'].iloc[-1]) else None,
                'atr': float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else None,
                'bb_upper': float(bb_data['BBU_20_2.0'].iloc[-1]) if 'BBU_20_2.0' in bb_data.columns and not pd.isna(bb_data['BBU_20_2.0'].iloc[-1]) else None,
                'bb_middle': float(bb_data['BBM_20_2.0'].iloc[-1]) if 'BBM_20_2.0' in bb_data.columns and not pd.isna(bb_data['BBM_20_2.0'].iloc[-1]) else None,
                'bb_lower': float(bb_data['BBL_20_2.0'].iloc[-1]) if 'BBL_20_2.0' in bb_data.columns and not pd.isna(bb_data['BBL_20_2.0'].iloc[-1]) else None,
                'adx': float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else None,
                'current_price': float(close.iloc[-1])
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate indicators: {e}")
            return {}
    
    async def _detect_patterns(self, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Detect chart patterns and structure breaks"""
        try:
            patterns = {
                'trend_break': False,
                'support_break': False,
                'resistance_break': False,
                'liquidity_sweep': False,
                'double_top': False,
                'double_bottom': False,
                'head_shoulders': False,
                'triangle': False
            }
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Trend break detection using EMA crossover
            if indicators.get('ema_fast') and indicators.get('ema_slow'):
                if len(close) >= 2:
                    ema_fast_prev = self.ta.ema(df['close'][:-1], length=self.config['ema_fast']).iloc[-1]
                    ema_slow_prev = self.ta.ema(df['close'][:-1], length=self.config['ema_slow']).iloc[-1]
                    
                    # Check for EMA crossover
                    if (ema_fast_prev <= ema_slow_prev and indicators['ema_fast'] > indicators['ema_slow']) or \
                       (ema_fast_prev >= ema_slow_prev and indicators['ema_fast'] < indicators['ema_slow']):
                        patterns['trend_break'] = True
            
            # Support/Resistance break detection
            if len(df) >= 20:
                recent_highs = high[-20:]
                recent_lows = low[-20:]
                current_price = close[-1]
                
                # Resistance break
                resistance_level = np.max(recent_highs[:-1])
                if current_price > resistance_level * 1.001:  # 0.1% buffer
                    patterns['resistance_break'] = True
                
                # Support break
                support_level = np.min(recent_lows[:-1])
                if current_price < support_level * 0.999:  # 0.1% buffer
                    patterns['support_break'] = True
            
            # Simple pattern detection based on price action
            if len(df) >= 10:
                recent_closes = close[-10:]
                
                # Double top pattern (simplified)
                if len(recent_closes) >= 5:
                    peaks = []
                    for i in range(1, len(recent_closes) - 1):
                        if recent_closes[i] > recent_closes[i-1] and recent_closes[i] > recent_closes[i+1]:
                            peaks.append(recent_closes[i])
                    
                    if len(peaks) >= 2:
                        if abs(peaks[-1] - peaks[-2]) / peaks[-1] < 0.02:  # Within 2%
                            patterns['double_top'] = True
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to detect patterns: {e}")
            return {}
    
    async def _calculate_levels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate support and resistance levels"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Calculate pivot points
            if len(df) >= 3:
                pivot = (high[-2] + low[-2] + close[-2]) / 3
                r1 = 2 * pivot - low[-2]
                s1 = 2 * pivot - high[-2]
                r2 = pivot + (high[-2] - low[-2])
                s2 = pivot - (high[-2] - low[-2])
            else:
                pivot = r1 = s1 = r2 = s2 = close[-1]
            
            # Calculate recent support/resistance
            lookback = min(50, len(df))
            recent_highs = high[-lookback:]
            recent_lows = low[-lookback:]
            
            resistance = np.percentile(recent_highs, 95)
            support = np.percentile(recent_lows, 5)
            
            return {
                'pivot': float(pivot),
                'resistance_1': float(r1),
                'support_1': float(s1),
                'resistance_2': float(r2),
                'support_2': float(s2),
                'dynamic_resistance': float(resistance),
                'dynamic_support': float(support)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate levels: {e}")
            return {}
    
    async def _determine_trend(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Determine overall trend direction and strength"""
        try:
            trend_signals = []
            
            # EMA trend
            if indicators.get('ema_fast') and indicators.get('ema_slow'):
                if indicators['ema_fast'] > indicators['ema_slow']:
                    trend_signals.append(1)
                else:
                    trend_signals.append(-1)
            
            # SMA trend
            if indicators.get('sma_20') and indicators.get('sma_50'):
                if indicators['sma_20'] > indicators['sma_50']:
                    trend_signals.append(1)
                else:
                    trend_signals.append(-1)
            
            # Price vs SMA
            if indicators.get('current_price') and indicators.get('sma_20'):
                if indicators['current_price'] > indicators['sma_20']:
                    trend_signals.append(1)
                else:
                    trend_signals.append(-1)
            
            # ADX for trend strength
            trend_strength = 'weak'
            if indicators.get('adx'):
                if indicators['adx'] > 25:
                    trend_strength = 'strong'
                elif indicators['adx'] > 15:
                    trend_strength = 'moderate'
            
            # Overall trend
            if trend_signals:
                avg_signal = sum(trend_signals) / len(trend_signals)
                if avg_signal > 0.3:
                    direction = 'bullish'
                elif avg_signal < -0.3:
                    direction = 'bearish'
                else:
                    direction = 'sideways'
            else:
                direction = 'unknown'
            
            return {
                'direction': direction,
                'strength': trend_strength,
                'signal_score': sum(trend_signals) if trend_signals else 0,
                'adx': indicators.get('adx')
            }
            
        except Exception as e:
            logger.error(f"Failed to determine trend: {e}")
            return {}
    
    async def _calculate_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volatility metrics"""
        try:
            close = df['close']
            
            # Historical volatility
            returns = close.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # ATR-based volatility
            atr_volatility = None
            if len(df) >= 14:
                atr = self.ta.atr(df['high'], df['low'], df['close'])
                if not pd.isna(atr.iloc[-1]):
                    atr_volatility = float(atr.iloc[-1]) / close.iloc[-1]
            
            return {
                'historical_volatility': float(volatility),
                'atr_volatility': atr_volatility,
                'volatility_regime': 'high' if volatility > 0.3 else 'medium' if volatility > 0.15 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate volatility: {e}")
            return {}
    
    async def _generate_signals(
        self, 
        indicators: Dict[str, Any], 
        patterns: Dict[str, Any], 
        trend: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate trading signals based on analysis"""
        try:
            signals = {
                'buy_signals': [],
                'sell_signals': [],
                'overall_signal': 'neutral',
                'signal_strength': 0.0
            }
            
            signal_score = 0
            
            # RSI signals
            if indicators.get('rsi'):
                if indicators['rsi'] < 30:
                    signals['buy_signals'].append('rsi_oversold')
                    signal_score += 1
                elif indicators['rsi'] > 70:
                    signals['sell_signals'].append('rsi_overbought')
                    signal_score -= 1
            
            # MACD signals
            if indicators.get('macd') and indicators.get('macd_signal'):
                if indicators['macd'] > indicators['macd_signal']:
                    signals['buy_signals'].append('macd_bullish')
                    signal_score += 1
                else:
                    signals['sell_signals'].append('macd_bearish')
                    signal_score -= 1
            
            # Trend signals
            if trend.get('direction') == 'bullish':
                signals['buy_signals'].append('trend_bullish')
                signal_score += 1
            elif trend.get('direction') == 'bearish':
                signals['sell_signals'].append('trend_bearish')
                signal_score -= 1
            
            # Pattern signals
            if patterns.get('resistance_break'):
                signals['buy_signals'].append('resistance_break')
                signal_score += 1
            if patterns.get('support_break'):
                signals['sell_signals'].append('support_break')
                signal_score -= 1
            
            # Overall signal
            if signal_score > 1:
                signals['overall_signal'] = 'buy'
            elif signal_score < -1:
                signals['overall_signal'] = 'sell'
            
            signals['signal_strength'] = abs(signal_score) / 5.0  # Normalize to 0-1
            
            return signals
            
        except Exception as e:
            logger.error(f"Failed to generate signals: {e}")
            return {}
    
    async def _calculate_confidence(
        self, 
        indicators: Dict[str, Any], 
        patterns: Dict[str, Any], 
        trend: Dict[str, Any], 
        signals: Dict[str, Any]
    ) -> float:
        """Calculate overall confidence in the analysis"""
        try:
            confidence_factors = []
            
            # Data quality
            non_null_indicators = sum(1 for v in indicators.values() if v is not None)
            total_indicators = len(indicators)
            if total_indicators > 0:
                data_quality = non_null_indicators / total_indicators
                confidence_factors.append(data_quality)
            
            # Trend strength
            if trend.get('strength') == 'strong':
                confidence_factors.append(0.9)
            elif trend.get('strength') == 'moderate':
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.5)
            
            # Signal consistency
            signal_strength = signals.get('signal_strength', 0)
            confidence_factors.append(signal_strength)
            
            # Pattern confirmation
            pattern_count = sum(1 for v in patterns.values() if v)
            pattern_confidence = min(pattern_count / 3.0, 1.0)
            confidence_factors.append(pattern_confidence)
            
            return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
            
        except Exception as e:
            logger.error(f"Failed to calculate confidence: {e}")
            return 0.5
    
    def _get_recommendation(self, signals: Dict[str, Any], confidence: float) -> str:
        """Get trading recommendation based on signals and confidence"""
        try:
            if confidence < 0.4:
                return 'hold'
            
            signal = signals.get('overall_signal', 'neutral')
            strength = signals.get('signal_strength', 0)
            
            if signal == 'buy' and strength > 0.6:
                return 'strong_buy'
            elif signal == 'buy':
                return 'buy'
            elif signal == 'sell' and strength > 0.6:
                return 'strong_sell'
            elif signal == 'sell':
                return 'sell'
            else:
                return 'hold'
                
        except Exception as e:
            logger.error(f"Failed to get recommendation: {e}")
            return 'hold'
    
    async def get_stop_loss_take_profit(
        self, 
        symbol: str, 
        entry_price: float, 
        direction: str
    ) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        try:
            # Get recent analysis for ATR
            analysis = await self.analyze_symbol(symbol)
            if not analysis or not analysis.get('indicators'):
                # Default to 2% stop loss, 4% take profit
                if direction.lower() == 'buy':
                    stop_loss = entry_price * 0.98
                    take_profit = entry_price * 1.04
                else:
                    stop_loss = entry_price * 1.02
                    take_profit = entry_price * 0.96
                return stop_loss, take_profit
            
            atr = analysis['indicators'].get('atr')
            if atr:
                # Use ATR-based levels
                atr_multiplier = 2.0
                if direction.lower() == 'buy':
                    stop_loss = entry_price - (atr * atr_multiplier)
                    take_profit = entry_price + (atr * atr_multiplier * 2)
                else:
                    stop_loss = entry_price + (atr * atr_multiplier)
                    take_profit = entry_price - (atr * atr_multiplier * 2)
            else:
                # Fallback to percentage-based
                if direction.lower() == 'buy':
                    stop_loss = entry_price * 0.98
                    take_profit = entry_price * 1.04
                else:
                    stop_loss = entry_price * 1.02
                    take_profit = entry_price * 0.96
            
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Failed to calculate stop loss/take profit: {e}")
            # Default fallback
            if direction.lower() == 'buy':
                return entry_price * 0.98, entry_price * 1.04
            else:
                return entry_price * 1.02, entry_price * 0.96
    
    async def start_continuous_analysis(self, symbols: List[str], interval: int = 300) -> None:
        """Start continuous analysis for multiple symbols"""
        try:
            self.running = True
            logger.info(f"Starting continuous analysis for {len(symbols)} symbols")
            
            # Create analysis tasks for each symbol
            tasks = []
            for symbol in symbols:
                task = asyncio.create_task(self._analysis_worker(symbol, interval))
                tasks.append(task)
            
            # Wait for all tasks
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Failed to start continuous analysis: {e}")
    
    async def _analysis_worker(self, symbol: str, interval: int) -> None:
        """Worker for continuous analysis of a single symbol"""
        try:
            while self.running:
                await self.analyze_symbol(symbol)
                await asyncio.sleep(interval)
                
        except Exception as e:
            logger.error(f"Analysis worker failed for {symbol}: {e}")
    
    async def stop_continuous_analysis(self) -> None:
        """Stop continuous analysis"""
        try:
            self.running = False
            logger.info("Stopped continuous analysis")
            
        except Exception as e:
            logger.error(f"Failed to stop continuous analysis: {e}")


# Global instance
_chart_agent: Optional[ChartAnalysisAgent] = None


async def get_chart_agent(orchestrator: TraeOrchestrator) -> ChartAnalysisAgent:
    """Get or create chart agent instance"""
    global _chart_agent
    if _chart_agent is None:
        _chart_agent = ChartAnalysisAgent(orchestrator)
    return _chart_agent