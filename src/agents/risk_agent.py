"""
Risk Management Agent with position sizing and drawdown protection
Migrated for global agent hub with orchestrator integration
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from enum import Enum

from ..core.trae_orchestrator import TraeOrchestrator, Event, EventType

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels for position sizing"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class PositionSize:
    """Position size calculation result"""
    symbol: str
    quantity: float
    risk_amount: float
    risk_percentage: float
    stop_loss: float
    take_profit: float
    max_loss: float
    position_value: float
    leverage: float = 1.0


@dataclass
class RiskMetrics:
    """Risk metrics for portfolio"""
    total_equity: float
    available_equity: float
    used_margin: float
    daily_pnl: float
    daily_drawdown: float
    max_drawdown: float
    risk_per_trade: float
    total_exposure: float
    open_positions: int
    risk_level: RiskLevel


class RiskManagementAgent:
    """Risk management agent with comprehensive risk controls"""
    
    def __init__(self, orchestrator: TraeOrchestrator):
        self.orchestrator = orchestrator
        self.running = False
        
        # Configuration
        self.config = {
            'max_risk_per_trade': 0.02,  # 2% per trade
            'max_daily_drawdown': 0.05,  # 5% daily drawdown limit
            'max_total_drawdown': 0.20,  # 20% total drawdown limit
            'max_open_positions': 10,
            'initial_equity': 10000.0,
            'risk_reward_ratio': 2.0,
            'correlation_limit': 0.7
        }
        
        # Risk tracking
        self.daily_trades: List[Dict[str, Any]] = []
        self.daily_pnl: float = 0.0
        self.max_drawdown: float = 0.0
        self.peak_equity: float = 0.0
        self.current_equity: float = 0.0
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        
        # Risk limits
        self.trading_halted: bool = False
        self.halt_reason: Optional[str] = None
        self.last_reset: datetime = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Initialize with default equity
        self.current_equity = self.config['initial_equity']
        self.peak_equity = self.current_equity
    
    async def calculate_position_size(
        self, 
        symbol: str, 
        entry_price: float, 
        stop_loss: float, 
        direction: str,
        risk_multiplier: float = 1.0
    ) -> Optional[PositionSize]:
        """Calculate position size based on risk management rules"""
        try:
            # Check if trading is halted
            if self.trading_halted:
                logger.warning(f"Trading halted: {self.halt_reason}")
                return None
            
            # Reset daily tracking if new day
            await self._check_daily_reset()
            
            # Get current risk metrics
            risk_metrics = await self.get_risk_metrics()
            
            # Check daily drawdown limit
            if risk_metrics.daily_drawdown >= self.config['max_daily_drawdown']:
                await self._halt_trading("Daily drawdown limit exceeded")
                return None
            
            # Check maximum drawdown limit
            if risk_metrics.max_drawdown >= self.config['max_total_drawdown']:
                await self._halt_trading("Maximum drawdown limit exceeded")
                return None
            
            # Check maximum open positions
            if len(self.open_positions) >= self.config['max_open_positions']:
                logger.warning("Maximum open positions reached")
                return None
            
            # Calculate risk amount
            base_risk_amount = self.current_equity * self.config['max_risk_per_trade']
            adjusted_risk_amount = base_risk_amount * risk_multiplier
            
            # Adjust for risk level
            if risk_metrics.risk_level == RiskLevel.HIGH:
                adjusted_risk_amount *= 0.5
            elif risk_metrics.risk_level == RiskLevel.EXTREME:
                adjusted_risk_amount *= 0.25
            
            # Calculate position size
            price_diff = abs(entry_price - stop_loss)
            if price_diff == 0:
                logger.warning("Entry price equals stop loss")
                return None
            
            # Calculate quantity
            quantity = adjusted_risk_amount / price_diff
            
            # Round quantity to appropriate precision
            quantity = self._round_quantity(symbol, quantity)
            
            # Calculate take profit
            take_profit = await self._calculate_take_profit(entry_price, stop_loss, direction)
            
            # Calculate position value and max loss
            position_value = quantity * entry_price
            max_loss = quantity * price_diff
            
            position_size = PositionSize(
                symbol=symbol,
                quantity=quantity,
                risk_amount=adjusted_risk_amount,
                risk_percentage=adjusted_risk_amount / self.current_equity,
                stop_loss=stop_loss,
                take_profit=take_profit,
                max_loss=max_loss,
                position_value=position_value
            )
            
            # Publish risk calculation event
            await self.orchestrator.publish_event(Event(
                type=EventType.RISK_CALCULATION,
                data={
                    'symbol': symbol,
                    'position_size': {
                        'quantity': quantity,
                        'risk_amount': adjusted_risk_amount,
                        'risk_percentage': adjusted_risk_amount / self.current_equity,
                        'max_loss': max_loss
                    },
                    'risk_metrics': {
                        'current_equity': self.current_equity,
                        'daily_drawdown': risk_metrics.daily_drawdown,
                        'max_drawdown': risk_metrics.max_drawdown,
                        'risk_level': risk_metrics.risk_level.value
                    },
                    'timestamp': datetime.now().isoformat()
                },
                source='RiskAgent'
            ))
            
            return position_size
            
        except Exception as e:
            logger.error(f"Failed to calculate position size for {symbol}: {e}")
            return None
    
    def _round_quantity(self, symbol: str, quantity: float) -> float:
        """Round quantity to appropriate precision based on symbol"""
        try:
            # Default rounding rules
            if 'BTC' in symbol.upper():
                return round(quantity, 6)
            elif 'ETH' in symbol.upper():
                return round(quantity, 4)
            elif any(crypto in symbol.upper() for crypto in ['USDT', 'USDC', 'USD']):
                return round(quantity, 2)
            elif any(fx in symbol.upper() for fx in ['EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD']):
                return round(quantity, 0)  # Forex lots
            else:
                return round(quantity, 2)  # Stocks
                
        except Exception as e:
            logger.error(f"Failed to round quantity: {e}")
            return round(quantity, 2)
    
    async def _calculate_take_profit(self, entry_price: float, stop_loss: float, direction: str) -> float:
        """Calculate take profit based on risk-reward ratio"""
        try:
            risk_distance = abs(entry_price - stop_loss)
            reward_distance = risk_distance * self.config['risk_reward_ratio']
            
            if direction.lower() == 'buy':
                return entry_price + reward_distance
            else:
                return entry_price - reward_distance
                
        except Exception as e:
            logger.error(f"Failed to calculate take profit: {e}")
            return entry_price
    
    async def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics"""
        try:
            # Calculate daily P&L
            daily_pnl = sum(trade.get('pnl', 0) for trade in self.daily_trades)
            
            # Calculate daily drawdown
            daily_start_equity = self.current_equity - daily_pnl
            daily_drawdown = abs(min(0, daily_pnl)) / daily_start_equity if daily_start_equity > 0 else 0
            
            # Update peak equity and max drawdown
            if self.current_equity > self.peak_equity:
                self.peak_equity = self.current_equity
            
            current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity if self.peak_equity > 0 else 0
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            # Calculate total exposure
            total_exposure = sum(pos.get('position_value', 0) for pos in self.open_positions.values())
            
            # Calculate used margin (simplified)
            used_margin = total_exposure * 0.1  # Assume 10:1 leverage
            available_equity = max(0, self.current_equity - used_margin)
            
            # Determine risk level
            risk_level = self._determine_risk_level(daily_drawdown, current_drawdown, len(self.open_positions))
            
            return RiskMetrics(
                total_equity=self.current_equity,
                available_equity=available_equity,
                used_margin=used_margin,
                daily_pnl=daily_pnl,
                daily_drawdown=daily_drawdown,
                max_drawdown=current_drawdown,
                risk_per_trade=self.config['max_risk_per_trade'],
                total_exposure=total_exposure,
                open_positions=len(self.open_positions),
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"Failed to get risk metrics: {e}")
            return RiskMetrics(
                total_equity=self.current_equity,
                available_equity=self.current_equity,
                used_margin=0,
                daily_pnl=0,
                daily_drawdown=0,
                max_drawdown=self.max_drawdown,
                risk_per_trade=self.config['max_risk_per_trade'],
                total_exposure=0,
                open_positions=0,
                risk_level=RiskLevel.LOW
            )
    
    def _determine_risk_level(self, daily_drawdown: float, max_drawdown: float, open_positions: int) -> RiskLevel:
        """Determine current risk level based on metrics"""
        try:
            risk_score = 0
            
            # Daily drawdown factor
            if daily_drawdown > 0.04:  # 4%
                risk_score += 3
            elif daily_drawdown > 0.02:  # 2%
                risk_score += 2
            elif daily_drawdown > 0.01:  # 1%
                risk_score += 1
            
            # Max drawdown factor
            if max_drawdown > 0.15:  # 15%
                risk_score += 3
            elif max_drawdown > 0.10:  # 10%
                risk_score += 2
            elif max_drawdown > 0.05:  # 5%
                risk_score += 1
            
            # Position count factor
            if open_positions > 8:
                risk_score += 2
            elif open_positions > 5:
                risk_score += 1
            
            # Determine risk level
            if risk_score >= 6:
                return RiskLevel.EXTREME
            elif risk_score >= 4:
                return RiskLevel.HIGH
            elif risk_score >= 2:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
                
        except Exception as e:
            logger.error(f"Failed to determine risk level: {e}")
            return RiskLevel.MEDIUM
    
    async def validate_trade(self, trade_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate if a trade should be executed based on risk rules"""
        try:
            symbol = trade_data.get('symbol')
            direction = trade_data.get('direction')
            entry_price = trade_data.get('entry_price')
            stop_loss = trade_data.get('stop_loss')
            
            if not all([symbol, direction, entry_price, stop_loss]):
                return False, "Missing required trade data"
            
            # Check if trading is halted
            if self.trading_halted:
                return False, f"Trading halted: {self.halt_reason}"
            
            # Check correlation with existing positions
            correlation_risk = await self._check_correlation_risk(symbol, direction)
            if correlation_risk:
                return False, "High correlation risk with existing positions"
            
            # Calculate position size
            position_size = await self.calculate_position_size(
                symbol, entry_price, stop_loss, direction
            )
            
            if position_size is None:
                return False, "Position size calculation failed"
            
            # Check minimum position size
            if position_size.quantity <= 0:
                return False, "Position size too small"
            
            # Check maximum position value
            max_position_value = self.current_equity * 0.2  # 20% of equity
            if position_size.position_value > max_position_value:
                return False, "Position value exceeds maximum allowed"
            
            return True, "Trade validated successfully"
            
        except Exception as e:
            logger.error(f"Failed to validate trade: {e}")
            return False, f"Validation error: {str(e)}"
    
    async def _check_correlation_risk(self, symbol: str, direction: str) -> bool:
        """Check if new position would create excessive correlation risk"""
        try:
            # Simplified correlation check
            # In a real implementation, this would use actual correlation data
            
            similar_positions = 0
            for pos_symbol, pos_data in self.open_positions.items():
                if pos_data.get('direction') == direction:
                    # Check for similar asset classes
                    if self._are_correlated(symbol, pos_symbol):
                        similar_positions += 1
            
            # Limit correlated positions
            return similar_positions >= 3
            
        except Exception as e:
            logger.error(f"Failed to check correlation risk: {e}")
            return False
    
    def _are_correlated(self, symbol1: str, symbol2: str) -> bool:
        """Simple correlation check based on symbol patterns"""
        try:
            # Crypto correlation
            crypto_pairs = ['BTC', 'ETH', 'USDT', 'USDC']
            if any(crypto in symbol1.upper() for crypto in crypto_pairs) and \
               any(crypto in symbol2.upper() for crypto in crypto_pairs):
                return True
            
            # Forex major pairs correlation
            major_pairs = ['EUR', 'GBP', 'USD', 'JPY']
            if any(pair in symbol1.upper() for pair in major_pairs) and \
               any(pair in symbol2.upper() for pair in major_pairs):
                return True
            
            # Same base symbol
            if symbol1.split('/')[0] == symbol2.split('/')[0]:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check correlation: {e}")
            return False
    
    async def add_position(self, position_data: Dict[str, Any]) -> bool:
        """Add a new position to tracking"""
        try:
            symbol = position_data.get('symbol')
            if not symbol:
                return False
            
            self.open_positions[symbol] = {
                'symbol': symbol,
                'direction': position_data.get('direction'),
                'quantity': position_data.get('quantity'),
                'entry_price': position_data.get('entry_price'),
                'stop_loss': position_data.get('stop_loss'),
                'take_profit': position_data.get('take_profit'),
                'position_value': position_data.get('position_value'),
                'unrealized_pnl': 0.0,
                'entry_time': datetime.now().isoformat()
            }
            
            # Publish position added event
            await self.orchestrator.publish_event(Event(
                type=EventType.POSITION_OPENED,
                data={
                    'symbol': symbol,
                    'position_data': self.open_positions[symbol],
                    'total_positions': len(self.open_positions),
                    'timestamp': datetime.now().isoformat()
                },
                source='RiskAgent'
            ))
            
            await self._publish_risk_update()
            return True
            
        except Exception as e:
            logger.error(f"Failed to add position: {e}")
            return False
    
    async def close_position(self, symbol: str, exit_price: float, reason: str = "manual") -> bool:
        """Close a position and update P&L"""
        try:
            if symbol not in self.open_positions:
                logger.warning(f"Position {symbol} not found")
                return False
            
            position = self.open_positions[symbol]
            
            # Calculate P&L
            entry_price = position['entry_price']
            quantity = position['quantity']
            direction = position['direction']
            
            if direction.lower() == 'buy':
                pnl = (exit_price - entry_price) * quantity
            else:
                pnl = (entry_price - exit_price) * quantity
            
            # Update equity
            self.current_equity += pnl
            self.daily_pnl += pnl
            
            # Record trade
            trade_record = {
                'symbol': symbol,
                'direction': direction,
                'quantity': quantity,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'reason': reason,
                'entry_time': position.get('entry_time'),
                'exit_time': datetime.now().isoformat()
            }
            
            self.daily_trades.append(trade_record)
            
            # Remove from open positions
            del self.open_positions[symbol]
            
            # Publish position closed event
            await self.orchestrator.publish_event(Event(
                type=EventType.POSITION_CLOSED,
                data={
                    'symbol': symbol,
                    'trade_record': trade_record,
                    'current_equity': self.current_equity,
                    'daily_pnl': self.daily_pnl,
                    'timestamp': datetime.now().isoformat()
                },
                source='RiskAgent'
            ))
            
            await self._publish_risk_update()
            return True
            
        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
            return False
    
    async def update_position_pnl(self, symbol: str, current_price: float) -> None:
        """Update unrealized P&L for a position"""
        try:
            if symbol not in self.open_positions:
                return
            
            position = self.open_positions[symbol]
            entry_price = position['entry_price']
            quantity = position['quantity']
            direction = position['direction']
            
            if direction.lower() == 'buy':
                unrealized_pnl = (current_price - entry_price) * quantity
            else:
                unrealized_pnl = (entry_price - current_price) * quantity
            
            position['unrealized_pnl'] = unrealized_pnl
            position['current_price'] = current_price
            
        except Exception as e:
            logger.error(f"Failed to update P&L for {symbol}: {e}")
    
    async def check_stop_loss_take_profit(self, symbol: str, current_price: float) -> Optional[str]:
        """Check if stop loss or take profit should be triggered"""
        try:
            if symbol not in self.open_positions:
                return None
            
            position = self.open_positions[symbol]
            direction = position['direction']
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']
            
            if direction.lower() == 'buy':
                if current_price <= stop_loss:
                    return 'stop_loss'
                elif current_price >= take_profit:
                    return 'take_profit'
            else:
                if current_price >= stop_loss:
                    return 'stop_loss'
                elif current_price <= take_profit:
                    return 'take_profit'
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to check stop loss/take profit for {symbol}: {e}")
            return None
    
    async def _check_daily_reset(self) -> None:
        """Check if daily tracking should be reset"""
        try:
            now = datetime.now()
            current_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            
            if current_day > self.last_reset:
                # Reset daily tracking
                self.daily_trades = []
                self.daily_pnl = 0.0
                self.last_reset = current_day
                
                # Resume trading if halted due to daily limits
                if self.trading_halted and 'daily' in self.halt_reason.lower():
                    await self.resume_trading()
                
                logger.info("Daily risk tracking reset")
                
        except Exception as e:
            logger.error(f"Failed to check daily reset: {e}")
    
    async def _halt_trading(self, reason: str) -> None:
        """Halt all trading due to risk limits"""
        try:
            self.trading_halted = True
            self.halt_reason = reason
            
            logger.warning(f"Trading halted: {reason}")
            
            # Publish trading halt event
            await self.orchestrator.publish_event(Event(
                type=EventType.TRADING_HALTED,
                data={
                    'reason': reason,
                    'current_equity': self.current_equity,
                    'daily_pnl': self.daily_pnl,
                    'max_drawdown': self.max_drawdown,
                    'timestamp': datetime.now().isoformat()
                },
                source='RiskAgent'
            ))
            
        except Exception as e:
            logger.error(f"Failed to halt trading: {e}")
    
    async def resume_trading(self) -> bool:
        """Resume trading after halt"""
        try:
            # Check if conditions allow resuming
            risk_metrics = await self.get_risk_metrics()
            
            if risk_metrics.daily_drawdown >= self.config['max_daily_drawdown']:
                logger.warning("Cannot resume: Daily drawdown limit still exceeded")
                return False
            
            if risk_metrics.max_drawdown >= self.config['max_total_drawdown']:
                logger.warning("Cannot resume: Maximum drawdown limit still exceeded")
                return False
            
            self.trading_halted = False
            self.halt_reason = None
            
            logger.info("Trading resumed")
            
            # Publish trading resumed event
            await self.orchestrator.publish_event(Event(
                type=EventType.TRADING_RESUMED,
                data={
                    'current_equity': self.current_equity,
                    'risk_metrics': {
                        'daily_drawdown': risk_metrics.daily_drawdown,
                        'max_drawdown': risk_metrics.max_drawdown,
                        'risk_level': risk_metrics.risk_level.value
                    },
                    'timestamp': datetime.now().isoformat()
                },
                source='RiskAgent'
            ))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume trading: {e}")
            return False
    
    async def _publish_risk_update(self) -> None:
        """Publish risk metrics update"""
        try:
            risk_metrics = await self.get_risk_metrics()
            
            await self.orchestrator.publish_event(Event(
                type=EventType.RISK_UPDATE,
                data={
                    'risk_metrics': {
                        'total_equity': risk_metrics.total_equity,
                        'available_equity': risk_metrics.available_equity,
                        'daily_pnl': risk_metrics.daily_pnl,
                        'daily_drawdown': risk_metrics.daily_drawdown,
                        'max_drawdown': risk_metrics.max_drawdown,
                        'total_exposure': risk_metrics.total_exposure,
                        'open_positions': risk_metrics.open_positions,
                        'risk_level': risk_metrics.risk_level.value
                    },
                    'trading_halted': self.trading_halted,
                    'halt_reason': self.halt_reason,
                    'timestamp': datetime.now().isoformat()
                },
                source='RiskAgent'
            ))
            
        except Exception as e:
            logger.error(f"Failed to publish risk update: {e}")
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        try:
            risk_metrics = await self.get_risk_metrics()
            
            # Calculate position summaries
            position_summaries = []
            total_unrealized_pnl = 0
            
            for symbol, position in self.open_positions.items():
                unrealized_pnl = position.get('unrealized_pnl', 0)
                total_unrealized_pnl += unrealized_pnl
                
                position_summaries.append({
                    'symbol': symbol,
                    'direction': position['direction'],
                    'quantity': position['quantity'],
                    'entry_price': position['entry_price'],
                    'current_price': position.get('current_price'),
                    'unrealized_pnl': unrealized_pnl,
                    'stop_loss': position['stop_loss'],
                    'take_profit': position['take_profit']
                })
            
            return {
                'equity': {
                    'total': self.current_equity,
                    'available': risk_metrics.available_equity,
                    'peak': self.peak_equity
                },
                'pnl': {
                    'daily_realized': self.daily_pnl,
                    'total_unrealized': total_unrealized_pnl,
                    'total_combined': self.daily_pnl + total_unrealized_pnl
                },
                'drawdown': {
                    'daily': risk_metrics.daily_drawdown,
                    'maximum': risk_metrics.max_drawdown
                },
                'positions': {
                    'count': len(self.open_positions),
                    'total_exposure': risk_metrics.total_exposure,
                    'details': position_summaries
                },
                'risk': {
                    'level': risk_metrics.risk_level.value,
                    'trading_halted': self.trading_halted,
                    'halt_reason': self.halt_reason
                },
                'trades_today': len(self.daily_trades),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get portfolio summary: {e}")
            return {}


# Global instance
_risk_agent: Optional[RiskManagementAgent] = None


async def get_risk_agent(orchestrator: TraeOrchestrator) -> RiskManagementAgent:
    """Get or create risk agent instance"""
    global _risk_agent
    if _risk_agent is None:
        _risk_agent = RiskManagementAgent(orchestrator)
    return _risk_agent