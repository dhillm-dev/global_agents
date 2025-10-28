"""
Execution Agent for publishing trade orders via Trae Orchestrator
Migrated for global agent hub with orchestrator integration
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import uuid
from enum import Enum

from ..core.trae_orchestrator import TraeOrchestrator, Event, EventType
from .risk_agent import get_risk_agent

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(Enum):
    """Order sides"""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "PENDING"
    SENT = "SENT"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class ExecutionAgent:
    """Execution agent for managing trade orders via orchestrator events"""
    
    def __init__(self, orchestrator: TraeOrchestrator):
        self.orchestrator = orchestrator
        self.running = False
        self.pending_orders: Dict[str, Dict[str, Any]] = {}
        self.order_history: List[Dict[str, Any]] = []
        self.execution_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'cancelled_orders': 0
        }
    
    async def execute_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Execute a trade order by emitting an orchestrator event"""
        try:
            # Validate inputs
            if not symbol or quantity <= 0:
                logger.error("Invalid trade parameters")
                return None
            
            # Normalize side
            side = side.upper()
            if side not in [OrderSide.BUY.value, OrderSide.SELL.value]:
                logger.error(f"Invalid order side: {side}")
                return None
            
            # Generate order ID
            order_id = str(uuid.uuid4())
            
            # Get risk agent for validation
            risk_agent = await get_risk_agent(self.orchestrator)
            
            # Prepare trade data for validation
            trade_data = {
                'symbol': symbol,
                'quantity': quantity,
                'entry_price': price or 0.0,
                'side': side,
                'order_type': order_type,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            
            # Validate trade with risk management
            is_valid, validation_message = await risk_agent.validate_trade(trade_data)
            if not is_valid:
                logger.warning(f"Trade validation failed: {validation_message}")
                await self._record_failed_order(order_id, symbol, side, quantity, validation_message)
                return None
            
            # Create order object
            order = await self._create_order(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata=metadata or {}
            )
            
            # Emit trade order request event
            await self.orchestrator.emit_event(Event(
                type=EventType.TRADE_ORDER_REQUEST,
                source='ExecutionAgent',
                data=order,
                timestamp=datetime.now(timezone.utc),
                priority=3
            ))
            
            # Track pending order
            self.pending_orders[order_id] = order
            self.execution_stats['total_orders'] += 1
            
            logger.info(f"Trade order emitted: {order_id} {side} {quantity} {symbol}")
            return order_id
            
        except Exception as e:
            logger.error(f"Failed to execute trade: {e}")
            return None
    
    async def _create_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create an order object"""
        timestamp = datetime.now(timezone.utc).isoformat()
        return {
            'order_id': order_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'order_type': order_type,
            'price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'status': OrderStatus.PENDING.value,
            'created_at': timestamp,
            'updated_at': timestamp,
            'metadata': metadata or {}
        }
    
    async def cancel_order(self, order_id: str, reason: str = "manual") -> bool:
        """Cancel a pending order"""
        try:
            if order_id not in self.pending_orders:
                logger.warning(f"Order not found: {order_id}")
                return False
            
            order = self.pending_orders[order_id]
            order['status'] = OrderStatus.CANCELLED.value
            order['updated_at'] = datetime.now(timezone.utc).isoformat()
            order['cancel_reason'] = reason
            
            # Emit order update
            await self.orchestrator.emit_event(Event(
                type=EventType.ORDER_EXECUTED,
                source='ExecutionAgent',
                data={'order': order, 'status': 'CANCELLED'},
                timestamp=datetime.now(timezone.utc),
                priority=3
            ))
            
            # Update stats and history
            self.order_history.append(order)
            self.execution_stats['cancelled_orders'] += 1
            del self.pending_orders[order_id]
            
            logger.info(f"Order cancelled: {order_id} ({reason})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def handle_order_update(self, update_data: Dict[str, Any]) -> None:
        """Handle external order updates (e.g., from broker execution)"""
        try:
            order_id = update_data.get('order_id')
            status = update_data.get('status')
            
            if not order_id or order_id not in self.pending_orders:
                logger.warning("Unknown order update received")
                return
            
            order = self.pending_orders[order_id]
            order['status'] = status
            order['updated_at'] = datetime.now(timezone.utc).isoformat()
            order['fill_price'] = update_data.get('fill_price')
            order['fill_quantity'] = update_data.get('fill_quantity')
            order['broker_order_id'] = update_data.get('broker_order_id')
            
            # Update stats
            if status == OrderStatus.FILLED.value:
                self.execution_stats['successful_orders'] += 1
            elif status in [OrderStatus.REJECTED.value, OrderStatus.FAILED.value]:
                self.execution_stats['failed_orders'] += 1
            
            # Emit order executed event
            await self.orchestrator.emit_event(Event(
                type=EventType.ORDER_EXECUTED,
                source='ExecutionAgent',
                data={'order': order, 'status': status},
                timestamp=datetime.now(timezone.utc),
                priority=4
            ))
            
            # Move to history if terminal state
            if status in [OrderStatus.FILLED.value, OrderStatus.CANCELLED.value, OrderStatus.REJECTED.value, OrderStatus.FAILED.value]:
                self.order_history.append(order)
                del self.pending_orders[order_id]
            
        except Exception as e:
            logger.error(f"Error handling order update: {e}")
    
    async def _record_failed_order(self, order_id: str, symbol: str, side: str, quantity: float, reason: str) -> None:
        """Record a failed order attempt"""
        try:
            order = {
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'status': OrderStatus.FAILED.value,
                'reason': reason,
                'created_at': datetime.now(timezone.utc).isoformat()
            }
            self.order_history.append(order)
            self.execution_stats['failed_orders'] += 1
        except Exception as e:
            logger.error(f"Failed to record failed order: {e}")
    
    async def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        try:
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'pending_orders': len(self.pending_orders),
                'total_orders': self.execution_stats['total_orders'],
                'successful_orders': self.execution_stats['successful_orders'],
                'failed_orders': self.execution_stats['failed_orders'],
                'cancelled_orders': self.execution_stats['cancelled_orders']
            }
        except Exception as e:
            logger.error(f"Failed to get execution stats: {e}")
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'pending_orders': 0,
                'total_orders': 0,
                'successful_orders': 0,
                'failed_orders': 0,
                'cancelled_orders': 0,
                'error': str(e)
            }
    
    async def start_order_monitoring(self) -> None:
        """Start background monitoring (placeholder for future broker integration)"""
        try:
            if self.running:
                logger.warning("Order monitoring already running")
                return
            
            self.running = True
            logger.info("Order monitoring started")
        except Exception as e:
            logger.error(f"Error starting order monitoring: {e}")
    
    async def stop_order_monitoring(self) -> None:
        """Stop background monitoring"""
        try:
            self.running = False
            logger.info("Order monitoring stopped")
        except Exception as e:
            logger.error(f"Error stopping order monitoring: {e}")


# Global instance
_execution_agent: Optional[ExecutionAgent] = None


async def get_execution_agent(orchestrator: TraeOrchestrator) -> ExecutionAgent:
    """Get or create execution agent instance"""
    global _execution_agent
    if _execution_agent is None:
        _execution_agent = ExecutionAgent(orchestrator)
    return _execution_agent