"""
MetaAPI Broker Implementation
Provides trading functionality through MetaAPI cloud service for MetaTrader integration
"""
import asyncio
from metaapi_cloud_sdk import MetaApi
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import logging
from decimal import Decimal

from .base import (
    BrokerBase, Order, Position, AccountInfo, MarketData,
    OrderSide, OrderType, OrderStatus
)

logger = logging.getLogger(__name__)

class MetaAPIBroker(BrokerBase):
    """MetaAPI broker implementation for MetaTrader integration"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.token = config.get("token")
        self.account_id = config.get("account_id")
        self.application = config.get("application", "TradingAgents")
        self.domain = config.get("domain", "agiliumtrade.agiliumtrade.ai")
        
        self.api: Optional[MetaApi] = None
        self.account = None
        self.connection = None
    
    async def connect(self) -> bool:
        """Establish connection to MetaAPI"""
        try:
            # Initialize MetaAPI
            self.api = MetaApi(self.token, {
                'application': self.application,
                'domain': self.domain
            })
            
            # Get account
            self.account = await self.api.metatrader_account_api.get_account(self.account_id)
            
            # Wait for account to be deployed
            await self.account.wait_deployed()
            
            # Create connection
            self.connection = self.account.get_streaming_connection()
            await self.connection.connect()
            
            # Wait for synchronization
            await self.connection.wait_synchronized()
            
            self.is_connected = True
            logger.info(f"Connected to MetaAPI account {self.account_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MetaAPI: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Close connection to MetaAPI"""
        try:
            if self.connection:
                self.connection.close()
                self.connection = None
            self.account = None
            self.api = None
            self.is_connected = False
            logger.info("Disconnected from MetaAPI")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from MetaAPI: {e}")
            return False
    
    async def place_order(self, 
                         symbol: str, 
                         side: OrderSide, 
                         quantity: float, 
                         order_type: OrderType = OrderType.MARKET,
                         price: Optional[float] = None,
                         stop_price: Optional[float] = None,
                         **kwargs) -> Order:
        """Place a trading order through MetaAPI"""
        
        if not self.validate_order_params(symbol, side, quantity, order_type, price):
            raise ValueError("Invalid order parameters")
        
        try:
            # Convert parameters to MetaAPI format
            action_type = "ORDER_TYPE_BUY" if side == OrderSide.BUY else "ORDER_TYPE_SELL"
            
            # Build trade request
            trade_request = {
                'actionType': 'ORDER_TYPE_BUY_MARKET' if side == OrderSide.BUY and order_type == OrderType.MARKET 
                             else 'ORDER_TYPE_SELL_MARKET' if side == OrderSide.SELL and order_type == OrderType.MARKET
                             else 'ORDER_TYPE_BUY_LIMIT' if side == OrderSide.BUY and order_type == OrderType.LIMIT
                             else 'ORDER_TYPE_SELL_LIMIT',
                'symbol': symbol,
                'volume': quantity,
            }
            
            # Add price for limit orders
            if order_type == OrderType.LIMIT and price:
                trade_request['openPrice'] = price
            
            # Add stop loss and take profit if provided
            if kwargs.get('stop_loss'):
                trade_request['stopLoss'] = kwargs['stop_loss']
            if kwargs.get('take_profit'):
                trade_request['takeProfit'] = kwargs['take_profit']
            
            # Place the order
            result = await self.connection.create_market_order(trade_request)
            
            return Order(
                id=result.get('orderId', ''),
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity,
                price=price,
                status=OrderStatus.PENDING,  # Will be updated by events
                filled_quantity=0.0,
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"Failed to place order through MetaAPI: {e}")
            raise
    
    async def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        """Cancel an existing order"""
        try:
            await self.connection.cancel_order(order_id)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_order(self, order_id: str, symbol: str = None) -> Optional[Order]:
        """Get order details by ID"""
        try:
            orders = await self.connection.get_orders()
            for order_data in orders:
                if order_data.get('id') == order_id:
                    return self._convert_metaapi_order(order_data)
            return None
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None
    
    async def get_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all orders, optionally filtered by symbol"""
        try:
            orders_data = await self.connection.get_orders()
            orders = []
            
            for order_data in orders_data:
                if symbol and order_data.get('symbol') != symbol:
                    continue
                order = self._convert_metaapi_order(order_data)
                if order:
                    orders.append(order)
            
            return orders
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get all positions, optionally filtered by symbol"""
        try:
            positions_data = await self.connection.get_positions()
            positions = []
            
            for pos_data in positions_data:
                if symbol and pos_data.get('symbol') != symbol:
                    continue
                position = self._convert_metaapi_position(pos_data)
                if position:
                    positions.append(position)
            
            return positions
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    async def get_account_info(self) -> AccountInfo:
        """Get account information"""
        try:
            account_info = await self.connection.get_account_information()
            
            return AccountInfo(
                balance=float(account_info.get('balance', 0)),
                equity=float(account_info.get('equity', 0)),
                margin=float(account_info.get('margin', 0)),
                free_margin=float(account_info.get('freeMargin', 0)),
                margin_level=float(account_info.get('marginLevel', 0)),
                currency=account_info.get('currency', 'USD'),
                leverage=int(account_info.get('leverage', 1))
            )
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise
    
    async def get_market_data(self, symbol: str) -> MarketData:
        """Get current market data for symbol"""
        try:
            # Get symbol price
            price = await self.connection.get_symbol_price(symbol)
            
            return MarketData(
                symbol=symbol,
                bid=float(price.get('bid', 0)),
                ask=float(price.get('ask', 0)),
                last=float(price.get('bid', 0)),  # Use bid as last
                volume=float(price.get('volume', 0)),
                timestamp=datetime.fromtimestamp(
                    price.get('time', 0) / 1000, 
                    tz=timezone.utc
                ) if price.get('time') else datetime.now(timezone.utc)
            )
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            raise
    
    async def get_historical_data(self, 
                                 symbol: str, 
                                 timeframe: str, 
                                 start_time: datetime, 
                                 end_time: Optional[datetime] = None,
                                 limit: Optional[int] = None) -> List[Dict]:
        """Get historical price data"""
        try:
            # Convert timeframe to MetaAPI format
            metaapi_timeframe = self._convert_timeframe(timeframe)
            
            # Get historical data
            candles = await self.connection.get_historical_candles(
                symbol=symbol,
                timeframe=metaapi_timeframe,
                startTime=start_time,
                limit=limit or 1000
            )
            
            historical_data = []
            for candle in candles:
                historical_data.append({
                    "timestamp": datetime.fromtimestamp(
                        candle.get('time', 0) / 1000, 
                        tz=timezone.utc
                    ).isoformat(),
                    "open": float(candle.get('open', 0)),
                    "high": float(candle.get('high', 0)),
                    "low": float(candle.get('low', 0)),
                    "close": float(candle.get('close', 0)),
                    "volume": float(candle.get('tickVolume', 0))
                })
            
            return historical_data
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return []
    
    async def get_symbols(self) -> List[str]:
        """Get available trading symbols"""
        try:
            symbols_data = await self.connection.get_symbols()
            return [symbol.get('symbol', '') for symbol in symbols_data]
        except Exception as e:
            logger.error(f"Failed to get symbols: {e}")
            return []
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to MetaAPI format"""
        mapping = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w", "1M": "1mn"
        }
        return mapping.get(timeframe, "1m")
    
    def _convert_order_status(self, metaapi_state: str) -> OrderStatus:
        """Convert MetaAPI order state to OrderStatus"""
        mapping = {
            "ORDER_STATE_STARTED": OrderStatus.PENDING,
            "ORDER_STATE_PLACED": OrderStatus.OPEN,
            "ORDER_STATE_CANCELED": OrderStatus.CANCELLED,
            "ORDER_STATE_PARTIAL": OrderStatus.PARTIALLY_FILLED,
            "ORDER_STATE_FILLED": OrderStatus.FILLED,
            "ORDER_STATE_REJECTED": OrderStatus.REJECTED,
        }
        return mapping.get(metaapi_state, OrderStatus.PENDING)
    
    def _convert_metaapi_order(self, order_data: Dict) -> Optional[Order]:
        """Convert MetaAPI order data to Order object"""
        try:
            # Determine side from order type
            order_type_str = order_data.get('type', '')
            side = OrderSide.BUY if 'BUY' in order_type_str else OrderSide.SELL
            
            # Determine order type
            if 'MARKET' in order_type_str:
                order_type = OrderType.MARKET
            elif 'LIMIT' in order_type_str:
                order_type = OrderType.LIMIT
            elif 'STOP' in order_type_str:
                order_type = OrderType.STOP
            else:
                order_type = OrderType.MARKET
            
            return Order(
                id=order_data.get('id', ''),
                symbol=order_data.get('symbol', ''),
                side=side,
                type=order_type,
                quantity=float(order_data.get('volume', 0)),
                price=float(order_data.get('openPrice', 0)) if order_data.get('openPrice') else None,
                status=self._convert_order_status(order_data.get('state', 'ORDER_STATE_STARTED')),
                filled_quantity=float(order_data.get('currentVolume', 0)),
                timestamp=datetime.fromtimestamp(
                    order_data.get('time', 0) / 1000, 
                    tz=timezone.utc
                ) if order_data.get('time') else datetime.now(timezone.utc)
            )
        except Exception as e:
            logger.error(f"Failed to convert MetaAPI order: {e}")
            return None
    
    def _convert_metaapi_position(self, pos_data: Dict) -> Optional[Position]:
        """Convert MetaAPI position data to Position object"""
        try:
            # Determine side
            position_type = pos_data.get('type', 'POSITION_TYPE_BUY')
            side = "long" if position_type == 'POSITION_TYPE_BUY' else "short"
            
            return Position(
                symbol=pos_data.get('symbol', ''),
                side=side,
                size=float(pos_data.get('volume', 0)),
                entry_price=float(pos_data.get('openPrice', 0)),
                current_price=float(pos_data.get('currentPrice', 0)),
                unrealized_pnl=float(pos_data.get('unrealizedProfit', 0)),
                realized_pnl=float(pos_data.get('profit', 0)),
                timestamp=datetime.fromtimestamp(
                    pos_data.get('time', 0) / 1000, 
                    tz=timezone.utc
                ) if pos_data.get('time') else datetime.now(timezone.utc)
            )
        except Exception as e:
            logger.error(f"Failed to convert MetaAPI position: {e}")
            return None