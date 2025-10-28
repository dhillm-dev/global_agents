"""
CCXT Broker Implementation
Provides trading functionality through CCXT library for cryptocurrency exchanges
"""
import asyncio
import ccxt.async_support as ccxt
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import logging
from decimal import Decimal

from .base import (
    BrokerBase, Order, Position, AccountInfo, MarketData,
    OrderSide, OrderType, OrderStatus
)

logger = logging.getLogger(__name__)

class CCXTBroker(BrokerBase):
    """CCXT broker implementation for cryptocurrency exchanges"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.exchange_name = config.get("exchange", "binance")
        self.api_key = config.get("api_key")
        self.secret = config.get("secret")
        self.password = config.get("password")  # For some exchanges
        self.sandbox = config.get("sandbox", True)
        self.enable_rate_limit = config.get("enable_rate_limit", True)
        
        self.exchange: Optional[ccxt.Exchange] = None
    
    async def connect(self) -> bool:
        """Establish connection to cryptocurrency exchange"""
        try:
            # Get exchange class
            exchange_class = getattr(ccxt, self.exchange_name)
            
            # Configure exchange
            config = {
                'apiKey': self.api_key,
                'secret': self.secret,
                'enableRateLimit': self.enable_rate_limit,
                'sandbox': self.sandbox,
            }
            
            # Add password for exchanges that require it
            if self.password:
                config['password'] = self.password
            
            self.exchange = exchange_class(config)
            
            # Test connection by loading markets
            await self.exchange.load_markets()
            
            # Get account info to verify credentials
            balance = await self.exchange.fetch_balance()
            
            self.is_connected = True
            logger.info(f"Connected to {self.exchange_name} ({'sandbox' if self.sandbox else 'live'})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to {self.exchange_name}: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Close connection to exchange"""
        try:
            if self.exchange:
                await self.exchange.close()
                self.exchange = None
            self.is_connected = False
            logger.info(f"Disconnected from {self.exchange_name}")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from {self.exchange_name}: {e}")
            return False
    
    async def place_order(self, 
                         symbol: str, 
                         side: OrderSide, 
                         quantity: float, 
                         order_type: OrderType = OrderType.MARKET,
                         price: Optional[float] = None,
                         stop_price: Optional[float] = None,
                         **kwargs) -> Order:
        """Place a trading order on the exchange"""
        
        if not self.validate_order_params(symbol, side, quantity, order_type, price):
            raise ValueError("Invalid order parameters")
        
        try:
            # Convert parameters to CCXT format
            ccxt_side = side.value
            ccxt_type = self._convert_order_type(order_type)
            
            # Place order based on type
            if order_type == OrderType.MARKET:
                result = await self.exchange.create_market_order(
                    symbol, ccxt_side, quantity, None, None, kwargs
                )
            elif order_type == OrderType.LIMIT:
                result = await self.exchange.create_limit_order(
                    symbol, ccxt_side, quantity, price, None, kwargs
                )
            elif order_type == OrderType.STOP:
                # Use stop market order
                result = await self.exchange.create_order(
                    symbol, 'stop_market', ccxt_side, quantity, None, 
                    {'stopPrice': stop_price, **kwargs}
                )
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            return Order(
                id=result['id'],
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity,
                price=price,
                status=self._convert_order_status(result.get('status', 'open')),
                filled_quantity=float(result.get('filled', 0)),
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"Failed to place order on {self.exchange_name}: {e}")
            raise
    
    async def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        """Cancel an existing order"""
        try:
            await self.exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_order(self, order_id: str, symbol: str = None) -> Optional[Order]:
        """Get order details by ID"""
        try:
            result = await self.exchange.fetch_order(order_id, symbol)
            return self._convert_ccxt_order(result)
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None
    
    async def get_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all orders, optionally filtered by symbol"""
        try:
            if symbol:
                orders_data = await self.exchange.fetch_open_orders(symbol)
            else:
                orders_data = await self.exchange.fetch_open_orders()
            
            orders = []
            for order_data in orders_data:
                order = self._convert_ccxt_order(order_data)
                if order:
                    orders.append(order)
            return orders
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get all positions, optionally filtered by symbol"""
        try:
            # Not all exchanges support positions (mainly derivatives)
            if hasattr(self.exchange, 'fetch_positions'):
                positions_data = await self.exchange.fetch_positions(symbol)
                positions = []
                for pos_data in positions_data:
                    position = self._convert_ccxt_position(pos_data)
                    if position and position.size > 0:
                        positions.append(position)
                return positions
            else:
                # For spot exchanges, simulate positions from balance
                return await self._get_spot_positions(symbol)
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    async def get_account_info(self) -> AccountInfo:
        """Get account information"""
        try:
            balance = await self.exchange.fetch_balance()
            
            # Calculate totals
            total_balance = 0.0
            total_used = 0.0
            
            for currency, amounts in balance.items():
                if isinstance(amounts, dict) and 'total' in amounts:
                    total_balance += float(amounts.get('total', 0))
                    total_used += float(amounts.get('used', 0))
            
            return AccountInfo(
                balance=total_balance,
                equity=total_balance,
                margin=total_used,
                free_margin=total_balance - total_used,
                margin_level=0.0,  # Not applicable for spot trading
                currency="USDT",  # Default to USDT for crypto
                leverage=1  # Default for spot trading
            )
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise
    
    async def get_market_data(self, symbol: str) -> MarketData:
        """Get current market data for symbol"""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            
            return MarketData(
                symbol=symbol,
                bid=float(ticker.get('bid', 0)),
                ask=float(ticker.get('ask', 0)),
                last=float(ticker.get('last', 0)),
                volume=float(ticker.get('baseVolume', 0)),
                timestamp=datetime.fromtimestamp(
                    ticker.get('timestamp', 0) / 1000, 
                    tz=timezone.utc
                )
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
            # Convert timeframe to CCXT format
            ccxt_timeframe = self._convert_timeframe(timeframe)
            
            # Convert datetime to timestamp
            since = int(start_time.timestamp() * 1000)
            
            # Fetch OHLCV data
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol, ccxt_timeframe, since, limit
            )
            
            candles = []
            for candle in ohlcv:
                candles.append({
                    "timestamp": datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc).isoformat(),
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5])
                })
            
            return candles
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return []
    
    async def get_symbols(self) -> List[str]:
        """Get available trading symbols"""
        try:
            markets = await self.exchange.load_markets()
            return list(markets.keys())
        except Exception as e:
            logger.error(f"Failed to get symbols: {e}")
            return []
    
    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert OrderType to CCXT format"""
        mapping = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP: "stop_market",
            OrderType.STOP_LIMIT: "stop_limit"
        }
        return mapping.get(order_type, "market")
    
    def _convert_order_status(self, ccxt_status: str) -> OrderStatus:
        """Convert CCXT order status to OrderStatus"""
        mapping = {
            "open": OrderStatus.OPEN,
            "closed": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED
        }
        return mapping.get(ccxt_status, OrderStatus.PENDING)
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to CCXT format"""
        mapping = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w", "1M": "1M"
        }
        return mapping.get(timeframe, "1m")
    
    def _convert_ccxt_order(self, order_data: Dict) -> Optional[Order]:
        """Convert CCXT order data to Order object"""
        try:
            side = OrderSide.BUY if order_data.get('side') == 'buy' else OrderSide.SELL
            
            return Order(
                id=order_data.get('id', ''),
                symbol=order_data.get('symbol', ''),
                side=side,
                type=OrderType.MARKET,  # Simplified
                quantity=float(order_data.get('amount', 0)),
                price=float(order_data.get('price', 0)) if order_data.get('price') else None,
                status=self._convert_order_status(order_data.get('status', 'open')),
                filled_quantity=float(order_data.get('filled', 0)),
                timestamp=datetime.fromtimestamp(
                    order_data.get('timestamp', 0) / 1000, 
                    tz=timezone.utc
                ) if order_data.get('timestamp') else datetime.now(timezone.utc)
            )
        except Exception as e:
            logger.error(f"Failed to convert CCXT order: {e}")
            return None
    
    def _convert_ccxt_position(self, pos_data: Dict) -> Optional[Position]:
        """Convert CCXT position data to Position object"""
        try:
            size = float(pos_data.get('size', 0))
            if size == 0:
                return None
            
            return Position(
                symbol=pos_data.get('symbol', ''),
                side=pos_data.get('side', 'long'),
                size=abs(size),
                entry_price=float(pos_data.get('entryPrice', 0)),
                current_price=float(pos_data.get('markPrice', 0)),
                unrealized_pnl=float(pos_data.get('unrealizedPnl', 0)),
                realized_pnl=0.0,  # Not always available
                timestamp=datetime.now(timezone.utc)
            )
        except Exception as e:
            logger.error(f"Failed to convert CCXT position: {e}")
            return None
    
    async def _get_spot_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get positions for spot trading (simulate from balance)"""
        try:
            balance = await self.exchange.fetch_balance()
            positions = []
            
            for currency, amounts in balance.items():
                if isinstance(amounts, dict):
                    total = float(amounts.get('total', 0))
                    if total > 0:
                        # Create a position for each non-zero balance
                        positions.append(Position(
                            symbol=f"{currency}/USDT",  # Assume USDT pairs
                            side="long",
                            size=total,
                            entry_price=0.0,  # Unknown for spot
                            current_price=0.0,  # Would need separate API call
                            unrealized_pnl=0.0,
                            realized_pnl=0.0,
                            timestamp=datetime.now(timezone.utc)
                        ))
            
            return positions
        except Exception as e:
            logger.error(f"Failed to get spot positions: {e}")
            return []