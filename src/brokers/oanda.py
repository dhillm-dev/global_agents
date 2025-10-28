"""
OANDA Broker Implementation
Provides trading functionality through OANDA's REST API
"""
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import logging
from decimal import Decimal

from .base import (
    BrokerBase, Order, Position, AccountInfo, MarketData,
    OrderSide, OrderType, OrderStatus
)

logger = logging.getLogger(__name__)

class OandaBroker(BrokerBase):
    """OANDA broker implementation using REST API"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.account_id = config.get("account_id")
        self.environment = config.get("environment", "practice")
        
        # Set base URL based on environment
        if self.environment == "live":
            self.base_url = "https://api-fxtrade.oanda.com"
            self.stream_url = "https://stream-fxtrade.oanda.com"
        else:
            self.base_url = "https://api-fxpractice.oanda.com"
            self.stream_url = "https://stream-fxpractice.oanda.com"
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def connect(self) -> bool:
        """Establish connection to OANDA"""
        try:
            self.session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Test connection by getting account info
            account_info = await self.get_account_info()
            if account_info:
                self.is_connected = True
                self.account_info = account_info
                logger.info(f"Connected to OANDA {self.environment} environment")
                return True
            
            return False
        except Exception as e:
            logger.error(f"Failed to connect to OANDA: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Close connection to OANDA"""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            self.is_connected = False
            logger.info("Disconnected from OANDA")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from OANDA: {e}")
            return False
    
    async def place_order(self, 
                         symbol: str, 
                         side: OrderSide, 
                         quantity: float, 
                         order_type: OrderType = OrderType.MARKET,
                         price: Optional[float] = None,
                         stop_price: Optional[float] = None,
                         **kwargs) -> Order:
        """Place a trading order with OANDA"""
        
        if not self.validate_order_params(symbol, side, quantity, order_type, price):
            raise ValueError("Invalid order parameters")
        
        # Convert to OANDA format
        units = int(quantity) if side == OrderSide.BUY else -int(quantity)
        
        order_data = {
            "order": {
                "instrument": symbol,
                "units": str(units),
                "type": self._convert_order_type(order_type),
                "timeInForce": kwargs.get("time_in_force", "FOK")
            }
        }
        
        # Add price for limit orders
        if order_type == OrderType.LIMIT and price:
            order_data["order"]["price"] = str(price)
        
        # Add stop loss and take profit if provided
        if kwargs.get("stop_loss"):
            order_data["order"]["stopLossOnFill"] = {
                "price": str(kwargs["stop_loss"])
            }
        
        if kwargs.get("take_profit"):
            order_data["order"]["takeProfitOnFill"] = {
                "price": str(kwargs["take_profit"])
            }
        
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/orders"
            async with self.session.post(url, json=order_data) as response:
                if response.status == 201:
                    result = await response.json()
                    order_info = result.get("orderCreateTransaction", {})
                    
                    return Order(
                        id=order_info.get("id", ""),
                        symbol=symbol,
                        side=side,
                        type=order_type,
                        quantity=quantity,
                        price=price,
                        status=OrderStatus.PENDING,
                        timestamp=datetime.now(timezone.utc)
                    )
                else:
                    error_text = await response.text()
                    raise Exception(f"Order placement failed: {error_text}")
        
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/orders/{order_id}/cancel"
            async with self.session.put(url) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order details by ID"""
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/orders/{order_id}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    result = await response.json()
                    order_data = result.get("order", {})
                    return self._convert_oanda_order(order_data)
                return None
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None
    
    async def get_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all orders, optionally filtered by symbol"""
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/orders"
            params = {}
            if symbol:
                params["instrument"] = symbol
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    result = await response.json()
                    orders = []
                    for order_data in result.get("orders", []):
                        order = self._convert_oanda_order(order_data)
                        if order:
                            orders.append(order)
                    return orders
                return []
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get all positions, optionally filtered by symbol"""
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/positions"
            async with self.session.get(url) as response:
                if response.status == 200:
                    result = await response.json()
                    positions = []
                    for pos_data in result.get("positions", []):
                        position = self._convert_oanda_position(pos_data)
                        if position and (not symbol or position.symbol == symbol):
                            positions.append(position)
                    return positions
                return []
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    async def get_account_info(self) -> AccountInfo:
        """Get account information"""
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    result = await response.json()
                    account_data = result.get("account", {})
                    
                    return AccountInfo(
                        balance=float(account_data.get("balance", 0)),
                        equity=float(account_data.get("NAV", 0)),
                        margin=float(account_data.get("marginUsed", 0)),
                        free_margin=float(account_data.get("marginAvailable", 0)),
                        margin_level=float(account_data.get("marginRate", 0)) * 100,
                        currency=account_data.get("currency", "USD"),
                        leverage=int(1 / float(account_data.get("marginRate", 0.02)))
                    )
                else:
                    raise Exception(f"Failed to get account info: {response.status}")
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise
    
    async def get_market_data(self, symbol: str) -> MarketData:
        """Get current market data for symbol"""
        try:
            url = f"{self.base_url}/v3/instruments/{symbol}/candles"
            params = {
                "count": 1,
                "granularity": "M1",
                "price": "BA"  # Bid and Ask
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    result = await response.json()
                    candles = result.get("candles", [])
                    if candles:
                        candle = candles[-1]
                        bid_data = candle.get("bid", {})
                        ask_data = candle.get("ask", {})
                        
                        return MarketData(
                            symbol=symbol,
                            bid=float(bid_data.get("c", 0)),
                            ask=float(ask_data.get("c", 0)),
                            last=float(ask_data.get("c", 0)),
                            volume=float(candle.get("volume", 0)),
                            timestamp=datetime.fromisoformat(candle.get("time", "").replace("Z", "+00:00"))
                        )
                raise Exception("No market data available")
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
            url = f"{self.base_url}/v3/instruments/{symbol}/candles"
            params = {
                "granularity": self._convert_timeframe(timeframe),
                "from": start_time.isoformat() + "Z",
                "price": "MBA"  # Mid, Bid, Ask
            }
            
            if end_time:
                params["to"] = end_time.isoformat() + "Z"
            if limit:
                params["count"] = limit
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    result = await response.json()
                    candles = []
                    for candle in result.get("candles", []):
                        mid_data = candle.get("mid", {})
                        candles.append({
                            "timestamp": candle.get("time"),
                            "open": float(mid_data.get("o", 0)),
                            "high": float(mid_data.get("h", 0)),
                            "low": float(mid_data.get("l", 0)),
                            "close": float(mid_data.get("c", 0)),
                            "volume": float(candle.get("volume", 0))
                        })
                    return candles
                return []
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return []
    
    async def get_symbols(self) -> List[str]:
        """Get available trading symbols"""
        try:
            url = f"{self.base_url}/v3/accounts/{self.account_id}/instruments"
            async with self.session.get(url) as response:
                if response.status == 200:
                    result = await response.json()
                    return [inst.get("name", "") for inst in result.get("instruments", [])]
                return []
        except Exception as e:
            logger.error(f"Failed to get symbols: {e}")
            return []
    
    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert OrderType to OANDA format"""
        mapping = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.STOP: "STOP",
            OrderType.STOP_LIMIT: "STOP"
        }
        return mapping.get(order_type, "MARKET")
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to OANDA granularity"""
        mapping = {
            "1m": "M1", "5m": "M5", "15m": "M15", "30m": "M30",
            "1h": "H1", "4h": "H4", "1d": "D", "1w": "W", "1M": "M"
        }
        return mapping.get(timeframe, "M1")
    
    def _convert_oanda_order(self, order_data: Dict) -> Optional[Order]:
        """Convert OANDA order data to Order object"""
        try:
            units = int(order_data.get("units", 0))
            side = OrderSide.BUY if units > 0 else OrderSide.SELL
            quantity = abs(units)
            
            return Order(
                id=order_data.get("id", ""),
                symbol=order_data.get("instrument", ""),
                side=side,
                type=OrderType.MARKET,  # Simplified
                quantity=quantity,
                price=float(order_data.get("price", 0)) if order_data.get("price") else None,
                status=OrderStatus.PENDING,  # Simplified
                timestamp=datetime.now(timezone.utc)
            )
        except Exception as e:
            logger.error(f"Failed to convert OANDA order: {e}")
            return None
    
    def _convert_oanda_position(self, pos_data: Dict) -> Optional[Position]:
        """Convert OANDA position data to Position object"""
        try:
            long_units = float(pos_data.get("long", {}).get("units", 0))
            short_units = float(pos_data.get("short", {}).get("units", 0))
            
            if long_units == 0 and short_units == 0:
                return None
            
            net_units = long_units + short_units
            side = "long" if net_units > 0 else "short"
            
            return Position(
                symbol=pos_data.get("instrument", ""),
                side=side,
                size=abs(net_units),
                entry_price=float(pos_data.get("long", {}).get("averagePrice", 0)) if net_units > 0 
                           else float(pos_data.get("short", {}).get("averagePrice", 0)),
                current_price=0.0,  # Would need separate API call
                unrealized_pnl=float(pos_data.get("unrealizedPL", 0)),
                realized_pnl=float(pos_data.get("realizedPL", 0)),
                timestamp=datetime.now(timezone.utc)
            )
        except Exception as e:
            logger.error(f"Failed to convert OANDA position: {e}")
            return None