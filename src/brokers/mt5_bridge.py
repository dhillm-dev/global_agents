"""
MT5 Bridge Broker Implementation
Connects to local Windows MT5 Bridge API for MetaTrader5 trading
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

class MT5BridgeBroker(BrokerBase):
    """MT5 Bridge broker implementation using HTTP API"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.bridge_url = config.get("bridge_url", "http://localhost:8787")
        self.bridge_token = config.get("bridge_token", "changeme")
        self.timeout = config.get("timeout", 30)
        self.retry_attempts = config.get("retry_attempts", 3)
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.headers = {
            "X-Token": self.bridge_token,
            "Content-Type": "application/json"
        }
    
    async def connect(self) -> bool:
        """Establish connection to MT5 Bridge"""
        try:
            self.session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
            
            # Test connection by getting account info
            account_info = await self.get_account_info()
            if account_info:
                self.is_connected = True
                self.account_info = account_info
                logger.info(f"Connected to MT5 Bridge at {self.bridge_url}")
                return True
            
            return False
        except Exception as e:
            logger.error(f"Failed to connect to MT5 Bridge: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Close connection to MT5 Bridge"""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            self.is_connected = False
            logger.info("Disconnected from MT5 Bridge")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from MT5 Bridge: {e}")
            return False
    
    async def place_order(self, 
                         symbol: str, 
                         side: OrderSide, 
                         quantity: float, 
                         order_type: OrderType = OrderType.MARKET,
                         price: Optional[float] = None,
                         stop_price: Optional[float] = None,
                         **kwargs) -> Order:
        """Place a trading order through MT5 Bridge"""
        
        if not self.validate_order_params(symbol, side, quantity, order_type, price):
            raise ValueError("Invalid order parameters")
        
        order_data = {
            "symbol": symbol,
            "side": side.value,
            "volume": quantity,
            "type": order_type.value
        }
        
        if price:
            order_data["price"] = price
        if stop_price:
            order_data["stop_price"] = stop_price
        
        # Add MT5-specific parameters
        order_data.update({
            "deviation": kwargs.get("deviation", 10),
            "magic": kwargs.get("magic", 0),
            "comment": kwargs.get("comment", "Global Agent Hub"),
            "type_time": kwargs.get("type_time", "GTC"),
            "type_filling": kwargs.get("type_filling", "FOK")
        })
        
        try:
            for attempt in range(self.retry_attempts):
                try:
                    async with self.session.post(f"{self.bridge_url}/order", json=order_data) as response:
                        if response.status == 200:
                            result = await response.json()
                            
                            if result.get("result") and result["result"].get("retcode") == 10009:  # TRADE_RETCODE_DONE
                                return Order(
                                    id=str(result["result"].get("order", "")),
                                    symbol=symbol,
                                    side=side,
                                    type=order_type,
                                    quantity=quantity,
                                    price=price,
                                    status=OrderStatus.FILLED,
                                    filled_quantity=quantity,
                                    timestamp=datetime.now(timezone.utc)
                                )
                            else:
                                error_msg = f"MT5 order failed: {result.get('error', 'Unknown error')}"
                                raise Exception(error_msg)
                        else:
                            error_text = await response.text()
                            raise Exception(f"Bridge API error: {error_text}")
                
                except aiohttp.ClientError as e:
                    if attempt == self.retry_attempts - 1:
                        raise
                    logger.warning(f"Retry {attempt + 1}/{self.retry_attempts} for order placement: {e}")
                    await asyncio.sleep(1)
        
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order (not implemented in basic bridge)"""
        logger.warning("Order cancellation not implemented in basic MT5 bridge")
        return False
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order details by ID (not implemented in basic bridge)"""
        logger.warning("Get order by ID not implemented in basic MT5 bridge")
        return None
    
    async def get_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all orders (not implemented in basic bridge)"""
        logger.warning("Get orders not implemented in basic MT5 bridge")
        return []
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get all positions through MT5 Bridge"""
        try:
            async with self.session.get(f"{self.bridge_url}/positions") as response:
                if response.status == 200:
                    positions_data = await response.json()
                    positions = []
                    
                    for pos_data in positions_data:
                        if symbol and pos_data.get("symbol") != symbol:
                            continue
                        
                        position = Position(
                            symbol=pos_data.get("symbol", ""),
                            side="long" if pos_data.get("type", 0) == 0 else "short",
                            size=float(pos_data.get("volume", 0)),
                            entry_price=float(pos_data.get("price_open", 0)),
                            current_price=float(pos_data.get("price_current", 0)),
                            unrealized_pnl=float(pos_data.get("profit", 0)),
                            realized_pnl=0.0,  # Not available in basic bridge
                            timestamp=datetime.fromtimestamp(
                                pos_data.get("time", 0), 
                                tz=timezone.utc
                            )
                        )
                        positions.append(position)
                    
                    return positions
                else:
                    logger.error(f"Failed to get positions: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    async def get_account_info(self) -> AccountInfo:
        """Get account information through MT5 Bridge"""
        try:
            async with self.session.get(f"{self.bridge_url}/account") as response:
                if response.status == 200:
                    account_data = await response.json()
                    
                    return AccountInfo(
                        balance=float(account_data.get("balance", 0)),
                        equity=float(account_data.get("equity", 0)),
                        margin=float(account_data.get("margin", 0)),
                        free_margin=float(account_data.get("margin_free", 0)),
                        margin_level=float(account_data.get("margin_level", 0)),
                        currency=account_data.get("currency", "USD"),
                        leverage=int(account_data.get("leverage", 1))
                    )
                else:
                    # Fallback for basic bridge without account endpoint
                    logger.warning("Account endpoint not available, using default values")
                    return AccountInfo(
                        balance=10000.0,
                        equity=10000.0,
                        margin=0.0,
                        free_margin=10000.0,
                        margin_level=0.0,
                        currency="USD",
                        leverage=100
                    )
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise
    
    async def get_market_data(self, symbol: str) -> MarketData:
        """Get current market data for symbol (not implemented in basic bridge)"""
        try:
            # Basic bridge doesn't have market data endpoint
            # Return dummy data or implement tick endpoint
            logger.warning("Market data not available in basic MT5 bridge")
            return MarketData(
                symbol=symbol,
                bid=1.0000,
                ask=1.0001,
                last=1.0000,
                volume=0.0,
                timestamp=datetime.now(timezone.utc)
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
        """Get historical price data (not implemented in basic bridge)"""
        logger.warning("Historical data not implemented in basic MT5 bridge")
        return []
    
    async def get_symbols(self) -> List[str]:
        """Get available trading symbols (not implemented in basic bridge)"""
        try:
            async with self.session.get(f"{self.bridge_url}/symbols") as response:
                if response.status == 200:
                    symbols_data = await response.json()
                    return symbols_data.get("symbols", [])
                else:
                    # Return common forex pairs as fallback
                    return [
                        "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD",
                        "NZDUSD", "EURJPY", "GBPJPY", "EURGBP", "AUDJPY", "EURAUD"
                    ]
        except Exception as e:
            logger.error(f"Failed to get symbols: {e}")
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Check MT5 Bridge connection health"""
        try:
            async with self.session.get(f"{self.bridge_url}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    return {
                        "status": "healthy",
                        "connected": True,
                        "bridge_url": self.bridge_url,
                        "mt5_connected": health_data.get("mt5_connected", False),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                else:
                    return {
                        "status": "bridge_error",
                        "connected": False,
                        "error": f"Bridge returned {response.status}",
                        "timestamp": datetime.utcnow().isoformat()
                    }
        except Exception as e:
            return {
                "status": "connection_error",
                "connected": False,
                "error": str(e),
                "bridge_url": self.bridge_url,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is supported (basic validation)"""
        # Basic validation for forex pairs
        return len(symbol) >= 6 and symbol.isalpha()