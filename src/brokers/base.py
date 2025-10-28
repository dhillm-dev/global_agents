"""
Broker Base Class - Abstract interface for all broker implementations
Provides unified API for trading across different platforms
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal
from enum import Enum
import asyncio
from dataclasses import dataclass
from datetime import datetime

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Position:
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime

@dataclass
class Order:
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    timestamp: Optional[datetime] = None

@dataclass
class AccountInfo:
    balance: float
    equity: float
    margin: float
    free_margin: float
    margin_level: float
    currency: str
    leverage: int

@dataclass
class MarketData:
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    timestamp: datetime

class BrokerBase(ABC):
    """Abstract base class for all broker implementations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_connected = False
        self.account_info: Optional[AccountInfo] = None
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to broker"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Close connection to broker"""
        pass
    
    @abstractmethod
    async def place_order(self, 
                         symbol: str, 
                         side: OrderSide, 
                         quantity: float, 
                         order_type: OrderType = OrderType.MARKET,
                         price: Optional[float] = None,
                         stop_price: Optional[float] = None,
                         **kwargs) -> Order:
        """Place a trading order"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        pass
    
    @abstractmethod
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order details by ID"""
        pass
    
    @abstractmethod
    async def get_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all orders, optionally filtered by symbol"""
        pass
    
    @abstractmethod
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get all positions, optionally filtered by symbol"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        """Get account information"""
        pass
    
    @abstractmethod
    async def get_market_data(self, symbol: str) -> MarketData:
        """Get current market data for symbol"""
        pass
    
    @abstractmethod
    async def get_historical_data(self, 
                                 symbol: str, 
                                 timeframe: str, 
                                 start_time: datetime, 
                                 end_time: Optional[datetime] = None,
                                 limit: Optional[int] = None) -> List[Dict]:
        """Get historical price data"""
        pass
    
    @abstractmethod
    async def get_symbols(self) -> List[str]:
        """Get available trading symbols"""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Check broker connection health"""
        try:
            if not self.is_connected:
                return {"status": "disconnected", "error": "Not connected to broker"}
            
            # Try to get account info as a health check
            account = await self.get_account_info()
            return {
                "status": "healthy",
                "connected": True,
                "account_currency": account.currency if account else "unknown",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "connected": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is supported"""
        # Override in subclasses for broker-specific validation
        return len(symbol) > 0
    
    def validate_order_params(self, 
                            symbol: str, 
                            side: OrderSide, 
                            quantity: float, 
                            order_type: OrderType,
                            price: Optional[float] = None) -> bool:
        """Validate order parameters"""
        if not self.validate_symbol(symbol):
            return False
        
        if quantity <= 0:
            return False
        
        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and price is None:
            return False
        
        return True
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()