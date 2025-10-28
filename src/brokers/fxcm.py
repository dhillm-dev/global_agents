"""
FXCM Broker Implementation
Provides trading functionality through FXCM REST API
"""
import asyncio
import aiohttp
import fxcmpy
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import logging
from decimal import Decimal

from .base import (
    BrokerBase, Order, Position, AccountInfo, MarketData,
    OrderSide, OrderType, OrderStatus
)

logger = logging.getLogger(__name__)

class FXCMBroker(BrokerBase):
    """FXCM broker implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.access_token = config.get("access_token")
        self.server = config.get("server", "demo")  # demo or real
        self.log_level = config.get("log_level", "error")
        
        self.api: Optional[fxcmpy.fxcmpy] = None
    
    async def connect(self) -> bool:
        """Establish connection to FXCM"""
        try:
            # Initialize FXCM connection
            self.api = fxcmpy.fxcmpy(
                access_token=self.access_token,
                log_level=self.log_level,
                server=self.server
            )
            
            # Test connection
            account_info = self.api.get_accounts()
            if account_info.empty:
                raise Exception("No accounts found")
            
            self.is_connected = True
            logger.info(f"Connected to FXCM ({self.server})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to FXCM: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Close connection to FXCM"""
        try:
            if self.api:
                self.api.close()
                self.api = None
            self.is_connected = False
            logger.info("Disconnected from FXCM")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from FXCM: {e}")
            return False
    
    async def place_order(self, 
                         symbol: str, 
                         side: OrderSide, 
                         quantity: float, 
                         order_type: OrderType = OrderType.MARKET,
                         price: Optional[float] = None,
                         stop_price: Optional[float] = None,
                         **kwargs) -> Order:
        """Place a trading order with FXCM"""
        
        if not self.validate_order_params(symbol, side, quantity, order_type, price):
            raise ValueError("Invalid order parameters")
        
        try:
            # Convert parameters to FXCM format
            is_buy = side == OrderSide.BUY
            amount = int(quantity * 1000)  # FXCM uses units (1000 = 1 lot)
            
            # Place order based on type
            if order_type == OrderType.MARKET:
                order_id = self.api.create_market_buy_order(symbol, amount) if is_buy else \
                          self.api.create_market_sell_order(symbol, amount)
            elif order_type == OrderType.LIMIT:
                if not price:
                    raise ValueError("Price required for limit orders")
                order_id = self.api.create_entry_order(
                    symbol=symbol,
                    is_buy=is_buy,
                    amount=amount,
                    rate=price,
                    is_in_pips=False
                )
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            # Get order details
            order_data = self.api.get_order(order_id)
            
            return Order(
                id=str(order_id),
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity,
                price=price,
                status=self._convert_order_status(order_data.get('status', 'O')),
                filled_quantity=0.0,  # Will be updated when filled
                timestamp=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"Failed to place order with FXCM: {e}")
            raise
    
    async def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        """Cancel an existing order"""
        try:
            result = self.api.delete_order(int(order_id))
            return result is not None
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_order(self, order_id: str, symbol: str = None) -> Optional[Order]:
        """Get order details by ID"""
        try:
            order_data = self.api.get_order(int(order_id))
            return self._convert_fxcm_order(order_data)
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None
    
    async def get_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all orders, optionally filtered by symbol"""
        try:
            orders_df = self.api.get_orders()
            orders = []
            
            for _, order_data in orders_df.iterrows():
                if symbol and order_data.get('currency') != symbol:
                    continue
                order = self._convert_fxcm_order(order_data)
                if order:
                    orders.append(order)
            
            return orders
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get all positions, optionally filtered by symbol"""
        try:
            positions_df = self.api.get_open_positions()
            positions = []
            
            for _, pos_data in positions_df.iterrows():
                if symbol and pos_data.get('currency') != symbol:
                    continue
                position = self._convert_fxcm_position(pos_data)
                if position:
                    positions.append(position)
            
            return positions
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    async def get_account_info(self) -> AccountInfo:
        """Get account information"""
        try:
            accounts_df = self.api.get_accounts()
            if accounts_df.empty:
                raise Exception("No account information available")
            
            account = accounts_df.iloc[0]
            
            return AccountInfo(
                balance=float(account.get('balance', 0)),
                equity=float(account.get('equity', 0)),
                margin=float(account.get('usableMargin', 0)),
                free_margin=float(account.get('usableMargin', 0)),
                margin_level=float(account.get('marginCallStatus', 0)),
                currency=account.get('accountCurrency', 'USD'),
                leverage=int(account.get('leverage', 1))
            )
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            raise
    
    async def get_market_data(self, symbol: str) -> MarketData:
        """Get current market data for symbol"""
        try:
            # Get latest price data
            prices = self.api.get_last_price(symbol)
            
            return MarketData(
                symbol=symbol,
                bid=float(prices.get('Bid', 0)),
                ask=float(prices.get('Ask', 0)),
                last=float(prices.get('Bid', 0)),  # Use bid as last
                volume=0.0,  # FXCM doesn't provide volume in this endpoint
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
        """Get historical price data"""
        try:
            # Convert timeframe to FXCM format
            fxcm_timeframe = self._convert_timeframe(timeframe)
            
            # Get historical data
            if end_time:
                data = self.api.get_candles(
                    symbol, 
                    period=fxcm_timeframe,
                    start=start_time,
                    end=end_time
                )
            else:
                data = self.api.get_candles(
                    symbol, 
                    period=fxcm_timeframe,
                    start=start_time,
                    number=limit or 100
                )
            
            candles = []
            for timestamp, row in data.iterrows():
                candles.append({
                    "timestamp": timestamp.isoformat(),
                    "open": float(row['bidopen']),
                    "high": float(row['bidhigh']),
                    "low": float(row['bidlow']),
                    "close": float(row['bidclose']),
                    "volume": float(row.get('tickqty', 0))
                })
            
            return candles
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return []
    
    async def get_symbols(self) -> List[str]:
        """Get available trading symbols"""
        try:
            instruments = self.api.get_instruments()
            return instruments['instrument'].tolist()
        except Exception as e:
            logger.error(f"Failed to get symbols: {e}")
            return []
    
    def _convert_order_status(self, fxcm_status: str) -> OrderStatus:
        """Convert FXCM order status to OrderStatus"""
        mapping = {
            "O": OrderStatus.OPEN,      # Open
            "F": OrderStatus.FILLED,    # Filled
            "C": OrderStatus.CANCELLED, # Cancelled
            "R": OrderStatus.REJECTED,  # Rejected
            "P": OrderStatus.PENDING,   # Pending
            "W": OrderStatus.OPEN,      # Waiting
        }
        return mapping.get(fxcm_status, OrderStatus.PENDING)
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to FXCM format"""
        mapping = {
            "1m": "m1", "5m": "m5", "15m": "m15", "30m": "m30",
            "1h": "H1", "4h": "H4", "1d": "D1", "1w": "W1", "1M": "M1"
        }
        return mapping.get(timeframe, "m1")
    
    def _convert_fxcm_order(self, order_data) -> Optional[Order]:
        """Convert FXCM order data to Order object"""
        try:
            # Determine side from FXCM data
            is_buy = order_data.get('isBuy', True)
            side = OrderSide.BUY if is_buy else OrderSide.SELL
            
            # Convert amount from units to lots
            amount_units = float(order_data.get('amount', 0))
            quantity = amount_units / 1000.0
            
            return Order(
                id=str(order_data.get('orderId', '')),
                symbol=order_data.get('currency', ''),
                side=side,
                type=OrderType.LIMIT,  # Simplified - FXCM has various types
                quantity=quantity,
                price=float(order_data.get('rate', 0)) if order_data.get('rate') else None,
                status=self._convert_order_status(order_data.get('status', 'O')),
                filled_quantity=0.0,  # Not directly available
                timestamp=datetime.now(timezone.utc)
            )
        except Exception as e:
            logger.error(f"Failed to convert FXCM order: {e}")
            return None
    
    def _convert_fxcm_position(self, pos_data) -> Optional[Position]:
        """Convert FXCM position data to Position object"""
        try:
            # Determine side
            is_buy = pos_data.get('isBuy', True)
            side = "long" if is_buy else "short"
            
            # Convert amount from units to lots
            amount_units = float(pos_data.get('amount', 0))
            size = amount_units / 1000.0
            
            return Position(
                symbol=pos_data.get('currency', ''),
                side=side,
                size=size,
                entry_price=float(pos_data.get('open', 0)),
                current_price=float(pos_data.get('close', 0)),
                unrealized_pnl=float(pos_data.get('grossPL', 0)),
                realized_pnl=0.0,  # Not directly available
                timestamp=datetime.now(timezone.utc)
            )
        except Exception as e:
            logger.error(f"Failed to convert FXCM position: {e}")
            return None