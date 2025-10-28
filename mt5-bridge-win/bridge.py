"""
MT5 Windows Bridge - Local FastAPI server for MetaTrader5 integration
Runs on Windows to provide MT5 access to the Linux-based Global Agent Hub
"""
import MetaTrader5 as mt5
import os
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security token from environment
TOKEN = os.getenv("MT5_BRIDGE_TOKEN", "changeme")

app = FastAPI(
    title="MT5 Windows Bridge",
    description="MetaTrader5 Bridge API for Global Agent Hub",
    version="1.0.0"
)

# Pydantic models
class OrderRequest(BaseModel):
    symbol: str
    side: str  # "buy" or "sell"
    volume: float
    type: str = "market"
    price: Optional[float] = None
    stop_price: Optional[float] = None
    deviation: int = 10
    magic: int = 0
    comment: str = "Global Agent Hub"
    type_time: str = "GTC"
    type_filling: str = "FOK"

class HealthResponse(BaseModel):
    status: str
    mt5_connected: bool
    account_info: Optional[Dict[str, Any]] = None
    timestamp: str

# Security dependency
def verify_token(x_token: Optional[str] = Header(None)):
    if x_token != TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    return x_token

@app.on_event("startup")
async def startup_event():
    """Initialize MT5 connection on startup"""
    try:
        if not mt5.initialize():
            error = mt5.last_error()
            logger.error(f"MT5 initialization failed: {error}")
            raise RuntimeError(f"MT5 initialization failed: {error}")
        
        # Get account info to verify connection
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("Failed to get MT5 account info")
        else:
            logger.info(f"MT5 Bridge started successfully. Account: {account_info.login}")
    
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup MT5 connection on shutdown"""
    try:
        mt5.shutdown()
        logger.info("MT5 Bridge shutdown completed")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check if MT5 is initialized
        account_info = mt5.account_info()
        mt5_connected = account_info is not None
        
        account_data = None
        if mt5_connected:
            account_data = {
                "login": account_info.login,
                "server": account_info.server,
                "currency": account_info.currency,
                "balance": account_info.balance,
                "equity": account_info.equity,
                "margin": account_info.margin,
                "margin_free": account_info.margin_free,
                "margin_level": account_info.margin_level,
                "leverage": account_info.leverage
            }
        
        return HealthResponse(
            status="healthy" if mt5_connected else "mt5_disconnected",
            mt5_connected=mt5_connected,
            account_info=account_data,
            timestamp=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="error",
            mt5_connected=False,
            timestamp=datetime.utcnow().isoformat()
        )

@app.post("/order")
async def place_order(order: OrderRequest, token: str = Depends(verify_token)):
    """Place a trading order through MT5"""
    try:
        # Convert side to MT5 action
        if order.side.lower() == "buy":
            action = mt5.ORDER_TYPE_BUY
        elif order.side.lower() == "sell":
            action = mt5.ORDER_TYPE_SELL
        else:
            raise ValueError(f"Invalid order side: {order.side}")
        
        # Prepare order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": order.symbol,
            "volume": order.volume,
            "type": action,
            "deviation": order.deviation,
            "magic": order.magic,
            "comment": order.comment,
            "type_time": getattr(mt5, f"ORDER_TIME_{order.type_time}", mt5.ORDER_TIME_GTC),
            "type_filling": getattr(mt5, f"ORDER_FILLING_{order.type_filling}", mt5.ORDER_FILLING_FOK)
        }
        
        # Add price for limit orders
        if order.type.lower() == "limit" and order.price:
            request["price"] = order.price
        
        # Send order
        result = mt5.order_send(request)
        
        # Prepare response
        response_data = {
            "request": request,
            "result": result._asdict() if result else None,
            "error": mt5.last_error()
        }
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Order placed successfully: {result.order}")
        else:
            logger.error(f"Order failed: {result.retcode if result else 'No result'}")
        
        return response_data
    
    except Exception as e:
        logger.error(f"Order placement error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/positions")
async def get_positions(token: str = Depends(verify_token)):
    """Get all open positions"""
    try:
        positions = mt5.positions_get()
        if positions is None:
            return []
        
        positions_list = []
        for pos in positions:
            positions_list.append({
                "ticket": pos.ticket,
                "symbol": pos.symbol,
                "type": pos.type,
                "volume": pos.volume,
                "price_open": pos.price_open,
                "price_current": pos.price_current,
                "profit": pos.profit,
                "swap": pos.swap,
                "comment": pos.comment,
                "time": pos.time,
                "magic": pos.magic
            })
        
        return positions_list
    
    except Exception as e:
        logger.error(f"Get positions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/account")
async def get_account_info(token: str = Depends(verify_token)):
    """Get account information"""
    try:
        account_info = mt5.account_info()
        if account_info is None:
            raise HTTPException(status_code=500, detail="Failed to get account info")
        
        return {
            "login": account_info.login,
            "server": account_info.server,
            "currency": account_info.currency,
            "balance": account_info.balance,
            "equity": account_info.equity,
            "margin": account_info.margin,
            "margin_free": account_info.margin_free,
            "margin_level": account_info.margin_level,
            "leverage": account_info.leverage,
            "profit": account_info.profit,
            "credit": account_info.credit,
            "margin_so_mode": account_info.margin_so_mode,
            "margin_so_call": account_info.margin_so_call,
            "margin_so_so": account_info.margin_so_so,
            "margin_initial": account_info.margin_initial,
            "margin_maintenance": account_info.margin_maintenance,
            "assets": account_info.assets,
            "liabilities": account_info.liabilities,
            "commission_blocked": account_info.commission_blocked
        }
    
    except Exception as e:
        logger.error(f"Get account info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/symbols")
async def get_symbols(token: str = Depends(verify_token)):
    """Get available trading symbols"""
    try:
        symbols = mt5.symbols_get()
        if symbols is None:
            return {"symbols": []}
        
        symbol_list = [symbol.name for symbol in symbols if symbol.visible]
        return {"symbols": symbol_list}
    
    except Exception as e:
        logger.error(f"Get symbols error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market_data/{symbol}")
async def get_market_data(symbol: str, token: str = Depends(verify_token)):
    """Get current market data for a symbol"""
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
        
        return {
            "symbol": symbol,
            "bid": tick.bid,
            "ask": tick.ask,
            "last": tick.last,
            "volume": tick.volume,
            "time": tick.time,
            "flags": tick.flags,
            "volume_real": tick.volume_real
        }
    
    except Exception as e:
        logger.error(f"Get market data error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("MT5_BRIDGE_HOST", "0.0.0.0")
    port = int(os.getenv("MT5_BRIDGE_PORT", "8787"))
    
    logger.info(f"Starting MT5 Bridge on {host}:{port}")
    logger.info(f"Token: {TOKEN}")
    
    uvicorn.run(
        "bridge:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )