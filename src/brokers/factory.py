"""
Broker Factory - Creates broker instances based on configuration
Supports multiple broker providers: OANDA, FXCM, CCXT, MetaAPI, MT5 Bridge
"""
import os
from typing import Dict, Any, Optional
from .base import BrokerBase
from .oanda import OANDABroker
from .fxcm import FXCMBroker
from .ccxt_broker import CCXTBroker
from .metaapi import MetaAPIBroker
from .mt5_bridge import MT5BridgeBroker
import logging

logger = logging.getLogger(__name__)

class BrokerFactory:
    """Factory class for creating broker instances"""
    
    SUPPORTED_BROKERS = {
        "oanda": OANDABroker,
        "fxcm": FXCMBroker,
        "ccxt": CCXTBroker,
        "metaapi": MetaAPIBroker,
        "mt5_bridge": MT5BridgeBroker
    }
    
    @classmethod
    def create_broker(cls, 
                     broker_type: Optional[str] = None, 
                     config: Optional[Dict[str, Any]] = None) -> BrokerBase:
        """
        Create a broker instance based on type and configuration
        
        Args:
            broker_type: Type of broker (oanda, fxcm, ccxt, metaapi, mt5_bridge)
            config: Broker-specific configuration
            
        Returns:
            BrokerBase: Configured broker instance
            
        Raises:
            ValueError: If broker type is not supported
            KeyError: If required configuration is missing
        """
        # Get broker type from environment if not provided
        if broker_type is None:
            broker_type = os.getenv("BROKER_PROVIDER", "oanda").lower()
        
        # Validate broker type
        if broker_type not in cls.SUPPORTED_BROKERS:
            raise ValueError(
                f"Unsupported broker type: {broker_type}. "
                f"Supported brokers: {list(cls.SUPPORTED_BROKERS.keys())}"
            )
        
        # Use default config if not provided
        if config is None:
            config = cls._get_default_config(broker_type)
        
        # Create broker instance
        broker_class = cls.SUPPORTED_BROKERS[broker_type]
        
        try:
            broker = broker_class(config)
            logger.info(f"Created {broker_type} broker instance")
            return broker
        except Exception as e:
            logger.error(f"Failed to create {broker_type} broker: {e}")
            raise
    
    @classmethod
    def _get_default_config(cls, broker_type: str) -> Dict[str, Any]:
        """Get default configuration for broker type from environment variables"""
        
        if broker_type == "oanda":
            return {
                "api_key": os.getenv("OANDA_API_KEY"),
                "account_id": os.getenv("OANDA_ACCOUNT_ID"),
                "environment": os.getenv("OANDA_ENVIRONMENT", "practice"),  # practice or live
                "base_url": os.getenv("OANDA_BASE_URL")
            }
        
        elif broker_type == "fxcm":
            return {
                "access_token": os.getenv("FXCM_ACCESS_TOKEN"),
                "log_level": os.getenv("FXCM_LOG_LEVEL", "error"),
                "server": os.getenv("FXCM_SERVER", "demo")  # demo or real
            }
        
        elif broker_type == "ccxt":
            return {
                "exchange": os.getenv("CCXT_EXCHANGE", "binance"),
                "api_key": os.getenv("CCXT_API_KEY"),
                "secret": os.getenv("CCXT_SECRET"),
                "password": os.getenv("CCXT_PASSWORD"),  # For some exchanges
                "sandbox": os.getenv("CCXT_SANDBOX", "true").lower() == "true",
                "enable_rate_limit": True
            }
        
        elif broker_type == "metaapi":
            return {
                "token": os.getenv("METAAPI_TOKEN"),
                "account_id": os.getenv("METAAPI_ACCOUNT_ID"),
                "region": os.getenv("METAAPI_REGION", "new-york"),
                "reliability": os.getenv("METAAPI_RELIABILITY", "high")
            }
        
        elif broker_type == "mt5_bridge":
            return {
                "bridge_url": os.getenv("MT5_BRIDGE_URL", "http://localhost:8787"),
                "bridge_token": os.getenv("MT5_BRIDGE_TOKEN", "changeme"),
                "timeout": int(os.getenv("MT5_BRIDGE_TIMEOUT", "30")),
                "retry_attempts": int(os.getenv("MT5_BRIDGE_RETRIES", "3"))
            }
        
        else:
            return {}
    
    @classmethod
    def get_available_brokers(cls) -> list:
        """Get list of available broker types"""
        return list(cls.SUPPORTED_BROKERS.keys())
    
    @classmethod
    def validate_config(cls, broker_type: str, config: Dict[str, Any]) -> bool:
        """Validate broker configuration"""
        
        if broker_type == "oanda":
            required_fields = ["api_key", "account_id"]
            return all(config.get(field) for field in required_fields)
        
        elif broker_type == "fxcm":
            required_fields = ["access_token"]
            return all(config.get(field) for field in required_fields)
        
        elif broker_type == "ccxt":
            required_fields = ["exchange", "api_key", "secret"]
            return all(config.get(field) for field in required_fields)
        
        elif broker_type == "metaapi":
            required_fields = ["token", "account_id"]
            return all(config.get(field) for field in required_fields)
        
        elif broker_type == "mt5_bridge":
            required_fields = ["bridge_url", "bridge_token"]
            return all(config.get(field) for field in required_fields)
        
        return False

# Convenience function for creating brokers
def create_broker(broker_type: Optional[str] = None, 
                 config: Optional[Dict[str, Any]] = None) -> BrokerBase:
    """Convenience function to create a broker instance"""
    return BrokerFactory.create_broker(broker_type, config)

# Global broker instance (lazy-loaded)
_global_broker: Optional[BrokerBase] = None

async def get_broker() -> BrokerBase:
    """Get or create global broker instance"""
    global _global_broker
    
    if _global_broker is None:
        _global_broker = create_broker()
        await _global_broker.connect()
    
    return _global_broker

async def close_broker():
    """Close global broker connection"""
    global _global_broker
    
    if _global_broker is not None:
        await _global_broker.disconnect()
        _global_broker = None