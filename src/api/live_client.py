"""
Live trading client wrapper for Polymarket CLOB.

Wraps the official py-clob-client library to provide:
- Automatic credential derivation
- Simplified order placement
- Balance checking
- Order management
"""
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Constants from py-clob-client
POLYGON_CHAIN_ID = 137


@dataclass
class LiveOrderResult:
    """Result of a live order placement."""
    success: bool
    order_id: Optional[str] = None
    message: str = ""
    fills: Optional[List[Dict]] = None
    error: Optional[str] = None


class LiveTradingClient:
    """
    Wrapper around py-clob-client for live trading.
    
    Handles:
    - L1/L2 authentication
    - Order creation and signing
    - Order placement and cancellation
    - Balance checking
    """
    
    HOST = "https://clob.polymarket.com"
    
    def __init__(
        self,
        private_key: str,
        funder: Optional[str] = None,
        signature_type: int = 0,
        chain_id: int = POLYGON_CHAIN_ID,
    ):
        """
        Initialize live trading client.
        
        Args:
            private_key: Wallet private key (with 0x prefix)
            funder: Address that holds funds (optional, for proxy wallets)
            signature_type: 0=EOA, 1=PolyProxy/Email, 2=Gnosis
            chain_id: Polygon chain ID (137 for mainnet)
        """
        self.private_key = private_key
        self.funder = funder
        self.signature_type = signature_type
        self.chain_id = chain_id
        self._client = None
        self._initialized = False
        
    def _ensure_initialized(self):
        """Lazy initialization of the client."""
        if self._initialized:
            return
            
        try:
            from py_clob_client.client import ClobClient
            
            self._client = ClobClient(
                self.HOST,
                key=self.private_key,
                chain_id=self.chain_id,
                signature_type=self.signature_type,
                funder=self.funder,
            )
            
            # Derive API credentials
            creds = self._client.create_or_derive_api_creds()
            self._client.set_api_creds(creds)
            
            self._initialized = True
            logger.info("LiveTradingClient initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LiveTradingClient: {e}")
            raise
    
    def get_balance(self) -> Dict[str, Any]:
        """
        Get USDC balance on Polymarket.
        
        Returns:
            Dict with balance information
        """
        self._ensure_initialized()
        try:
            # Note: Balance is typically checked via Polygon RPC, not CLOB API
            # This is a placeholder - actual implementation needs web3 integration
            return {"balance": 0, "message": "Balance check requires on-chain query"}
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return {"balance": 0, "error": str(e)}
    
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """
        Get all open orders.
        
        Returns:
            List of open orders
        """
        self._ensure_initialized()
        try:
            from py_clob_client.clob_types import OpenOrderParams
            orders = self._client.get_orders(OpenOrderParams())
            return orders if orders else []
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []
    
    def create_limit_order(
        self,
        token_id: str,
        side: str,  # "BUY" or "SELL"
        size: float,
        price: float,
    ) -> LiveOrderResult:
        """
        Create and submit a limit order.
        
        Args:
            token_id: Market token ID
            side: "BUY" or "SELL"
            size: Number of shares
            price: Limit price (0-1)
            
        Returns:
            LiveOrderResult with order details
        """
        self._ensure_initialized()
        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY, SELL
            
            order_side = BUY if side.upper() == "BUY" else SELL
            
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=order_side,
            )
            
            # Create signed order
            signed_order = self._client.create_order(order_args)
            
            # Submit order
            response = self._client.post_order(signed_order, OrderType.GTC)
            
            logger.info(f"Limit order submitted: {response}")
            
            return LiveOrderResult(
                success=True,
                order_id=response.get("orderID") or response.get("id"),
                message="Order placed successfully",
                fills=response.get("fills", []),
            )
            
        except Exception as e:
            logger.error(f"Failed to create limit order: {e}")
            return LiveOrderResult(
                success=False,
                error=str(e),
                message=f"Order failed: {e}",
            )
    
    def create_market_order(
        self,
        token_id: str,
        side: str,  # "BUY" or "SELL"
        amount: float,  # Dollar amount for BUY, shares for SELL
    ) -> LiveOrderResult:
        """
        Create and submit a market order.
        
        Args:
            token_id: Market token ID
            side: "BUY" or "SELL"
            amount: Dollar amount (for BUY) or shares (for SELL)
            
        Returns:
            LiveOrderResult with order details
        """
        self._ensure_initialized()
        try:
            from py_clob_client.clob_types import MarketOrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY, SELL
            
            order_side = BUY if side.upper() == "BUY" else SELL
            
            order_args = MarketOrderArgs(
                token_id=token_id,
                amount=amount,
                side=order_side,
            )
            
            # Create signed market order (Fill-or-Kill)
            signed_order = self._client.create_market_order(order_args)
            
            # Submit order
            response = self._client.post_order(signed_order, OrderType.FOK)
            
            logger.info(f"Market order submitted: {response}")
            
            return LiveOrderResult(
                success=True,
                order_id=response.get("orderID") or response.get("id"),
                message="Market order executed",
                fills=response.get("fills", []),
            )
            
        except Exception as e:
            logger.error(f"Failed to create market order: {e}")
            return LiveOrderResult(
                success=False,
                error=str(e),
                message=f"Order failed: {e}",
            )
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        self._ensure_initialized()
        try:
            self._client.cancel(order_id)
            logger.info(f"Order {order_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def cancel_all_orders(self) -> bool:
        """
        Cancel all open orders.
        
        Returns:
            True if all orders cancelled
        """
        self._ensure_initialized()
        try:
            self._client.cancel_all()
            logger.info("All orders cancelled")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return False
    
    def get_midpoint(self, token_id: str) -> Optional[float]:
        """
        Get midpoint price for a token.
        
        Args:
            token_id: Market token ID
            
        Returns:
            Midpoint price or None
        """
        self._ensure_initialized()
        try:
            result = self._client.get_midpoint(token_id)
            return float(result.get("mid", 0))
        except Exception as e:
            logger.error(f"Failed to get midpoint: {e}")
            return None
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection and authentication.
        
        Returns:
            Dict with connection status
        """
        self._ensure_initialized()
        try:
            # Test with a simple API call
            server_time = self._client.get_server_time()
            open_orders = len(self.get_open_orders())
            
            return {
                "connected": True,
                "server_time": server_time,
                "open_orders_count": open_orders,
                "message": "Connection successful",
            }
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
                "message": f"Connection failed: {e}",
            }
