"""
Polymarket CLOB API Client.

Provides an async wrapper around the py-clob-client library with additional
features including rate limiting, retry logic, and comprehensive error handling.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, List

import aiohttp

from src.api.exceptions import (
    PolymarketError,
    PolymarketAPIError,
    RateLimitError,
    AuthenticationError,
    MarketNotFoundError,
    OrderError,
    InsufficientBalanceError,
    InvalidOrderError,
    OrderNotFoundError,
    OrderCancellationError,
)
from src.api.rate_limiter import RateLimiter
from src.api.utils import (
    retry_with_backoff,
    normalize_side,
    normalize_order_type,
    parse_orderbook_response,
    calculate_spread,
)

logger = logging.getLogger(__name__)


class PolymarketClient:
    """
    Asynchronous client for the Polymarket CLOB API.
    
    This client wraps the py-clob-client library to provide async operations
    with additional features like rate limiting, retry logic, and comprehensive
    error handling.
    
    Attributes:
        host: API base URL
        chain_id: Blockchain chain ID (137 for Polygon)
        is_authenticated: Whether the client has valid credentials
    
    Example:
        # Read-only client
        client = PolymarketClient()
        await client.connect()
        markets = await client.get_markets()
        await client.close()
        
        # Authenticated client
        client = PolymarketClient(
            private_key="0x...",
            funder="0x..."
        )
        await client.connect()
        await client.create_or_derive_api_creds()
        order = await client.create_order({...})
        await client.close()
    """
    
    DEFAULT_HOST = "https://clob.polymarket.com"
    DEFAULT_CHAIN_ID = 137  # Polygon
    
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        private_key: Optional[str] = None,
        chain_id: int = DEFAULT_CHAIN_ID,
        signature_type: int = 1,
        funder: Optional[str] = None,
        max_requests_per_minute: int = 60,
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        """
        Initialize the Polymarket client.
        
        Args:
            host: API base URL
            private_key: Private key for authenticated requests
            chain_id: Blockchain chain ID (137 for Polygon)
            signature_type: Signature type (0=EOA, 1=Email/Magic, 2=Browser proxy)
            funder: Funder address for order placement
            max_requests_per_minute: Rate limit for API requests
            max_retries: Maximum retry attempts for failed requests
            timeout: Request timeout in seconds
        """
        self.host = host.rstrip("/")
        self.private_key = private_key
        self.chain_id = chain_id
        self.signature_type = signature_type
        self.funder = funder
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Rate limiter
        self._rate_limiter = RateLimiter(max_requests_per_minute=max_requests_per_minute)
        
        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Thread pool for running sync py-clob-client operations
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # py-clob-client instance (lazy initialized)
        self._clob_client = None
        
        # API credentials
        self._api_creds: Optional[dict] = None
        
        # Request logging
        self._request_count = 0
        self._last_request_time: Optional[float] = None
        
        logger.info(
            f"PolymarketClient initialized: host={host}, "
            f"chain_id={chain_id}, authenticated={bool(private_key)}"
        )
    
    @property
    def is_authenticated(self) -> bool:
        """Check if the client has valid credentials."""
        return self._api_creds is not None and self.private_key is not None
    
    async def connect(self) -> None:
        """
        Initialize the HTTP session and py-clob-client.
        
        Must be called before making any API requests.
        """
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"Content-Type": "application/json"}
            )
        
        # Initialize py-clob-client if private key provided
        if self.private_key:
            await self._init_clob_client()
        
        logger.info("PolymarketClient connected")
    
    async def _init_clob_client(self) -> None:
        """Initialize the py-clob-client in a thread pool."""
        def _init():
            try:
                from py_clob_client.client import ClobClient
                
                if self.private_key:
                    client = ClobClient(
                        self.host,
                        key=self.private_key,
                        chain_id=self.chain_id,
                        signature_type=self.signature_type,
                        funder=self.funder
                    )
                else:
                    client = ClobClient(self.host)
                
                return client
            except ImportError:
                logger.warning(
                    "py-clob-client not installed. "
                    "Order signing will not be available."
                )
                return None
            except Exception as e:
                logger.error(f"Failed to initialize py-clob-client: {e}")
                raise
        
        loop = asyncio.get_running_loop()
        self._clob_client = await loop.run_in_executor(self._executor, _init)
    
    async def close(self) -> None:
        """Close the HTTP session and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()
        
        self._executor.shutdown(wait=False)
        
        logger.info(
            f"PolymarketClient closed after {self._request_count} requests"
        )
    
    async def __aenter__(self) -> "PolymarketClient":
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
    
    # =========================================================================
    # Internal Request Methods
    # =========================================================================
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the HTTP session."""
        if self._session is None or self._session.closed:
            await self.connect()
        return self._session
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        data: Optional[dict] = None,
        authenticated: bool = False,
    ) -> Any:
        """
        Make an HTTP request to the API.
        
        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint path
            params: Query parameters
            data: JSON body data
            authenticated: Whether to include auth headers
        
        Returns:
            Parsed JSON response
        
        Raises:
            PolymarketAPIError: For API errors
            RateLimitError: When rate limited
            AuthenticationError: For auth failures
        """
        async with self._rate_limiter:
            return await self._request_internal(
                method, endpoint, params, data, authenticated
            )
    
    async def _request_internal(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        data: Optional[dict] = None,
        authenticated: bool = False,
    ) -> Any:
        """Internal request method without rate limiting."""
        session = await self._get_session()
        url = f"{self.host}{endpoint}"
        
        headers = {"Content-Type": "application/json"}
        
        if authenticated:
            if not self._api_creds:
                raise AuthenticationError(
                    401,
                    "API credentials required. Call create_or_derive_api_creds() first."
                )
            headers.update({
                "POLY_ADDRESS": self._api_creds.get("address", ""),
                "POLY_SIGNATURE": self._api_creds.get("signature", ""),
                "POLY_TIMESTAMP": str(self._api_creds.get("timestamp", "")),
                "POLY_NONCE": str(self._api_creds.get("nonce", "")),
                "POLY_API_KEY": self._api_creds.get("api_key", ""),
                "POLY_PASSPHRASE": self._api_creds.get("passphrase", ""),
                "POLY_SECRET": self._api_creds.get("secret", ""),
            })
        
        self._request_count += 1
        self._last_request_time = time.time()
        
        logger.debug(f"Request {self._request_count}: {method} {url}")
        
        try:
            async with session.request(
                method,
                url,
                params=params,
                json=data,
                headers=headers
            ) as response:
                response_text = await response.text()
                
                logger.debug(
                    f"Response {response.status}: {response_text[:200]}..."
                    if len(response_text) > 200 else
                    f"Response {response.status}: {response_text}"
                )
                
                # Handle error responses
                if response.status == 429:
                    retry_after = response.headers.get("Retry-After")
                    raise RateLimitError(
                        429,
                        "Rate limit exceeded",
                        retry_after=float(retry_after) if retry_after else None
                    )
                
                if response.status in (401, 403):
                    raise AuthenticationError(response.status, response_text)
                
                if response.status == 404:
                    raise MarketNotFoundError(
                        404,
                        f"Resource not found: {endpoint}"
                    )
                
                if response.status >= 400:
                    # Try to parse error message
                    error_msg = response_text
                    try:
                        import json
                        error_data = json.loads(response_text)
                        error_msg = error_data.get("message", response_text)
                        
                        # Check for specific order errors
                        if "insufficient" in error_msg.lower():
                            raise InsufficientBalanceError(
                                response.status,
                                error_msg
                            )
                        if "invalid" in error_msg.lower() and "order" in error_msg.lower():
                            raise InvalidOrderError(
                                response.status,
                                error_msg
                            )
                    except (json.JSONDecodeError, KeyError):
                        pass
                    
                    raise PolymarketAPIError(response.status, error_msg)
                
                # Parse successful response
                if response_text:
                    try:
                        import json
                        return json.loads(response_text)
                    except json.JSONDecodeError:
                        return response_text
                return None
                
        except aiohttp.ClientError as e:
            logger.error(f"Network error during {method} {url}: {e}")
            raise PolymarketError(f"Network error: {e}")
    
    async def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        data: Optional[dict] = None,
        authenticated: bool = False,
    ) -> Any:
        """Make a request with automatic retry on failure."""
        return await retry_with_backoff(
            self._request,
            method,
            endpoint,
            params,
            data,
            authenticated,
            max_retries=self.max_retries,
        )
    
    async def _run_sync(self, func, *args, **kwargs) -> Any:
        """Run a synchronous function in the thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: func(*args, **kwargs)
        )
    
    # =========================================================================
    # Authentication
    # =========================================================================
    
    async def create_or_derive_api_creds(self) -> dict:
        """
        Get or create API credentials for authenticated requests.
        
        This uses the py-clob-client to derive credentials from the
        private key. Must be called before making authenticated requests.
        
        Returns:
            Dictionary containing API credentials
        
        Raises:
            AuthenticationError: If credentials cannot be created
        """
        if not self._clob_client:
            raise AuthenticationError(
                401,
                "Private key required to create API credentials"
            )
        
        try:
            creds = await self._run_sync(
                self._clob_client.create_or_derive_api_creds
            )
            self._api_creds = creds
            logger.info("API credentials created successfully")
            return creds
        except Exception as e:
            logger.error(f"Failed to create API credentials: {e}")
            raise AuthenticationError(401, str(e))
    
    def set_api_creds(self, creds: dict) -> None:
        """
        Set API credentials for authenticated requests.
        
        Args:
            creds: Dictionary containing API credentials
                   (api_key, secret, passphrase, etc.)
        """
        self._api_creds = creds
        if self._clob_client:
            try:
                self._clob_client.set_api_creds(creds)
            except Exception as e:
                logger.warning(f"Failed to set creds on clob client: {e}")
        logger.info("API credentials set")
    
    # =========================================================================
    # Health Check
    # =========================================================================
    
    async def get_ok(self) -> bool:
        """
        Check if the API is healthy.
        
        Returns:
            True if the API is responding, False otherwise
        """
        try:
            response = await self._request("GET", "/")
            return response == "OK" or bool(response)
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    async def get_server_time(self) -> str:
        """
        Get the current server time.
        
        Returns:
            Server time string
        """
        response = await self._request("GET", "/time")
        return str(response)
    
    # =========================================================================
    # Markets
    # =========================================================================
    
    async def get_markets(self, next_cursor: str = "") -> List[dict]:
        """
        Get all markets with pagination.
        
        Args:
            next_cursor: Pagination cursor for fetching more results
        
        Returns:
            List of market dictionaries
        """
        params = {}
        if next_cursor:
            params["next_cursor"] = next_cursor
        
        response = await self._request_with_retry("GET", "/markets", params=params)
        return response if isinstance(response, list) else response.get("data", [])
    
    async def get_simplified_markets(self) -> dict:
        """
        Get simplified market data.
        
        Returns a mapping of token IDs to basic market info.
        
        Returns:
            Dictionary mapping token IDs to market data
        """
        response = await self._request_with_retry("GET", "/simplified-markets")
        return response if isinstance(response, dict) else {}
    
    async def get_market(self, condition_id: str) -> dict:
        """
        Get a single market by condition ID.
        
        Args:
            condition_id: The market's condition ID
        
        Returns:
            Market details dictionary
        
        Raises:
            MarketNotFoundError: If the market doesn't exist
        """
        return await self._request_with_retry("GET", f"/markets/{condition_id}")
    
    # =========================================================================
    # Orderbook
    # =========================================================================
    
    async def get_order_book(self, token_id: str) -> dict:
        """
        Get the orderbook for a token.
        
        Args:
            token_id: The token ID to get the orderbook for
        
        Returns:
            Orderbook with bids and asks
        """
        response = await self._request_with_retry(
            "GET",
            "/book",
            params={"token_id": token_id}
        )
        return parse_orderbook_response(response)
    
    async def get_order_books(self, token_ids: List[str]) -> List[dict]:
        """
        Get orderbooks for multiple tokens.
        
        Args:
            token_ids: List of token IDs
        
        Returns:
            List of orderbooks (same order as input)
        """
        tasks = [self.get_order_book(token_id) for token_id in token_ids]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    # =========================================================================
    # Pricing
    # =========================================================================
    
    async def get_midpoint(self, token_id: str) -> float:
        """
        Get the midpoint price for a token.
        
        The midpoint is the average of the best bid and best ask.
        
        Args:
            token_id: The token ID
        
        Returns:
            Midpoint price as a float
        """
        response = await self._request_with_retry(
            "GET",
            "/midpoint",
            params={"token_id": token_id}
        )
        
        if isinstance(response, dict):
            return float(response.get("mid", response.get("midpoint", 0)))
        return float(response)
    
    async def get_price(self, token_id: str, side: str) -> float:
        """
        Get the price for a specific side.
        
        Args:
            token_id: The token ID
            side: Order side (BUY or SELL)
        
        Returns:
            Price as a float
        """
        side = normalize_side(side)
        response = await self._request_with_retry(
            "GET",
            "/price",
            params={"token_id": token_id, "side": side}
        )
        
        if isinstance(response, dict):
            return float(response.get("price", 0))
        return float(response)
    
    async def get_spread(self, token_id: str) -> dict:
        """
        Get the bid-ask spread for a token.
        
        Args:
            token_id: The token ID
        
        Returns:
            Dictionary with spread metrics:
                - best_bid: Best bid price
                - best_ask: Best ask price
                - spread: Absolute spread
                - spread_percent: Spread as percentage of midpoint
                - midpoint: Midpoint price
        """
        orderbook = await self.get_order_book(token_id)
        return calculate_spread(orderbook)
    
    async def get_last_trade_price(self, token_id: str) -> float:
        """
        Get the last trade price for a token.
        
        Args:
            token_id: The token ID
        
        Returns:
            Last trade price as a float
        """
        response = await self._request_with_retry(
            "GET",
            "/last-trade-price",
            params={"token_id": token_id}
        )
        
        if isinstance(response, dict):
            return float(response.get("price", 0))
        return float(response)
    
    # =========================================================================
    # Orders (Authenticated)
    # =========================================================================
    
    async def create_order(self, order_args: dict) -> dict:
        """
        Create and sign a limit order.
        
        Uses the py-clob-client to properly sign the order.
        
        Args:
            order_args: Order parameters:
                - token_id: Token to trade
                - price: Limit price (0.0 to 1.0)
                - size: Order size in USDC
                - side: BUY or SELL
        
        Returns:
            Signed order ready for submission
        
        Raises:
            AuthenticationError: If not authenticated
            InvalidOrderError: If order parameters are invalid
        """
        if not self._clob_client or not self.is_authenticated:
            raise AuthenticationError(
                401,
                "Authentication required to create orders"
            )
        
        try:
            from py_clob_client.clob_types import OrderArgs
            from py_clob_client.order_builder.constants import BUY, SELL
            
            side = BUY if normalize_side(order_args.get("side", "")) == "BUY" else SELL
            
            args = OrderArgs(
                token_id=order_args["token_id"],
                price=order_args["price"],
                size=order_args["size"],
                side=side,
            )
            
            signed_order = await self._run_sync(
                self._clob_client.create_order, args
            )
            
            logger.info(
                f"Created order: {side} {order_args['size']} @ {order_args['price']}"
            )
            return signed_order
            
        except KeyError as e:
            raise InvalidOrderError(400, f"Missing required field: {e}")
        except Exception as e:
            logger.error(f"Failed to create order: {e}")
            raise OrderError(400, str(e))
    
    async def create_market_order(self, order_args: dict) -> dict:
        """
        Create and sign a market order.
        
        Market orders execute immediately at the best available price.
        
        Args:
            order_args: Order parameters:
                - token_id: Token to trade
                - amount: USDC amount to spend/receive
                - side: BUY or SELL
        
        Returns:
            Signed order ready for submission
        
        Raises:
            AuthenticationError: If not authenticated
            InvalidOrderError: If order parameters are invalid
        """
        if not self._clob_client or not self.is_authenticated:
            raise AuthenticationError(
                401,
                "Authentication required to create orders"
            )
        
        try:
            from py_clob_client.clob_types import MarketOrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY, SELL
            
            side = BUY if normalize_side(order_args.get("side", "")) == "BUY" else SELL
            
            args = MarketOrderArgs(
                token_id=order_args["token_id"],
                amount=order_args["amount"],
                side=side,
                order_type=OrderType.FOK,
            )
            
            signed_order = await self._run_sync(
                self._clob_client.create_market_order, args
            )
            
            logger.info(
                f"Created market order: {side} ${order_args['amount']}"
            )
            return signed_order
            
        except KeyError as e:
            raise InvalidOrderError(400, f"Missing required field: {e}")
        except Exception as e:
            logger.error(f"Failed to create market order: {e}")
            raise OrderError(400, str(e))
    
    async def post_order(self, signed_order: dict, order_type: str = "GTC") -> dict:
        """
        Submit a signed order to the exchange.
        
        Args:
            signed_order: Signed order from create_order or create_market_order
            order_type: Order type (GTC, FOK, GTD)
        
        Returns:
            Order confirmation with order ID
        
        Raises:
            OrderError: If order submission fails
            InsufficientBalanceError: If not enough funds
        """
        if not self._clob_client:
            raise AuthenticationError(
                401,
                "Authentication required to post orders"
            )
        
        try:
            from py_clob_client.clob_types import OrderType
            
            ot = getattr(OrderType, normalize_order_type(order_type))
            
            result = await self._run_sync(
                self._clob_client.post_order, signed_order, ot
            )
            
            logger.info(f"Order posted: {result}")
            return result
            
        except Exception as e:
            error_msg = str(e).lower()
            if "insufficient" in error_msg:
                raise InsufficientBalanceError(400, str(e))
            logger.error(f"Failed to post order: {e}")
            raise OrderError(400, str(e))
    
    async def get_order(self, order_id: str) -> dict:
        """
        Get an order by ID.
        
        Args:
            order_id: The order ID
        
        Returns:
            Order details
        
        Raises:
            OrderNotFoundError: If order doesn't exist
        """
        try:
            return await self._request_with_retry(
                "GET",
                f"/order/{order_id}",
                authenticated=True
            )
        except MarketNotFoundError:
            raise OrderNotFoundError(order_id)
    
    async def get_orders(self, params: Optional[dict] = None) -> List[dict]:
        """
        Get open orders.
        
        Args:
            params: Optional filter parameters:
                - market: Filter by market condition ID
                - state: Filter by state (OPEN, CLOSED, etc.)
        
        Returns:
            List of open orders
        """
        return await self._request_with_retry(
            "GET",
            "/orders",
            params=params,
            authenticated=True
        )
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order by ID.
        
        Args:
            order_id: The order ID to cancel
        
        Returns:
            True if cancelled successfully
        
        Raises:
            OrderNotFoundError: If order doesn't exist
            OrderCancellationError: If cancellation fails
        """
        if not self._clob_client:
            raise AuthenticationError(
                401,
                "Authentication required to cancel orders"
            )
        
        try:
            result = await self._run_sync(self._clob_client.cancel, order_id)
            logger.info(f"Order cancelled: {order_id}")
            return bool(result)
        except Exception as e:
            if "not found" in str(e).lower():
                raise OrderNotFoundError(order_id)
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise OrderCancellationError(400, str(e))
    
    async def cancel_all_orders(self) -> int:
        """
        Cancel all open orders.
        
        Returns:
            Number of orders cancelled
        
        Raises:
            AuthenticationError: If not authenticated
        """
        if not self._clob_client:
            raise AuthenticationError(
                401,
                "Authentication required to cancel orders"
            )
        
        try:
            result = await self._run_sync(self._clob_client.cancel_all)
            count = len(result) if isinstance(result, list) else 0
            logger.info(f"Cancelled {count} orders")
            return count
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            raise OrderCancellationError(400, str(e))
    
    # =========================================================================
    # Trades
    # =========================================================================
    
    async def get_trades(self, params: Optional[dict] = None) -> List[dict]:
        """
        Get trade history.
        
        Args:
            params: Optional filter parameters:
                - market: Filter by market condition ID
                - maker: Filter by maker address
                - taker: Filter by taker address
        
        Returns:
            List of trades
        """
        return await self._request_with_retry(
            "GET",
            "/trades",
            params=params,
            authenticated=True
        )
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_rate_limit_status(self) -> dict:
        """
        Get current rate limit status.
        
        Returns:
            Dictionary with remaining requests and wait time
        """
        return {
            "remaining": self._rate_limiter.get_remaining(),
            "wait_time": self._rate_limiter.get_wait_time(),
            "total_requests": self._request_count,
        }
    
    def __repr__(self) -> str:
        return (
            f"PolymarketClient(host={self.host!r}, "
            f"authenticated={self.is_authenticated})"
        )
