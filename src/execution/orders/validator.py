"""
Order validation for trade execution.
"""
import logging
from typing import Optional

from src.config.settings import Config
from src.execution.orders.models import Order, OrderType, ValidationResult
from src.execution.positions import PositionManager

logger = logging.getLogger(__name__)


class OrderValidator:
    """
    Validates orders before execution.
    
    Checks:
    - Size limits
    - Price validity
    - Balance sufficiency
    - Position availability
    - Market tradability
    - Risk limits
    """
    
    # Default limits
    MIN_ORDER_SIZE = 1.0
    MAX_ORDER_SIZE = 10000.0
    MAX_PRICE_DEVIATION = 0.20  # 20% from mid
    
    def __init__(
        self,
        config: Config,
        position_manager: Optional[PositionManager] = None
    ):
        """
        Initialize validator.
        
        Args:
            config: Application configuration
            position_manager: For position-based validation
        """
        self.config = config
        self.position_manager = position_manager
        
        # From config
        self.max_position_pct = config.max_position_size_pct
        self.max_daily_loss_pct = config.max_daily_loss_pct
    
    def validate(self, order: Order, balance: float = None) -> ValidationResult:
        """
        Run all validations.
        
        Args:
            order: Order to validate
            balance: Current available balance
            
        Returns:
            ValidationResult with errors/warnings
        """
        result = ValidationResult(is_valid=True)
        
        # Size validation
        size_result = self.validate_size(order)
        result.merge(size_result)
        
        # Price validation
        price_result = self.validate_price(order)
        result.merge(price_result)
        
        # Balance validation (for buys)
        if balance is not None and order.action == "BUY":
            balance_result = self.validate_balance(order, balance)
            result.merge(balance_result)
        
        # Position validation (for sells)
        if order.action == "SELL" and self.position_manager:
            position_result = self.validate_position(order)
            result.merge(position_result)
        
        # Risk limits
        risk_result = self.validate_risk_limits(order)
        result.merge(risk_result)
        
        return result
    
    def validate_size(self, order: Order) -> ValidationResult:
        """
        Validate order size.
        
        Checks:
        - Minimum order size
        - Maximum order size
        """
        result = ValidationResult(is_valid=True)
        
        if order.size < self.MIN_ORDER_SIZE:
            result.add_error(
                f"Order size {order.size} below minimum {self.MIN_ORDER_SIZE}"
            )
        
        if order.size > self.MAX_ORDER_SIZE:
            result.add_error(
                f"Order size {order.size} exceeds maximum {self.MAX_ORDER_SIZE}"
            )
        
        if order.size <= 0:
            result.add_error("Order size must be positive")
        
        return result
    
    def validate_price(self, order: Order) -> ValidationResult:
        """
        Validate order price.
        
        Checks:
        - Price in valid range (0-1)
        - Limit price reasonable
        """
        result = ValidationResult(is_valid=True)
        
        if order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                result.add_error("Limit order requires limit_price")
                return result
            
            if not (0 < order.limit_price < 1):
                result.add_error(
                    f"Limit price {order.limit_price} must be between 0 and 1"
                )
            
            # Warning for extreme prices
            if order.limit_price < 0.05:
                result.add_warning(
                    f"Low limit price {order.limit_price} - may be hard to fill"
                )
            
            if order.limit_price > 0.95:
                result.add_warning(
                    f"High limit price {order.limit_price} - may be hard to fill"
                )
        
        return result
    
    def validate_balance(self, order: Order, balance: float) -> ValidationResult:
        """
        Validate sufficient balance for buy orders.
        
        Args:
            order: Order to validate
            balance: Available balance
        """
        result = ValidationResult(is_valid=True)
        
        if order.action != "BUY":
            return result
        
        # Estimate cost
        price = order.limit_price or 0.5  # Use mid if no limit
        estimated_cost = order.size * price * 1.02  # 2% for fees
        
        if estimated_cost > balance:
            result.add_error(
                f"Insufficient balance: need ${estimated_cost:.2f}, have ${balance:.2f}"
            )
        elif estimated_cost > balance * 0.9:
            result.add_warning(
                f"Order uses {(estimated_cost/balance)*100:.1f}% of available balance"
            )
        
        return result
    
    def validate_position(self, order: Order) -> ValidationResult:
        """
        Validate sufficient position for sell orders.
        
        Args:
            order: Order to validate
        """
        result = ValidationResult(is_valid=True)
        
        if order.action != "SELL":
            return result
        
        if not self.position_manager:
            result.add_warning("Cannot verify position - no position manager")
            return result
        
        position = self.position_manager.get_position(order.market_id, order.side)
        
        if position is None:
            result.add_error(
                f"No position to sell in {order.market_id} {order.side}"
            )
        elif position.size < order.size:
            result.add_error(
                f"Insufficient position: have {position.size}, selling {order.size}"
            )
        
        return result
    
    def validate_market(
        self,
        order: Order,
        market_active: bool = True,
        market_resolved: bool = False,
        has_liquidity: bool = True
    ) -> ValidationResult:
        """
        Validate market is tradeable.
        
        Args:
            order: Order to validate
            market_active: Whether market is active
            market_resolved: Whether market has resolved
            has_liquidity: Whether market has liquidity
        """
        result = ValidationResult(is_valid=True)
        
        if not market_active:
            result.add_error(f"Market {order.market_id} is not active")
        
        if market_resolved:
            result.add_error(f"Market {order.market_id} has already resolved")
        
        if not has_liquidity:
            result.add_warning(f"Market {order.market_id} has low liquidity")
        
        return result
    
    def validate_risk_limits(self, order: Order) -> ValidationResult:
        """
        Validate order against risk limits.
        
        Checks:
        - Max position per market
        - Max exposure per category
        """
        result = ValidationResult(is_valid=True)
        
        if not self.position_manager:
            return result
        
        # Check max position size per market
        total_exposure = self.position_manager.get_total_exposure()
        new_exposure = order.size * (order.limit_price or 0.5)
        
        if total_exposure > 0:
            position_pct = new_exposure / total_exposure
            if position_pct > self.max_position_pct:
                result.add_warning(
                    f"Order is {position_pct*100:.1f}% of portfolio "
                    f"(limit: {self.max_position_pct*100:.1f}%)"
                )
        
        return result
    
    def quick_validate(self, order: Order) -> bool:
        """
        Quick validation for basic checks only.
        
        Returns True if order passes basic checks.
        """
        if order.size <= 0:
            return False
        
        if order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                return False
            if not (0 < order.limit_price < 1):
                return False
        
        return True
