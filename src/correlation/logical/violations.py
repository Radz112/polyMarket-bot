from typing import List, Dict, Optional
from src.models import PriceSnapshot
from .rules import LogicalRule, LogicalRuleType, RuleViolation

class ViolationDetector:
    def __init__(self):
        pass

    def check_rule(self, rule: LogicalRule, prices: Dict[str, float]) -> Optional[RuleViolation]:
        """
        Check if current prices violate the rule.
        prices: dict of market_id -> yes_price
        """
        if rule.rule_type == LogicalRuleType.THRESHOLD_ORDERING:
            return self._check_threshold_ordering(rule, prices)
        return None

    def _check_threshold_ordering(self, rule: LogicalRule, prices: Dict[str, float]) -> Optional[RuleViolation]:
        # Metadata: lower_strike_market, higher_strike_market
        # Logic: P(lower_strike) >= P(higher_strike)
        # e.g. P(>90k) >= P(>100k)
        
        low_id = rule.metadata["lower_strike_market"]
        high_id = rule.metadata["higher_strike_market"]
        
        if low_id not in prices or high_id not in prices:
            return None
            
        p_low = prices[low_id]
        p_high = prices[high_id]
        
        # Violation if P(high) > P(low) + tolerance
        # Higher strike trading HIGHER than lower strike.
        # e.g. >100k is 0.60, >90k is 0.50
        # Buy >90k, Sell >100k.
        # Cost: 0.50. Income: 0.60. Net: +0.10 right now?
        # If BTC lands > 100k: Payoff >90k=1, Payoff >100k=-1. Net 0.
        # If BTC lands < 90k: Payoff 0, Payoff 0. Net 0.
        # If BTC lands 90-100k: Payoff >90k=1, Payoff >100k=0. Net +1.
        # Arbitrage is guaranteed.
        
        diff = p_high - p_low
        if diff > rule.tolerance:
            return RuleViolation(
                rule=rule,
                deviation=diff,
                profit_opportunity=diff, # This is immediate credit arb? Or guaranteed profit?
                suggested_trades=[
                    (low_id, "YES", 1.0), # Buy cheaper (prob 1 if other hits)
                    (high_id, "NO", 1.0)  # Short expensive? (Shorting usually means splitting NO?)
                    # Simplification: Shorting YES means Buying NO.
                    # P(YES) + P(NO) = 1 (ignoring spread)
                    # If P_high_YES > P_low_YES
                    # Then 1 - P_high_NO > 1 - P_low_NO
                    # P_low_NO > P_high_NO
                    
                    # Strategy: Buy Low_YES, Buy High_NO.
                    # Cost: P_low + (1 - P_high)
                    # = 1 + P_low - P_high
                    # Since P_high > P_low, Cost < 1.
                    # Payoff:
                    # >100k: Low(YES)=1, High(NO)=0. Sum=1.
                    # <90k: Low(YES)=0, High(NO)=1. Sum=1.
                    # 90-100k: Low(YES)=1, High(NO)=1. Sum=2.
                    # Min payoff is 1. Cost < 1. Risk free.
                ],
                details=f"Violation: Higher strike ({rule.metadata['higher_val']}) price {p_high:.2f} > Lower strike ({rule.metadata['lower_val']}) price {p_low:.2f}"
            )
        return None
