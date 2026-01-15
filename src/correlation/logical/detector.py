from typing import List, Dict
from src.models import Market
from .rules import LogicalRule, LogicalRuleType
from .parser import ThresholdParser

class LogicalRuleDetector:
    def __init__(self):
        self.parser = ThresholdParser()

    def detect_rules(self, markets: List[Market]) -> List[LogicalRule]:
        rules = []
        rules.extend(self.detect_threshold_ordering(markets))
        # rules.extend(self.detect_mutually_exclusive(markets)) # To be implemented
        return rules

    def detect_threshold_ordering(self, markets: List[Market]) -> List[LogicalRule]:
        """
        Detects ordering constraints.
        If Asset Price > X implies Asset Price > Y (where X > Y).
        Then P(>X) <= P(>Y).
        Violation if Price(>X) > Price(>Y). (Higher strike is more expensive?)
        Wait:
        If BTC > 100k, surely BTC > 90k.
        So P(>100k) should be <= P(>90k).
        Correct.
        """
        groups = self.parser.group_markets(markets)
        rules = []
        
        for asset, items in groups.items():
            # items are sorted by value: 90k, 100k, 110k
            # for "above" direction:
            # 90k is easiest (matches all >90), 110k is hardest.
            # So P(90k) >= P(100k) >= P(110k).
            # We can create pairwise rules.
            
            # Filter for "above" direction only for now
            items_above = [x for x in items if x[1].direction == "above"]
            
            for i in range(len(items_above) - 1):
                m_low, info_low = items_above[i]
                m_high, info_high = items_above[i+1]
                
                # Rule: Price(Low Strike) >= Price(High Strike)
                # m_low (90k) >= m_high (100k)
                
                rule = LogicalRule(
                    rule_type=LogicalRuleType.THRESHOLD_ORDERING,
                    market_ids=[m_low.id, m_high.id],
                    constraint_desc=f"P({m_low.slug}) >= P({m_high.slug})",
                    metadata={
                        "asset": asset,
                        "lower_strike_market": m_low.id, # The one that should be MORE expensive (easier to hit)
                        "higher_strike_market": m_high.id,
                        "lower_val": info_low.value,
                        "higher_val": info_high.value
                    }
                )
                rules.append(rule)
                
        return rules
