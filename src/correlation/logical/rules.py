from dataclasses import dataclass, field
from enum import Enum
from typing import List, Callable, Dict, Optional, Tuple, Any

class LogicalRuleType(Enum):
    THRESHOLD_ORDERING = "threshold_ordering"  # BTC > $100K implies BTC > $90K
    SUBSET_SUPERSET = "subset_superset"        # Win PA contributes to Win National
    MUTUALLY_EXCLUSIVE = "mutually_exclusive"  # Trump wins XOR Biden wins
    EXHAUSTIVE = "exhaustive"                  # All outcomes sum to 100%
    TEMPORAL = "temporal"                      # Before X implies by Y
    CONDITIONAL = "conditional"                # If A then likely B

@dataclass
class LogicalRule:
    rule_type: LogicalRuleType
    market_ids: List[str]
    constraint_desc: str  # Human-readable: "P(A) <= P(B)"
    tolerance: float = 0.02 # Allowed deviation (e.g., 2 cents)
    confidence: float = 1.0
    
    # Metadata for execution
    # e.g. for threshold ordering: {"lower_market": id, "higher_market": id}
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RuleViolation:
    rule: LogicalRule
    deviation: float
    profit_opportunity: float  # Guaranteed profit
    # List of (market_id, side 'YES'/'NO', amount)
    # e.g. [("market_A", "YES", 1.0), ("market_B", "NO", 1.0)]
    suggested_trades: List[Tuple[str, str, float]] 
    details: str
