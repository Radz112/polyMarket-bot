import pytest
from src.models import Market
from src.correlation.logical.parser import ThresholdParser
from src.correlation.logical.detector import LogicalRuleDetector
from src.correlation.logical.violations import ViolationDetector
from src.correlation.logical.rules import LogicalRuleType

@pytest.fixture
def parser():
    return ThresholdParser()

@pytest.fixture
def detector():
    return LogicalRuleDetector()

def test_parser_btc(parser):
    q = "Will Bitcoin go above $100,000?"
    info = parser.parse_threshold(q)
    assert info.asset == "BTC"
    assert info.value == 100000
    assert info.direction == "above"

def test_parser_k_suffix(parser):
    q = "Will BTC hit 90k?"
    info = parser.parse_threshold(q)
    assert info.asset == "BTC"
    assert info.value == 90000

def test_simple_ordering(detector):
    m1 = Market(condition_id="1", slug="btc-90k", question="BTC > 90k", entities={})
    m2 = Market(condition_id="2", slug="btc-100k", question="BTC > 100k", entities={})
    
    rules = detector.detect_rules([m1, m2])
    assert len(rules) == 1
    r = rules[0]
    assert r.rule_type == LogicalRuleType.THRESHOLD_ORDERING
    # Lower strike (90k) should correspond to "lower_market" (easier to hit) in metadata?
    # Logic in detector: m_low is index i (lower val).
    # metadata: lower_strike_market = m_low.id
    assert r.metadata["lower_strike_market"] == "1"
    assert r.metadata["higher_strike_market"] == "2"

def test_violation_detection():
    detector = ViolationDetector()
    # Mock rule
    # P(>90k) >= P(>100k)
    # Violation if P(>100k) > P(>90k)
    from src.correlation.logical.rules import LogicalRule
    rule = LogicalRule(
        rule_type=LogicalRuleType.THRESHOLD_ORDERING,
        market_ids=["90k", "100k"],
        constraint_desc="",
        metadata={"lower_strike_market": "90k", "higher_strike_market": "100k", "lower_val": 90000, "higher_val": 100000}
    )
    
    # Case 1: Normal prices
    # 90k at 0.60, 100k at 0.50. No violation.
    prices = {"90k": 0.60, "100k": 0.50}
    v = detector.check_rule(rule, prices)
    assert v is None
    
    # Case 2: Inverted prices (Arbitrage)
    # 90k at 0.50, 100k at 0.60. Violation!
    prices_bad = {"90k": 0.50, "100k": 0.60}
    v_bad = detector.check_rule(rule, prices_bad)
    assert v_bad is not None
    assert v_bad.deviation == pytest.approx(0.10)
    assert v_bad.profit_opportunity == pytest.approx(0.10)
