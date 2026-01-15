import logging
import asyncio
import pytest
from src.models import Market, MarketCorrelation, CorrelationType
from src.correlation.merger import CorrelationMerger
from src.correlation.graph import CorrelationGraph
from src.correlation.logical.rules import LogicalRule, LogicalRuleType

@pytest.fixture
def merger():
    return CorrelationMerger()

def test_merger_logic(merger):
    # 1. String Match
    c1 = MarketCorrelation(
        market_a_id="1", market_b_id="2", correlation_type=CorrelationType.POSITIVE,
        expected_relationship="", confidence=0.0, metadata={}
    )
    str_matches = [(c1, 0.8)]
    
    # 2. Stat Match
    c2 = MarketCorrelation(
        market_a_id="1", market_b_id="2", correlation_type=CorrelationType.POSITIVE,
        expected_relationship="", confidence=0.0, metadata={}
    )
    stat_matches = [(c2, 0.9)]
    
    # 3. Logical Rule
    rule = LogicalRule(
        rule_type=LogicalRuleType.THRESHOLD_ORDERING,
        market_ids=["2", "3"],
        constraint_desc="P(2)>=P(3)"
    )
    
    merged = merger.merge_detections(str_matches, stat_matches, [rule])
    
    assert len(merged) == 2 # 1-2, and 2-3
    
    # Check 1-2 (Combined)
    m12 = next(m for m in merged if "1" in [m.market_a_id, m.market_b_id] and "2" in [m.market_a_id, m.market_b_id])
    assert "string" in m12.metadata["detection_methods"]
    assert "statistical" in m12.metadata["detection_methods"]
    assert m12.confidence > 0.8 # Boosted
    
    # Check 2-3 (Logical)
    m23 = next(m for m in merged if "3" in [m.market_a_id, m.market_b_id])
    assert "logical" in m23.metadata["detection_methods"]
    assert m23.confidence == 1.0

def test_graph_clusters():
    c1 = MarketCorrelation(
        market_a_id="1", market_b_id="2", correlation_type=CorrelationType.POSITIVE, 
        expected_relationship="", confidence=0.9, metadata={}
    )
    c2 = MarketCorrelation(
        market_a_id="2", market_b_id="3", correlation_type=CorrelationType.POSITIVE, 
        expected_relationship="", confidence=0.9, metadata={}
    )
    c3 = MarketCorrelation(
        market_a_id="4", market_b_id="5", correlation_type=CorrelationType.POSITIVE, 
        expected_relationship="", confidence=0.9, metadata={}
    )
    
    graph = CorrelationGraph([c1, c2, c3])
    clusters = graph.get_clusters()
    
    # Should be {1,2,3} and {4,5}
    assert len(clusters) == 2
    sizes = sorted([len(c) for c in clusters])
    assert sizes == [2, 3]
