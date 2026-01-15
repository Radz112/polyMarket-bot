"""
Verification tests for Pydantic data models.
"""

import pytest
from datetime import datetime, timedelta
from src.models import (
    Market, Token, MarketSummary,
    Orderbook, OrderbookEntry,
    PriceSnapshot, PriceHistory,
    MarketCorrelation, MarketGroup, CorrelationType,
    Signal, ScoredSignal, SignalType,
    Position, Trade, Side
)

class TestMarketModels:
    def test_market_creation(self):
        token = Token(token_id="123", outcome="YES", price=0.5)
        market = Market(
            condition_id="m123",
            slug="will-it-rain",
            question="Will it rain?",
            tokens=[token]
        )
        assert market.id == "m123"
        assert market.active is True
        assert len(market.tokens) == 1
        assert market.tokens[0].token_id == "123"

    def test_market_from_api_response(self):
        data = {
            "condition_id": "m456",
            "slug": "btc-price",
            "question": "BTC > 100k?",
            "tokens": [
                {"token_id": "t1", "outcome": "YES", "price": 0.3},
                {"token_id": "t2", "outcome": "NO", "price": 0.7}
            ]
        }
        market = Market.from_api_response(data)
        assert market.id == "m456"
        assert len(market.tokens) == 2
        assert market.tokens[0].outcome == "YES"

class TestOrderbookModels:
    def test_orderbook_computed_properties(self):
        ob = Orderbook(
            market_id="m1",
            token_id="t1",
            bids=[OrderbookEntry(price=0.40, size=100)],
            asks=[OrderbookEntry(price=0.50, size=100)]
        )
        assert ob.best_bid == 0.40
        assert ob.best_ask == 0.50
        assert ob.spread == pytest.approx(0.10)
        assert ob.mid_price == 0.45
    
    def test_empty_orderbook(self):
        ob = Orderbook(market_id="m1", token_id="t1")
        assert ob.best_bid is None
        assert ob.best_ask is None
        assert ob.spread is None
        assert ob.mid_price is None

    def test_from_api_response(self):
        data = {
            "market": "m2",
            "asset_id": "t2",
            "bids": [("0.45", "100")],
            "asks": [("0.55", "50")],
            "timestamp": "1672531200000"  # ms timestamp
        }
        ob = Orderbook.from_api_response(data)
        assert ob.market_id == "m2"
        assert len(ob.bids) == 1
        assert ob.bids[0].price == 0.45
        assert isinstance(ob.timestamp, datetime)

class TestPriceModels:
    def test_price_history(self):
        now = datetime.utcnow()
        p1 = PriceSnapshot(
            market_id="m1", token_id="t1", 
            timestamp=now - timedelta(hours=1),
            yes_price=0.4, no_price=0.6
        )
        p2 = PriceSnapshot(
            market_id="m1", token_id="t1",
            timestamp=now,
            yes_price=0.5, no_price=0.5
        )
        history = PriceHistory(market_id="m1", snapshots=[p1, p2])
        
        # Test get_price_at
        at_p1 = history.get_price_at(now - timedelta(hours=1))
        assert at_p1 == p1
        
        at_p2 = history.get_price_at(now)
        assert at_p2 == p2
        
        # Test range
        in_range = history.get_range(now - timedelta(minutes=30), now + timedelta(minutes=1))
        assert len(in_range) == 1
        assert in_range[0] == p2

class TestPositionModels:
    def test_position_pnl(self):
        pos = Position(
            id="p1", market_id="m1",
            side=Side.YES, size=100,
            entry_price=0.50,
            current_price=0.60,
            opened_at=datetime.utcnow()
        )
        assert pos.market_value == 60.0
        assert pos.cost_basis == 50.0
        assert pos.unrealized_pnl == 10.0
        assert pos.unrealized_pnl_pct == 20.0
        
    def test_zero_cost_basis(self):
        pos = Position(
            id="p2", market_id="m2", side=Side.NO, size=10,
            entry_price=0, current_price=0.5,
            opened_at=datetime.utcnow()
        )
        assert pos.unrealized_pnl_pct == 0.0

class TestSignalModels:
    def test_signal_creation(self):
        sig = Signal(
            id="s1",
            signal_type=SignalType.DIVERGENCE,
            market_ids=["m1", "m2"],
            divergence_amount=0.1,
            expected_value=0.5,
            actual_value=0.6,
            confidence=0.8,
            score=85
        )
        assert sig.signal_type == "divergence"
        assert sig.score == 85

class TestCorrelationModels:
    def test_correlation_creation(self):
        corr = MarketCorrelation(
            market_a_id="m1", market_b_id="m2",
            correlation_type=CorrelationType.INVERSE,
            expected_relationship="A+B=1",
            confidence=0.95
        )
        assert corr.correlation_type == "inverse"
