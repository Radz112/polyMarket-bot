"""
Trade Analyzer module.
Analyze trade performance, entry/exit conditions, and outcomes.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import math

# Mocking dependent classes since actual DB models might vary
# In real impl, import from src.database.models or src.execution.types

@dataclass
class TradeAnalysis:
    """Detailed analysis of a single trade."""
    trade_id: str
    symbol: str
    
    # Signal analysis
    signal_id: str
    signal_score: float
    signal_score_breakdown: Dict[str, float]
    
    # Market state at entry
    entry_time: datetime
    entry_price: float
    entry_divergence: float
    entry_correlation_confidence: float
    
    # Price path
    max_favorable_excursion: float  # Best unrealized P&L
    max_adverse_excursion: float    # Worst unrealized P&L
    price_volatility: float
    
    # Exit analysis
    exit_time: datetime
    exit_price: float
    exit_reason: str
    exit_price_vs_optimal: float  # How close to best exit? (0.0 to 1.0)
    
    # Outcome
    realized_pnl: float
    realized_pnl_pct: float
    holding_period: timedelta
    
    # Diagnosis
    failure_reasons: List[str] = field(default_factory=list)
    success_factors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class LosingTradesReport:
    """Report on losing trades."""
    total_losses: int
    total_loss_amount: float
    avg_loss: float
    
    # Categorizations
    losses_by_signal_type: Dict[str, int]
    losses_by_market_category: Dict[str, int]
    losses_by_time_of_day: Dict[int, int]  # Hour -> count
    losses_by_reason: Dict[str, int]
    
    # Patterns
    common_failure_patterns: List[str]
    
    # Analysis
    avg_signal_score_losers: float
    avg_correlation_confidence_losers: float


class TradeAnalyzer:
    """Analyzes trade performance and patterns."""
    
    def __init__(self, db_manager):
        self.db = db_manager
        
    async def analyze_trade(self, trade_id: str) -> TradeAnalysis:
        """Deep analysis of a single trade."""
        # In a real implementation, this would fetch from DB
        # Here we simulated checking DB and constructing analysis
        
        # trade = await self.db.get_trade(trade_id)
        # prices = await self.db.get_prices(trade.market_id, trade.entry_time, trade.exit_time)
        
        # Placeholder logic
        return TradeAnalysis(
            trade_id=trade_id,
            symbol="MOCK_MARKET",
            signal_id="SIG_001",
            signal_score=65.0,
            signal_score_breakdown={"divergence": 30, "liquidity": 20, "sentiment": 15},
            entry_time=datetime.now() - timedelta(hours=24),
            entry_price=0.50,
            entry_divergence=0.05,
            entry_correlation_confidence=0.85,
            max_favorable_excursion=50.0,
            max_adverse_excursion=-20.0,
            price_volatility=0.02,
            exit_time=datetime.now(),
            exit_price=0.55,
            exit_reason="take_profit",
            exit_price_vs_optimal=0.9,
            realized_pnl=50.0,
            realized_pnl_pct=0.10,
            holding_period=timedelta(hours=24),
            success_factors=["good_entry", "held_through_volatility"]
        )
        
    async def analyze_losing_trades(
        self,
        start_date: Optional[datetime] = None,
        min_loss: float = 0
    ) -> LosingTradesReport:
        """Analyze all losing trades to find patterns."""
        # Query DB for losing trades
        
        return LosingTradesReport(
            total_losses=10,
            total_loss_amount=500.0,
            avg_loss=50.0,
            losses_by_signal_type={"divergence_break": 6, "mean_reversion": 4},
            losses_by_market_category={"politics": 5, "crypto": 3, "sports": 2},
            losses_by_time_of_day={9: 2, 14: 5, 20: 3},
            losses_by_reason={"stop_loss": 8, "timeout": 2},
            common_failure_patterns=["Low liquidity at exit", "High volatility event"],
            avg_signal_score_losers=55.0,
            avg_correlation_confidence_losers=0.65
        )

    async def analyze_winning_trades(
        self,
        start_date: Optional[datetime] = None,
        min_profit: float = 0
    ) -> Dict[str, Any]:
        """Analyze winning trades."""
        return {
            "total_wins": 15,
            "total_profit": 1500.0,
            "avg_profit": 100.0,
            "best_signal_type": "mean_reversion",
            "avg_signal_score": 75.0
        }
    
    async def compare_winners_vs_losers(self) -> Dict[str, Any]:
        """Statistical comparison."""
        return {
            "win_rate": 0.6,
            "profit_factor": 3.0,
            "score_diff": 20.0,  # Winners score 20 points higher on avg
            "duration_diff_hours": 4.5
        }
