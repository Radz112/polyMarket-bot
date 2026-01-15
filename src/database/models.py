from datetime import datetime
from typing import List, Optional
from sqlalchemy import String, Float, Boolean, DateTime, ForeignKey, Integer, Enum as SQLEnum, ARRAY, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from src.models.correlation import CorrelationType
from src.correlation.logical.rules import LogicalRuleType
from src.models.signal import SignalType
from src.models.position import Side

class Base(DeclarativeBase):
    pass

class MarketModel(Base):
    __tablename__ = "markets"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    slug: Mapped[str] = mapped_column(String, nullable=False)
    question: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    category: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    subcategory: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    entities: Mapped[dict] = mapped_column(JSONB, default={})
    end_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    outcome: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    clob_token_ids: Mapped[List[str]] = mapped_column(ARRAY(String), default=[])  # YES/NO token IDs for CLOB API

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class MarketCorrelationModel(Base):
    __tablename__ = "market_correlations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_a_id: Mapped[str] = mapped_column(ForeignKey("markets.id"), nullable=False)
    market_b_id: Mapped[str] = mapped_column(ForeignKey("markets.id"), nullable=False)
    correlation_type: Mapped[CorrelationType] = mapped_column(SQLEnum(CorrelationType), nullable=False)
    expected_relationship: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    manual_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Detailed Scores
    detection_methods: Mapped[List[str]] = mapped_column(ARRAY(String), default=[])
    string_similarity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    statistical_correlation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lead_lag_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    leader_market_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    
    historical_correlation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint('market_a_id', 'market_b_id', name='uq_market_pair'),
    )

class PriceSnapshotModel(Base):
    __tablename__ = "price_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[str] = mapped_column(ForeignKey("markets.id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    yes_price: Mapped[float] = mapped_column(Float, nullable=False)
    no_price: Mapped[float] = mapped_column(Float, nullable=False)
    yes_volume: Mapped[float] = mapped_column(Float, default=0.0)
    no_volume: Mapped[float] = mapped_column(Float, default=0.0)

    __table_args__ = (
        Index('idx_market_timestamp', 'market_id', 'timestamp'),
    )

class SignalModel(Base):
    __tablename__ = "signals"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    signal_type: Mapped[SignalType] = mapped_column(SQLEnum(SignalType), nullable=False)
    market_ids: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=False)
    divergence_amount: Mapped[float] = mapped_column(Float, nullable=False)
    expected_value: Mapped[float] = mapped_column(Float, nullable=False)
    actual_value: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default={})
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    acted_on: Mapped[bool] = mapped_column(Boolean, default=False)
    outcome_pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

class TradeModel(Base):
    __tablename__ = "trades"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    market_id: Mapped[str] = mapped_column(ForeignKey("markets.id"), nullable=False)
    side: Mapped[Side] = mapped_column(SQLEnum(Side), nullable=False)
    size: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    fees: Mapped[float] = mapped_column(Float, default=0.0)
    order_id: Mapped[str] = mapped_column(String, nullable=False)
    is_paper: Mapped[bool] = mapped_column(Boolean, default=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

class PositionModel(Base):
    __tablename__ = "positions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    market_id: Mapped[str] = mapped_column(ForeignKey("markets.id"), nullable=False)
    side: Mapped[Side] = mapped_column(SQLEnum(Side), nullable=False)
    size: Mapped[float] = mapped_column(Float, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    realized_pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

class LogicalRuleModel(Base):
    __tablename__ = "logical_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    rule_type: Mapped[LogicalRuleType] = mapped_column(SQLEnum(LogicalRuleType), nullable=False)
    market_ids: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=False)
    constraint_desc: Mapped[str] = mapped_column(String, nullable=False)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default={})
    tolerance: Mapped[float] = mapped_column(Float, default=0.02)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    active: Mapped[bool] = mapped_column(Boolean, default=True)

    __table_args__ = (
        Index('idx_logical_rules_markets', 'market_ids', postgresql_using='gin'),
    )

class RuleViolationModel(Base):
    __tablename__ = "rule_violations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    rule_id: Mapped[int] = mapped_column(ForeignKey("logical_rules.id"), nullable=False)
    
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    deviation: Mapped[float] = mapped_column(Float, nullable=False)
    profit_opportunity: Mapped[float] = mapped_column(Float, nullable=False)
    suggested_trades: Mapped[dict] = mapped_column(JSONB, nullable=False) # List of trade tuples
    details: Mapped[str] = mapped_column(String, nullable=True)
    
    acted_on: Mapped[bool] = mapped_column(Boolean, default=False)
    outcome_profit: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


# Paper Trading Models

class PaperPortfolioModel(Base):
    """Paper trading portfolio state."""
    __tablename__ = "paper_portfolio"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    balance: Mapped[float] = mapped_column(Float, nullable=False)
    initial_balance: Mapped[float] = mapped_column(Float, nullable=False)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    total_fees: Mapped[float] = mapped_column(Float, default=0.0)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class PaperPositionModel(Base):
    """Paper trading positions."""
    __tablename__ = "paper_positions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    market_id: Mapped[str] = mapped_column(String, nullable=False)
    market_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    side: Mapped[str] = mapped_column(String, nullable=False)  # 'YES' or 'NO'
    size: Mapped[float] = mapped_column(Float, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    exit_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    realized_pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String, default="open")  # 'open', 'closed', 'resolved'
    
    signal_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    signal_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class PaperTradeModel(Base):
    """Paper trading trade records."""
    __tablename__ = "paper_trades"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    position_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    market_id: Mapped[str] = mapped_column(String, nullable=False)
    market_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    side: Mapped[str] = mapped_column(String, nullable=False)  # 'YES' or 'NO'
    action: Mapped[str] = mapped_column(String, nullable=False)  # 'BUY' or 'SELL'
    size: Mapped[float] = mapped_column(Float, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    fees: Mapped[float] = mapped_column(Float, nullable=False)
    total: Mapped[float] = mapped_column(Float, nullable=False)
    
    signal_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    signal_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    realized_pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    holding_period_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PaperSnapshotModel(Base):
    """Paper trading portfolio snapshots for equity curve."""
    __tablename__ = "paper_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    cash_balance: Mapped[float] = mapped_column(Float, nullable=False)
    positions_value: Mapped[float] = mapped_column(Float, nullable=False)
    total_value: Mapped[float] = mapped_column(Float, nullable=False)
    unrealized_pnl: Mapped[float] = mapped_column(Float, nullable=False)
    realized_pnl: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        Index('idx_paper_snapshots_time', 'timestamp'),
    )
