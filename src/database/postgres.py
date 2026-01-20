import logging
import os
from datetime import datetime
from typing import List, Optional, Any
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select, update, delete, text
from sqlalchemy.exc import SQLAlchemyError

from src.database.models import (
    Base, MarketModel, MarketCorrelationModel, PriceSnapshotModel,
    SignalModel, TradeModel, PositionModel
)
from src.models import (
    Market, MarketCorrelation, PriceSnapshot, Signal, Trade, Position
)

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, config=None):
        self._engine = None
        self._session_factory = None
        self.config = config

    async def connect(self):
        """Initialize the database connection pool."""
        if self._engine:
            return

        if self.config:
            db_user = self.config.postgres_user
            db_password = self.config.postgres_password
            db_host = self.config.postgres_host
            db_port = self.config.postgres_port
            db_name = self.config.postgres_db
        else:
            db_user = os.getenv("POSTGRES_USER", "postgres")
            db_password = os.getenv("POSTGRES_PASSWORD", "postgres")
            db_host = os.getenv("POSTGRES_HOST", "localhost")
            db_port = os.getenv("POSTGRES_PORT", "5432")
            db_name = os.getenv("POSTGRES_DB", "polymarket_bot")
        
        database_url = f"postgresql+asyncpg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        try:
            self._engine = create_async_engine(database_url, echo=False)
            self._session_factory = async_sessionmaker(
                self._engine, expire_on_commit=False, class_=AsyncSession
            )
            logger.info("Connected to PostgreSQL")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    async def disconnect(self):
        """Close the database connection pool."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Disconnected from PostgreSQL")

    async def health_check(self) -> bool:
        """Check database connectivity."""
        if not self._engine:
            return False
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            return False

    # --- Markets ---

    async def upsert_market(self, market: Market):
        """Insert or update a market."""
        async with self._session_factory() as session:
            try:
                # Check existence
                stmt = select(MarketModel).where(MarketModel.id == market.id)
                result = await session.execute(stmt)
                existing = result.scalars().first()

                if existing:
                    existing.slug = market.slug
                    existing.question = market.question
                    existing.description = market.description
                    existing.category = market.category
                    existing.subcategory = market.subcategory
                    existing.entities = market.entities
                    existing.end_date = market.end_date
                    existing.active = market.active
                    existing.resolved = market.resolved
                    existing.outcome = market.outcome
                    existing.clob_token_ids = market.clob_token_ids
                else:
                    new_market = MarketModel(
                        id=market.id,
                        slug=market.slug,
                        question=market.question,
                        description=market.description,
                        category=market.category,
                        subcategory=market.subcategory,
                        entities=market.entities,
                        end_date=market.end_date,
                        active=market.active,
                        resolved=market.resolved,
                        outcome=market.outcome,
                        clob_token_ids=market.clob_token_ids
                    )
                    session.add(new_market)
                await session.commit()
            except SQLAlchemyError as e:
                logger.error(f"Error upserting market {market.id}: {e}")
                await session.rollback()
                raise

    async def upsert_markets(self, markets: List[Market]):
        # Naive implementation loop for now, optimize with bulk operations later if needed
        for m in markets:
            await self.upsert_market(m)

    async def get_market(self, market_id: str) -> Optional[Market]:
        async with self._session_factory() as session:
            result = await session.execute(select(MarketModel).where(MarketModel.id == market_id))
            model = result.scalars().first()
            if model:
                # Conversion logic from ORM to Pydantic
                return Market(
                    condition_id=model.id,
                    slug=model.slug,
                    question=model.question,
                    description=model.description,
                    category=model.category,
                    subcategory=model.subcategory,
                    entities=model.entities or {},
                    end_date=model.end_date,
                    active=model.active,
                    closed=not model.active,
                    resolved=model.resolved,
                    outcome=model.outcome,
                    clob_token_ids=model.clob_token_ids or []
                )
            return None

    async def get_markets_by_category(self, category: str) -> List[Market]:
        async with self._session_factory() as session:
            stmt = select(MarketModel).where(MarketModel.category == category)
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [
                 Market(
                    condition_id=m.id,
                    slug=m.slug,
                    question=m.question,
                    description=m.description,
                    category=m.category,
                    subcategory=m.subcategory,
                    entities=m.entities or {},
                    end_date=m.end_date,
                    active=m.active,
                    closed=not m.active,
                    resolved=m.resolved,
                    outcome=m.outcome
                ) for m in models
            ]
    async def get_active_markets(self) -> List[Market]:
        async with self._session_factory() as session:
            result = await session.execute(select(MarketModel).where(MarketModel.active == True))
            models = result.scalars().all()
            return [
                 Market(
                    condition_id=m.id,
                    slug=m.slug,
                    question=m.question,
                    description=m.description,
                    category=m.category,
                    subcategory=m.subcategory,
                    entities=m.entities or {},
                    end_date=m.end_date,
                    active=m.active,
                    closed=not m.active,
                    resolved=m.resolved,
                    outcome=m.outcome
                ) for m in models
            ]
            
    async def search_markets(self, query: str) -> List[Market]:
        async with self._session_factory() as session:
            stmt = select(MarketModel).where(
                MarketModel.question.ilike(f"%{query}%") | MarketModel.slug.ilike(f"%{query}%")
            )
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [
                Market(
                    condition_id=m.id,
                    slug=m.slug,
                    question=m.question,
                    description=m.description,
                    category=m.category,
                    subcategory=m.subcategory,
                    end_date=m.end_date,
                    active=m.active,
                    resolved=m.resolved,
                    outcome=m.outcome,
                    tokens=[]
                ) for m in models
            ]

    # --- Correlations ---

    async def upsert_correlation(self, correlation: MarketCorrelation) -> None:
        async with self._session_factory() as session:
            try:
                stmt = select(MarketCorrelationModel).where(
                    MarketCorrelationModel.market_a_id == correlation.market_a_id,
                    MarketCorrelationModel.market_b_id == correlation.market_b_id
                )
                result = await session.execute(stmt)
                existing = result.scalars().first()
                
                if existing:
                    existing.correlation_type = correlation.correlation_type
                    existing.expected_relationship = correlation.expected_relationship
                    existing.confidence = correlation.confidence
                    existing.manual_verified = correlation.manual_verified
                    existing.historical_correlation = correlation.historical_correlation
                else:
                    new_corr = MarketCorrelationModel(
                        market_a_id=correlation.market_a_id,
                        market_b_id=correlation.market_b_id,
                        correlation_type=correlation.correlation_type,
                        expected_relationship=correlation.expected_relationship,
                        confidence=correlation.confidence,
                        manual_verified=correlation.manual_verified,
                        historical_correlation=correlation.historical_correlation
                    )
                    session.add(new_corr)
                await session.commit()
            except SQLAlchemyError as e:
                logger.error(f"Error upserting correlation: {e}")
                await session.rollback()

    async def get_correlations_for_market(self, market_id: str) -> List[MarketCorrelation]:
        async with self._session_factory() as session:
            stmt = select(MarketCorrelationModel).where(
                (MarketCorrelationModel.market_a_id == market_id) | 
                (MarketCorrelationModel.market_b_id == market_id)
            )
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [
                MarketCorrelation(
                    id=str(m.id),
                    market_a_id=m.market_a_id,
                    market_b_id=m.market_b_id,
                    correlation_type=m.correlation_type,
                    expected_relationship=m.expected_relationship,
                    confidence=m.confidence,
                    manual_verified=m.manual_verified,
                    historical_correlation=m.historical_correlation,
                    created_at=m.created_at,
                    updated_at=m.updated_at
                ) for m in models
            ]

    async def get_all_correlations(self) -> List[MarketCorrelation]:
        async with self._session_factory() as session:
            stmt = select(MarketCorrelationModel)
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [
                MarketCorrelation(
                    id=str(m.id), # Model id is int, Pydantic expects str optional
                    market_a_id=m.market_a_id,
                    market_b_id=m.market_b_id,
                    correlation_type=m.correlation_type,
                    expected_relationship=m.expected_relationship,
                    confidence=m.confidence,
                    manual_verified=m.manual_verified,
                    historical_correlation=m.historical_correlation,
                    created_at=m.created_at,
                    updated_at=m.updated_at
                ) for m in models
            ]

    async def delete_correlation(self, market_a_id: str, market_b_id: str) -> None:
        async with self._session_factory() as session:
            try:
                stmt = delete(MarketCorrelationModel).where(
                    MarketCorrelationModel.market_a_id == market_a_id,
                    MarketCorrelationModel.market_b_id == market_b_id
                )
                await session.execute(stmt)
                await session.commit()
            except SQLAlchemyError as e:
                logger.error(f"Error deleting correlation: {e}")
                await session.rollback()

    # --- Prices ---

    async def save_price_snapshot(self, snapshot: PriceSnapshot) -> None:
        async with self._session_factory() as session:
            try:
                new_snap = PriceSnapshotModel(
                    market_id=snapshot.market_id,
                    timestamp=snapshot.timestamp,
                    yes_price=snapshot.yes_price,
                    no_price=snapshot.no_price,
                    yes_volume=snapshot.volume or 0.0,
                    no_volume=0.0 # Standardize volume field usage
                )
                session.add(new_snap)
                await session.commit()
            except SQLAlchemyError as e:
                logger.error(f"Error saving price snapshot: {e}")
                await session.rollback()

    async def save_price_snapshots(self, snapshots: List[PriceSnapshot]) -> None:
        async with self._session_factory() as session:
            try:
                for snapshot in snapshots:
                    new_snap = PriceSnapshotModel(
                        market_id=snapshot.market_id,
                        timestamp=snapshot.timestamp,
                        yes_price=snapshot.yes_price,
                        no_price=snapshot.no_price,
                        yes_volume=snapshot.volume or 0.0,
                        no_volume=0.0
                    )
                    session.add(new_snap)
                await session.commit()
            except SQLAlchemyError as e:
                logger.error(f"Error saving price snapshots: {e}")
                await session.rollback()

    async def get_price_history(self, market_id: str, start: datetime, end: datetime) -> List[PriceSnapshot]:
        async with self._session_factory() as session:
            stmt = select(PriceSnapshotModel).where(
                PriceSnapshotModel.market_id == market_id,
                PriceSnapshotModel.timestamp >= start,
                PriceSnapshotModel.timestamp <= end
            ).order_by(PriceSnapshotModel.timestamp.asc())
            
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [
                PriceSnapshot(
                    market_id=m.market_id,
                    token_id="", # Model might not store token_id explicitly if related to market
                    timestamp=m.timestamp,
                    yes_price=m.yes_price,
                    no_price=m.no_price,
                    volume=m.yes_volume + (m.no_volume or 0.0)
                ) for m in models
            ]

    async def get_latest_price(self, market_id: str) -> Optional[PriceSnapshot]:
        async with self._session_factory() as session:
            stmt = select(PriceSnapshotModel).where(
                PriceSnapshotModel.market_id == market_id
            ).order_by(PriceSnapshotModel.timestamp.desc()).limit(1)
            result = await session.execute(stmt)
            model = result.scalars().first()
            if model:
                return PriceSnapshot(
                    market_id=model.market_id,
                    token_id="",
                    timestamp=model.timestamp,
                    yes_price=model.yes_price,
                    no_price=model.no_price,
                    volume=model.yes_volume + (model.no_volume or 0.0)
                )
            return None

    # --- Signals ---

    async def save_signal(self, signal: Signal) -> None:
        async with self._session_factory() as session:
            try:
                new_sig = SignalModel(
                    id=signal.id,
                    signal_type=signal.signal_type,
                    market_ids=signal.market_ids,
                    divergence_amount=signal.divergence_amount,
                    expected_value=signal.expected_value,
                    actual_value=signal.actual_value,
                    confidence=signal.confidence,
                    score=signal.score,
                    metadata_=signal.metadata
                )
                session.add(new_sig)
                await session.commit()
            except SQLAlchemyError as e:
                logger.error(f"Error saving signal: {e}")
                await session.rollback()

    async def get_recent_signals(self, limit: int = 100) -> List[Signal]:
        async with self._session_factory() as session:
            stmt = select(SignalModel).order_by(SignalModel.created_at.desc()).limit(limit)
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [
                Signal(
                    id=m.id,
                    signal_type=m.signal_type,
                    market_ids=m.market_ids,
                    divergence_amount=m.divergence_amount,
                    expected_value=m.expected_value,
                    actual_value=m.actual_value,
                    confidence=m.confidence,
                    score=m.score,
                    metadata=m.metadata_ or {}
                ) for m in models
            ]

    async def mark_signal_acted(self, signal_id: str, outcome: float) -> None:
        async with self._session_factory() as session:
            try:
                stmt = select(SignalModel).where(SignalModel.id == signal_id)
                result = await session.execute(stmt)
                existing = result.scalars().first()
                if existing:
                    existing.acted_on = True
                    existing.outcome_pnl = outcome
                    await session.commit()
            except SQLAlchemyError as e:
                logger.error(f"Error marking signal acted: {e}")
                await session.rollback()

    # --- Trades & Positions ---
    
    async def save_trade(self, trade: Trade) -> None:
         async with self._session_factory() as session:
            try:
                new_trade = TradeModel(
                    id=trade.id,
                    market_id=trade.market_id,
                    side=trade.side,
                    size=trade.size,
                    price=trade.price,
                    fees=trade.fees,
                    order_id=trade.order_id,
                    is_paper=trade.is_paper,
                    created_at=trade.timestamp
                )
                session.add(new_trade)
                await session.commit()
            except Exception as e:
                logger.error(f"Error saving trade: {e}")
                await session.rollback()

    async def get_trades(self, market_id: str = None, start: datetime = None) -> List[Trade]:
        async with self._session_factory() as session:
            stmt = select(TradeModel)
            if market_id:
                stmt = stmt.where(TradeModel.market_id == market_id)
            if start:
                stmt = stmt.where(TradeModel.created_at >= start)
            
            stmt = stmt.order_by(TradeModel.created_at.desc())
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [
                Trade(
                    id=m.id,
                    market_id=m.market_id,
                    side=m.side,
                    action="BUY", # Should store action in DB model too, defaulting to BUY for now
                    size=m.size,
                    price=m.price,
                    fees=m.fees,
                    timestamp=m.created_at,
                    order_id=m.order_id,
                    is_paper=m.is_paper
                ) for m in models
            ]

    async def save_position(self, position: Position) -> None:
        async with self._session_factory() as session:
            try:
                stmt = select(PositionModel).where(PositionModel.id == position.id)
                result = await session.execute(stmt)
                existing = result.scalars().first()
                if existing:
                    existing.size = position.size
                    existing.closed_at = position.closed_at
                    existing.realized_pnl = position.realized_pnl if hasattr(position, 'realized_pnl') else None
                else:
                    new_pos = PositionModel(
                        id=position.id,
                        market_id=position.market_id,
                        side=position.side,
                        size=position.size,
                        entry_price=position.entry_price,
                        opened_at=position.opened_at
                    )
                    session.add(new_pos)
                await session.commit()
            except Exception as e:
                logger.error(f"Error saving position: {e}")
                await session.rollback()

    async def get_open_positions(self) -> List[Position]:
         async with self._session_factory() as session:
            stmt = select(PositionModel).where(PositionModel.closed_at.is_(None))
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [
                Position(
                    id=m.id,
                    market_id=m.market_id,
                    side=m.side,
                    size=m.size,
                    entry_price=m.entry_price,
                    opened_at=m.opened_at
                ) for m in models
            ]

    async def close_position(self, position_id: str, realized_pnl: float) -> None:
        async with self._session_factory() as session:
            try:
                stmt = select(PositionModel).where(PositionModel.id == position_id)
                result = await session.execute(stmt)
                existing = result.scalars().first()
                if existing:
                    existing.closed_at = datetime.utcnow()
                    existing.realized_pnl = realized_pnl
                    await session.commit()
            except Exception as e:
                logger.error(f"Error closing position: {e}")
                await session.rollback()
