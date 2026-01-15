import logging
from typing import List, Optional, Tuple, Dict
from sqlalchemy import select, delete
from sqlalchemy.orm import joinedload

from src.database.postgres import DatabaseManager
from src.database.models import MarketCorrelationModel, LogicalRuleModel, RuleViolationModel
from src.models import MarketCorrelation
from src.correlation.logical.rules import LogicalRule, LogicalRuleType, RuleViolation

logger = logging.getLogger(__name__)

class CorrelationStore:
    def __init__(self, db: DatabaseManager):
        self.db = db

    async def save_correlations_batch(self, correlations: List[MarketCorrelation]):
        """Bulk upsert correlations."""
        # For now, loop invalidation. Improve with bulk insert later.
        for corr in correlations:
            await self.db.upsert_correlation(corr)
            
    async def save_logical_rule(self, rule: LogicalRule):
        # We need a method in DB manager or direct session access
        # Adding direct session usage here for clarity, 
        # but ideally we extend DatabaseManager.
        async with self.db._session_factory() as session:
            try:
                model = LogicalRuleModel(
                    rule_type=rule.rule_type,
                    market_ids=rule.market_ids,
                    constraint_desc=rule.constraint_desc,
                    metadata_=rule.metadata,
                    tolerance=rule.tolerance,
                    confidence=rule.confidence
                )
                session.add(model)
                await session.commit()
            except Exception as e:
                logger.error(f"Error saving logical rule: {e}")
                await session.rollback()

    async def save_violation(self, violation: RuleViolation, rule_db_id: int):
        async with self.db._session_factory() as session:
            try:
                model = RuleViolationModel(
                    rule_id=rule_db_id,
                    deviation=violation.deviation,
                    profit_opportunity=violation.profit_opportunity,
                    suggested_trades=[
                        {"market_id": t[0], "side": t[1], "size": t[2]} 
                        for t in violation.suggested_trades
                    ],
                    details=violation.details
                )
                session.add(model)
                await session.commit()
            except Exception as e:
                logger.error(f"Error saving violation: {e}")
                await session.rollback()

    async def get_all_active(self) -> List[MarketCorrelation]:
        """Get all active correlations."""
        return await self.get_all_correlations()

    async def get_all_correlations(self) -> List[MarketCorrelation]:
        # Reuse existing db method logic or new query
        # Currently DatabaseManager doesn't have get_all_correlations
        # We can add it or query here.
        async with self.db._session_factory() as session:
            stmt = select(MarketCorrelationModel)
            result = await session.execute(stmt)
            models = result.scalars().all()
            return [self._model_to_entity(m) for m in models]
            
    def _model_to_entity(self, m: MarketCorrelationModel) -> MarketCorrelation:
        meta = {}
        # Convert new columns to metadata dict for now, 
        # until MarketCorrelation entity is updated to have these fields explicitly
        # Or we updated the entity?
        # The Implementation Plan didn't specify updating the Pydantic model structure deeply,
        # but the merger uses metadata.
        
        # Populate metadata from columns
        meta["detection_methods"] = m.detection_methods or []
        meta["string_similarity"] = m.string_similarity
        meta["statistical_correlation"] = m.statistical_correlation
        meta["lead_lag_seconds"] = m.lead_lag_seconds
        meta["leader_market_id"] = m.leader_market_id
        
        return MarketCorrelation(
            market_a_id=m.market_a_id,
            market_b_id=m.market_b_id,
            correlation_type=m.correlation_type,
            expected_relationship=m.expected_relationship,
            confidence=m.confidence,
            manual_verified=m.manual_verified,
            historical_correlation=m.historical_correlation,
            metadata=meta
        )
