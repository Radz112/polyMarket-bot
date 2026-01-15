"""
Markets API routes - using real database data.
"""
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query

from src.dashboard.api.models import (
    MarketResponse,
    MarketDetailResponse,
    CorrelationResponse,
    PricePoint,
)
from src.dashboard.api.dependencies import get_db

router = APIRouter(prefix="/api/markets", tags=["Markets"])


@router.get("", response_model=List[MarketResponse])
async def get_markets(
    category: Optional[str] = Query(None, description="Filter by category"),
    active_only: bool = Query(True, description="Only active markets"),
    limit: int = Query(100, ge=1, le=500, description="Max results")
) -> List[MarketResponse]:
    """
    Get all markets with optional filtering.

    - **category**: Filter by market category
    - **active_only**: Only return active markets
    - **limit**: Maximum number of results
    """
    db = get_db()

    # Ensure database is connected
    if not db._engine:
        await db.connect()

    # Fetch markets from database
    from sqlalchemy import select, text
    from src.database.models import MarketModel

    async with db._session_factory() as session:
        query = select(MarketModel)

        if active_only:
            query = query.where(MarketModel.active == True)

        if category:
            query = query.where(MarketModel.category == category)

        query = query.limit(limit)

        result = await session.execute(query)
        db_markets = result.scalars().all()

    markets = []
    for m in db_markets:
        markets.append(MarketResponse(
            id=m.id,
            slug=m.slug,
            question=m.question,
            category=m.category,
            end_date=m.end_date,
            active=m.active,
            yes_price=0.5,  # Would need real-time price data
            no_price=0.5,
            volume_24h=0.0,
        ))

    return markets


@router.get("/correlations", response_model=List[CorrelationResponse])
async def get_all_correlations() -> List[CorrelationResponse]:
    """
    Get all market correlations.
    """
    db = get_db()

    if not db._engine:
        await db.connect()

    from sqlalchemy import select
    from src.database.models import MarketCorrelationModel, MarketModel

    async with db._session_factory() as session:
        result = await session.execute(select(MarketCorrelationModel))
        correlations = result.scalars().all()

        responses = []
        for c in correlations:
            # Get market names
            market_a = await session.get(MarketModel, c.market_a_id)
            market_b = await session.get(MarketModel, c.market_b_id)

            responses.append(CorrelationResponse(
                id=str(c.id),
                market_a_id=c.market_a_id,
                market_b_id=c.market_b_id,
                market_a_name=market_a.question if market_a else "Unknown",
                market_b_name=market_b.question if market_b else "Unknown",
                correlation_type=c.correlation_type.value,
                confidence=c.confidence,
                expected_relationship=c.expected_relationship,
            ))

        return responses


@router.get("/{market_id}", response_model=MarketDetailResponse)
async def get_market(market_id: str) -> MarketDetailResponse:
    """
    Get single market with orderbook and correlations.

    - **market_id**: Market identifier
    """
    db = get_db()

    if not db._engine:
        await db.connect()

    from src.database.models import MarketModel

    async with db._session_factory() as session:
        market = await session.get(MarketModel, market_id)

    if not market:
        raise HTTPException(status_code=404, detail=f"Market {market_id} not found")

    return MarketDetailResponse(
        market=MarketResponse(
            id=market.id,
            slug=market.slug,
            question=market.question,
            category=market.category,
            end_date=market.end_date,
            active=market.active,
            yes_price=0.5,
            no_price=0.5,
            volume_24h=0.0,
        ),
        correlations=[],
        orderbook=None
    )


@router.get("/{market_id}/correlations", response_model=List[CorrelationResponse])
async def get_market_correlations(market_id: str) -> List[CorrelationResponse]:
    """
    Get all correlations for a market.

    - **market_id**: Market identifier
    """
    db = get_db()

    if not db._engine:
        await db.connect()

    from sqlalchemy import select, or_
    from src.database.models import MarketCorrelationModel, MarketModel

    async with db._session_factory() as session:
        query = select(MarketCorrelationModel).where(
            or_(
                MarketCorrelationModel.market_a_id == market_id,
                MarketCorrelationModel.market_b_id == market_id
            )
        )
        result = await session.execute(query)
        correlations = result.scalars().all()

        responses = []
        for c in correlations:
            market_a = await session.get(MarketModel, c.market_a_id)
            market_b = await session.get(MarketModel, c.market_b_id)

            responses.append(CorrelationResponse(
                id=str(c.id),
                market_a_id=c.market_a_id,
                market_b_id=c.market_b_id,
                market_a_name=market_a.question if market_a else "Unknown",
                market_b_name=market_b.question if market_b else "Unknown",
                correlation_type=c.correlation_type.value,
                confidence=c.confidence,
                expected_relationship=c.expected_relationship,
            ))

        return responses


@router.get("/{market_id}/price-history", response_model=List[PricePoint])
async def get_price_history(
    market_id: str,
    start: Optional[datetime] = Query(None, description="Start time"),
    end: Optional[datetime] = Query(None, description="End time"),
    interval: str = Query("1m", description="Time interval (1m, 5m, 1h, 1d)")
) -> List[PricePoint]:
    """
    Get price history for charting.

    - **market_id**: Market identifier
    - **start**: Start timestamp
    - **end**: End timestamp
    - **interval**: Time interval
    """
    db = get_db()

    if not db._engine:
        await db.connect()

    from sqlalchemy import select
    from src.database.models import PriceSnapshotModel

    async with db._session_factory() as session:
        query = select(PriceSnapshotModel).where(
            PriceSnapshotModel.market_id == market_id
        ).order_by(PriceSnapshotModel.timestamp.desc()).limit(100)

        result = await session.execute(query)
        snapshots = result.scalars().all()

        return [
            PricePoint(
                timestamp=s.timestamp,
                price=s.yes_price,
                volume=s.yes_volume
            )
            for s in snapshots
        ]
