"""
Historical data provider for backtesting.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from .config import PriceSnapshot, OrderbookSnapshot, MarketResolution

logger = logging.getLogger(__name__)


@dataclass
class MockMarketData:
    """Mock market data for backtesting."""
    market_id: str
    question: str
    category: str
    prices: List[PriceSnapshot]
    resolution: Optional[MarketResolution] = None


class HistoricalDataProvider:
    """
    Provides historical data for backtesting.
    
    Supports loading from database or using mock data.
    """
    
    def __init__(self, db: Optional[Any] = None):
        self.db = db
        self.cache: Dict[str, List[PriceSnapshot]] = {}
        self.market_info: Dict[str, dict] = {}
        self.resolutions: Dict[str, MarketResolution] = {}
        self._data_loaded = False
    
    async def load_data(
        self,
        market_ids: List[str],
        start_date: datetime,
        end_date: datetime
    ):
        """
        Load all required historical data into memory.
        
        Args:
            market_ids: List of market IDs to load
            start_date: Start of date range
            end_date: End of date range
        """
        logger.info(f"Loading data for {len(market_ids)} markets from {start_date} to {end_date}")
        
        for market_id in market_ids:
            if market_id not in self.cache:
                prices = await self._load_prices(market_id, start_date, end_date)
                self.cache[market_id] = prices
        
        self._data_loaded = True
        logger.info(f"Loaded {sum(len(p) for p in self.cache.values())} price points")
    
    async def _load_prices(
        self,
        market_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[PriceSnapshot]:
        """Load prices for a single market."""
        if self.db:
            # Load from database
            return await self._load_from_db(market_id, start_date, end_date)
        else:
            # Return empty - will use mock data
            return []
    
    async def _load_from_db(
        self,
        market_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[PriceSnapshot]:
        """Load prices from database."""
        # Placeholder - would query historical_prices table
        return []
    
    def load_mock_data(self, mock_data: List[MockMarketData]):
        """
        Load mock data for testing.
        
        Args:
            mock_data: List of mock market data
        """
        for data in mock_data:
            self.cache[data.market_id] = data.prices
            self.market_info[data.market_id] = {
                "question": data.question,
                "category": data.category,
            }
            if data.resolution:
                self.resolutions[data.market_id] = data.resolution
        
        self._data_loaded = True
        logger.info(f"Loaded mock data for {len(mock_data)} markets")
    
    def get_price_at(
        self,
        market_id: str,
        timestamp: datetime
    ) -> Optional[PriceSnapshot]:
        """
        Get price at specific timestamp.
        
        Returns the most recent price <= timestamp.
        """
        if market_id not in self.cache:
            return None
        
        prices = self.cache[market_id]
        if not prices:
            return None
        
        # Find most recent price before or at timestamp
        result = None
        for price in prices:
            if price.timestamp <= timestamp:
                result = price
            else:
                break
        
        return result
    
    def get_prices_range(
        self,
        market_id: str,
        start: datetime,
        end: datetime
    ) -> List[PriceSnapshot]:
        """Get all prices in a date range."""
        if market_id not in self.cache:
            return []
        
        return [
            p for p in self.cache[market_id]
            if start <= p.timestamp <= end
        ]
    
    def get_all_prices_at(
        self,
        timestamp: datetime
    ) -> Dict[str, PriceSnapshot]:
        """Get prices for all markets at a timestamp."""
        result = {}
        for market_id in self.cache:
            price = self.get_price_at(market_id, timestamp)
            if price:
                result[market_id] = price
        return result
    
    def get_market_resolution(
        self,
        market_id: str
    ) -> Optional[MarketResolution]:
        """Get resolution data for a market."""
        return self.resolutions.get(market_id)
    
    def get_resolutions_before(
        self,
        timestamp: datetime
    ) -> List[MarketResolution]:
        """Get all resolutions before a timestamp."""
        return [
            r for r in self.resolutions.values()
            if r.resolution_time <= timestamp
        ]
    
    def get_market_info(self, market_id: str) -> dict:
        """Get market metadata."""
        return self.market_info.get(market_id, {"question": "Unknown", "category": "Other"})
    
    def get_market_ids(self) -> List[str]:
        """Get all loaded market IDs."""
        return list(self.cache.keys())
    
    @property
    def is_loaded(self) -> bool:
        """Check if data is loaded."""
        return self._data_loaded


def generate_mock_prices(
    market_id: str,
    start_date: datetime,
    end_date: datetime,
    initial_price: float = 0.50,
    volatility: float = 0.02,
    time_step: timedelta = timedelta(hours=1)
) -> List[PriceSnapshot]:
    """
    Generate mock price data for testing.
    
    Uses a random walk with drift.
    """
    import random
    
    prices = []
    current_price = initial_price
    current_time = start_date
    
    while current_time <= end_date:
        # Random walk with mean reversion
        change = random.gauss(0, volatility)
        mean_reversion = (0.50 - current_price) * 0.01
        current_price = max(0.01, min(0.99, current_price + change + mean_reversion))
        
        prices.append(PriceSnapshot(
            market_id=market_id,
            timestamp=current_time,
            yes_price=current_price,
            no_price=1 - current_price,
            yes_volume=random.uniform(1000, 10000),
            no_volume=random.uniform(1000, 10000),
        ))
        
        current_time += time_step
    
    return prices


def create_mock_dataset(
    num_markets: int = 5,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    resolution_rate: float = 0.3
) -> List[MockMarketData]:
    """
    Create a complete mock dataset for testing.
    
    Args:
        num_markets: Number of markets to generate
        start_date: Start date (default: 30 days ago)
        end_date: End date (default: now)
        resolution_rate: Fraction of markets that resolve
        
    Returns:
        List of MockMarketData objects
    """
    import random
    
    if start_date is None:
        start_date = datetime.utcnow() - timedelta(days=30)
    if end_date is None:
        end_date = datetime.utcnow()
    
    categories = ["Politics", "Crypto", "Economics", "Sports", "Entertainment"]
    questions = [
        "Will {subject} happen by end of month?",
        "Will {subject} exceed expectations?",
        "{subject} outcome determination",
        "Prediction: {subject}",
    ]
    subjects = [
        "Bitcoin price", "Election result", "Fed rate decision",
        "Championship winner", "Movie box office", "Tech earnings",
        "GDP growth", "Inflation report", "Market rally", "Tech IPO"
    ]
    
    mock_data = []
    
    for i in range(num_markets):
        market_id = f"mock_market_{i+1}"
        question = random.choice(questions).format(subject=random.choice(subjects))
        category = random.choice(categories)
        
        # Generate prices with varying characteristics
        initial_price = random.uniform(0.30, 0.70)
        volatility = random.uniform(0.01, 0.04)
        
        prices = generate_mock_prices(
            market_id=market_id,
            start_date=start_date,
            end_date=end_date,
            initial_price=initial_price,
            volatility=volatility,
        )
        
        # Maybe add resolution
        resolution = None
        if random.random() < resolution_rate and prices:
            # Resolve near the end
            resolution_time = end_date - timedelta(days=random.randint(1, 5))
            final_price = prices[-1].yes_price
            outcome = "YES" if final_price > 0.50 else "NO"
            resolution = MarketResolution(
                market_id=market_id,
                resolution_time=resolution_time,
                outcome=outcome,
            )
        
        mock_data.append(MockMarketData(
            market_id=market_id,
            question=question,
            category=category,
            prices=prices,
            resolution=resolution,
        ))
    
    return mock_data
