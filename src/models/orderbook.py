from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, computed_field, ConfigDict

class OrderbookEntry(BaseModel):
    """Single price level"""
    price: float
    size: float

class Orderbook(BaseModel):
    """Full orderbook snapshot"""
    model_config = ConfigDict(populate_by_name=True)
    
    market_id: str
    token_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    bids: List[OrderbookEntry] = []
    asks: List[OrderbookEntry] = []
    
    @computed_field
    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None
    
    @computed_field
    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None
    
    @computed_field
    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return float(self.best_ask) - float(self.best_bid)
        return None
    
    @computed_field
    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return (float(self.best_bid) + float(self.best_ask)) / 2
        return None
    
    @classmethod
    def from_api_response(cls, data: dict, market_id: str = "", token_id: str = "") -> "Orderbook":
        """Create from API response"""
        bids = [OrderbookEntry(price=float(p), size=float(s)) for p, s in data.get("bids", [])]
        asks = [OrderbookEntry(price=float(p), size=float(s)) for p, s in data.get("asks", [])]
        
        # Handle timestamp - sometimes string, sometimes int/float
        ts_val = data.get("timestamp")
        timestamp = datetime.utcnow()
        if ts_val:
            try:
                # If numeric (ms or s)
                if isinstance(ts_val, (int, float)):
                    # Assume ms if large
                    timestamp = datetime.fromtimestamp(ts_val / 1000.0 if ts_val > 2e10 else ts_val)
                elif isinstance(ts_val, str):
                    timestamp = datetime.fromisoformat(ts_val.replace("Z", "+00:00"))
            except Exception:
                pass
        
        return cls(
            market_id=data.get("market", market_id),
            token_id=data.get("asset_id", token_id),
            timestamp=timestamp,
            bids=bids,
            asks=asks
        )

class OrderbookUpdate(BaseModel):
    """WebSocket orderbook update"""
    market_id: str
    token_id: str
    timestamp: datetime
    side: str
    price: float
    size: float
    action: str
