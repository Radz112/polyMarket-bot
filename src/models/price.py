from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict

class PriceSnapshot(BaseModel):
    """Point-in-time price"""
    model_config = ConfigDict(populate_by_name=True)
    
    market_id: str
    token_id: str
    timestamp: datetime
    yes_price: float
    no_price: float
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    volume: Optional[float] = None

class PriceHistory(BaseModel):
    """Time series of prices"""
    model_config = ConfigDict(populate_by_name=True)
    
    market_id: str
    snapshots: List[PriceSnapshot] = []
    
    def get_price_at(self, timestamp: datetime) -> Optional[PriceSnapshot]:
        """Get price at or before timestamp"""
        if not self.snapshots:
            return None
            
        # Sort if not sorted (assuming chronological usually)
        # Binary search could be better for large history
        closest = None
        for snapshot in self.snapshots:
            if snapshot.timestamp <= timestamp:
                if closest is None or snapshot.timestamp > closest.timestamp:
                    closest = snapshot
        return closest
    
    def get_range(self, start: datetime, end: datetime) -> List[PriceSnapshot]:
        """Get prices in range"""
        return [
            s for s in self.snapshots 
            if start <= s.timestamp <= end
        ]
