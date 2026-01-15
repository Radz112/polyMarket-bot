from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, computed_field, ConfigDict

class Side(str, Enum):
    YES = "YES"
    NO = "NO"

class Position(BaseModel):
    """Open position"""
    model_config = ConfigDict(populate_by_name=True)
    
    id: str
    market_id: str
    market_name: Optional[str] = None
    side: Side
    size: float
    entry_price: float
    current_price: float = 0
    opened_at: datetime
    closed_at: Optional[datetime] = None
    
    @computed_field
    @property
    def market_value(self) -> float:
        return self.size * self.current_price
    
    @computed_field
    @property
    def cost_basis(self) -> float:
        return self.size * self.entry_price
    
    @computed_field
    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis
    
    @computed_field
    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100

class Trade(BaseModel):
    """Executed trade"""
    model_config = ConfigDict(populate_by_name=True)
    
    id: str
    market_id: str
    side: Side
    action: str  # "BUY" or "SELL"
    size: float
    price: float
    fees: float
    timestamp: datetime
    order_id: Optional[str] = None
    signal_id: Optional[str] = None
    is_paper: bool = True
    realized_pnl: Optional[float] = None
