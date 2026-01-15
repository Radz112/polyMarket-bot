from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

class SignalType(str, Enum):
    DIVERGENCE = "divergence"
    THRESHOLD_VIOLATION = "threshold_violation"
    INVERSE_SUM_DEVIATION = "inverse_sum_deviation"
    LAGGING_MARKET = "lagging_market"

class Signal(BaseModel):
    """Trading signal"""
    model_config = ConfigDict(populate_by_name=True)
    
    id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    signal_type: SignalType
    market_ids: List[str]
    divergence_amount: float
    expected_value: float
    actual_value: float
    confidence: float = Field(ge=0, le=1)
    score: float = Field(ge=0, le=100)
    metadata: Dict[str, Any] = {}

class ScoredSignal(BaseModel):
    """Signal with scoring breakdown"""
    model_config = ConfigDict(populate_by_name=True)
    
    signal: Signal
    overall_score: float
    component_scores: Dict[str, float]
    recommended_action: str
    recommended_size: float
    recommended_price: float
