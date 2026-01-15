from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict

class CorrelationType(str, Enum):
    EQUIVALENT = "equivalent"      # Same outcome, different wording
    MATHEMATICAL = "mathematical"  # A implies B, thresholds
    INVERSE = "inverse"           # A + B ≈ 100%
    CAUSAL = "causal"             # A influences B

class MarketCorrelation(BaseModel):
    """Relationship between two markets"""
    model_config = ConfigDict(populate_by_name=True)

    id: Optional[str] = None
    market_a_id: str
    market_b_id: str
    correlation_type: CorrelationType
    expected_relationship: str  # e.g., "A <= B", "A + B = 1"
    confidence: float = Field(ge=0, le=1)
    manual_verified: bool = False
    historical_correlation: Optional[float] = None
    metadata: dict = Field(default_factory=dict)  # Detection method details
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class MarketGroup(BaseModel):
    """Collection of related markets"""
    model_config = ConfigDict(populate_by_name=True)
    
    id: str
    name: str
    description: Optional[str] = None
    market_ids: List[str] = []
    correlation_type: CorrelationType
