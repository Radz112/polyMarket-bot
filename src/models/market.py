from datetime import datetime
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, ConfigDict

class Token(BaseModel):
    """Token (YES/NO share) representation"""
    model_config = ConfigDict(populate_by_name=True)
    
    token_id: str
    outcome: str  # "YES" or "NO"
    price: Optional[float] = None

class Market(BaseModel):
    """Full market representation"""
    model_config = ConfigDict(populate_by_name=True)
    
    id: str = Field(..., alias="condition_id")
    slug: str = Field(default="")  # Some API responses might lack this or call it differently
    question: str
    description: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    end_date: Optional[datetime] = None
    created_at: Optional[datetime] = None
    active: bool = True
    closed: bool = False
    resolved: bool = False
    outcome: Optional[str] = None
    tokens: List[Token] = []
    clob_token_ids: List[str] = []  # CLOB API token IDs for YES/NO outcomes
    min_order_size: float = 0.01
    tick_size: float = 0.01
    entities: Dict[str, Any] = {}  # Extracted entities for similarity detection

    @property
    def yes_token_id(self) -> Optional[str]:
        """Get the YES token ID (first in list)."""
        return self.clob_token_ids[0] if self.clob_token_ids else None

    @property
    def no_token_id(self) -> Optional[str]:
        """Get the NO token ID (second in list)."""
        return self.clob_token_ids[1] if len(self.clob_token_ids) > 1 else None
    
    @classmethod
    def from_api_response(cls, data: dict) -> "Market":
        """Create from API response"""
        # Handle field mapping quirks from different API endpoints here
        # For now, we assume data keys roughly match or are handled by aliases
        # We might need to construct tokens manually if they are nested differently
        token_data = data.get("tokens", [])
        tokens = []
        
        # Sometimes tokens come as dicts directly, sometimes we need to map them
        if token_data and isinstance(token_data, list):
            for t in token_data:
                if isinstance(t, dict):
                    tokens.append(Token(**t))
        
        # Ensure we have 'tokens' in the init data
        init_data = data.copy()
        if "tokens" not in init_data:
            init_data["tokens"] = tokens
            
        # Map generic 'id' if 'condition_id' missing but 'question_id' or 'id' present?
        # The prompt says id: str. The API usually returns condition_id.
        # We used alias="condition_id", so we expect data["condition_id"].
        
        return cls(**init_data)

class MarketSummary(BaseModel):
    """Lightweight market for lists"""
    model_config = ConfigDict(populate_by_name=True)
    
    id: str
    slug: str
    question: str
    category: Optional[str] = None
    end_date: Optional[datetime] = None
    active: bool = True
    yes_price: Optional[float] = None
    no_price: Optional[float] = None
