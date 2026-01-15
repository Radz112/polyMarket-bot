import re
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime
from src.models import Market

@dataclass
class ThresholdInfo:
    asset: str
    value: float
    direction: str # "above" or "below"
    date_str: Optional[str] = None
    
class ThresholdParser:
    def __init__(self):
        # Regex patterns
        # Identify price thresholds: "$100k", "100,000", "0.50"
        # Identify assets: "BTC", "Bitcoin", "ETH", "Fed Rates"
        pass

    def parse_threshold(self, question: str) -> Optional[ThresholdInfo]:
        """
        Extract threshold info from question.
        """
        # Simplify text
        text = question.lower()
        
        # 1. Detect Asset (Simple heuristics for now)
        asset = None
        if "btc" in text or "bitcoin" in text:
            asset = "BTC"
        elif "eth" in text or "ethereum" in text:
            asset = "ETH"
        elif "sol" in text or "solana" in text:
            asset = "SOL"
        elif "fed" in text and "rate" in text:
            asset = "FED_RATE"
            
        if not asset:
            return None
            
        # 2. Detect Value
        # Patterns: "$100k", "$100,000", "100k", "100000"
        # We need to handle "k" = 1000
        # regex to find number near $ or keywords
        
        # Look for $ followed by number
        # or number followed by k
        
        value = None
        
        # Try finding $100k, $100,000
        # r'\$([\d,]+(\.\d+)?)(k)?'
        dollar_match = re.search(r'\$\s?([\d,]+(\.\d+)?)\s?(k|m)?', text)
        if dollar_match:
            num_str = dollar_match.group(1).replace(",", "")
            val = float(num_str)
            suffix = dollar_match.group(3)
            if suffix == 'k':
                val *= 1000
            elif suffix == 'm':
                val *= 1000000
            value = val
        if value is None:
            # Fallback for "90k" without dollar, "100,000" without dollar
            # Find numbers that might have k/m suffix
            # r'\b(\d+(\.\d+)?)(\s?[km])?\b'
            
            matches = re.finditer(r'\b(\d{2,3}(,\d{3})*(\.\d+)?)(\s?[km])?\b', text)
            for m in matches:
                # Need context check? e.g. "year 2024" is bad. 
                # but "90k" is usually price.
                
                raw_val = m.group(1).replace(",", "")
                val = float(raw_val)
                suffix = m.group(4)
                
                if suffix:
                    suffix = suffix.strip()
                    if suffix == 'k': val *= 1000
                    elif suffix == 'm': val *= 1000000
                
                # Heuristic: Filter out likely years if no suffix
                if not suffix:
                    if 2000 <= val <= 2100:
                        continue
                        
                value = val
                break

        if value is None:
            return None
            
        # 3. Detect Direction
        # "above", ">", "hit", "reach" -> usually implies >=
        # "below", "<"
        direction = "above" # default for "will btc hit x"
        if "below" in text or "<" in text:
            direction = "below"
            
        return ThresholdInfo(asset=asset, value=value, direction=direction)

    def group_markets(self, markets: List[Market]) -> Dict[str, List[Tuple[Market, ThresholdInfo]]]:
        """
        Group markets by Asset.
        Returns: {"BTC": [(m1, info1), (m2, info2)]}
        """
        groups = {}
        for m in markets:
            info = self.parse_threshold(m.question)
            if info:
                key = info.asset
                # Include date if pertinent?
                # For now just asset
                if key not in groups:
                    groups[key] = []
                groups[key].append((m, info))
                
        # Sort each group by threshold value
        for key in groups:
            groups[key].sort(key=lambda x: x[1].value)
            
        return groups
