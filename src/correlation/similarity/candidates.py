from typing import List, Dict, Tuple
from src.models.market import Market
from collections import defaultdict

class CandidateGenerator:
    def __init__(self):
        pass

    def block_by_category(self, markets: List[Market]) -> Dict[str, List[Market]]:
        """Group markets by Category or Subcategory"""
        blocks = defaultdict(list)
        for m in markets:
            key = m.subcategory if m.subcategory else (m.category or "Other")
            blocks[key].append(m)
        return blocks

    def generate_candidates(self, markets: List[Market]) -> List[Tuple[Market, Market]]:
        """
        Generate pairs of markets to compare.
        Strategies:
        1. Same subcategory (always compare)
        2. Same category (if no subcategory)
        
        Returns a list of unique pairs.
        """
        candidates = []
        seen = set()
        
        # 1. Block by Category/Subcategory
        blocks = self.block_by_category(markets)
        
        for key, group in blocks.items():
            # If group is too large, we might skip full comparison, but for now assuming < 1000 per subcat
            if len(group) < 2:
                continue
                
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    m1 = group[i]
                    m2 = group[j]
                    
                    # Sort IDs to ensure stable set key
                    pair_key = tuple(sorted((m1.id, m2.id)))
                    if pair_key not in seen:
                        candidates.append((m1, m2))
                        seen.add(pair_key)
        
        return candidates
