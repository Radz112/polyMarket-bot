from typing import Tuple, Dict, Any, Optional
import re
from .taxonomy import CATEGORIES
from .entities import EntityExtractor
import logging

logger = logging.getLogger(__name__)

class MarketCategorizer:
    def __init__(self):
        self.extractor = EntityExtractor()
        self.categories = CATEGORIES

    def categorize(self, market) -> Tuple[str, Optional[str], Dict[str, Any]]:
        """
        Categorize a market based on its question and description.
        Returns: (Category, Subcategory, Entities)
        """
        text = market.question
        if market.description:
             text += " " + market.description
        
        # 1. Extract Entities
        entities = self.extractor.extract(text)
        
        # 2. Determine Category
        # We search for keywords in the question primarily
        question_lower = market.question.lower()
        
        best_category = "Other"
        best_subcategory = None
        max_score = 0
        
        for cat, data in self.categories.items():
            for subcat, keywords in data["subcategories"].items():
                score = 0
                for kw in keywords:
                    # Use simple regex for word boundary if keyword is alphanumeric
                    # Handles multi-word keywords too
                    pattern = r'\b' + re.escape(kw) + r'\b'
                    if re.search(pattern, question_lower):
                        score += 1
                
                # Boost logic: exact matches or important keywords could count more
                if score > max_score:
                    max_score = score
                    best_category = cat
                    best_subcategory = subcat
        
        # Fallback: Use Gamma API category if provided and our logic is weak
        if max_score == 0 and market.category:
            # Map API category to our taxonomy if possible, else use it as is
            # For now, just trust our taxonomy. If no match, it stays "Other".
            pass

        return best_category, best_subcategory, entities
