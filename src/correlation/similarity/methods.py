from typing import Dict, Any, List
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def levenshtein_similarity(text_a: str, text_b: str) -> float:
    """Normalized token sort ratio (0.0 to 1.0)"""
    return fuzz.token_sort_ratio(text_a, text_b) / 100.0

def jaccard_similarity(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Jaccard index of token sets"""
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0

def entity_similarity(entities_a: Dict[str, Any], entities_b: Dict[str, Any]) -> float:
    """
    Compare extracted entities. 
    Focus on specific keys: people, orgs, custom (years, prices).
    """
    score = 0.0
    checks = 0
    
    # 1. Compare People (High Value)
    people_a = set(entities_a.get("people", []))
    people_b = set(entities_b.get("people", []))
    if people_a or people_b:
        checks += 1
        if people_a.intersection(people_b):
            score += 1.0
            
    # 2. Compare Orgs
    orgs_a = set(entities_a.get("orgs", []))
    orgs_b = set(entities_b.get("orgs", []))
    if orgs_a or orgs_b:
        checks += 1
        if orgs_a.intersection(orgs_b):
            score += 0.8 # Slightly less weight than people

    # 3. Compare Custom (Years)
    years_a = set(entities_a.get("custom", {}).get("years", []))
    years_b = set(entities_b.get("custom", {}).get("years", []))
    if years_a or years_b:
        checks += 1
        if years_a.intersection(years_b):
             score += 1.0
        elif years_a and years_b:
             # Different years => probably not correlated
             score -= 0.5 

    if checks == 0:
        return 0.0
        
    return max(0.0, score / checks)

class TfidfSimilarity:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.matrix = None
        
    def fit(self, corpus: List[str]):
        if not corpus:
            return
        self.matrix = self.vectorizer.fit_transform(corpus)
        
    def predict(self, idx_a: int, idx_b: int) -> float:
        if self.matrix is None:
            return 0.0
        # Compute cosine similarity between two rows
        # fast sparse dot product
        return (self.matrix[idx_a] * self.matrix[idx_b].T)[0, 0]
