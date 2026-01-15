from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import time
import logging

from src.models.market import Market
from src.models.correlation import CorrelationType
from .preprocessing import TextPreprocessor
from .candidates import CandidateGenerator
from .methods import levenshtein_similarity, jaccard_similarity, entity_similarity, TfidfSimilarity

logger = logging.getLogger(__name__)

@dataclass
class SimilarityResult:
    market_a_id: str
    market_b_id: str
    overall_score: float
    question_similarity: float
    entity_overlap: float
    category_match: bool
    confidence: float
    breakdown: Dict[str, Any]

class StringSimilarityDetector:
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.preprocessor = TextPreprocessor()
        self.candidate_gen = CandidateGenerator()
        self.tfidf = TfidfSimilarity()

    def find_similar_markets(self, markets: List[Market]) -> List[SimilarityResult]:
        """
        Main pipeline:
        1. Preprocess all markets
        2. Fit TF-IDF
        3. Generate candidates
        4. Compute similarities
        5. Filter results
        """
        if not markets or len(markets) < 2:
            return []

        logger.info(f"Processing {len(markets)} markets for similarity...")
        
        # 1. Preprocess (Cache tokens on objects for speed)
        market_map = {m.id: m for m in markets}
        processed_data = {} # id -> {text, tokens}
        corpus = []
        market_ids = []
        
        for m in markets:
            norm_text = self.preprocessor.normalize(m.question)
            tokens = self.preprocessor.tokenize(norm_text)
            processed_data[m.id] = {
                "text": norm_text,
                "tokens": tokens
            }
            corpus.append(norm_text)
            market_ids.append(m.id)
            
        # 2. Fit TF-IDF
        self.tfidf.fit(corpus)
        id_to_idx = {mid: i for i, mid in enumerate(market_ids)}

        # 3. Generate Candidates
        candidates = self.candidate_gen.generate_candidates(markets)
        logger.info(f"Generated {len(candidates)} candidate pairs to compare.")
        
        results = []
        
        for m1, m2 in candidates:
            # 4. Compute Similarity
            d1 = processed_data[m1.id]
            d2 = processed_data[m2.id]
            
            # Scores
            lev_score = levenshtein_similarity(d1["text"], d2["text"])
            jac_score = jaccard_similarity(d1["tokens"], d2["tokens"])
            tfidf_score = self.tfidf.predict(id_to_idx[m1.id], id_to_idx[m2.id])
            ent_score = entity_similarity(m1.entities, m2.entities)
            
            # Weighted average
            # Weights: TF-IDF (0.3), Levenshtein (0.3), Entities (0.3), Jaccard (0.1)
            overall = (tfidf_score * 0.3) + (lev_score * 0.3) + (ent_score * 0.3) + (jac_score * 0.1)
            
            if overall >= self.threshold:
                res = SimilarityResult(
                    market_a_id=m1.id,
                    market_b_id=m2.id,
                    overall_score=overall,
                    question_similarity=lev_score, # Using Levenshtein as proxy for simple Q sim
                    entity_overlap=ent_score,
                    category_match=(m1.category == m2.category),
                    confidence=overall,
                    breakdown={
                        "levenshtein": lev_score,
                        "jaccard": jac_score,
                        "tfidf": tfidf_score,
                        "entities": ent_score
                    }
                )
                results.append(res)
                
        # Sort by score desc
        results.sort(key=lambda x: x.overall_score, reverse=True)
        return results
