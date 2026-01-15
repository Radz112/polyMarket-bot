from typing import List, Dict, Optional, Tuple
from src.models import MarketCorrelation, CorrelationType
from src.correlation.logical.rules import LogicalRule

class CorrelationMerger:
    def __init__(self):
        pass

    def merge_detections(
        self,
        string_matches: List[Tuple[MarketCorrelation, float]], # (corr, score)
        statistical_matches: List[Tuple[MarketCorrelation, float]],
        logical_rules: List[LogicalRule]
    ) -> List[MarketCorrelation]:
        """
        Merge detections from different sources into a unified list of MarketCorrelations.
        """
        merged_map: Dict[Tuple[str, str], MarketCorrelation] = {}
        
        # 1. Process String Matches
        for corr, score in string_matches:
            key = tuple(sorted([corr.market_a_id, corr.market_b_id]))
            
            if key not in merged_map:
                merged_map[key] = corr
                # Initialize metadata
                merged_map[key].metadata["detection_methods"] = ["string"]
                merged_map[key].metadata["string_score"] = score
            else:
                # Update existing
                existing = merged_map[key]
                existing.metadata["detection_methods"].append("string")
                existing.metadata["string_score"] = score
                # Boost confidence if string match confirms?
                # Usually we take the max or weighted avg
                existing.confidence = max(existing.confidence, score)

        # 2. Process Statistical Matches
        for corr, score in statistical_matches:
            key = tuple(sorted([corr.market_a_id, corr.market_b_id]))
            
            if key not in merged_map:
                merged_map[key] = corr
                merged_map[key].metadata["detection_methods"] = ["statistical"]
                merged_map[key].metadata["statistical_score"] = score
            else:
                existing = merged_map[key]
                if "statistical" not in existing.metadata.get("detection_methods", []):
                    existing.metadata["detection_methods"].append("statistical")
                existing.metadata["statistical_score"] = score
                
                # Boost confidence logic:
                # If both string and statistical exist, confidence should be high
                str_score = existing.metadata.get("string_score", 0.0)
                # Simple ensemble: weighted average or max
                # If aligned (both say correlated), boost.
                existing.confidence = (str_score + score) / 2 + 0.1 # Boost for dual confirmation
                existing.confidence = min(1.0, existing.confidence)
                
                # Check for conflict?
                # If str says positive and stat says negative?
                # Handled by individual detectors returning correlations.
                
        # 3. Process Logical Rules (Implicit correlations)
        # Logical rules define relationships. We can convert them to Correlations
        # e.g. Threshold Ordering -> Positive Correlation
        for rule in logical_rules:
            # We assume pairwise for now or create clique
            # Simple chain: m1, m2
            if len(rule.market_ids) >= 2:
                # Link adjacent pairs in the rule list? 
                # Or all pairs?
                # Usually pairwise
                for i in range(len(rule.market_ids) - 1):
                    id_a = rule.market_ids[i]
                    id_b = rule.market_ids[i+1]
                    key = tuple(sorted([id_a, id_b]))
                    
                    if key not in merged_map:
                        # Create new correlation
                        corr = MarketCorrelation(
                            market_a_id=id_a,
                            market_b_id=id_b,
                            correlation_type=CorrelationType.MATHEMATICAL,  # Logical rules define mathematical constraints
                            expected_relationship="logical_constraint",
                            confidence=1.0,  # Logical is hard constraint
                            manual_verified=True,  # It's math
                            metadata={}
                        )
                        corr.metadata["detection_methods"] = ["logical"]
                        corr.metadata["logical_rule_type"] = rule.rule_type.value
                        merged_map[key] = corr
                    else:
                        existing = merged_map[key]
                        if "logical" not in existing.metadata.get("detection_methods", []):
                             existing.metadata["detection_methods"].append("logical")
                        existing.confidence = 1.0 # Override with certainty

        return list(merged_map.values())
