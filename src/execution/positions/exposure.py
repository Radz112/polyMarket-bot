"""
Exposure tracking for portfolio risk management.
"""
import logging
from typing import Dict, List, Optional
from collections import defaultdict

from src.execution.positions.types import (
    Position,
    ConcentrationReport,
    ExposureSummary,
)
from src.execution.positions.manager import PositionManager

logger = logging.getLogger(__name__)


class ExposureTracker:
    """
    Tracks and analyzes portfolio exposure.
    
    Provides:
    - Net exposure by underlying/entity
    - Correlated exposure analysis
    - Category-level exposure
    - Concentration reporting
    """
    
    def __init__(self, position_manager: PositionManager):
        """
        Initialize exposure tracker.
        
        Args:
            position_manager: Manager for accessing positions
        """
        self.position_manager = position_manager
    
    def get_net_exposure(self) -> Dict[str, float]:
        """
        Get net exposure by underlying entity.
        
        For prediction markets, this groups by the underlying
        event/entity. For example:
        - Long Trump PA YES: +$500
        - Long Trump National NO: -$300 (NO is inverse)
        - Net Trump exposure: +$200
        
        Returns:
            Dict mapping entity/underlying to net exposure
        """
        exposure = defaultdict(float)
        
        for position in self.position_manager.get_all_positions():
            # Extract underlying from market name/id
            underlying = self._extract_underlying(position)
            
            # YES = positive exposure, NO = negative exposure
            if position.side == "YES":
                exposure[underlying] += position.market_value
            else:
                exposure[underlying] -= position.market_value
        
        return dict(exposure)
    
    def get_correlated_exposure(self, position: Position) -> float:
        """
        Get total exposure to markets correlated with this position.
        
        Args:
            position: Position to analyze
            
        Returns:
            Total exposure to correlated markets including this position
        """
        total = position.market_value
        
        # Get correlated position IDs
        for corr_id in position.correlated_positions:
            corr_pos = self.position_manager.get_position_by_id(corr_id)
            if corr_pos:
                # Add if same direction, subtract if opposite
                if self._same_direction(position, corr_pos):
                    total += corr_pos.market_value
                else:
                    # Opposite direction reduces net exposure
                    total -= corr_pos.market_value
        
        return total
    
    def get_category_exposure(self) -> Dict[str, float]:
        """
        Get exposure by category.
        
        Returns:
            Dict mapping category to total exposure
        """
        return self.position_manager.get_exposure_by_category()
    
    def get_exposure_summary(self) -> ExposureSummary:
        """
        Get comprehensive exposure summary.
        
        Returns:
            ExposureSummary with all exposure metrics
        """
        positions = self.position_manager.get_all_positions()
        
        if not positions:
            return ExposureSummary(
                total_exposure=0,
                net_exposure_by_underlying={},
                exposure_by_category={},
                largest_single_exposure=0,
                largest_category_exposure=0
            )
        
        total = sum(p.market_value for p in positions)
        by_underlying = self.get_net_exposure()
        by_category = self.get_category_exposure()
        
        largest_single = max(p.market_value for p in positions)
        largest_category = max(by_category.values()) if by_category else 0
        
        return ExposureSummary(
            total_exposure=total,
            net_exposure_by_underlying=by_underlying,
            exposure_by_category=by_category,
            largest_single_exposure=largest_single,
            largest_category_exposure=largest_category
        )
    
    def get_concentration_report(self) -> ConcentrationReport:
        """
        Generate portfolio concentration report.
        
        Returns:
            ConcentrationReport with diversification analysis
        """
        positions = self.position_manager.get_all_positions()
        
        if not positions:
            return ConcentrationReport(
                largest_position=None,
                largest_position_pct=0,
                largest_category="",
                largest_category_pct=0,
                correlation_clusters=[],
                diversification_score=100,  # Empty = fully diversified
                warnings=[]
            )
        
        total_exposure = sum(p.market_value for p in positions)
        if total_exposure == 0:
            total_exposure = 1  # Avoid division by zero
        
        # Find largest position
        largest_pos = max(positions, key=lambda p: p.market_value)
        largest_pos_pct = (largest_pos.market_value / total_exposure) * 100
        
        # Find largest category
        by_category = self.get_category_exposure()
        if by_category:
            largest_cat = max(by_category.items(), key=lambda x: x[1])
            largest_cat_name = largest_cat[0]
            largest_cat_pct = (largest_cat[1] / total_exposure) * 100
        else:
            largest_cat_name = ""
            largest_cat_pct = 0
        
        # Find correlation clusters
        clusters = self._find_correlation_clusters(positions)
        
        # Calculate diversification score
        div_score = self._calculate_diversification_score(
            positions, largest_pos_pct, largest_cat_pct, clusters
        )
        
        # Generate warnings
        warnings = self._generate_concentration_warnings(
            largest_pos_pct, largest_cat_pct, clusters
        )
        
        return ConcentrationReport(
            largest_position=largest_pos,
            largest_position_pct=largest_pos_pct,
            largest_category=largest_cat_name,
            largest_category_pct=largest_cat_pct,
            correlation_clusters=clusters,
            diversification_score=div_score,
            warnings=warnings
        )
    
    def _extract_underlying(self, position: Position) -> str:
        """
        Extract underlying entity from position.
        
        This is a simple heuristic - could be enhanced with NER or mapping.
        """
        # Try to extract key entity from market name
        name = position.market_name.lower()
        
        # Common patterns in prediction markets
        keywords = ["trump", "biden", "harris", "bitcoin", "ethereum", "fed", "election"]
        
        for kw in keywords:
            if kw in name:
                return kw.capitalize()
        
        # Fall back to category
        return position.category
    
    def _same_direction(self, pos1: Position, pos2: Position) -> bool:
        """Check if two positions are in the same direction."""
        # Same side on related markets = same direction
        # Opposite side on inverse markets = same direction
        # This is simplified - real implementation would use correlation data
        return pos1.side == pos2.side
    
    def _find_correlation_clusters(self, positions: List[Position]) -> List[List[str]]:
        """
        Find clusters of correlated positions.
        
        Uses union-find to group positions by their correlations.
        """
        if not positions:
            return []
        
        # Build adjacency from correlated_positions
        parent = {p.id: p.id for p in positions}
        
        def find(x):
            if parent.get(x, x) != x:
                parent[x] = find(parent[x])
            return parent.get(x, x)
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Union correlated positions
        for pos in positions:
            for corr_id in pos.correlated_positions:
                if corr_id in parent:
                    union(pos.id, corr_id)
        
        # Group by root
        clusters = defaultdict(list)
        for pos in positions:
            root = find(pos.id)
            clusters[root].append(pos.id)
        
        # Return clusters with more than 1 position
        return [ids for ids in clusters.values() if len(ids) > 1]
    
    def _calculate_diversification_score(
        self,
        positions: List[Position],
        largest_pos_pct: float,
        largest_cat_pct: float,
        clusters: List[List[str]]
    ) -> float:
        """
        Calculate diversification score (0-100).
        
        Higher = more diversified
        """
        score = 100.0
        
        # Penalize concentrated positions
        if largest_pos_pct > 50:
            score -= 40
        elif largest_pos_pct > 30:
            score -= 25
        elif largest_pos_pct > 20:
            score -= 15
        elif largest_pos_pct > 10:
            score -= 5
        
        # Penalize concentrated categories
        if largest_cat_pct > 70:
            score -= 30
        elif largest_cat_pct > 50:
            score -= 20
        elif largest_cat_pct > 40:
            score -= 10
        
        # Penalize correlation clusters
        total_in_clusters = sum(len(c) for c in clusters)
        cluster_pct = (total_in_clusters / len(positions)) * 100 if positions else 0
        if cluster_pct > 50:
            score -= 15
        elif cluster_pct > 30:
            score -= 10
        
        # Bonus for number of positions
        if len(positions) >= 10:
            score += 10
        elif len(positions) >= 5:
            score += 5
        
        return max(0, min(100, score))
    
    def _generate_concentration_warnings(
        self,
        largest_pos_pct: float,
        largest_cat_pct: float,
        clusters: List[List[str]]
    ) -> List[str]:
        """Generate warnings about concentration risks."""
        warnings = []
        
        if largest_pos_pct > 30:
            warnings.append(
                f"High single-position concentration: {largest_pos_pct:.1f}% in one position"
            )
        
        if largest_cat_pct > 50:
            warnings.append(
                f"High category concentration: {largest_cat_pct:.1f}% in one category"
            )
        
        if len(clusters) > 0:
            total_clustered = sum(len(c) for c in clusters)
            warnings.append(
                f"Found {len(clusters)} correlation clusters with {total_clustered} positions"
            )
        
        return warnings
