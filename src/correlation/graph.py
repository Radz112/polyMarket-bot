from typing import List, Dict, Set, Optional
import networkx as nx

from src.models import MarketCorrelation

class CorrelationGraph:
    def __init__(self, correlations: List[MarketCorrelation]):
        self.graph = nx.Graph()
        self._build_graph(correlations)

    def _build_graph(self, correlations: List[MarketCorrelation]):
        for corr in correlations:
            if corr.confidence > 0.3: # Filter weak ones
                self.graph.add_edge(
                    corr.market_a_id, 
                    corr.market_b_id, 
                    weight=corr.confidence,
                    type=corr.correlation_type
                )

    def get_clusters(self) -> List[List[str]]:
        """Find connected components (clusters of correlated markets)."""
        return [list(c) for c in nx.connected_components(self.graph)]

    def get_neighbors(self, market_id: str) -> List[str]:
        if market_id in self.graph:
            return list(self.graph.neighbors(market_id))
        return []

    def shortest_path(self, market_a: str, market_b: str) -> List[str]:
        try:
            return nx.shortest_path(self.graph, source=market_a, target=market_b)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
