from typing import List, Dict
from src.correlation.store import CorrelationStore
from src.correlation.graph import CorrelationGraph
from src.models import MarketCorrelation

class CorrelationQueries:
    def __init__(self, store: CorrelationStore):
        self.store = store
        self.graph = None 

    async def load_graph(self):
        correlations = await self.store.get_all_correlations()
        self.graph = CorrelationGraph(correlations)

    async def get_market_clusters(self) -> List[List[str]]:
        if not self.graph:
            await self.load_graph()
        return self.graph.get_clusters()

    async def find_related_markets(self, market_id: str) -> List[str]:
        if not self.graph:
            await self.load_graph()
        return self.graph.get_neighbors(market_id)
        
    # TODO: Add more specific query wrappers
