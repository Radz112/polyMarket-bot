"""
Comprehensive test suite for Dashboard API.
"""
import pytest
from datetime import datetime
from fastapi.testclient import TestClient

from src.dashboard.api.main import app
from src.dashboard.api.websocket import ws_manager


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


# ============================================================================
# System Tests
# ============================================================================

class TestSystem:
    """Test system endpoints."""
    
    def test_root(self, client):
        """Root should return API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Polymarket Bot API"
        assert "version" in data
    
    def test_health(self, client):
        """Health check should return status."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "uptime_seconds" in data
    
    def test_stats(self, client):
        """Stats should return counts."""
        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert "markets_tracked" in data
        assert "active_positions" in data


# ============================================================================
# Market Tests
# ============================================================================

class TestMarkets:
    """Test market endpoints."""
    
    def test_get_markets(self, client):
        """Should return list of markets."""
        response = client.get("/api/markets")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_markets_with_category(self, client):
        """Should filter by category."""
        response = client.get("/api/markets?category=Politics")
        assert response.status_code == 200
        data = response.json()
        for market in data:
            assert market["category"].lower() == "politics"
    
    def test_get_markets_with_limit(self, client):
        """Should respect limit."""
        response = client.get("/api/markets?limit=1")
        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 1
    
    def test_get_market_not_found(self, client):
        """Should return 404 for missing market."""
        response = client.get("/api/markets/nonexistent")
        assert response.status_code == 404


# ============================================================================
# Signal Tests
# ============================================================================

class TestSignals:
    """Test signal endpoints."""
    
    def test_get_signals(self, client):
        """Should return list of signals."""
        response = client.get("/api/signals")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_signals_with_score(self, client):
        """Should filter by minimum score."""
        response = client.get("/api/signals?min_score=50")
        assert response.status_code == 200


# ============================================================================
# Position Tests
# ============================================================================

class TestPositions:
    """Test position endpoints."""
    
    def test_get_positions(self, client):
        """Should return list of positions."""
        response = client.get("/api/positions")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_positions_closed(self, client):
        """Should return closed positions."""
        response = client.get("/api/positions?status=closed")
        assert response.status_code == 200
    
    def test_get_position_not_found(self, client):
        """Should return 404 for missing position."""
        response = client.get("/api/positions/nonexistent")
        assert response.status_code == 404


# ============================================================================
# Portfolio Tests
# ============================================================================

class TestPortfolio:
    """Test portfolio endpoints."""
    
    def test_get_portfolio(self, client):
        """Should return portfolio state."""
        response = client.get("/api/portfolio")
        assert response.status_code == 200
        data = response.json()
        assert "cash_balance" in data
        assert "total_value" in data
        assert "positions" in data
    
    def test_get_portfolio_history(self, client):
        """Should return portfolio history."""
        response = client.get("/api/portfolio/history")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_performance(self, client):
        """Should return performance metrics."""
        response = client.get("/api/portfolio/performance")
        assert response.status_code == 200
        data = response.json()
        assert "total_trades" in data
        assert "win_rate" in data


# ============================================================================
# Order Tests
# ============================================================================

class TestOrders:
    """Test order endpoints."""
    
    def test_get_orders(self, client):
        """Should return list of orders."""
        response = client.get("/api/orders")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_create_order(self, client):
        """Should create an order."""
        response = client.post("/api/orders", json={
            "market_id": "test_market",
            "side": "YES",
            "action": "BUY",
            "order_type": "limit",
            "size": 100,
            "limit_price": 0.50
        })
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["size"] == 100
    
    def test_create_order_invalid_side(self, client):
        """Should reject invalid side."""
        response = client.post("/api/orders", json={
            "market_id": "test",
            "side": "INVALID",
            "action": "BUY",
            "size": 100
        })
        assert response.status_code == 422  # Validation error


# ============================================================================
# Risk Tests
# ============================================================================

class TestRisk:
    """Test risk endpoints."""
    
    def test_get_risk_dashboard(self, client):
        """Should return risk state."""
        response = client.get("/api/risk")
        assert response.status_code == 200
        data = response.json()
        assert "trading_allowed" in data
        assert "risk_score" in data
        assert "risk_level" in data
    
    def test_get_breakers(self, client):
        """Should return circuit breakers."""
        response = client.get("/api/risk/breakers")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_halt_trading(self, client):
        """Should halt trading."""
        response = client.post("/api/risk/halt", json={
            "reason": "Test halt"
        })
        assert response.status_code == 200
        
        # Check it's halted
        response = client.get("/api/risk")
        assert response.json()["trading_allowed"] == False
        
        # Resume
        client.post("/api/risk/resume")


# ============================================================================
# Settings Tests
# ============================================================================

class TestSettings:
    """Test settings endpoints."""
    
    def test_get_settings(self, client):
        """Should return settings."""
        response = client.get("/api/settings")
        assert response.status_code == 200
        data = response.json()
        assert "paper_trading" in data
        assert "max_position_size_pct" in data
    
    def test_update_settings(self, client):
        """Should update settings."""
        response = client.put("/api/settings", json={
            "auto_trade": True
        })
        assert response.status_code == 200
    
    def test_get_limits(self, client):
        """Should return risk limits."""
        response = client.get("/api/settings/limits")
        assert response.status_code == 200
        data = response.json()
        assert "max_position_pct" in data


# ============================================================================
# WebSocket Tests
# ============================================================================

class TestWebSocket:
    """Test WebSocket connections."""
    
    def test_signals_websocket(self, client):
        """Should connect to signals WebSocket."""
        with client.websocket_connect("/ws/signals") as websocket:
            assert websocket is not None
    
    def test_positions_websocket(self, client):
        """Should connect to positions WebSocket."""
        with client.websocket_connect("/ws/positions") as websocket:
            assert websocket is not None
    
    def test_portfolio_websocket(self, client):
        """Should connect to portfolio WebSocket."""
        with client.websocket_connect("/ws/portfolio") as websocket:
            assert websocket is not None
    
    def test_risk_websocket(self, client):
        """Should connect to risk WebSocket."""
        with client.websocket_connect("/ws/risk") as websocket:
            assert websocket is not None


# ============================================================================
# WebSocket Manager Tests
# ============================================================================

class TestWebSocketManager:
    """Test WebSocket manager."""
    
    def test_connection_count_empty(self):
        """Should return 0 for empty."""
        assert ws_manager.get_connection_count("nonexistent") == 0
    
    @pytest.mark.asyncio
    async def test_broadcast_no_connections(self):
        """Should not fail with no connections."""
        await ws_manager.broadcast("signals", {"type": "test"})


# ============================================================================
# OpenAPI Tests
# ============================================================================

class TestOpenAPI:
    """Test OpenAPI documentation."""
    
    def test_openapi_schema(self, client):
        """Should return OpenAPI schema."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data
    
    def test_swagger_ui(self, client):
        """Swagger docs should be accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc(self, client):
        """ReDoc should be accessible."""
        response = client.get("/redoc")
        assert response.status_code == 200
