"""
FastAPI main application for Dashboard API.
"""
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.dashboard.api.routes import (
    markets_router,
    signals_router,
    positions_router,
    portfolio_router,
    orders_router,
    risk_router,
    settings_router,
    system_router,
)
from src.dashboard.api.websocket import ws_manager
from src.dashboard.api.dependencies import app_state
from src.config.settings import Config

# Create FastAPI app
app = FastAPI(
    title="Polymarket Bot API",
    description="API for Polymarket arbitrage bot dashboard",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(markets_router)
app.include_router(signals_router)
app.include_router(positions_router)
app.include_router(portfolio_router)
app.include_router(orders_router)
app.include_router(risk_router)
app.include_router(settings_router)
app.include_router(system_router)


# ============ Startup/Shutdown ============

@app.on_event("startup")
async def startup_event():
    """Initialize application state on startup."""
    config = Config()
    app_state.initialize(config)
    print(f"🚀 Polymarket Bot API started")
    print(f"   Paper trading: {config.paper_trading}")
    print(f"   Docs: http://localhost:8000/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("👋 Polymarket Bot API shutting down")


# ============ WebSocket Endpoints ============

@app.websocket("/ws/signals")
async def signals_websocket(websocket: WebSocket):
    """
    Real-time signal updates.
    
    Messages:
    - {"type": "new_signal", "signal": {...}}
    - {"type": "signal_update", "signal": {...}}
    - {"type": "signal_expired", "signal_id": "..."}
    """
    await ws_manager.connect("signals", websocket)
    try:
        while True:
            # Keep connection alive, receive any client messages
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect("signals", websocket)


@app.websocket("/ws/positions")
async def positions_websocket(websocket: WebSocket):
    """
    Real-time position updates.
    
    Messages:
    - {"type": "position_update", "position": {...}}
    - {"type": "position_closed", "position_id": "...", "pnl": ...}
    """
    await ws_manager.connect("positions", websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect("positions", websocket)


@app.websocket("/ws/portfolio")
async def portfolio_websocket(websocket: WebSocket):
    """
    Real-time portfolio updates (every second).
    
    Messages:
    - {"type": "portfolio_update", "value": ..., "pnl": ...}
    """
    await ws_manager.connect("portfolio", websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect("portfolio", websocket)


@app.websocket("/ws/risk")
async def risk_websocket(websocket: WebSocket):
    """
    Real-time risk alerts.
    
    Messages:
    - {"type": "risk_alert", "alert": {...}}
    - {"type": "breaker_tripped", "breaker": "..."}
    """
    await ws_manager.connect("risk", websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect("risk", websocket)


# ============ Root ============

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Polymarket Bot API",
        "version": "1.0.0",
        "docs": "/docs"
    }


# ============ Run Server ============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
