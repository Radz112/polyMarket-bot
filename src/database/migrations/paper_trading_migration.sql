-- Paper Trading Tables Migration
-- Run this to add paper trading tables to the database

-- Paper portfolio state
CREATE TABLE IF NOT EXISTS paper_portfolio (
    id SERIAL PRIMARY KEY,
    balance FLOAT NOT NULL,
    initial_balance FLOAT NOT NULL,
    realized_pnl FLOAT DEFAULT 0.0,
    total_fees FLOAT DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Paper positions
CREATE TABLE IF NOT EXISTS paper_positions (
    id TEXT PRIMARY KEY,
    market_id TEXT NOT NULL,
    market_name TEXT,
    side TEXT NOT NULL,  -- 'YES' or 'NO'
    size FLOAT NOT NULL,
    entry_price FLOAT NOT NULL,
    opened_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    closed_at TIMESTAMP WITH TIME ZONE,
    exit_price FLOAT,
    realized_pnl FLOAT,
    status TEXT DEFAULT 'open',  -- 'open', 'closed', 'resolved'
    signal_id TEXT,
    signal_score FLOAT
);

-- Paper trades
CREATE TABLE IF NOT EXISTS paper_trades (
    id TEXT PRIMARY KEY,
    position_id TEXT,
    market_id TEXT NOT NULL,
    market_name TEXT,
    side TEXT NOT NULL,  -- 'YES' or 'NO'
    action TEXT NOT NULL,  -- 'BUY' or 'SELL'
    size FLOAT NOT NULL,
    price FLOAT NOT NULL,
    fees FLOAT NOT NULL,
    total FLOAT NOT NULL,
    signal_id TEXT,
    signal_score FLOAT,
    realized_pnl FLOAT,
    holding_period_seconds FLOAT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Paper portfolio snapshots for equity curve
CREATE TABLE IF NOT EXISTS paper_snapshots (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    cash_balance FLOAT NOT NULL,
    positions_value FLOAT NOT NULL,
    total_value FLOAT NOT NULL,
    unrealized_pnl FLOAT NOT NULL,
    realized_pnl FLOAT NOT NULL
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_paper_snapshots_time ON paper_snapshots(timestamp);
CREATE INDEX IF NOT EXISTS idx_paper_positions_status ON paper_positions(status);
CREATE INDEX IF NOT EXISTS idx_paper_positions_market ON paper_positions(market_id);
CREATE INDEX IF NOT EXISTS idx_paper_trades_timestamp ON paper_trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_paper_trades_market ON paper_trades(market_id);
