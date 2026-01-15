-- Markets
CREATE TABLE IF NOT EXISTS markets (
    id TEXT PRIMARY KEY,
    slug TEXT UNIQUE NOT NULL,
    question TEXT NOT NULL,
    description TEXT,
    category TEXT,
    subcategory TEXT,
    end_date TIMESTAMP,
    active BOOLEAN DEFAULT TRUE,
    resolved BOOLEAN DEFAULT FALSE,
    outcome TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Market correlations
CREATE TABLE IF NOT EXISTS market_correlations (
    id SERIAL PRIMARY KEY,
    market_a_id TEXT NOT NULL REFERENCES markets(id),
    market_b_id TEXT NOT NULL REFERENCES markets(id),
    correlation_type TEXT NOT NULL,
    expected_relationship TEXT,
    confidence FLOAT DEFAULT 0.5,
    manual_verified BOOLEAN DEFAULT FALSE,
    historical_correlation FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(market_a_id, market_b_id)
);

-- Price snapshots
CREATE TABLE IF NOT EXISTS price_snapshots (
    id SERIAL PRIMARY KEY,
    market_id TEXT NOT NULL REFERENCES markets(id),
    token_id TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    yes_price FLOAT,
    no_price FLOAT,
    volume FLOAT
);
CREATE INDEX IF NOT EXISTS idx_price_snapshots_market_time ON price_snapshots(market_id, timestamp);

-- Signals
CREATE TABLE IF NOT EXISTS signals (
    id TEXT PRIMARY KEY,
    signal_type TEXT NOT NULL,
    market_ids TEXT[] NOT NULL,
    divergence_amount FLOAT,
    expected_value FLOAT,
    actual_value FLOAT,
    confidence FLOAT,
    score FLOAT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    acted_on BOOLEAN DEFAULT FALSE,
    outcome FLOAT
);

-- Trades
CREATE TABLE IF NOT EXISTS trades (
    id TEXT PRIMARY KEY,
    market_id TEXT NOT NULL,
    side TEXT NOT NULL,
    action TEXT NOT NULL,
    size FLOAT NOT NULL,
    price FLOAT NOT NULL,
    fees FLOAT DEFAULT 0,
    order_id TEXT,
    signal_id TEXT REFERENCES signals(id),
    is_paper BOOLEAN DEFAULT TRUE,
    realized_pnl FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Positions
CREATE TABLE IF NOT EXISTS positions (
    id TEXT PRIMARY KEY,
    market_id TEXT NOT NULL,
    side TEXT NOT NULL,
    size FLOAT NOT NULL,
    entry_price FLOAT NOT NULL,
    opened_at TIMESTAMP DEFAULT NOW(),
    closed_at TIMESTAMP,
    realized_pnl FLOAT,
    status TEXT DEFAULT 'open'
);
