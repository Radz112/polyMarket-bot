/**
 * Performance Analytics Types
 */

export type TimeRange = '1d' | '7d' | '30d' | '90d' | '1y' | 'all';

export interface EquityPoint {
    timestamp: string;
    value: number;
    drawdown: number;
    trades?: number;
}

export interface TradeStats {
    totalTrades: number;
    winningTrades: number;
    losingTrades: number;
    winRate: number;
    averageWin: number;
    averageLoss: number;
    largestWin: number;
    largestLoss: number;
    avgHoldingPeriod: number; // seconds
    profitFactor: number;
    expectancy: number;
    currentStreak: number;
    maxWinStreak: number;
    maxLossStreak: number;
}

export interface DrawdownPeriod {
    startDate: string;
    endDate: string | null;
    depth: number;
    currentPct: number;
    maxPct: number;
    maxDate: string;
    duration: number; // seconds
    recoveryDuration: number | null;
}

export interface CategoryPerformance {
    category: string;
    trades: number;
    winRate: number;
    profit: number;
    loss: number;
    netPnl: number;
    avgTrade: number;
}

export interface ReturnPeriod {
    period: string;
    return: number;
    returnPct: number;
}

export interface PerformanceData {
    totalReturn: number;
    totalReturnPct: number;
    winRate: number;
    profitFactor: number;
    sharpeRatio: number;
    equityCurve: EquityPoint[];
    returnsByPeriod: ReturnPeriod[];
    returnsDistribution: number[];
    tradeStats: TradeStats;
    drawdowns: DrawdownPeriod[];
    byCategory: CategoryPerformance[];
}

export interface Trade {
    id: string;
    timestamp: string;
    marketId: string;
    marketName: string;
    side: 'YES' | 'NO';
    action: 'BUY' | 'SELL';
    size: number;
    price: number;
    fees: number;
    realizedPnl: number | null;
    signalId?: string;
    signalScore?: number;
}

export interface TradeFilters {
    dateFrom?: string;
    dateTo?: string;
    side?: 'YES' | 'NO';
    action?: 'BUY' | 'SELL';
    minPnl?: number;
    maxPnl?: number;
    category?: string;
}
