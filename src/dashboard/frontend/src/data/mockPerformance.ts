/**
 * Mock performance data for development
 */
import type { PerformanceData, Trade, EquityPoint } from '../types/performance';

const now = new Date();
const generateEquityCurve = (): EquityPoint[] => {
    const points: EquityPoint[] = [];
    let value = 10000;
    let maxValue = value;

    for (let i = 30; i >= 0; i--) {
        const date = new Date(now.getTime() - i * 24 * 3600000);
        const change = (Math.random() - 0.45) * 200; // Slight upward bias
        value = Math.max(value + change, 8000);
        maxValue = Math.max(maxValue, value);
        const drawdown = ((maxValue - value) / maxValue) * 100;

        points.push({
            timestamp: date.toISOString(),
            value: Math.round(value * 100) / 100,
            drawdown: Math.round(drawdown * 100) / 100,
            trades: Math.floor(Math.random() * 3),
        });
    }
    return points;
};

export const mockPerformanceData: PerformanceData = {
    totalReturn: 1847.32,
    totalReturnPct: 18.47,
    winRate: 0.62,
    profitFactor: 1.85,
    sharpeRatio: 1.42,
    equityCurve: generateEquityCurve(),
    returnsByPeriod: [
        { period: 'Today', return: 125.50, returnPct: 1.05 },
        { period: 'This Week', return: 342.80, returnPct: 2.95 },
        { period: 'This Month', return: 1847.32, returnPct: 18.47 },
        { period: 'Last Month', return: -234.10, returnPct: -2.29 },
        { period: 'YTD', return: 2156.45, returnPct: 21.56 },
    ],
    returnsDistribution: [
        -150, -120, -80, -50, -30, -20, -10, 5, 15, 25, 35, 50, 75, 100, 125, 180
    ],
    tradeStats: {
        totalTrades: 156,
        winningTrades: 97,
        losingTrades: 59,
        winRate: 0.622,
        averageWin: 45.32,
        averageLoss: -28.67,
        largestWin: 285.00,
        largestLoss: -142.50,
        avgHoldingPeriod: 43200, // 12 hours
        profitFactor: 1.85,
        expectancy: 11.84,
        currentStreak: 3,
        maxWinStreak: 8,
        maxLossStreak: 4,
    },
    drawdowns: [
        {
            startDate: new Date(now.getTime() - 5 * 24 * 3600000).toISOString(),
            endDate: null,
            depth: 3.2,
            currentPct: 3.2,
            maxPct: 8.5,
            maxDate: new Date(now.getTime() - 15 * 24 * 3600000).toISOString(),
            duration: 5 * 24 * 3600,
            recoveryDuration: null,
        },
        {
            startDate: new Date(now.getTime() - 20 * 24 * 3600000).toISOString(),
            endDate: new Date(now.getTime() - 12 * 24 * 3600000).toISOString(),
            depth: 8.5,
            currentPct: 0,
            maxPct: 8.5,
            maxDate: new Date(now.getTime() - 15 * 24 * 3600000).toISOString(),
            duration: 8 * 24 * 3600,
            recoveryDuration: 3 * 24 * 3600,
        },
    ],
    byCategory: [
        { category: 'Politics', trades: 45, winRate: 0.68, profit: 1250, loss: -380, netPnl: 870, avgTrade: 19.33 },
        { category: 'Crypto', trades: 38, winRate: 0.58, profit: 890, loss: -520, netPnl: 370, avgTrade: 9.74 },
        { category: 'Economics', trades: 28, winRate: 0.64, profit: 560, loss: -280, netPnl: 280, avgTrade: 10.00 },
        { category: 'Sports', trades: 22, winRate: 0.55, profit: 420, loss: -340, netPnl: 80, avgTrade: 3.64 },
        { category: 'Entertainment', trades: 23, winRate: 0.61, profit: 380, loss: -133, netPnl: 247, avgTrade: 10.74 },
    ],
};

export const mockTrades: Trade[] = [
    {
        id: 'trade_1',
        timestamp: new Date(now.getTime() - 1800000).toISOString(),
        marketId: 'market_1',
        marketName: 'Will Trump win the 2024 election?',
        side: 'YES',
        action: 'SELL',
        size: 250,
        price: 0.54,
        fees: 5.00,
        realizedPnl: 42.50,
        signalId: 'sig_1',
        signalScore: 78,
    },
    {
        id: 'trade_2',
        timestamp: new Date(now.getTime() - 3600000).toISOString(),
        marketId: 'market_1',
        marketName: 'Will Trump win the 2024 election?',
        side: 'YES',
        action: 'BUY',
        size: 250,
        price: 0.48,
        fees: 5.00,
        realizedPnl: null,
    },
    {
        id: 'trade_3',
        timestamp: new Date(now.getTime() - 7200000).toISOString(),
        marketId: 'market_3',
        marketName: 'Will Bitcoin reach $100k?',
        side: 'NO',
        action: 'SELL',
        size: 150,
        price: 0.62,
        fees: 3.00,
        realizedPnl: -18.50,
        signalId: 'sig_2',
        signalScore: 65,
    },
    {
        id: 'trade_4',
        timestamp: new Date(now.getTime() - 86400000).toISOString(),
        marketId: 'market_2',
        marketName: 'Will Trump win popular vote?',
        side: 'YES',
        action: 'BUY',
        size: 200,
        price: 0.32,
        fees: 4.00,
        realizedPnl: null,
        signalId: 'sig_3',
        signalScore: 82,
    },
];

export default mockPerformanceData;
