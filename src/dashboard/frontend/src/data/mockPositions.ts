/**
 * Mock positions data for development
 */
import type { Position } from '../types';

const now = new Date();

export const mockPositions: Position[] = [
    {
        id: 'pos_1',
        marketId: 'market_1',
        marketName: 'Will Trump win the 2024 presidential election?',
        side: 'YES',
        size: 500,
        entryPrice: 0.48,
        currentPrice: 0.52,
        marketValue: 260,
        costBasis: 240,
        unrealizedPnl: 20,
        unrealizedPnlPct: 8.33,
        openedAt: new Date(now.getTime() - 3600000 * 24).toISOString(),
        timeHeld: 86400, // 1 day
        correlatedPositions: ['pos_2'],
    },
    {
        id: 'pos_2',
        marketId: 'market_2',
        marketName: 'Will Trump win the popular vote?',
        side: 'YES',
        size: 200,
        entryPrice: 0.32,
        currentPrice: 0.35,
        marketValue: 70,
        costBasis: 64,
        unrealizedPnl: 6,
        unrealizedPnlPct: 9.38,
        openedAt: new Date(now.getTime() - 3600000 * 12).toISOString(),
        timeHeld: 43200, // 12 hours
        correlatedPositions: ['pos_1'],
    },
    {
        id: 'pos_3',
        marketId: 'market_3',
        marketName: 'Will Bitcoin reach $100,000 in 2024?',
        side: 'NO',
        size: 300,
        entryPrice: 0.62,
        currentPrice: 0.58,
        marketValue: 174,
        costBasis: 186,
        unrealizedPnl: -12,
        unrealizedPnlPct: -6.45,
        openedAt: new Date(now.getTime() - 3600000 * 48).toISOString(),
        timeHeld: 172800, // 2 days
        correlatedPositions: [],
    },
];

export default mockPositions;
