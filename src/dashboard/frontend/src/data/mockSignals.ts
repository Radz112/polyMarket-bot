/**
 * Mock signals data for development
 */
import type { ExtendedSignal } from '../types';

const now = new Date();
const addMinutes = (mins: number) => new Date(now.getTime() + mins * 60000).toISOString();

export const mockSignals: ExtendedSignal[] = [
    {
        id: 'sig_1',
        signalType: 'divergence',
        markets: [
            {
                id: 'market_1',
                slug: 'trump-wins-2024',
                question: 'Will Trump win the 2024 presidential election?',
                category: 'Politics',
                endDate: '2024-11-06',
                active: true,
                yesPrice: 0.52,
                noPrice: 0.48,
                volume24h: 125000,
            },
            {
                id: 'market_2',
                slug: 'trump-popular-vote',
                question: 'Will Trump win the popular vote?',
                category: 'Politics',
                endDate: '2024-11-06',
                active: true,
                yesPrice: 0.35,
                noPrice: 0.65,
                volume24h: 45000,
            },
        ],
        divergenceAmount: 0.17,
        score: 82,
        recommendedAction: 'BUY',
        recommendedSize: 500,
        recommendedPrice: 0.35,
        detectedAt: new Date(now.getTime() - 120000).toISOString(),
        expiresAt: addMinutes(3),
        status: 'active',
        urgency: 'high',
        componentScores: [
            { name: 'Divergence', score: 90, weight: 0.3, contribution: 27 },
            { name: 'Liquidity', score: 75, weight: 0.25, contribution: 18.75 },
            { name: 'Confidence', score: 85, weight: 0.25, contribution: 21.25 },
            { name: 'Time Value', score: 60, weight: 0.2, contribution: 12 },
        ],
        explanation: 'Electoral win implies popular vote is possible, but popular vote priced too low relative to electoral.',
        expectedRelationship: 'A ≥ B',
        actualValue: 0.35,
        maxExecutableSize: 2000,
        estimatedFillPrice: 0.36,
        estimatedFees: 10,
        expectedProfit: 75,
    },
    {
        id: 'sig_2',
        signalType: 'mathematical',
        markets: [
            {
                id: 'market_3',
                slug: 'bitcoin-100k-2024',
                question: 'Will Bitcoin reach $100,000 in 2024?',
                category: 'Crypto',
                endDate: '2024-12-31',
                active: true,
                yesPrice: 0.42,
                noPrice: 0.58,
                volume24h: 89000,
            },
            {
                id: 'market_4',
                slug: 'bitcoin-50k-2024',
                question: 'Will Bitcoin stay above $50,000 in 2024?',
                category: 'Crypto',
                endDate: '2024-12-31',
                active: true,
                yesPrice: 0.78,
                noPrice: 0.22,
                volume24h: 67000,
            },
        ],
        divergenceAmount: 0.08,
        score: 65,
        recommendedAction: 'BUY',
        recommendedSize: 250,
        recommendedPrice: 0.42,
        detectedAt: new Date(now.getTime() - 300000).toISOString(),
        expiresAt: addMinutes(2),
        status: 'active',
        urgency: 'medium',
        componentScores: [
            { name: 'Divergence', score: 65, weight: 0.3, contribution: 19.5 },
            { name: 'Liquidity', score: 80, weight: 0.25, contribution: 20 },
            { name: 'Confidence', score: 55, weight: 0.25, contribution: 13.75 },
            { name: 'Time Value', score: 50, weight: 0.2, contribution: 10 },
        ],
        explanation: 'If BTC reaches 100k, it necessarily stays above 50k. Mathematical implication.',
        expectedRelationship: 'A → B',
        actualValue: 0.78,
        maxExecutableSize: 1500,
        estimatedFillPrice: 0.43,
        estimatedFees: 5,
        expectedProfit: 35,
    },
];

export default mockSignals;
