/**
 * TypeScript types for Market Scanner
 */

export interface Market {
    id: string;
    slug: string;
    question: string;
    category: string;
    endDate: string;
    active: boolean;
    yesPrice: number;
    noPrice: number;
    volume24h: number;
}

export interface Correlation {
    id: string;
    marketA: Market;
    marketB: Market;
    type: CorrelationType;
    expectedRelationship: string;
    expectedValue: number;
    actualValue: number;
    divergence: number;
    confidence: number;
    verified: boolean;
    signalScore?: number;
    signal?: Signal;
    hasSignal: boolean;
    createdAt: string;
}

export type CorrelationType =
    | 'equivalent'
    | 'mathematical'
    | 'inverse'
    | 'causal'
    | 'temporal';

export interface Signal {
    id: string;
    signalType: string;
    markets: Market[];
    divergenceAmount: number;
    score: number;
    recommendedAction: 'BUY' | 'SELL';
    recommendedSize: number;
    recommendedPrice?: number;
    detectedAt: string;
    expiresAt?: string;
    status: 'active' | 'traded' | 'expired';
}

export interface ScannerFilters {
    category: string | null;
    correlationType: CorrelationType | null;
    minDivergence: number;
    minConfidence: number;
    showOnlyOpportunities: boolean;
    verifiedOnly: boolean;
}

export type SortOption =
    | 'marketA'
    | 'marketB'
    | 'type'
    | 'divergence'
    | 'score'
    | 'confidence';

export interface WebSocketMessage {
    type: 'divergence_update' | 'new_correlation' | 'correlation_removed' |
    'new_signal' | 'signal_update' | 'signal_expired' |
    'position_update' | 'position_closed';
    correlationId?: string;
    updates?: Partial<Correlation>;
    correlation?: Correlation;
    signal?: Signal;
    signalId?: string;
    position?: Position;
    positionId?: string;
    pnl?: number;
}

export interface Position {
    id: string;
    marketId: string;
    marketName: string;
    side: 'YES' | 'NO';
    size: number;
    entryPrice: number;
    currentPrice: number;
    marketValue: number;
    costBasis: number;
    unrealizedPnl: number;
    unrealizedPnlPct: number;
    openedAt: string;
    timeHeld: number; // seconds
    correlatedPositions: string[];
}

export interface ComponentScore {
    name: string;
    score: number;
    weight: number;
    contribution: number;
}

export interface ExtendedSignal extends Signal {
    urgency: 'low' | 'medium' | 'high' | 'critical';
    componentScores: ComponentScore[];
    explanation: string;
    expectedRelationship: string;
    actualValue: number;
    maxExecutableSize: number;
    estimatedFillPrice: number;
    estimatedFees: number;
    expectedProfit: number;
}
