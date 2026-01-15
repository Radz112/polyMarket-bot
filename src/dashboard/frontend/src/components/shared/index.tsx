/**
 * Shared components
 */
import React from 'react';
import type { Market, CorrelationType } from '../../types';
import './shared.css';

// Format price as cents
export const formatPrice = (price: number): string => {
    return `${(price * 100).toFixed(1)}¢`;
};

// Market Cell - displays market name and prices
export const MarketCell: React.FC<{ market: Market }> = ({ market }) => (
    <div className="market-cell">
        <div className="market-name" title={market.question}>
            {market.question.length > 40
                ? market.question.substring(0, 40) + '...'
                : market.question}
        </div>
        <div className="market-prices">
            <span className="price yes">YES: {formatPrice(market.yesPrice)}</span>
            <span className="price no">NO: {formatPrice(market.noPrice)}</span>
        </div>
    </div>
);

// Price Badge - colored price display
export const PriceBadge: React.FC<{
    price: number;
    type?: 'yes' | 'no' | 'neutral';
}> = ({ price, type = 'neutral' }) => (
    <span className={`price-badge ${type}`}>
        {formatPrice(price)}
    </span>
);

// Score Badge - signal score display
export const ScoreBadge: React.FC<{ score: number }> = ({ score }) => {
    const level = score >= 80 ? 'high' : score >= 60 ? 'medium' : 'low';
    return (
        <span className={`score-badge ${level}`}>
            {score.toFixed(0)}
        </span>
    );
};

// Correlation Type Badge
const typeLabels: Record<CorrelationType, string> = {
    equivalent: 'EQ',
    mathematical: 'MATH',
    inverse: 'INV',
    causal: 'CAUSE',
    temporal: 'TIME',
};

const typeColors: Record<CorrelationType, string> = {
    equivalent: '#3b82f6',
    mathematical: '#8b5cf6',
    inverse: '#ef4444',
    causal: '#22c55e',
    temporal: '#f59e0b',
};

export const CorrelationTypeBadge: React.FC<{
    type: CorrelationType;
    large?: boolean;
}> = ({ type, large }) => (
    <span
        className={`correlation-type-badge ${large ? 'large' : ''}`}
        style={{ backgroundColor: typeColors[type] }}
        title={type}
    >
        {typeLabels[type]}
    </span>
);

// Divergence Display
export const DivergenceDisplay: React.FC<{
    expected: number;
    actual: number;
    divergence: number;
}> = ({ expected, actual, divergence }) => {
    const isSignificant = Math.abs(divergence) > 0.03;
    const isPositive = divergence > 0;

    return (
        <div className={`divergence-display ${isSignificant ? 'significant' : ''}`}>
            <div className={`divergence-value ${isPositive ? 'positive' : 'negative'}`}>
                {isPositive ? '+' : ''}{(divergence * 100).toFixed(1)}¢
            </div>
            <div className="divergence-detail">
                <span>Exp: {formatPrice(expected)}</span>
                <span>Act: {formatPrice(actual)}</span>
            </div>
        </div>
    );
};

// Loading Spinner
export const LoadingSpinner: React.FC = () => (
    <div className="loading-spinner">
        <div className="spinner"></div>
    </div>
);

// Empty State
export const EmptyState: React.FC<{ message: string }> = ({ message }) => (
    <div className="empty-state">
        <p>{message}</p>
    </div>
);
