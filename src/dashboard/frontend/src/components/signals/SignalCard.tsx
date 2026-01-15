/**
 * Signal Card Component
 */
import React from 'react';
import type { ExtendedSignal } from '../../types';
import { ScoreBadge, formatPrice } from '../shared';
import { CountdownTimer } from '../shared/CountdownTimer';
import './SignalCard.css';

interface Props {
    signal: ExtendedSignal;
    onSelect: () => void;
    onTrade: (signal: ExtendedSignal) => void;
    onDismiss: (signal: ExtendedSignal) => void;
}

const formatTimeAgo = (dateStr: string): string => {
    const seconds = Math.floor((Date.now() - new Date(dateStr).getTime()) / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    return `${hours}h ago`;
};

const urgencyColors: Record<string, string> = {
    critical: '#ef4444',
    high: '#f59e0b',
    medium: '#3b82f6',
    low: '#6b7280',
};

export const SignalCard: React.FC<Props> = ({
    signal,
    onSelect,
    onTrade,
    onDismiss,
}) => {
    return (
        <div
            className={`signal-card urgency-${signal.urgency}`}
            onClick={onSelect}
            style={{ borderLeftColor: urgencyColors[signal.urgency] }}
        >
            {/* Header */}
            <div className="signal-header">
                <ScoreBadge score={signal.score} />
                <span className="signal-type">{signal.signalType}</span>
                <span className="signal-time">{formatTimeAgo(signal.detectedAt)}</span>
            </div>

            {/* Markets */}
            <div className="signal-markets">
                {signal.markets.map((market) => (
                    <div key={market.id} className="signal-market">
                        <span className="market-name">{market.question}</span>
                        <span className="market-price">{formatPrice(market.yesPrice)}</span>
                    </div>
                ))}
            </div>

            {/* Divergence */}
            <div className="signal-divergence">
                <span className="divergence-label">Divergence</span>
                <span className="divergence-value">
                    {(signal.divergenceAmount * 100).toFixed(1)}¢
                </span>
            </div>

            {/* Recommendation */}
            <div className="signal-recommendation">
                <span className={`action ${signal.recommendedAction.toLowerCase()}`}>
                    {signal.recommendedAction}
                </span>
                <span className="size">Size: ${signal.recommendedSize.toFixed(0)}</span>
            </div>

            {/* Expiry */}
            {signal.expiresAt && (
                <div className="signal-expiry">
                    <span className="expiry-label">Expires in</span>
                    <CountdownTimer expiresAt={signal.expiresAt} />
                </div>
            )}

            {/* Actions */}
            <div className="signal-actions">
                <button
                    className="trade-button"
                    onClick={(e) => {
                        e.stopPropagation();
                        onTrade(signal);
                    }}
                >
                    Trade
                </button>
                <button
                    className="dismiss-button"
                    onClick={(e) => {
                        e.stopPropagation();
                        onDismiss(signal);
                    }}
                >
                    Dismiss
                </button>
            </div>
        </div>
    );
};

export default SignalCard;
