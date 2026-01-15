/**
 * Position Card Component
 */
import React from 'react';
import type { Position } from '../../types';
import { formatPrice } from '../shared';
import './PositionCard.css';

interface Props {
    position: Position;
    onSelect: () => void;
    onClose: (position: Position) => void;
}

const formatDuration = (seconds: number): string => {
    if (seconds < 3600) {
        const mins = Math.floor(seconds / 60);
        return `${mins}m`;
    }
    const hours = Math.floor(seconds / 3600);
    if (hours < 24) {
        return `${hours}h`;
    }
    const days = Math.floor(hours / 24);
    return `${days}d`;
};

export const PositionCard: React.FC<Props> = ({
    position,
    onSelect,
    onClose,
}) => {
    const pnlClass = position.unrealizedPnl >= 0 ? 'positive' : 'negative';

    return (
        <div className="position-card" onClick={onSelect}>
            <div className="position-header">
                <div className="market-name">{position.marketName}</div>
                <span className={`side-badge ${position.side.toLowerCase()}`}>
                    {position.side}
                </span>
            </div>

            <div className="position-details">
                <div className="detail-row">
                    <span className="label">Size</span>
                    <span className="value">${position.size.toFixed(2)}</span>
                </div>
                <div className="detail-row">
                    <span className="label">Entry</span>
                    <span className="value">{formatPrice(position.entryPrice)}</span>
                </div>
                <div className="detail-row">
                    <span className="label">Current</span>
                    <span className="value">{formatPrice(position.currentPrice)}</span>
                </div>
            </div>

            <div className={`position-pnl ${pnlClass}`}>
                <div className="pnl-value">
                    {position.unrealizedPnl >= 0 ? '+' : ''}${position.unrealizedPnl.toFixed(2)}
                </div>
                <div className="pnl-pct">
                    ({position.unrealizedPnlPct >= 0 ? '+' : ''}
                    {position.unrealizedPnlPct.toFixed(1)}%)
                </div>
            </div>

            <div className="position-time">
                Held for {formatDuration(position.timeHeld)}
            </div>

            <div className="position-actions">
                <button
                    className="close-button"
                    onClick={(e) => {
                        e.stopPropagation();
                        onClose(position);
                    }}
                >
                    Close
                </button>
            </div>
        </div>
    );
};

export default PositionCard;
