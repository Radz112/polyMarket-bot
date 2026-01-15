/**
 * Position Detail Modal
 */
import React, { useState } from 'react';
import type { Position } from '../../types';
import { formatPrice } from '../shared';
import './PositionDetailModal.css';

interface Props {
    position: Position;
    onClose: () => void;
    onClosePosition: (position: Position) => void;
    onReducePosition: (position: Position, size: number) => void;
}

const formatDuration = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const days = Math.floor(hours / 24);
    const remainingHours = hours % 24;
    if (days > 0) {
        return `${days}d ${remainingHours}h`;
    }
    const mins = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${mins}m`;
};

export const PositionDetailModal: React.FC<Props> = ({
    position,
    onClose,
    onClosePosition,
    onReducePosition,
}) => {
    const [reduceSize, setReduceSize] = useState(0);
    const pnlClass = position.unrealizedPnl >= 0 ? 'positive' : 'negative';

    const handleReduce = () => {
        if (reduceSize > 0) {
            onReducePosition(position, reduceSize);
            onClose();
        }
    };

    const handleCloseAll = () => {
        onClosePosition(position);
        onClose();
    };

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="position-detail-modal" onClick={(e) => e.stopPropagation()}>
                <header className="modal-header">
                    <h2>Position Details</h2>
                    <button className="close-btn" onClick={onClose}>×</button>
                </header>

                {/* Position Info */}
                <section className="position-info">
                    <h3>{position.marketName}</h3>
                    <span className={`side-badge large ${position.side.toLowerCase()}`}>
                        {position.side}
                    </span>
                </section>

                {/* Stats */}
                <section className="position-stats">
                    <div className="stat">
                        <span className="label">Size</span>
                        <span className="value">${position.size.toFixed(2)}</span>
                    </div>
                    <div className="stat">
                        <span className="label">Entry Price</span>
                        <span className="value">{formatPrice(position.entryPrice)}</span>
                    </div>
                    <div className="stat">
                        <span className="label">Current Price</span>
                        <span className="value">{formatPrice(position.currentPrice)}</span>
                    </div>
                    <div className="stat">
                        <span className="label">Market Value</span>
                        <span className="value">${position.marketValue.toFixed(2)}</span>
                    </div>
                    <div className="stat">
                        <span className="label">Cost Basis</span>
                        <span className="value">${position.costBasis.toFixed(2)}</span>
                    </div>
                    <div className="stat">
                        <span className="label">Time Held</span>
                        <span className="value">{formatDuration(position.timeHeld)}</span>
                    </div>
                </section>

                {/* P&L */}
                <section className={`pnl-section ${pnlClass}`}>
                    <div className="pnl-main">
                        <span className="label">Unrealized P&L</span>
                        <span className="value">
                            {position.unrealizedPnl >= 0 ? '+' : ''}${position.unrealizedPnl.toFixed(2)}
                        </span>
                    </div>
                    <div className="pnl-pct">
                        ({position.unrealizedPnlPct >= 0 ? '+' : ''}{position.unrealizedPnlPct.toFixed(1)}%)
                    </div>
                </section>

                {/* Correlated Positions Warning */}
                {position.correlatedPositions.length > 0 && (
                    <section className="correlated-warning">
                        <h4>⚠️ Correlated Positions</h4>
                        <p>
                            You have {position.correlatedPositions.length} other position(s)
                            that may be affected by this market.
                        </p>
                    </section>
                )}

                {/* Chart Placeholder */}
                <section className="chart-placeholder">
                    <div className="placeholder-content">
                        📈 Price chart since entry coming soon
                    </div>
                </section>

                {/* Actions */}
                <section className="actions-section">
                    <div className="close-full">
                        <button className="close-full-btn" onClick={handleCloseAll}>
                            Close Full Position
                        </button>
                        <span className="estimate">
                            Est. P&L: {position.unrealizedPnl >= 0 ? '+' : ''}${position.unrealizedPnl.toFixed(2)}
                        </span>
                    </div>

                    <div className="reduce-partial">
                        <h4>Reduce Position</h4>
                        <div className="reduce-slider">
                            <input
                                type="range"
                                min={0}
                                max={position.size}
                                step={1}
                                value={reduceSize}
                                onChange={(e) => setReduceSize(+e.target.value)}
                            />
                            <span className="reduce-amount">
                                ${reduceSize.toFixed(2)} of ${position.size.toFixed(2)}
                            </span>
                        </div>
                        <button
                            className="reduce-btn"
                            onClick={handleReduce}
                            disabled={reduceSize === 0}
                        >
                            Reduce
                        </button>
                    </div>
                </section>
            </div>
        </div>
    );
};

export default PositionDetailModal;
