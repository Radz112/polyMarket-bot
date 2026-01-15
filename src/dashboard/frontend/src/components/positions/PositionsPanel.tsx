/**
 * Positions Panel - Main Component
 */
import React, { useState, useMemo } from 'react';
import type { Position } from '../../types';
import { PositionCard } from './PositionCard';
import { PositionDetailModal } from './PositionDetailModal';
import { usePositionUpdates } from '../../hooks/usePositionUpdates';
import { EmptyState } from '../shared';
import './PositionsPanel.css';

type SortOption = 'pnl' | 'size' | 'time';

const sortPositions = (positions: Position[], sortBy: SortOption): Position[] => {
    return [...positions].sort((a, b) => {
        switch (sortBy) {
            case 'pnl':
                return b.unrealizedPnl - a.unrealizedPnl;
            case 'size':
                return b.size - a.size;
            case 'time':
                return b.timeHeld - a.timeHeld;
            default:
                return 0;
        }
    });
};

export const PositionsPanel: React.FC = () => {
    const { positions, connected, closePosition, reducePosition } = usePositionUpdates();
    const [selectedPosition, setSelectedPosition] = useState<Position | null>(null);
    const [sortBy, setSortBy] = useState<SortOption>('pnl');

    const sortedPositions = useMemo(
        () => sortPositions(positions, sortBy),
        [positions, sortBy]
    );

    const totalValue = positions.reduce((sum, p) => sum + p.marketValue, 0);
    const totalPnL = positions.reduce((sum, p) => sum + p.unrealizedPnl, 0);
    const pnlClass = totalPnL >= 0 ? 'positive' : 'negative';

    const handleClosePosition = async (position: Position) => {
        const result = await closePosition(position.id);
        if (result.success) {
            setSelectedPosition(null);
        }
    };

    const handleReducePosition = async (position: Position, size: number) => {
        const result = await reducePosition(position.id, size);
        if (result.success) {
            setSelectedPosition(null);
        }
    };

    return (
        <div className="positions-panel">
            <header className="positions-header">
                <div className="header-left">
                    <h2>Open Positions</h2>
                    <span className="position-count">{positions.length}</span>
                    <div className={`connection-dot ${connected ? 'connected' : ''}`} />
                </div>
                <div className="positions-summary">
                    <div className="summary-item">
                        <span className="label">Total Value</span>
                        <span className="value">${totalValue.toFixed(2)}</span>
                    </div>
                    <div className={`summary-item ${pnlClass}`}>
                        <span className="label">Unrealized P&L</span>
                        <span className="value">
                            {totalPnL >= 0 ? '+' : ''}${totalPnL.toFixed(2)}
                        </span>
                    </div>
                </div>
            </header>

            <div className="sort-options">
                <span className="sort-label">Sort by:</span>
                <button
                    className={sortBy === 'pnl' ? 'active' : ''}
                    onClick={() => setSortBy('pnl')}
                >
                    P&L
                </button>
                <button
                    className={sortBy === 'size' ? 'active' : ''}
                    onClick={() => setSortBy('size')}
                >
                    Size
                </button>
                <button
                    className={sortBy === 'time' ? 'active' : ''}
                    onClick={() => setSortBy('time')}
                >
                    Time
                </button>
            </div>

            <div className="positions-list">
                {sortedPositions.length > 0 ? (
                    sortedPositions.map((position) => (
                        <PositionCard
                            key={position.id}
                            position={position}
                            onSelect={() => setSelectedPosition(position)}
                            onClose={handleClosePosition}
                        />
                    ))
                ) : (
                    <EmptyState message="No open positions" />
                )}
            </div>

            {selectedPosition && (
                <PositionDetailModal
                    position={selectedPosition}
                    onClose={() => setSelectedPosition(null)}
                    onClosePosition={handleClosePosition}
                    onReducePosition={handleReducePosition}
                />
            )}
        </div>
    );
};

export default PositionsPanel;
