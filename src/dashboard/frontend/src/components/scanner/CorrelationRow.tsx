/**
 * Correlation Row Component
 */
import React from 'react';
import type { Correlation } from '../../types';
import {
    MarketCell,
    CorrelationTypeBadge,
    DivergenceDisplay,
    ScoreBadge,
} from '../shared';
import './CorrelationRow.css';

interface Props {
    correlation: Correlation;
    onClick: () => void;
}

const getDivergenceClass = (divergence: number): string => {
    const abs = Math.abs(divergence);
    if (abs >= 0.05) return 'row-opportunity';
    if (abs >= 0.03) return 'row-divergent';
    return '';
};

export const CorrelationRow: React.FC<Props> = ({ correlation, onClick }) => {
    const divergenceClass = getDivergenceClass(correlation.divergence);

    const handleTrade = (e: React.MouseEvent) => {
        e.stopPropagation();
        // TODO: Open trade modal
        console.log('Trade clicked for', correlation.id);
    };

    return (
        <tr className={`correlation-row ${divergenceClass}`} onClick={onClick}>
            <td className="market-col">
                <MarketCell market={correlation.marketA} />
            </td>
            <td className="market-col">
                <MarketCell market={correlation.marketB} />
            </td>
            <td className="type-col">
                <CorrelationTypeBadge type={correlation.type} />
            </td>
            <td className="relationship-col">
                <span className="relationship">{correlation.expectedRelationship}</span>
            </td>
            <td className="divergence-col">
                <DivergenceDisplay
                    expected={correlation.expectedValue}
                    actual={correlation.actualValue}
                    divergence={correlation.divergence}
                />
            </td>
            <td className="confidence-col">
                <span className="confidence">
                    {(correlation.confidence * 100).toFixed(0)}%
                </span>
                {correlation.verified && (
                    <span className="verified-badge" title="Verified">✓</span>
                )}
            </td>
            <td className="score-col">
                {correlation.signalScore ? (
                    <ScoreBadge score={correlation.signalScore} />
                ) : (
                    <span className="no-signal">—</span>
                )}
            </td>
            <td className="actions-col">
                {correlation.hasSignal && (
                    <button className="trade-btn" onClick={handleTrade}>
                        Trade
                    </button>
                )}
                <button className="view-btn" onClick={onClick}>
                    View
                </button>
            </td>
        </tr>
    );
};

export default CorrelationRow;
