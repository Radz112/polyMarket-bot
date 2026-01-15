/**
 * Correlation Detail Panel
 */
import React from 'react';
import type { Correlation } from '../../types';
import {
    MarketCell,
    CorrelationTypeBadge,
    DivergenceDisplay,
    ScoreBadge,
    formatPrice,
} from '../shared';
import './CorrelationDetail.css';

interface Props {
    correlation: Correlation;
    onClose: () => void;
}

export const CorrelationDetail: React.FC<Props> = ({ correlation, onClose }) => {
    const handleTrade = () => {
        console.log('Execute trade for', correlation.id);
        // TODO: Open trade modal
    };

    return (
        <div className="correlation-detail-overlay" onClick={onClose}>
            <div className="correlation-detail-panel" onClick={(e) => e.stopPropagation()}>
                <header className="detail-header">
                    <h2>Correlation Details</h2>
                    <button className="close-btn" onClick={onClose}>
                        ×
                    </button>
                </header>

                <div className="markets-comparison">
                    <div className="market-card">
                        <div className="market-card-header">Market A</div>
                        <h3>{correlation.marketA.question}</h3>
                        <div className="market-prices-large">
                            <div className="price-row">
                                <span className="label">YES</span>
                                <span className="value yes">{formatPrice(correlation.marketA.yesPrice)}</span>
                            </div>
                            <div className="price-row">
                                <span className="label">NO</span>
                                <span className="value no">{formatPrice(correlation.marketA.noPrice)}</span>
                            </div>
                        </div>
                        <div className="market-meta">
                            <span className="category">{correlation.marketA.category}</span>
                            <span className="volume">Vol: ${(correlation.marketA.volume24h / 1000).toFixed(0)}k</span>
                        </div>
                    </div>

                    <div className="correlation-indicator">
                        <CorrelationTypeBadge type={correlation.type} large />
                        <div className="relationship-text">{correlation.expectedRelationship}</div>
                        <div className="confidence-row">
                            <span>Confidence:</span>
                            <strong>{(correlation.confidence * 100).toFixed(0)}%</strong>
                            {correlation.verified && <span className="verified">✓ Verified</span>}
                        </div>
                    </div>

                    <div className="market-card">
                        <div className="market-card-header">Market B</div>
                        <h3>{correlation.marketB.question}</h3>
                        <div className="market-prices-large">
                            <div className="price-row">
                                <span className="label">YES</span>
                                <span className="value yes">{formatPrice(correlation.marketB.yesPrice)}</span>
                            </div>
                            <div className="price-row">
                                <span className="label">NO</span>
                                <span className="value no">{formatPrice(correlation.marketB.noPrice)}</span>
                            </div>
                        </div>
                        <div className="market-meta">
                            <span className="category">{correlation.marketB.category}</span>
                            <span className="volume">Vol: ${(correlation.marketB.volume24h / 1000).toFixed(0)}k</span>
                        </div>
                    </div>
                </div>

                <div className="divergence-section">
                    <h3>Current Divergence</h3>
                    <div className="divergence-large">
                        <DivergenceDisplay
                            expected={correlation.expectedValue}
                            actual={correlation.actualValue}
                            divergence={correlation.divergence}
                        />
                    </div>
                </div>

                {correlation.hasSignal && correlation.signalScore && (
                    <div className="signal-section">
                        <h3>Trading Signal</h3>
                        <div className="signal-card">
                            <div className="signal-score">
                                <span className="label">Score</span>
                                <ScoreBadge score={correlation.signalScore} />
                            </div>
                            <div className="signal-recommendation">
                                <p>
                                    Divergence detected between these correlated markets.
                                    Consider trading to capture the mispricing.
                                </p>
                            </div>
                            <button className="trade-btn-large" onClick={handleTrade}>
                                Execute Trade
                            </button>
                        </div>
                    </div>
                )}

                <div className="chart-placeholder">
                    <div className="placeholder-content">
                        📈 Price comparison chart coming soon
                    </div>
                </div>
            </div>
        </div>
    );
};

export default CorrelationDetail;
