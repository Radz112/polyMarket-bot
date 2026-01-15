/**
 * Signal Detail Modal
 */
import React, { useState } from 'react';
import type { ExtendedSignal } from '../../types';
import { ScoreBadge, formatPrice } from '../shared';
import './SignalDetailModal.css';

interface Props {
    signal: ExtendedSignal;
    onClose: () => void;
    onTrade: (signal: ExtendedSignal, size: number) => void;
}

export const SignalDetailModal: React.FC<Props> = ({
    signal,
    onClose,
    onTrade,
}) => {
    const [tradeSize, setTradeSize] = useState(signal.recommendedSize);

    const handleTrade = () => {
        onTrade(signal, tradeSize);
        onClose();
    };

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="signal-detail-modal" onClick={(e) => e.stopPropagation()}>
                <header className="modal-header">
                    <h2>Signal Details</h2>
                    <button className="close-btn" onClick={onClose}>×</button>
                </header>

                {/* Score Breakdown */}
                <section className="score-breakdown">
                    <h3>Score Breakdown</h3>
                    <div className="score-total">
                        <ScoreBadge score={signal.score} />
                    </div>
                    <div className="score-components">
                        {signal.componentScores.map((cs) => (
                            <div key={cs.name} className="score-component">
                                <div className="component-header">
                                    <span className="component-name">{cs.name}</span>
                                    <span className="component-score">{cs.score}</span>
                                </div>
                                <div className="component-bar">
                                    <div
                                        className="component-fill"
                                        style={{ width: `${cs.score}%` }}
                                    />
                                </div>
                                <div className="component-meta">
                                    <span>Weight: {(cs.weight * 100).toFixed(0)}%</span>
                                    <span>Contribution: {cs.contribution.toFixed(1)}</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </section>

                {/* Markets */}
                <section className="markets-section">
                    <h3>Markets Involved</h3>
                    {signal.markets.map((market) => (
                        <div key={market.id} className="market-detail-card">
                            <div className="market-question">{market.question}</div>
                            <div className="market-prices">
                                <div className="price">
                                    <span className="label">YES</span>
                                    <span className="value yes">{formatPrice(market.yesPrice)}</span>
                                </div>
                                <div className="price">
                                    <span className="label">NO</span>
                                    <span className="value no">{formatPrice(market.noPrice)}</span>
                                </div>
                            </div>
                            <div className="market-volume">
                                Vol 24h: ${(market.volume24h / 1000).toFixed(0)}k
                            </div>
                        </div>
                    ))}
                </section>

                {/* Explanation */}
                <section className="explanation-section">
                    <h3>Why is this a signal?</h3>
                    <p>{signal.explanation}</p>
                    <div className="expected-vs-actual">
                        <div className="row">
                            <span className="label">Expected</span>
                            <span className="value">{signal.expectedRelationship}</span>
                        </div>
                        <div className="row">
                            <span className="label">Actual Value</span>
                            <span className="value">{formatPrice(signal.actualValue)}</span>
                        </div>
                        <div className="row highlight">
                            <span className="label">Divergence</span>
                            <span className="value">{(signal.divergenceAmount * 100).toFixed(1)}¢</span>
                        </div>
                    </div>
                </section>

                {/* Trade Form */}
                <section className="trade-form">
                    <h3>Execute Trade</h3>
                    <div className={`trade-action ${signal.recommendedAction.toLowerCase()}`}>
                        {signal.recommendedAction}
                    </div>

                    <div className="size-input">
                        <label>Trade Size ($)</label>
                        <input
                            type="number"
                            value={tradeSize}
                            onChange={(e) => setTradeSize(+e.target.value)}
                            min={10}
                            max={signal.maxExecutableSize}
                        />
                        <span className="max-size">Max: ${signal.maxExecutableSize.toFixed(0)}</span>
                    </div>

                    <div className="trade-preview">
                        <div className="preview-row">
                            <span>Estimated Fill</span>
                            <span>{formatPrice(signal.estimatedFillPrice)}</span>
                        </div>
                        <div className="preview-row">
                            <span>Estimated Fees</span>
                            <span>${signal.estimatedFees.toFixed(2)}</span>
                        </div>
                        <div className="preview-row profit">
                            <span>Expected Profit</span>
                            <span>${signal.expectedProfit.toFixed(2)}</span>
                        </div>
                    </div>

                    <button className="execute-button" onClick={handleTrade}>
                        Execute Trade
                    </button>
                </section>
            </div>
        </div>
    );
};

export default SignalDetailModal;
