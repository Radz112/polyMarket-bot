/**
 * Drawdown Analysis Component
 */
import React from 'react';
import type { DrawdownPeriod } from '../../types/performance';
import './DrawdownAnalysis.css';

interface Props {
    drawdowns: DrawdownPeriod[];
}

const formatDate = (date: string) => {
    return new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
};

const formatDuration = (seconds: number): string => {
    const days = Math.floor(seconds / 86400);
    if (days > 0) return `${days}d`;
    const hours = Math.floor(seconds / 3600);
    return `${hours}h`;
};

export const DrawdownAnalysis: React.FC<Props> = ({ drawdowns }) => {
    const current = drawdowns[0];

    return (
        <div className="drawdown-analysis">
            <h3>Drawdown Analysis</h3>

            {/* Current Drawdown Meter */}
            <div className="current-drawdown">
                <div className="drawdown-header">
                    <span>Current Drawdown</span>
                    <span className="drawdown-pct">
                        {(current?.currentPct || 0).toFixed(1)}%
                    </span>
                </div>
                <div className="drawdown-meter">
                    <div
                        className="drawdown-fill"
                        style={{ width: `${Math.min(current?.currentPct || 0, 100)}%` }}
                    />
                </div>
            </div>

            {/* Max Drawdown */}
            <div className="max-drawdown">
                <div className="max-drawdown-header">
                    <span className="label">Maximum Drawdown</span>
                    <span className="value">{(current?.maxPct || 0).toFixed(1)}%</span>
                </div>
                {current?.maxDate && (
                    <span className="date">{formatDate(current.maxDate)}</span>
                )}
            </div>

            {/* Drawdown Periods Table */}
            {drawdowns.length > 0 && (
                <div className="drawdown-table-container">
                    <table className="drawdown-table">
                        <thead>
                            <tr>
                                <th>Start</th>
                                <th>End</th>
                                <th>Depth</th>
                                <th>Duration</th>
                                <th>Recovery</th>
                            </tr>
                        </thead>
                        <tbody>
                            {drawdowns.slice(0, 5).map((dd, i) => (
                                <tr key={i}>
                                    <td>{formatDate(dd.startDate)}</td>
                                    <td>{dd.endDate ? formatDate(dd.endDate) : 'Ongoing'}</td>
                                    <td className="negative">-{dd.depth.toFixed(1)}%</td>
                                    <td>{formatDuration(dd.duration)}</td>
                                    <td>{dd.recoveryDuration ? formatDuration(dd.recoveryDuration) : '—'}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
};

export default DrawdownAnalysis;
