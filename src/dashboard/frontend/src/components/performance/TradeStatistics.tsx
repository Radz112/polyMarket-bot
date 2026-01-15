/**
 * Trade Statistics Component
 */
import React from 'react';
import type { TradeStats } from '../../types/performance';
import './TradeStatistics.css';

interface Props {
    stats: TradeStats;
}

const formatDuration = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    if (hours < 24) return `${hours}h`;
    const days = Math.floor(hours / 24);
    return `${days}d ${hours % 24}h`;
};

export const TradeStatistics: React.FC<Props> = ({ stats }) => {
    return (
        <div className="trade-statistics">
            <h3>Trade Statistics</h3>

            <div className="stats-grid">
                <div className="stat-row">
                    <span className="label">Total Trades</span>
                    <span className="value">{stats.totalTrades}</span>
                </div>
                <div className="stat-row">
                    <span className="label">Winning Trades</span>
                    <span className="value positive">{stats.winningTrades}</span>
                </div>
                <div className="stat-row">
                    <span className="label">Losing Trades</span>
                    <span className="value negative">{stats.losingTrades}</span>
                </div>
                <div className="stat-row">
                    <span className="label">Win Rate</span>
                    <span className="value">{(stats.winRate * 100).toFixed(1)}%</span>
                </div>

                <div className="stat-divider" />

                <div className="stat-row">
                    <span className="label">Average Win</span>
                    <span className="value positive">${stats.averageWin.toFixed(2)}</span>
                </div>
                <div className="stat-row">
                    <span className="label">Average Loss</span>
                    <span className="value negative">${Math.abs(stats.averageLoss).toFixed(2)}</span>
                </div>
                <div className="stat-row">
                    <span className="label">Largest Win</span>
                    <span className="value positive">${stats.largestWin.toFixed(2)}</span>
                </div>
                <div className="stat-row">
                    <span className="label">Largest Loss</span>
                    <span className="value negative">${Math.abs(stats.largestLoss).toFixed(2)}</span>
                </div>

                <div className="stat-divider" />

                <div className="stat-row">
                    <span className="label">Avg Holding Period</span>
                    <span className="value">{formatDuration(stats.avgHoldingPeriod)}</span>
                </div>
                <div className="stat-row">
                    <span className="label">Profit Factor</span>
                    <span className={`value ${stats.profitFactor >= 1 ? 'positive' : 'negative'}`}>
                        {stats.profitFactor.toFixed(2)}
                    </span>
                </div>
                <div className="stat-row">
                    <span className="label">Expectancy</span>
                    <span className={`value ${stats.expectancy >= 0 ? 'positive' : 'negative'}`}>
                        ${stats.expectancy.toFixed(2)}
                    </span>
                </div>
            </div>

            <div className="streaks">
                <div className="streak-item">
                    <span className="streak-label">Current Streak</span>
                    <span className={`streak-value ${stats.currentStreak > 0 ? 'positive' : 'negative'}`}>
                        {stats.currentStreak > 0
                            ? `${stats.currentStreak} wins`
                            : `${Math.abs(stats.currentStreak)} losses`}
                    </span>
                </div>
                <div className="streak-item">
                    <span className="streak-label">Max Win Streak</span>
                    <span className="streak-value positive">{stats.maxWinStreak}</span>
                </div>
                <div className="streak-item">
                    <span className="streak-label">Max Loss Streak</span>
                    <span className="streak-value negative">{stats.maxLossStreak}</span>
                </div>
            </div>
        </div>
    );
};

export default TradeStatistics;
