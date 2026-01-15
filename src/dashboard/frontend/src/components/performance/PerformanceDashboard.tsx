/**
 * Performance Dashboard - Main Component
 */
import React, { useState } from 'react';
import type { TimeRange } from '../../types/performance';
import { mockPerformanceData } from '../../data/mockPerformance';
import { MetricCard } from './MetricCard';
import { TimeRangeSelector } from './TimeRangeSelector';
import { EquityCurveChart } from './EquityCurveChart';
import { TradeStatistics } from './TradeStatistics';
import { DrawdownAnalysis } from './DrawdownAnalysis';
import { PerformanceByCategory } from './PerformanceByCategory';
import { TradeHistoryTable } from './TradeHistoryTable';
import './PerformanceDashboard.css';

export const PerformanceDashboard: React.FC = () => {
    const [timeRange, setTimeRange] = useState<TimeRange>('30d');
    const performance = mockPerformanceData;

    return (
        <div className="performance-dashboard">
            <header className="dashboard-header">
                <h2>Performance</h2>
                <TimeRangeSelector value={timeRange} onChange={setTimeRange} />
            </header>

            {/* Key Metrics */}
            <div className="metrics-row">
                <MetricCard
                    label="Total Return"
                    value={performance.totalReturn}
                    format="currency"
                    subValue={`${performance.totalReturnPct >= 0 ? '+' : ''}${performance.totalReturnPct.toFixed(1)}%`}
                    trend={performance.totalReturn >= 0 ? 'up' : 'down'}
                />
                <MetricCard
                    label="Win Rate"
                    value={performance.winRate}
                    format="percent"
                    benchmark={0.5}
                />
                <MetricCard
                    label="Profit Factor"
                    value={performance.profitFactor}
                    format="ratio"
                    benchmark={1.0}
                />
                <MetricCard
                    label="Sharpe Ratio"
                    value={performance.sharpeRatio}
                    format="ratio"
                    benchmark={1.0}
                />
            </div>

            {/* Equity Curve */}
            <section className="section">
                <h3>Equity Curve</h3>
                <EquityCurveChart data={performance.equityCurve} />
            </section>

            {/* Statistics Grid */}
            <div className="stats-grid">
                <TradeStatistics stats={performance.tradeStats} />
                <DrawdownAnalysis drawdowns={performance.drawdowns} />
            </div>

            {/* Performance by Category */}
            <section className="section">
                <PerformanceByCategory data={performance.byCategory} />
            </section>

            {/* Trade History */}
            <section className="section">
                <TradeHistoryTable />
            </section>
        </div>
    );
};

export default PerformanceDashboard;
