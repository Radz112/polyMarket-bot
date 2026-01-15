/**
 * Performance by Category Component
 */
import React from 'react';
import type { CategoryPerformance } from '../../types/performance';
import './PerformanceByCategory.css';

interface Props {
    data: CategoryPerformance[];
}

export const PerformanceByCategory: React.FC<Props> = ({ data }) => {
    const maxPnl = Math.max(...data.map(d => Math.abs(d.netPnl)));

    return (
        <div className="performance-by-category">
            <h3>Performance by Category</h3>

            {/* Bar Chart */}
            <div className="category-bars">
                {data.map((cat) => (
                    <div key={cat.category} className="category-bar-row">
                        <span className="category-name">{cat.category}</span>
                        <div className="bar-container">
                            <div
                                className={`bar ${cat.netPnl >= 0 ? 'positive' : 'negative'}`}
                                style={{ width: `${(Math.abs(cat.netPnl) / maxPnl) * 100}%` }}
                            />
                        </div>
                        <span className={`pnl-value ${cat.netPnl >= 0 ? 'positive' : 'negative'}`}>
                            {cat.netPnl >= 0 ? '+' : ''}${cat.netPnl.toFixed(0)}
                        </span>
                    </div>
                ))}
            </div>

            {/* Table */}
            <div className="category-table-container">
                <table className="category-table">
                    <thead>
                        <tr>
                            <th>Category</th>
                            <th>Trades</th>
                            <th>Win Rate</th>
                            <th>Net P&L</th>
                            <th>Avg Trade</th>
                        </tr>
                    </thead>
                    <tbody>
                        {data.map((cat) => (
                            <tr key={cat.category}>
                                <td>{cat.category}</td>
                                <td>{cat.trades}</td>
                                <td>{(cat.winRate * 100).toFixed(0)}%</td>
                                <td className={cat.netPnl >= 0 ? 'positive' : 'negative'}>
                                    {cat.netPnl >= 0 ? '+' : ''}${cat.netPnl.toFixed(2)}
                                </td>
                                <td className={cat.avgTrade >= 0 ? 'positive' : 'negative'}>
                                    ${cat.avgTrade.toFixed(2)}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default PerformanceByCategory;
