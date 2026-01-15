/**
 * Trade History Table Component
 */
import React, { useState } from 'react';
import type { Trade } from '../../types/performance';
import { mockTrades } from '../../data/mockPerformance';
import { formatPrice } from '../shared';
import './TradeHistoryTable.css';

export const TradeHistoryTable: React.FC = () => {
    const [trades] = useState<Trade[]>(mockTrades);
    const [page, setPage] = useState(1);
    const perPage = 10;

    const formatDateTime = (date: string) => {
        const d = new Date(date);
        return d.toLocaleString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
        });
    };

    const totalPages = Math.ceil(trades.length / perPage);
    const displayedTrades = trades.slice((page - 1) * perPage, page * perPage);

    return (
        <div className="trade-history">
            <h3>Trade History</h3>

            <div className="table-container">
                <table className="trades-table">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Market</th>
                            <th>Side</th>
                            <th>Action</th>
                            <th>Size</th>
                            <th>Price</th>
                            <th>P&L</th>
                            <th>Signal</th>
                        </tr>
                    </thead>
                    <tbody>
                        {displayedTrades.map((trade) => (
                            <tr key={trade.id}>
                                <td>{formatDateTime(trade.timestamp)}</td>
                                <td className="market-name">{trade.marketName}</td>
                                <td>
                                    <span className={`side-badge ${trade.side.toLowerCase()}`}>
                                        {trade.side}
                                    </span>
                                </td>
                                <td>
                                    <span className={`action-badge ${trade.action.toLowerCase()}`}>
                                        {trade.action}
                                    </span>
                                </td>
                                <td>${trade.size.toFixed(2)}</td>
                                <td>{formatPrice(trade.price)}</td>
                                <td className={trade.realizedPnl !== null
                                    ? (trade.realizedPnl >= 0 ? 'positive' : 'negative')
                                    : ''
                                }>
                                    {trade.realizedPnl !== null
                                        ? `${trade.realizedPnl >= 0 ? '+' : ''}$${trade.realizedPnl.toFixed(2)}`
                                        : '—'}
                                </td>
                                <td>
                                    {trade.signalScore && (
                                        <span className="signal-score">{trade.signalScore}</span>
                                    )}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {totalPages > 1 && (
                <div className="pagination">
                    <button
                        disabled={page === 1}
                        onClick={() => setPage(p => p - 1)}
                    >
                        ← Prev
                    </button>
                    <span className="page-info">
                        Page {page} of {totalPages}
                    </span>
                    <button
                        disabled={page === totalPages}
                        onClick={() => setPage(p => p + 1)}
                    >
                        Next →
                    </button>
                </div>
            )}
        </div>
    );
};

export default TradeHistoryTable;
