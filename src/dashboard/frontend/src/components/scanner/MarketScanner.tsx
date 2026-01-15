/**
 * Market Scanner - Main Component
 */
import React, { useState, useMemo } from 'react';
import type { Correlation, ScannerFilters, SortOption } from '../../types';
import { ScannerFiltersBar } from './ScannerFilters';
import { CorrelationTable } from './CorrelationTable';
import { CorrelationDetail } from './CorrelationDetail';
import { useCorrelationUpdates } from '../../hooks/useCorrelationUpdates';
import { LoadingSpinner } from '../shared';
import './MarketScanner.css';

interface Props {
    refreshInterval?: number;
}

const defaultFilters: ScannerFilters = {
    category: null,
    correlationType: null,
    minDivergence: 0,
    minConfidence: 0,
    showOnlyOpportunities: false,
    verifiedOnly: false,
};

export const MarketScanner: React.FC<Props> = ({ refreshInterval = 5000 }) => {
    const { correlations, connected, error, refresh } = useCorrelationUpdates();
    const [filters, setFilters] = useState<ScannerFilters>(defaultFilters);
    const [sortBy, setSortBy] = useState<SortOption>('divergence');
    const [selectedPair, setSelectedPair] = useState<Correlation | null>(null);

    // Filter correlations
    const filteredCorrelations = useMemo(() => {
        return correlations.filter((corr) => {
            if (filters.category &&
                corr.marketA.category.toLowerCase() !== filters.category.toLowerCase()) {
                return false;
            }
            if (filters.correlationType && corr.type !== filters.correlationType) {
                return false;
            }
            if (Math.abs(corr.divergence) * 100 < filters.minDivergence) {
                return false;
            }
            if (corr.confidence < filters.minConfidence) {
                return false;
            }
            if (filters.showOnlyOpportunities && !corr.hasSignal) {
                return false;
            }
            if (filters.verifiedOnly && !corr.verified) {
                return false;
            }
            return true;
        });
    }, [correlations, filters]);

    // Sort correlations
    const sortedCorrelations = useMemo(() => {
        return [...filteredCorrelations].sort((a, b) => {
            switch (sortBy) {
                case 'marketA':
                    return a.marketA.question.localeCompare(b.marketA.question);
                case 'marketB':
                    return a.marketB.question.localeCompare(b.marketB.question);
                case 'type':
                    return a.type.localeCompare(b.type);
                case 'divergence':
                    return Math.abs(b.divergence) - Math.abs(a.divergence);
                case 'confidence':
                    return b.confidence - a.confidence;
                case 'score':
                    return (b.signalScore || 0) - (a.signalScore || 0);
                default:
                    return 0;
            }
        });
    }, [filteredCorrelations, sortBy]);

    return (
        <div className="market-scanner">
            <header className="scanner-header">
                <div className="header-left">
                    <h1>Market Scanner</h1>
                    <span className="correlation-count">
                        {filteredCorrelations.length} correlations
                    </span>
                </div>
                <div className="header-right">
                    <div className={`connection-status ${connected ? 'connected' : 'disconnected'}`}>
                        <span className="status-dot"></span>
                        {connected ? 'Live' : 'Offline'}
                    </div>
                    <button className="refresh-btn" onClick={refresh}>
                        ↻ Refresh
                    </button>
                </div>
            </header>

            <ScannerFiltersBar filters={filters} onChange={setFilters} />

            {correlations.length === 0 ? (
                <LoadingSpinner />
            ) : (
                <CorrelationTable
                    correlations={sortedCorrelations}
                    sortBy={sortBy}
                    onSort={setSortBy}
                    onSelect={setSelectedPair}
                />
            )}

            {selectedPair && (
                <CorrelationDetail
                    correlation={selectedPair}
                    onClose={() => setSelectedPair(null)}
                />
            )}
        </div>
    );
};

export default MarketScanner;
