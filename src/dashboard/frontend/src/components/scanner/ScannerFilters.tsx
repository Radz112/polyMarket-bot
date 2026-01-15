/**
 * Scanner Filters Component
 */
import React from 'react';
import type { ScannerFilters, CorrelationType } from '../../types';
import { mockCategories } from '../../data/mockData';
import './ScannerFilters.css';

interface Props {
    filters: ScannerFilters;
    onChange: (filters: ScannerFilters) => void;
}

const correlationTypes: CorrelationType[] = [
    'equivalent',
    'mathematical',
    'inverse',
    'causal',
    'temporal',
];

export const ScannerFiltersBar: React.FC<Props> = ({ filters, onChange }) => {
    const updateFilter = <K extends keyof ScannerFilters>(
        key: K,
        value: ScannerFilters[K]
    ) => {
        onChange({ ...filters, [key]: value });
    };

    return (
        <div className="scanner-filters">
            <div className="filter-group">
                <label>Category</label>
                <select
                    value={filters.category || ''}
                    onChange={(e) => updateFilter('category', e.target.value || null)}
                >
                    <option value="">All Categories</option>
                    {mockCategories.map((cat) => (
                        <option key={cat} value={cat.toLowerCase()}>
                            {cat}
                        </option>
                    ))}
                </select>
            </div>

            <div className="filter-group">
                <label>Type</label>
                <select
                    value={filters.correlationType || ''}
                    onChange={(e) =>
                        updateFilter('correlationType', (e.target.value as CorrelationType) || null)
                    }
                >
                    <option value="">All Types</option>
                    {correlationTypes.map((type) => (
                        <option key={type} value={type}>
                            {type.charAt(0).toUpperCase() + type.slice(1)}
                        </option>
                    ))}
                </select>
            </div>

            <div className="filter-group slider">
                <label>
                    Min Divergence: <span>{filters.minDivergence}¢</span>
                </label>
                <input
                    type="range"
                    min={0}
                    max={10}
                    step={0.5}
                    value={filters.minDivergence}
                    onChange={(e) => updateFilter('minDivergence', +e.target.value)}
                />
            </div>

            <div className="filter-group slider">
                <label>
                    Min Confidence: <span>{(filters.minConfidence * 100).toFixed(0)}%</span>
                </label>
                <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={filters.minConfidence}
                    onChange={(e) => updateFilter('minConfidence', +e.target.value)}
                />
            </div>

            <div className="filter-group checkbox">
                <label>
                    <input
                        type="checkbox"
                        checked={filters.showOnlyOpportunities}
                        onChange={(e) => updateFilter('showOnlyOpportunities', e.target.checked)}
                    />
                    Opportunities only
                </label>
            </div>

            <div className="filter-group checkbox">
                <label>
                    <input
                        type="checkbox"
                        checked={filters.verifiedOnly}
                        onChange={(e) => updateFilter('verifiedOnly', e.target.checked)}
                    />
                    Verified only
                </label>
            </div>
        </div>
    );
};

export default ScannerFiltersBar;
