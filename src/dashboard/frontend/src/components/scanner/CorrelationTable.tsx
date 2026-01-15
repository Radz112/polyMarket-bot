/**
 * Correlation Table Component
 */
import React from 'react';
import type { Correlation, SortOption } from '../../types';
import { CorrelationRow } from './CorrelationRow';
import { EmptyState } from '../shared';
import './CorrelationTable.css';

interface Props {
    correlations: Correlation[];
    sortBy: SortOption;
    onSort: (option: SortOption) => void;
    onSelect: (correlation: Correlation) => void;
}

const sortOptions: { key: SortOption; label: string }[] = [
    { key: 'marketA', label: 'Market A' },
    { key: 'marketB', label: 'Market B' },
    { key: 'type', label: 'Type' },
    { key: 'divergence', label: 'Divergence' },
    { key: 'confidence', label: 'Confidence' },
    { key: 'score', label: 'Score' },
];

export const CorrelationTable: React.FC<Props> = ({
    correlations,
    sortBy,
    onSort,
    onSelect,
}) => {
    if (correlations.length === 0) {
        return <EmptyState message="No correlations match your filters" />;
    }

    return (
        <div className="correlation-table-wrapper">
            <table className="correlation-table">
                <thead>
                    <tr>
                        <th
                            className={sortBy === 'marketA' ? 'sorted' : ''}
                            onClick={() => onSort('marketA')}
                        >
                            Market A
                            {sortBy === 'marketA' && <span className="sort-indicator">↓</span>}
                        </th>
                        <th
                            className={sortBy === 'marketB' ? 'sorted' : ''}
                            onClick={() => onSort('marketB')}
                        >
                            Market B
                            {sortBy === 'marketB' && <span className="sort-indicator">↓</span>}
                        </th>
                        <th
                            className={sortBy === 'type' ? 'sorted' : ''}
                            onClick={() => onSort('type')}
                        >
                            Type
                            {sortBy === 'type' && <span className="sort-indicator">↓</span>}
                        </th>
                        <th>Relationship</th>
                        <th
                            className={sortBy === 'divergence' ? 'sorted' : ''}
                            onClick={() => onSort('divergence')}
                        >
                            Divergence
                            {sortBy === 'divergence' && <span className="sort-indicator">↓</span>}
                        </th>
                        <th
                            className={sortBy === 'confidence' ? 'sorted' : ''}
                            onClick={() => onSort('confidence')}
                        >
                            Confidence
                            {sortBy === 'confidence' && <span className="sort-indicator">↓</span>}
                        </th>
                        <th
                            className={sortBy === 'score' ? 'sorted' : ''}
                            onClick={() => onSort('score')}
                        >
                            Score
                            {sortBy === 'score' && <span className="sort-indicator">↓</span>}
                        </th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {correlations.map((correlation) => (
                        <CorrelationRow
                            key={correlation.id}
                            correlation={correlation}
                            onClick={() => onSelect(correlation)}
                        />
                    ))}
                </tbody>
            </table>
        </div>
    );
};

export default CorrelationTable;
