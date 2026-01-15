/**
 * Time Range Selector Component
 */
import React from 'react';
import type { TimeRange } from '../../types/performance';
import './TimeRangeSelector.css';

interface Props {
    value: TimeRange;
    onChange: (range: TimeRange) => void;
}

const ranges: { value: TimeRange; label: string }[] = [
    { value: '1d', label: '1D' },
    { value: '7d', label: '7D' },
    { value: '30d', label: '30D' },
    { value: '90d', label: '90D' },
    { value: '1y', label: '1Y' },
    { value: 'all', label: 'All' },
];

export const TimeRangeSelector: React.FC<Props> = ({ value, onChange }) => {
    return (
        <div className="time-range-selector">
            {ranges.map((range) => (
                <button
                    key={range.value}
                    className={value === range.value ? 'active' : ''}
                    onClick={() => onChange(range.value)}
                >
                    {range.label}
                </button>
            ))}
        </div>
    );
};

export default TimeRangeSelector;
