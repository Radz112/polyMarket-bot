/**
 * Metric Card Component
 */
import React from 'react';
import './MetricCard.css';

interface Props {
    label: string;
    value: number | undefined;
    format: 'currency' | 'percent' | 'ratio' | 'number';
    subValue?: string;
    trend?: 'up' | 'down' | 'neutral';
    benchmark?: number;
}

const formatValue = (value: number | undefined, format: string): string => {
    if (value === undefined) return '—';

    switch (format) {
        case 'currency':
            return `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
        case 'percent':
            return `${(value * 100).toFixed(1)}%`;
        case 'ratio':
            return value.toFixed(2);
        case 'number':
            return value.toLocaleString();
        default:
            return String(value);
    }
};

export const MetricCard: React.FC<Props> = ({
    label,
    value,
    format,
    subValue,
    trend,
    benchmark,
}) => {
    const trendClass = trend || (value !== undefined && benchmark !== undefined
        ? value >= benchmark ? 'up' : 'down'
        : 'neutral');

    return (
        <div className={`metric-card trend-${trendClass}`}>
            <div className="metric-label">{label}</div>
            <div className="metric-value">
                {formatValue(value, format)}
                {trend && (
                    <span className="trend-icon">
                        {trend === 'up' ? '↑' : trend === 'down' ? '↓' : ''}
                    </span>
                )}
            </div>
            {subValue && <div className="metric-sub">{subValue}</div>}
            {benchmark !== undefined && value !== undefined && (
                <div className="metric-benchmark">
                    vs benchmark: {formatValue(benchmark, format)}
                </div>
            )}
        </div>
    );
};

export default MetricCard;
