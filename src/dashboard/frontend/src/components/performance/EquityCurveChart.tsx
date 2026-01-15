/**
 * Equity Curve Chart Component (SVG-based, no external deps)
 */
import React from 'react';
import type { EquityPoint } from '../../types/performance';
import './EquityCurveChart.css';

interface Props {
    data: EquityPoint[];
}

export const EquityCurveChart: React.FC<Props> = ({ data }) => {
    if (!data || data.length === 0) {
        return <div className="chart-empty">No data available</div>;
    }

    const width = 800;
    const height = 300;
    const padding = { top: 20, right: 60, bottom: 40, left: 70 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // Calculate scales
    const values = data.map((d) => d.value);
    const minValue = Math.min(...values) * 0.98;
    const maxValue = Math.max(...values) * 1.02;
    const valueRange = maxValue - minValue;

    const drawdowns = data.map((d) => d.drawdown);
    const maxDrawdown = Math.max(...drawdowns, 1);

    const xScale = (i: number) => padding.left + (i / (data.length - 1)) * chartWidth;
    const yScale = (v: number) => padding.top + chartHeight - ((v - minValue) / valueRange) * chartHeight;
    const ddScale = (d: number) => padding.top + (d / maxDrawdown) * (chartHeight * 0.3);

    // Generate path for equity curve
    const equityPath = data
        .map((d, i) => `${i === 0 ? 'M' : 'L'} ${xScale(i)} ${yScale(d.value)}`)
        .join(' ');

    // Generate area for equity curve
    const equityArea = `${equityPath} L ${xScale(data.length - 1)} ${yScale(minValue)} L ${xScale(0)} ${yScale(minValue)} Z`;

    // Generate path for drawdown
    const drawdownPath = data
        .map((d, i) => `${i === 0 ? 'M' : 'L'} ${xScale(i)} ${ddScale(d.drawdown)}`)
        .join(' ');

    // Format date for axis
    const formatDate = (date: string) => {
        const d = new Date(date);
        return `${d.getMonth() + 1}/${d.getDate()}`;
    };

    // Y-axis ticks
    const yTicks = Array.from({ length: 5 }, (_, i) => minValue + (valueRange * i) / 4);

    return (
        <div className="equity-curve-chart">
            <svg viewBox={`0 0 ${width} ${height}`}>
                {/* Grid lines */}
                <g className="grid">
                    {yTicks.map((tick, i) => (
                        <line
                            key={i}
                            x1={padding.left}
                            y1={yScale(tick)}
                            x2={width - padding.right}
                            y2={yScale(tick)}
                            stroke="var(--border)"
                            strokeDasharray="4 4"
                        />
                    ))}
                </g>

                {/* Equity area fill */}
                <path
                    d={equityArea}
                    fill="url(#equityGradient)"
                    opacity={0.2}
                />

                {/* Equity line */}
                <path
                    d={equityPath}
                    fill="none"
                    stroke="var(--accent)"
                    strokeWidth={2}
                />

                {/* Drawdown area */}
                <path
                    d={`${drawdownPath} L ${xScale(data.length - 1)} ${padding.top} L ${xScale(0)} ${padding.top} Z`}
                    fill="var(--red)"
                    opacity={0.1}
                />

                {/* Y-axis labels */}
                <g className="y-axis">
                    {yTicks.map((tick, i) => (
                        <text
                            key={i}
                            x={padding.left - 10}
                            y={yScale(tick)}
                            textAnchor="end"
                            dominantBaseline="middle"
                            fill="var(--text-muted)"
                            fontSize={11}
                        >
                            ${tick.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                        </text>
                    ))}
                </g>

                {/* X-axis labels */}
                <g className="x-axis">
                    {data.filter((_, i) => i % Math.ceil(data.length / 6) === 0).map((d, i, arr) => (
                        <text
                            key={i}
                            x={xScale(data.indexOf(d))}
                            y={height - 10}
                            textAnchor="middle"
                            fill="var(--text-muted)"
                            fontSize={11}
                        >
                            {formatDate(d.timestamp)}
                        </text>
                    ))}
                </g>

                {/* Data points */}
                {data.map((d, i) => (
                    <circle
                        key={i}
                        cx={xScale(i)}
                        cy={yScale(d.value)}
                        r={3}
                        fill="var(--accent)"
                        opacity={0}
                        className="data-point"
                    >
                        <title>
                            {formatDate(d.timestamp)}: ${d.value.toFixed(2)} (DD: {d.drawdown.toFixed(1)}%)
                        </title>
                    </circle>
                ))}

                {/* Gradient definition */}
                <defs>
                    <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="var(--accent)" />
                        <stop offset="100%" stopColor="var(--accent)" stopOpacity={0} />
                    </linearGradient>
                </defs>
            </svg>

            <div className="chart-legend">
                <div className="legend-item">
                    <span className="legend-color" style={{ background: 'var(--accent)' }}></span>
                    Portfolio Value
                </div>
                <div className="legend-item">
                    <span className="legend-color" style={{ background: 'var(--red)', opacity: 0.3 }}></span>
                    Drawdown
                </div>
            </div>
        </div>
    );
};

export default EquityCurveChart;
