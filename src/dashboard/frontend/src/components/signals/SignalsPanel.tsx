/**
 * Signals Panel - Main Component
 */
import React, { useState } from 'react';
import type { ExtendedSignal } from '../../types';
import { SignalCard } from './SignalCard';
import { SignalDetailModal } from './SignalDetailModal';
import { useSignalUpdates } from '../../hooks/useSignalUpdates';
import { EmptyState } from '../shared';
import './SignalsPanel.css';

export const SignalsPanel: React.FC = () => {
    const { signals, connected, dismissSignal, tradeSignal } = useSignalUpdates();
    const [activeTab, setActiveTab] = useState<'active' | 'history'>('active');
    const [selectedSignal, setSelectedSignal] = useState<ExtendedSignal | null>(null);

    const activeSignals = signals.filter((s) => s.status === 'active');
    const historySignals = signals.filter((s) => s.status !== 'active');

    const handleTrade = async (signal: ExtendedSignal, size?: number) => {
        const result = await tradeSignal(signal.id, size || signal.recommendedSize);
        if (result.success) {
            setSelectedSignal(null);
        }
    };

    const handleDismiss = (signal: ExtendedSignal) => {
        dismissSignal(signal.id);
    };

    return (
        <div className="signals-panel">
            <header className="signals-header">
                <div className="header-left">
                    <h2>Signals</h2>
                    <div className={`connection-dot ${connected ? 'connected' : ''}`} />
                </div>
                <div className="tabs">
                    <button
                        className={activeTab === 'active' ? 'active' : ''}
                        onClick={() => setActiveTab('active')}
                    >
                        Active ({activeSignals.length})
                    </button>
                    <button
                        className={activeTab === 'history' ? 'active' : ''}
                        onClick={() => setActiveTab('history')}
                    >
                        History
                    </button>
                </div>
            </header>

            <div className="signals-list">
                {activeTab === 'active' ? (
                    activeSignals.length > 0 ? (
                        activeSignals.map((signal) => (
                            <SignalCard
                                key={signal.id}
                                signal={signal}
                                onSelect={() => setSelectedSignal(signal)}
                                onTrade={handleTrade}
                                onDismiss={handleDismiss}
                            />
                        ))
                    ) : (
                        <EmptyState message="No active signals" />
                    )
                ) : historySignals.length > 0 ? (
                    historySignals.map((signal) => (
                        <SignalCard
                            key={signal.id}
                            signal={signal}
                            onSelect={() => setSelectedSignal(signal)}
                            onTrade={handleTrade}
                            onDismiss={handleDismiss}
                        />
                    ))
                ) : (
                    <EmptyState message="No signal history" />
                )}
            </div>

            {selectedSignal && (
                <SignalDetailModal
                    signal={selectedSignal}
                    onClose={() => setSelectedSignal(null)}
                    onTrade={handleTrade}
                />
            )}
        </div>
    );
};

export default SignalsPanel;
