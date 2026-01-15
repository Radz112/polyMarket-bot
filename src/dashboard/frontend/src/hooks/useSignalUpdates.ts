/**
 * Hook for real-time signal updates via WebSocket
 */
import { useState, useEffect, useCallback } from 'react';
import type { ExtendedSignal, WebSocketMessage } from '../types';
import { mockSignals } from '../data/mockSignals';

const WS_URL = 'ws://localhost:8000/ws/signals';

export const useSignalUpdates = () => {
    const [signals, setSignals] = useState<ExtendedSignal[]>([]);
    const [connected, setConnected] = useState(false);

    useEffect(() => {
        // Initial load with mock data
        setSignals(mockSignals);

        // WebSocket connection
        let ws: WebSocket | null = null;
        let reconnectTimer: number;

        const connect = () => {
            try {
                ws = new WebSocket(WS_URL);

                ws.onopen = () => {
                    setConnected(true);
                    console.log('Signal WebSocket connected');
                };

                ws.onmessage = (event) => {
                    const data: WebSocketMessage = JSON.parse(event.data);

                    if (data.type === 'new_signal' && data.signal) {
                        setSignals(prev => [data.signal as ExtendedSignal, ...prev]);
                    } else if (data.type === 'signal_update' && data.signalId) {
                        setSignals(prev =>
                            prev.map(s =>
                                s.id === data.signalId ? { ...s, ...data.signal } : s
                            )
                        );
                    } else if (data.type === 'signal_expired' && data.signalId) {
                        setSignals(prev =>
                            prev.map(s =>
                                s.id === data.signalId ? { ...s, status: 'expired' as const } : s
                            )
                        );
                    }
                };

                ws.onclose = () => {
                    setConnected(false);
                    reconnectTimer = window.setTimeout(connect, 5000);
                };
            } catch (err) {
                setConnected(false);
            }
        };

        connect();

        return () => {
            if (ws) ws.close();
            if (reconnectTimer) clearTimeout(reconnectTimer);
        };
    }, []);

    const dismissSignal = useCallback(async (signalId: string) => {
        try {
            await fetch(`http://localhost:8000/api/signals/${signalId}/dismiss`, {
                method: 'POST',
            });
            setSignals(prev => prev.filter(s => s.id !== signalId));
        } catch (err) {
            console.error('Error dismissing signal:', err);
        }
    }, []);

    const tradeSignal = useCallback(async (signalId: string, size: number) => {
        try {
            const response = await fetch(`http://localhost:8000/api/signals/${signalId}/trade`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ size }),
            });

            const result = await response.json();

            if (result.success) {
                setSignals(prev =>
                    prev.map(s =>
                        s.id === signalId ? { ...s, status: 'traded' as const } : s
                    )
                );
                return { success: true };
            } else {
                return { success: false, error: result.detail || result.error || 'Trade failed' };
            }
        } catch (err) {
            console.error('Error trading signal:', err);
            return { success: false, error: 'Connection error' };
        }
    }, []);

    return { signals, connected, dismissSignal, tradeSignal };
};

export default useSignalUpdates;
