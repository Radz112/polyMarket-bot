/**
 * Hook for real-time correlation updates via WebSocket
 */
import { useState, useEffect, useCallback } from 'react';
import type { Correlation, WebSocketMessage } from '../types';
import { mockCorrelations } from '../data/mockData';

const WS_URL = 'ws://localhost:8000/ws/signals';

export const useCorrelationUpdates = () => {
    const [correlations, setCorrelations] = useState<Correlation[]>([]);
    const [connected, setConnected] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        // Initial load - use mock data for now
        setCorrelations(mockCorrelations);

        // Try WebSocket connection
        let ws: WebSocket | null = null;
        let reconnectTimer: number;

        const connect = () => {
            try {
                ws = new WebSocket(WS_URL);

                ws.onopen = () => {
                    setConnected(true);
                    setError(null);
                    console.log('WebSocket connected');
                };

                ws.onmessage = (event) => {
                    const data: WebSocketMessage = JSON.parse(event.data);

                    if (data.type === 'divergence_update' && data.correlationId) {
                        setCorrelations(prev =>
                            prev.map(c =>
                                c.id === data.correlationId
                                    ? { ...c, ...data.updates }
                                    : c
                            )
                        );
                    } else if (data.type === 'new_correlation' && data.correlation) {
                        setCorrelations(prev => [data.correlation!, ...prev]);
                    } else if (data.type === 'correlation_removed' && data.correlationId) {
                        setCorrelations(prev =>
                            prev.filter(c => c.id !== data.correlationId)
                        );
                    }
                };

                ws.onerror = () => {
                    setError('WebSocket error');
                    setConnected(false);
                };

                ws.onclose = () => {
                    setConnected(false);
                    // Reconnect after 5 seconds
                    reconnectTimer = window.setTimeout(connect, 5000);
                };
            } catch (err) {
                setError('Failed to connect');
                setConnected(false);
            }
        };

        connect();

        return () => {
            if (ws) ws.close();
            if (reconnectTimer) clearTimeout(reconnectTimer);
        };
    }, []);

    const refresh = useCallback(() => {
        // Refresh from API
        setCorrelations(mockCorrelations);
    }, []);

    return { correlations, connected, error, refresh };
};

export default useCorrelationUpdates;
