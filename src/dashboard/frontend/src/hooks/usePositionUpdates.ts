/**
 * Hook for real-time position updates via WebSocket
 */
import { useState, useEffect, useCallback } from 'react';
import type { Position, WebSocketMessage } from '../types';
import { mockPositions } from '../data/mockPositions';

const WS_URL = 'ws://localhost:8000/ws/positions';

export const usePositionUpdates = () => {
    const [positions, setPositions] = useState<Position[]>([]);
    const [connected, setConnected] = useState(false);

    useEffect(() => {
        // Initial load with mock data
        setPositions(mockPositions);

        // WebSocket connection
        let ws: WebSocket | null = null;
        let reconnectTimer: number;

        const connect = () => {
            try {
                ws = new WebSocket(WS_URL);

                ws.onopen = () => {
                    setConnected(true);
                    console.log('Position WebSocket connected');
                };

                ws.onmessage = (event) => {
                    const data: WebSocketMessage = JSON.parse(event.data);

                    if (data.type === 'position_update' && data.position) {
                        setPositions(prev => {
                            const exists = prev.find(p => p.id === data.position!.id);
                            if (exists) {
                                return prev.map(p =>
                                    p.id === data.position!.id ? data.position! : p
                                );
                            }
                            return [data.position!, ...prev];
                        });
                    } else if (data.type === 'position_closed' && data.positionId) {
                        setPositions(prev =>
                            prev.filter(p => p.id !== data.positionId)
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

    const closePosition = useCallback(async (positionId: string) => {
        try {
            const response = await fetch(`http://localhost:8000/api/positions/${positionId}/close`, {
                method: 'POST',
            });
            const result = await response.json();
            if (result.success) {
                setPositions(prev => prev.filter(p => p.id !== positionId));
                return { success: true };
            }
            return { success: false, error: result.error || 'Failed to close' };
        } catch (err) {
            console.error('Error closing position:', err);
            return { success: false, error: 'Connection error' };
        }
    }, []);

    const reducePosition = useCallback(async (positionId: string, size: number) => {
        try {
            const response = await fetch(`http://localhost:8000/api/positions/${positionId}/reduce`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ size }),
            });
            const result = await response.json();
            if (result.success) {
                setPositions(prev =>
                    prev.map(p =>
                        p.id === positionId ? { ...p, size: p.size - size } : p
                    )
                );
                return { success: true };
            }
            return { success: false, error: result.error || 'Failed to reduce' };
        } catch (err) {
            console.error('Error reducing position:', err);
            return { success: false, error: 'Connection error' };
        }
    }, []);

    return { positions, connected, closePosition, reducePosition };
};

export default usePositionUpdates;
