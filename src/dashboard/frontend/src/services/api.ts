/**
 * API service for Dashboard
 */
const API_BASE = 'http://localhost:8000';
const WS_BASE = 'ws://localhost:8000';

export const api = {
    // Markets
    async getMarkets(category?: string) {
        const params = new URLSearchParams();
        if (category) params.set('category', category);
        const res = await fetch(`${API_BASE}/api/markets?${params}`);
        return res.json();
    },

    async getMarket(id: string) {
        const res = await fetch(`${API_BASE}/api/markets/${id}`);
        return res.json();
    },

    // Correlations
    async getCorrelations(filters?: { type?: string; minConfidence?: number }) {
        const params = new URLSearchParams();
        if (filters?.type) params.set('correlation_type', filters.type);
        if (filters?.minConfidence) params.set('min_confidence', String(filters.minConfidence));
        const res = await fetch(`${API_BASE}/api/correlations?${params}`);
        return res.json();
    },

    // Signals
    async getSignals(status = 'active') {
        const res = await fetch(`${API_BASE}/api/signals?status=${status}`);
        return res.json();
    },

    async tradeSignal(signalId: string, size?: number) {
        const res = await fetch(`${API_BASE}/api/signals/${signalId}/trade`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ size }),
        });
        return res.json();
    },

    // Portfolio
    async getPortfolio() {
        const res = await fetch(`${API_BASE}/api/portfolio`);
        return res.json();
    },

    // Health
    async getHealth() {
        const res = await fetch(`${API_BASE}/api/health`);
        return res.json();
    },
};

export const createWebSocket = (channel: string) => {
    return new WebSocket(`${WS_BASE}/ws/${channel}`);
};

export default api;
