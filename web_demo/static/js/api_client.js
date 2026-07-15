class FloodcontrolApiError extends Error {
    constructor(message, status, payload) {
        super(message);
        this.status = status;
        this.payload = payload || {};
    }
}

class FloodcontrolApiClient {
    async request(path, options = {}) {
        const response = await fetch(path, {
            headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
            ...options,
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new FloodcontrolApiError(
                payload.message || `HTTP ${response.status}`,
                response.status,
                payload,
            );
        }
        return payload;
    }

    status() {
        return this.request('/api/status');
    }

    start(payload) {
        return this.request('/api/sessions', {
            method: 'POST',
            body: JSON.stringify(payload),
        });
    }

    sessionStatus(sessionId) {
        return this.request(`/api/sessions/${encodeURIComponent(sessionId)}/status`);
    }

    updateText(sessionId, text) {
        return this.request(`/api/sessions/${encodeURIComponent(sessionId)}/text`, {
            method: 'POST',
            body: JSON.stringify({ text }),
        });
    }

    updateGuidance(sessionId, guidance) {
        return this.request(`/api/sessions/${encodeURIComponent(sessionId)}/guidance`, {
            method: 'PUT',
            body: JSON.stringify(guidance),
        });
    }

    updateRoute(sessionId, route) {
        return this.request(`/api/sessions/${encodeURIComponent(sessionId)}/route`, {
            method: 'PUT',
            body: JSON.stringify(route),
        });
    }

    clearRoute(sessionId) {
        return this.request(`/api/sessions/${encodeURIComponent(sessionId)}/route`, {
            method: 'DELETE',
        });
    }

    pause(sessionId) {
        return this.request(`/api/sessions/${encodeURIComponent(sessionId)}/pause`, { method: 'POST' });
    }

    resume(sessionId) {
        return this.request(`/api/sessions/${encodeURIComponent(sessionId)}/resume`, { method: 'POST' });
    }

    nextChunk(sessionId, waitMs = 500) {
        return this.request(
            `/api/sessions/${encodeURIComponent(sessionId)}/chunks/next?wait_ms=${waitMs}`,
        );
    }

    reset(sessionId) {
        return this.request(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
    }
}

window.FloodcontrolApiClient = FloodcontrolApiClient;
window.FloodcontrolApiError = FloodcontrolApiError;
