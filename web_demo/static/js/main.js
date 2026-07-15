class FloodcontrolMotionApp {
    constructor() {
        this.api = new FloodcontrolApiClient();
        this.sessionId = null;
        this.sessionEpoch = null;
        this.frames = [];
        this.fetchingChunk = false;
        this.paused = false;
        this.pendingForceStart = false;
        this.targetFps = 20;
        this.frameInterval = 1000 / this.targetFps;
        this.lastFrameTime = performance.now();
        this.frameCount = 0;
        this.fpsFrames = 0;
        this.fpsTimestamp = performance.now();
        this.bindElements();
        this.renderer = new MotionRenderer(
            document.getElementById('renderCanvas'),
            document.getElementById('canvas-container'),
        );
        this.routeEditor = new RouteEditor({
            canvas: document.getElementById('renderCanvas'),
            textarea: this.routePoints,
            renderer: this.renderer,
            onCommit: (points, reference, source) => this.sendRoute(points, reference, source),
        });
        this.bindActions();
        this.beforeUnload();
        this.playbackLoop();
        this.statusLoop();
    }

    bindElements() {
        const byId = id => document.getElementById(id);
        this.motionText = byId('motionText');
        this.seed = byId('seed');
        this.cfgMode = byId('cfgMode');
        this.cfgText = byId('cfgText');
        this.cfgConstraint = byId('cfgConstraint');
        this.cfgJoint = byId('cfgJoint');
        this.startBtn = byId('startBtn');
        this.updateTextBtn = byId('updateTextBtn');
        this.updateGuidanceBtn = byId('updateGuidanceBtn');
        this.pauseBtn = byId('pauseBtn');
        this.resetBtn = byId('resetBtn');
        this.routePoints = byId('routePoints');
        this.routeReference = byId('routeReference');
        this.routeEndBehavior = byId('routeEndBehavior');
        this.routeDuration = byId('routeDuration');
        this.updateRouteBtn = byId('updateRouteBtn');
        this.clearRouteBtn = byId('clearRouteBtn');
        this.statusEl = byId('status');
        this.bufferSize = byId('bufferSize');
        this.commitIndex = byId('commitIndex');
        this.windowState = byId('windowState');
        this.fpsEl = byId('fps');
        this.frameCountEl = byId('frameCount');
        this.conflictWarning = byId('conflictWarning');
        this.forceTakeoverBtn = byId('forceTakeoverBtn');
        this.cancelTakeoverBtn = byId('cancelTakeoverBtn');
    }

    bindActions() {
        this.startBtn.addEventListener('click', () => this.start(false));
        this.updateTextBtn.addEventListener('click', () => this.updateText());
        this.updateGuidanceBtn.addEventListener('click', () => this.updateGuidance());
        this.pauseBtn.addEventListener('click', () => this.togglePause());
        this.resetBtn.addEventListener('click', () => this.reset());
        this.updateRouteBtn.addEventListener('click', () => {
            try {
                this.sendRoute(this.routeEditor.parseTextarea(), this.routeReference.value, 'manual');
            } catch (error) {
                this.showError(error);
            }
        });
        this.clearRouteBtn.addEventListener('click', () => this.clearRoute());
        this.forceTakeoverBtn.addEventListener('click', () => {
            this.conflictWarning.hidden = true;
            this.start(true);
        });
        this.cancelTakeoverBtn.addEventListener('click', () => {
            this.conflictWarning.hidden = true;
        });
    }

    guidance() {
        const values = {
            mode: this.cfgMode.value,
            scale_text: Number(this.cfgText.value),
            scale_constraint: Number(this.cfgConstraint.value),
            scale_joint: Number(this.cfgJoint.value),
        };
        if (![values.scale_text, values.scale_constraint, values.scale_joint].every(Number.isFinite)) {
            throw new Error('CFG scales must be finite numbers');
        }
        return values;
    }

    async start(force) {
        this.setStatus('Starting…');
        try {
            let initialRoute = null;
            if (this.routePoints.value.trim()) {
                initialRoute = {
                    points_xz: this.routeEditor.parseTextarea(),
                    duration_seconds: Number(this.routeDuration.value),
                    reference: this.routeReference.value,
                    end_behavior: this.routeEndBehavior.value,
                    source: 'manual',
                };
            }
            const payload = await this.api.start({
                text: this.motionText.value,
                seed: Number.parseInt(this.seed.value || '0', 10),
                force,
                guidance: this.guidance(),
                route: initialRoute,
            });
            const session = payload.session;
            this.sessionId = session.session_id;
            this.sessionEpoch = session.session_epoch;
            this.frames = [];
            this.frameCount = 0;
            this.paused = false;
            this.renderer.clearMotion();
            this.setControls(true);
            this.setStatus('Running');
            if (session.route) this.renderer.setActiveRoute(session.route.points_xz);
            this.pumpChunks();
        } catch (error) {
            if (error instanceof FloodcontrolApiError && error.status === 409 && error.payload.conflict) {
                this.conflictWarning.hidden = false;
                this.setStatus('Session conflict');
                return;
            }
            this.showError(error);
        }
    }

    async updateText() {
        if (!this.sessionId) return;
        try {
            await this.api.updateText(this.sessionId, this.motionText.value);
        } catch (error) { this.showError(error); }
    }

    async updateGuidance() {
        if (!this.sessionId) return;
        try {
            await this.api.updateGuidance(this.sessionId, this.guidance());
        } catch (error) { this.showError(error); }
    }

    async sendRoute(points, reference, source) {
        if (!this.sessionId) return;
        const duration = Number(this.routeDuration.value);
        if (!Number.isFinite(duration) || duration <= 0) {
            this.showError(new Error('Route duration must be positive'));
            return;
        }
        try {
            const payload = await this.api.updateRoute(this.sessionId, {
                points_xz: points,
                duration_seconds: duration,
                reference,
                end_behavior: this.routeEndBehavior.value,
                source,
            });
            this.renderer.setActiveRoute(payload.route.points_xz);
            this.renderer.setDraftRoute([]);
        } catch (error) { this.showError(error); }
    }

    async clearRoute() {
        if (!this.sessionId) return;
        try {
            await this.api.clearRoute(this.sessionId);
            this.routeEditor.clear();
            this.renderer.setActiveRoute([]);
        } catch (error) { this.showError(error); }
    }

    async togglePause() {
        if (!this.sessionId) return;
        try {
            if (this.paused) {
                await this.api.resume(this.sessionId);
                this.paused = false;
                this.pauseBtn.textContent = 'Pause';
                this.pumpChunks();
            } else {
                await this.api.pause(this.sessionId);
                this.paused = true;
                this.pauseBtn.textContent = 'Resume';
            }
        } catch (error) { this.showError(error); }
    }

    async reset() {
        const sessionId = this.sessionId;
        this.sessionId = null;
        if (sessionId) {
            try { await this.api.reset(sessionId); } catch (error) { console.warn(error); }
        }
        this.frames = [];
        this.sessionEpoch = null;
        this.paused = false;
        this.frameCount = 0;
        this.renderer.clearMotion();
        this.setControls(false);
        this.setStatus('Idle');
    }

    async pumpChunks() {
        if (!this.sessionId || this.fetchingChunk || this.paused || this.frames.length >= 12) return;
        const sessionId = this.sessionId;
        this.fetchingChunk = true;
        try {
            const payload = await this.api.nextChunk(sessionId, 500);
            if (sessionId !== this.sessionId) return;
            if (payload.status === 'success') {
                const chunk = payload.chunk;
                if (this.sessionEpoch === null) this.sessionEpoch = chunk.session_epoch;
                if (chunk.session_epoch === this.sessionEpoch) this.frames.push(...chunk.frames);
            }
        } catch (error) {
            if (sessionId === this.sessionId) this.showError(error);
        } finally {
            this.fetchingChunk = false;
            if (sessionId === this.sessionId && !this.paused) {
                setTimeout(() => this.pumpChunks(), this.frames.length < 8 ? 0 : 50);
            }
        }
    }

    playbackLoop(timestamp = performance.now()) {
        requestAnimationFrame(time => this.playbackLoop(time));
        if (this.frames.length && timestamp - this.lastFrameTime >= this.frameInterval) {
            const frame = this.frames.shift();
            this.renderer.renderFrame(frame);
            this.lastFrameTime = timestamp;
            this.frameCount += 1;
            this.fpsFrames += 1;
            this.frameCountEl.textContent = String(this.frameCount);
            this.pumpChunks();
        }
        if (timestamp - this.fpsTimestamp >= 1000) {
            this.fpsEl.textContent = String(this.fpsFrames);
            this.fpsFrames = 0;
            this.fpsTimestamp = timestamp;
        }
    }

    async statusLoop() {
        try {
            const payload = this.sessionId
                ? await this.api.sessionStatus(this.sessionId)
                : await this.api.status();
            const session = payload.session;
            if (session) {
                this.bufferSize.textContent = `${session.buffered_chunks} / ${session.buffer_capacity_chunks}`;
                this.commitIndex.textContent = String(session.commit_index);
                this.windowState.textContent = `${session.window_origin} · epoch ${session.window_epoch}`;
                if (session.error) this.setStatus(`Error: ${session.error}`);
                else this.setStatus(session.state);
                if (session.route) this.renderer.setActiveRoute(session.route.points_xz);
            } else if (!this.sessionId && payload.runtime_status) {
                this.setStatus(payload.runtime_status);
            }
        } catch (error) {
            if (error instanceof FloodcontrolApiError && error.status === 404) {
                this.showError(error);
            }
        }
        setTimeout(() => this.statusLoop(), 500);
    }

    setControls(active) {
        this.startBtn.disabled = active;
        this.updateTextBtn.disabled = !active;
        this.updateGuidanceBtn.disabled = !active;
        this.pauseBtn.disabled = !active;
        this.resetBtn.disabled = !active;
        this.updateRouteBtn.disabled = !active;
        this.clearRouteBtn.disabled = !active;
        if (!active) this.pauseBtn.textContent = 'Pause';
    }

    setStatus(text) { this.statusEl.textContent = text; }

    showError(error) {
        console.error(error);
        if (error instanceof FloodcontrolApiError && error.status === 404) {
            this.sessionId = null;
            this.sessionEpoch = null;
            this.frames = [];
            this.paused = false;
            this.setControls(false);
        }
        this.setStatus(error.message || String(error));
    }

    beforeUnload() {
        window.addEventListener('beforeunload', () => {
            if (!this.sessionId) return;
            const url = `/api/sessions/${encodeURIComponent(this.sessionId)}/reset`;
            navigator.sendBeacon(url, new Blob(['{}'], { type: 'application/json' }));
        });
    }
}

window.addEventListener('DOMContentLoaded', () => new FloodcontrolMotionApp());
