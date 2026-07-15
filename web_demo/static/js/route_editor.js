class RouteEditor {
    constructor({ canvas, textarea, renderer, onCommit }) {
        this.canvas = canvas;
        this.textarea = textarea;
        this.renderer = renderer;
        this.onCommit = onCommit;
        this.points = [];
        this.drawing = false;
        canvas.addEventListener('pointerdown', event => this.pointerDown(event));
        canvas.addEventListener('pointermove', event => this.pointerMove(event));
        canvas.addEventListener('pointerup', event => this.pointerUp(event));
        canvas.addEventListener('pointercancel', event => this.pointerUp(event));
    }

    parseTextarea() {
        const lines = this.textarea.value.split(/\r?\n/).map(line => line.trim()).filter(Boolean);
        const points = lines.map((line, index) => {
            const values = line.split(/[\s,]+/).map(Number);
            if (values.length !== 2 || values.some(value => !Number.isFinite(value))) {
                throw new Error(`Route line ${index + 1} must contain finite x z values`);
            }
            return values;
        });
        if (!points.length) throw new Error('Route requires at least one XZ point');
        return points;
    }

    setTextarea(points) {
        this.textarea.value = points.map(point => `${point[0].toFixed(3)} ${point[1].toFixed(3)}`).join('\n');
    }

    pointerDown(event) {
        if (!event.shiftKey || event.button !== 0) return;
        const point = this.renderer.groundPoint(event);
        if (!point) return;
        event.preventDefault();
        this.drawing = true;
        this.points = [point];
        this.renderer.controls.enabled = false;
        this.renderer.setDraftRoute(this.points);
        this.canvas.setPointerCapture(event.pointerId);
    }

    pointerMove(event) {
        if (!this.drawing) return;
        const point = this.renderer.groundPoint(event);
        if (!point) return;
        const previous = this.points[this.points.length - 1];
        if (Math.hypot(point[0] - previous[0], point[1] - previous[1]) < 0.05) return;
        this.points.push(point);
        this.renderer.setDraftRoute(this.points);
        this.setTextarea(this.points);
    }

    pointerUp(event) {
        if (!this.drawing) return;
        this.drawing = false;
        this.renderer.controls.enabled = true;
        if (this.canvas.hasPointerCapture(event.pointerId)) {
            this.canvas.releasePointerCapture(event.pointerId);
        }
        if (this.points.length > 1) this.onCommit(this.points, 'world', 'canvas');
    }

    clear() {
        this.points = [];
        this.textarea.value = '';
        this.renderer.setDraftRoute([]);
    }
}

window.RouteEditor = RouteEditor;
