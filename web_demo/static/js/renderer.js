class MotionRenderer {
    constructor(canvas, container) {
        this.canvas = canvas;
        this.container = container;
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xffffff);
        this.camera = new THREE.PerspectiveCamera(
            60,
            container.clientWidth / container.clientHeight,
            0.1,
            1000,
        );
        this.camera.position.set(3, 1.5, 3);
        this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.shadowMap.enabled = true;
        this.controls = new THREE.OrbitControls(this.camera, canvas);
        this.controls.target.set(0, 1, 0);
        this.controls.enableDamping = true;
        this.currentRoot = new THREE.Vector3(0, 1, 0);
        this.lastUserInteraction = 0;

        this.scene.add(new THREE.AmbientLight(0xffffff, 0.7));
        const key = new THREE.DirectionalLight(0xffffff, 0.8);
        key.position.set(5, 8, 3);
        key.castShadow = true;
        this.scene.add(key);
        const ground = new THREE.Mesh(
            new THREE.PlaneGeometry(1000, 1000),
            new THREE.ShadowMaterial({ opacity: 0.15 }),
        );
        ground.rotation.x = -Math.PI / 2;
        ground.receiveShadow = true;
        this.scene.add(ground);
        const grid = new THREE.GridHelper(1000, 1000, 0xdddddd, 0xeeeeee);
        grid.position.y = 0.01;
        this.scene.add(grid);

        this.skeleton = new Skeleton3D(this.scene);
        this.activeRoute = this.createLine(0x24a148, 0.9);
        this.draftRoute = this.createLine(0xff3b30, 0.7);
        this.raycaster = new THREE.Raycaster();
        this.pointer = new THREE.Vector2();
        this.groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);

        this.controls.addEventListener('start', () => { this.lastUserInteraction = Date.now(); });
        window.addEventListener('resize', () => this.resize());
        this.animate();
    }

    createLine(color, opacity) {
        const line = new THREE.Line(
            new THREE.BufferGeometry(),
            new THREE.LineBasicMaterial({ color, transparent: true, opacity }),
        );
        line.frustumCulled = false;
        this.scene.add(line);
        return line;
    }

    setLine(line, pointsXZ, y = 0.06) {
        const points = (pointsXZ || []).map(
            point => new THREE.Vector3(Number(point[0]), y, Number(point[1])),
        );
        line.geometry.dispose();
        line.geometry = new THREE.BufferGeometry().setFromPoints(points);
        line.visible = points.length > 1;
    }

    setActiveRoute(pointsXZ) { this.setLine(this.activeRoute, pointsXZ, 0.07); }
    setDraftRoute(pointsXZ) { this.setLine(this.draftRoute, pointsXZ, 0.08); }

    groundPoint(event) {
        const rect = this.canvas.getBoundingClientRect();
        this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        this.raycaster.setFromCamera(this.pointer, this.camera);
        const point = new THREE.Vector3();
        return this.raycaster.ray.intersectPlane(this.groundPlane, point)
            ? [point.x, point.z]
            : null;
    }

    renderFrame(frame) {
        if (!frame || !Array.isArray(frame.joints)) return;
        this.skeleton.updatePose(frame.joints);
        const root = frame.root_motion || frame.joints[0];
        this.currentRoot.set(Number(root[0]), Number(root[1]), Number(root[2]));
    }

    clearMotion() {
        this.skeleton.clearTrail();
        this.setActiveRoute([]);
        this.setDraftRoute([]);
    }

    resize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        if (Date.now() - this.lastUserInteraction > 2000) {
            const offset = this.camera.position.clone().sub(this.controls.target);
            const target = this.currentRoot.clone();
            target.y = 1.0;
            this.controls.target.lerp(target, 0.15);
            this.camera.position.lerp(target.clone().add(offset), 0.15);
        }
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}

window.MotionRenderer = MotionRenderer;
