"""Thin Flask routes over the authoritative ``WebRuntime`` facade."""

from __future__ import annotations

from flask import jsonify, render_template, request

from web_demo.runtime.web_runtime import (
    SessionConflictError,
    SessionNotFoundError,
    WebRuntime,
)

from .responses import error, success
from .schemas import (
    StartSessionRequest,
    UpdateGuidanceRequest,
    UpdateRouteRequest,
    UpdateTextRequest,
)


def register_routes(app, runtime: WebRuntime):
    """Register HTTP transport only; all mutable state remains in runtime."""

    @app.errorhandler(SessionConflictError)
    def handle_conflict(exc):
        return jsonify(
            error(
                str(exc),
                conflict=True,
                active_session_id=exc.active_session_id,
            )
        ), 409

    @app.errorhandler(SessionNotFoundError)
    def handle_missing_session(exc):
        return jsonify(error(str(exc))), 404

    @app.errorhandler(ValueError)
    @app.errorhandler(TypeError)
    def handle_bad_request(exc):
        return jsonify(error(str(exc))), 400

    @app.errorhandler(RuntimeError)
    def handle_runtime_failure(exc):
        message = str(exc)
        status = 503 if "BLOCKED_ON_" in message else 409
        return jsonify(error(message)), status

    @app.errorhandler(Exception)
    def handle_unexpected(exc):
        app.logger.exception("Unhandled Web runtime error", exc_info=exc)
        return jsonify(error("internal Web runtime error")), 500

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            web_defaults=runtime.config.template_defaults(),
        )

    @app.get("/api/status")
    def runtime_status():
        return jsonify(runtime.status())

    @app.post("/api/sessions")
    def start_session():
        values = StartSessionRequest.from_payload(request.get_json(silent=True))
        session = runtime.start_session(
            text=values.text,
            seed=values.seed,
            initial_world_xz=values.initial_world_xz,
            initial_yaw=values.initial_yaw,
            force=values.force,
            guidance=values.guidance,
            initial_route=(
                None
                if values.route is None
                else {
                    "points_xz": values.route.points_xz,
                    "duration_seconds": values.route.duration_seconds,
                    "reference": values.route.reference.value,
                    "end_behavior": values.route.end_behavior.value,
                    "source": values.route.source,
                }
            ),
        )
        return jsonify(success(session=session)), 201

    @app.get("/api/sessions/<session_id>/status")
    def session_status(session_id: str):
        return jsonify(runtime.status(session_id))

    @app.post("/api/sessions/<session_id>/text")
    def update_text(session_id: str):
        values = UpdateTextRequest.from_payload(request.get_json(silent=True))
        return jsonify(success(text=runtime.update_text(session_id, values.text)))

    @app.put("/api/sessions/<session_id>/route")
    def update_route(session_id: str):
        values = UpdateRouteRequest.from_payload(request.get_json(silent=True))
        route = runtime.update_route(
            session_id,
            points_xz=values.points_xz,
            duration_seconds=values.duration_seconds,
            reference=values.reference.value,
            end_behavior=values.end_behavior.value,
            source=values.source,
        )
        return jsonify(success(route=route))

    @app.delete("/api/sessions/<session_id>/route")
    def clear_route(session_id: str):
        return jsonify(success(route=runtime.clear_route(session_id)))

    @app.put("/api/sessions/<session_id>/guidance")
    def update_guidance(session_id: str):
        values = UpdateGuidanceRequest.from_payload(request.get_json(silent=True))
        return jsonify(
            success(
                guidance=runtime.update_guidance(session_id, values.guidance)
            )
        )

    @app.post("/api/sessions/<session_id>/pause")
    def pause(session_id: str):
        return jsonify(success(session=runtime.pause(session_id)))

    @app.post("/api/sessions/<session_id>/resume")
    def resume(session_id: str):
        return jsonify(success(session=runtime.resume(session_id)))

    @app.get("/api/sessions/<session_id>/chunks/next")
    def next_chunk(session_id: str):
        wait_ms = request.args.get("wait_ms", "500")
        try:
            wait_seconds = min(1.0, max(0.0, int(wait_ms) / 1000.0))
        except ValueError as exc:
            raise ValueError("wait_ms must be an integer") from exc
        chunk = runtime.pop_chunk(session_id, timeout=wait_seconds)
        if chunk is None:
            return jsonify({"status": "waiting"})
        return jsonify(success(chunk=chunk.to_payload()))

    def _reset(session_id: str):
        runtime.reset(session_id)
        return jsonify(success(message="session reset"))

    app.add_url_rule(
        "/api/sessions/<session_id>",
        endpoint="delete_session",
        view_func=_reset,
        methods=["DELETE"],
    )
    app.add_url_rule(
        "/api/sessions/<session_id>/reset",
        endpoint="reset_session_beacon",
        view_func=_reset,
        methods=["POST"],
    )
    return app


__all__ = ["register_routes"]
