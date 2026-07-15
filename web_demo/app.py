"""Flask application factory for the Floodcontrol streaming demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask
from flask_cors import CORS

from web_demo.api.routes import register_routes
from web_demo.config import load_web_config
from web_demo.runtime.web_runtime import WebRuntime


def create_app(config_path: str | Path, *, runtime: WebRuntime | None = None) -> Flask:
    """Create a server whose HTTP layer depends only on ``WebRuntime``."""

    config = load_web_config(config_path)
    selected_runtime = runtime or WebRuntime(config)
    app = Flask(__name__)
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    app.extensions["floodcontrol_runtime"] = selected_runtime
    CORS(app)
    register_routes(app, selected_runtime)

    @app.after_request
    def add_no_cache_headers(response):
        response.headers["Cache-Control"] = (
            "no-store, no-cache, must-revalidate, max-age=0"
        )
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Floodcontrol real-time hybrid motion Web server"
    )
    parser.add_argument("--config", default="../configs/stream.yaml")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    app = create_app(args.config)
    try:
        app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
    finally:
        app.extensions["floodcontrol_runtime"].shutdown()


if __name__ == "__main__":
    main()


__all__ = ["create_app", "main"]
