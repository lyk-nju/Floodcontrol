"""Flask server bootstrap for the real-time 3D motion generation demo."""

import threading
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask
from flask_cors import CORS
from web_demo.model_manager import WEB_MIGRATION_ERROR, get_model_manager
from web_demo.api.routes import register_routes
from web_demo.config import load_debug_preset_cfg, load_traj_mask_cfg
from web_demo.services.session_service import SessionService

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
CORS(app)

# Global model manager (lazy loaded)
model_manager = None
model_config_path = None  # Will be set once at startup
traj_mask_cfg = None
debug_preset_cfg = None
model_init_lock = threading.Lock()
session_service = SessionService(consumption_timeout=5.0)


def init_model():
    """Initialize model manager"""
    global model_manager, traj_mask_cfg
    if model_manager is None:
        with model_init_lock:
            if model_manager is None:
                if model_config_path is None:
                    raise RuntimeError("model_config_path not set. Server not properly initialized.")
                print(f"Initializing model manager with config: {model_config_path}")
                model_manager = get_model_manager(
                    config_path=model_config_path,
                    traj_mask_cfg=traj_mask_cfg,
                )
                print("Model manager ready!")
    return model_manager


def load_debug_preset_sample():
    """Fail explicitly while the root5/body265 Web bridge is unavailable."""
    cfg = debug_preset_cfg or {}
    if not bool(cfg.get("enabled", False)):
        return None
    raise RuntimeError(WEB_MIGRATION_ERROR)


@app.after_request
def add_no_cache_headers(response):
    """Avoid stale JS/CSS while iterating on the web demo."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def _handle_consumption_timeout(session_id: str, elapsed: float) -> bool:
    """Reset generation when the active client stops consuming frames."""
    if not model_manager or not model_manager.is_generating:
        return False
    print(
        f"No frame consumed for {elapsed:.1f}s - client disconnected, auto-resetting..."
    )
    if not model_manager.reset():
        print("Auto-reset skipped because the generation thread did not stop cleanly")
        return False
    print("Generation reset due to client disconnect (no frame consumption)")
    return True


def start_consumption_monitor():
    """Start the consumption monitoring thread if not already running."""
    session_service.start_consumption_monitor(_handle_consumption_timeout)
    print("Consumption monitor started")


def get_trajectory_defaults() -> dict:
    traj_cfg = traj_mask_cfg or {}
    route_mode = str(traj_cfg.get("route_mode", "relative_to_actor"))
    return {
        "route_mode": route_mode,
        "horizon_tokens": int(traj_cfg.get("horizon_tokens", 20)),
        "delay_enabled": bool(traj_cfg.get("update_delay_enabled", True)),
        "delay_tokens": int(
            traj_cfg.get("update_delay_tokens", traj_cfg.get("horizon_tokens", 20))
        ),
    }


register_routes(
    app,
    init_model=init_model,
    get_model_manager=lambda: model_manager,
    load_debug_preset_sample=load_debug_preset_sample,
    start_consumption_monitor=start_consumption_monitor,
    session_service=session_service,
    get_trajectory_defaults=get_trajectory_defaults,
)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Flask server for real-time 3D motion generation')
    parser.add_argument('--config', type=str, default='../configs/stream.yaml',
                        help='Path to config yaml file (default: ../configs/stream.yaml)')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the server on (default: 5000)')
    args = parser.parse_args()

    model_config_path = args.config
    traj_mask_cfg = load_traj_mask_cfg(model_config_path)
    debug_preset_cfg = load_debug_preset_cfg(model_config_path)
    
    print("Starting Flask server...")
    print(f"Config file: {model_config_path}")
    print("Trajectory config source: traj_mask section in main config")
    if debug_preset_cfg and bool(debug_preset_cfg.get("enabled", False)):
        print(f"Web demo debug preset enabled: {debug_preset_cfg}")
    print("Note: Model will be loaded on first generation request")
    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
