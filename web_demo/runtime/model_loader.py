"""Model loading boundary for the new hybrid Web runtime."""

from __future__ import annotations

from pathlib import Path

from .model_bundle import ModelBundle


WEB_RUNTIME_BLOCKER = (
    "BLOCKED_ON_LDF_CHECKPOINT: the strict four-frame BodyVAE and atomic "
    "InferenceSession runtime are ready, but no formally trained hybrid LDF "
    "checkpoint/loader contract exists yet. Finish LDF training, freeze its "
    "Root/Body prediction and checkpoint schema, then configure Web model paths."
)


def resolve_repo_path(path: str | Path | None) -> Path | None:
    """Resolve a configured path relative to the Floodcontrol repository."""

    if path is None or not str(path).strip():
        return None
    value = Path(path).expanduser()
    if value.is_absolute():
        return value.resolve()
    root = Path(__file__).resolve().parents[2]
    return (root / value).resolve()


def load_model_bundle(_config) -> ModelBundle:
    """Fail explicitly until the formal LDF checkpoint contract is frozen."""

    raise RuntimeError(WEB_RUNTIME_BLOCKER)


__all__ = ["WEB_RUNTIME_BLOCKER", "load_model_bundle", "resolve_repo_path"]
