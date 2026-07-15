"""Small JSON response helpers for the Web API boundary."""

from __future__ import annotations


def success(**payload):
    return {"status": "success", **payload}


def error(message: str, **payload):
    return {"status": "error", "message": message, **payload}


__all__ = ["error", "success"]
