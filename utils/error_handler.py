from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from models.responses import GenericResponse
from utils.exceptions import AppException


P = ParamSpec("P")
R = TypeVar("R")


def safe_execute(func: Callable[P, R]) -> Callable[P, R | GenericResponse]:
    """Wrap orchestrator calls so callers always receive structured responses."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | GenericResponse:
        try:
            return func(*args, **kwargs)
        except AppException as exc:
            return GenericResponse(success=False, error=f"{exc.code}: {exc.message}")
        except Exception as exc:  # pragma: no cover - defensive fallback
            return GenericResponse(success=False, error=f"unhandled_error: {exc}")

    return wrapper

