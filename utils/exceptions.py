from __future__ import annotations

from models.schemas import SkippedReason


class AppException(Exception):
    """Base application exception with optional machine-readable code."""

    def __init__(self, message: str, code: str = "app_error") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class ValidationException(AppException):
    def __init__(self, message: str) -> None:
        super().__init__(message=message, code="validation_error")


class EmbeddingException(AppException):
    def __init__(self, message: str) -> None:
        super().__init__(message=message, code="embedding_error")


class RetrievalException(AppException):
    def __init__(self, message: str) -> None:
        super().__init__(message=message, code="retrieval_error")


class ParseException(AppException):
    def __init__(self, message: str, skipped_reason: SkippedReason) -> None:
        super().__init__(message=message, code="parse_error")
        self.skipped_reason = skipped_reason

