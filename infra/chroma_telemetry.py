from __future__ import annotations

from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent
from overrides import override


class NoOpProductTelemetry(ProductTelemetryClient):
    """No-op telemetry client to avoid PostHog runtime issues in local/dev runs."""

    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        _ = event


