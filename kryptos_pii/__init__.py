"""Kryptos PII Protection: a platform extension.

The security logic lives here and nowhere else. The orchestrator reaches it
through the extension contract, the SDK and the MCP server are thin adapters
over the same :func:`engine.run`, and none of them re-implement detection.
"""

from kryptos_pii.engine import EXTENSION_NAME, EXTENSION_VERSION, OPERATIONS, Outcome, run

__all__ = ["EXTENSION_NAME", "EXTENSION_VERSION", "OPERATIONS", "Outcome", "run"]
__version__ = EXTENSION_VERSION
