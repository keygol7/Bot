"""Small time helpers (stdlib only)."""

from __future__ import annotations

from datetime import datetime, timezone


def parse_iso8601(s: str | None) -> float | None:
    """Parse an ISO-8601 timestamp (e.g. ``2026-09-27T13:00:00Z``) to epoch seconds.

    Returns ``None`` for missing/unparseable input. Naive timestamps are treated as UTC.
    """
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()
