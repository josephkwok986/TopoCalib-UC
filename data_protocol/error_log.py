"""Append-only project error log helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .constants import DEFAULT_ERROR_LOG


def append_error(
    message: str,
    *,
    error_log: Path = DEFAULT_ERROR_LOG,
    context: Mapping[str, Any] | None = None,
) -> None:
    """Record a major preprocessing error in the repository-level ERROR.txt."""

    record = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "message": message,
        "context": dict(context or {}),
    }
    with error_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        f.write("\n")
