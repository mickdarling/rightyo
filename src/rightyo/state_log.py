"""An opt-in local record of each live decision's exact state and answer (#199).

Live and offline decisions on the same turns disagree, and the difference must come from
something live only. This keeps, for each decided turn, the bounded state RightyO sent to
the decision model and what came back, so the two can be compared turn by turn.

It holds transcript text, so it is off unless the operator names a file, the file is
created `0600` and never followed through a symlink, and writing stops for good once the
file reaches `max_bytes`. A failed write is reported once and never affects a decision.
"""

from __future__ import annotations

import json
import os
import stat
import threading
import time
from pathlib import Path
from typing import Any, Callable

from rightyo.contracts import ProviderDecision

DEFAULT_MAX_BYTES = 50 * 1024 * 1024


class DecisionStateLog:
    def __init__(
        self,
        path: Path,
        max_bytes: int = DEFAULT_MAX_BYTES,
        report: Callable[[str], None] | None = None,
    ) -> None:
        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError("the decision state log needs an absolute path")
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("invalid decision state log size")
        self.path = path
        self.max_bytes = max_bytes
        self.report = report
        self._lock = threading.Lock()
        self._stopped = False

    def __call__(self, state: dict[str, Any], decision: ProviderDecision) -> None:
        line = (
            json.dumps(
                {
                    "logged_at_ms": int(time.time() * 1000),
                    "state": state,
                    "decision": {
                        "label": decision.label,
                        "attention_choice": decision.attention_choice or decision.label,
                        "recipient": decision.recipient,
                        "confidence": decision.confidence,
                        "probabilities": decision.probabilities,
                        "model": decision.model,
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        with self._lock:
            if self._stopped:
                return
            try:
                self._append(line)
            except OSError as error:
                self._stop(f"decision_state_log stopped reason={type(error).__name__}")

    def _append(self, line: bytes) -> None:
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC
        descriptor = os.open(self.path, flags, 0o600)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
                self._stop("decision_state_log stopped reason=not_a_private_file")
                return
            if info.st_size + len(line) > self.max_bytes:
                self._stop("decision_state_log stopped reason=full")
                return
            written = 0
            while written < len(line):
                written += os.write(descriptor, line[written:])
        finally:
            os.close(descriptor)

    def _stop(self, note: str) -> None:
        self._stopped = True
        if self.report is not None:
            try:
                self.report(note)
            except Exception:  # noqa: BLE001 - a report must never affect a decision
                pass
