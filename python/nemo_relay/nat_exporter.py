# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NAT-oriented NeMo Relay event exporter.

The exporter writes canonical ATOF event JSONL that NVIDIA NeMo Agent Toolkit
can replay into its intermediate-step telemetry stream.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Literal

from nemo_relay import subscribers


class NatTelemetryExporter:
    """Write Relay events as JSONL for NeMo Agent Toolkit ingestion."""

    def __init__(self, path: str | Path, *, mode: Literal["append", "overwrite"] = "overwrite") -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._closed = False
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if mode == "overwrite":
            self._path.write_text("")
        elif mode == "append":
            self._path.touch()
        else:
            raise ValueError("mode must be 'append' or 'overwrite'")

    @property
    def path(self) -> str:
        """Return the JSONL output path."""
        return str(self._path)

    def subscriber(self, event) -> None:
        """Subscriber callback that writes one event per JSONL line."""
        with self._lock:
            if self._closed:
                return
            with self._path.open("a") as output:
                output.write(_event_to_json(event))
                output.write("\n")

    def register(self, name: str) -> None:
        """Register this exporter as a global Relay subscriber."""
        subscribers.register(name, self.subscriber)

    def deregister(self, name: str) -> bool:
        """Deregister a previously registered exporter subscriber."""
        return subscribers.deregister(name)

    def force_flush(self) -> None:
        """Wait for queued Relay subscriber callbacks to finish."""
        subscribers.flush()

    def shutdown(self) -> None:
        """Flush queued callbacks and reject future writes."""
        self.force_flush()
        with self._lock:
            self._closed = True

    def __repr__(self) -> str:
        return f"<NatTelemetryExporter path={self._path!s}>"


def _event_to_json(event) -> str:
    if hasattr(event, "to_json"):
        return event.to_json()
    if hasattr(event, "to_dict"):
        return json.dumps(event.to_dict(), separators=(",", ":"))
    return json.dumps(event, separators=(",", ":"))


__all__ = ["NatTelemetryExporter"]
