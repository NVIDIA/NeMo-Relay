# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create the direct Hermes result and lifecycle receipt inside a Harbor task."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import time
import tomllib
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parent
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))
from relay_version import require_supported_version
from typing import Any

SCHEMA_VERSION = "harbor-hermes-switchyard.phase1.v1"
MAX_DIAGNOSTIC_BYTES = 1024 * 1024
HERMES_SESSION = Path("/logs/agent/hermes-session.jsonl")
HERMES_LOG = Path("/logs/agent/hermes.txt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def checked_artifact_root(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError("artifact root must be absolute")
    resolved = path.resolve(strict=False)
    allowed = Path("/logs/agent").resolve()
    if resolved == allowed or allowed not in resolved.parents:
        raise ValueError("artifact root must be a child of /logs/agent")
    resolved.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(resolved, 0o700)
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("initialize", "complete"))
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--hermes-repository", required=True)
    parser.add_argument("--hermes-commit", required=True)
    parser.add_argument("--switchyard-commit", required=True)
    parser.add_argument("--relay-wheel-sha256", required=True)
    parser.add_argument("--relay-config", type=Path, required=True)
    parser.add_argument("--switchyard-manifest", type=Path, required=True)
    parser.add_argument("--switchyard-library", type=Path, required=True)
    parser.add_argument("--session-handle", default="")
    parser.add_argument("--started-at", type=float)
    parser.add_argument("--error-type", default="")
    return parser.parse_args()


def initialize(args: argparse.Namespace, root: Path) -> None:
    with args.switchyard_manifest.open("rb") as stream:
        manifest = tomllib.load(stream)
    plugin_id = manifest.get("plugin", {}).get("id")
    if plugin_id != "nvidia.switchyard":
        raise ValueError(f"unexpected Switchyard plugin id: {plugin_id!r}")

    config_digest = sha256(args.relay_config)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "initialized",
        "activation_mode": "relay_standard_dynamic",
        "session_handle": args.session_handle or None,
        "dependencies": {
            "nemo_relay": {
                "version": importlib.metadata.version("nemo-relay"),
                "wheel_sha256": args.relay_wheel_sha256,
            },
            "hermes": {
                "repository": args.hermes_repository,
                "commit": args.hermes_commit,
            },
            "switchyard": {
                "commit": args.switchyard_commit,
                "plugin_id": plugin_id,
                "manifest_sha256": sha256(args.switchyard_manifest),
                "library_sha256": sha256(args.switchyard_library),
            },
        },
        "relay_config_sha256": config_digest,
        "dynamic_plugin_ids": [plugin_id],
        "routing_contract": {
            "relay_outer_lifecycle": True,
            "execution_intercept_owner": plugin_id,
            "provider_http_client_owner": "switchyard-llm-client",
            "separate_switchyard_service": False,
        },
        "artifacts": {
            "root": str(root),
            "atof": str(root / "relay" / "trajectory.atof.jsonl"),
            "atif_directory": str(root / "relay" / "atif"),
            "bounded_diagnostics": str(root / "diagnostics" / "hermes-tail.txt"),
        },
        "cleanup": {
            "plugin_host_closed": False,
            "exporters_flushed": False,
            "completion_marker_written": False,
        },
    }
    try:
        require_supported_version(receipt["dependencies"]["nemo_relay"]["version"])
    except ValueError as error:
        raise RuntimeError("Hermes environment did not install a supported nemo-relay release") from error
    (root / "relay" / "atif").mkdir(mode=0o700, parents=True, exist_ok=True)
    (root / "diagnostics").mkdir(mode=0o700, parents=True, exist_ok=True)
    atomic_json(root / "direct-hermes-receipt.json", receipt)


def _read_session_messages(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    messages: list[dict[str, Any]] = []
    session_id: str | None = None
    if not path.is_file():
        return messages, session_id
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        candidate = payload.get("session_id") or payload.get("id")
        if isinstance(candidate, str) and candidate:
            session_id = candidate
        nested = payload.get("messages")
        if isinstance(nested, list):
            messages.extend(item for item in nested if isinstance(item, dict))
        elif payload.get("role"):
            messages.append(payload)
    return messages, session_id


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return ""


def _last_assistant_response(messages: list[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            text = _text_content(message.get("content"))
            if text:
                return text
    return None


def _response_from_cli_log(text: str) -> tuple[str | None, str | None]:
    """Recover quiet-mode output when Hermes produced an empty session export."""
    lines = text.splitlines()
    marker_index: int | None = None
    session_id: str | None = None
    for index, line in enumerate(lines):
        match = re.fullmatch(r"\s*session_id:\s*(\S+)\s*", line)
        if match:
            marker_index = index
            session_id = match.group(1)
    if marker_index is None:
        return None, None
    response = "\n".join(lines[marker_index + 1 :]).strip()
    return response or None, session_id


def _write_bounded_diagnostics(root: Path) -> str:
    destination = root / "diagnostics" / "hermes-tail.txt"
    if not HERMES_LOG.is_file():
        destination.write_text("", encoding="utf-8")
        os.chmod(destination, 0o600)
        return ""
    size = HERMES_LOG.stat().st_size
    with HERMES_LOG.open("rb") as stream:
        if size > MAX_DIAGNOSTIC_BYTES:
            stream.seek(-MAX_DIAGNOSTIC_BYTES, os.SEEK_END)
        content = stream.read(MAX_DIAGNOSTIC_BYTES)
    destination.write_bytes(content)
    os.chmod(destination, 0o600)
    return content.decode("utf-8", errors="replace")


def complete(args: argparse.Namespace, root: Path) -> None:
    receipt_path = root / "direct-hermes-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    messages, exported_session_id = _read_session_messages(HERMES_SESSION)
    session_response = _last_assistant_response(messages)
    diagnostic_text = _write_bounded_diagnostics(root)
    log_response, log_session_id = _response_from_cli_log(diagnostic_text)
    # Quiet mode sometimes leaves the exported session empty, so successful
    # runs recover their response from stdout. A non-zero agent command may
    # instead print an error after ``session_id:``; never promote that text to
    # a completed response. The deterministic Phase 1 late-failure injection
    # happens after a successful inherited run and intentionally exercises the
    # bounded stdout recovery path.
    allow_log_response = not args.error_type or args.error_type == "InjectedPostResponseFailure"
    response = session_response or (log_response if allow_log_response else None)
    exported_session_id = exported_session_id or log_session_id
    lowered = diagnostic_text.lower()
    cleanup_failure = any(
        marker in lowered
        for marker in (
            "plugin configuration cleanup failed",
            "plugin subscriber flush failed",
            "exporter flush failed",
            "plugin teardown failed",
        )
    )
    late_failure = bool(args.error_type or cleanup_failure)
    if response and late_failure:
        status = "preserved_completed_response"
    elif response:
        status = "completed"
    else:
        status = "failed"

    ended_at = time.time()
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "final_response": response,
        "session_id": exported_session_id or args.session_handle or None,
        "timing": {
            "started_at_unix": args.started_at,
            "ended_at_unix": ended_at,
            "duration_seconds": (max(0.0, ended_at - args.started_at) if args.started_at is not None else None),
        },
        "error": (
            {
                "type": args.error_type or "RelayCleanupError",
                "phase": (
                    "shutdown"
                    if cleanup_failure or args.error_type == "InjectedPostResponseFailure"
                    else "agent"
                ),
            }
            if late_failure
            else None
        ),
    }
    atomic_json(root / "direct-hermes-result.json", result)

    receipt["status"] = "completed" if status != "failed" else "failed"
    receipt["result_status"] = status
    receipt["cleanup"] = {
        "plugin_host_closed": not cleanup_failure,
        "exporters_flushed": not cleanup_failure,
        "completion_marker_written": True,
        "late_failure": late_failure,
    }
    atomic_json(receipt_path, receipt)
    completion = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if status != "failed" else "failed",
        "result_status": status,
        "completed_at_unix": ended_at,
    }
    atomic_json(root / "completion.json", completion)
    marker = root / ".complete"
    marker.write_text("completed\n", encoding="utf-8")
    os.chmod(marker, 0o600)


def main() -> int:
    args = parse_args()
    root = checked_artifact_root(args.artifact_root)
    if args.mode == "initialize":
        initialize(args, root)
    else:
        complete(args, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
