# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]


def load_finalizer():
    path = EXAMPLE_ROOT / "scripts" / "finalize_artifacts.py"
    spec = importlib.util.spec_from_file_location("phase1_finalizer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_args(tmp_path: Path, *, error_type: str = "") -> argparse.Namespace:
    config = tmp_path / "plugins.toml"
    config.write_text("version = 1\n", encoding="utf-8")
    manifest = tmp_path / "relay-plugin.toml"
    manifest.write_text('[plugin]\nid = "nvidia.switchyard"\n', encoding="utf-8")
    library = tmp_path / "libswitchyard.so"
    library.write_bytes(b"native-plugin-test")
    return argparse.Namespace(
        relay_config=config,
        switchyard_manifest=manifest,
        switchyard_library=library,
        relay_wheel_sha256="a" * 64,
        hermes_repository="https://github.com/bbednarski9/hermes-agent.git",
        hermes_commit="a3d472f0e6bdc376df87b1436a461c4796db6747",
        switchyard_commit="8daac03edf8544144833af1fd009b3da737715bc",
        session_handle="phase1-session",
        started_at=1.0,
        error_type=error_type,
    )


def test_completed_response_is_preserved_after_post_response_failure(tmp_path: Path, monkeypatch) -> None:
    module = load_finalizer()
    root = tmp_path / "artifacts"
    root.mkdir()
    session = tmp_path / "hermes-session.jsonl"
    session.write_text(
        json.dumps(
            {
                "session_id": "phase1-session",
                "messages": [
                    {"role": "user", "content": "solve"},
                    {"role": "assistant", "content": "completed answer"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    log = tmp_path / "hermes.txt"
    log.write_text("normal shutdown\n", encoding="utf-8")
    monkeypatch.setattr(module, "HERMES_SESSION", session)
    monkeypatch.setattr(module, "HERMES_LOG", log)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "0.7.1")

    args = make_args(tmp_path, error_type="InjectedPostResponseFailure")
    module.initialize(args, root)
    module.complete(args, root)

    result = json.loads((root / "direct-hermes-result.json").read_text())
    receipt = json.loads((root / "direct-hermes-receipt.json").read_text())
    assert result["status"] == "preserved_completed_response"
    assert result["final_response"] == "completed answer"
    assert result["error"]["type"] == "InjectedPostResponseFailure"
    assert receipt["cleanup"]["plugin_host_closed"] is True
    assert receipt["cleanup"]["exporters_flushed"] is True
    assert (root / ".complete").read_text() == "completed\n"


def test_no_response_never_creates_a_passed_completion(tmp_path: Path, monkeypatch) -> None:
    module = load_finalizer()
    root = tmp_path / "artifacts"
    root.mkdir()
    session = tmp_path / "hermes-session.jsonl"
    session.write_text('{"messages": []}\n', encoding="utf-8")
    log = tmp_path / "hermes.txt"
    log.write_text("agent stopped\n", encoding="utf-8")
    monkeypatch.setattr(module, "HERMES_SESSION", session)
    monkeypatch.setattr(module, "HERMES_LOG", log)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "0.7.1")

    args = make_args(tmp_path)
    module.initialize(args, root)
    module.complete(args, root)

    completion = json.loads((root / "completion.json").read_text())
    assert completion["status"] == "failed"


def test_empty_session_uses_bounded_quiet_cli_output(tmp_path: Path, monkeypatch) -> None:
    module = load_finalizer()
    root = tmp_path / "artifacts"
    root.mkdir()
    session = tmp_path / "hermes-session.jsonl"
    session.write_text("", encoding="utf-8")
    log = tmp_path / "hermes.txt"
    log.write_text("startup warning\n\nsession_id: cli-session\ncompleted\nanswer\n", encoding="utf-8")
    monkeypatch.setattr(module, "HERMES_SESSION", session)
    monkeypatch.setattr(module, "HERMES_LOG", log)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "0.7.1")

    args = make_args(tmp_path)
    module.initialize(args, root)
    module.complete(args, root)

    result = json.loads((root / "direct-hermes-result.json").read_text())
    assert result["status"] == "completed"
    assert result["session_id"] == "cli-session"
    assert result["final_response"] == "completed\nanswer"


def test_failed_agent_output_is_not_promoted_to_a_completed_response(tmp_path: Path, monkeypatch) -> None:
    module = load_finalizer()
    root = tmp_path / "artifacts"
    root.mkdir()
    session = tmp_path / "hermes-session.jsonl"
    session.write_text("", encoding="utf-8")
    log = tmp_path / "hermes.txt"
    log.write_text(
        "session_id: failed-session\nAPI call failed after 3 retries: provider returned HTTP 400\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "HERMES_SESSION", session)
    monkeypatch.setattr(module, "HERMES_LOG", log)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "0.7.1")

    args = make_args(tmp_path, error_type="NonZeroAgentExitCodeError")
    module.initialize(args, root)
    module.complete(args, root)

    result = json.loads((root / "direct-hermes-result.json").read_text())
    completion = json.loads((root / "completion.json").read_text())
    assert result["status"] == "failed"
    assert result["final_response"] is None
    assert result["error"] == {"type": "NonZeroAgentExitCodeError", "phase": "agent"}
    assert completion["status"] == "failed"
