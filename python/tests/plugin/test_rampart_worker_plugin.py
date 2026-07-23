# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the Rampart Python worker plugin."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("nemo_relay_plugin")

from nemo_relay_plugin import PluginContext  # noqa: E402

FAILURE_REPLACEMENT = "[REDACTED:PII_DETECTION_FAILURE]"


@pytest.fixture(name="worker", scope="module")
def worker_fixture() -> Any:
    example_root = Path(__file__).parents[3] / "examples/rampart-pii-worker-plugin"
    sys.path.insert(0, str(example_root))
    try:
        yield importlib.import_module("nemo_relay_rampart_worker.worker")
    finally:
        sys.path.remove(str(example_root))
        for name in tuple(sys.modules):
            if name == "nemo_relay_rampart_worker" or name.startswith("nemo_relay_rampart_worker."):
                sys.modules.pop(name, None)


def test_manifest_integrity_matches_entrypoint() -> None:
    example_root = Path(__file__).parents[3] / "examples/rampart-pii-worker-plugin"
    manifest = tomllib.loads((example_root / "relay-plugin.toml").read_text(encoding="utf-8"))
    artifact = example_root / manifest["source"]["artifact"]

    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()

    assert manifest["integrity"]["sha256"] == f"sha256:{digest}"
    assert manifest["load"]["entrypoint"] == "nemo_relay_rampart_worker.worker:main"


def test_schema_matches_worker_configuration(worker: Any) -> None:
    example_root = Path(__file__).parents[3] / "examples/rampart-pii-worker-plugin"
    schema = json.loads((example_root / "config.schema.json").read_text(encoding="utf-8"))

    assert set(schema["properties"]) == set(worker._CONFIG_FIELDS)
    assert schema["properties"]["max_latency_ms"]["default"] == worker.DEFAULT_MAX_LATENCY_MS
    assert "model_id" not in schema["properties"]


def test_worker_validates_closed_configuration(worker: Any) -> None:
    plugin = worker.RampartWorkerPlugin()

    assert plugin.validate({}) == []
    diagnostics = plugin.validate(
        {
            "unknown": True,
            "max_latency_ms": 0,
            "priority": 2**31 - 1,
        }
    )
    assert [(item.field, item.code) for item in diagnostics] == [
        ("unknown", "nvidia.rampart_pii.invalid_config"),
        ("max_latency_ms", "nvidia.rampart_pii.invalid_config"),
        ("priority", "nvidia.rampart_pii.invalid_config"),
    ]
    diagnostics = plugin.validate(
        {
            "input": False,
            "output": False,
            "tool_input": False,
            "tool_output": False,
        }
    )
    assert len(diagnostics) == 1
    assert diagnostics[0].message == "at least one managed sanitization surface must be enabled"


async def test_worker_registers_sanitizers_and_fail_closed_backstops(
    worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_classifier = MagicMock(return_value=lambda _text: [])
    monkeypatch.setattr(
        worker,
        "load_classifier",
        load_classifier,
    )
    context = MagicMock(spec=PluginContext)

    await worker.RampartWorkerPlugin().register(context, {})

    load_classifier.assert_called_once_with(
        worker.DEFAULT_MODEL_ID,
        worker.DEFAULT_MODEL_REVISION,
        False,
    )
    assert context.register_scope_sanitize_start_guardrail.call_count == 1
    assert context.register_scope_sanitize_end_guardrail.call_count == 1
    assert context.register_llm_sanitize_request_guardrail.call_count == 1
    assert context.register_llm_sanitize_response_guardrail.call_count == 1
    assert context.register_tool_sanitize_request_guardrail.call_count == 1
    assert context.register_tool_sanitize_response_guardrail.call_count == 1

    _name, backstop = context.register_scope_sanitize_start_guardrail.call_args.args
    fields = {
        "data": {"safe": True},
        "category_profile": {"subtype": "test"},
        "metadata": None,
    }
    assert backstop({"name": "scope"}, fields) is fields

    _name, sanitize_request = context.register_llm_sanitize_request_guardrail.call_args.args
    sanitized = await sanitize_request(
        {
            "headers": {},
            "content": {
                "model": "test-model",
                "messages": [{"role": "user", "content": "email alex@example.com"}],
            },
        }
    )
    assert sanitized["content"]["model"] == "test-model"
    assert sanitized["content"]["messages"][0]["content"] == "email [REDACTED:EMAIL]"


async def test_worker_fails_activation_before_registration_when_model_is_unavailable(
    worker: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(_model: str, _revision: str, _network: bool) -> Any:
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(worker, "load_classifier", unavailable)
    context = MagicMock(spec=PluginContext)

    with pytest.raises(RuntimeError, match="model unavailable"):
        await worker.RampartWorkerPlugin().register(context, {})

    context.register_scope_sanitize_start_guardrail.assert_not_called()
    context.register_llm_sanitize_request_guardrail.assert_not_called()


def _detect_terms(*terms: tuple[str, str]) -> Callable[[str], list[dict[str, object]]]:
    def classify(text: str) -> list[dict[str, object]]:
        detections = []
        for term, label in terms:
            start = text.index(term)
            detections.append(
                {
                    "start": start,
                    "end": start + len(term),
                    "entity_group": label,
                    "score": 0.99,
                }
            )
        return detections

    return classify


def test_worker_combines_deterministic_and_contextual_detection(worker: Any) -> None:
    sanitizer = worker.RampartSanitizer(
        _detect_terms(("Alex", "GIVEN_NAME"), ("Rivera", "SURNAME")),
        max_latency_ms=1_000,
    )

    sanitized = sanitizer.sanitize("Alex Rivera can be reached at alex@example.com")

    assert sanitized == ("[REDACTED:GIVEN_NAME] [REDACTED:SURNAME] can be reached at [REDACTED:EMAIL]")


def test_worker_applies_rampart_keep_set(worker: Any) -> None:
    value = "Phoenix"
    sanitizer = worker.RampartSanitizer(
        lambda _text: [
            {
                "start": 0,
                "end": len(value),
                "entity_group": "CITY",
                "score": 0.99,
            }
        ],
        max_latency_ms=1_000,
    )

    assert sanitizer.sanitize(value) == value


def test_worker_uses_rampart_recall_threshold(worker: Any) -> None:
    value = "Alex"

    def sanitizer(score: float) -> Any:
        return worker.RampartSanitizer(
            lambda _text: [
                {
                    "start": 0,
                    "end": len(value),
                    "entity_group": "GIVEN_NAME",
                    "score": score,
                }
            ],
            max_latency_ms=1_000,
        )

    assert sanitizer(0.39).sanitize(value) == value
    assert sanitizer(0.4).sanitize(value) == "[REDACTED:GIVEN_NAME]"


def test_worker_matches_rampart_hyphen_inference_fold(worker: Any) -> None:
    classified: list[str] = []

    def classify(text: str) -> list[dict[str, object]]:
        classified.append(text)
        return []

    sanitizer = worker.RampartSanitizer(classify, max_latency_ms=1_000)

    assert sanitizer.sanitize("Jean-Baptiste") == "Jean-Baptiste"
    assert classified == ["Jean Baptiste"]


@pytest.mark.parametrize(
    "value",
    [
        "us-west-2",
        "trace_id: 550e8400-e29b-41d4-a716-446655440000",
        "span id 4bf92f3577b34da6a3ce929d0e0e4736",
        "HTTP status 404",
    ],
)
def test_worker_preserves_operational_values(worker: Any, value: str) -> None:
    sanitizer = worker.RampartSanitizer(
        lambda text: [
            {
                "start": 0,
                "end": len(text),
                "entity_group": "SURNAME",
                "score": 0.99,
            }
        ],
        max_latency_ms=1_000,
    )

    assert sanitizer.sanitize(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "550e8400-e29b-41d4-a716-446655440000",
        "4bf92f3577b34da6a3ce929d0e0e4736",
    ],
)
def test_worker_does_not_assume_unlabeled_ids_are_operational(
    worker: Any,
    value: str,
) -> None:
    sanitizer = worker.RampartSanitizer(lambda _text: [], max_latency_ms=1_000)

    assert sanitizer.sanitize(value) == "[REDACTED:IDENTIFIER]"


def test_worker_fails_closed_on_classifier_error(worker: Any) -> None:
    def fail(_text: str) -> list[dict[str, object]]:
        raise RuntimeError("classifier unavailable")

    sanitizer = worker.RampartSanitizer(fail, max_latency_ms=1_000)

    assert sanitizer.sanitize("ordinary content") == FAILURE_REPLACEMENT


@pytest.mark.parametrize(
    "detection",
    [
        {
            "start": 0,
            "end": 4,
            "entity_group": "UNSUPPORTED",
            "score": 0.99,
        },
        {
            "start": -1,
            "end": 4,
            "entity_group": "GIVEN_NAME",
            "score": 0.99,
        },
        {
            "start": 0,
            "end": 4,
            "entity_group": "GIVEN_NAME",
            "score": float("nan"),
        },
    ],
)
def test_worker_fails_closed_on_invalid_model_output(
    worker: Any,
    detection: dict[str, object],
) -> None:
    sanitizer = worker.RampartSanitizer(
        lambda _text: [detection],
        max_latency_ms=1_000,
    )

    assert sanitizer.sanitize("Alex") == FAILURE_REPLACEMENT


def test_worker_fails_closed_when_latency_budget_is_exceeded(worker: Any) -> None:
    times = iter((1.0, 1.2))
    sanitizer = worker.RampartSanitizer(
        lambda _text: [],
        max_latency_ms=100,
        clock=lambda: next(times),
    )

    assert sanitizer.sanitize("ordinary content") == FAILURE_REPLACEMENT


def test_worker_sanitizes_content_without_changing_llm_structure(worker: Any) -> None:
    sanitizer = worker.RampartSanitizer(lambda _text: [], max_latency_ms=1_000)
    request = {
        "headers": {"authorization": "not-exported-by-relay"},
        "content": {
            "model": "model@example.com",
            "messages": [
                {
                    "role": "user",
                    "content": "Contact alex@example.com",
                }
            ],
        },
    }

    sanitized = worker.sanitize_llm_request(request, sanitizer)

    assert sanitized["headers"] == request["headers"]
    assert sanitized["content"]["model"] == "model@example.com"
    assert sanitized["content"]["messages"][0]["content"] == "Contact [REDACTED:EMAIL]"
    assert request["content"]["messages"][0]["content"] == "Contact alex@example.com"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("Contact alex@example.com", "Contact [REDACTED:EMAIL]"),
        (["Contact alex@example.com"], ["Contact [REDACTED:EMAIL]"]),
        (
            [
                {
                    "model": "model@example.com",
                    "content": "Contact alex@example.com",
                }
            ],
            [
                {
                    "model": "model@example.com",
                    "content": "Contact [REDACTED:EMAIL]",
                }
            ],
        ),
    ],
)
def test_worker_sanitizes_root_llm_content(
    worker: Any,
    payload: Any,
    expected: Any,
) -> None:
    sanitizer = worker.RampartSanitizer(lambda _text: [], max_latency_ms=1_000)

    sanitized = worker.sanitize_llm_response(payload, sanitizer)

    assert sanitized == expected


def test_worker_sanitizes_tool_content_and_preserves_operational_ids(worker: Any) -> None:
    sanitizer = worker.RampartSanitizer(lambda _text: [], max_latency_ms=1_000)
    payload = {
        "email": "alex@example.com",
        "region": "us-west-2",
        "trace_id": "550e8400-e29b-41d4-a716-446655440000",
    }

    sanitized = worker.sanitize_tool_payload(payload, sanitizer)

    assert sanitized == {
        "email": "[REDACTED:EMAIL]",
        "region": "us-west-2",
        "trace_id": "550e8400-e29b-41d4-a716-446655440000",
    }
    assert payload["email"] == "alex@example.com"


def test_worker_fails_closed_on_oversized_content(worker: Any) -> None:
    sanitizer = worker.RampartSanitizer(lambda _text: [], max_latency_ms=1_000)
    payload = {"content": "x" * 65_537}

    assert worker.sanitize_tool_payload(payload, sanitizer) == FAILURE_REPLACEMENT


def test_worker_fails_closed_on_deep_content_without_recursive_fallback(worker: Any) -> None:
    sanitizer = worker.RampartSanitizer(lambda _text: [], max_latency_ms=1_000)
    payload: dict[str, Any] = {"content": "alex@example.com"}
    for _ in range(2_000):
        payload = {"nested": payload}

    assert worker.sanitize_llm_response(payload, sanitizer) == FAILURE_REPLACEMENT


def test_benchmark_does_not_depend_on_host_rampart_integration() -> None:
    benchmark = (Path(__file__).parents[3] / "examples/rampart-pii-worker-plugin/benchmark.py").read_text(
        encoding="utf-8"
    )

    assert "nemo_relay.integrations.rampart" not in benchmark
    assert "in_process" not in benchmark
