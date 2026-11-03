# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consumer regressions for public Python declarations."""

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_STREAM_SETUP = textwrap.dedent(
    """\
    from collections.abc import AsyncIterator

    from nemo_relay import Json, LLMRequest, llm

    async def produce(request: LLMRequest) -> AsyncIterator[Json]:
        yield {"text": "hello"}

    def collect(chunk: Json) -> None:
        pass

    def finalize() -> Json:
        return {"text": "hello"}

    """
)


def _check_consumer(consumer_path: Path, source: str) -> subprocess.CompletedProcess[str]:
    consumer_path.write_text(textwrap.dedent(source), encoding="utf-8")
    repository_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    # The canonical suite adds standard-library paths that interfere with ty's typeshed.
    environment.pop("PYTHONPATH", None)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "ty",
            "check",
            "--project",
            str(repository_root),
            "--python",
            sys.executable,
            "--extra-search-path",
            str(repository_root / "python"),
            "--python-version",
            f"{sys.version_info.major}.{sys.version_info.minor}",
            "--output-format",
            "concise",
            "--color",
            "never",
            "--no-progress",
            "--error",
            "unresolved-attribute",
            "--error",
            "invalid-argument-type",
            "--error",
            "invalid-await",
            str(consumer_path),
        ],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            """\
            import nemo_relay

            fields: nemo_relay.JsonObject = {"attempt": 2}
            nemo_relay.log("info", "message", target="consumer", fields=fields)
            nemo_relay.trace("message", target="consumer", fields=fields)
            nemo_relay.debug("message", target="consumer", fields=fields)
            nemo_relay.info("message", target="consumer", fields=fields)
            nemo_relay.warn("message", target="consumer", fields=fields)
            nemo_relay.error("message", target="consumer", fields=fields)
            """,
            id="logging-helpers",
        ),
        pytest.param(
            """\
            from nemo_relay import (
                OpenTelemetryConfig,
                OpenTelemetryFileSinkConfig,
                OpenTelemetrySubscriber,
            )

            def subscribe(
                endpoint_configuration: OpenTelemetryConfig,
                file_configuration: OpenTelemetryFileSinkConfig,
            ) -> tuple[OpenTelemetrySubscriber, OpenTelemetrySubscriber]:
                return (
                    OpenTelemetrySubscriber(endpoint_configuration),
                    OpenTelemetrySubscriber(file_configuration),
                )
            """,
            id="subscriber-configurations",
        ),
        pytest.param(
            """\
            from nemo_relay import AnnotatedLLMRequest, Json

            request = AnnotatedLLMRequest([])
            request.store = True
            request.previous_response_id = "previous-response"
            request.truncation = "auto"
            request.reasoning = {"effort": "low"}
            request.include = ["reasoning.encrypted_content"]
            request.user = "consumer"
            request.metadata = {"purpose": "typing"}
            request.service_tier = "default"
            request.parallel_tool_calls = False
            request.max_output_tokens = 128
            request.max_tool_calls = 2
            request.top_logprobs = 3
            request.stream = True

            storage_setting: bool | None = request.store
            previous_response: str | None = request.previous_response_id
            truncation_setting: Json = request.truncation
            reasoning_settings: Json = request.reasoning
            included_fields: Json = request.include
            user_identifier: str | None = request.user
            request_metadata: Json = request.metadata
            service_tier: str | None = request.service_tier
            parallel_calls: bool | None = request.parallel_tool_calls
            output_limit: int | None = request.max_output_tokens
            tool_limit: int | None = request.max_tool_calls
            probability_count: int | None = request.top_logprobs
            streaming_setting: bool | None = request.stream

            request.store = None
            request.previous_response_id = None
            request.truncation = None
            request.reasoning = None
            request.include = None
            request.user = None
            request.metadata = None
            request.service_tier = None
            request.parallel_tool_calls = None
            request.max_output_tokens = None
            request.max_tool_calls = None
            request.top_logprobs = None
            request.stream = None
            """,
            id="request-properties",
        ),
        pytest.param(
            _STREAM_SETUP
            + textwrap.dedent(
                """\
                async def consume(request: LLMRequest) -> None:
                    stream = await llm.stream_execute("consumer", request, produce, collect, finalize)
                    try:
                        async for chunk in stream:
                            pass
                    finally:
                        await stream.aclose()
                """
            ),
            id="stream-iteration-and-close",
        ),
    ],
)
def test_supported_public_calls_type_check(tmp_path: Path, source: str) -> None:
    completed = _check_consumer(tmp_path / "consumer.py", source)

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_awaiting_stream_is_rejected(tmp_path: Path) -> None:
    source = _STREAM_SETUP + textwrap.dedent(
        """\
        async def consume(request: LLMRequest) -> None:
            stream = await llm.stream_execute("consumer", request, produce, collect, finalize)
            await stream
        """
    )
    consumer_path = tmp_path / "invalid_stream_consumer.py"
    completed = _check_consumer(consumer_path, source)
    diagnostics = completed.stdout + completed.stderr

    assert completed.returncode == 1, diagnostics
    assert re.findall(r"\b(?:error|warning)\[([^]]+)\]", diagnostics) == ["invalid-await"], diagnostics
    assert str(consumer_path) in diagnostics, diagnostics
