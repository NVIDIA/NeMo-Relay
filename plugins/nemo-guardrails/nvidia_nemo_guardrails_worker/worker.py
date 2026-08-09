# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run NeMo Guardrails 0.23 input rails as a Relay final-input policy."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
from typing import Any, Callable, cast

import nemoguardrails
from nemoguardrails.rails.llm.config import RailsConfig
from nemoguardrails.rails.llm.llmrails import LLMRails
from nemoguardrails.rails.llm.options import RailStatus, RailType

from nemo_relay_plugin import (
    AnnotatedLlmRequest,
    ConfigDiagnostic,
    DiagnosticLevel,
    Json,
    LlmFinalInputPolicyOutcome,
    LlmRequest,
    PluginContext,
    PolicyFailureMode,
    WorkerPlugin,
    serve_plugin,
)

PLUGIN_ID = "nvidia.nemo_guardrails"
SUPPORTED_NEMO_GUARDRAILS_VERSION = "0.23.0"
DEFAULT_BLOCKED_MESSAGE = "Request blocked by NeMo Guardrails input policy."
_ALLOWED_CONFIG_FIELDS = frozenset(
    {
        "config_path",
        "config_yaml",
        "colang_content",
        "priority",
        "timeout_ms",
        "failure_mode",
        "max_concurrency",
        "blocked_message",
    }
)
_SUPPORTED_MESSAGE_ROLES = frozenset({"system", "developer", "user", "assistant"})


class WorkerConfigurationError(ValueError):
    """Report one invalid worker configuration field."""

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field


class UnsupportedInputError(ValueError):
    """Report provider-neutral input that the preview cannot evaluate safely."""


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    """Validated worker settings."""

    config_path: str | None
    config_yaml: str | None
    colang_content: str | None
    priority: int
    timeout_ms: int
    failure_mode: PolicyFailureMode
    max_concurrency: int
    blocked_message: str


EngineFactory = Callable[[RailsConfig], Any]


class NemoGuardrailsWorker(WorkerPlugin):
    """Install a Guardrails 0.23 input check at Relay final-input time."""

    plugin_id = PLUGIN_ID

    def __init__(self, engine_factory: EngineFactory = LLMRails) -> None:
        self._engine_factory = engine_factory
        self._prepared_raw_config: Json | None = None
        self._prepared_config: WorkerConfig | None = None
        self._engine: Any | None = None

    def validate(self, config: Json) -> list[ConfigDiagnostic | dict[str, Any]]:
        """Validate Relay settings and compile the Guardrails configuration once."""
        try:
            self._prepare(config)
        except WorkerConfigurationError as exc:
            return [
                ConfigDiagnostic(
                    level=DiagnosticLevel.ERROR,
                    code="nvidia.nemo_guardrails.invalid_config",
                    component=self.plugin_id,
                    field=exc.field,
                    message=str(exc),
                )
            ]
        except Exception as exc:  # noqa: BLE001 - third-party config failures become diagnostics.
            return [
                ConfigDiagnostic(
                    level=DiagnosticLevel.ERROR,
                    code="nvidia.nemo_guardrails.config_load_failed",
                    component=self.plugin_id,
                    message=f"failed to load NeMo Guardrails configuration: {exc}",
                )
            ]
        return []

    def register(self, ctx: PluginContext, config: Json) -> None:
        """Register one bounded asynchronous final-input policy."""
        settings, engine = self._prepare(config)
        limiter = asyncio.Semaphore(settings.max_concurrency)

        async def check_final_input(
            _name: str,
            request: LlmRequest,
            annotated_request: AnnotatedLlmRequest | None,
        ) -> LlmFinalInputPolicyOutcome:
            try:
                messages = _guardrails_messages(annotated_request)
            except UnsupportedInputError as exc:
                return LlmFinalInputPolicyOutcome.reject(
                    "nemo_guardrails.unsupported_input",
                    str(exc),
                    evidence=_evidence("unsupported"),
                )

            async with limiter:
                result = await engine.check_async(messages, rail_types=[RailType.INPUT])
            return _policy_outcome(result, request, annotated_request, settings)

        ctx.register_llm_final_input_policy(
            "input",
            check_final_input,
            priority=settings.priority,
            timeout_ms=settings.timeout_ms,
            failure_mode=settings.failure_mode,
        )

    def _prepare(self, raw_config: Json) -> tuple[WorkerConfig, Any]:
        if self._prepared_raw_config == raw_config and self._prepared_config is not None and self._engine is not None:
            return self._prepared_config, self._engine

        if nemoguardrails.__version__ != SUPPORTED_NEMO_GUARDRAILS_VERSION:
            raise WorkerConfigurationError(
                "NeMo Guardrails worker requires "
                f"nemoguardrails=={SUPPORTED_NEMO_GUARDRAILS_VERSION}, found {nemoguardrails.__version__!r}"
            )
        settings = _parse_config(raw_config)
        rails_config = _load_rails_config(settings)
        engine = self._engine_factory(rails_config)
        self._prepared_raw_config = copy.deepcopy(raw_config)
        self._prepared_config = settings
        self._engine = engine
        return settings, engine


def _parse_config(value: Json) -> WorkerConfig:
    if not isinstance(value, dict):
        raise WorkerConfigurationError("plugin config must be a JSON object")
    unknown = sorted(set(value) - _ALLOWED_CONFIG_FIELDS)
    if unknown:
        raise WorkerConfigurationError(f"unknown configuration field {unknown[0]!r}", field=unknown[0])

    config_path = _optional_nonempty_string(value.get("config_path"), "config_path")
    config_yaml = _optional_nonempty_string(value.get("config_yaml"), "config_yaml")
    if (config_path is None) == (config_yaml is None):
        raise WorkerConfigurationError(
            "configure exactly one of config_path or config_yaml",
            field="config_path",
        )
    colang_content = _optional_string(value.get("colang_content"), "colang_content")
    if config_path is not None and colang_content is not None:
        raise WorkerConfigurationError("colang_content requires config_yaml", field="colang_content")

    priority = _integer(value.get("priority", 0), "priority")
    timeout_ms = _positive_integer(value.get("timeout_ms", 30_000), "timeout_ms")
    max_concurrency = _positive_integer(value.get("max_concurrency", 16), "max_concurrency")
    blocked_message = _nonempty_string(value.get("blocked_message", DEFAULT_BLOCKED_MESSAGE), "blocked_message")
    failure_mode_value = value.get("failure_mode", PolicyFailureMode.FAIL_CLOSED.value)
    if not isinstance(failure_mode_value, str):
        raise WorkerConfigurationError("failure_mode must be fail_closed or fail_open", field="failure_mode")
    try:
        failure_mode = PolicyFailureMode(failure_mode_value)
    except ValueError as exc:
        raise WorkerConfigurationError(
            "failure_mode must be fail_closed or fail_open",
            field="failure_mode",
        ) from exc

    return WorkerConfig(
        config_path=config_path,
        config_yaml=config_yaml,
        colang_content=colang_content,
        priority=priority,
        timeout_ms=timeout_ms,
        failure_mode=failure_mode,
        max_concurrency=max_concurrency,
        blocked_message=blocked_message,
    )


def _load_rails_config(config: WorkerConfig) -> RailsConfig:
    if config.config_path is not None:
        return RailsConfig.from_path(config.config_path)
    assert config.config_yaml is not None
    return RailsConfig.from_content(
        yaml_content=config.config_yaml,
        colang_content=config.colang_content,
    )


def _guardrails_messages(annotated_request: AnnotatedLlmRequest | None) -> list[dict[str, Any]]:
    if not isinstance(annotated_request, dict):
        raise UnsupportedInputError("NeMo Guardrails input policy requires a Relay request codec.")

    messages: list[dict[str, Any]] = []
    instructions = annotated_request.get("instructions")
    if instructions is not None:
        messages.append({"role": "system", "content": _text_content(instructions, "instructions")})

    annotated_messages = annotated_request.get("messages")
    if not isinstance(annotated_messages, list):
        raise UnsupportedInputError("NeMo Guardrails input policy requires an annotated messages array.")
    for index, message in enumerate(annotated_messages):
        if not isinstance(message, dict):
            raise UnsupportedInputError(f"Annotated message {index} must be an object.")
        message_object = cast(dict[str, Json], message)
        role = message_object.get("role")
        if not isinstance(role, str) or role not in _SUPPORTED_MESSAGE_ROLES:
            raise UnsupportedInputError(f"Annotated message {index} uses unsupported role {role!r}.")
        if role == "assistant" and message_object.get("content") is None:
            raise UnsupportedInputError(
                f"Annotated message {index} has no text content; native tool traffic is not supported by this preview."
            )
        content = _text_content(message_object.get("content"), f"message {index} content")
        guardrails_role = "system" if role == "developer" else role
        messages.append({"role": guardrails_role, "content": content})
    return messages


def _text_content(value: Json, field: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        raise UnsupportedInputError(
            f"{field} is composite or multimodal; only plain text is supported by this preview."
        )
    raise UnsupportedInputError(f"{field} must be plain text.")


def _policy_outcome(
    result: Any,
    request: LlmRequest,
    annotated_request: AnnotatedLlmRequest | None,
    config: WorkerConfig,
) -> LlmFinalInputPolicyOutcome:
    status = getattr(result.status, "value", result.status)
    rail = getattr(result, "rail", None)
    evidence = _evidence(str(status), rail=rail)
    if status == RailStatus.PASSED.value:
        return LlmFinalInputPolicyOutcome.allow(evidence=evidence)
    if status == RailStatus.BLOCKED.value:
        return LlmFinalInputPolicyOutcome.reject(
            "nemo_guardrails.input_blocked",
            config.blocked_message,
            evidence=evidence,
        )
    if status == RailStatus.MODIFIED.value:
        content = getattr(result, "content", None)
        if not isinstance(content, str):
            raise RuntimeError("NeMo Guardrails returned a modified result without string content")
        transformed = copy.deepcopy(annotated_request)
        if not isinstance(transformed, dict):
            raise RuntimeError("NeMo Guardrails modified input without an annotated Relay request")
        _replace_last_user_content(transformed, content)
        return LlmFinalInputPolicyOutcome.transform(
            copy.deepcopy(request),
            transformed,
            evidence=evidence,
        )
    raise RuntimeError(f"NeMo Guardrails returned unsupported status {status!r}")


def _replace_last_user_content(annotated_request: AnnotatedLlmRequest, content: str) -> None:
    messages = annotated_request.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            message_object = cast(dict[str, Json], message)
            if message_object.get("role") == "user":
                message_object["content"] = content
                return
    raise RuntimeError("NeMo Guardrails modified input but the request has no user message")


def _evidence(status: str, *, rail: Any | None = None) -> dict[str, Json]:
    evidence: dict[str, Json] = {
        "policy": PLUGIN_ID,
        "engine": "LLMRails",
        "library_version": SUPPORTED_NEMO_GUARDRAILS_VERSION,
        "rail_type": "input",
        "status": status,
    }
    if isinstance(rail, str) and rail:
        evidence["rail"] = rail
    return evidence


def _optional_string(value: Json, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise WorkerConfigurationError(f"{field} must be a string", field=field)
    return value


def _optional_nonempty_string(value: Json, field: str) -> str | None:
    result = _optional_string(value, field)
    if result is not None and not result.strip():
        raise WorkerConfigurationError(f"{field} must not be empty", field=field)
    return result


def _nonempty_string(value: Json, field: str) -> str:
    result = _optional_nonempty_string(value, field)
    if result is None:
        raise WorkerConfigurationError(f"{field} must be a string", field=field)
    return result


def _integer(value: Json, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkerConfigurationError(f"{field} must be an integer", field=field)
    return value


def _positive_integer(value: Json, field: str) -> int:
    result = _integer(value, field)
    if result <= 0:
        raise WorkerConfigurationError(f"{field} must be greater than zero", field=field)
    return result


async def main() -> None:
    """Serve the worker entrypoint declared by relay-plugin.toml."""
    await serve_plugin(NemoGuardrailsWorker())


if __name__ == "__main__":
    asyncio.run(main())
