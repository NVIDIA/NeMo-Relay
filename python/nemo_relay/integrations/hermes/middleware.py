# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-execution validation middleware for Hermes Agent tool and LLM calls."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import nemo_relay

_logger = logging.getLogger(__name__)

_SECRET_KEYS = frozenset({"api_key", "token", "secret", "password", "authorization", "bearer"})
_REDACTED = "***REDACTED***"


def _redact_secrets(args: Mapping[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of *args* with secret-looking values redacted."""
    redacted: dict[str, Any] = {}
    for key, value in args.items():
        if key.lower() in _SECRET_KEYS:
            redacted[key] = _REDACTED
        else:
            redacted[key] = value
    return redacted


def _payload_size(payload: Mapping[str, Any] | None) -> int:
    """Rough byte-size estimate of a JSON-serialisable mapping."""
    if payload is None:
        return 0
    import json
    try:
        return len(json.dumps(payload, default=str).encode())
    except Exception:
        return 0


class NemoRelayHermesMiddleware:
    """Middleware that guards Hermes Agent tool and LLM calls.

    Features:
    * **Secret redaction** -- sanitises tool arguments before they reach
      the observability pipeline.
    * **Tool blocking** -- denies invocations of tools in a configurable
      block-list.
    * **Payload size guard** -- rejects LLM requests whose payload
      exceeds a configurable byte threshold.
    """

    def __init__(
        self,
        agent_name: str = "hermes-agent",
        blocked_tools: Sequence[str] | None = None,
        max_tool_args_size: int = 65536,
    ) -> None:
        self._agent_name = agent_name
        self._blocked_tools = frozenset(blocked_tools or ())
        self._max_tool_args_size = max_tool_args_size
        self._registered = False

    # -- registration ----------------------------------------------------------

    def register(self) -> None:
        """Register all guardrails and intercepts with the NeMo Relay runtime."""
        if self._registered:
            return
        self._registered = True

        # Tool argument sanitisation
        try:
            nemo_relay.guardrails.register_tool_sanitize_request(
                f"{self._agent_name}-redact",
                10,
                self._sanitize_tool_args,
            )
        except Exception:
            _logger.debug("Failed to register tool sanitiser", exc_info=True)

        # Tool blocking
        if self._blocked_tools:
            try:
                nemo_relay.guardrails.register_tool_sanitize_request(
                    f"{self._agent_name}-block",
                    5,
                    self._block_denied_tools,
                )
            except Exception:
                _logger.debug("Failed to register tool blocker", exc_info=True)

        # LLM payload size guard
        try:
            nemo_relay.intercepts.register_llm_request(
                f"{self._agent_name}-size-guard",
                10,
                False,
                self._guard_llm_payload_size,
            )
        except Exception:
            _logger.debug("Failed to register LLM size guard", exc_info=True)

    # -- guard callbacks -------------------------------------------------------

    def _sanitize_tool_args(
        self, tool_name: str, args: Mapping[str, Any]
    ) -> dict[str, Any]:
        return _redact_secrets(args)

    def _block_denied_tools(
        self, tool_name: str, args: Mapping[str, Any]
    ) -> dict[str, Any]:
        if tool_name in self._blocked_tools:
            raise ValueError(f"Tool {tool_name!r} is blocked by Hermes middleware policy")
        return dict(args)

    def _guard_llm_payload_size(
        self,
        name: str,
        request: nemo_relay.LLMRequest,
        annotated: nemo_relay.AnnotatedLLMRequest | None,
    ) -> nemo_relay.LLMRequestInterceptOutcome:
        size = _payload_size(request.content)
        if size > self._max_tool_args_size:
            raise ValueError(
                f"LLM payload ({size} bytes) exceeds configured limit"
                f" ({self._max_tool_args_size} bytes)"
            )
        return nemo_relay.LLMRequestInterceptOutcome(request, annotated)

    # -- properties ------------------------------------------------------------

    @property
    def agent_name(self) -> str:
        return self._agent_name

    @property
    def blocked_tools(self) -> frozenset[str]:
        return self._blocked_tools

    @property
    def max_tool_args_size(self) -> int:
        return self._max_tool_args_size
