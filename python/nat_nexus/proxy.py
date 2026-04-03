# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proxy types, backends, and declarative proxy API.

This submodule contains all proxy-related types, storage backends, and the
declarative proxy API for configuring and managing the NexusProxy lifecycle.

Types:
    NexusProxy              - Proxy runtime with online learning and hints injection.
    InMemoryBackend         - In-memory storage backend (default).
    RedisBackend            - Redis-backed storage backend.
    MetadataEnvelope        - Envelope wrapping trie/accumulator metadata.
    ParallelHint            - Hint for parallel execution opportunities.
    AgentHints              - Aggregated hints injected into LLM requests.
    PredictionMetrics       - Statistical metrics for LLM call predictions.
    LlmCallPrediction       - Predicted LLM call statistics.
    SensitivityConfig       - Weights for latency sensitivity scoring.
    StorageBackendProtocol  - Protocol for custom storage backends.

Functions:
    set_latency_sensitivity(value)  - Set latency sensitivity on the current scope.
    set_use_proxy(enabled)          - Buffer proxy enable/disable intent.
    set_proxy_backend(backend)      - Buffer backend choice.
    set_proxy_sensitivity(config)   - Buffer sensitivity configuration.
    set_dynamo_intercept(enabled)   - Enable/disable DynamoIntercept.
    ensure_proxy()                  - Materialize proxy from buffered settings (async).
    teardown_proxy()                - Tear down the active proxy.
    proxy_active()                  - Check whether a proxy is currently active.

Example::

    from nat_nexus.proxy import NexusProxy, InMemoryBackend, set_use_proxy, ensure_proxy

    # Declarative API
    set_use_proxy(True)
    await ensure_proxy()

    # Or explicit builder
    proxy = NexusProxy("my-agent", InMemoryBackend())
    await proxy.register()
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from nat_nexus._native import (
    AgentHints,
    InMemoryBackend,
    LlmCallPrediction,
    MetadataEnvelope,
    NexusProxy,
    ParallelHint,
    PredictionMetrics,
    RedisBackend,
    SensitivityConfig,
    ensure_proxy,
    proxy_active,
    set_dynamo_intercept,
    set_latency_sensitivity,
    set_proxy_backend,
    set_proxy_sensitivity,
    set_use_proxy,
    teardown_proxy,
)


@runtime_checkable
class StorageBackendProtocol(Protocol):
    """Protocol for custom storage backends.

    All methods are async. Implement this protocol and pass an instance
    to ``NexusProxy()`` or ``set_proxy_backend()`` as the backend argument.

    Data is passed as plain dicts (JSON-serializable) across the boundary.
    """

    async def store_run(self, record: dict) -> None: ...

    async def load_plan(self, agent_id: str) -> dict | None: ...

    async def list_runs(self, agent_id: str) -> list[dict]: ...

    async def store_trie(self, agent_id: str, envelope: dict) -> None: ...

    async def load_trie(self, agent_id: str) -> dict | None: ...

    async def store_accumulators(self, agent_id: str, state: dict) -> None: ...

    async def load_accumulators(self, agent_id: str) -> dict | None: ...


__all__ = [
    # Types
    "NexusProxy",
    "InMemoryBackend",
    "RedisBackend",
    "MetadataEnvelope",
    "ParallelHint",
    "AgentHints",
    "PredictionMetrics",
    "LlmCallPrediction",
    "SensitivityConfig",
    "StorageBackendProtocol",
    # Functions
    "set_latency_sensitivity",
    "set_use_proxy",
    "set_proxy_backend",
    "set_proxy_sensitivity",
    "set_dynamo_intercept",
    "ensure_proxy",
    "teardown_proxy",
    "proxy_active",
]
