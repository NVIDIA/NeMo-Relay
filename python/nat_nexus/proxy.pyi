# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Type stubs for nat_nexus.proxy.

Provides static type information for the proxy submodule, including
proxy types, storage backends, and the declarative proxy API.
"""

from typing import Any, Optional, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Proxy types
# ---------------------------------------------------------------------------

class ParallelHint:
    """A parallel execution hint annotating a tool with a group.

    When ``explicit`` is ``True``, the hint was user-annotated;
    when ``False``, it was learned from historical data.
    """

    @property
    def tool_name(self) -> str:
        """Name of the tool this hint applies to."""
        ...
    @property
    def group_id(self) -> str:
        """Identifier of the parallel group."""
        ...
    @property
    def explicit(self) -> bool:
        """Whether the hint was explicitly annotated (vs learned)."""
        ...

class MetadataEnvelope:
    """Per-request metadata injected by the proxy's LLM request intercept.

    Carries run identity, agent identity, parallelism hints, and
    an open-ended extensions map.
    """

    @property
    def run_id(self) -> str:
        """Unique run identifier (UUID as hyphenated string)."""
        ...
    @property
    def agent_id(self) -> str:
        """Identifier of the agent that owns this run."""
        ...
    @property
    def parallel_hints(self) -> list[ParallelHint]:
        """Parallel execution hints attached to this request."""
        ...
    @property
    def extensions(self) -> Any:
        """Open-ended extensions map (JSON-serializable)."""
        ...

class InMemoryBackend:
    """An in-memory storage backend for testing and single-process use.

    Example::

        backend = InMemoryBackend()
        proxy = NexusProxy("my-agent", backend)
    """

    def __init__(self) -> None:
        """Create a new empty in-memory backend."""
        ...

class NexusProxy:
    """A stateful proxy that wires Nexus event subscribers and intercepts
    to a storage backend for telemetry capture and metadata injection.

    Example::

        proxy = NexusProxy("my-agent", InMemoryBackend())
        await proxy.register()
        # ... run agent workload ...
        proxy.deregister()
    """

    def __init__(
        self,
        agent_id: str,
        backend: InMemoryBackend | RedisBackend | StorageBackendProtocol,
        *,
        llm_intercept_priority: int = 100,
        tool_intercept_priority: int = 100,
        sensitivity_config: Optional[SensitivityConfig] = None,
        dynamo_intercept: bool = False,
    ) -> None:
        """Create a new proxy for the given agent.

        Args:
            agent_id: Identifier of the agent.
            backend: Storage backend (InMemoryBackend, RedisBackend, or any object
                implementing StorageBackendProtocol).
            llm_intercept_priority: Priority for the LLM request intercept (default 100).
            tool_intercept_priority: Priority for the tool execution intercept (default 100).
            sensitivity_config: Optional sensitivity configuration. When provided,
                a LatencySensitivityLearner is created and added to the learner pipeline.
            dynamo_intercept: Enable DynamoIntercept for AgentHints injection (default False).
        """
        ...
    async def register(self) -> None:
        """Wire the proxy's subscriber and intercepts with the Nexus runtime.

        Must be awaited. Spawns a background drain task for async telemetry.
        """
        ...
    def deregister(self) -> None:
        """Remove the proxy's subscriber and intercepts from the Nexus runtime.

        Safe to call multiple times.
        """
        ...
    def store_plan(self, extensions: Optional[Any] = None) -> None:
        """Seed the hot cache with a minimal execution plan for testing.

        Args:
            extensions: Optional JSON-serializable extensions to include.
        """
        ...
    @property
    def agent_id(self) -> str:
        """The agent identifier this proxy is associated with."""
        ...

# ---------------------------------------------------------------------------
# Proxy v2 types
# ---------------------------------------------------------------------------

class AgentHints:
    """Typed agent hints injected into LLM request headers by the proxy.

    Read-only. Created internally by the proxy's LLM intercept from
    prediction trie data.
    """

    @property
    def osl(self) -> int:
        """Output Sequence Length (tokens), from prediction output_tokens p90."""
        ...
    @property
    def iat(self) -> int:
        """Inter-Arrival Time (ms), from prediction interarrival_ms mean."""
        ...
    @property
    def priority(self) -> int:
        """Engine scheduler priority (sensitivity_scale - latency_sensitivity)."""
        ...
    @property
    def latency_sensitivity(self) -> float:
        """Sensitivity score (float for Dynamo compatibility)."""
        ...
    @property
    def prefix_id(self) -> str:
        """KV cache prefix identity."""
        ...
    @property
    def total_requests(self) -> int:
        """Expected total requests (remaining_calls mean + call_index)."""
        ...

class PredictionMetrics:
    """Aggregated statistics for a single prediction metric.

    Read-only. Contains sample count, mean, and percentile values
    computed from streaming accumulators.
    """

    @property
    def sample_count(self) -> int:
        """Number of samples in the accumulator."""
        ...
    @property
    def mean(self) -> float:
        """Mean value."""
        ...
    @property
    def p50(self) -> float:
        """50th percentile (median)."""
        ...
    @property
    def p90(self) -> float:
        """90th percentile."""
        ...
    @property
    def p95(self) -> float:
        """95th percentile."""
        ...

class LlmCallPrediction:
    """Predictions for an LLM call at a given position in the call hierarchy.

    Read-only. Contains prediction metrics for remaining calls,
    inter-arrival time, output tokens, and an optional latency
    sensitivity score.
    """

    @property
    def remaining_calls(self) -> PredictionMetrics:
        """Expected remaining LLM calls after this one."""
        ...
    @property
    def interarrival_ms(self) -> PredictionMetrics:
        """Expected time in ms until the next LLM call."""
        ...
    @property
    def output_tokens(self) -> PredictionMetrics:
        """Expected output token count for this call."""
        ...
    @property
    def latency_sensitivity(self) -> Optional[int]:
        """Auto-computed latency sensitivity score, or ``None``."""
        ...

class SensitivityConfig:
    """Configuration for auto-sensitivity scoring in the learner pipeline.

    Constructable from Python. Pass to ``NexusProxy`` as ``sensitivity_config``
    to enable the latency sensitivity learner.

    Example::

        config = SensitivityConfig(w_critical=0.7, w_fanout=0.3)
        proxy = NexusProxy("my-agent", backend, sensitivity_config=config)
    """

    def __init__(
        self,
        *,
        sensitivity_scale: int = 5,
        w_critical: float = 0.5,
        w_fanout: float = 0.3,
        w_position: float = 0.2,
        w_parallel: float = 0.0,
    ) -> None:
        """Create a sensitivity configuration.

        Args:
            sensitivity_scale: Integer scale for quantized sensitivity (1..=scale). Default 5.
            w_critical: Weight for the critical-path signal. Default 0.5.
            w_fanout: Weight for the fan-out signal. Default 0.3.
            w_position: Weight for the U-shaped position signal. Default 0.2.
            w_parallel: Weight for the parallel-penalty signal. Default 0.0.
        """
        ...
    @property
    def sensitivity_scale(self) -> int:
        """Integer scale for quantized sensitivity."""
        ...
    @property
    def w_critical(self) -> float:
        """Weight for the critical-path signal."""
        ...
    @property
    def w_fanout(self) -> float:
        """Weight for the fan-out signal."""
        ...
    @property
    def w_position(self) -> float:
        """Weight for the U-shaped position signal."""
        ...
    @property
    def w_parallel(self) -> float:
        """Weight for the parallel-penalty signal."""
        ...

class RedisBackend:
    """A Redis-backed storage backend for cross-process shared state.

    Constructed asynchronously via the ``connect()`` static method.
    Once created, pass to ``NexusProxy`` as the backend argument.

    .. note::
        A ``RedisBackend`` can only be passed to one ``NexusProxy``.
        Attempting to reuse it raises ``RuntimeError``.

    Example::

        backend = await RedisBackend.connect("redis://127.0.0.1:6379", "nexus:")
        proxy = NexusProxy("my-agent", backend)
    """

    @staticmethod
    async def connect(url: str, key_prefix: str) -> RedisBackend:
        """Connect to Redis and return a new backend.

        Args:
            url: Redis connection URL (e.g. ``redis://127.0.0.1:6379``).
            key_prefix: String prepended to every Redis key (e.g. ``nexus:``).

        Raises:
            RuntimeError: If the connection cannot be established.
        """
        ...

@runtime_checkable
class StorageBackendProtocol(Protocol):
    """Protocol for custom storage backends.

    All methods are async. Implement this protocol and pass an instance
    to ``NexusProxy()`` or ``set_proxy_backend()`` as the backend argument.

    Data is passed as plain dicts (JSON-serializable) across the boundary.
    """

    async def store_run(self, record: dict[str, Any]) -> None: ...
    async def load_plan(self, agent_id: str) -> dict[str, Any] | None: ...
    async def list_runs(self, agent_id: str) -> list[dict[str, Any]]: ...
    async def store_trie(self, agent_id: str, envelope: dict[str, Any]) -> None: ...
    async def load_trie(self, agent_id: str) -> dict[str, Any] | None: ...
    async def store_accumulators(self, agent_id: str, state: dict[str, Any]) -> None: ...
    async def load_accumulators(self, agent_id: str) -> dict[str, Any] | None: ...

# ---------------------------------------------------------------------------
# Declarative proxy API
# ---------------------------------------------------------------------------

def set_use_proxy(enabled: bool) -> None:
    """Enable or disable the declarative proxy.

    Buffers the intent without creating a proxy. Call ``ensure_proxy()``
    to materialize. Setting to ``False`` after ``ensure_proxy()`` will
    teardown the active proxy.

    Args:
        enabled: Whether to enable the proxy.
    """
    ...

def set_proxy_backend(backend: InMemoryBackend | RedisBackend | StorageBackendProtocol) -> None:
    """Set the storage backend for the declarative proxy.

    Buffers the backend choice. Applied when ``ensure_proxy()`` is called.

    Args:
        backend: An InMemoryBackend, RedisBackend, or StorageBackendProtocol instance.
    """
    ...

def set_proxy_sensitivity(config: SensitivityConfig) -> None:
    """Set the sensitivity configuration for the declarative proxy.

    Buffers the configuration. Applied when ``ensure_proxy()`` is called.

    Args:
        config: A SensitivityConfig instance.
    """
    ...

def set_dynamo_intercept(enabled: bool) -> None:
    """Enable or disable DynamoIntercept for the declarative proxy.

    When enabled, AgentHints are injected into LLM request bodies.

    Args:
        enabled: Whether to enable DynamoIntercept.
    """
    ...

async def ensure_proxy() -> None:
    """Create and register the proxy from buffered configuration.

    Must be awaited. Creates a NexusProxy from the settings configured
    via ``set_use_proxy``, ``set_proxy_backend``, etc. If the proxy
    is already active, this is a no-op.

    If called without prior ``set_use_proxy(True)``, implicitly enables.
    Defaults: InMemoryBackend, NAT-matching sensitivity weights.
    """
    ...

def teardown_proxy() -> None:
    """Deregister and remove the active declarative proxy.

    Safe to call when no proxy is active.
    """
    ...

def proxy_active() -> bool:
    """Check whether the declarative proxy is currently active.

    Returns:
        True if ensure_proxy() has been called and teardown_proxy() has not.
    """
    ...

# ---------------------------------------------------------------------------
# Scope-level latency sensitivity
# ---------------------------------------------------------------------------

def set_latency_sensitivity(value: int) -> None:
    """Set latency sensitivity on the current (top) scope.

    Uses max-merge semantics: if the scope already has a higher value,
    this call is a no-op. Does not push a new scope.

    Args:
        value: Positive integer sensitivity value.

    Raises:
        RuntimeError: If the scope stack is unavailable.
        ValueError: If value is 0.
    """
    ...
