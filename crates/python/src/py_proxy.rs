// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python-facing proxy type wrappers for Nexus proxy types.
//!
//! Exposes [`PyNexusProxy`], [`PyInMemoryBackend`], [`PyMetadataEnvelope`],
//! [`PyParallelHint`], [`PyAgentHints`], [`PyPredictionMetrics`],
//! [`PyLlmCallPrediction`], [`PySensitivityConfig`], and [`PyRedisBackend`]
//! as `#[pyclass]` types. `NexusProxy` is monomorphized on `AnyBackend`
//! enabling both `InMemoryBackend` and `RedisBackend` from Python.

use std::sync::Arc;

use pyo3::prelude::*;

use nvidia_nat_nexus_proxy::trie::{LlmCallPrediction, PredictionMetrics, SensitivityConfig};
use nvidia_nat_nexus_proxy::AgentHints;
use nvidia_nat_nexus_proxy::RedisBackend;
use nvidia_nat_nexus_proxy::{
    AnyBackend, ExecutionPlan, InMemoryBackend, LatencySensitivityLearner, MetadataEnvelope,
    NexusProxy, ParallelHint,
};

use crate::convert::{json_to_py, py_to_json};
use crate::py_storage::PyStorageBackend;

// ---------------------------------------------------------------------------
// ParallelHint (read-only data wrapper)
// ---------------------------------------------------------------------------

/// Annotates a tool with a parallel execution group.
///
/// Properties:
///     tool_name (str): Name of the tool this hint applies to.
///     group_id (str): Identifier of the parallel group.
///     explicit (bool): Whether this hint was explicitly annotated or learned.
#[pyclass(name = "ParallelHint", from_py_object)]
#[derive(Clone)]
pub struct PyParallelHint {
    pub(crate) inner: ParallelHint,
}

#[pymethods]
impl PyParallelHint {
    #[getter]
    fn tool_name(&self) -> String {
        self.inner.tool_name.clone()
    }

    #[getter]
    fn group_id(&self) -> String {
        self.inner.group_id.clone()
    }

    #[getter]
    fn explicit(&self) -> bool {
        self.inner.explicit
    }

    fn __repr__(&self) -> String {
        format!(
            "ParallelHint(tool_name='{}', group_id='{}')",
            self.inner.tool_name, self.inner.group_id
        )
    }
}

// ---------------------------------------------------------------------------
// MetadataEnvelope (read-only data wrapper)
// ---------------------------------------------------------------------------

/// Per-request metadata injected by the LLM request intercept.
///
/// Properties:
///     run_id (str): Unique identifier for the current run (UUID as string).
///     agent_id (str): Identifier of the agent that owns this run.
///     parallel_hints (list[ParallelHint]): Parallel execution hints.
///     extensions (Any): Open-ended extensions map (JSON-serializable).
#[pyclass(name = "MetadataEnvelope", from_py_object)]
#[derive(Clone)]
pub struct PyMetadataEnvelope {
    pub(crate) inner: MetadataEnvelope,
}

#[pymethods]
impl PyMetadataEnvelope {
    #[getter]
    fn run_id(&self) -> String {
        self.inner.run_id.to_string()
    }

    #[getter]
    fn agent_id(&self) -> String {
        self.inner.agent_id.clone()
    }

    #[getter]
    fn parallel_hints(&self) -> Vec<PyParallelHint> {
        self.inner
            .parallel_hints
            .iter()
            .map(|h| PyParallelHint { inner: h.clone() })
            .collect()
    }

    #[getter]
    fn extensions(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &self.inner.extensions)
    }

    fn __repr__(&self) -> String {
        format!(
            "MetadataEnvelope(run_id='{}', agent_id='{}')",
            self.inner.run_id, self.inner.agent_id
        )
    }
}

// ---------------------------------------------------------------------------
// InMemoryBackend (constructable wrapper)
// ---------------------------------------------------------------------------

/// An in-memory storage backend for testing and single-process use.
///
/// Create an instance and pass it to ``NexusProxy()`` as the backend argument.
/// The proxy constructs its own internal backend; this object signals intent.
#[pyclass(name = "InMemoryBackend")]
pub struct PyInMemoryBackend {
    #[allow(dead_code)]
    pub(crate) inner: InMemoryBackend,
}

#[pymethods]
impl PyInMemoryBackend {
    #[new]
    fn new() -> Self {
        Self {
            inner: InMemoryBackend::new(),
        }
    }

    fn __repr__(&self) -> String {
        "<InMemoryBackend>".to_string()
    }
}

// ---------------------------------------------------------------------------
// AgentHints (read-only data wrapper)
// ---------------------------------------------------------------------------

/// Typed agent hints injected into LLM request headers.
///
/// Properties:
///     osl (int): Output Sequence Length (tokens).
///     iat (int): Inter-Arrival Time (ms).
///     priority (int): Engine scheduler priority.
///     latency_sensitivity (float): Sensitivity score.
///     prefix_id (str): KV cache prefix identity.
///     total_requests (int): Expected total requests.
#[pyclass(name = "AgentHints", from_py_object)]
#[derive(Clone)]
pub struct PyAgentHints {
    pub(crate) inner: AgentHints,
}

#[pymethods]
impl PyAgentHints {
    #[getter]
    fn osl(&self) -> u32 {
        self.inner.osl
    }

    #[getter]
    fn iat(&self) -> u32 {
        self.inner.iat
    }

    #[getter]
    fn priority(&self) -> i32 {
        self.inner.priority
    }

    #[getter]
    fn latency_sensitivity(&self) -> f64 {
        self.inner.latency_sensitivity
    }

    #[getter]
    fn prefix_id(&self) -> String {
        self.inner.prefix_id.clone()
    }

    #[getter]
    fn total_requests(&self) -> u32 {
        self.inner.total_requests
    }

    fn __repr__(&self) -> String {
        format!(
            "AgentHints(osl={}, iat={}, priority={}, latency_sensitivity={}, prefix_id='{}', total_requests={})",
            self.inner.osl, self.inner.iat, self.inner.priority,
            self.inner.latency_sensitivity, self.inner.prefix_id, self.inner.total_requests
        )
    }
}

// ---------------------------------------------------------------------------
// PredictionMetrics (read-only data wrapper)
// ---------------------------------------------------------------------------

/// Aggregated statistics for a single metric from profiler data.
///
/// Properties:
///     sample_count (int): Number of samples.
///     mean (float): Mean value.
///     p50 (float): 50th percentile (median).
///     p90 (float): 90th percentile.
///     p95 (float): 95th percentile.
#[pyclass(name = "PredictionMetrics", from_py_object)]
#[derive(Clone)]
pub struct PyPredictionMetrics {
    pub(crate) inner: PredictionMetrics,
}

#[pymethods]
impl PyPredictionMetrics {
    #[getter]
    fn sample_count(&self) -> u32 {
        self.inner.sample_count
    }

    #[getter]
    fn mean(&self) -> f64 {
        self.inner.mean
    }

    #[getter]
    fn p50(&self) -> f64 {
        self.inner.p50
    }

    #[getter]
    fn p90(&self) -> f64 {
        self.inner.p90
    }

    #[getter]
    fn p95(&self) -> f64 {
        self.inner.p95
    }

    fn __repr__(&self) -> String {
        format!(
            "PredictionMetrics(sample_count={}, mean={:.2}, p50={:.2}, p90={:.2}, p95={:.2})",
            self.inner.sample_count,
            self.inner.mean,
            self.inner.p50,
            self.inner.p90,
            self.inner.p95
        )
    }
}

// ---------------------------------------------------------------------------
// LlmCallPrediction (read-only data wrapper, nested PredictionMetrics)
// ---------------------------------------------------------------------------

/// Predictions for an LLM call at a given position in the call hierarchy.
///
/// Properties:
///     remaining_calls (PredictionMetrics): How many more LLM calls are expected.
///     interarrival_ms (PredictionMetrics): Expected time until next LLM call.
///     output_tokens (PredictionMetrics): Expected output token count.
///     latency_sensitivity (int | None): Auto-computed sensitivity score.
#[pyclass(name = "LlmCallPrediction", from_py_object)]
#[derive(Clone)]
pub struct PyLlmCallPrediction {
    pub(crate) inner: LlmCallPrediction,
}

#[pymethods]
impl PyLlmCallPrediction {
    #[getter]
    fn remaining_calls(&self) -> PyPredictionMetrics {
        PyPredictionMetrics {
            inner: self.inner.remaining_calls.clone(),
        }
    }

    #[getter]
    fn interarrival_ms(&self) -> PyPredictionMetrics {
        PyPredictionMetrics {
            inner: self.inner.interarrival_ms.clone(),
        }
    }

    #[getter]
    fn output_tokens(&self) -> PyPredictionMetrics {
        PyPredictionMetrics {
            inner: self.inner.output_tokens.clone(),
        }
    }

    #[getter]
    fn latency_sensitivity(&self) -> Option<u32> {
        self.inner.latency_sensitivity
    }

    fn __repr__(&self) -> String {
        format!(
            "LlmCallPrediction(latency_sensitivity={:?})",
            self.inner.latency_sensitivity
        )
    }
}

// ---------------------------------------------------------------------------
// SensitivityConfig (constructable with defaults)
// ---------------------------------------------------------------------------

/// Configuration for auto-sensitivity scoring.
///
/// All parameters have defaults matching NAT's trie_builder defaults.
///
/// Args:
///     sensitivity_scale (int): Integer scale for quantized sensitivity (default: 5).
///     w_critical (float): Weight for the critical-path signal (default: 0.5).
///     w_fanout (float): Weight for the fan-out signal (default: 0.3).
///     w_position (float): Weight for the U-shaped position signal (default: 0.2).
///     w_parallel (float): Weight for the parallel-penalty signal (default: 0.0).
#[pyclass(name = "SensitivityConfig", from_py_object)]
#[derive(Clone)]
pub struct PySensitivityConfig {
    pub(crate) inner: SensitivityConfig,
}

#[pymethods]
impl PySensitivityConfig {
    #[new]
    #[pyo3(signature = (*, sensitivity_scale=5, w_critical=0.5, w_fanout=0.3, w_position=0.2, w_parallel=0.0))]
    fn new(
        sensitivity_scale: u32,
        w_critical: f64,
        w_fanout: f64,
        w_position: f64,
        w_parallel: f64,
    ) -> Self {
        Self {
            inner: SensitivityConfig {
                sensitivity_scale,
                w_critical,
                w_fanout,
                w_position,
                w_parallel,
            },
        }
    }

    #[getter]
    fn sensitivity_scale(&self) -> u32 {
        self.inner.sensitivity_scale
    }

    #[getter]
    fn w_critical(&self) -> f64 {
        self.inner.w_critical
    }

    #[getter]
    fn w_fanout(&self) -> f64 {
        self.inner.w_fanout
    }

    #[getter]
    fn w_position(&self) -> f64 {
        self.inner.w_position
    }

    #[getter]
    fn w_parallel(&self) -> f64 {
        self.inner.w_parallel
    }

    fn __repr__(&self) -> String {
        format!(
            "SensitivityConfig(sensitivity_scale={}, w_critical={}, w_fanout={}, w_position={}, w_parallel={})",
            self.inner.sensitivity_scale, self.inner.w_critical, self.inner.w_fanout,
            self.inner.w_position, self.inner.w_parallel
        )
    }
}

// ---------------------------------------------------------------------------
// RedisBackend (async constructor, Option for take semantics)
// ---------------------------------------------------------------------------

/// A Redis-backed storage backend for cross-process shared state.
///
/// Use the async ``connect()`` static method to create an instance, then
/// pass it to ``NexusProxy()`` as the backend argument. The backend is
/// consumed when passed to a proxy and cannot be reused.
///
/// Example::
///
///     backend = await RedisBackend.connect("redis://127.0.0.1:6379", "nexus:")
///     proxy = NexusProxy("my-agent", backend)
#[pyclass(name = "RedisBackend")]
pub struct PyRedisBackend {
    inner: Option<RedisBackend>,
}

#[pymethods]
impl PyRedisBackend {
    /// Connect to Redis and return a new ``RedisBackend``.
    ///
    /// Args:
    ///     url (str): Redis connection URL (e.g. ``redis://127.0.0.1:6379``).
    ///     key_prefix (str): String prepended to every Redis key (e.g. ``"nexus:"``).
    ///
    /// Returns:
    ///     Awaitable[RedisBackend]: A connected Redis backend.
    #[staticmethod]
    fn connect(py: Python<'_>, url: String, key_prefix: String) -> PyResult<Bound<'_, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let backend = RedisBackend::new(&url, key_prefix)
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(PyRedisBackend {
                inner: Some(backend),
            })
        })
    }

    fn __repr__(&self) -> String {
        "<RedisBackend>".to_string()
    }
}

impl PyRedisBackend {
    /// Take the inner RedisBackend, consuming it. Returns error if already taken.
    pub(crate) fn take_inner(&mut self) -> PyResult<RedisBackend> {
        self.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "RedisBackend already consumed (passed to NexusProxy)",
            )
        })
    }
}

// ---------------------------------------------------------------------------
// NexusProxy (monomorphized on AnyBackend)
// ---------------------------------------------------------------------------

/// The central proxy that wires Nexus event subscribers and intercepts to a
/// storage backend.
///
/// Construct with an ``agent_id`` and either an ``InMemoryBackend`` or
/// ``RedisBackend``. Call ``register()`` (an awaitable) to wire the proxy
/// with Nexus, and ``deregister()`` to cleanly remove all registrations.
///
/// Example::
///
///     backend = InMemoryBackend()
///     proxy = NexusProxy("my-agent", backend)
///     await proxy.register()
///     # ... run agent ...
///     proxy.deregister()
///
/// Args:
///     agent_id (str): Identifier for the agent.
///     backend (InMemoryBackend | RedisBackend): Storage backend.
///     llm_intercept_priority (int): Priority for the LLM request intercept (default 100).
///     tool_intercept_priority (int): Priority for the tool execution intercept (default 100).
///     sensitivity_config (SensitivityConfig | None): Optional sensitivity config.
///         When provided, creates a LatencySensitivityLearner for the learner pipeline.
#[pyclass(name = "NexusProxy")]
pub struct PyNexusProxy {
    inner: Arc<tokio::sync::Mutex<Option<NexusProxy>>>,
}

#[pymethods]
impl PyNexusProxy {
    #[new]
    #[pyo3(signature = (agent_id, backend, *, llm_intercept_priority=100, tool_intercept_priority=100, sensitivity_config=None, dynamo_intercept=false))]
    fn new(
        agent_id: String,
        backend: &Bound<'_, PyAny>,
        llm_intercept_priority: i32,
        tool_intercept_priority: i32,
        sensitivity_config: Option<PySensitivityConfig>,
        dynamo_intercept: bool,
    ) -> PyResult<Self> {
        let any_backend = if backend.is_instance_of::<PyInMemoryBackend>() {
            Box::new(InMemoryBackend::new()) as AnyBackend
        } else if backend.is_instance_of::<PyRedisBackend>() {
            let mut py_redis: PyRefMut<'_, PyRedisBackend> = backend.extract()?;
            Box::new(py_redis.take_inner()?) as AnyBackend
        } else if backend.hasattr("store_run")? && backend.hasattr("load_plan")? {
            // Duck-typing: treat as a Python StorageBackendProtocol implementation.
            Box::new(PyStorageBackend::new(backend.clone().unbind())) as AnyBackend
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "backend must be InMemoryBackend, RedisBackend, or implement StorageBackendProtocol \
                 (7 async methods: store_run, load_plan, list_runs, store_trie, load_trie, \
                 store_accumulators, load_accumulators)",
            ));
        };

        let mut builder = NexusProxy::builder()
            .agent_id(agent_id.clone())
            .backend(any_backend)
            .llm_intercept_priority(llm_intercept_priority)
            .tool_intercept_priority(tool_intercept_priority)
            .dynamo_intercept(dynamo_intercept);

        if let Some(sc) = sensitivity_config {
            builder = builder.learner(Box::new(LatencySensitivityLearner::new(agent_id, sc.inner)));
        }

        let proxy = builder
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(tokio::sync::Mutex::new(Some(proxy))),
        })
    }

    /// Register the proxy with the Nexus runtime.
    ///
    /// This is an awaitable that wires the event subscriber and both intercepts
    /// with the Nexus global context.
    ///
    /// Returns:
    ///     None
    fn register<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = inner.lock().await;
            let proxy = guard.as_mut().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("proxy already shut down")
            })?;
            proxy
                .register()
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(())
        })
    }

    /// Deregister the proxy from the Nexus runtime.
    ///
    /// Removes the event subscriber and both intercepts. Safe to call multiple
    /// times.
    fn deregister(&self) -> PyResult<()> {
        let mut guard = self.inner.try_lock().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "proxy is locked by an async operation; try again after await completes",
            )
        })?;
        let proxy = guard
            .as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("proxy already shut down"))?;
        proxy
            .deregister()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }

    /// The agent identifier.
    #[getter]
    fn agent_id(&self) -> PyResult<String> {
        let guard = self.inner.try_lock().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "proxy is locked by an async operation; try again after await completes",
            )
        })?;
        let proxy = guard
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("proxy already shut down"))?;
        Ok(proxy.agent_id().to_string())
    }

    /// Pre-seed the backend with an execution plan so the LLM request intercept
    /// has metadata to inject immediately.
    ///
    /// Args:
    ///     extensions: A JSON-serializable object for the extensions field (default: ``{}``).
    #[pyo3(signature = (extensions=None))]
    fn store_plan(&self, extensions: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let mut guard = self.inner.try_lock().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "proxy is locked by an async operation; try again after await completes",
            )
        })?;
        let proxy = guard
            .as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("proxy already shut down"))?;

        let ext = match extensions {
            Some(obj) if !obj.is_none() => py_to_json(obj)?,
            _ => serde_json::json!({}),
        };

        let plan = ExecutionPlan {
            agent_id: proxy.agent_id().to_string(),
            parallel_groups: vec![],
            metadata_template: MetadataEnvelope {
                run_id: uuid::Uuid::new_v4(),
                agent_id: proxy.agent_id().to_string(),
                parallel_hints: vec![],
                extensions: ext,
            },
        };

        // Store in backend (StorageBackendDyn::store_plan via dyn dispatch)
        proxy
            .backend()
            .store_plan(&plan)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Also update hot cache so intercepts see it immediately
        let mut cache = proxy.hot_cache().write().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("hot cache lock poisoned: {e}"))
        })?;
        cache.plan = Some(plan);

        Ok(())
    }

    fn __repr__(&self) -> String {
        "<NexusProxy>".to_string()
    }
}

// ---------------------------------------------------------------------------
// Declarative proxy API (#[pyfunction] wrappers)
// ---------------------------------------------------------------------------

/// Enable or disable the declarative proxy.
///
/// Buffers the intent without creating a proxy. Call ``ensure_proxy()``
/// to materialize. Setting to ``False`` after ``ensure_proxy()`` will
/// teardown the active proxy.
///
/// Args:
///     enabled (bool): Whether to enable the proxy.
#[pyfunction]
fn set_use_proxy(enabled: bool) -> PyResult<()> {
    nvidia_nat_nexus_proxy::set_use_proxy(enabled);
    Ok(())
}

/// Set the storage backend for the declarative proxy.
///
/// Buffers the backend choice. Applied when ``ensure_proxy()`` is called.
///
/// Args:
///     backend (InMemoryBackend | RedisBackend): An InMemoryBackend or RedisBackend instance.
#[pyfunction]
fn set_proxy_backend(backend: &Bound<'_, PyAny>) -> PyResult<()> {
    let any_backend = if backend.is_instance_of::<PyInMemoryBackend>() {
        Box::new(InMemoryBackend::new()) as AnyBackend
    } else if backend.is_instance_of::<PyRedisBackend>() {
        let mut py_redis: PyRefMut<'_, PyRedisBackend> = backend.extract()?;
        Box::new(py_redis.take_inner()?) as AnyBackend
    } else if backend.hasattr("store_run")? && backend.hasattr("load_plan")? {
        // Duck-typing: treat as a Python StorageBackendProtocol implementation.
        Box::new(PyStorageBackend::new(backend.clone().unbind())) as AnyBackend
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "backend must be InMemoryBackend, RedisBackend, or implement StorageBackendProtocol \
             (7 async methods: store_run, load_plan, list_runs, store_trie, load_trie, \
             store_accumulators, load_accumulators)",
        ));
    };
    nvidia_nat_nexus_proxy::set_proxy_backend(any_backend);
    Ok(())
}

/// Set the sensitivity configuration for the declarative proxy.
///
/// Buffers the configuration. Applied when ``ensure_proxy()`` is called.
///
/// Args:
///     config (SensitivityConfig): A SensitivityConfig instance.
#[pyfunction]
fn set_proxy_sensitivity(config: PySensitivityConfig) -> PyResult<()> {
    nvidia_nat_nexus_proxy::set_proxy_sensitivity(config.inner);
    Ok(())
}

/// Enable or disable DynamoIntercept for the declarative proxy.
///
/// When enabled, AgentHints are injected into LLM request bodies.
///
/// Args:
///     enabled (bool): Whether to enable DynamoIntercept.
#[pyfunction]
fn set_dynamo_intercept(enabled: bool) -> PyResult<()> {
    nvidia_nat_nexus_proxy::set_dynamo_intercept(enabled);
    Ok(())
}

/// Create and register the proxy from buffered configuration.
///
/// Must be awaited. Creates a NexusProxy from the settings configured
/// via ``set_use_proxy``, ``set_proxy_backend``, etc. If the proxy
/// is already active, this is a no-op.
///
/// Returns:
///     Awaitable[None]
#[pyfunction]
fn ensure_proxy(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        nvidia_nat_nexus_proxy::ensure_proxy()
            .await
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    })
}

/// Deregister and remove the active declarative proxy.
///
/// Safe to call when no proxy is active.
#[pyfunction]
fn teardown_proxy() -> PyResult<()> {
    nvidia_nat_nexus_proxy::teardown_proxy()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Check whether the declarative proxy is currently active.
///
/// Returns:
///     bool: True if ensure_proxy() has been called and teardown_proxy() has not.
#[pyfunction]
fn proxy_active() -> bool {
    nvidia_nat_nexus_proxy::proxy_active()
}

// ---------------------------------------------------------------------------
// Scope-level latency sensitivity
// ---------------------------------------------------------------------------

/// Set latency sensitivity on the current (top) scope.
///
/// Uses max-merge semantics: if the scope already has a higher value, the
/// call is a no-op. Does not push a new scope.
///
/// Args:
///     value (int): Positive integer sensitivity value.
///
/// Raises:
///     RuntimeError: If the scope stack is unavailable.
///     ValueError: If value is 0.
#[pyfunction]
fn set_latency_sensitivity(value: u32) -> PyResult<()> {
    if value == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "sensitivity must be positive (> 0)",
        ));
    }
    nvidia_nat_nexus_proxy::set_latency_sensitivity(value)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyParallelHint>()?;
    m.add_class::<PyMetadataEnvelope>()?;
    m.add_class::<PyInMemoryBackend>()?;
    m.add_class::<PyNexusProxy>()?;
    m.add_class::<PyAgentHints>()?;
    m.add_class::<PyPredictionMetrics>()?;
    m.add_class::<PyLlmCallPrediction>()?;
    m.add_class::<PySensitivityConfig>()?;
    m.add_class::<PyRedisBackend>()?;
    // Declarative proxy API
    m.add_function(wrap_pyfunction!(set_use_proxy, m)?)?;
    m.add_function(wrap_pyfunction!(set_proxy_backend, m)?)?;
    m.add_function(wrap_pyfunction!(set_proxy_sensitivity, m)?)?;
    m.add_function(wrap_pyfunction!(set_dynamo_intercept, m)?)?;
    m.add_function(wrap_pyfunction!(ensure_proxy, m)?)?;
    m.add_function(wrap_pyfunction!(teardown_proxy, m)?)?;
    m.add_function(wrap_pyfunction!(proxy_active, m)?)?;
    // Scope-level latency sensitivity
    m.add_function(wrap_pyfunction!(set_latency_sensitivity, m)?)?;
    Ok(())
}

#[cfg(test)]
#[path = "py_proxy_coverage_tests.rs"]
mod coverage_tests;
