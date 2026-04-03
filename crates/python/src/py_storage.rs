// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bridge from Python storage backend objects to Rust `StorageBackendDyn`.
//!
//! `PyStorageBackend` wraps a Python object implementing the
//! `StorageBackendProtocol` (7 async methods) and implements
//! `StorageBackendDyn` by acquiring the GIL and calling the
//! corresponding Python method for each operation.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use pyo3::prelude::*;

use nvidia_nat_nexus_proxy::error::{ProxyError, Result};
use nvidia_nat_nexus_proxy::storage::StorageBackendDyn;
use nvidia_nat_nexus_proxy::trie::serialization::TrieEnvelope;
use nvidia_nat_nexus_proxy::trie::AccumulatorState;
use nvidia_nat_nexus_proxy::types::{ExecutionPlan, RunRecord};

use crate::convert::{json_to_py, py_to_json};

/// Wraps a Python object implementing `StorageBackendProtocol` and bridges
/// it to the Rust `StorageBackendDyn` trait. Each method acquires the GIL,
/// serializes Rust types to Python dicts via JSON, calls the Python method,
/// converts the resulting coroutine to a Rust future, and deserializes
/// the result back.
pub struct PyStorageBackend {
    /// Arc-wrapped so we can cheaply clone into each async block without
    /// needing the GIL for `Py::clone_ref`.
    inner: Arc<Py<PyAny>>,
}

// `Arc<Py<PyAny>>` is Send + Sync. Py<PyAny> is Send. All access goes
// through the GIL so Sync is safe.
unsafe impl Send for PyStorageBackend {}
unsafe impl Sync for PyStorageBackend {}

impl PyStorageBackend {
    /// Create a new `PyStorageBackend` wrapping the given Python object.
    pub fn new(obj: Py<PyAny>) -> Self {
        Self {
            inner: Arc::new(obj),
        }
    }
}

impl StorageBackendDyn for PyStorageBackend {
    fn store_run_dyn<'a>(
        &'a self,
        record: &'a RunRecord,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        let inner = self.inner.clone();
        let record_json = serde_json::to_value(record)
            .map_err(|e| ProxyError::Internal(format!("serialize RunRecord: {e}")));
        Box::pin(async move {
            let record_json = record_json?;
            let fut = Python::attach(|py| -> std::result::Result<_, ProxyError> {
                let dict = json_to_py(py, &record_json)
                    .map_err(|e| ProxyError::Internal(format!("json_to_py: {e}")))?;
                let coro = inner
                    .call_method1(py, "store_run", (dict,))
                    .map_err(|e| ProxyError::Internal(format!("call store_run: {e}")))?;
                pyo3_async_runtimes::tokio::into_future(coro.into_bound(py))
                    .map_err(|e| ProxyError::Internal(format!("into_future: {e}")))
            })?;
            fut.await
                .map_err(|e| ProxyError::Internal(format!("Python store_run: {e}")))?;
            Ok(())
        })
    }

    fn load_plan_dyn<'a>(
        &'a self,
        agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<ExecutionPlan>>> + Send + 'a>> {
        let inner = self.inner.clone();
        let agent_id = agent_id.to_string();
        Box::pin(async move {
            let fut = Python::attach(|py| -> std::result::Result<_, ProxyError> {
                let coro = inner
                    .call_method1(py, "load_plan", (agent_id.as_str(),))
                    .map_err(|e| ProxyError::Internal(format!("call load_plan: {e}")))?;
                pyo3_async_runtimes::tokio::into_future(coro.into_bound(py))
                    .map_err(|e| ProxyError::Internal(format!("into_future: {e}")))
            })?;
            let result = fut
                .await
                .map_err(|e| ProxyError::Internal(format!("Python load_plan: {e}")))?;
            Python::attach(|py| {
                let obj = result.bind(py);
                if obj.is_none() {
                    return Ok(None);
                }
                let json_val = py_to_json(obj)
                    .map_err(|e| ProxyError::Internal(format!("py_to_json: {e}")))?;
                let plan: ExecutionPlan = serde_json::from_value(json_val)
                    .map_err(|e| ProxyError::Internal(format!("deserialize ExecutionPlan: {e}")))?;
                Ok(Some(plan))
            })
        })
    }

    fn list_runs_dyn<'a>(
        &'a self,
        agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<RunRecord>>> + Send + 'a>> {
        let inner = self.inner.clone();
        let agent_id = agent_id.to_string();
        Box::pin(async move {
            let fut = Python::attach(|py| -> std::result::Result<_, ProxyError> {
                let coro = inner
                    .call_method1(py, "list_runs", (agent_id.as_str(),))
                    .map_err(|e| ProxyError::Internal(format!("call list_runs: {e}")))?;
                pyo3_async_runtimes::tokio::into_future(coro.into_bound(py))
                    .map_err(|e| ProxyError::Internal(format!("into_future: {e}")))
            })?;
            let result = fut
                .await
                .map_err(|e| ProxyError::Internal(format!("Python list_runs: {e}")))?;
            Python::attach(|py| {
                let obj = result.bind(py);
                let json_val = py_to_json(obj)
                    .map_err(|e| ProxyError::Internal(format!("py_to_json: {e}")))?;
                let runs: Vec<RunRecord> = serde_json::from_value(json_val).map_err(|e| {
                    ProxyError::Internal(format!("deserialize Vec<RunRecord>: {e}"))
                })?;
                Ok(runs)
            })
        })
    }

    fn store_trie<'a>(
        &'a self,
        agent_id: &'a str,
        envelope: &'a TrieEnvelope,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        let inner = self.inner.clone();
        let agent_id = agent_id.to_string();
        let envelope_json = serde_json::to_value(envelope)
            .map_err(|e| ProxyError::Internal(format!("serialize TrieEnvelope: {e}")));
        Box::pin(async move {
            let envelope_json = envelope_json?;
            let fut = Python::attach(|py| -> std::result::Result<_, ProxyError> {
                let dict = json_to_py(py, &envelope_json)
                    .map_err(|e| ProxyError::Internal(format!("json_to_py: {e}")))?;
                let coro = inner
                    .call_method1(py, "store_trie", (agent_id.as_str(), dict))
                    .map_err(|e| ProxyError::Internal(format!("call store_trie: {e}")))?;
                pyo3_async_runtimes::tokio::into_future(coro.into_bound(py))
                    .map_err(|e| ProxyError::Internal(format!("into_future: {e}")))
            })?;
            fut.await
                .map_err(|e| ProxyError::Internal(format!("Python store_trie: {e}")))?;
            Ok(())
        })
    }

    fn load_trie<'a>(
        &'a self,
        agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<TrieEnvelope>>> + Send + 'a>> {
        let inner = self.inner.clone();
        let agent_id = agent_id.to_string();
        Box::pin(async move {
            let fut = Python::attach(|py| -> std::result::Result<_, ProxyError> {
                let coro = inner
                    .call_method1(py, "load_trie", (agent_id.as_str(),))
                    .map_err(|e| ProxyError::Internal(format!("call load_trie: {e}")))?;
                pyo3_async_runtimes::tokio::into_future(coro.into_bound(py))
                    .map_err(|e| ProxyError::Internal(format!("into_future: {e}")))
            })?;
            let result = fut
                .await
                .map_err(|e| ProxyError::Internal(format!("Python load_trie: {e}")))?;
            Python::attach(|py| {
                let obj = result.bind(py);
                if obj.is_none() {
                    return Ok(None);
                }
                let json_val = py_to_json(obj)
                    .map_err(|e| ProxyError::Internal(format!("py_to_json: {e}")))?;
                let envelope: TrieEnvelope = serde_json::from_value(json_val)
                    .map_err(|e| ProxyError::Internal(format!("deserialize TrieEnvelope: {e}")))?;
                Ok(Some(envelope))
            })
        })
    }

    fn store_accumulators<'a>(
        &'a self,
        agent_id: &'a str,
        state: &'a AccumulatorState,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        let inner = self.inner.clone();
        let agent_id = agent_id.to_string();
        let state_json = serde_json::to_value(state)
            .map_err(|e| ProxyError::Internal(format!("serialize AccumulatorState: {e}")));
        Box::pin(async move {
            let state_json = state_json?;
            let fut = Python::attach(|py| -> std::result::Result<_, ProxyError> {
                let dict = json_to_py(py, &state_json)
                    .map_err(|e| ProxyError::Internal(format!("json_to_py: {e}")))?;
                let coro = inner
                    .call_method1(py, "store_accumulators", (agent_id.as_str(), dict))
                    .map_err(|e| ProxyError::Internal(format!("call store_accumulators: {e}")))?;
                pyo3_async_runtimes::tokio::into_future(coro.into_bound(py))
                    .map_err(|e| ProxyError::Internal(format!("into_future: {e}")))
            })?;
            fut.await
                .map_err(|e| ProxyError::Internal(format!("Python store_accumulators: {e}")))?;
            Ok(())
        })
    }

    fn load_accumulators<'a>(
        &'a self,
        agent_id: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<AccumulatorState>>> + Send + 'a>> {
        let inner = self.inner.clone();
        let agent_id = agent_id.to_string();
        Box::pin(async move {
            let fut = Python::attach(|py| -> std::result::Result<_, ProxyError> {
                let coro = inner
                    .call_method1(py, "load_accumulators", (agent_id.as_str(),))
                    .map_err(|e| ProxyError::Internal(format!("call load_accumulators: {e}")))?;
                pyo3_async_runtimes::tokio::into_future(coro.into_bound(py))
                    .map_err(|e| ProxyError::Internal(format!("into_future: {e}")))
            })?;
            let result = fut
                .await
                .map_err(|e| ProxyError::Internal(format!("Python load_accumulators: {e}")))?;
            Python::attach(|py| {
                let obj = result.bind(py);
                if obj.is_none() {
                    return Ok(None);
                }
                let json_val = py_to_json(obj)
                    .map_err(|e| ProxyError::Internal(format!("py_to_json: {e}")))?;
                let state: AccumulatorState = serde_json::from_value(json_val).map_err(|e| {
                    ProxyError::Internal(format!("deserialize AccumulatorState: {e}"))
                })?;
                Ok(Some(state))
            })
        })
    }
}
