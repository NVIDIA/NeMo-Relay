// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::CString;

use chrono::Utc;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use serde_json::json;
use uuid::Uuid;

use nvidia_nat_nexus_proxy::storage::StorageBackendDyn;
use nvidia_nat_nexus_proxy::trie::data_models::PredictionTrieNode;
use nvidia_nat_nexus_proxy::trie::{AccumulatorState, NodeAccumulators, RunningStats};
use nvidia_nat_nexus_proxy::types::{
    CallKind, CallRecord, ExecutionPlan, MetadataEnvelope, ParallelGroup, ParallelHint, RunRecord,
};

use super::*;

fn load_module<'py>(py: Python<'py>, code: &str) -> Bound<'py, PyModule> {
    let code = CString::new(code).unwrap();
    let file_name = CString::new("py_storage_coverage_tests.py").unwrap();
    let module_name = CString::new("py_storage_coverage_tests").unwrap();
    PyModule::from_code(py, &code, &file_name, &module_name).unwrap()
}

fn with_event_loop<'py, T>(py: Python<'py>, f: impl FnOnce(Bound<'py, PyAny>) -> T) -> T {
    let asyncio = py.import("asyncio").unwrap();
    let event_loop = asyncio.call_method0("new_event_loop").unwrap();
    asyncio
        .call_method1("set_event_loop", (&event_loop,))
        .unwrap();
    let result = f(event_loop.clone().into_any());
    asyncio
        .call_method1("set_event_loop", (py.None(),))
        .unwrap();
    event_loop.call_method0("close").unwrap();
    result
}

fn sample_plan(agent_id: &str) -> ExecutionPlan {
    ExecutionPlan {
        agent_id: agent_id.to_string(),
        parallel_groups: vec![ParallelGroup {
            group_id: "g1".into(),
            tool_names: vec!["search".into(), "lookup".into()],
        }],
        metadata_template: MetadataEnvelope {
            run_id: Uuid::new_v4(),
            agent_id: agent_id.to_string(),
            parallel_hints: vec![ParallelHint {
                tool_name: "search".into(),
                group_id: "g1".into(),
                explicit: true,
            }],
            extensions: json!({"team": "qa"}),
        },
    }
}

fn sample_run(agent_id: &str) -> RunRecord {
    let now = Utc::now();
    RunRecord {
        id: Uuid::new_v4(),
        agent_id: agent_id.to_string(),
        calls: vec![CallRecord {
            kind: CallKind::Llm,
            name: "chat".into(),
            started_at: now,
            ended_at: Some(now),
            metadata_snapshot: Some(MetadataEnvelope {
                run_id: Uuid::new_v4(),
                agent_id: agent_id.to_string(),
                parallel_hints: vec![],
                extensions: json!({"source": "test"}),
            }),
            output_tokens: Some(12),
        }],
        started_at: now,
        ended_at: Some(now),
    }
}

fn sample_trie(agent_id: &str) -> TrieEnvelope {
    TrieEnvelope::new(PredictionTrieNode::new("root"), agent_id)
}

fn sample_accumulators() -> AccumulatorState {
    let mut state = AccumulatorState::default();
    let mut node = NodeAccumulators::default();
    let mut stats = RunningStats::new();
    stats.add_sample(3.0);
    stats.add_sample(5.0);
    node.remaining_calls.insert(1, stats);
    state.nodes.insert("root".into(), node);
    state
}

#[test]
fn test_py_storage_backend_roundtrips_all_supported_methods() {
    Python::initialize();

    let agent_id = "agent-storage";
    let plan = sample_plan(agent_id);
    let run = sample_run(agent_id);
    let trie = sample_trie(agent_id);
    let accumulators = sample_accumulators();

    let (backend_obj, backend) = Python::attach(|py| {
        let module = load_module(
            py,
            r#"
class Backend:
    def __init__(self):
        self.calls = []
        self.plan = None
        self.runs = []
        self.trie = None
        self.accumulators = None

    async def store_run(self, record):
        self.calls.append(("store_run", record))

    async def load_plan(self, agent_id):
        self.calls.append(("load_plan", agent_id))
        return self.plan

    async def list_runs(self, agent_id):
        self.calls.append(("list_runs", agent_id))
        return self.runs

    async def store_trie(self, agent_id, envelope):
        self.calls.append(("store_trie", agent_id, envelope))

    async def load_trie(self, agent_id):
        self.calls.append(("load_trie", agent_id))
        return self.trie

    async def store_accumulators(self, agent_id, state):
        self.calls.append(("store_accumulators", agent_id, state))

    async def load_accumulators(self, agent_id):
        self.calls.append(("load_accumulators", agent_id))
        return self.accumulators
"#,
        );

        let backend_obj = module.getattr("Backend").unwrap().call0().unwrap();
        backend_obj
            .setattr(
                "plan",
                crate::convert::json_to_py(py, &serde_json::to_value(&plan).unwrap()).unwrap(),
            )
            .unwrap();
        backend_obj
            .setattr(
                "runs",
                crate::convert::json_to_py(py, &serde_json::to_value(vec![run.clone()]).unwrap())
                    .unwrap(),
            )
            .unwrap();
        backend_obj
            .setattr(
                "trie",
                crate::convert::json_to_py(py, &serde_json::to_value(&trie).unwrap()).unwrap(),
            )
            .unwrap();
        backend_obj
            .setattr(
                "accumulators",
                crate::convert::json_to_py(py, &serde_json::to_value(&accumulators).unwrap())
                    .unwrap(),
            )
            .unwrap();

        let backend_obj = backend_obj.unbind();
        let backend = PyStorageBackend::new(backend_obj.clone_ref(py));
        (backend_obj, backend)
    });

    Python::attach(|py| {
        with_event_loop(py, |event_loop| {
            pyo3_async_runtimes::tokio::run_until_complete(event_loop, async move {
                backend.store_run_dyn(&run).await.unwrap();

                let loaded_plan = backend.load_plan_dyn(agent_id).await.unwrap().unwrap();
                assert_eq!(
                    serde_json::to_value(&loaded_plan).unwrap(),
                    serde_json::to_value(&plan).unwrap()
                );

                let runs = backend.list_runs_dyn(agent_id).await.unwrap();
                assert_eq!(runs.len(), 1);
                assert_eq!(
                    serde_json::to_value(&runs[0]).unwrap(),
                    serde_json::to_value(&run).unwrap()
                );

                backend.store_trie(agent_id, &trie).await.unwrap();
                let loaded_trie = backend.load_trie(agent_id).await.unwrap().unwrap();
                assert_eq!(
                    serde_json::to_value(&loaded_trie).unwrap(),
                    serde_json::to_value(&trie).unwrap()
                );

                backend
                    .store_accumulators(agent_id, &accumulators)
                    .await
                    .unwrap();
                let loaded_accumulators =
                    backend.load_accumulators(agent_id).await.unwrap().unwrap();
                assert_eq!(
                    serde_json::to_value(&loaded_accumulators).unwrap(),
                    serde_json::to_value(&accumulators).unwrap()
                );

                Ok::<(), PyErr>(())
            })
            .unwrap();
        });
    });

    Python::attach(|py| {
        let calls =
            crate::convert::py_to_json(backend_obj.bind(py).getattr("calls").unwrap().as_any())
                .unwrap();
        let calls = calls.as_array().unwrap();
        assert_eq!(calls.len(), 7);
        assert_eq!(calls[0][0], json!("store_run"));
        assert_eq!(calls[1], json!(["load_plan", agent_id]));
        assert_eq!(calls[2], json!(["list_runs", agent_id]));
        assert_eq!(calls[3][0], json!("store_trie"));
        assert_eq!(calls[4], json!(["load_trie", agent_id]));
        assert_eq!(calls[5][0], json!("store_accumulators"));
        assert_eq!(calls[6], json!(["load_accumulators", agent_id]));
    });
}

#[test]
fn test_py_storage_backend_covers_none_and_error_paths() {
    Python::initialize();

    let trie = sample_trie("agent-storage");

    let (none_backend, failing_backend) = Python::attach(|py| {
        let module = load_module(
            py,
            r#"
class NoneBackend:
    async def store_run(self, record):
        return None

    async def load_plan(self, agent_id):
        return None

    async def list_runs(self, agent_id):
        return []

    async def store_trie(self, agent_id, envelope):
        return None

    async def load_trie(self, agent_id):
        return None

    async def store_accumulators(self, agent_id, state):
        return None

    async def load_accumulators(self, agent_id):
        return None

class FailingBackend:
    async def store_run(self, record):
        raise RuntimeError("store_run boom")

    async def load_plan(self, agent_id):
        return {"bad": True}

    async def list_runs(self, agent_id):
        return {"bad": True}

    async def store_trie(self, agent_id, envelope):
        raise RuntimeError("store_trie boom")

    async def load_trie(self, agent_id):
        return {"bad": True}

    async def store_accumulators(self, agent_id, state):
        raise RuntimeError("store_accumulators boom")

    async def load_accumulators(self, agent_id):
        return {"bad": True}
"#,
        );

        let none_backend = PyStorageBackend::new(
            module
                .getattr("NoneBackend")
                .unwrap()
                .call0()
                .unwrap()
                .unbind(),
        );
        let failing_backend = PyStorageBackend::new(
            module
                .getattr("FailingBackend")
                .unwrap()
                .call0()
                .unwrap()
                .unbind(),
        );
        (none_backend, failing_backend)
    });

    let run = sample_run("agent-storage");
    let accumulators = sample_accumulators();

    Python::attach(|py| {
        with_event_loop(py, |event_loop| {
            pyo3_async_runtimes::tokio::run_until_complete(event_loop, async move {
                assert!(none_backend.load_plan_dyn("agent").await.unwrap().is_none());
                assert!(none_backend.load_trie("agent").await.unwrap().is_none());
                assert!(none_backend
                    .load_accumulators("agent")
                    .await
                    .unwrap()
                    .is_none());

                let err = failing_backend.store_run_dyn(&run).await.unwrap_err();
                assert!(err.to_string().contains("Python store_run"));

                let err = failing_backend.load_plan_dyn("agent").await.unwrap_err();
                assert!(err.to_string().contains("deserialize ExecutionPlan"));

                let err = failing_backend.list_runs_dyn("agent").await.unwrap_err();
                assert!(err.to_string().contains("deserialize Vec<RunRecord>"));

                let err = failing_backend
                    .store_trie("agent", &trie)
                    .await
                    .unwrap_err();
                assert!(err.to_string().contains("Python store_trie"));

                let err = failing_backend.load_trie("agent").await.unwrap_err();
                assert!(err.to_string().contains("deserialize TrieEnvelope"));

                let err = failing_backend
                    .store_accumulators("agent", &accumulators)
                    .await
                    .unwrap_err();
                assert!(err.to_string().contains("Python store_accumulators"));

                let err = failing_backend
                    .load_accumulators("agent")
                    .await
                    .unwrap_err();
                assert!(err.to_string().contains("deserialize AccumulatorState"));

                Ok::<(), PyErr>(())
            })
            .unwrap();
        });
    });
}

#[test]
fn test_py_storage_backend_reuses_cached_task_locals_in_background_tasks() {
    Python::initialize();

    let run = sample_run("agent-storage");

    let (backend_obj, backend) = Python::attach(|py| {
        let module = load_module(
            py,
            r#"
class Backend:
    def __init__(self):
        self.calls = []

    async def store_run(self, record):
        self.calls.append(("store_run", record["agent_id"]))

    async def load_plan(self, agent_id):
        return None

    async def list_runs(self, agent_id):
        return []

    async def store_trie(self, agent_id, envelope):
        return None

    async def load_trie(self, agent_id):
        return None

    async def store_accumulators(self, agent_id, state):
        return None

    async def load_accumulators(self, agent_id):
        return None
"#,
        );

        let backend_obj = module.getattr("Backend").unwrap().call0().unwrap().unbind();
        let backend = PyStorageBackend::new(backend_obj.clone_ref(py));
        (backend_obj, backend)
    });

    Python::attach(|py| {
        with_event_loop(py, |event_loop| {
            pyo3_async_runtimes::tokio::run_until_complete(event_loop, async move {
                backend.store_run_dyn(&run).await.unwrap();

                let run = run.clone();
                tokio::spawn(async move { backend.store_run_dyn(&run).await })
                    .await
                    .unwrap()
                    .unwrap();

                Ok::<(), PyErr>(())
            })
            .unwrap();
        });
    });

    Python::attach(|py| {
        let calls =
            crate::convert::py_to_json(backend_obj.bind(py).getattr("calls").unwrap().as_any())
                .unwrap();
        assert_eq!(
            calls,
            json!([
                ["store_run", "agent-storage"],
                ["store_run", "agent-storage"]
            ])
        );
    });
}
