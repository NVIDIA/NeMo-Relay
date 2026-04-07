// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::CString;

use pyo3::prelude::*;
use pyo3::types::PyModule;
use serde_json::json;
use uuid::Uuid;

use nvidia_nat_nexus_core::types as core_types;
use nvidia_nat_nexus_proxy::trie::data_models::{LlmCallPrediction, PredictionMetrics};
use nvidia_nat_nexus_proxy::{AgentHints, MetadataEnvelope, ParallelHint};

use super::*;

fn load_module<'py>(py: Python<'py>, code: &str) -> Bound<'py, PyModule> {
    let code = CString::new(code).unwrap();
    let file_name = CString::new("py_proxy_coverage_tests.py").unwrap();
    let module_name = CString::new("py_proxy_coverage_tests").unwrap();
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

#[test]
fn test_proxy_data_wrappers_and_module_register() {
    Python::initialize();
    Python::attach(|py| {
        let hint = PyParallelHint {
            inner: ParallelHint {
                tool_name: "search".into(),
                group_id: "g1".into(),
                explicit: true,
            },
        };
        assert_eq!(hint.tool_name(), "search");
        assert_eq!(hint.group_id(), "g1");
        assert!(hint.explicit());
        assert!(hint.__repr__().contains("ParallelHint"));

        let envelope = PyMetadataEnvelope {
            inner: MetadataEnvelope {
                run_id: Uuid::new_v4(),
                agent_id: "agent-1".into(),
                parallel_hints: vec![ParallelHint {
                    tool_name: "lookup".into(),
                    group_id: "g2".into(),
                    explicit: false,
                }],
                extensions: json!({"env": "test"}),
            },
        };
        assert_eq!(envelope.agent_id(), "agent-1");
        assert_eq!(envelope.parallel_hints().len(), 1);
        assert_eq!(
            crate::convert::py_to_json(envelope.extensions(py).unwrap().bind(py)).unwrap(),
            json!({"env": "test"})
        );
        assert!(envelope.__repr__().contains("MetadataEnvelope"));

        let backend = PyInMemoryBackend::new();
        assert_eq!(backend.__repr__(), "<InMemoryBackend>");

        let agent_hints = PyAgentHints {
            inner: AgentHints {
                osl: 64,
                iat: 25,
                priority: 3,
                latency_sensitivity: 2.0,
                prefix_id: "agent-depth".into(),
                total_requests: 4,
            },
        };
        assert_eq!(agent_hints.osl(), 64);
        assert_eq!(agent_hints.iat(), 25);
        assert_eq!(agent_hints.priority(), 3);
        assert_eq!(agent_hints.latency_sensitivity(), 2.0);
        assert_eq!(agent_hints.prefix_id(), "agent-depth");
        assert_eq!(agent_hints.total_requests(), 4);
        assert!(agent_hints.__repr__().contains("AgentHints"));

        let metrics = PyPredictionMetrics {
            inner: PredictionMetrics {
                sample_count: 7,
                mean: 2.5,
                p50: 2.0,
                p90: 4.0,
                p95: 4.5,
            },
        };
        assert_eq!(metrics.sample_count(), 7);
        assert_eq!(metrics.mean(), 2.5);
        assert_eq!(metrics.p50(), 2.0);
        assert_eq!(metrics.p90(), 4.0);
        assert_eq!(metrics.p95(), 4.5);
        assert!(metrics.__repr__().contains("PredictionMetrics"));

        let prediction = PyLlmCallPrediction {
            inner: LlmCallPrediction {
                remaining_calls: PredictionMetrics {
                    sample_count: 2,
                    mean: 1.0,
                    p50: 1.0,
                    p90: 1.0,
                    p95: 1.0,
                },
                interarrival_ms: PredictionMetrics::default(),
                output_tokens: PredictionMetrics::default(),
                latency_sensitivity: Some(5),
            },
        };
        assert_eq!(prediction.remaining_calls().sample_count(), 2);
        assert_eq!(prediction.interarrival_ms().sample_count(), 0);
        assert_eq!(prediction.output_tokens().sample_count(), 0);
        assert_eq!(prediction.latency_sensitivity(), Some(5));
        assert!(prediction.__repr__().contains("LlmCallPrediction"));

        let config = PySensitivityConfig::new(9, 0.7, 0.2, 0.1, 0.05);
        assert_eq!(config.sensitivity_scale(), 9);
        assert_eq!(config.w_critical(), 0.7);
        assert_eq!(config.w_fanout(), 0.2);
        assert_eq!(config.w_position(), 0.1);
        assert_eq!(config.w_parallel(), 0.05);
        assert!(config.__repr__().contains("SensitivityConfig"));

        let redis = PyRedisBackend { inner: None };
        assert_eq!(redis.__repr__(), "<RedisBackend>");

        let module = PyModule::new(py, "_proxy_test").unwrap();
        register(&module).unwrap();
        for name in [
            "ParallelHint",
            "MetadataEnvelope",
            "InMemoryBackend",
            "NexusProxy",
            "AgentHints",
            "PredictionMetrics",
            "LlmCallPrediction",
            "SensitivityConfig",
            "RedisBackend",
            "set_use_proxy",
            "set_proxy_backend",
            "set_proxy_sensitivity",
            "set_dynamo_intercept",
            "ensure_proxy",
            "teardown_proxy",
            "proxy_active",
            "set_latency_sensitivity",
        ] {
            assert!(module.getattr(name).is_ok(), "missing export {name}");
        }
    });
}

#[test]
fn test_py_nexus_proxy_constructor_and_lock_error_paths() {
    Python::initialize();
    Python::attach(|py| {
        let backend = Py::new(py, PyInMemoryBackend::new()).unwrap();
        let config = PySensitivityConfig::new(7, 0.6, 0.2, 0.15, 0.05);
        let proxy = PyNexusProxy::new(
            "agent-proxy".into(),
            backend.bind(py).as_any(),
            10,
            20,
            Some(config),
            true,
        )
        .unwrap();

        assert_eq!(proxy.agent_id().unwrap(), "agent-proxy");
        proxy.store_plan(None).unwrap();
        let extensions = crate::convert::json_to_py(py, &json!({"tier": "gold"})).unwrap();
        proxy.store_plan(Some(extensions.bind(py))).unwrap();
        assert_eq!(proxy.__repr__(), "<NexusProxy>");

        let _guard = proxy.inner.try_lock().unwrap();
        assert!(proxy
            .deregister()
            .unwrap_err()
            .to_string()
            .contains("try again after await completes"));
        assert!(proxy
            .agent_id()
            .unwrap_err()
            .to_string()
            .contains("try again after await completes"));
        assert!(proxy
            .store_plan(None)
            .unwrap_err()
            .to_string()
            .contains("try again after await completes"));
    });
}

#[test]
fn test_py_nexus_proxy_register_and_backend_validation_paths() {
    Python::initialize();
    Python::attach(|py| {
        let backend = Py::new(py, PyInMemoryBackend::new()).unwrap();
        let proxy = PyNexusProxy::new(
            "agent-async".into(),
            backend.bind(py).as_any(),
            100,
            200,
            None,
            false,
        )
        .unwrap();
        let proxy = Py::new(py, proxy).unwrap();
        let helper = load_module(
            py,
            r#"
async def register_proxy(proxy):
    await proxy.register()
"#,
        );

        with_event_loop(py, |event_loop| {
            let future = helper
                .getattr("register_proxy")
                .unwrap()
                .call1((proxy.bind(py),))
                .unwrap();
            event_loop
                .call_method1("run_until_complete", (&future,))
                .unwrap();
        });

        proxy.bind(py).borrow().deregister().unwrap();

        let module = load_module(
            py,
            r#"
class StorageBackend:
    async def store_run(self, record):
        return None

    async def load_plan(self, agent_id):
        return None
"#,
        );
        let duck_backend = module.getattr("StorageBackend").unwrap().call0().unwrap();
        let duck_proxy = PyNexusProxy::new(
            "agent-duck".into(),
            duck_backend.as_any(),
            1,
            2,
            None,
            false,
        )
        .unwrap();
        assert_eq!(duck_proxy.agent_id().unwrap(), "agent-duck");

        let err = match PyNexusProxy::new("bad".into(), py.None().bind(py), 1, 2, None, false) {
            Ok(_) => panic!("expected invalid backend type to fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("backend must be"));
    });
}

#[test]
fn test_declarative_proxy_helpers_and_validation() {
    Python::initialize();
    Python::attach(|py| {
        teardown_proxy().ok();
        set_use_proxy(false).unwrap();
        assert!(!proxy_active());

        let backend = Py::new(py, PyInMemoryBackend::new()).unwrap();
        set_proxy_backend(backend.bind(py).as_any()).unwrap();

        let module = load_module(
            py,
            r#"
class StorageBackend:
    async def store_run(self, record):
        return None

    async def load_plan(self, agent_id):
        return None
"#,
        );
        let duck_backend = module.getattr("StorageBackend").unwrap().call0().unwrap();
        set_proxy_backend(duck_backend.as_any()).unwrap();

        let err = set_proxy_backend(py.None().bind(py)).unwrap_err();
        assert!(err.to_string().contains("backend must be"));

        set_proxy_sensitivity(PySensitivityConfig::new(6, 0.5, 0.3, 0.2, 0.1)).unwrap();
        set_dynamo_intercept(true).unwrap();
        set_use_proxy(true).unwrap();
        let module = PyModule::new(py, "_proxy_runtime").unwrap();
        register(&module).unwrap();
        let helper = load_module(
            py,
            r#"
async def ensure_proxy_async(func):
    await func()
"#,
        );

        with_event_loop(py, |event_loop| {
            let future = helper
                .getattr("ensure_proxy_async")
                .unwrap()
                .call1((module.getattr("ensure_proxy").unwrap(),))
                .unwrap();
            event_loop
                .call_method1("run_until_complete", (&future,))
                .unwrap();
        });

        assert!(proxy_active());
        teardown_proxy().unwrap();
        set_use_proxy(false).unwrap();
        assert!(!proxy_active());

        let err = set_latency_sensitivity(0).unwrap_err();
        assert!(err.to_string().contains("positive"));

        let scope = nvidia_nat_nexus_core::nat_nexus_push_scope(
            "latency-scope",
            core_types::ScopeType::Agent,
            None,
            core_types::ScopeAttributes::empty(),
            None,
            None,
        )
        .unwrap();
        set_latency_sensitivity(3).unwrap();
        nvidia_nat_nexus_core::nat_nexus_pop_scope(&scope.uuid).unwrap();
    });
}

#[test]
fn test_py_redis_backend_take_inner_error() {
    let mut backend = PyRedisBackend { inner: None };
    let err = match backend.take_inner() {
        Ok(_) => panic!("expected missing Redis backend to fail"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("already consumed"));
}
