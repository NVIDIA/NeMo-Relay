// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use pyo3::types::PyModule;
use serde_json::json;

fn py_dict<'py>(py: Python<'py>, value: serde_json::Value) -> Bound<'py, pyo3::PyAny> {
    crate::convert::json_to_py(py, &value)
        .unwrap()
        .into_bound(py)
}

#[test]
fn py_api_helpers_and_scope_lifecycle_round_trip() {
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "_py_api_cov").unwrap();
        register(&module).unwrap();
        assert!(module.getattr("create_scope_stack").is_ok());
        assert!(module.getattr("llm_call_end").is_ok());

        let stack = create_scope_stack();
        set_thread_scope_stack(&stack);
        sync_thread_scope_stack(&stack);
        assert!(py_scope_stack_active());

        let handle = get_handle().unwrap();
        assert_eq!(handle.inner.name, "root");

        let data = py_dict(py, json!({"payload": true}));
        let metadata = py_dict(py, json!({"meta": true}));
        let child = push_scope(
            "child",
            PyScopeType::Tool,
            Some(handle.clone()),
            Some(PyScopeAttributes {
                inner: nemo_flow::types::scope::ScopeAttributes::PARALLEL,
            }),
            Some(&data),
            Some(&metadata),
        )
        .unwrap();
        assert_eq!(child.inner.name, "child");

        event(
            "mark",
            Some(child.clone()),
            Some(&py_dict(py, json!({"step": 1}))),
            Some(&py_dict(py, json!({"source": "cov"}))),
        )
        .unwrap();

        let tool = tool_call(
            "tool",
            &py_dict(py, json!({"arg": 1})),
            Some(child.clone()),
            Some(PyToolAttributes {
                inner: nemo_flow::types::tool::ToolAttributes::LOCAL,
            }),
            Some(&py_dict(py, json!({"tool_data": true}))),
            Some(&py_dict(py, json!({"tool_meta": true}))),
            Some("tool-call".to_string()),
        )
        .unwrap();
        tool_call_end(
            &tool,
            &py_dict(py, json!({"result": 2})),
            Some(&py_dict(py, json!({"done": true}))),
            Some(&py_dict(py, json!({"status": "ok"}))),
        )
        .unwrap();

        let llm_request = PyLLMRequest {
            inner: nemo_flow::types::llm::LLMRequest {
                headers: serde_json::Map::new(),
                content: json!({"messages": [], "model": "demo"}),
            },
        };
        let llm = llm_call(
            "llm",
            llm_request,
            Some(child.clone()),
            Some(PyLLMAttributes {
                inner: nemo_flow::types::llm::LLMAttributes::STATELESS
                    | nemo_flow::types::llm::LLMAttributes::STREAMING,
            }),
            Some(&py_dict(py, json!({"llm_data": true}))),
            Some(&py_dict(py, json!({"llm_meta": true}))),
            Some("demo-model".to_string()),
        )
        .unwrap();
        llm_call_end(
            &llm,
            &py_dict(py, json!({"response": "ok"})),
            Some(&py_dict(py, json!({"tokens": 10}))),
            Some(&py_dict(py, json!({"finish_reason": "stop"}))),
        )
        .unwrap();

        pop_scope(&child).unwrap();
        assert_eq!(get_handle().unwrap().inner.name, "root");
    });
}

#[test]
fn to_py_err_and_forward_stream_to_channel_cover_private_helpers() {
    Python::initialize();
    let err = to_py_err(nemo_flow::error::FlowError::Internal("boom".into()));
    assert!(err.to_string().contains("boom"));

    let runtime = tokio::runtime::Runtime::new().unwrap();
    runtime.block_on(async {
        let stream: RustJsonStream = Box::pin(tokio_stream::iter(vec![
            Ok(json!({"chunk": 1})),
            Ok(json!({"chunk": 2})),
        ]));
        let (tx, mut rx) = tokio::sync::mpsc::channel(2);

        forward_stream_to_channel(stream, tx).await;

        assert_eq!(rx.recv().await.unwrap().unwrap(), json!({"chunk": 1}));
        assert_eq!(rx.recv().await.unwrap().unwrap(), json!({"chunk": 2}));
        assert!(rx.recv().await.is_none());
    });
}
