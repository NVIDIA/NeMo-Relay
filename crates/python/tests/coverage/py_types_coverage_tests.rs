// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nemo_flow::api::llm::{llm_call, llm_call_end};
use nemo_flow::api::scope::{pop_scope, push_scope};
use nemo_flow::types::event::Event;
use nemo_flow::types::llm::{LLMAttributes, LLMHandle};
use nemo_flow::types::scope::{ScopeAttributes, ScopeHandle, ScopeType};
use nemo_flow::types::tool::{ToolAttributes, ToolHandle};
use pyo3::types::{PyList, PyModule};
use serde_json::json;
use uuid::Uuid;

#[test]
fn test_register_exposes_all_type_bindings() {
    Python::initialize();
    Python::attach(|py| {
        let module = PyModule::new(py, "_types_test").unwrap();
        register(&module).unwrap();

        assert!(module.getattr("ScopeStack").is_ok());
        assert!(module.getattr("LlmStream").is_ok());
        assert!(module.getattr("ScopeAttributes").is_ok());
        assert!(module.getattr("ToolAttributes").is_ok());
        assert!(module.getattr("LLMAttributes").is_ok());
        assert!(module.getattr("ScopeType").is_ok());
        assert!(module.getattr("ScopeHandle").is_ok());
        assert!(module.getattr("ToolHandle").is_ok());
        assert!(module.getattr("LLMHandle").is_ok());
        assert!(module.getattr("LLMRequest").is_ok());
        assert!(module.getattr("ScopeStartEvent").is_ok());
        assert!(module.getattr("ScopeEndEvent").is_ok());
        assert!(module.getattr("ToolStartEvent").is_ok());
        assert!(module.getattr("ToolEndEvent").is_ok());
        assert!(module.getattr("LLMStartEvent").is_ok());
        assert!(module.getattr("LLMEndEvent").is_ok());
        assert!(module.getattr("MarkEvent").is_ok());
        assert!(module.getattr("AtifExporter").is_ok());
        assert!(module.getattr("OpenInferenceConfig").is_ok());
        assert!(module.getattr("OpenInferenceSubscriber").is_ok());
        assert!(module.getattr("OpenTelemetryConfig").is_ok());
        assert!(module.getattr("OpenTelemetrySubscriber").is_ok());
        assert!(module.getattr("OpenAIChatCodec").is_ok());
        assert!(module.getattr("OpenAIResponsesCodec").is_ok());
        assert!(module.getattr("AnthropicMessagesCodec").is_ok());
    });
}

#[test]
fn test_bitflags_handles_and_event_wrappers_expose_expected_fields() {
    Python::initialize();
    let scope_attrs =
        PyScopeAttributes::new(PyScopeAttributes::PARALLEL | PyScopeAttributes::RELOCATABLE);
    assert!(scope_attrs.is_parallel());
    assert!(scope_attrs.is_relocatable());
    assert_eq!(
        scope_attrs.value(),
        PyScopeAttributes::PARALLEL | PyScopeAttributes::RELOCATABLE
    );
    assert!(scope_attrs.__repr__().contains("ScopeAttributes"));

    let tool_attrs = PyToolAttributes::new(PyToolAttributes::LOCAL);
    assert!(tool_attrs.is_local());
    assert_eq!(tool_attrs.value(), PyToolAttributes::LOCAL);
    assert!(tool_attrs.__repr__().contains("ToolAttributes"));

    let llm_attrs = PyLLMAttributes::new(PyLLMAttributes::STATELESS | PyLLMAttributes::STREAMING);
    assert!(llm_attrs.is_stateless());
    assert!(llm_attrs.is_streaming());
    assert_eq!(
        llm_attrs.value(),
        PyLLMAttributes::STATELESS | PyLLMAttributes::STREAMING
    );
    assert!(llm_attrs.__repr__().contains("LLMAttributes"));

    let scope_variants = [
        (PyScopeType::Agent, ScopeType::Agent),
        (PyScopeType::Function, ScopeType::Function),
        (PyScopeType::Tool, ScopeType::Tool),
        (PyScopeType::Llm, ScopeType::Llm),
        (PyScopeType::Retriever, ScopeType::Retriever),
        (PyScopeType::Embedder, ScopeType::Embedder),
        (PyScopeType::Reranker, ScopeType::Reranker),
        (PyScopeType::Guardrail, ScopeType::Guardrail),
        (PyScopeType::Evaluator, ScopeType::Evaluator),
        (PyScopeType::Custom, ScopeType::Custom),
        (PyScopeType::Unknown, ScopeType::Unknown),
    ];
    for (py_variant, core_variant) in scope_variants {
        let py_round_trip = PyScopeType::from(core_variant);
        let core_round_trip: ScopeType = py_variant.clone().into();
        assert!(py_variant == py_round_trip);
        assert!(core_round_trip == core_variant);
    }

    Python::attach(|py| {
        let parent_uuid = Uuid::now_v7();
        let scope = PyScopeHandle::from(ScopeHandle::new(
            "scope".into(),
            ScopeType::Tool,
            ScopeAttributes::PARALLEL,
            Some(parent_uuid),
            Some(json!({"scope": true})),
            Some(json!({"meta": "scope"})),
        ));
        assert!(scope.scope_type() == PyScopeType::Tool);
        assert_eq!(scope.parent_uuid(), Some(parent_uuid.to_string()));
        assert_eq!(
            py_to_json(scope.data(py).unwrap().bind(py)).unwrap(),
            json!({"scope": true})
        );
        assert_eq!(
            py_to_json(scope.metadata(py).unwrap().bind(py)).unwrap(),
            json!({"meta": "scope"})
        );
        assert!(scope.__repr__().contains("ScopeHandle"));

        let tool = PyToolHandle::from(ToolHandle::new(
            "tool".into(),
            ToolAttributes::LOCAL,
            Some(parent_uuid),
            Some(json!({"tool": true})),
            Some(json!({"meta": "tool"})),
        ));
        assert_eq!(tool.parent_uuid(), Some(parent_uuid.to_string()));
        assert_eq!(tool.attributes().value(), PyToolAttributes::LOCAL);
        assert_eq!(
            py_to_json(tool.data(py).unwrap().bind(py)).unwrap(),
            json!({"tool": true})
        );
        assert_eq!(
            py_to_json(tool.metadata(py).unwrap().bind(py)).unwrap(),
            json!({"meta": "tool"})
        );
        assert!(tool.__repr__().contains("ToolHandle"));

        let llm = PyLLMHandle::from(LLMHandle::new(
            "llm".into(),
            LLMAttributes::STATELESS | LLMAttributes::STREAMING,
            Some(parent_uuid),
            Some(json!({"llm": true})),
            Some(json!({"meta": "llm"})),
        ));
        assert_eq!(llm.parent_uuid(), Some(parent_uuid.to_string()));
        assert_eq!(
            llm.attributes().value(),
            PyLLMAttributes::STATELESS | PyLLMAttributes::STREAMING
        );
        assert_eq!(
            py_to_json(llm.data(py).unwrap().bind(py)).unwrap(),
            json!({"llm": true})
        );
        assert_eq!(
            py_to_json(llm.metadata(py).unwrap().bind(py)).unwrap(),
            json!({"meta": "llm"})
        );
        assert!(llm.__repr__().contains("LLMHandle"));

        let request = PyLLMRequest {
            inner: LLMRequest {
                headers: serde_json::Map::from_iter([("x-trace".into(), json!("1"))]),
                content: json!({"prompt": "hello"}),
            },
        };
        assert_eq!(
            py_to_json(request.headers(py).unwrap().bind(py)).unwrap(),
            json!({"x-trace": "1"})
        );
        assert_eq!(
            py_to_json(request.content(py).unwrap().bind(py)).unwrap(),
            json!({"prompt": "hello"})
        );
        assert_eq!(request.__repr__(), "LLMRequest(...)");

        let event = match Event::mark(
            Some(parent_uuid),
            Uuid::now_v7(),
            "event",
            Some(json!({"event": true})),
            Some(json!({"meta": "event"})),
        ) {
            Event::Mark(inner) => PyMarkEvent { inner },
            _ => unreachable!(),
        };
        assert_eq!(event.kind(), "Mark");
        assert_eq!(event.parent_uuid(), Some(parent_uuid.to_string()));
        assert_eq!(
            py_to_json(event.data(py).unwrap().bind(py)).unwrap(),
            json!({"event": true})
        );
        assert_eq!(
            py_to_json(event.metadata(py).unwrap().bind(py)).unwrap(),
            json!({"meta": "event"})
        );
        assert!(event.timestamp().contains('T'));

        let tool_event = match Event::tool_start(
            Some(parent_uuid),
            Uuid::now_v7(),
            "tool-event",
            Some(json!({"event": true})),
            Some(json!({"meta": "event"})),
            ToolAttributes::LOCAL,
            Some(json!({"input": true})),
            Some("tool-1".into()),
        ) {
            Event::ToolStart(inner) => PyToolStartEvent { inner },
            _ => unreachable!(),
        };
        assert_eq!(tool_event.kind(), "ToolStart");
        assert_eq!(
            py_to_json(tool_event.input(py).unwrap().bind(py)).unwrap(),
            json!({"input": true})
        );
        assert_eq!(tool_event.tool_call_id(), Some("tool-1".into()));
        assert_eq!(tool_event.attributes().value(), PyToolAttributes::LOCAL);

        let llm_event = match Event::llm_end(
            Some(parent_uuid),
            Uuid::now_v7(),
            "llm-event",
            Some(json!({"event": true})),
            Some(json!({"meta": "event"})),
            LLMAttributes::STATELESS,
            Some(json!({"output": true})),
            Some("model".into()),
            None,
        ) {
            Event::LLMEnd(inner) => PyLLMEndEvent { inner },
            _ => unreachable!(),
        };
        assert_eq!(llm_event.kind(), "LLMEnd");
        assert_eq!(
            py_to_json(llm_event.output(py).unwrap().bind(py)).unwrap(),
            json!({"output": true})
        );
        assert_eq!(llm_event.model_name(), Some("model".into()));
        assert_eq!(llm_event.attributes().value(), PyLLMAttributes::STATELESS);
    });
}

#[test]
fn test_atif_exporter_methods_cover_register_export_and_clear() {
    Python::initialize();
    Python::attach(|py| {
        let tool_def = json_to_py(py, &json!({"name": "typed_tool"})).unwrap();
        let tool_defs = PyList::empty(py);
        tool_defs.append(tool_def.bind(py)).unwrap();
        let extra = json_to_py(py, &json!({"team": "qa"})).unwrap();

        let exporter = PyAtifExporter::new(
            "session-types-rust".into(),
            "py-agent".into(),
            "1.0.0".into(),
            Some("typed-model".into()),
            Some(&tool_defs),
            Some(extra.bind(py)),
        )
        .unwrap();

        let subscriber_name = format!("py_types_atif_{}", Uuid::now_v7());
        exporter.register(subscriber_name.clone()).unwrap();
        let scope = push_scope(
            "atif_root",
            ScopeType::Agent,
            None,
            ScopeAttributes::empty(),
            None,
            None,
        )
        .unwrap();
        let request = LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": [{"role": "user", "content": "hello"}], "model": "typed-model"}),
        };

        let handle = llm_call(
            "atif_llm",
            &request,
            Some(&scope),
            LLMAttributes::empty(),
            None,
            None,
            Some("typed-model".into()),
            None,
        )
        .unwrap();
        llm_call_end(&handle, json!({"content": "world"}), None, None, None).unwrap();

        let exported = py_to_json(exporter.export(py).unwrap().bind(py)).unwrap();
        let exported_json: serde_json::Value =
            serde_json::from_str(&exporter.export_json().unwrap()).unwrap();
        assert_eq!(exported["session_id"], json!("session-types-rust"));
        assert_eq!(exported["agent"]["name"], json!("py-agent"));
        assert_eq!(
            exported["agent"]["tool_definitions"],
            json!([{"name": "typed_tool"}])
        );
        assert_eq!(exported["agent"]["extra"], json!({"team": "qa"}));
        assert_eq!(exported_json["session_id"], json!("session-types-rust"));
        assert!(!exported["steps"].as_array().unwrap().is_empty());

        exporter.clear();
        let cleared = py_to_json(exporter.export(py).unwrap().bind(py)).unwrap();
        assert_eq!(cleared["steps"], json!([]));

        pop_scope(&scope.uuid).unwrap();
        assert!(exporter.deregister(subscriber_name.clone()).unwrap());
        assert!(!exporter.deregister(subscriber_name).unwrap());
        assert_eq!(exporter.__repr__(), "<AtifExporter>");
    });
}

#[test]
fn test_open_telemetry_config_and_subscriber_cover_lifecycle() {
    Python::initialize();
    Python::attach(|py| {
        let mut config = PyOpenTelemetryConfig::new();
        config.endpoint = Some("http://localhost:4318/v1/traces".into());
        config.service_name = "py-agent".into();
        config.service_namespace = Some("agents".into());
        config.service_version = Some("1.0.0".into());
        config.instrumentation_scope = "py-scope".into();
        config.timeout_millis = 1250;
        config.set_header("authorization".into(), "Bearer token".into());
        config.set_resource_attribute("deployment.environment".into(), "test".into());

        assert!(config.__repr__().contains("OpenTelemetryConfig"));
        assert_eq!(
            py_to_json(config.headers(py).unwrap().bind(py)).unwrap(),
            json!({"authorization": "Bearer token"})
        );
        assert_eq!(
            py_to_json(config.resource_attributes(py).unwrap().bind(py)).unwrap(),
            json!({"deployment.environment": "test"})
        );

        let config = pyo3::Py::new(py, config).unwrap();
        let subscriber = PyOpenTelemetrySubscriber::new(config.bind(py).borrow()).unwrap();
        let subscriber_name = format!("py_otel_{}", Uuid::now_v7().simple());
        subscriber.register(subscriber_name.clone()).unwrap();
        assert!(subscriber.deregister(subscriber_name.clone()).unwrap());
        assert!(!subscriber.deregister(subscriber_name).unwrap());
        subscriber.force_flush().unwrap();
        subscriber.shutdown().unwrap();
        assert_eq!(subscriber.__repr__(), "<OpenTelemetrySubscriber>");
    });
}

#[test]
fn test_open_telemetry_config_rejects_invalid_inputs() {
    Python::initialize();
    Python::attach(|py| {
        let mut config = PyOpenTelemetryConfig::new();
        let bad_headers = PyList::empty(py);
        assert!(config.set_headers(&bad_headers.into_any()).is_err());

        let bad_attrs = json_to_py(py, &json!({"env": 1})).unwrap();
        assert!(config.set_resource_attributes(bad_attrs.bind(py)).is_err());

        config.transport = "invalid".into();
        let err = config.to_rust_config().unwrap_err();
        assert!(err.to_string().contains("transport must be"));
    });
}

#[test]
fn test_open_inference_config_and_subscriber_cover_lifecycle() {
    Python::initialize();
    Python::attach(|py| {
        let mut config = PyOpenInferenceConfig::new();
        config.endpoint = Some("http://localhost:4318/v1/traces".into());
        config.service_name = "py-agent".into();
        config.service_namespace = Some("agents".into());
        config.service_version = Some("1.0.0".into());
        config.instrumentation_scope = "py-scope".into();
        config.timeout_millis = 1250;
        config.set_header("authorization".into(), "Bearer token".into());
        config.set_resource_attribute("deployment.environment".into(), "test".into());

        assert!(config.__repr__().contains("OpenInferenceConfig"));
        assert_eq!(
            py_to_json(config.headers(py).unwrap().bind(py)).unwrap(),
            json!({"authorization": "Bearer token"})
        );
        assert_eq!(
            py_to_json(config.resource_attributes(py).unwrap().bind(py)).unwrap(),
            json!({"deployment.environment": "test"})
        );

        let config = pyo3::Py::new(py, config).unwrap();
        let subscriber = PyOpenInferenceSubscriber::new(config.bind(py).borrow()).unwrap();
        let subscriber_name = format!("py_openinference_{}", Uuid::now_v7().simple());
        subscriber.register(subscriber_name.clone()).unwrap();
        assert!(subscriber.deregister(subscriber_name.clone()).unwrap());
        assert!(!subscriber.deregister(subscriber_name).unwrap());
        subscriber.force_flush().unwrap();
        subscriber.shutdown().unwrap();
        assert_eq!(subscriber.__repr__(), "<OpenInferenceSubscriber>");
    });
}

#[test]
fn test_open_inference_config_rejects_invalid_inputs() {
    Python::initialize();
    Python::attach(|py| {
        let mut config = PyOpenInferenceConfig::new();
        let bad_headers = PyList::empty(py);
        assert!(config.set_headers(&bad_headers.into_any()).is_err());

        let bad_attrs = json_to_py(py, &json!({"env": 1})).unwrap();
        assert!(config.set_resource_attributes(bad_attrs.bind(py)).is_err());

        config.transport = "invalid".into();
        let err = config.to_rust_config().unwrap_err();
        assert!(err.to_string().contains("transport must be"));
    });
}
