// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
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
        assert!(module.getattr("EventType").is_ok());
        assert!(module.getattr("ScopeHandle").is_ok());
        assert!(module.getattr("ToolHandle").is_ok());
        assert!(module.getattr("LLMHandle").is_ok());
        assert!(module.getattr("LLMRequest").is_ok());
        assert!(module.getattr("Event").is_ok());
        assert!(module.getattr("AtifExporter").is_ok());
        assert!(module.getattr("OpenInferenceConfig").is_ok());
        assert!(module.getattr("OpenInferenceSubscriber").is_ok());
        assert!(module.getattr("OpenTelemetryConfig").is_ok());
        assert!(module.getattr("OpenTelemetrySubscriber").is_ok());
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
        (PyScopeType::Agent, core_types::ScopeType::Agent),
        (PyScopeType::Function, core_types::ScopeType::Function),
        (PyScopeType::Tool, core_types::ScopeType::Tool),
        (PyScopeType::Llm, core_types::ScopeType::Llm),
        (PyScopeType::Retriever, core_types::ScopeType::Retriever),
        (PyScopeType::Embedder, core_types::ScopeType::Embedder),
        (PyScopeType::Reranker, core_types::ScopeType::Reranker),
        (PyScopeType::Guardrail, core_types::ScopeType::Guardrail),
        (PyScopeType::Evaluator, core_types::ScopeType::Evaluator),
        (PyScopeType::Custom, core_types::ScopeType::Custom),
        (PyScopeType::Unknown, core_types::ScopeType::Unknown),
    ];
    for (py_variant, core_variant) in scope_variants {
        let py_round_trip = PyScopeType::from(core_variant);
        let core_round_trip: core_types::ScopeType = py_variant.clone().into();
        assert!(py_variant == py_round_trip);
        assert!(core_round_trip == core_variant);
    }

    let event_variants = [
        (PyEventType::Start, core_types::EventType::Start),
        (PyEventType::End, core_types::EventType::End),
        (PyEventType::Mark, core_types::EventType::Mark),
    ];
    for (py_variant, core_variant) in event_variants {
        let py_round_trip = PyEventType::from(core_variant);
        let core_round_trip: core_types::EventType = py_variant.clone().into();
        assert!(py_variant == py_round_trip);
        assert!(core_round_trip == core_variant);
    }

    Python::attach(|py| {
        let parent_uuid = Uuid::new_v4();
        let scope = PyScopeHandle::from(core_types::ScopeHandle::new(
            "scope".into(),
            core_types::ScopeType::Tool,
            core_types::ScopeAttributes::PARALLEL,
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

        let tool = PyToolHandle::from(core_types::ToolHandle::new(
            "tool".into(),
            core_types::ToolAttributes::LOCAL,
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

        let llm = PyLLMHandle::from(core_types::LLMHandle::new(
            "llm".into(),
            core_types::LLMAttributes::STATELESS | core_types::LLMAttributes::STREAMING,
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
            inner: core_types::LLMRequest {
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

        let mut inner_event = core_types::Event::new(
            Some(parent_uuid),
            Uuid::new_v4(),
            Some("event".into()),
            Some(json!({"event": true})),
            Some(json!({"meta": "event"})),
            None,
            core_types::EventType::Mark,
            Some(core_types::ScopeType::Guardrail),
        );
        inner_event.input = Some(json!({"input": true}));
        inner_event.output = Some(json!({"output": true}));
        inner_event.model_name = Some("model".into());
        inner_event.tool_call_id = Some("tool-1".into());
        inner_event.root_uuid = Some(Uuid::new_v4());

        let event = PyEvent::from(inner_event.clone());
        assert_eq!(event.parent_uuid(), Some(parent_uuid.to_string()));
        assert!(event.event_type() == PyEventType::Mark);
        assert!(event.scope_type() == Some(PyScopeType::Guardrail));
        assert_eq!(
            py_to_json(event.data(py).unwrap().bind(py)).unwrap(),
            json!({"event": true})
        );
        assert_eq!(
            py_to_json(event.metadata(py).unwrap().bind(py)).unwrap(),
            json!({"meta": "event"})
        );
        assert_eq!(
            py_to_json(event.input(py).unwrap().bind(py)).unwrap(),
            json!({"input": true})
        );
        assert_eq!(
            py_to_json(event.output(py).unwrap().bind(py)).unwrap(),
            json!({"output": true})
        );
        assert_eq!(event.model_name(), Some("model".into()));
        assert_eq!(event.tool_call_id(), Some("tool-1".into()));
        assert_eq!(
            event.root_uuid(),
            inner_event.root_uuid.map(|uuid| uuid.to_string())
        );
        assert!(event.timestamp().contains('T'));
        assert!(event.__repr__().contains("Event("));
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

        let subscriber_name = format!("py_types_atif_{}", Uuid::new_v4());
        exporter.register(subscriber_name.clone()).unwrap();
        let scope = nvidia_nat_nexus_core::nat_nexus_push_scope(
            "atif_root",
            core_types::ScopeType::Agent,
            None,
            core_types::ScopeAttributes::empty(),
            None,
            None,
        )
        .unwrap();
        let request = core_types::LLMRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": [{"role": "user", "content": "hello"}], "model": "typed-model"}),
        };

        let handle = nvidia_nat_nexus_core::nat_nexus_llm_call(
            "atif_llm",
            &request,
            Some(&scope),
            core_types::LLMAttributes::empty(),
            None,
            None,
            Some("typed-model".into()),
        )
        .unwrap();
        nvidia_nat_nexus_core::nat_nexus_llm_call_end(
            &handle,
            json!({"content": "world"}),
            None,
            None,
        )
        .unwrap();

        let exported = py_to_json(
            exporter
                .export(py, Some(scope.uuid.to_string()))
                .unwrap()
                .bind(py),
        )
        .unwrap();
        let exported_json: serde_json::Value =
            serde_json::from_str(&exporter.export_json(Some(scope.uuid.to_string())).unwrap())
                .unwrap();
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
        let cleared = py_to_json(
            exporter
                .export(py, Some(scope.uuid.to_string()))
                .unwrap()
                .bind(py),
        )
        .unwrap();
        assert_eq!(cleared["steps"], json!([]));

        let invalid_export = exporter.export_json(Some("not-a-uuid".into())).unwrap_err();
        assert!(invalid_export.to_string().contains("Invalid UUID"));

        nvidia_nat_nexus_core::nat_nexus_pop_scope(&scope.uuid).unwrap();
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
        let subscriber_name = format!("py_otel_{}", Uuid::new_v4().simple());
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
        let subscriber_name = format!("py_openinference_{}", Uuid::new_v4().simple());
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
