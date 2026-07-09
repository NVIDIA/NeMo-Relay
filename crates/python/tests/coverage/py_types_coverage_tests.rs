// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coverage tests for py types coverage in the NeMo Relay Python crate.

use super::*;
use std::ffi::CString;

use nemo_relay::api::event::{
    BaseEvent, CategoryProfile, Event, EventCategory, MarkEvent, ScopeCategory, ScopeEvent,
    llm_attributes_to_strings, scope_attributes_to_strings, tool_attributes_to_strings,
};
use nemo_relay::api::llm::{LlmAttributes, LlmHandle};
use nemo_relay::api::llm::{LlmCallParams, llm_call, llm_call_end};
use nemo_relay::api::scope::{PushScopeParams, pop_scope, push_scope};
use nemo_relay::api::scope::{ScopeAttributes, ScopeHandle, ScopeType};
use nemo_relay::api::tool::{ToolAttributes, ToolHandle};
use nemo_relay::codec::request::{
    AnnotatedLlmRequest as AnnotatedLLMRequest, Message, MessageContent,
};
use nemo_relay::codec::response::{
    AnnotatedLlmResponse as AnnotatedLLMResponse, ApiSpecificResponse, CostEstimate, CostSource,
    FinishReason, ResponseToolCall, Usage,
};
use pyo3::types::{PyDict, PyList, PyModule};
use serde_json::json;
use uuid::Uuid;

fn with_event_loop<T>(py: Python<'_>, f: impl FnOnce(Bound<'_, PyAny>) -> T) -> T {
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

fn base_event(
    parent_uuid: Uuid,
    name: &str,
    data: serde_json::Value,
    metadata: serde_json::Value,
) -> BaseEvent {
    BaseEvent::builder()
        .parent_uuid(parent_uuid)
        .name(name)
        .data(data)
        .metadata(metadata)
        .build()
}

#[test]
fn test_register_exposes_all_type_bindings() {
    let _python = crate::test_support::init_python_test();
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
        assert!(module.getattr("ScopeEvent").is_ok());
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
    let _python = crate::test_support::init_python_test();
    let scope_attrs =
        PyScopeAttributes::new(PyScopeAttributes::PARALLEL | PyScopeAttributes::RELOCATABLE);
    assert!(scope_attrs.is_parallel());
    assert!(scope_attrs.is_relocatable());
    assert_eq!(
        scope_attrs.value(),
        PyScopeAttributes::PARALLEL | PyScopeAttributes::RELOCATABLE
    );
    assert!(scope_attrs.__repr__().contains("ScopeAttributes"));

    let tool_attrs = PyToolAttributes::new(PyToolAttributes::REMOTE);
    assert!(tool_attrs.is_remote());
    assert_eq!(tool_attrs.value(), PyToolAttributes::REMOTE);
    assert!(tool_attrs.__repr__().contains("ToolAttributes"));

    let llm_attrs = PyLLMAttributes::new(PyLLMAttributes::STATEFUL | PyLLMAttributes::STREAMING);
    assert!(llm_attrs.is_stateful());
    assert!(llm_attrs.is_streaming());
    assert_eq!(
        llm_attrs.value(),
        PyLLMAttributes::STATEFUL | PyLLMAttributes::STREAMING
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
        let scope = PyScopeHandle::from(
            ScopeHandle::builder()
                .name("scope")
                .scope_type(ScopeType::Tool)
                .attributes(ScopeAttributes::PARALLEL)
                .parent_uuid(parent_uuid)
                .data(json!({"scope": true}))
                .metadata(json!({"meta": "scope"}))
                .build(),
        );
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

        let tool = PyToolHandle::from(
            ToolHandle::builder()
                .name("tool")
                .attributes(ToolAttributes::REMOTE)
                .parent_uuid(parent_uuid)
                .data(json!({"tool": true}))
                .metadata(json!({"meta": "tool"}))
                .build(),
        );
        assert_eq!(tool.parent_uuid(), Some(parent_uuid.to_string()));
        assert_eq!(tool.attributes().value(), PyToolAttributes::REMOTE);
        assert_eq!(
            py_to_json(tool.data(py).unwrap().bind(py)).unwrap(),
            json!({"tool": true})
        );
        assert_eq!(
            py_to_json(tool.metadata(py).unwrap().bind(py)).unwrap(),
            json!({"meta": "tool"})
        );
        assert!(tool.__repr__().contains("ToolHandle"));

        let llm = PyLLMHandle::from(
            LlmHandle::builder()
                .name("llm")
                .attributes(LlmAttributes::STATEFUL | LlmAttributes::STREAMING)
                .parent_uuid(parent_uuid)
                .data(json!({"llm": true}))
                .metadata(json!({"meta": "llm"}))
                .build(),
        );
        assert_eq!(llm.parent_uuid(), Some(parent_uuid.to_string()));
        assert_eq!(
            llm.attributes().value(),
            PyLLMAttributes::STATEFUL | PyLLMAttributes::STREAMING
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
            inner: LlmRequest {
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

        let event = match Event::Mark(MarkEvent::new(
            base_event(
                parent_uuid,
                "event",
                json!({"event": true}),
                json!({"meta": "event"}),
            ),
            None,
            None,
        )) {
            Event::Mark(inner) => PyMarkEvent { inner },
            _ => unreachable!(),
        };
        assert_eq!(event.kind(), "mark");
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

        let tool_event = match Event::Scope(ScopeEvent::new(
            base_event(
                parent_uuid,
                "tool-event",
                json!({"input": true}),
                json!({"meta": "event"}),
            ),
            ScopeCategory::Start,
            tool_attributes_to_strings(ToolAttributes::REMOTE),
            EventCategory::tool(),
            Some(CategoryProfile::builder().tool_call_id("tool-1").build()),
        )) {
            Event::Scope(inner) => PyScopeEvent { inner },
            _ => unreachable!(),
        };
        assert_eq!(tool_event.kind(), "scope");
        assert_eq!(tool_event.scope_category(), "start");
        assert_eq!(tool_event.category(), "tool");
        assert_eq!(
            py_to_json(tool_event.data(py).unwrap().bind(py)).unwrap(),
            json!({"input": true})
        );
        assert_eq!(
            py_to_json(tool_event.category_profile(py).unwrap().bind(py)).unwrap(),
            json!({"tool_call_id": "tool-1"})
        );
        assert_eq!(tool_event.attributes(), vec!["remote"]);

        let llm_event = match Event::Scope(ScopeEvent::new(
            base_event(
                parent_uuid,
                "llm-event",
                json!({"output": true}),
                json!({"meta": "event"}),
            ),
            ScopeCategory::End,
            llm_attributes_to_strings(LlmAttributes::STATEFUL),
            EventCategory::llm(),
            Some(CategoryProfile::builder().model_name("model").build()),
        )) {
            Event::Scope(inner) => PyScopeEvent { inner },
            _ => unreachable!(),
        };
        assert_eq!(llm_event.kind(), "scope");
        assert_eq!(llm_event.scope_category(), "end");
        assert_eq!(llm_event.category(), "llm");
        assert_eq!(
            py_to_json(llm_event.data(py).unwrap().bind(py)).unwrap(),
            json!({"output": true})
        );
        assert_eq!(
            py_to_json(llm_event.category_profile(py).unwrap().bind(py)).unwrap(),
            json!({"model_name": "model"})
        );
        assert_eq!(llm_event.attributes(), vec!["stateful"]);
    });
}

#[test]
fn test_atif_exporter_methods_cover_register_export_and_clear() {
    let _python = crate::test_support::init_python_test();
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
            PushScopeParams::builder()
                .name("atif_root")
                .scope_type(ScopeType::Agent)
                .build(),
        )
        .unwrap();
        let request = LlmRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": [{"role": "user", "content": "hello"}], "model": "typed-model"}),
        };

        let handle = llm_call(
            LlmCallParams::builder()
                .name("atif_llm")
                .request(&request)
                .parent(&scope)
                .model_name("typed-model")
                .build(),
        )
        .unwrap();
        llm_call_end(
            nemo_relay::api::llm::LlmCallEndParams::builder()
                .handle(&handle)
                .response(json!({"content": "world"}))
                .build(),
        )
        .unwrap();

        let exported = py_to_json(exporter.export(py).unwrap().bind(py)).unwrap();
        let exported_json: serde_json::Value =
            serde_json::from_str(&exporter.export_json(py).unwrap()).unwrap();
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

        pop_scope(
            nemo_relay::api::scope::PopScopeParams::builder()
                .handle_uuid(&scope.uuid)
                .build(),
        )
        .unwrap();
        assert!(exporter.deregister(subscriber_name.clone()).unwrap());
        assert!(!exporter.deregister(subscriber_name).unwrap());
        assert_eq!(exporter.__repr__(), "<AtifExporter>");
    });
}

#[test]
fn test_open_telemetry_config_and_subscriber_cover_lifecycle() {
    let _python = crate::test_support::init_python_test();
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
        subscriber.force_flush(py).unwrap();
        subscriber.shutdown(py).unwrap();
        assert_eq!(subscriber.__repr__(), "<OpenTelemetrySubscriber>");
    });
}

#[test]
fn test_open_telemetry_config_rejects_invalid_inputs() {
    let _python = crate::test_support::init_python_test();
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
    let _python = crate::test_support::init_python_test();
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
        subscriber.force_flush(py).unwrap();
        subscriber.shutdown(py).unwrap();
        assert_eq!(subscriber.__repr__(), "<OpenInferenceSubscriber>");
    });
}

#[test]
fn test_open_inference_config_rejects_invalid_inputs() {
    let _python = crate::test_support::init_python_test();
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

#[test]
fn test_stream_request_event_and_handle_wrappers_cover_remaining_methods() {
    let _python = crate::test_support::init_python_test();

    let scope_or = PyScopeAttributes::new(PyScopeAttributes::PARALLEL)
        .__or__(&PyScopeAttributes::new(PyScopeAttributes::RELOCATABLE));
    assert_eq!(
        scope_or.value(),
        PyScopeAttributes::PARALLEL | PyScopeAttributes::RELOCATABLE
    );
    let scope_and = scope_or.__and__(&PyScopeAttributes::new(PyScopeAttributes::PARALLEL));
    assert_eq!(scope_and.value(), PyScopeAttributes::PARALLEL);

    let tool_or = PyToolAttributes::new(PyToolAttributes::REMOTE).__or__(&PyToolAttributes::new(0));
    assert_eq!(tool_or.value(), PyToolAttributes::REMOTE);
    let tool_and = tool_or.__and__(&PyToolAttributes::new(PyToolAttributes::REMOTE));
    assert_eq!(tool_and.value(), PyToolAttributes::REMOTE);

    let llm_or = PyLLMAttributes::new(PyLLMAttributes::STATEFUL)
        .__or__(&PyLLMAttributes::new(PyLLMAttributes::STREAMING));
    assert_eq!(
        llm_or.value(),
        PyLLMAttributes::STATEFUL | PyLLMAttributes::STREAMING
    );
    let llm_and = llm_or.__and__(&PyLLMAttributes::new(PyLLMAttributes::STREAMING));
    assert_eq!(llm_and.value(), PyLLMAttributes::STREAMING);

    Python::attach(|py| {
        let stack = PyScopeStack(nemo_relay::api::runtime::create_scope_stack());
        assert_eq!(stack.__repr__(), "<ScopeStack>");

        let parent_uuid = Uuid::now_v7();
        let scope = PyScopeHandle::from(
            ScopeHandle::builder()
                .name("scope")
                .scope_type(ScopeType::Agent)
                .attributes(ScopeAttributes::PARALLEL | ScopeAttributes::RELOCATABLE)
                .parent_uuid(parent_uuid)
                .data(json!({"scope": true}))
                .metadata(json!({"scope_meta": true}))
                .build(),
        );
        assert!(!scope.uuid().is_empty());
        assert_eq!(scope.name(), "scope");
        assert_eq!(
            scope.attributes().value(),
            PyScopeAttributes::PARALLEL | PyScopeAttributes::RELOCATABLE
        );

        let tool = PyToolHandle::from(
            ToolHandle::builder()
                .name("tool")
                .attributes(ToolAttributes::REMOTE)
                .parent_uuid(parent_uuid)
                .data(json!({"tool": true}))
                .metadata(json!({"tool_meta": true}))
                .build(),
        );
        assert!(!tool.uuid().is_empty());
        assert_eq!(tool.name(), "tool");

        let llm = PyLLMHandle::from(
            LlmHandle::builder()
                .name("llm")
                .attributes(LlmAttributes::STATEFUL | LlmAttributes::STREAMING)
                .parent_uuid(parent_uuid)
                .data(json!({"llm": true}))
                .metadata(json!({"llm_meta": true}))
                .build(),
        );
        assert!(!llm.uuid().is_empty());
        assert_eq!(llm.name(), "llm");

        let headers = PyDict::new(py);
        headers.set_item("x-trace", "1").unwrap();
        let content = json_to_py(py, &json!({"model": "demo", "messages": []})).unwrap();
        let request = PyLLMRequest::new(headers.as_any(), content.bind(py)).unwrap();
        assert_eq!(
            py_to_json(request.headers(py).unwrap().bind(py)).unwrap(),
            json!({"x-trace": "1"})
        );
        assert_eq!(
            py_to_json(request.content(py).unwrap().bind(py)).unwrap(),
            json!({"model": "demo", "messages": []})
        );
        assert_eq!(request.__repr__(), "LLMRequest(...)");

        let annotated_request = AnnotatedLLMRequest {
            messages: vec![
                Message::System {
                    content: MessageContent::Text("system".into()),
                    name: None,
                },
                Message::User {
                    content: MessageContent::Text("user".into()),
                    name: None,
                },
            ],
            model: Some("codec-model".into()),
            params: None,
            tools: None,
            tool_choice: None,
            store: None,
            previous_response_id: None,
            truncation: None,
            reasoning: None,
            include: None,
            user: None,
            metadata: None,
            service_tier: None,
            parallel_tool_calls: None,
            max_output_tokens: None,
            max_tool_calls: None,
            top_logprobs: None,
            stream: None,
            extra: serde_json::Map::new(),
        };
        let annotated_response = AnnotatedLLMResponse {
            id: Some("resp-1".into()),
            model: Some("codec-model".into()),
            message: Some(MessageContent::Text("done".into())),
            tool_calls: Some(vec![ResponseToolCall {
                id: "call-1".into(),
                name: "lookup".into(),
                arguments: json!({"city": "NYC"}),
            }]),
            finish_reason: Some(FinishReason::Complete),
            usage: Some(Usage {
                prompt_tokens: Some(1),
                completion_tokens: Some(2),
                total_tokens: Some(3),
                cache_read_tokens: None,
                cache_write_tokens: None,
                cost: None,
            }),
            api_specific: Some(ApiSpecificResponse::Custom {
                api_name: "custom".into(),
                data: json!({"ok": true}),
            }),
            optimization_summary: None,
            extra: serde_json::Map::from_iter([("extra".into(), json!(true))]),
        };

        let scope_start = match Event::Scope(ScopeEvent::new(
            base_event(
                parent_uuid,
                "scope-start",
                json!({"phase": "start"}),
                json!({"meta": true}),
            ),
            ScopeCategory::Start,
            scope_attributes_to_strings(ScopeAttributes::PARALLEL),
            EventCategory::agent(),
            None,
        )) {
            Event::Scope(inner) => PyScopeEvent { inner },
            _ => unreachable!(),
        };
        assert_eq!(scope_start.kind(), "scope");
        assert_eq!(scope_start.scope_category(), "start");
        assert_eq!(scope_start.name(), "scope-start");
        assert_eq!(scope_start.category(), "agent");
        assert_eq!(scope_start.attributes(), vec!["parallel".to_string()]);

        let scope_end = match Event::Scope(ScopeEvent::new(
            base_event(
                parent_uuid,
                "scope-end",
                json!({"phase": "end"}),
                json!({"meta": true}),
            ),
            ScopeCategory::End,
            scope_attributes_to_strings(ScopeAttributes::RELOCATABLE),
            EventCategory::tool(),
            None,
        )) {
            Event::Scope(inner) => PyScopeEvent { inner },
            _ => unreachable!(),
        };
        assert_eq!(scope_end.kind(), "scope");
        assert_eq!(scope_end.scope_category(), "end");
        assert_eq!(scope_end.category(), "tool");

        let tool_end = match Event::Scope(ScopeEvent::new(
            base_event(
                parent_uuid,
                "tool-end",
                json!({"output": 1}),
                json!({"meta": true}),
            ),
            ScopeCategory::End,
            tool_attributes_to_strings(ToolAttributes::REMOTE),
            EventCategory::tool(),
            Some(CategoryProfile::builder().tool_call_id("call-1").build()),
        )) {
            Event::Scope(inner) => PyScopeEvent { inner },
            _ => unreachable!(),
        };
        assert_eq!(tool_end.kind(), "scope");
        assert_eq!(tool_end.scope_category(), "end");
        assert_eq!(
            py_to_json(tool_end.data(py).unwrap().bind(py)).unwrap(),
            json!({"output": 1})
        );
        assert_eq!(
            py_to_json(tool_end.category_profile(py).unwrap().bind(py)).unwrap(),
            json!({"tool_call_id": "call-1"})
        );

        let llm_start = match Event::Scope(ScopeEvent::new(
            base_event(
                parent_uuid,
                "llm-start",
                json!({"input": true}),
                json!({"meta": true}),
            ),
            ScopeCategory::Start,
            llm_attributes_to_strings(LlmAttributes::STATEFUL),
            EventCategory::llm(),
            Some(
                CategoryProfile::builder()
                    .model_name("demo-model")
                    .annotated_request(std::sync::Arc::new(annotated_request.clone()))
                    .build(),
            ),
        )) {
            Event::Scope(inner) => PyScopeEvent { inner },
            _ => unreachable!(),
        };
        let mut expected_start_profile = json!({"model_name": "demo-model"});
        expected_start_profile.as_object_mut().unwrap().insert(
            "annotated_request".into(),
            serde_json::to_value(&annotated_request).unwrap(),
        );
        assert_eq!(
            py_to_json(llm_start.category_profile(py).unwrap().bind(py)).unwrap(),
            expected_start_profile
        );

        let llm_end = match Event::Scope(ScopeEvent::new(
            base_event(
                parent_uuid,
                "llm-end",
                json!({"output": true}),
                json!({"meta": true}),
            ),
            ScopeCategory::End,
            llm_attributes_to_strings(LlmAttributes::STREAMING),
            EventCategory::llm(),
            Some(
                CategoryProfile::builder()
                    .model_name("demo-model")
                    .annotated_response(std::sync::Arc::new(annotated_response.clone()))
                    .build(),
            ),
        )) {
            Event::Scope(inner) => PyScopeEvent { inner },
            _ => unreachable!(),
        };
        let mut expected_end_profile = json!({"model_name": "demo-model"});
        expected_end_profile.as_object_mut().unwrap().insert(
            "annotated_response".into(),
            serde_json::to_value(&annotated_response).unwrap(),
        );
        assert_eq!(
            py_to_json(llm_end.category_profile(py).unwrap().bind(py)).unwrap(),
            expected_end_profile
        );

        let mark = match Event::Mark(MarkEvent::new(
            base_event(
                parent_uuid,
                "mark",
                json!({"mark": true}),
                json!({"meta": true}),
            ),
            None,
            None,
        )) {
            Event::Mark(inner) => PyMarkEvent { inner },
            _ => unreachable!(),
        };
        assert_eq!(mark.kind(), "mark");
        assert_eq!(mark.name(), "mark");

        with_event_loop(py, |event_loop| {
            let runner = PyModule::from_code(
                py,
                &CString::new(
                    r#"
async def next_item(stream):
    return await stream.__anext__()
"#,
                )
                .unwrap(),
                &CString::new("py_types_stream_runner.py").unwrap(),
                &CString::new("py_types_stream_runner").unwrap(),
            )
            .unwrap();
            let (tx_ok, rx_ok) = tokio::sync::mpsc::channel(2);
            tx_ok.blocking_send(Ok(json!({"chunk": 1}))).unwrap();
            drop(tx_ok);
            let stream_ok = pyo3::Py::new(
                py,
                PyLlmStream {
                    receiver: tokio::sync::Mutex::new(rx_ok),
                },
            )
            .unwrap();
            {
                let ok_ref = stream_ok.bind(py).borrow();
                let _ = PyLlmStream::__aiter__(ok_ref);
            }
            let ok_chunk = event_loop
                .call_method1(
                    "run_until_complete",
                    (runner
                        .getattr("next_item")
                        .unwrap()
                        .call1((stream_ok.clone_ref(py),))
                        .unwrap(),),
                )
                .unwrap();
            assert_eq!(
                crate::convert::py_to_json(&ok_chunk).unwrap(),
                json!({"chunk": 1})
            );

            let (tx_err, rx_err) = tokio::sync::mpsc::channel(1);
            tx_err
                .blocking_send(Err(nemo_relay::error::FlowError::Internal(
                    "stream boom".into(),
                )))
                .unwrap();
            drop(tx_err);
            let stream_err = pyo3::Py::new(
                py,
                PyLlmStream {
                    receiver: tokio::sync::Mutex::new(rx_err),
                },
            )
            .unwrap();
            let err = event_loop
                .call_method1(
                    "run_until_complete",
                    (runner
                        .getattr("next_item")
                        .unwrap()
                        .call1((stream_err.clone_ref(py),))
                        .unwrap(),),
                )
                .unwrap_err();
            assert!(err.to_string().contains("stream boom"));

            let (tx_done, rx_done) = tokio::sync::mpsc::channel(1);
            drop(tx_done);
            let stream_done = pyo3::Py::new(
                py,
                PyLlmStream {
                    receiver: tokio::sync::Mutex::new(rx_done),
                },
            )
            .unwrap();
            let stop = event_loop
                .call_method1(
                    "run_until_complete",
                    (runner
                        .getattr("next_item")
                        .unwrap()
                        .call1((stream_done.clone_ref(py),))
                        .unwrap(),),
                )
                .unwrap_err();
            assert!(stop.to_string().contains("StopAsyncIteration"));
        });
    });
}

#[test]
fn test_python_side_core_type_constructors_cover_exposed_entrypoints() {
    let _python = crate::test_support::init_python_test();
    Python::attach(|py| {
        let module = PyModule::new(py, "_types_python_side").unwrap();
        register(&module).unwrap();

        let scope_attrs = module
            .getattr("ScopeAttributes")
            .unwrap()
            .call1((3_u32,))
            .unwrap();
        assert_eq!(
            scope_attrs
                .getattr("value")
                .unwrap()
                .extract::<u32>()
                .unwrap(),
            3
        );
        assert!(
            scope_attrs
                .getattr("is_parallel")
                .unwrap()
                .extract::<bool>()
                .unwrap()
        );
        assert!(
            scope_attrs
                .repr()
                .unwrap()
                .to_str()
                .unwrap()
                .contains("ScopeAttributes")
        );

        let tool_attrs = module
            .getattr("ToolAttributes")
            .unwrap()
            .call1((1_u32,))
            .unwrap();
        assert_eq!(
            tool_attrs
                .getattr("value")
                .unwrap()
                .extract::<u32>()
                .unwrap(),
            1
        );
        assert!(
            tool_attrs
                .getattr("is_remote")
                .unwrap()
                .extract::<bool>()
                .unwrap()
        );

        let llm_attrs = module
            .getattr("LLMAttributes")
            .unwrap()
            .call1((3_u32,))
            .unwrap();
        assert_eq!(
            llm_attrs
                .getattr("value")
                .unwrap()
                .extract::<u32>()
                .unwrap(),
            3
        );
        assert!(
            llm_attrs
                .getattr("is_stateful")
                .unwrap()
                .extract::<bool>()
                .unwrap()
        );
        assert!(
            llm_attrs
                .getattr("is_streaming")
                .unwrap()
                .extract::<bool>()
                .unwrap()
        );

        let scope_type = module
            .getattr("ScopeType")
            .unwrap()
            .getattr("Agent")
            .unwrap();
        assert!(
            scope_type
                .repr()
                .unwrap()
                .to_str()
                .unwrap()
                .contains("ScopeType.Agent")
        );

        let headers = PyDict::new(py);
        headers.set_item("x-trace", "1").unwrap();
        let content = json_to_py(py, &json!({"model": "demo", "messages": []})).unwrap();
        let request = module
            .getattr("LLMRequest")
            .unwrap()
            .call1((headers, content.bind(py)))
            .unwrap();
        assert_eq!(
            py_to_json(request.getattr("headers").unwrap().as_any()).unwrap(),
            json!({"x-trace": "1"})
        );
        assert_eq!(
            py_to_json(request.getattr("content").unwrap().as_any()).unwrap(),
            json!({"model": "demo", "messages": []})
        );

        let contribution_fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../types/tests/fixtures/llm_optimization_contribution_v1.json"
        ))
        .unwrap();
        let contributions = json_to_py(py, &json!([contribution_fixture.clone()])).unwrap();
        let kwargs = PyDict::new(py);
        kwargs
            .set_item("optimization_contributions", contributions)
            .unwrap();
        let outcome = module
            .getattr("LLMRequestInterceptOutcome")
            .unwrap()
            .call((request.clone(),), Some(&kwargs))
            .unwrap();
        let round_trip = py_to_json(
            outcome
                .getattr("optimization_contributions")
                .unwrap()
                .as_any(),
        )
        .unwrap();
        assert_eq!(round_trip, json!([contribution_fixture]));
        assert_eq!(
            round_trip[0]["future_top_level_field"],
            json!({"preserved": true})
        );

        let bad_headers = PyList::empty(py);
        let err = module
            .getattr("LLMRequest")
            .unwrap()
            .call1((bad_headers, py.None()))
            .unwrap_err();
        assert!(err.to_string().contains("not an instance of 'dict'"));
    });
}

#[test]
fn test_annotated_llm_types_and_builtin_codecs_cover_mutators_and_codecs() {
    let _python = crate::test_support::init_python_test();
    Python::attach(|py| {
        let messages = json_to_py(
            py,
            &json!([
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "Where is the weather?"},
                {
                    "role": "assistant",
                    "content": "Calling tool",
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{\"city\":\"NYC\"}"}
                    }]
                }
            ]),
        )
        .unwrap();
        let params = json_to_py(
            py,
            &json!({"temperature": 0.2, "max_tokens": 64, "top_p": 0.9, "stop": ["DONE"]}),
        )
        .unwrap();
        let tools = json_to_py(
            py,
            &json!([{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up weather",
                    "parameters": {"type": "object"}
                }
            }]),
        )
        .unwrap();
        let tool_choice = json_to_py(
            py,
            &json!({"type": "function", "function": {"name": "lookup"}}),
        )
        .unwrap();
        let extra = json_to_py(py, &json!({"provider": "test"})).unwrap();

        let mut annotated = PyAnnotatedLLMRequest::new(
            messages.bind(py),
            Some("demo-model".into()),
            Some(params.bind(py)),
            Some(tools.bind(py)),
            Some(tool_choice.bind(py)),
            Some(extra.bind(py)),
        )
        .unwrap();
        assert_eq!(annotated.model(), Some("demo-model".into()));
        assert_eq!(annotated.system_prompt(), Some("You are terse.".into()));
        assert_eq!(
            annotated.last_user_message(),
            Some("Where is the weather?".into())
        );
        assert!(annotated.has_tool_calls());
        assert!(annotated.__repr__().contains("AnnotatedLLMRequest"));
        assert_eq!(
            py_to_json(annotated.messages(py).unwrap().bind(py)).unwrap()[0]["role"],
            json!("system")
        );
        assert_eq!(
            py_to_json(annotated.params(py).unwrap().bind(py)).unwrap()["max_tokens"],
            json!(64)
        );
        assert_eq!(
            py_to_json(annotated.tools(py).unwrap().bind(py)).unwrap()[0]["function"]["name"],
            json!("lookup")
        );
        assert_eq!(
            py_to_json(annotated.tool_choice(py).unwrap().bind(py)).unwrap()["function"]["name"],
            json!("lookup")
        );
        assert_eq!(
            py_to_json(annotated.extra(py).unwrap().bind(py)).unwrap()["provider"],
            json!("test")
        );
        assert_eq!(annotated.store(), None);
        assert_eq!(annotated.previous_response_id(), None);
        assert!(annotated.truncation(py).unwrap().bind(py).is_none());
        assert!(annotated.reasoning(py).unwrap().bind(py).is_none());
        assert!(annotated.include(py).unwrap().bind(py).is_none());
        assert_eq!(annotated.user(), None);
        assert!(annotated.metadata(py).unwrap().bind(py).is_none());
        assert_eq!(annotated.service_tier(), None);
        assert_eq!(annotated.parallel_tool_calls(), None);
        assert_eq!(annotated.max_output_tokens(), None);
        assert_eq!(annotated.max_tool_calls(), None);
        assert_eq!(annotated.top_logprobs(), None);
        assert_eq!(annotated.stream(), None);

        let updated_messages =
            json_to_py(py, &json!([{"role": "user", "content": "updated"}])).unwrap();
        annotated.set_messages(updated_messages.bind(py)).unwrap();
        annotated.set_model(Some("updated-model".into()));
        let updated_params = json_to_py(py, &json!({"temperature": 0.7})).unwrap();
        annotated.set_params(updated_params.bind(py)).unwrap();
        let updated_tools = json_to_py(
            py,
            &json!([{
                "type": "function",
                "function": {"name": "updated", "parameters": {"type": "object"}}
            }]),
        )
        .unwrap();
        annotated.set_tools(updated_tools.bind(py)).unwrap();
        let updated_choice = json_to_py(py, &json!("auto")).unwrap();
        annotated.set_tool_choice(updated_choice.bind(py)).unwrap();
        annotated.set_store(Some(true));
        annotated.set_previous_response_id(Some("resp_1".into()));
        let updated_truncation = json_to_py(py, &json!("disabled")).unwrap();
        annotated
            .set_truncation(updated_truncation.bind(py))
            .unwrap();
        let updated_reasoning = json_to_py(py, &json!({"effort": "low"})).unwrap();
        annotated.set_reasoning(updated_reasoning.bind(py)).unwrap();
        let updated_include = json_to_py(py, &json!(["reasoning.encrypted_content"])).unwrap();
        annotated.set_include(updated_include.bind(py)).unwrap();
        annotated.set_user(Some("user-1".into()));
        let updated_metadata = json_to_py(py, &json!({"tenant": "qa"})).unwrap();
        annotated.set_metadata(updated_metadata.bind(py)).unwrap();
        annotated.set_service_tier(Some("default".into()));
        annotated.set_parallel_tool_calls(Some(false));
        annotated.set_max_output_tokens(Some(128));
        annotated.set_max_tool_calls(Some(3));
        annotated.set_top_logprobs(Some(2));
        annotated.set_stream(Some(true));
        let updated_extra = json_to_py(py, &json!({"updated": true})).unwrap();
        annotated.set_extra(updated_extra.bind(py)).unwrap();
        assert_eq!(annotated.model(), Some("updated-model".into()));
        assert_eq!(annotated.last_user_message(), Some("updated".into()));
        assert_eq!(annotated.store(), Some(true));
        assert_eq!(annotated.previous_response_id(), Some("resp_1".into()));
        assert_eq!(
            py_to_json(annotated.truncation(py).unwrap().bind(py)).unwrap(),
            json!("disabled")
        );
        assert_eq!(
            py_to_json(annotated.reasoning(py).unwrap().bind(py)).unwrap(),
            json!({"effort": "low"})
        );
        assert_eq!(
            py_to_json(annotated.include(py).unwrap().bind(py)).unwrap(),
            json!(["reasoning.encrypted_content"])
        );
        assert_eq!(annotated.user(), Some("user-1".into()));
        assert_eq!(
            py_to_json(annotated.metadata(py).unwrap().bind(py)).unwrap(),
            json!({"tenant": "qa"})
        );
        assert_eq!(annotated.service_tier(), Some("default".into()));
        assert_eq!(annotated.parallel_tool_calls(), Some(false));
        assert_eq!(annotated.max_output_tokens(), Some(128));
        assert_eq!(annotated.max_tool_calls(), Some(3));
        assert_eq!(annotated.top_logprobs(), Some(2));
        assert_eq!(annotated.stream(), Some(true));
        assert_eq!(
            py_to_json(annotated.extra(py).unwrap().bind(py)).unwrap(),
            json!({"updated": true})
        );

        annotated.set_params(py.None().bind(py)).unwrap();
        annotated.set_tools(py.None().bind(py)).unwrap();
        annotated.set_tool_choice(py.None().bind(py)).unwrap();
        annotated.set_truncation(py.None().bind(py)).unwrap();
        annotated.set_reasoning(py.None().bind(py)).unwrap();
        annotated.set_include(py.None().bind(py)).unwrap();
        annotated.set_metadata(py.None().bind(py)).unwrap();
        assert!(annotated.params(py).unwrap().bind(py).is_none());
        assert!(annotated.tools(py).unwrap().bind(py).is_none());
        assert!(annotated.tool_choice(py).unwrap().bind(py).is_none());
        assert!(annotated.truncation(py).unwrap().bind(py).is_none());
        assert!(annotated.reasoning(py).unwrap().bind(py).is_none());
        assert!(annotated.include(py).unwrap().bind(py).is_none());
        assert!(annotated.metadata(py).unwrap().bind(py).is_none());

        let bad_messages = json_to_py(py, &json!([{"content": "missing role"}])).unwrap();
        let err = PyAnnotatedLLMRequest::new(bad_messages.bind(py), None, None, None, None, None)
            .err()
            .unwrap();
        assert!(err.to_string().contains("invalid messages"));
        let bad_params = json_to_py(py, &json!({"temperature": "hot"})).unwrap();
        assert!(annotated.set_params(bad_params.bind(py)).is_err());
        let bad_tools = json_to_py(py, &json!({"tool": "bad"})).unwrap();
        assert!(annotated.set_tools(bad_tools.bind(py)).is_err());
        let bad_choice = json_to_py(py, &json!({"bad": true})).unwrap();
        assert!(annotated.set_tool_choice(bad_choice.bind(py)).is_err());
        let bad_extra = PyList::empty(py);
        assert!(annotated.set_extra(&bad_extra.into_any()).is_err());

        let response = PyAnnotatedLLMResponse {
            inner: AnnotatedLLMResponse {
                id: Some("resp-42".into()),
                model: Some("demo-model".into()),
                message: Some(MessageContent::Text("hello".into())),
                tool_calls: Some(vec![ResponseToolCall {
                    id: "call-1".into(),
                    name: "lookup".into(),
                    arguments: json!({"city": "NYC"}),
                }]),
                finish_reason: Some(FinishReason::Complete),
                usage: Some(Usage {
                    prompt_tokens: Some(2),
                    completion_tokens: Some(3),
                    total_tokens: Some(5),
                    cache_read_tokens: Some(1),
                    cache_write_tokens: None,
                    cost: Some(CostEstimate {
                        total: Some(0.000_001),
                        currency: "USD".into(),
                        input: Some(0.000_000_2),
                        output: Some(0.000_000_8),
                        cache_read: None,
                        cache_write: None,
                        source: CostSource::ProviderReported,
                        pricing_provider: Some("test-provider".into()),
                        pricing_model: Some("demo-model".into()),
                        pricing_as_of: Some("2026-06-04".into()),
                        pricing_source: Some("https://example.test/pricing".into()),
                    }),
                }),
                api_specific: Some(ApiSpecificResponse::Custom {
                    api_name: "custom".into(),
                    data: json!({"debug": true}),
                }),
                optimization_summary: Some(
                    serde_json::from_value(json!({
                        "schema_version": "1",
                        "calculation_version": "1",
                        "status": "partial",
                        "limitations": ["missing_pricing"],
                        "tokens_saved": {"prompt_tokens": 2, "total_tokens": 2},
                        "contributions": []
                    }))
                    .unwrap(),
                ),
                extra: serde_json::Map::from_iter([("trace".into(), json!("abc"))]),
            },
        };
        assert_eq!(response.id(), Some("resp-42".into()));
        assert_eq!(response.model(), Some("demo-model".into()));
        assert_eq!(
            py_to_json(response.message(py).unwrap().bind(py)).unwrap(),
            json!("hello")
        );
        assert_eq!(
            py_to_json(response.tool_calls(py).unwrap().bind(py)).unwrap()[0]["name"],
            json!("lookup")
        );
        assert_eq!(response.finish_reason(), Some("complete".into()));
        assert_eq!(
            py_to_json(response.usage(py).unwrap().bind(py)).unwrap()["total_tokens"],
            json!(5)
        );
        assert_eq!(
            py_to_json(response.usage(py).unwrap().bind(py)).unwrap()["cost"]["pricing_provider"],
            json!("test-provider")
        );
        assert_eq!(
            py_to_json(response.optimization_summary(py).unwrap().bind(py)).unwrap()["tokens_saved"]
                ["prompt_tokens"],
            json!(2)
        );
        assert_eq!(
            py_to_json(response.api_specific(py).unwrap().bind(py)).unwrap()["api_name"],
            json!("custom")
        );
        assert_eq!(
            py_to_json(response.extra(py).unwrap().bind(py)).unwrap()["trace"],
            json!("abc")
        );
        assert_eq!(response.response_text(), Some("hello".into()));
        assert!(response.has_tool_calls());
        assert!(response.__repr__().contains("AnnotatedLLMResponse"));

        let response_without_api_specific = PyAnnotatedLLMResponse {
            inner: AnnotatedLLMResponse {
                id: None,
                model: None,
                message: None,
                tool_calls: None,
                finish_reason: None,
                usage: None,
                api_specific: None,
                optimization_summary: None,
                extra: serde_json::Map::new(),
            },
        };
        assert!(
            response_without_api_specific
                .api_specific(py)
                .unwrap()
                .bind(py)
                .is_none()
        );
        assert!(
            response_without_api_specific
                .optimization_summary(py)
                .unwrap()
                .bind(py)
                .is_none()
        );

        let chat_request = PyLLMRequest {
            inner: nemo_relay::api::llm::LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 16
                }),
            },
        };
        let chat_codec = PyOpenAIChatCodec::new();
        let chat_decoded = chat_codec.decode(&chat_request).unwrap();
        assert_eq!(chat_decoded.model(), Some("gpt-4o-mini".into()));
        let chat_encoded = chat_codec.encode(&chat_decoded, &chat_request).unwrap();
        assert_eq!(chat_encoded.inner.content["model"], json!("gpt-4o-mini"));
        let chat_response = chat_codec
            .decode_response(
                json_to_py(
                    py,
                    &json!({
                        "id": "chatcmpl-1",
                        "model": "gpt-4o-mini",
                        "choices": [{
                            "message": {"role": "assistant", "content": "hello"},
                            "finish_reason": "stop"
                        }]
                    }),
                )
                .unwrap()
                .bind(py),
            )
            .unwrap();
        assert_eq!(chat_response.response_text(), Some("hello".into()));
        assert_eq!(chat_codec.__repr__(), "<OpenAIChatCodec>");

        let responses_request = PyLLMRequest {
            inner: nemo_relay::api::llm::LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({
                    "model": "gpt-4o-mini",
                    "instructions": "Be helpful",
                    "input": [{"role": "user", "content": "hi"}],
                    "max_output_tokens": 32
                }),
            },
        };
        let responses_codec = PyOpenAIResponsesCodec::new();
        let responses_decoded = responses_codec.decode(&responses_request).unwrap();
        assert_eq!(responses_decoded.system_prompt(), Some("Be helpful".into()));
        let responses_encoded = responses_codec
            .encode(&responses_decoded, &responses_request)
            .unwrap();
        assert_eq!(
            responses_encoded.inner.content["instructions"],
            json!("Be helpful")
        );
        let responses_response = responses_codec
            .decode_response(
                json_to_py(
                    py,
                    &json!({
                        "id": "resp-1",
                        "model": "gpt-4o-mini",
                        "status": "completed",
                        "output": [{
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "done"}]
                        }]
                    }),
                )
                .unwrap()
                .bind(py),
            )
            .unwrap();
        assert_eq!(responses_response.response_text(), Some("done".into()));
        assert_eq!(responses_codec.__repr__(), "<OpenAIResponsesCodec>");

        let anthropic_request = PyLLMRequest {
            inner: nemo_relay::api::llm::LlmRequest {
                headers: serde_json::Map::new(),
                content: json!({
                    "model": "claude-sonnet-4-20250514",
                    "system": "Be careful",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 64
                }),
            },
        };
        let anthropic_codec = PyAnthropicMessagesCodec::new();
        let anthropic_decoded = anthropic_codec.decode(&anthropic_request).unwrap();
        assert_eq!(anthropic_decoded.system_prompt(), Some("Be careful".into()));
        let anthropic_encoded = anthropic_codec
            .encode(&anthropic_decoded, &anthropic_request)
            .unwrap();
        assert_eq!(
            anthropic_encoded.inner.content["system"],
            json!("Be careful")
        );
        let anthropic_response = anthropic_codec
            .decode_response(
                json_to_py(
                    py,
                    &json!({
                        "id": "msg-1",
                        "model": "claude-sonnet-4-20250514",
                        "content": [{"type": "text", "text": "done"}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 1, "output_tokens": 2}
                    }),
                )
                .unwrap()
                .bind(py),
            )
            .unwrap();
        assert_eq!(anthropic_response.response_text(), Some("done".into()));
        assert_eq!(anthropic_codec.__repr__(), "<AnthropicMessagesCodec>");
    });
}

#[test]
fn test_forced_serialization_error_hooks_cover_unreachable_wrappers() {
    let _python = crate::test_support::init_python_test();
    Python::attach(|py| {
        let tool_def = json_to_py(py, &json!({"name": "typed_tool"})).unwrap();
        let tool_defs = PyList::empty(py);
        tool_defs.append(tool_def.bind(py)).unwrap();
        let extra = json_to_py(py, &json!({"team": "qa"})).unwrap();

        let exporter = PyAtifExporter::new(
            "session-serialization".into(),
            "py-agent".into(),
            "1.0.0".into(),
            Some("typed-model".into()),
            Some(&tool_defs),
            Some(extra.bind(py)),
        )
        .unwrap();

        let annotated = PyAnnotatedLLMRequest {
            inner: AnnotatedLLMRequest {
                messages: vec![
                    Message::System {
                        content: MessageContent::Text("system".into()),
                        name: None,
                    },
                    Message::User {
                        content: MessageContent::Text("user".into()),
                        name: None,
                    },
                ],
                model: Some("demo-model".into()),
                params: Some(nemo_relay::codec::request::GenerationParams {
                    temperature: Some(0.1),
                    max_tokens: Some(8),
                    ..Default::default()
                }),
                tools: Some(vec![nemo_relay::codec::request::ToolDefinition {
                    tool_type: "function".into(),
                    function: nemo_relay::codec::request::FunctionDefinition {
                        name: "lookup".into(),
                        description: None,
                        parameters: Some(json!({"type": "object"})),
                    },
                }]),
                tool_choice: Some(nemo_relay::codec::request::ToolChoice::Auto),
                store: None,
                previous_response_id: None,
                truncation: None,
                reasoning: None,
                include: None,
                user: None,
                metadata: None,
                service_tier: None,
                parallel_tool_calls: None,
                max_output_tokens: None,
                max_tool_calls: None,
                top_logprobs: None,
                stream: None,
                extra: serde_json::Map::new(),
            },
        };

        let response = PyAnnotatedLLMResponse {
            inner: AnnotatedLLMResponse {
                id: Some("resp-42".into()),
                model: Some("demo-model".into()),
                message: Some(MessageContent::Text("hello".into())),
                tool_calls: Some(vec![ResponseToolCall {
                    id: "call-1".into(),
                    name: "lookup".into(),
                    arguments: json!({"city": "NYC"}),
                }]),
                finish_reason: Some(FinishReason::Complete),
                usage: Some(Usage {
                    prompt_tokens: Some(2),
                    completion_tokens: Some(3),
                    total_tokens: Some(5),
                    cache_read_tokens: Some(1),
                    cache_write_tokens: None,
                    cost: None,
                }),
                api_specific: Some(ApiSpecificResponse::Custom {
                    api_name: "custom".into(),
                    data: json!({"debug": true}),
                }),
                optimization_summary: Some(
                    serde_json::from_value(json!({
                        "schema_version": "1",
                        "calculation_version": "1",
                        "status": "partial",
                        "limitations": ["test"],
                        "tokens_saved": {},
                        "contributions": []
                    }))
                    .unwrap(),
                ),
                extra: serde_json::Map::new(),
            },
        };

        type ForcedCaseFn = fn(
            Python<'_>,
            &PyAtifExporter,
            &PyAnnotatedLLMRequest,
            &PyAnnotatedLLMResponse,
        ) -> PyResult<()>;
        type ForcedCase<'a> = (u64, &'a str, ForcedCaseFn);

        let forced_cases: &[ForcedCase<'_>] = &[
            (
                FORCE_ATIF_EXPORT_VALUE_SERIALIZATION_ERROR,
                "forced serialization failure",
                |py, exporter, _, _| exporter.export(py).map(|_| ()),
            ),
            (
                FORCE_ATIF_EXPORT_JSON_SERIALIZATION_ERROR,
                "forced serialization failure",
                |py, exporter, _, _| exporter.export_json(py).map(|_| ()),
            ),
            (
                FORCE_ANNOTATED_REQUEST_MESSAGES_SERIALIZATION_ERROR,
                "forced serialization failure",
                |py, _, annotated, _| annotated.messages(py).map(|_| ()),
            ),
            (
                FORCE_ANNOTATED_REQUEST_PARAMS_SERIALIZATION_ERROR,
                "forced serialization failure",
                |py, _, annotated, _| annotated.params(py).map(|_| ()),
            ),
            (
                FORCE_ANNOTATED_REQUEST_TOOLS_SERIALIZATION_ERROR,
                "forced serialization failure",
                |py, _, annotated, _| annotated.tools(py).map(|_| ()),
            ),
            (
                FORCE_ANNOTATED_REQUEST_TOOL_CHOICE_SERIALIZATION_ERROR,
                "forced serialization failure",
                |py, _, annotated, _| annotated.tool_choice(py).map(|_| ()),
            ),
            (
                FORCE_ANNOTATED_RESPONSE_MESSAGE_SERIALIZATION_ERROR,
                "forced serialization failure",
                |py, _, _, response| response.message(py).map(|_| ()),
            ),
            (
                FORCE_ANNOTATED_RESPONSE_TOOL_CALLS_SERIALIZATION_ERROR,
                "forced serialization failure",
                |py, _, _, response| response.tool_calls(py).map(|_| ()),
            ),
            (
                FORCE_ANNOTATED_RESPONSE_USAGE_SERIALIZATION_ERROR,
                "forced serialization failure",
                |py, _, _, response| response.usage(py).map(|_| ()),
            ),
            (
                FORCE_ANNOTATED_RESPONSE_OPTIMIZATION_SUMMARY_SERIALIZATION_ERROR,
                "forced serialization failure",
                |py, _, _, response| response.optimization_summary(py).map(|_| ()),
            ),
            (
                FORCE_ANNOTATED_RESPONSE_API_SPECIFIC_SERIALIZATION_ERROR,
                "forced serialization failure",
                |py, _, _, response| response.api_specific(py).map(|_| ()),
            ),
        ];

        for (mask, expected, call) in forced_cases {
            set_forced_serialization_mask_for_tests(*mask);
            let err = call(py, &exporter, &annotated, &response).unwrap_err();
            assert!(err.to_string().contains(expected));
        }
        set_forced_serialization_mask_for_tests(0);
    });
}

#[test]
fn test_python_visible_type_none_and_error_paths_cover_remaining_branches() {
    let _python = crate::test_support::init_python_test();
    Python::attach(|py| {
        let types_module = PyModule::new(py, "_types_none_paths").unwrap();
        register(&types_module).unwrap();

        let helpers = PyModule::from_code(
            py,
            &CString::new(
                r#"
def run(types):
    exporter = types.AtifExporter("session-1", "agent", "1.0.0")
    _ = exporter.export()
    _ = exporter.export_json()
    _ = repr(exporter)
    _ = exporter.deregister("missing-exporter")

    otel = types.OpenTelemetryConfig()
    otel.transport = "grpc"
    otel.endpoint = "http://127.0.0.1:4317"
    otel_grpc_error = None
    try:
        otel_subscriber = types.OpenTelemetrySubscriber(otel)
        _ = repr(otel_subscriber)
        otel_subscriber.shutdown()
    except RuntimeError as err:
        otel_grpc_error = str(err)

    oi = types.OpenInferenceConfig()
    oi.transport = "grpc"
    oi.endpoint = "http://127.0.0.1:4317"
    oi_grpc_error = None
    try:
        oi_subscriber = types.OpenInferenceSubscriber(oi)
        _ = repr(oi_subscriber)
        oi_subscriber.shutdown()
    except RuntimeError as err:
        oi_grpc_error = str(err)

    request_error = None
    try:
        types.LLMRequest([], {"model": "demo"})
    except TypeError as err:
        request_error = str(err)

    annotated = types.AnnotatedLLMRequest([{"role": "user", "content": "hello"}])
    annotated.model = None
    params_is_none = annotated.params is None
    tools_is_none = annotated.tools is None
    tool_choice_is_none = annotated.tool_choice is None

    invalid_params_error = None
    invalid_tools_error = None
    invalid_tool_choice_error = None
    invalid_extra_error = None
    invalid_messages_setter_error = None
    invalid_params_setter_error = None
    invalid_tools_setter_error = None
    invalid_tool_choice_setter_error = None
    invalid_extra_setter_error = None

    try:
        types.AnnotatedLLMRequest([{"role": "user", "content": "hello"}], params=[])
    except ValueError as err:
        invalid_params_error = str(err)

    try:
        types.AnnotatedLLMRequest([{"role": "user", "content": "hello"}], tools={})
    except ValueError as err:
        invalid_tools_error = str(err)

    try:
        types.AnnotatedLLMRequest([{"role": "user", "content": "hello"}], tool_choice=[])
    except ValueError as err:
        invalid_tool_choice_error = str(err)

    try:
        types.AnnotatedLLMRequest([{"role": "user", "content": "hello"}], extra=[])
    except ValueError as err:
        invalid_extra_error = str(err)

    try:
        annotated.messages = {}
    except ValueError as err:
        invalid_messages_setter_error = str(err)

    try:
        annotated.params = []
    except ValueError as err:
        invalid_params_setter_error = str(err)

    try:
        annotated.tools = {}
    except ValueError as err:
        invalid_tools_setter_error = str(err)

    try:
        annotated.tool_choice = []
    except ValueError as err:
        invalid_tool_choice_setter_error = str(err)

    try:
        annotated.extra = []
    except ValueError as err:
        invalid_extra_setter_error = str(err)

    chat_codec = types.OpenAIChatCodec()
    chat_response = chat_codec.decode_response({
        "id": "chatcmpl-none-1",
        "choices": [{
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop"
        }]
    })

    responses_codec = types.OpenAIResponsesCodec()
    responses_response = responses_codec.decode_response({
        "status": "completed",
        "output": []
    })

    anthropic_codec = types.AnthropicMessagesCodec()
    anthropic_response = anthropic_codec.decode_response({
        "content": [],
        "stop_reason": "end_turn"
    })

    return {
        "request_error": request_error,
        "otel_grpc_error": otel_grpc_error,
        "oi_grpc_error": oi_grpc_error,
        "params_is_none": params_is_none,
        "tools_is_none": tools_is_none,
        "tool_choice_is_none": tool_choice_is_none,
        "invalid_params_error": invalid_params_error,
        "invalid_tools_error": invalid_tools_error,
        "invalid_tool_choice_error": invalid_tool_choice_error,
        "invalid_extra_error": invalid_extra_error,
        "invalid_messages_setter_error": invalid_messages_setter_error,
        "invalid_params_setter_error": invalid_params_setter_error,
        "invalid_tools_setter_error": invalid_tools_setter_error,
        "invalid_tool_choice_setter_error": invalid_tool_choice_setter_error,
        "invalid_extra_setter_error": invalid_extra_setter_error,
        "chat_tool_calls_is_none": chat_response.tool_calls is None,
        "chat_usage_is_none": chat_response.usage is None,
        "chat_api_specific_is_none": chat_response.api_specific is None,
        "responses_message_is_none": responses_response.message is None,
        "responses_tool_calls_is_none": responses_response.tool_calls is None,
        "responses_usage_is_none": responses_response.usage is None,
        "responses_api_specific_is_none": responses_response.api_specific is None,
        "anthropic_message_is_none": anthropic_response.message is None,
        "anthropic_tool_calls_is_none": anthropic_response.tool_calls is None,
        "anthropic_api_specific_is_none": anthropic_response.api_specific is None,
        "anthropic_usage_is_none": anthropic_response.usage is None,
    }
"#,
            )
            .unwrap(),
            &CString::new("py_types_none_paths.py").unwrap(),
            &CString::new("py_types_none_paths").unwrap(),
        )
        .unwrap();

        let result = helpers
            .getattr("run")
            .unwrap()
            .call1((types_module.clone(),))
            .unwrap();
        let result_json = py_to_json(&result).unwrap();

        assert!(result_json["request_error"].as_str().is_some());
        assert!(result_json["otel_grpc_error"].as_str().is_some());
        assert!(result_json["oi_grpc_error"].is_null());
        assert_eq!(result_json["params_is_none"], json!(true));
        assert_eq!(result_json["tools_is_none"], json!(true));
        assert_eq!(result_json["tool_choice_is_none"], json!(true));
        assert!(result_json["invalid_params_error"].as_str().is_some());
        assert!(result_json["invalid_tools_error"].as_str().is_some());
        assert!(result_json["invalid_tool_choice_error"].as_str().is_some());
        assert!(result_json["invalid_extra_error"].as_str().is_some());
        assert!(
            result_json["invalid_messages_setter_error"]
                .as_str()
                .is_some()
        );
        assert!(
            result_json["invalid_params_setter_error"]
                .as_str()
                .is_some()
        );
        assert!(result_json["invalid_tools_setter_error"].as_str().is_some());
        assert!(
            result_json["invalid_tool_choice_setter_error"]
                .as_str()
                .is_some()
        );
        assert!(result_json["invalid_extra_setter_error"].as_str().is_some());
        assert!(result_json["chat_tool_calls_is_none"].as_bool().is_some());
        assert!(result_json["chat_usage_is_none"].as_bool().is_some());
        assert!(result_json["chat_api_specific_is_none"].as_bool().is_some());
        assert!(result_json["responses_message_is_none"].as_bool().is_some());
        assert!(
            result_json["responses_tool_calls_is_none"]
                .as_bool()
                .is_some()
        );
        assert!(result_json["responses_usage_is_none"].as_bool().is_some());
        assert!(
            result_json["responses_api_specific_is_none"]
                .as_bool()
                .is_some()
        );
        assert!(result_json["anthropic_message_is_none"].as_bool().is_some());
        assert!(
            result_json["anthropic_tool_calls_is_none"]
                .as_bool()
                .is_some()
        );
        assert!(
            result_json["anthropic_api_specific_is_none"]
                .as_bool()
                .is_some()
        );
        assert!(result_json["anthropic_usage_is_none"].as_bool().is_some());
    });
}

#[test]
fn test_python_visible_wrappers_cover_pyclass_trampolines() {
    let _python = crate::test_support::init_python_test();
    Python::attach(|py| {
        let types_module = PyModule::new(py, "_types_py_visible").unwrap();
        register(&types_module).unwrap();
        let api_module = PyModule::new(py, "_types_py_visible_api").unwrap();
        crate::py_api::register(&api_module).unwrap();

        let helpers = PyModule::from_code(
            py,
            &CString::new(
                r#"
events = []

def subscriber(event):
    payload = {
        "kind": event.kind,
        "uuid": event.uuid,
        "name": event.name,
        "parent": event.parent_uuid,
        "timestamp": event.timestamp,
        "data": event.data,
        "metadata": event.metadata,
    }
    if hasattr(event, "attributes"):
        payload["attributes"] = event.attributes
    if hasattr(event, "scope_category"):
        payload["scope_category"] = event.scope_category
    if hasattr(event, "category"):
        payload["category"] = event.category
    if hasattr(event, "category_profile"):
        payload["category_profile"] = event.category_profile
    if hasattr(event, "data_schema"):
        payload["data_schema"] = event.data_schema
    if hasattr(event, "annotated_request"):
        payload["annotated_request"] = None if event.annotated_request is None else event.annotated_request.model
    if hasattr(event, "annotated_response"):
        payload["annotated_response"] = None if event.annotated_response is None else event.annotated_response.model
    events.append(payload)

def run(types, api):
    sa = types.ScopeAttributes(types.ScopeAttributes.PARALLEL | types.ScopeAttributes.RELOCATABLE)
    ta = types.ToolAttributes(types.ToolAttributes.REMOTE)
    la = types.LLMAttributes(types.LLMAttributes.STATEFUL | types.LLMAttributes.STREAMING)
    _ = sa.is_parallel
    _ = sa.is_relocatable
    _ = sa | types.ScopeAttributes(types.ScopeAttributes.PARALLEL)
    _ = sa & types.ScopeAttributes(types.ScopeAttributes.PARALLEL)
    _ = repr(sa)
    _ = ta.is_remote
    _ = ta | types.ToolAttributes(types.ToolAttributes.REMOTE)
    _ = ta & types.ToolAttributes(types.ToolAttributes.REMOTE)
    _ = repr(ta)
    _ = la.is_stateful
    _ = la.is_streaming
    _ = la | types.LLMAttributes(types.LLMAttributes.STREAMING)
    _ = la & types.LLMAttributes(types.LLMAttributes.STREAMING)
    _ = repr(la)

    otel = types.OpenTelemetryConfig()
    otel.headers = {"authorization": "Bearer token"}
    otel.resource_attributes = {"env": "test"}
    _ = otel.headers
    _ = otel.resource_attributes
    _ = repr(otel)

    oi = types.OpenInferenceConfig()
    oi.headers = {"authorization": "Bearer token"}
    oi.resource_attributes = {"env": "test"}
    _ = oi.headers
    _ = oi.resource_attributes
    _ = repr(oi)

    request = types.LLMRequest({"x-trace": "1"}, {"model": "demo-model", "messages": []})
    _ = request.headers
    _ = request.content
    _ = repr(request)

    annotated = types.AnnotatedLLMRequest(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {
                "role": "assistant",
                "content": "assistant",
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"}
                }]
            }
        ],
        model="codec-model",
        params={"temperature": 0.1, "max_tokens": 8},
        tools=[{
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object"}}
        }],
        tool_choice="auto",
        extra={"provider": "demo"},
    )
    _ = annotated.messages
    _ = annotated.model
    _ = annotated.params
    _ = annotated.tools
    _ = annotated.tool_choice
    _ = annotated.extra
    _ = annotated.system_prompt()
    _ = annotated.last_user_message()
    _ = annotated.has_tool_calls()
    annotated.messages = [{"role": "user", "content": "updated"}]
    annotated.model = "updated-model"
    annotated.params = None
    annotated.tools = None
    annotated.tool_choice = None
    annotated.extra = {"updated": True}
    _ = repr(annotated)

    stack = api.create_scope_stack()
    _ = repr(stack)
    api.set_thread_scope_stack(stack)
    api.sync_thread_scope_stack(stack)
    root = api.get_handle()
    _ = root.uuid
    _ = root.name
    _ = root.scope_type
    _ = root.attributes
    _ = root.parent_uuid
    _ = root.data
    _ = root.metadata
    _ = repr(root)

    api.register_subscriber("types_py_visible_subscriber", subscriber)
    child = api.push_scope(
        "child",
        types.ScopeType.Tool,
        handle=root,
        attributes=sa,
        data={"payload": True},
        metadata={"meta": True},
    )
    _ = child.uuid
    _ = child.name
    _ = child.scope_type
    _ = child.attributes
    _ = child.parent_uuid
    _ = child.data
    _ = child.metadata
    _ = repr(child)

    tool = api.tool_call(
        "tool",
        {"arg": 1},
        handle=child,
        attributes=ta,
        data={"tool_data": True},
        metadata={"tool_meta": True},
        tool_call_id="call-1",
    )
    _ = tool.uuid
    _ = tool.name
    _ = tool.attributes
    _ = tool.parent_uuid
    _ = tool.data
    _ = tool.metadata
    _ = repr(tool)
    api.tool_call_end(tool, {"result": 2}, data={"done": True}, metadata={"status": "ok"})

    llm = api.llm_call(
        "llm",
        request,
        handle=child,
        attributes=la,
        data={"llm_data": True},
        metadata={"llm_meta": True},
        model_name="demo-model",
    )
    _ = llm.uuid
    _ = llm.name
    _ = llm.attributes
    _ = llm.parent_uuid
    _ = llm.data
    _ = llm.metadata
    _ = repr(llm)
    api.llm_call_end(llm, {"response": "ok"}, data={"tokens": 1}, metadata={"finish_reason": "stop"})

    api.event("mark", handle=child, data={"step": 1}, metadata={"source": "py"})
    api.pop_scope(child)
    api.flush_subscribers()
    api.deregister_subscriber("types_py_visible_subscriber")

    chat_codec = types.OpenAIChatCodec()
    chat_decoded = chat_codec.decode(request)
    _ = chat_decoded.messages
    _ = chat_codec.encode(chat_decoded, request)
    chat_response = chat_codec.decode_response({
        "id": "chatcmpl-1",
        "model": "gpt-4o-mini",
        "choices": [{
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop"
        }]
    })
    _ = chat_response.id
    _ = chat_response.model
    _ = chat_response.message
    _ = chat_response.tool_calls
    _ = chat_response.finish_reason
    _ = chat_response.usage
    _ = chat_response.api_specific
    _ = chat_response.extra
    _ = chat_response.response_text()
    _ = chat_response.has_tool_calls()
    _ = repr(chat_response)
    _ = repr(chat_codec)

    responses_codec = types.OpenAIResponsesCodec()
    responses_request = types.LLMRequest({}, {
        "model": "gpt-4o-mini",
        "instructions": "Be helpful",
        "input": [{"role": "user", "content": "hi"}],
        "max_output_tokens": 4
    })
    responses_decoded = responses_codec.decode(responses_request)
    _ = responses_codec.encode(responses_decoded, responses_request)
    responses_response = responses_codec.decode_response({
        "id": "resp-1",
        "model": "gpt-4o-mini",
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "done"}]
        }]
    })
    _ = responses_response.response_text()
    _ = repr(responses_codec)

    anthropic_codec = types.AnthropicMessagesCodec()
    anthropic_request = types.LLMRequest({}, {
        "model": "claude-sonnet-4-20250514",
        "system": "Be careful",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4
    })
    anthropic_decoded = anthropic_codec.decode(anthropic_request)
    _ = anthropic_codec.encode(anthropic_decoded, anthropic_request)
    anthropic_response = anthropic_codec.decode_response({
        "id": "msg-1",
        "model": "claude-sonnet-4-20250514",
        "content": [{"type": "text", "text": "done"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1}
    })
    _ = anthropic_response.response_text()
    _ = repr(anthropic_codec)

    return events
"#,
            )
            .unwrap(),
            &CString::new("py_types_visible_helpers.py").unwrap(),
            &CString::new("py_types_visible_helpers").unwrap(),
        )
        .unwrap();

        let events = helpers
            .getattr("run")
            .unwrap()
            .call1((types_module.clone(), api_module.clone()))
            .unwrap();
        let events_json = py_to_json(&events).unwrap();
        assert!(
            events_json
                .as_array()
                .unwrap()
                .iter()
                .any(|event| event["kind"] == "scope"
                    && event["category"] == "tool"
                    && event["scope_category"] == "start")
        );
        assert!(
            events_json
                .as_array()
                .unwrap()
                .iter()
                .any(|event| event["kind"] == "scope"
                    && event["category"] == "llm"
                    && event["scope_category"] == "end")
        );
    });
}
