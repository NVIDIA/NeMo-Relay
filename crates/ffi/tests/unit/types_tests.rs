// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for types in the NeMo Relay FFI crate.

use super::*;
use std::ffi::{CStr, CString};
use std::sync::Arc;

use nemo_relay::api::event::{
    BaseEvent, CategoryProfile, EventCategory, MarkEvent, ScopeCategory, ScopeEvent,
    llm_attributes_to_strings, scope_attributes_to_strings, tool_attributes_to_strings,
};
use nemo_relay::api::runtime::create_scope_stack;
use serde_json::json;
use uuid::Uuid;

fn take_string(ptr: *mut c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    let value = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap().to_string();
    unsafe { convert::nemo_relay_string_free(ptr) };
    Some(value)
}

fn mark_event(
    name: &str,
    parent_uuid: Option<Uuid>,
    data: Option<serde_json::Value>,
    metadata: Option<serde_json::Value>,
) -> Event {
    Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .parent_uuid_opt(parent_uuid)
            .name(name)
            .data_opt(data)
            .metadata_opt(metadata)
            .build(),
        None,
        None,
    ))
}

struct ScopeEventFixture {
    scope_category: ScopeCategory,
    scope_type: ScopeType,
    name: &'static str,
    parent_uuid: Option<Uuid>,
    data: Option<serde_json::Value>,
    metadata: Option<serde_json::Value>,
    attributes: Vec<String>,
    category_profile: Option<CategoryProfile>,
}

fn make_scope_event(fixture: ScopeEventFixture) -> Event {
    Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .parent_uuid_opt(fixture.parent_uuid)
            .name(fixture.name)
            .data_opt(fixture.data)
            .metadata_opt(fixture.metadata)
            .build(),
        fixture.scope_category,
        fixture.attributes,
        EventCategory::from(fixture.scope_type),
        fixture.category_profile,
    ))
}

#[test]
fn test_scope_handle_accessors_and_null_metadata_guard() {
    assert!(unsafe { nemo_relay_scope_handle_metadata(std::ptr::null()) }.is_null());

    let parent_uuid = Uuid::now_v7();
    let handle = FfiScopeHandle(
        ScopeHandle::builder()
            .name("scope")
            .scope_type(ScopeType::Tool)
            .attributes(ScopeAttributes::PARALLEL)
            .parent_uuid(parent_uuid)
            .data(json!({"data": true}))
            .metadata(json!({"meta": true}))
            .build(),
    );

    assert_eq!(
        take_string(unsafe { nemo_relay_scope_handle_name(&handle) }),
        Some("scope".into())
    );
    assert_eq!(
        unsafe { nemo_relay_scope_handle_scope_type(&handle) } as i32,
        NemoRelayScopeType::Tool as i32
    );
    assert_eq!(
        unsafe { nemo_relay_scope_handle_attributes(&handle) },
        ScopeAttributes::PARALLEL.bits()
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_scope_handle_parent_uuid(&handle) }),
        Some(parent_uuid.to_string())
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_scope_handle_data(&handle) }),
        Some(r#"{"data":true}"#.into())
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_scope_handle_metadata(&handle) }),
        Some(r#"{"meta":true}"#.into())
    );
}

#[test]
fn test_scope_type_conversions_and_handle_null_guards() {
    let scope_types = [
        (NemoRelayScopeType::Agent, ScopeType::Agent),
        (NemoRelayScopeType::Function, ScopeType::Function),
        (NemoRelayScopeType::Tool, ScopeType::Tool),
        (NemoRelayScopeType::Llm, ScopeType::Llm),
        (NemoRelayScopeType::Retriever, ScopeType::Retriever),
        (NemoRelayScopeType::Embedder, ScopeType::Embedder),
        (NemoRelayScopeType::Reranker, ScopeType::Reranker),
        (NemoRelayScopeType::Guardrail, ScopeType::Guardrail),
        (NemoRelayScopeType::Evaluator, ScopeType::Evaluator),
        (NemoRelayScopeType::Custom, ScopeType::Custom),
        (NemoRelayScopeType::Unknown, ScopeType::Unknown),
    ];

    for (ffi, core) in scope_types {
        let round_trip: NemoRelayScopeType = core.into();
        assert_eq!(round_trip as i32, ffi as i32);
        let back: ScopeType = ffi.into();
        assert_eq!(back, core);
    }

    assert!(unsafe { nemo_relay_scope_handle_uuid(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_relay_scope_handle_name(std::ptr::null()) }.is_null());
    assert_eq!(
        unsafe { nemo_relay_scope_handle_scope_type(std::ptr::null()) } as i32,
        NemoRelayScopeType::Unknown as i32
    );
    assert_eq!(
        unsafe { nemo_relay_scope_handle_attributes(std::ptr::null()) },
        0
    );
    assert!(unsafe { nemo_relay_scope_handle_parent_uuid(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_relay_scope_handle_data(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_relay_scope_handle_metadata(std::ptr::null()) }.is_null());
}

#[test]
fn test_tool_and_llm_handle_accessors_and_null_guards() {
    let parent_uuid = Uuid::now_v7();
    let tool = FfiToolHandle(
        ToolHandle::builder()
            .name("tool")
            .attributes(ToolAttributes::REMOTE)
            .parent_uuid(parent_uuid)
            .build(),
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_tool_handle_uuid(&tool) }),
        Some(tool.0.uuid.to_string())
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_tool_handle_name(&tool) }),
        Some("tool".into())
    );
    assert_eq!(
        unsafe { nemo_relay_tool_handle_attributes(&tool) },
        ToolAttributes::REMOTE.bits()
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_tool_handle_parent_uuid(&tool) }),
        Some(parent_uuid.to_string())
    );

    let llm = FfiLLMHandle(
        LlmHandle::builder()
            .name("llm")
            .attributes(LlmAttributes::STATEFUL | LlmAttributes::STREAMING)
            .parent_uuid(parent_uuid)
            .build(),
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_llm_handle_uuid(&llm) }),
        Some(llm.0.uuid.to_string())
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_llm_handle_name(&llm) }),
        Some("llm".into())
    );
    assert_eq!(
        unsafe { nemo_relay_llm_handle_attributes(&llm) },
        (LlmAttributes::STATEFUL | LlmAttributes::STREAMING).bits()
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_llm_handle_parent_uuid(&llm) }),
        Some(parent_uuid.to_string())
    );

    assert!(unsafe { nemo_relay_tool_handle_uuid(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_relay_tool_handle_name(std::ptr::null()) }.is_null());
    assert_eq!(
        unsafe { nemo_relay_tool_handle_attributes(std::ptr::null()) },
        0
    );
    assert!(unsafe { nemo_relay_tool_handle_parent_uuid(std::ptr::null()) }.is_null());

    assert!(unsafe { nemo_relay_llm_handle_uuid(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_relay_llm_handle_name(std::ptr::null()) }.is_null());
    assert_eq!(
        unsafe { nemo_relay_llm_handle_attributes(std::ptr::null()) },
        0
    );
    assert!(unsafe { nemo_relay_llm_handle_parent_uuid(std::ptr::null()) }.is_null());
}

#[test]
fn test_llm_request_null_inputs_event_null_guards_and_free_nulls() {
    let request_ptr = unsafe { nemo_relay_llm_request_new(std::ptr::null(), std::ptr::null()) };
    assert!(!request_ptr.is_null());
    assert_eq!(
        take_string(unsafe { nemo_relay_llm_request_headers(request_ptr) }),
        Some("{}".into())
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_llm_request_content(request_ptr) }),
        Some("null".into())
    );
    unsafe { nemo_relay_llm_request_free(request_ptr) };

    assert!(unsafe { nemo_relay_llm_request_headers(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_relay_llm_request_content(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_relay_event_uuid(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_relay_event_name(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_relay_event_kind(std::ptr::null()) }.is_null());
    assert_eq!(unsafe { nemo_relay_event_attributes(std::ptr::null()) }, 0);
    assert!(unsafe { nemo_relay_event_data(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_relay_event_metadata(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_relay_event_timestamp(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_relay_event_input(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_relay_event_output(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_relay_event_model_name(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_relay_event_tool_call_id(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_relay_event_parent_uuid(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_relay_event_scope_type(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_relay_event_atof_version(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_relay_event_scope_category(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_relay_event_category(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_relay_event_attributes_json(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_relay_event_category_profile(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_relay_event_data_schema(std::ptr::null()) }.is_null());

    unsafe {
        nemo_relay_scope_handle_free(std::ptr::null_mut());
        nemo_relay_tool_handle_free(std::ptr::null_mut());
        nemo_relay_llm_handle_free(std::ptr::null_mut());
        nemo_relay_llm_request_free(std::ptr::null_mut());
        nemo_relay_event_free(std::ptr::null_mut());
        nemo_relay_scope_stack_free(std::ptr::null_mut());
        nemo_relay_atif_exporter_free(std::ptr::null_mut());
        nemo_relay_otel_subscriber_free(std::ptr::null_mut());
        nemo_relay_openinference_subscriber_free(std::ptr::null_mut());
        nemo_relay_adaptive_runtime_free(std::ptr::null_mut());
    }
}

#[test]
fn test_valid_free_functions_and_none_backed_accessors() {
    let scope_ptr = Box::into_raw(Box::new(FfiScopeHandle(
        ScopeHandle::builder()
            .name("scope-none")
            .scope_type(ScopeType::Function)
            .build(),
    )));
    assert_eq!(
        take_string(unsafe { nemo_relay_scope_handle_parent_uuid(scope_ptr) }),
        None
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_scope_handle_data(scope_ptr) }),
        None
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_scope_handle_metadata(scope_ptr) }),
        None
    );
    unsafe { nemo_relay_scope_handle_free(scope_ptr) };

    let tool_ptr = Box::into_raw(Box::new(FfiToolHandle(
        ToolHandle::builder().name("tool-none").build(),
    )));
    assert_eq!(
        take_string(unsafe { nemo_relay_tool_handle_parent_uuid(tool_ptr) }),
        None
    );
    unsafe { nemo_relay_tool_handle_free(tool_ptr) };

    let llm_ptr = Box::into_raw(Box::new(FfiLLMHandle(
        LlmHandle::builder().name("llm-none").build(),
    )));
    assert_eq!(
        take_string(unsafe { nemo_relay_llm_handle_parent_uuid(llm_ptr) }),
        None
    );
    unsafe { nemo_relay_llm_handle_free(llm_ptr) };

    let request_ptr = Box::into_raw(Box::new(FfiLLMRequest(LlmRequest {
        headers: serde_json::Map::new(),
        content: json!(null),
    })));
    unsafe { nemo_relay_llm_request_free(request_ptr) };

    let event_ptr = Box::into_raw(Box::new(FfiEvent(mark_event(
        "free-event",
        None,
        None,
        None,
    ))));
    unsafe { nemo_relay_event_free(event_ptr) };

    let stack_ptr = Box::into_raw(Box::new(FfiScopeStack(create_scope_stack())));
    unsafe { nemo_relay_scope_stack_free(stack_ptr) };

    let exporter_ptr = Box::into_raw(Box::new(FfiAtifExporter(
        nemo_relay::observability::atif::AtifExporter::new(
            "session".into(),
            nemo_relay::observability::atif::AtifAgentInfo {
                name: "ffi-agent".into(),
                version: "1.0.0".into(),
                model_name: None,
                tool_definitions: None,
                extra: None,
            },
        ),
    )));
    unsafe { nemo_relay_atif_exporter_free(exporter_ptr) };

    let adaptive_ptr = Box::into_raw(Box::new(FfiAdaptiveRuntime(std::sync::Mutex::new(None))));
    unsafe { nemo_relay_adaptive_runtime_free(adaptive_ptr) };
}

#[test]
fn test_llm_request_new_invalid_inputs_fall_back_to_defaults() {
    let invalid_headers = CString::new(r#"["not-an-object"]"#).unwrap();
    let invalid_content = CString::new("{").unwrap();
    let request_ptr =
        unsafe { nemo_relay_llm_request_new(invalid_headers.as_ptr(), invalid_content.as_ptr()) };
    assert!(!request_ptr.is_null());
    assert_eq!(
        take_string(unsafe { nemo_relay_llm_request_headers(request_ptr) }),
        Some("{}".into())
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_llm_request_content(request_ptr) }),
        Some("null".into())
    );
    unsafe { nemo_relay_llm_request_free(request_ptr) };
}

#[test]
fn test_llm_request_and_event_accessors() {
    let headers = CString::new(r#"{"header":"value"}"#).unwrap();
    let content = CString::new(r#"{"prompt":"hi"}"#).unwrap();
    let request_ptr = unsafe { nemo_relay_llm_request_new(headers.as_ptr(), content.as_ptr()) };
    assert!(!request_ptr.is_null());
    assert_eq!(
        take_string(unsafe { nemo_relay_llm_request_headers(request_ptr) }),
        Some(r#"{"header":"value"}"#.into())
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_llm_request_content(request_ptr) }),
        Some(r#"{"prompt":"hi"}"#.into())
    );
    unsafe { nemo_relay_llm_request_free(request_ptr) };

    let parent_uuid = Uuid::now_v7();
    let scope_event = make_scope_event(ScopeEventFixture {
        scope_category: ScopeCategory::Start,
        scope_type: ScopeType::Guardrail,
        name: "ffi-event",
        parent_uuid: Some(parent_uuid),
        data: Some(json!({"data": 1})),
        metadata: Some(json!({"meta": 2})),
        attributes: scope_attributes_to_strings(ScopeAttributes::empty()),
        category_profile: None,
    });
    let ffi_event = FfiEvent(scope_event.clone());

    assert_eq!(
        take_string(unsafe { nemo_relay_event_kind(&ffi_event) }),
        Some("scope".into())
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_event_scope_category(&ffi_event) }),
        Some("start".into())
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_event_atof_version(&ffi_event) }),
        Some("0.1".into())
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_event_category(&ffi_event) }),
        Some("guardrail".into())
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_event_attributes_json(&ffi_event) }),
        Some("[]".into())
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_event_category_profile(&ffi_event) }),
        None
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_event_data_schema(&ffi_event) }),
        None
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_event_uuid(&ffi_event) }),
        Some(scope_event.uuid().to_string())
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_event_name(&ffi_event) }),
        Some("ffi-event".into())
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_event_data(&ffi_event) }),
        Some(r#"{"data":1}"#.into())
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_event_metadata(&ffi_event) }),
        Some(r#"{"meta":2}"#.into())
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_event_scope_type(&ffi_event) }),
        Some("guardrail".into())
    );
    assert_eq!(
        unsafe { nemo_relay_event_attributes(&ffi_event) },
        ScopeAttributes::empty().bits()
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_event_parent_uuid(&ffi_event) }),
        Some(parent_uuid.to_string())
    );
    assert!(
        take_string(unsafe { nemo_relay_event_timestamp(&ffi_event) })
            .unwrap()
            .contains('T')
    );

    assert_eq!(
        take_string(unsafe { nemo_relay_event_input(&ffi_event) }),
        Some(r#"{"data":1}"#.into())
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_event_output(&ffi_event) }),
        None
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_event_model_name(&ffi_event) }),
        None
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_event_tool_call_id(&ffi_event) }),
        None
    );

    let llm_event = make_scope_event(ScopeEventFixture {
        scope_category: ScopeCategory::Start,
        scope_type: ScopeType::Llm,
        name: "ffi-llm",
        parent_uuid: Some(parent_uuid),
        data: Some(json!({"input": true})),
        metadata: None,
        attributes: llm_attributes_to_strings(LlmAttributes::empty()),
        category_profile: Some(CategoryProfile::builder().model_name("model").build()),
    });
    let ffi_llm_event = FfiEvent(llm_event);
    assert_eq!(
        take_string(unsafe { nemo_relay_event_input(&ffi_llm_event) }),
        Some(r#"{"input":true}"#.into())
    );
    assert_eq!(
        unsafe { nemo_relay_event_attributes(&ffi_llm_event) },
        LlmAttributes::empty().bits()
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_event_model_name(&ffi_llm_event) }),
        Some("model".into())
    );
    let llm_profile = take_string(unsafe { nemo_relay_event_category_profile(&ffi_llm_event) })
        .expect("expected category profile");
    let llm_profile: serde_json::Value = serde_json::from_str(&llm_profile).unwrap();
    assert_eq!(llm_profile["model_name"], json!("model"));
    assert_eq!(
        take_string(unsafe { nemo_relay_event_scope_type(&ffi_llm_event) }),
        Some("llm".into())
    );

    let tool_event = make_scope_event(ScopeEventFixture {
        scope_category: ScopeCategory::End,
        scope_type: ScopeType::Tool,
        name: "ffi-tool",
        parent_uuid: Some(parent_uuid),
        data: Some(json!({"output": true})),
        metadata: None,
        attributes: tool_attributes_to_strings(ToolAttributes::empty()),
        category_profile: Some(
            CategoryProfile::builder()
                .tool_call_id("tool-call-id")
                .build(),
        ),
    });
    let ffi_tool_event = FfiEvent(tool_event);
    assert_eq!(
        take_string(unsafe { nemo_relay_event_output(&ffi_tool_event) }),
        Some(r#"{"output":true}"#.into())
    );
    assert_eq!(
        unsafe { nemo_relay_event_attributes(&ffi_tool_event) },
        ToolAttributes::empty().bits()
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_event_tool_call_id(&ffi_tool_event) }),
        Some("tool-call-id".into())
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_event_scope_type(&ffi_tool_event) }),
        Some("tool".into())
    );

    let mark_event = mark_event("ffi-mark", Some(parent_uuid), None, None);
    let ffi_mark_event = FfiEvent(mark_event);
    assert_eq!(
        take_string(unsafe { nemo_relay_event_scope_type(&ffi_mark_event) }),
        None
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_event_atof_version(&ffi_mark_event) }),
        Some("0.1".into())
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_event_scope_category(&ffi_mark_event) }),
        None
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_event_category(&ffi_mark_event) }),
        None
    );
    assert_eq!(
        take_string(unsafe { nemo_relay_event_attributes_json(&ffi_mark_event) }),
        None
    );
    assert_eq!(unsafe { nemo_relay_event_attributes(&ffi_mark_event) }, 0);
}

#[test]
fn test_annotated_event_accessors_and_codec_handles() {
    let annotated_request = nemo_relay::codec::request::AnnotatedLlmRequest {
        messages: vec![nemo_relay::codec::request::Message::User {
            content: nemo_relay::codec::request::MessageContent::Text("hello".into()),
            name: Some("tester".into()),
        }],
        model: Some("gpt-test".into()),
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
        extra: serde_json::Map::from_iter([("provider".into(), json!("ffi"))]),
    };
    let llm_start = make_scope_event(ScopeEventFixture {
        scope_category: ScopeCategory::Start,
        scope_type: ScopeType::Llm,
        name: "annotated-start",
        parent_uuid: None,
        data: Some(json!({"input": "value"})),
        metadata: None,
        attributes: llm_attributes_to_strings(LlmAttributes::STREAMING),
        category_profile: Some(
            CategoryProfile::builder()
                .model_name("gpt-test")
                .annotated_request(Arc::new(annotated_request))
                .build(),
        ),
    });
    let ffi_start = FfiEvent(llm_start);
    let annotated_request_json =
        take_string(unsafe { nemo_relay_event_annotated_request(&ffi_start) })
            .expect("expected annotated request json");
    let annotated_request_value: serde_json::Value =
        serde_json::from_str(&annotated_request_json).unwrap();
    assert_eq!(annotated_request_value["model"], json!("gpt-test"));
    assert_eq!(annotated_request_value["provider"], json!("ffi"));
    assert!(unsafe { nemo_relay_event_annotated_response(&ffi_start) }.is_null());

    let annotated_response = nemo_relay::codec::response::AnnotatedLlmResponse {
        id: Some("resp_123".into()),
        model: Some("gpt-test".into()),
        message: Some(nemo_relay::codec::request::MessageContent::Text(
            "done".into(),
        )),
        tool_calls: None,
        finish_reason: Some(nemo_relay::codec::response::FinishReason::Complete),
        usage: Some(nemo_relay::codec::response::Usage {
            prompt_tokens: Some(10),
            completion_tokens: Some(5),
            total_tokens: Some(15),
            cache_read_tokens: None,
            cache_write_tokens: None,
            cost: Some(nemo_relay::codec::response::CostEstimate {
                total: Some(0.000_01),
                currency: "USD".into(),
                input: Some(0.000_004),
                output: Some(0.000_006),
                cache_read: None,
                cache_write: None,
                source: nemo_relay::codec::response::CostSource::ProviderReported,
                pricing_provider: Some("ffi-provider".into()),
                pricing_model: Some("gpt-test".into()),
                pricing_as_of: Some("2026-06-04".into()),
                pricing_source: Some("https://example.test/pricing".into()),
            }),
        }),
        api_specific: None,
        extra: serde_json::Map::from_iter([("trace".into(), json!(true))]),
    };
    let llm_end = make_scope_event(ScopeEventFixture {
        scope_category: ScopeCategory::End,
        scope_type: ScopeType::Llm,
        name: "annotated-end",
        parent_uuid: None,
        data: Some(json!({"output": "value"})),
        metadata: None,
        attributes: llm_attributes_to_strings(LlmAttributes::STATEFUL),
        category_profile: Some(
            CategoryProfile::builder()
                .model_name("gpt-test")
                .annotated_response(Arc::new(annotated_response))
                .build(),
        ),
    });
    let ffi_end = FfiEvent(llm_end);
    let annotated_response_json =
        take_string(unsafe { nemo_relay_event_annotated_response(&ffi_end) })
            .expect("expected annotated response json");
    let annotated_response_value: serde_json::Value =
        serde_json::from_str(&annotated_response_json).unwrap();
    assert_eq!(annotated_response_value["id"], json!("resp_123"));
    assert!(annotated_response_value.get("model_provider").is_none());
    assert_eq!(annotated_response_value["trace"], json!(true));
    assert_eq!(
        annotated_response_value["usage"]["cost"]["pricing_provider"],
        json!("ffi-provider")
    );
    assert!(unsafe { nemo_relay_event_annotated_request(&ffi_end) }.is_null());

    let scope_event = FfiEvent(make_scope_event(ScopeEventFixture {
        scope_category: ScopeCategory::Start,
        scope_type: ScopeType::Function,
        name: "plain-scope",
        parent_uuid: None,
        data: None,
        metadata: None,
        attributes: scope_attributes_to_strings(ScopeAttributes::PARALLEL),
        category_profile: None,
    }));
    assert!(unsafe { nemo_relay_event_annotated_request(&scope_event) }.is_null());
    assert!(unsafe { nemo_relay_event_annotated_response(&scope_event) }.is_null());

    let openai_chat = api::nemo_relay_openai_chat_codec_new();
    let openai_responses = api::nemo_relay_openai_responses_codec_new();
    let anthropic = api::nemo_relay_anthropic_messages_codec_new();
    assert!(!openai_chat.is_null());
    assert!(!openai_responses.is_null());
    assert!(!anthropic.is_null());

    unsafe {
        nemo_relay_codec_free(openai_chat);
        nemo_relay_codec_free(openai_responses);
        nemo_relay_codec_free(anthropic);
        nemo_relay_codec_free(std::ptr::null_mut());
    }
}

#[test]
fn test_event_accessor_none_and_null_pointer_paths_for_annotations() {
    assert!(unsafe { nemo_relay_event_annotated_request(std::ptr::null()) }.is_null());
    assert!(unsafe { nemo_relay_event_annotated_response(std::ptr::null()) }.is_null());

    let plain_start = FfiEvent(make_scope_event(ScopeEventFixture {
        scope_category: ScopeCategory::Start,
        scope_type: ScopeType::Llm,
        name: "plain-start",
        parent_uuid: None,
        data: None,
        metadata: None,
        attributes: Vec::new(),
        category_profile: None,
    }));
    assert_eq!(
        take_string(unsafe { nemo_relay_event_parent_uuid(&plain_start) }),
        None
    );
    assert!(unsafe { nemo_relay_event_annotated_request(&plain_start) }.is_null());

    let plain_end = FfiEvent(make_scope_event(ScopeEventFixture {
        scope_category: ScopeCategory::End,
        scope_type: ScopeType::Llm,
        name: "plain-end",
        parent_uuid: None,
        data: None,
        metadata: None,
        attributes: Vec::new(),
        category_profile: None,
    }));
    assert!(unsafe { nemo_relay_event_annotated_response(&plain_end) }.is_null());
}
