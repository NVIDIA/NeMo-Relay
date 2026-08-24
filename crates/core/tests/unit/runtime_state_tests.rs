// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for runtime middleware snapshot chains.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::{Map, json};

use super::*;
use crate::api::registry::{RegistryRecord, RequestIntercept};
use crate::api::runtime::EventMetadataInjectorFn;

#[tokio::test]
async fn event_metadata_injection_accepts_flat_otel_values_and_empty_output() {
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("valid-injection")
            .metadata(json!({"nv.test.existing": "original"}))
            .build(),
        None,
        None,
    ));
    let noop: EventMetadataInjectorFn = Arc::new(|_| Box::pin(async { Ok(BTreeMap::new()) }));
    let valid: EventMetadataInjectorFn = Arc::new(|_| {
        Box::pin(async {
            Ok(BTreeMap::from([
                ("region".into(), json!("us-west")),
                ("region_name".into(), json!("west")),
                ("region-name".into(), json!("west")),
                ("experiment.variant".into(), json!("value")),
                ("nv.test.boolean".into(), json!(true)),
                ("nv.test.number".into(), json!(42)),
                (
                    "nv.test.max_unsigned_integer".into(),
                    json!(i64::MAX as u64),
                ),
                ("nv.test.strings".into(), json!(["a", "b"])),
                ("nv.test.booleans".into(), json!([true, false])),
                ("nv.test.integers".into(), json!([1, 2])),
                ("nv.test.doubles".into(), json!([1.0, 2.5])),
                ("nv.test.numbers".into(), json!([1, 2.5])),
                ("nv.test.empty".into(), json!([])),
            ]))
        })
    });

    let injected = NemoRelayContextState::event_metadata_injection_snapshot_chain(
        event,
        &[
            RegistryRecord::new("noop", 0, noop),
            RegistryRecord::new("valid", 1, valid),
        ],
    )
    .await;
    let metadata = injected.metadata().expect("metadata should be an object");

    assert_eq!(metadata["nv.test.existing"], json!("original"));
    assert_eq!(metadata["region"], json!("us-west"));
    assert_eq!(metadata["region_name"], json!("west"));
    assert_eq!(metadata["region-name"], json!("west"));
    assert_eq!(metadata["experiment.variant"], json!("value"));
    assert_eq!(metadata["nv.test.boolean"], json!(true));
    assert_eq!(metadata["nv.test.number"], json!(42));
    assert_eq!(
        metadata["nv.test.max_unsigned_integer"],
        json!(i64::MAX as u64)
    );
    assert_eq!(metadata["nv.test.strings"], json!(["a", "b"]));
    assert_eq!(metadata["nv.test.booleans"], json!([true, false]));
    assert_eq!(metadata["nv.test.integers"], json!([1, 2]));
    assert_eq!(metadata["nv.test.doubles"], json!([1.0, 2.5]));
    assert_eq!(metadata["nv.test.numbers"], json!([1, 2.5]));
    assert_eq!(metadata["nv.test.empty"], json!([]));
}

#[tokio::test]
async fn event_metadata_injection_rejects_invalid_output_atomically() {
    let invalid_outputs = [
        BTreeMap::from([
            ("telemetry.accepted_if_valid".into(), json!(true)),
            ("invalid-value".into(), Json::Null),
        ]),
        BTreeMap::from([("".into(), json!(true))]),
        BTreeMap::from([("   ".into(), json!(true))]),
        BTreeMap::from([(".region".into(), json!(true))]),
        BTreeMap::from([("region.".into(), json!(true))]),
        BTreeMap::from([("literal..dots".into(), json!(true))]),
        BTreeMap::from([("display name".into(), json!(true))]),
        BTreeMap::from([("region/zone".into(), json!(true))]),
        BTreeMap::from([("region@name".into(), json!(true))]),
        BTreeMap::from([("region\nname".into(), json!(true))]),
        BTreeMap::from([("nv.test.null".into(), Json::Null)]),
        BTreeMap::from([("nv.test.object".into(), json!({"nested": true}))]),
        BTreeMap::from([("nv.test.nested_list".into(), json!([[1]]))]),
        BTreeMap::from([("nv.test.mixed_list".into(), json!([1, "two"]))]),
        BTreeMap::from([("nv.test.oversized_number".into(), json!(u64::MAX))]),
        BTreeMap::from([("nv.test.oversized_list".into(), json!([u64::MAX]))]),
    ];

    for invalid_output in invalid_outputs {
        let event = Event::Mark(MarkEvent::new(
            BaseEvent::builder()
                .name("invalid-injection")
                .metadata(json!({"nv.test.existing": "original"}))
                .build(),
            None,
            None,
        ));
        let expected = event.clone();
        let injector: EventMetadataInjectorFn = Arc::new(move |_| {
            let invalid_output = invalid_output.clone();
            Box::pin(async move { Ok(invalid_output) })
        });

        let injected = NemoRelayContextState::event_metadata_injection_snapshot_chain(
            event,
            &[RegistryRecord::new("invalid", 0, injector)],
        )
        .await;

        assert_eq!(injected, expected);
    }
}

#[tokio::test]
async fn event_metadata_injection_isolates_failures_and_preserves_non_object_metadata() {
    let first: EventMetadataInjectorFn = Arc::new(|_| {
        Box::pin(async { Ok(BTreeMap::from([("nv.test.first".into(), json!(true))])) })
    });
    let panicking: EventMetadataInjectorFn =
        Arc::new(|_| Box::pin(async { panic!("expected injector panic") }));
    let failing: EventMetadataInjectorFn = Arc::new(|_| {
        Box::pin(async { Err(FlowError::Internal("expected injector failure".into())) })
    });
    let later_called = Arc::new(AtomicBool::new(false));
    let later_called_by_callback = Arc::clone(&later_called);
    let later: EventMetadataInjectorFn = Arc::new(move |_| {
        later_called_by_callback.store(true, Ordering::Release);
        Box::pin(async { Ok(BTreeMap::from([("nv.test.later".into(), json!(true))])) })
    });
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder().name("failure-isolation").build(),
        None,
        None,
    ));

    let injected = NemoRelayContextState::event_metadata_injection_snapshot_chain(
        event,
        &[
            RegistryRecord::new("first", 0, first),
            RegistryRecord::new("panicking", 1, panicking),
            RegistryRecord::new("failing", 2, failing),
            RegistryRecord::new("later", 3, later),
        ],
    )
    .await;
    let metadata = injected.metadata().expect("metadata should be created");

    assert_eq!(metadata["nv.test.first"], json!(true));
    assert_eq!(metadata["nv.test.later"], json!(true));
    assert!(later_called.load(Ordering::Acquire));

    let scalar_metadata_event = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("non-object-metadata")
            .metadata(json!("preserved"))
            .build(),
        None,
        None,
    ));
    let expected = scalar_metadata_event.clone();
    let injector: EventMetadataInjectorFn = Arc::new(|_| {
        Box::pin(async { Ok(BTreeMap::from([("nv.test.omitted".into(), json!(true))])) })
    });

    let injected = NemoRelayContextState::event_metadata_injection_snapshot_chain(
        scalar_metadata_event,
        &[RegistryRecord::new("non-object", 0, injector)],
    )
    .await;

    assert_eq!(injected, expected);
}

#[tokio::test]
async fn sanitizer_snapshot_chains_fail_closed_on_callback_panics() {
    let event = Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("preserved-event")
            .data(json!({"event": "preserved"}))
            .metadata(json!({"metadata": "preserved"}))
            .build(),
        None,
        None,
    ));
    let event_sanitizer: EventSanitizeFn =
        Arc::new(|_, _| Box::pin(async { panic!("event sanitizer panic") }));
    let event_later_called = Arc::new(AtomicBool::new(false));
    let event_later_called_by_callback = Arc::clone(&event_later_called);
    let event_later_sanitizer: EventSanitizeFn = Arc::new(move |_, fields| {
        event_later_called_by_callback.store(true, Ordering::Release);
        Box::pin(async move { Ok(fields) })
    });
    let sanitized_event = NemoRelayContextState::event_sanitize_snapshot_chain(
        event.clone(),
        &[
            RegistryRecord::new("event-panic", 0, event_sanitizer),
            RegistryRecord::new("event-later", -1, event_later_sanitizer),
        ],
    )
    .await;
    assert_eq!(sanitized_event.data(), None);
    assert_eq!(sanitized_event.metadata(), None);
    assert!(!event_later_called.load(Ordering::Acquire));

    let tool_payload = json!({"tool": "preserved"});
    let tool_sanitizer: ToolSanitizeFn =
        Arc::new(|_, _| Box::pin(async { panic!("tool sanitizer panic") }));
    let tool_later_called = Arc::new(AtomicBool::new(false));
    let tool_later_called_by_callback = Arc::clone(&tool_later_called);
    let tool_later_sanitizer: ToolSanitizeFn = Arc::new(move |_, value| {
        tool_later_called_by_callback.store(true, Ordering::Release);
        Box::pin(async move { Ok(value) })
    });
    let tool_entries = vec![
        RegistryRecord::new("tool-panic", 0, tool_sanitizer),
        RegistryRecord::new("tool-later", -1, tool_later_sanitizer),
    ];
    assert_eq!(
        NemoRelayContextState::tool_sanitize_request_snapshot_chain(
            "tool",
            tool_payload.clone(),
            &tool_entries,
        )
        .await,
        None
    );
    assert!(!tool_later_called.load(Ordering::Acquire));
    let tool_response = json!({"tool_response": "preserved"});
    let tool_response_sanitizer: ToolSanitizeFn =
        Arc::new(|_, _| Box::pin(async { panic!("tool response sanitizer panic") }));
    let tool_response_later_called = Arc::new(AtomicBool::new(false));
    let tool_response_later_called_by_callback = Arc::clone(&tool_response_later_called);
    let tool_response_later_sanitizer: ToolSanitizeFn = Arc::new(move |_, value| {
        tool_response_later_called_by_callback.store(true, Ordering::Release);
        Box::pin(async move { Ok(value) })
    });
    assert_eq!(
        NemoRelayContextState::tool_sanitize_response_snapshot_chain(
            "tool",
            tool_response.clone(),
            &[
                RegistryRecord::new("tool-response-panic", 0, tool_response_sanitizer,),
                RegistryRecord::new("tool-response-later", -1, tool_response_later_sanitizer,)
            ],
        )
        .await,
        None
    );
    assert!(!tool_response_later_called.load(Ordering::Acquire));

    let request = LlmRequest {
        headers: Map::new(),
        content: json!({"llm": "preserved"}),
    };
    let llm_sanitizer: LlmSanitizeRequestFn =
        Arc::new(|_, _| Box::pin(async { panic!("LLM sanitizer panic") }));
    let llm_later_called = Arc::new(AtomicBool::new(false));
    let llm_later_called_by_callback = Arc::clone(&llm_later_called);
    let llm_later_sanitizer: LlmSanitizeRequestFn = Arc::new(move |request, _| {
        llm_later_called_by_callback.store(true, Ordering::Release);
        Box::pin(async move { Ok(Some(request)) })
    });
    let llm_entries = vec![
        RegistryRecord::new("llm-panic", 0, llm_sanitizer),
        RegistryRecord::new("llm-later", -1, llm_later_sanitizer),
    ];
    assert_eq!(
        NemoRelayContextState::llm_sanitize_request_snapshot_chain(
            request.clone(),
            LlmSanitizeRequestContext::default(),
            &llm_entries,
        )
        .await,
        None
    );
    assert!(!llm_later_called.load(Ordering::Acquire));
    let llm_response = json!({"llm_response": "preserved"});
    let llm_response_sanitizer: LlmSanitizeResponseFn =
        Arc::new(|_, _| Box::pin(async { panic!("LLM response sanitizer panic") }));
    let llm_response_later_called = Arc::new(AtomicBool::new(false));
    let llm_response_later_called_by_callback = Arc::clone(&llm_response_later_called);
    let llm_response_later_sanitizer: LlmSanitizeResponseFn = Arc::new(move |response, _| {
        llm_response_later_called_by_callback.store(true, Ordering::Release);
        Box::pin(async move { Ok(Some(response)) })
    });
    assert_eq!(
        NemoRelayContextState::llm_sanitize_response_snapshot_chain(
            llm_response.clone(),
            LlmSanitizeResponseContext::default(),
            &[
                RegistryRecord::new("llm-response-panic", 0, llm_response_sanitizer,),
                RegistryRecord::new("llm-response-later", -1, llm_response_later_sanitizer,)
            ],
        )
        .await,
        None
    );
    assert!(!llm_response_later_called.load(Ordering::Acquire));

    let tool_conditional: ToolConditionalFn =
        Arc::new(|_, _| Box::pin(async { panic!("tool conditional panic") }));
    let error = NemoRelayContextState::tool_conditional_execution_snapshot_chain(
        "tool",
        &tool_payload,
        &[RegistryRecord::new(
            "tool-conditional-panic",
            0,
            tool_conditional,
        )],
        &[],
        None,
        None,
    )
    .await
    .unwrap_err();
    assert!(matches!(
        error,
        FlowError::Internal(ref message) if message.contains("tool-conditional-panic")
    ));

    let llm_conditional: LlmConditionalFn =
        Arc::new(|_| Box::pin(async { panic!("LLM conditional panic") }));
    let error = NemoRelayContextState::llm_conditional_execution_snapshot_chain(
        &request,
        &[RegistryRecord::new(
            "llm-conditional-panic",
            0,
            llm_conditional,
        )],
        &[],
        None,
        None,
    )
    .await
    .unwrap_err();
    assert!(matches!(
        error,
        FlowError::Internal(ref message) if message.contains("llm-conditional-panic")
    ));

    let tool_intercept: ToolInterceptFn =
        Arc::new(|_, _| Box::pin(async { panic!("tool intercept panic") }));
    let error = NemoRelayContextState::tool_request_intercepts_snapshot_chain(
        "tool",
        tool_payload,
        &[RegistryRecord::new(
            "tool-intercept-panic",
            0,
            RequestIntercept::new(false, tool_intercept),
        )],
    )
    .await
    .unwrap_err();
    assert!(matches!(
        error,
        FlowError::Internal(ref message) if message.contains("tool-intercept-panic")
    ));

    let llm_intercept: LlmRequestInterceptFn =
        Arc::new(|_, _, _| Box::pin(async { panic!("LLM intercept panic") }));
    let error = NemoRelayContextState::llm_request_intercepts_snapshot_chain(
        "llm",
        request,
        None,
        &[RegistryRecord::new(
            "llm-intercept-panic",
            0,
            RequestIntercept::new(false, llm_intercept),
        )],
        false,
    )
    .await
    .unwrap_err();
    assert!(matches!(
        error,
        FlowError::Internal(ref message) if message.contains("llm-intercept-panic")
    ));
}
