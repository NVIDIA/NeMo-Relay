// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coverage tests for callable in the NeMo Relay WebAssembly crate.

use super::*;
use nemo_relay::codec::request::AnnotatedLlmRequest;
use serde_json::json;
use tokio_stream::StreamExt;
use wasm_bindgen::JsCast;
use wasm_bindgen::JsValue;

fn dummy_function() -> Function {
    JsValue::NULL.unchecked_into()
}

#[test]
fn pending_mark_dto_uses_camel_case_without_changing_canonical_fields() {
    let dto: JsPendingMarkSpec = serde_json::from_value(json!({
        "name": "request.optimized",
        "categoryProfile": {"subtype": "optimizer.saved_tokens"},
        "data": {"savedTokens": 12}
    }))
    .unwrap();
    let canonical: PendingMarkSpec = dto.into();
    assert_eq!(
        canonical
            .category_profile
            .as_ref()
            .unwrap()
            .subtype
            .as_deref(),
        Some("optimizer.saved_tokens")
    );

    let dto_json = serde_json::to_value(JsPendingMarkSpec::from(canonical)).unwrap();
    assert_eq!(
        dto_json["categoryProfile"]["subtype"],
        "optimizer.saved_tokens"
    );
    assert!(dto_json.get("category_profile").is_none());
    assert!(
        serde_json::from_value::<JsPendingMarkSpec>(json!({
            "name": "wire-name-is-invalid-in-js",
            "category_profile": {"subtype": "invalid"}
        }))
        .is_err()
    );
}

#[test]
fn native_tool_and_llm_wrapper_fallbacks_are_stable() {
    let tool = wrap_js_tool_fn(dummy_function());
    assert_eq!(tool("name", json!({"input": true})), Json::Null);

    let tool_conditional = wrap_js_tool_conditional_fn(dummy_function());
    assert_eq!(
        tool_conditional("name", &json!({"input": true})).unwrap(),
        None
    );

    let tool_intercept = wrap_js_tool_request_intercept_fn(dummy_function());
    assert_eq!(
        tool_intercept("name", json!({"input": true})).unwrap(),
        json!({"input": true})
    );

    let llm_request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"messages": []}),
    };
    let llm_intercept = wrap_js_llm_request_intercept_fn(dummy_function());
    let outcome = llm_intercept("llm", llm_request.clone(), None).unwrap();
    assert_eq!(outcome.request.content, llm_request.content);
    assert!(outcome.annotated_request.is_none());
    assert!(outcome.pending_marks.is_empty());

    let llm_sanitize = wrap_js_llm_sanitize_request_fn(dummy_function());
    assert_eq!(
        llm_sanitize(llm_request.clone()).content,
        llm_request.content
    );

    let llm_response = wrap_js_llm_response_fn(dummy_function());
    assert_eq!(
        llm_response(json!({"response": true})),
        json!({"response": true})
    );

    let llm_conditional = wrap_js_llm_conditional_fn(dummy_function());
    assert_eq!(llm_conditional(&llm_request).unwrap(), None);
}

#[tokio::test]
async fn native_async_wrapper_fallbacks_return_errors_or_defaults() {
    let tool_exec = wrap_js_tool_exec_fn(dummy_function());
    assert!(
        tool_exec(json!({"tool": 1}))
            .await
            .unwrap_err()
            .to_string()
            .contains("only supported on wasm32")
    );

    let llm_exec = wrap_js_llm_exec_fn(dummy_function());
    assert!(
        llm_exec(LlmRequest {
            headers: serde_json::Map::new(),
            content: json!({"messages": []}),
        })
        .await
        .unwrap_err()
        .to_string()
        .contains("only supported on wasm32")
    );

    let mut collector = wrap_js_collector_fn(dummy_function());
    assert!(collector(json!({"chunk": 1})).is_ok());

    let finalizer = wrap_js_finalizer_fn(dummy_function());
    assert_eq!(finalizer(), Json::Null);

    let subscriber = wrap_js_event_subscriber(dummy_function());
    subscriber(&Event::Mark(nemo_relay::api::event::MarkEvent::new(
        nemo_relay::api::event::BaseEvent::builder()
            .name("native-mark")
            .build(),
        None,
        None,
    )));
}

#[tokio::test]
async fn native_intercept_and_codec_fallbacks_are_callable() {
    let tool_intercept = wrap_js_tool_exec_intercept_fn(dummy_function());
    let next: ToolExecutionNextFn = Arc::new(|args| Box::pin(async move { Ok(args) }));
    assert_eq!(
        tool_intercept("tool", json!({"x": 1}), next).await.unwrap(),
        json!({"x": 1})
    );

    let llm_intercept = wrap_js_llm_exec_intercept_fn(dummy_function());
    let llm_next: LlmExecutionNextFn =
        Arc::new(|request| Box::pin(async move { Ok(serde_json::to_value(request).unwrap()) }));
    let llm_request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"messages": []}),
    };
    assert_eq!(
        llm_intercept("llm", llm_request.clone(), llm_next,)
            .await
            .unwrap(),
        serde_json::to_value(&llm_request).unwrap()
    );

    let llm_stream_intercept = wrap_js_llm_stream_exec_intercept_fn(dummy_function());
    let llm_stream_next: LlmStreamExecutionNextFn = Arc::new(|_request| {
        Box::pin(async move {
            Ok(
                Box::pin(tokio_stream::iter(vec![Ok(json!({"chunk": true}))]))
                    as Pin<Box<dyn tokio_stream::Stream<Item = Result<Json>> + Send>>,
            )
        })
    });
    let mut stream = llm_stream_intercept("llm", llm_request, llm_stream_next)
        .await
        .unwrap();
    assert_eq!(
        stream.next().await.transpose().unwrap(),
        Some(json!({"chunk": true}))
    );

    let codec = wrap_js_codec(dummy_function(), dummy_function());
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"messages": []}),
    };
    assert!(codec.decode(&request).is_err());
    let annotated = AnnotatedLlmRequest {
        messages: Vec::new(),
        model: None,
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
    assert!(codec.encode(&annotated, &request).is_err());

    assert!(
        std::panic::catch_unwind(|| wrap_js_response_codec(dummy_function())).is_err(),
        "non-wasm response codec wrapper should panic"
    );
}
