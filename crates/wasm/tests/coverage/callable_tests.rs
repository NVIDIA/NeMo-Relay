// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use serde_json::json;
use tokio_stream::StreamExt;
use wasm_bindgen::JsCast;
use wasm_bindgen::JsValue;

fn dummy_function() -> Function {
    JsValue::NULL.unchecked_into()
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

    let llm_request = LLMRequest {
        headers: serde_json::Map::new(),
        content: json!({"messages": []}),
    };
    let llm_intercept = wrap_js_llm_request_intercept_fn(dummy_function());
    let (request, annotated) = llm_intercept("llm", llm_request.clone(), None).unwrap();
    assert_eq!(request.content, llm_request.content);
    assert!(annotated.is_none());

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
        llm_exec(LLMRequest {
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
    subscriber(&Event::mark(
        None,
        uuid::Uuid::now_v7(),
        "native-mark",
        None,
        None,
    ));
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
    let llm_request = LLMRequest {
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
    let request = LLMRequest {
        headers: serde_json::Map::new(),
        content: json!({"messages": []}),
    };
    assert!(codec.decode(&request).is_err());
    let annotated = AnnotatedLLMRequest {
        messages: Vec::new(),
        model: None,
        params: None,
        tools: None,
        tool_choice: None,
        extra: serde_json::Map::new(),
    };
    assert!(codec.encode(&annotated, &request).is_err());

    assert!(
        std::panic::catch_unwind(|| wrap_js_response_codec(dummy_function())).is_err(),
        "non-wasm response codec wrapper should panic"
    );
}
