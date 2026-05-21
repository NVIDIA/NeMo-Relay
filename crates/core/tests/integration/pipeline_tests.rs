// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the Codec pipeline.
//! Tests verify decode/encode around the intercept chain,
//! both streaming and non-streaming paths, and merge-not-replace semantics.

#![allow(clippy::await_holding_lock)]

use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use serde_json::json;
use tokio_stream::Stream;

use nemo_flow::api::event::{Event, ScopeCategory};
use nemo_flow::api::llm::LlmRequest;
use nemo_flow::api::llm::{
    LlmCallExecuteParams, LlmStreamCallExecuteParams, llm_call_execute, llm_stream_call_execute,
};
use nemo_flow::api::registry::{deregister_llm_request_intercept, register_llm_request_intercept};
use nemo_flow::api::runtime::NemoFlowContextState;
use nemo_flow::api::runtime::global_context;
use nemo_flow::api::runtime::{LlmExecutionNextFn, LlmStreamExecutionNextFn};
use nemo_flow::api::runtime::{create_scope_stack, set_thread_scope_stack};
use nemo_flow::api::scope::ScopeType;
use nemo_flow::api::subscriber::{deregister_subscriber, register_subscriber};
use nemo_flow::codec::request::AnnotatedLlmRequest;
use nemo_flow::codec::request::MessageContent;
use nemo_flow::codec::response::AnnotatedLlmResponse;
use nemo_flow::codec::response::FinishReason;
use nemo_flow::codec::traits::{LlmCodec, LlmResponseCodec};
use nemo_flow::error::{FlowError, Result};
use nemo_flow::json::Json;

// ---------------------------------------------------------------------------
// Test isolation
// ---------------------------------------------------------------------------

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn is_scope_event(event: &Event, scope_type: ScopeType, scope_category: ScopeCategory) -> bool {
    event.scope_type() == Some(scope_type) && event.scope_category() == Some(scope_category)
}

fn reset_global() {
    let ctx = global_context();
    let mut state = ctx.write().unwrap();
    *state = NemoFlowContextState::new();
}

fn setup_isolated_thread() {
    let stack = create_scope_stack();
    set_thread_scope_stack(stack);
}

// ---------------------------------------------------------------------------
// TrackingCodec — records decode/encode calls and performs real transformations
// ---------------------------------------------------------------------------

struct TrackingCodec {
    id: String,
    decode_log: Arc<Mutex<Vec<String>>>,
    encode_log: Arc<Mutex<Vec<String>>>,
}

impl LlmCodec for TrackingCodec {
    fn decode(&self, request: &LlmRequest) -> Result<AnnotatedLlmRequest> {
        self.decode_log
            .lock()
            .unwrap()
            .push(format!("decode:{}", self.id));

        // Parse messages from content if present
        let messages = if let Some(msgs) = request.content.get("messages") {
            serde_json::from_value(msgs.clone()).unwrap_or_default()
        } else {
            vec![]
        };

        // Capture unmodeled keys into extra
        let mut extra = serde_json::Map::new();
        if let Some(obj) = request.content.as_object() {
            for (k, v) in obj {
                match k.as_str() {
                    "messages" | "model" => {}
                    _ => {
                        extra.insert(k.clone(), v.clone());
                    }
                }
            }
        }

        Ok(AnnotatedLlmRequest {
            messages,
            model: Some(self.id.clone()),
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
            extra,
        })
    }

    fn encode(&self, annotated: &AnnotatedLlmRequest, original: &LlmRequest) -> Result<LlmRequest> {
        self.encode_log
            .lock()
            .unwrap()
            .push(format!("encode:{}", self.id));

        // Merge-not-replace: overlay structured fields onto original content
        let mut content = original.content.clone();
        if let Some(obj) = content.as_object_mut() {
            if let Some(ref model) = annotated.model {
                obj.insert("model".into(), json!(model));
            }
            if !annotated.messages.is_empty() {
                obj.insert(
                    "messages".into(),
                    serde_json::to_value(&annotated.messages).unwrap(),
                );
            }
            // Merge extra fields back
            for (k, v) in &annotated.extra {
                obj.insert(k.clone(), v.clone());
            }
        }

        Ok(LlmRequest {
            headers: original.headers.clone(),
            content,
        })
    }
}

/// A codec whose decode always fails, for error propagation tests.
struct FailingCodec;

impl LlmCodec for FailingCodec {
    fn decode(&self, _request: &LlmRequest) -> Result<AnnotatedLlmRequest> {
        Err(FlowError::Internal("decode failed on purpose".into()))
    }

    fn encode(
        &self,
        _annotated: &AnnotatedLlmRequest,
        original: &LlmRequest,
    ) -> Result<LlmRequest> {
        Ok(original.clone())
    }
}

// ---------------------------------------------------------------------------
// Helper constructors
// ---------------------------------------------------------------------------

type TrackingCodecResult = (
    Arc<dyn LlmCodec>,
    Arc<Mutex<Vec<String>>>,
    Arc<Mutex<Vec<String>>>,
);

fn make_tracking_codec(id: &str) -> TrackingCodecResult {
    let decode_log = Arc::new(Mutex::new(Vec::new()));
    let encode_log = Arc::new(Mutex::new(Vec::new()));
    let codec = Arc::new(TrackingCodec {
        id: id.to_string(),
        decode_log: decode_log.clone(),
        encode_log: encode_log.clone(),
    });
    (codec, decode_log, encode_log)
}

fn make_llm_request(content: Json) -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content,
    }
}

fn noop_exec_fn() -> LlmExecutionNextFn {
    Arc::new(|_req| Box::pin(async move { Ok(json!({"response": "ok"})) }))
}

fn noop_stream_exec_fn() -> LlmStreamExecutionNextFn {
    Arc::new(|_req| {
        Box::pin(async {
            let stream: Pin<Box<dyn Stream<Item = Result<Json>> + Send>> =
                Box::pin(futures::stream::once(async { Ok(json!({"chunk": 1})) }));
            Ok(stream)
        })
    })
}

// ===========================================================================
// Decode runs before intercepts
// ===========================================================================

#[tokio::test]
async fn test_decode_runs_before_intercepts() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    // Create a TrackingCodec
    let (codec, decode_log, _encode_log) = make_tracking_codec("codec_A");

    // Register an annotated intercept that captures what it receives
    let captured = Arc::new(Mutex::new(None::<Option<AnnotatedLlmRequest>>));
    let cap = captured.clone();
    register_llm_request_intercept(
        "ann_i",
        1,
        false,
        Arc::new(move |_name, req, annotated| {
            *cap.lock().unwrap() = Some(annotated.clone());
            Ok((req, annotated))
        }),
    )
    .unwrap();

    let request = make_llm_request(json!({"messages": [{"role": "user", "content": "hello"}]}));

    let _result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("test_llm")
            .request(request)
            .func(noop_exec_fn())
            .codec(codec)
            .build(),
    )
    .await
    .unwrap();

    // Assert decode was called
    let dl = decode_log.lock().unwrap();
    assert_eq!(dl.len(), 1);
    assert_eq!(dl[0], "decode:codec_A");

    // Assert annotated intercept received Some(AnnotatedLlmRequest) with model == "codec_A"
    let cap_val = captured.lock().unwrap();
    let annotated = cap_val.as_ref().unwrap().as_ref().unwrap();
    assert_eq!(annotated.model, Some("codec_A".into()));

    // Cleanup
    deregister_llm_request_intercept("ann_i").unwrap();
}

// ===========================================================================
// Encode runs after intercepts
// ===========================================================================

#[tokio::test]
async fn test_encode_runs_after_intercepts() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let (codec, _decode_log, encode_log) = make_tracking_codec("codec_B");

    // Annotated intercept modifies the model field
    register_llm_request_intercept(
        "modify_model",
        1,
        false,
        Arc::new(|_name, req, annotated| {
            let mut ann = annotated.unwrap();
            ann.model = Some("modified".into());
            Ok((req, Some(ann)))
        }),
    )
    .unwrap();

    // We capture the request that reaches the exec function to verify encode happened
    let exec_request = Arc::new(Mutex::new(None::<LlmRequest>));
    let er = exec_request.clone();
    let func: LlmExecutionNextFn = Arc::new(move |req| {
        *er.lock().unwrap() = Some(req);
        Box::pin(async move { Ok(json!({"response": "ok"})) })
    });

    let request = make_llm_request(json!({"messages": [{"role": "user", "content": "hi"}]}));

    let _result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("test_llm")
            .request(request)
            .func(func)
            .codec(codec)
            .build(),
    )
    .await
    .unwrap();

    // Assert encode was called
    let el = encode_log.lock().unwrap();
    assert_eq!(el.len(), 1);
    assert_eq!(el[0], "encode:codec_B");

    // Assert the exec function received a request with model="modified"
    let captured_req = exec_request.lock().unwrap();
    let req = captured_req.as_ref().unwrap();
    assert_eq!(req.content["model"], json!("modified"));

    // Cleanup
    deregister_llm_request_intercept("modify_model").unwrap();
}

// ===========================================================================
// Intercept receives both LlmRequest and AnnotatedLlmRequest
// ===========================================================================

#[tokio::test]
async fn test_annotated_intercept_receives_both() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let (codec, _, _) = make_tracking_codec("codec_C");

    let captured_pair = Arc::new(Mutex::new(
        None::<(LlmRequest, Option<AnnotatedLlmRequest>)>,
    ));
    let cp = captured_pair.clone();
    register_llm_request_intercept(
        "capture_both",
        1,
        false,
        Arc::new(move |_name, req, annotated| {
            *cp.lock().unwrap() = Some((req.clone(), annotated.clone()));
            Ok((req, annotated))
        }),
    )
    .unwrap();

    let request = make_llm_request(json!({
        "messages": [{"role": "user", "content": "test"}],
        "model": "original-model"
    }));

    let _result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("test_llm")
            .request(request)
            .func(noop_exec_fn())
            .codec(codec)
            .build(),
    )
    .await
    .unwrap();

    let cap = captured_pair.lock().unwrap();
    let (req, ann) = cap.as_ref().unwrap();

    // LlmRequest is present with original content
    assert!(req.content.get("messages").is_some());

    // AnnotatedLlmRequest is Some with decoded fields
    let annotated = ann.as_ref().unwrap();
    assert_eq!(annotated.model, Some("codec_C".into()));
    assert!(!annotated.messages.is_empty());

    // Cleanup
    deregister_llm_request_intercept("capture_both").unwrap();
}

// ===========================================================================
// Intercept without codec receives None for annotated
// ===========================================================================

#[tokio::test]
async fn test_legacy_intercept_backward_compat() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    // Part 1: Legacy intercept with no Codec
    let legacy_called_1 = Arc::new(Mutex::new(false));
    let lc1 = legacy_called_1.clone();
    register_llm_request_intercept(
        "legacy_1",
        1,
        false,
        Arc::new(move |_name, mut req, annotated| {
            *lc1.lock().unwrap() = true;
            req.headers.insert("x-legacy".into(), json!("was-here"));
            Ok((req, annotated))
        }),
    )
    .unwrap();

    let exec_captured = Arc::new(Mutex::new(None::<LlmRequest>));
    let ec = exec_captured.clone();
    let func: LlmExecutionNextFn = Arc::new(move |req| {
        *ec.lock().unwrap() = Some(req);
        Box::pin(async move { Ok(json!({"response": "ok"})) })
    });

    let request = make_llm_request(json!({"prompt": "hi"}));
    let _result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("test_llm")
            .request(request)
            .func(func)
            .build(),
    )
    .await
    .unwrap();

    assert!(*legacy_called_1.lock().unwrap());
    let cap = exec_captured.lock().unwrap();
    assert_eq!(cap.as_ref().unwrap().headers["x-legacy"], json!("was-here"));

    // Cleanup part 1
    deregister_llm_request_intercept("legacy_1").unwrap();

    // Part 2: Legacy intercept WITH Codec — legacy intercept still runs
    reset_global();

    let (codec, _, _) = make_tracking_codec("codec_D");

    let legacy_called_2 = Arc::new(Mutex::new(false));
    let lc2 = legacy_called_2.clone();
    register_llm_request_intercept(
        "legacy_2",
        1,
        false,
        Arc::new(move |_name, mut req, annotated| {
            *lc2.lock().unwrap() = true;
            req.headers.insert("x-legacy-2".into(), json!("also-here"));
            Ok((req, annotated))
        }),
    )
    .unwrap();

    let exec_captured_2 = Arc::new(Mutex::new(None::<LlmRequest>));
    let ec2 = exec_captured_2.clone();
    let func2: LlmExecutionNextFn = Arc::new(move |req| {
        *ec2.lock().unwrap() = Some(req);
        Box::pin(async move { Ok(json!({"response": "ok"})) })
    });

    let request2 = make_llm_request(json!({"messages": [{"role": "user", "content": "hi"}]}));
    let _result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("test_llm")
            .request(request2)
            .func(func2)
            .codec(codec)
            .build(),
    )
    .await
    .unwrap();

    assert!(*legacy_called_2.lock().unwrap());
    // Legacy header modifications should be preserved through encode
    let cap2 = exec_captured_2.lock().unwrap();
    assert_eq!(
        cap2.as_ref().unwrap().headers["x-legacy-2"],
        json!("also-here")
    );

    // Cleanup
    deregister_llm_request_intercept("legacy_2").unwrap();
}

// ===========================================================================
// Streaming path also decodes
// ===========================================================================

#[tokio::test]
async fn test_stream_path_also_decodes() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let (codec, decode_log, _) = make_tracking_codec("codec_S");

    let captured_ann = Arc::new(Mutex::new(None::<Option<AnnotatedLlmRequest>>));
    let ca = captured_ann.clone();
    register_llm_request_intercept(
        "stream_ann",
        1,
        false,
        Arc::new(move |_name, req, annotated| {
            *ca.lock().unwrap() = Some(annotated.clone());
            Ok((req, annotated))
        }),
    )
    .unwrap();

    let request = make_llm_request(json!({"messages": [{"role": "user", "content": "stream me"}]}));

    let collector: Box<dyn FnMut(Json) -> Result<()> + Send> = Box::new(|_chunk| Ok(()));
    let finalizer: Box<dyn FnOnce() -> Json + Send> = Box::new(|| json!({"done": true}));

    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("test_stream")
            .request(request)
            .func(noop_stream_exec_fn())
            .collector(collector)
            .finalizer(finalizer)
            .codec(codec)
            .build(),
    )
    .await
    .unwrap();

    // Consume the stream to trigger full pipeline
    while let Some(_chunk) = stream.next().await {}

    // Assert decode was called
    let dl = decode_log.lock().unwrap();
    assert_eq!(dl.len(), 1);
    assert_eq!(dl[0], "decode:codec_S");

    // Assert annotated intercept received Some(AnnotatedLlmRequest)
    let ann_val = captured_ann.lock().unwrap();
    assert!(ann_val.as_ref().unwrap().is_some());
    assert_eq!(
        ann_val.as_ref().unwrap().as_ref().unwrap().model,
        Some("codec_S".into())
    );

    // Cleanup
    deregister_llm_request_intercept("stream_ann").unwrap();
}

// ===========================================================================
// Shared helper serves both paths
// ===========================================================================

#[tokio::test]
async fn test_shared_helper_both_paths() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let (codec, decode_log, encode_log) = make_tracking_codec("codec_shared");

    // Same annotated intercept that modifies model — used for both calls
    let ann_call_count = Arc::new(Mutex::new(0u32));
    let acc = ann_call_count.clone();
    register_llm_request_intercept(
        "shared_ann",
        1,
        false,
        Arc::new(move |_name, req, annotated| {
            *acc.lock().unwrap() += 1;
            Ok((req, annotated))
        }),
    )
    .unwrap();

    // Non-streaming call
    let request1 =
        make_llm_request(json!({"messages": [{"role": "user", "content": "non-stream"}]}));
    let _result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("test_llm")
            .request(request1)
            .func(noop_exec_fn())
            .codec(codec.clone())
            .build(),
    )
    .await
    .unwrap();

    // Streaming call
    let request2 = make_llm_request(json!({"messages": [{"role": "user", "content": "stream"}]}));
    let collector: Box<dyn FnMut(Json) -> Result<()> + Send> = Box::new(|_| Ok(()));
    let finalizer: Box<dyn FnOnce() -> Json + Send> = Box::new(|| json!({"done": true}));

    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("test_stream")
            .request(request2)
            .func(noop_stream_exec_fn())
            .collector(collector)
            .finalizer(finalizer)
            .codec(codec)
            .build(),
    )
    .await
    .unwrap();
    while stream.next().await.is_some() {}

    // Both paths should have triggered decode and encode
    let dl = decode_log.lock().unwrap();
    assert_eq!(dl.len(), 2, "decode should have been called twice");

    let el = encode_log.lock().unwrap();
    assert_eq!(el.len(), 2, "encode should have been called twice");

    // Annotated intercept should have fired twice
    assert_eq!(*ann_call_count.lock().unwrap(), 2);

    // Cleanup
    deregister_llm_request_intercept("shared_ann").unwrap();
}

// ===========================================================================
// Explicit codec parameter selects the correct codec
// ===========================================================================

#[tokio::test]
async fn test_explicit_codec_param_overrides() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    // Create two codecs
    let (_codec_a, _, _) = make_tracking_codec("A");
    let (codec_b, _, _) = make_tracking_codec("B");

    // Capture which model the annotated intercept sees
    let captured_model = Arc::new(Mutex::new(None::<String>));
    let cm = captured_model.clone();
    register_llm_request_intercept(
        "check_model",
        1,
        false,
        Arc::new(move |_name, req, annotated| {
            if let Some(ref ann) = annotated {
                *cm.lock().unwrap() = ann.model.clone();
            }
            Ok((req, annotated))
        }),
    )
    .unwrap();

    let request = make_llm_request(json!({"messages": [{"role": "user", "content": "test"}]}));

    // Pass codec B directly
    let _result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("test_llm")
            .request(request)
            .func(noop_exec_fn())
            .codec(codec_b)
            .build(),
    )
    .await
    .unwrap();

    // Annotated intercept should see model="B"
    let model = captured_model.lock().unwrap();
    assert_eq!(model.as_deref(), Some("B"));

    // Cleanup
    deregister_llm_request_intercept("check_model").unwrap();
}

// ===========================================================================
// Encode uses merge-not-replace semantics
// ===========================================================================

#[tokio::test]
async fn test_encode_merge_not_replace() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let (codec, _, _) = make_tracking_codec("codec_merge");

    // Annotated intercept modifies the model
    register_llm_request_intercept(
        "merge_mod",
        1,
        false,
        Arc::new(|_name, req, annotated| {
            let mut ann = annotated.unwrap();
            ann.model = Some("new_model".into());
            Ok((req, Some(ann)))
        }),
    )
    .unwrap();

    // Capture the request arriving at the exec function
    let exec_request = Arc::new(Mutex::new(None::<LlmRequest>));
    let er = exec_request.clone();
    let func: LlmExecutionNextFn = Arc::new(move |req| {
        *er.lock().unwrap() = Some(req);
        Box::pin(async move { Ok(json!({"response": "ok"})) })
    });

    // Original content has "stream": true and "custom_key": 42
    let request = make_llm_request(json!({
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
        "custom_key": 42
    }));

    let _result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("test_llm")
            .request(request)
            .func(func)
            .codec(codec)
            .build(),
    )
    .await
    .unwrap();

    // Check the encoded request
    let captured = exec_request.lock().unwrap();
    let req = captured.as_ref().unwrap();

    // The modified model should be present
    assert_eq!(req.content["model"], json!("new_model"));

    // The preserved keys from original should still be present (merge-not-replace)
    assert_eq!(req.content["stream"], json!(true));
    assert_eq!(req.content["custom_key"], json!(42));

    // Cleanup
    deregister_llm_request_intercept("merge_mod").unwrap();
}

// ===========================================================================
// Intercept chain respects priority order
// ===========================================================================

#[tokio::test]
async fn test_unified_chain_priority_order() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let (codec, _, _) = make_tracking_codec("codec_pri");

    let call_log = Arc::new(Mutex::new(Vec::<String>::new()));

    // Legacy intercept at priority 10
    let cl1 = call_log.clone();
    register_llm_request_intercept(
        "legacy_p10",
        10,
        false,
        Arc::new(move |_name, req, annotated| {
            cl1.lock().unwrap().push("legacy_p10".into());
            Ok((req, annotated))
        }),
    )
    .unwrap();

    // Annotated intercept at priority 5 (should run first)
    let cl2 = call_log.clone();
    register_llm_request_intercept(
        "annotated_p5",
        5,
        false,
        Arc::new(move |_name, req, annotated| {
            cl2.lock().unwrap().push("annotated_p5".into());
            Ok((req, annotated))
        }),
    )
    .unwrap();

    let request = make_llm_request(json!({"messages": [{"role": "user", "content": "order"}]}));

    let _result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("test_llm")
            .request(request)
            .func(noop_exec_fn())
            .codec(codec)
            .build(),
    )
    .await
    .unwrap();

    // Assert annotated (priority 5) ran before legacy (priority 10)
    let log = call_log.lock().unwrap();
    assert_eq!(log.len(), 2);
    assert_eq!(log[0], "annotated_p5");
    assert_eq!(log[1], "legacy_p10");

    // Cleanup
    deregister_llm_request_intercept("legacy_p10").unwrap();
    deregister_llm_request_intercept("annotated_p5").unwrap();
}

// ===========================================================================
// Edge case: No codec — annotated intercept receives None
// ===========================================================================

#[tokio::test]
async fn test_no_codec_annotated_intercept_receives_none() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    // Register annotated intercept but NO codec
    let captured_ann = Arc::new(Mutex::new(None::<Option<AnnotatedLlmRequest>>));
    let ca = captured_ann.clone();
    register_llm_request_intercept(
        "no_codec_ann",
        1,
        false,
        Arc::new(move |_name, req, annotated| {
            *ca.lock().unwrap() = Some(annotated.clone());
            Ok((req, annotated))
        }),
    )
    .unwrap();

    let request = make_llm_request(json!({"messages": [{"role": "user", "content": "hi"}]}));

    let _result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("test_llm")
            .request(request)
            .func(noop_exec_fn())
            .build(),
    )
    .await
    .unwrap();

    // Annotated intercept should receive None for AnnotatedLlmRequest
    let ann = captured_ann.lock().unwrap();
    assert!(ann.as_ref().unwrap().is_none());

    // Cleanup
    deregister_llm_request_intercept("no_codec_ann").unwrap();
}

// ===========================================================================
// Edge case: Decode error propagates
// ===========================================================================

#[tokio::test]
async fn test_decode_error_propagates() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    // Create a codec that always fails on decode
    let failing_codec: Arc<dyn LlmCodec> = Arc::new(FailingCodec);

    let request = make_llm_request(json!({"messages": []}));

    let result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("test_llm")
            .request(request)
            .func(noop_exec_fn())
            .codec(failing_codec)
            .build(),
    )
    .await;

    // The execute call should return Err with the decode error
    assert!(result.is_err());
    match result.unwrap_err() {
        FlowError::Internal(msg) => {
            assert!(msg.contains("decode failed on purpose"), "Got: {}", msg);
        }
        other => panic!("Expected Internal error from decode, got: {:?}", other),
    }
}

// ===========================================================================
// Response Codec integration tests
// ===========================================================================

/// Mock response codec that returns a fixed AnnotatedLlmResponse.
struct MockResponseCodec;

impl LlmResponseCodec for MockResponseCodec {
    fn decode_response(&self, _response: &Json) -> Result<AnnotatedLlmResponse> {
        Ok(AnnotatedLlmResponse {
            id: Some("mock-resp-id".into()),
            model: Some("mock-model".into()),
            message: Some(MessageContent::Text("mock response text".into())),
            tool_calls: None,
            finish_reason: Some(FinishReason::Complete),
            usage: None,
            api_specific: None,
            extra: serde_json::Map::new(),
        })
    }
}

/// Mock response codec that always fails.
struct FailingResponseCodec;

impl LlmResponseCodec for FailingResponseCodec {
    fn decode_response(&self, _response: &Json) -> Result<AnnotatedLlmResponse> {
        Err(FlowError::Internal("decode failed".into()))
    }
}

#[tokio::test]
async fn test_response_codec_populates_annotated_response() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::new()));
    let ec = events.clone();
    register_subscriber(
        "resp_codec_sub",
        Arc::new(move |e: &Event| {
            ec.lock().unwrap().push(e.clone());
        }),
    )
    .unwrap();

    let request = make_llm_request(json!({"messages": [{"role": "user", "content": "hello"}]}));
    let response_codec: Arc<dyn LlmResponseCodec> = Arc::new(MockResponseCodec);

    let _result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("test_llm")
            .request(request)
            .func(noop_exec_fn())
            .response_codec(response_codec)
            .build(),
    )
    .await
    .unwrap();

    let captured = events.lock().unwrap();
    let end_event = captured
        .iter()
        .find(|e| is_scope_event(e, ScopeType::Llm, ScopeCategory::End))
        .expect("expected LlmEnd event");

    let ann = end_event
        .annotated_response()
        .expect("annotated_response should be Some when response codec is active");
    assert_eq!(ann.id, Some("mock-resp-id".into()));
    assert_eq!(ann.response_text(), Some("mock response text"));

    drop(captured);
    deregister_subscriber("resp_codec_sub").unwrap();
}

#[tokio::test]
async fn test_response_codec_none_when_no_codec() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::new()));
    let ec = events.clone();
    register_subscriber(
        "no_resp_codec_sub",
        Arc::new(move |e: &Event| {
            ec.lock().unwrap().push(e.clone());
        }),
    )
    .unwrap();

    let request = make_llm_request(json!({"messages": [{"role": "user", "content": "hello"}]}));

    let _result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("test_llm")
            .request(request)
            .func(noop_exec_fn())
            .build(),
    )
    .await
    .unwrap();

    let captured = events.lock().unwrap();
    let end_event = captured
        .iter()
        .find(|e| is_scope_event(e, ScopeType::Llm, ScopeCategory::End))
        .expect("expected LlmEnd event");

    assert!(
        end_event.annotated_response().is_none(),
        "annotated_response should be None when no response codec"
    );

    drop(captured);
    deregister_subscriber("no_resp_codec_sub").unwrap();
}

#[tokio::test]
async fn test_response_codec_failure_non_fatal() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::new()));
    let ec = events.clone();
    register_subscriber(
        "fail_resp_codec_sub",
        Arc::new(move |e: &Event| {
            ec.lock().unwrap().push(e.clone());
        }),
    )
    .unwrap();

    let request = make_llm_request(json!({"messages": [{"role": "user", "content": "hello"}]}));
    let response_codec: Arc<dyn LlmResponseCodec> = Arc::new(FailingResponseCodec);

    // Pipeline should NOT return an error despite decode failure (non-fatal)
    let result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("test_llm")
            .request(request)
            .func(noop_exec_fn())
            .response_codec(response_codec)
            .build(),
    )
    .await;

    assert!(
        result.is_ok(),
        "Pipeline should succeed even when response codec fails"
    );

    let captured = events.lock().unwrap();
    let end_event = captured
        .iter()
        .find(|e| is_scope_event(e, ScopeType::Llm, ScopeCategory::End))
        .expect("expected LlmEnd event");

    assert!(
        end_event.annotated_response().is_none(),
        "annotated_response should be None when decode fails"
    );
    assert!(
        end_event.output().is_some(),
        "raw output should still be present"
    );

    drop(captured);
    deregister_subscriber("fail_resp_codec_sub").unwrap();
}

#[tokio::test]
async fn test_request_codec_populates_annotated_request() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::new()));
    let ec = events.clone();
    register_subscriber(
        "req_codec_ann_sub",
        Arc::new(move |e: &Event| {
            ec.lock().unwrap().push(e.clone());
        }),
    )
    .unwrap();

    let (codec, _, _) = make_tracking_codec("req_codec_test");

    let request = make_llm_request(json!({"messages": [{"role": "user", "content": "hello"}]}));

    let _result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("test_llm")
            .request(request)
            .func(noop_exec_fn())
            .codec(codec)
            .build(),
    )
    .await
    .unwrap();

    let captured = events.lock().unwrap();
    let start_event = captured
        .iter()
        .find(|e| is_scope_event(e, ScopeType::Llm, ScopeCategory::Start))
        .expect("expected LlmStart event");

    let ann = start_event
        .annotated_request()
        .expect("annotated_request should be Some when request codec is active");
    assert_eq!(ann.model, Some("req_codec_test".into()));

    drop(captured);
    deregister_subscriber("req_codec_ann_sub").unwrap();
}

#[tokio::test]
async fn test_stream_response_codec_populates_annotated_response() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::new()));
    let ec = events.clone();
    register_subscriber(
        "stream_resp_codec_sub",
        Arc::new(move |e: &Event| {
            ec.lock().unwrap().push(e.clone());
        }),
    )
    .unwrap();

    let request = make_llm_request(json!({"messages": [{"role": "user", "content": "stream me"}]}));
    let response_codec: Arc<dyn LlmResponseCodec> = Arc::new(MockResponseCodec);

    let collector: Box<dyn FnMut(Json) -> Result<()> + Send> = Box::new(|_chunk| Ok(()));
    let finalizer: Box<dyn FnOnce() -> Json + Send> =
        Box::new(|| json!({"aggregated": "response"}));

    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("test_stream")
            .request(request)
            .func(noop_stream_exec_fn())
            .collector(collector)
            .finalizer(finalizer)
            .response_codec(response_codec)
            .build(),
    )
    .await
    .unwrap();

    // Drain the stream to trigger finalization and END event
    while let Some(_chunk) = stream.next().await {}

    let captured = events.lock().unwrap();
    let end_event = captured
        .iter()
        .find(|e| is_scope_event(e, ScopeType::Llm, ScopeCategory::End))
        .expect("expected LlmEnd event after stream drain");

    let ann = end_event
        .annotated_response()
        .expect("annotated_response should be Some on stream path when response codec is active");
    assert_eq!(ann.id, Some("mock-resp-id".into()));
    assert_eq!(ann.response_text(), Some("mock response text"));

    drop(captured);
    deregister_subscriber("stream_resp_codec_sub").unwrap();
}
