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

use nemo_relay::api::event::{Event, PendingMarkSpec, ScopeCategory};
use nemo_relay::api::llm::{
    LlmCallExecuteParams, LlmStreamCallExecuteParams, llm_call_execute, llm_stream_call_execute,
};
use nemo_relay::api::llm::{LlmRequest, LlmRequestInterceptOutcome};
use nemo_relay::api::optimization::record_llm_optimization_contribution;
use nemo_relay::api::registry::{
    deregister_llm_request_intercept, deregister_llm_sanitize_request_guardrail,
    deregister_llm_sanitize_response_guardrail, register_llm_request_intercept,
    register_llm_sanitize_request_guardrail, register_llm_sanitize_response_guardrail,
};
use nemo_relay::api::runtime::NemoRelayContextState;
use nemo_relay::api::runtime::global_context;
use nemo_relay::api::runtime::{LlmExecutionNextFn, LlmJsonStream, LlmStreamExecutionNextFn};
use nemo_relay::api::runtime::{create_scope_stack, set_thread_scope_stack};
use nemo_relay::api::scope::ScopeType;
use nemo_relay::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use nemo_relay::codec::openai_chat::OpenAIChatCodec;
use nemo_relay::codec::optimization::{
    LlmOptimizationContribution, LlmOptimizationKind, LlmOptimizationModel,
    LlmOptimizationModelTransition,
};
use nemo_relay::codec::request::{AnnotatedLlmRequest, Message, MessageContent};
use nemo_relay::codec::response::FinishReason;
use nemo_relay::codec::response::{
    AnnotatedLlmResponse, CostEstimate, CostSource, PricingCatalog, PricingResolver, Usage,
    reset_active_pricing_resolver, set_active_pricing_resolver,
};
use nemo_relay::codec::traits::{LlmCodec, LlmResponseCodec};
use nemo_relay::error::{FlowError, Result};
use nemo_relay::json::Json;

// ---------------------------------------------------------------------------
// Test isolation
// ---------------------------------------------------------------------------

static TEST_MUTEX: Mutex<()> = Mutex::new(());

struct ResetPricingResolverGuard;

impl Drop for ResetPricingResolverGuard {
    fn drop(&mut self) {
        let _ = reset_active_pricing_resolver();
    }
}

fn is_scope_event(event: &Event, scope_type: ScopeType, scope_category: ScopeCategory) -> bool {
    event.scope_type() == Some(scope_type) && event.scope_category() == Some(scope_category)
}

fn reset_global() {
    let ctx = global_context();
    let mut state = ctx.write().unwrap();
    *state = NemoRelayContextState::new();
}

fn setup_isolated_thread() {
    let stack = create_scope_stack();
    set_thread_scope_stack(stack);
}

fn captured_events_snapshot(events: &Arc<Mutex<Vec<Event>>>) -> Vec<Event> {
    flush_subscribers().unwrap();
    events.lock().unwrap().clone()
}

fn install_mock_response_pricing() {
    let catalog = PricingCatalog::from_json_str(
        &json!({
            "version": 1,
            "entries": [
                {
                    "provider": "openai",
                    "model_id": "gpt-4o-mini",
                    "pricing_as_of": "2026-06-05",
                    "pricing_source": "test",
                    "rates": {
                        "input_per_million": 0.15,
                        "output_per_million": 0.60,
                        "cache_read_per_million": 0.075
                    },
                    "prompt_cache": {
                        "read_accounting": "included_in_prompt_tokens"
                    }
                }
            ]
        })
        .to_string(),
    )
    .unwrap();
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![catalog])).unwrap();
}

fn install_routed_response_pricing() {
    let catalog = PricingCatalog::from_json_str(
        &json!({
            "version": 1,
            "entries": [
                {
                    "provider": "test", "model_id": "baseline",
                    "pricing_as_of": "2026-07-08", "pricing_source": "test",
                    "rates": {"input_per_million": 4.0, "output_per_million": 8.0},
                    "prompt_cache": {"read_accounting": "included_in_prompt_tokens"}
                },
                {
                    "provider": "test", "model_id": "effective",
                    "pricing_as_of": "2026-07-08", "pricing_source": "test",
                    "rates": {"input_per_million": 1.0, "output_per_million": 2.0},
                    "prompt_cache": {"read_accounting": "included_in_prompt_tokens"}
                }
            ]
        })
        .to_string(),
    )
    .unwrap();
    set_active_pricing_resolver(PricingResolver::from_catalogs(vec![catalog])).unwrap();
}

fn routed_model_contribution() -> LlmOptimizationContribution {
    let mut contribution =
        LlmOptimizationContribution::new("test.router", LlmOptimizationKind::model_routing());
    contribution.model_transition = Some(LlmOptimizationModelTransition {
        baseline: Some(LlmOptimizationModel::new("baseline").with_provider("test")),
        effective: Some(LlmOptimizationModel::new("effective").with_provider("test")),
    });
    contribution
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

fn make_openai_chat_request(content: &str) -> LlmRequest {
    make_llm_request(json!({
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": content}],
    }))
}

fn make_openai_chat_response(content: &str) -> Json {
    json!({
        "id": "chatcmpl-test",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })
}

fn first_message_text(request: &AnnotatedLlmRequest) -> Option<&str> {
    match request.messages.first()? {
        Message::System {
            content: MessageContent::Text(text),
            ..
        }
        | Message::User {
            content: MessageContent::Text(text),
            ..
        }
        | Message::Tool {
            content: MessageContent::Text(text),
            ..
        } => Some(text.as_str()),
        Message::Assistant {
            content: Some(MessageContent::Text(text)),
            ..
        } => Some(text.as_str()),
        _ => None,
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
            Ok(nemo_relay::api::llm::LlmRequestInterceptOutcome::new(
                req, annotated,
            ))
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
        Arc::new(|_name, mut req, annotated| {
            let mut ann = annotated.unwrap();
            ann.model = Some("modified".into());
            req.headers.insert("x-codec-route".into(), json!("blue"));
            Ok(nemo_relay::api::llm::LlmRequestInterceptOutcome::new(
                req,
                Some(ann),
            ))
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
    assert_eq!(req.headers["x-codec-route"], json!("blue"));

    // Cleanup
    deregister_llm_request_intercept("modify_model").unwrap();
}

#[tokio::test]
async fn test_codec_rejects_raw_content_mutation_before_lifecycle() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::new()));
    let captured_events = events.clone();
    register_subscriber(
        "codec_raw_mutation_subscriber",
        Arc::new(move |event| captured_events.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let later_intercept_called = Arc::new(Mutex::new(false));
    register_llm_request_intercept(
        "codec_raw_mutation",
        1,
        false,
        Arc::new(|_name, mut request, annotated| {
            request.content["model"] = json!("raw-model-edit");
            Ok(LlmRequestInterceptOutcome::new(request, annotated)
                .with_pending_mark(PendingMarkSpec::builder().name("must.not.emit").build()))
        }),
    )
    .unwrap();
    let later_called = later_intercept_called.clone();
    register_llm_request_intercept(
        "codec_after_raw_mutation",
        2,
        false,
        Arc::new(move |_name, request, annotated| {
            *later_called.lock().unwrap() = true;
            Ok(LlmRequestInterceptOutcome::new(request, annotated))
        }),
    )
    .unwrap();

    let provider_called = Arc::new(Mutex::new(false));
    let called = provider_called.clone();
    let provider: LlmExecutionNextFn = Arc::new(move |_request| {
        *called.lock().unwrap() = true;
        Box::pin(async { Ok(json!({"response": "unexpected"})) })
    });
    let (codec, _, encode_log) = make_tracking_codec("codec_raw_read_only");
    let error = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("test_llm")
            .request(make_llm_request(json!({
                "model": "original",
                "messages": [{"role": "user", "content": "hello"}]
            })))
            .func(provider)
            .codec(codec)
            .build(),
    )
    .await
    .expect_err("raw content mutation must fail on the codec path");

    assert!(error.to_string().contains("request.content"));
    assert!(!*later_intercept_called.lock().unwrap());
    assert!(!*provider_called.lock().unwrap());
    assert!(encode_log.lock().unwrap().is_empty());
    assert!(captured_events_snapshot(&events).is_empty());

    deregister_llm_request_intercept("codec_raw_mutation").unwrap();
    deregister_llm_request_intercept("codec_after_raw_mutation").unwrap();
    deregister_subscriber("codec_raw_mutation_subscriber").unwrap();
}

#[tokio::test]
async fn test_codec_rejects_missing_annotation_before_lifecycle() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    register_llm_request_intercept(
        "codec_missing_annotation",
        1,
        false,
        Arc::new(|_name, request, _annotated| Ok(LlmRequestInterceptOutcome::new(request, None))),
    )
    .unwrap();

    let provider_called = Arc::new(Mutex::new(false));
    let called = provider_called.clone();
    let provider: LlmExecutionNextFn = Arc::new(move |_request| {
        *called.lock().unwrap() = true;
        Box::pin(async { Ok(json!({"response": "unexpected"})) })
    });
    let (codec, _, encode_log) = make_tracking_codec("codec_annotation_required");
    let error = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("test_llm")
            .request(make_llm_request(json!({
                "messages": [{"role": "user", "content": "hello"}]
            })))
            .func(provider)
            .codec(codec)
            .build(),
    )
    .await
    .expect_err("missing annotation must fail on the codec path");

    assert!(error.to_string().contains("omitted annotated_request"));
    assert!(!*provider_called.lock().unwrap());
    assert!(encode_log.lock().unwrap().is_empty());

    deregister_llm_request_intercept("codec_missing_annotation").unwrap();
}

#[tokio::test]
async fn test_stream_codec_rejects_raw_content_mutation_before_lifecycle() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::new()));
    let captured_events = events.clone();
    register_subscriber(
        "stream_codec_raw_mutation_subscriber",
        Arc::new(move |event| captured_events.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    register_llm_request_intercept(
        "stream_codec_raw_mutation",
        1,
        false,
        Arc::new(|_name, mut request, annotated| {
            request.content["model"] = json!("raw-stream-edit");
            Ok(LlmRequestInterceptOutcome::new(request, annotated))
        }),
    )
    .unwrap();

    let provider_called = Arc::new(Mutex::new(false));
    let called = provider_called.clone();
    let provider: LlmStreamExecutionNextFn = Arc::new(move |_request| {
        *called.lock().unwrap() = true;
        Box::pin(async {
            let stream: Pin<Box<dyn Stream<Item = Result<Json>> + Send>> =
                Box::pin(futures::stream::empty());
            Ok(stream)
        })
    });
    let (codec, _, _) = make_tracking_codec("stream_codec_raw_read_only");
    let result = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("test_stream")
            .request(make_llm_request(json!({
                "model": "original",
                "messages": [{"role": "user", "content": "hello"}]
            })))
            .func(provider)
            .collector(Box::new(|_| Ok(())))
            .finalizer(Box::new(|| json!({"done": true})))
            .codec(codec)
            .build(),
    )
    .await;
    let error = match result {
        Ok(_) => panic!("raw content mutation must fail on the streaming codec path"),
        Err(error) => error,
    };

    assert!(error.to_string().contains("request.content"));
    assert!(!*provider_called.lock().unwrap());
    assert!(captured_events_snapshot(&events).is_empty());

    deregister_llm_request_intercept("stream_codec_raw_mutation").unwrap();
    deregister_subscriber("stream_codec_raw_mutation_subscriber").unwrap();
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
            Ok(nemo_relay::api::llm::LlmRequestInterceptOutcome::new(
                req, annotated,
            ))
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
async fn test_canonical_intercept_with_and_without_codec() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    // Part 1: canonical intercept with no codec.
    let legacy_called_1 = Arc::new(Mutex::new(false));
    let lc1 = legacy_called_1.clone();
    register_llm_request_intercept(
        "legacy_1",
        1,
        false,
        Arc::new(move |_name, mut req, annotated| {
            *lc1.lock().unwrap() = true;
            req.headers.insert("x-legacy".into(), json!("was-here"));
            Ok(nemo_relay::api::llm::LlmRequestInterceptOutcome::new(
                req, annotated,
            ))
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

    // Part 2: canonical intercept with a codec.
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
            Ok(nemo_relay::api::llm::LlmRequestInterceptOutcome::new(
                req, annotated,
            ))
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
            Ok(nemo_relay::api::llm::LlmRequestInterceptOutcome::new(
                req, annotated,
            ))
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
            Ok(nemo_relay::api::llm::LlmRequestInterceptOutcome::new(
                req, annotated,
            ))
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
            Ok(nemo_relay::api::llm::LlmRequestInterceptOutcome::new(
                req, annotated,
            ))
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
            Ok(nemo_relay::api::llm::LlmRequestInterceptOutcome::new(
                req,
                Some(ann),
            ))
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
            Ok(nemo_relay::api::llm::LlmRequestInterceptOutcome::new(
                req, annotated,
            ))
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
            Ok(nemo_relay::api::llm::LlmRequestInterceptOutcome::new(
                req, annotated,
            ))
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
            Ok(nemo_relay::api::llm::LlmRequestInterceptOutcome::new(
                req, annotated,
            ))
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
            model: Some("gpt-4o-mini".into()),
            message: Some(MessageContent::Text("mock response text".into())),
            tool_calls: None,
            finish_reason: Some(FinishReason::Complete),
            usage: Some(Usage {
                prompt_tokens: Some(1_000),
                completion_tokens: Some(500),
                total_tokens: Some(1_500),
                cache_read_tokens: Some(200),
                cache_write_tokens: None,
                cost: None,
            }),
            optimization_summary: None,
            api_specific: None,
            extra: serde_json::Map::new(),
        })
    }
}

struct RoutedAliasResponseCodec {
    cost_source: CostSource,
}

impl LlmResponseCodec for RoutedAliasResponseCodec {
    fn decode_response(&self, _response: &Json) -> Result<AnnotatedLlmResponse> {
        let provider_reported = self.cost_source == CostSource::ProviderReported;
        Ok(AnnotatedLlmResponse {
            model: Some("provider-response-alias".to_string()),
            finish_reason: Some(FinishReason::Complete),
            usage: Some(Usage {
                prompt_tokens: Some(1_000),
                completion_tokens: Some(500),
                total_tokens: Some(1_500),
                cost: Some(CostEstimate {
                    total: Some(if provider_reported { 0.42 } else { 99.0 }),
                    currency: "USD".to_string(),
                    input: None,
                    output: None,
                    cache_read: None,
                    cache_write: None,
                    source: self.cost_source.clone(),
                    pricing_provider: Some("test".to_string()),
                    pricing_model: Some("provider-response-alias".to_string()),
                    pricing_as_of: Some(if provider_reported {
                        "2026-07-01".to_string()
                    } else {
                        "2020-01-01".to_string()
                    }),
                    pricing_source: Some(if provider_reported {
                        "provider".to_string()
                    } else {
                        "stale-alias-pricing".to_string()
                    }),
                }),
                ..Usage::default()
            }),
            ..AnnotatedLlmResponse::default()
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
    let _pricing_guard = ResetPricingResolverGuard;
    reset_global();
    setup_isolated_thread();
    install_mock_response_pricing();

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
            .name("openai")
            .request(request)
            .func(noop_exec_fn())
            .response_codec(response_codec)
            .build(),
    )
    .await
    .unwrap();

    let captured = captured_events_snapshot(&events);
    let end_event = captured
        .iter()
        .find(|e| is_scope_event(e, ScopeType::Llm, ScopeCategory::End))
        .expect("expected LlmEnd event");

    let ann = end_event
        .annotated_response()
        .expect("annotated_response should be Some when response codec is active");
    assert_eq!(ann.id, Some("mock-resp-id".into()));
    assert_eq!(ann.response_text(), Some("mock response text"));
    assert_eq!(
        ann.usage
            .as_ref()
            .and_then(|usage| usage.cost.as_ref())
            .and_then(|cost| cost.total),
        Some(0.000_435)
    );

    deregister_subscriber("resp_codec_sub").unwrap();
}

#[tokio::test]
async fn test_response_codec_annotation_uses_sanitized_managed_response() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::new()));
    let ec = events.clone();
    register_subscriber(
        "sanitized_resp_codec_sub",
        Arc::new(move |e: &Event| {
            ec.lock().unwrap().push(e.clone());
        }),
    )
    .unwrap();
    register_llm_sanitize_response_guardrail(
        "sanitize_resp_codec_annotation",
        1,
        Arc::new(|_response| make_openai_chat_response("Sanitized")),
    )
    .unwrap();

    let func: LlmExecutionNextFn =
        Arc::new(|_req| Box::pin(async move { Ok(make_openai_chat_response("SECRET")) }));
    let response_codec: Arc<dyn LlmResponseCodec> = Arc::new(OpenAIChatCodec);

    let result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(make_openai_chat_request("hello"))
            .func(func)
            .response_codec(response_codec)
            .build(),
    )
    .await
    .unwrap();
    assert_eq!(result["choices"][0]["message"]["content"], json!("SECRET"));

    let captured = captured_events_snapshot(&events);
    let end_event = captured
        .iter()
        .find(|e| is_scope_event(e, ScopeType::Llm, ScopeCategory::End))
        .expect("expected LlmEnd event");
    assert_eq!(
        end_event.output().unwrap()["choices"][0]["message"]["content"],
        json!("Sanitized")
    );
    let annotated = end_event
        .annotated_response()
        .expect("annotated_response should be decoded from sanitized output");
    assert_eq!(annotated.response_text(), Some("Sanitized"));
    assert!(
        !serde_json::to_string(end_event).unwrap().contains("SECRET"),
        "end event should not retain raw response content"
    );

    deregister_subscriber("sanitized_resp_codec_sub").unwrap();
    deregister_llm_sanitize_response_guardrail("sanitize_resp_codec_annotation").unwrap();
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

    let captured = captured_events_snapshot(&events);
    let end_event = captured
        .iter()
        .find(|e| is_scope_event(e, ScopeType::Llm, ScopeCategory::End))
        .expect("expected LlmEnd event");

    assert!(
        end_event.annotated_response().is_none(),
        "annotated_response should be None when no response codec"
    );

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

    let captured = captured_events_snapshot(&events);
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

    let captured = captured_events_snapshot(&events);
    let start_event = captured
        .iter()
        .find(|e| is_scope_event(e, ScopeType::Llm, ScopeCategory::Start))
        .expect("expected LlmStart event");

    let ann = start_event
        .annotated_request()
        .expect("annotated_request should be Some when request codec is active");
    assert_eq!(ann.model, Some("req_codec_test".into()));

    deregister_subscriber("req_codec_ann_sub").unwrap();
}

#[tokio::test]
async fn test_request_codec_annotation_uses_sanitized_start_payload() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::new()));
    let ec = events.clone();
    register_subscriber(
        "sanitized_req_codec_sub",
        Arc::new(move |e: &Event| {
            ec.lock().unwrap().push(e.clone());
        }),
    )
    .unwrap();
    register_llm_sanitize_request_guardrail(
        "sanitize_req_codec_annotation",
        1,
        Arc::new(|request| LlmRequest {
            headers: request.headers,
            content: make_openai_chat_request("Sanitized").content,
        }),
    )
    .unwrap();

    let request_codec: Arc<dyn LlmCodec> = Arc::new(OpenAIChatCodec);
    let _result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(make_openai_chat_request("SECRET"))
            .func(noop_exec_fn())
            .codec(request_codec)
            .build(),
    )
    .await
    .unwrap();

    let captured = captured_events_snapshot(&events);
    let start_event = captured
        .iter()
        .find(|e| is_scope_event(e, ScopeType::Llm, ScopeCategory::Start))
        .expect("expected LlmStart event");
    assert_eq!(
        start_event.input().unwrap()["content"]["messages"][0]["content"],
        json!("Sanitized")
    );
    let annotated = start_event
        .annotated_request()
        .expect("annotated_request should be decoded from sanitized input");
    assert_eq!(first_message_text(annotated), Some("Sanitized"));
    assert!(
        !serde_json::to_string(start_event)
            .unwrap()
            .contains("SECRET"),
        "start event should not retain raw request content"
    );

    deregister_subscriber("sanitized_req_codec_sub").unwrap();
    deregister_llm_sanitize_request_guardrail("sanitize_req_codec_annotation").unwrap();
}

#[tokio::test]
async fn test_stream_response_codec_populates_annotated_response() {
    let _lock = TEST_MUTEX.lock().unwrap();
    let _pricing_guard = ResetPricingResolverGuard;
    reset_global();
    setup_isolated_thread();
    install_mock_response_pricing();

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
            .name("openai")
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

    let captured = captured_events_snapshot(&events);
    let end_event = captured
        .iter()
        .find(|e| is_scope_event(e, ScopeType::Llm, ScopeCategory::End))
        .expect("expected LlmEnd event after stream drain");

    let ann = end_event
        .annotated_response()
        .expect("annotated_response should be Some on stream path when response codec is active");
    assert_eq!(ann.id, Some("mock-resp-id".into()));
    assert_eq!(ann.response_text(), Some("mock response text"));
    assert_eq!(
        ann.usage
            .as_ref()
            .and_then(|usage| usage.cost.as_ref())
            .and_then(|cost| cost.total),
        Some(0.000_435)
    );

    deregister_subscriber("stream_resp_codec_sub").unwrap();
}

#[tokio::test]
async fn managed_buffered_and_streaming_close_price_the_committed_route_not_response_alias() {
    let _lock = TEST_MUTEX.lock().unwrap();
    let _pricing_guard = ResetPricingResolverGuard;
    reset_global();
    setup_isolated_thread();
    install_routed_response_pricing();

    let events = Arc::new(Mutex::new(Vec::new()));
    let captured_events = events.clone();
    register_subscriber(
        "routed_alias_pricing_sub",
        Arc::new(move |event: &Event| captured_events.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("routed-buffered")
            .request(make_llm_request(json!({"model": "requested"})))
            .func(Arc::new(|_| {
                Box::pin(async {
                    assert!(record_llm_optimization_contribution(
                        routed_model_contribution()
                    ));
                    Ok(json!({"response": "done"}))
                })
            }))
            .response_codec(Arc::new(RoutedAliasResponseCodec {
                cost_source: CostSource::ModelPricing,
            }))
            .build(),
    )
    .await
    .unwrap();

    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("routed-streaming")
            .request(make_llm_request(json!({"model": "requested"})))
            .func(Arc::new(|_| {
                Box::pin(async {
                    assert!(record_llm_optimization_contribution(
                        routed_model_contribution()
                    ));
                    Ok(
                        Box::pin(tokio_stream::iter(vec![Ok(json!({"chunk": "done"}))]))
                            as LlmJsonStream,
                    )
                })
            }))
            .collector(Box::new(|_| Ok(())))
            .finalizer(Box::new(|| json!({"response": "done"})))
            .response_codec(Arc::new(RoutedAliasResponseCodec {
                cost_source: CostSource::ModelPricing,
            }))
            .build(),
    )
    .await
    .unwrap();
    while let Some(item) = stream.next().await {
        item.unwrap();
    }

    llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("routed-provider-reported")
            .request(make_llm_request(json!({"model": "requested"})))
            .func(Arc::new(|_| {
                Box::pin(async {
                    assert!(record_llm_optimization_contribution(
                        routed_model_contribution()
                    ));
                    Ok(json!({"response": "done"}))
                })
            }))
            .response_codec(Arc::new(RoutedAliasResponseCodec {
                cost_source: CostSource::ProviderReported,
            }))
            .build(),
    )
    .await
    .unwrap();

    let captured = captured_events_snapshot(&events);
    for name in ["routed-buffered", "routed-streaming"] {
        let response = captured
            .iter()
            .find(|event| {
                event.name() == name && is_scope_event(event, ScopeType::Llm, ScopeCategory::End)
            })
            .and_then(Event::annotated_response)
            .unwrap_or_else(|| panic!("missing annotated response for {name}"));
        let summary = response.optimization_summary.as_ref().unwrap();
        assert_eq!(summary.effective_model.as_ref().unwrap().model, "effective");
        assert_eq!(summary.actual_cost.as_ref().unwrap().total, Some(0.002));
        assert_eq!(
            summary
                .actual_cost
                .as_ref()
                .unwrap()
                .pricing_model
                .as_deref(),
            Some("effective")
        );
        assert_eq!(
            response
                .usage
                .as_ref()
                .and_then(|usage| usage.cost.as_ref())
                .and_then(|cost| cost.total),
            Some(0.002)
        );
    }

    let provider_response = captured
        .iter()
        .find(|event| {
            event.name() == "routed-provider-reported"
                && is_scope_event(event, ScopeType::Llm, ScopeCategory::End)
        })
        .and_then(Event::annotated_response)
        .expect("missing provider-reported annotated response");
    let provider_cost = provider_response
        .optimization_summary
        .as_ref()
        .and_then(|summary| summary.actual_cost.as_ref())
        .unwrap();
    assert_eq!(provider_cost.total, Some(0.42));
    assert_eq!(provider_cost.source, CostSource::ProviderReported);
    assert_eq!(provider_cost.pricing_source.as_deref(), Some("provider"));
    assert_eq!(
        provider_response
            .usage
            .as_ref()
            .and_then(|usage| usage.cost.as_ref())
            .map(|cost| cost.source.clone()),
        Some(CostSource::ProviderReported)
    );

    deregister_subscriber("routed_alias_pricing_sub").unwrap();
}

#[tokio::test]
async fn test_stream_response_codec_annotation_uses_sanitized_aggregated_response() {
    let _lock = TEST_MUTEX.lock().unwrap();
    reset_global();
    setup_isolated_thread();

    let events = Arc::new(Mutex::new(Vec::new()));
    let ec = events.clone();
    register_subscriber(
        "stream_sanitized_resp_codec_sub",
        Arc::new(move |e: &Event| {
            ec.lock().unwrap().push(e.clone());
        }),
    )
    .unwrap();
    register_llm_sanitize_response_guardrail(
        "stream_sanitize_resp_codec_annotation",
        1,
        Arc::new(|_response| make_openai_chat_response("Sanitized")),
    )
    .unwrap();

    let collector: Box<dyn FnMut(Json) -> Result<()> + Send> = Box::new(|_chunk| Ok(()));
    let finalizer: Box<dyn FnOnce() -> Json + Send> =
        Box::new(|| make_openai_chat_response("SECRET"));
    let response_codec: Arc<dyn LlmResponseCodec> = Arc::new(OpenAIChatCodec);

    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("openai")
            .request(make_openai_chat_request("stream me"))
            .func(noop_stream_exec_fn())
            .collector(collector)
            .finalizer(finalizer)
            .response_codec(response_codec)
            .build(),
    )
    .await
    .unwrap();

    while let Some(_chunk) = stream.next().await {}

    let captured = captured_events_snapshot(&events);
    let end_event = captured
        .iter()
        .find(|e| is_scope_event(e, ScopeType::Llm, ScopeCategory::End))
        .expect("expected LlmEnd event after stream drain");
    assert_eq!(
        end_event.output().unwrap()["choices"][0]["message"]["content"],
        json!("Sanitized")
    );
    let annotated = end_event
        .annotated_response()
        .expect("annotated_response should be decoded from sanitized stream output");
    assert_eq!(annotated.response_text(), Some("Sanitized"));
    assert!(
        !serde_json::to_string(end_event).unwrap().contains("SECRET"),
        "stream end event should not retain raw aggregated response content"
    );

    deregister_subscriber("stream_sanitized_resp_codec_sub").unwrap();
    deregister_llm_sanitize_response_guardrail("stream_sanitize_resp_codec_annotation").unwrap();
}
