// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for the adaptive plugin's `response_cache` feature
//! (exact-match).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

use nemo_relay::api::event::{Event, ScopeCategory};
use nemo_relay::api::llm::{
    LlmCallExecuteParams, LlmRequest, LlmStreamCallExecuteParams, llm_call_execute,
    llm_stream_call_execute,
};
use nemo_relay::api::runtime::{
    LlmExecutionNextFn, LlmJsonStream, LlmStreamExecutionNextFn, NemoRelayContextState,
    global_context,
};
use nemo_relay::api::scope::ScopeType;
use nemo_relay::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use nemo_relay::error::FlowError;
use nemo_relay::plugin::{
    PluginConfig, clear_plugin_configuration, initialize_plugins_exact, validate_plugin_config,
};
use nemo_relay_adaptive::plugin_component::{ComponentSpec, register_adaptive_component};
use nemo_relay_adaptive::{
    AcgComponentConfig, AdaptiveConfig, BackendSpec, ResponseCacheConfig, StateConfig,
};
use serde_json::{Value as Json, json};
use tokio::sync::Mutex;
use tokio_stream::StreamExt;

#[path = "response_cache_common.rs"]
mod response_cache_common;
use response_cache_common::{activate_cache, call, chat_request};

static TEST_MUTEX: Mutex<()> = Mutex::const_new(());

fn reset_global() {
    let _ = clear_plugin_configuration();
    let ctx = global_context();
    let mut state = ctx.write().unwrap();
    *state = NemoRelayContextState::new();
}

/// A provider stub that counts how many times it actually runs and returns a
/// canned body carrying token usage and cost.
fn counting_provider(calls: Arc<AtomicUsize>, body: Json) -> LlmExecutionNextFn {
    Arc::new(move |_req: LlmRequest| {
        let calls = Arc::clone(&calls);
        let body = body.clone();
        Box::pin(async move {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(body)
        })
    })
}

/// A provider stub that counts runs and always returns a transport-style error.
fn erroring_provider(calls: Arc<AtomicUsize>) -> LlmExecutionNextFn {
    Arc::new(move |_req: LlmRequest| {
        let calls = Arc::clone(&calls);
        Box::pin(async move {
            calls.fetch_add(1, Ordering::SeqCst);
            Err(FlowError::Internal("provider boom".to_string()))
        })
    })
}

fn sample_body() -> Json {
    json!({
        "id": "resp_abc",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "gpt-4o",
        "choices": [{"index": 0,
            "message": {"role": "assistant", "content": "The answer is 42."},
            "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1200, "completion_tokens": 80, "total_tokens": 1280, "cost_usd": 0.0123}
    })
}

/// A streaming provider stub that counts how many times it actually runs and
/// yields a fixed sequence of chunks.
fn counting_stream_provider(
    calls: Arc<AtomicUsize>,
    chunks: Vec<Json>,
) -> LlmStreamExecutionNextFn {
    Arc::new(move |_req: LlmRequest| {
        let calls = Arc::clone(&calls);
        let chunks = chunks.clone();
        Box::pin(async move {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(Box::pin(tokio_stream::iter(
                chunks.into_iter().map(Ok::<Json, FlowError>),
            )) as LlmJsonStream)
        })
    })
}

/// Runs one managed streaming call and drains it, returning the chunks the
/// caller observed (after the cache intercept and the managed pipeline).
async fn stream_call(provider: &LlmStreamExecutionNextFn, request: LlmRequest) -> Vec<Json> {
    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("openai")
            .request(request)
            .func(provider.clone())
            .collector(Box::new(|_chunk| Ok(())))
            .finalizer(Box::new(|| json!({"done": true})))
            .model_name("gpt-4o")
            .build(),
    )
    .await
    .unwrap();
    let mut collected = Vec::new();
    while let Some(item) = stream.next().await {
        collected.push(item.unwrap());
    }
    collected
}

#[tokio::test]
async fn exact_repeat_is_a_hit_that_skips_the_provider_and_returns_the_response_unchanged() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(ResponseCacheConfig::default()).await;

    let calls = Arc::new(AtomicUsize::new(0));
    let provider = counting_provider(Arc::clone(&calls), sample_body());

    let first = call(&provider, chat_request("What is the answer?")).await;
    let second = call(&provider, chat_request("What is the answer?")).await;

    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "provider should run once; the repeat must be served from cache"
    );

    // A reuse is shape-identical to a live call: the hit returns the stored
    // response UNCHANGED, including usage — never a stripped/different shape.
    assert_eq!(
        first, second,
        "a hit must return the stored response unchanged"
    );
    assert!(
        second.get("usage").is_some(),
        "a hit must preserve usage (savings are reported on the mark, not by mutating the body)"
    );
}

#[tokio::test]
async fn a_different_request_is_a_miss() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(ResponseCacheConfig::default()).await;

    let calls = Arc::new(AtomicUsize::new(0));
    let provider = counting_provider(Arc::clone(&calls), sample_body());

    call(&provider, chat_request("Question one?")).await;
    call(&provider, chat_request("A different question?")).await;

    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "distinct requests must each hit the provider"
    );
}

#[tokio::test]
async fn cosmetic_noise_still_hits_the_cache() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(ResponseCacheConfig::default()).await;

    let calls = Arc::new(AtomicUsize::new(0));
    let provider = counting_provider(Arc::clone(&calls), sample_body());

    // Same meaning, but with a streaming flag and a per-call user id that must
    // not affect the key, plus reordered top-level fields.
    let first = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false,
            "user": "user-A"
        }),
    };
    let second = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "user": "user-B",
            "stream": true,
            "messages": [{"role": "user", "content": "hello"}],
            "model": "gpt-4o"
        }),
    };

    call(&provider, first).await;
    call(&provider, second).await;

    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "field order, streaming flag, and per-call user id must not change the key"
    );
}

#[tokio::test]
async fn stateful_responses_calls_bypass_the_cache() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(ResponseCacheConfig::default()).await;

    let calls = Arc::new(AtomicUsize::new(0));
    let provider = counting_provider(Arc::clone(&calls), sample_body());

    let stateful = || LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "stateful"}],
            "store": true
        }),
    };

    call(&provider, stateful()).await;
    call(&provider, stateful()).await;

    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "stateful Responses calls (store=true) must never be cached"
    );
}

#[tokio::test]
async fn invalid_config_is_rejected_by_validation() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    register_adaptive_component().unwrap();

    let adaptive = AdaptiveConfig {
        response_cache: Some(ResponseCacheConfig {
            ttl_seconds: 0,
            bypass_rate: 2.0,
            ..ResponseCacheConfig::default()
        }),
        ..AdaptiveConfig::default()
    };
    let report = validate_plugin_config(&PluginConfig {
        components: vec![ComponentSpec::new(adaptive).into()],
        ..PluginConfig::default()
    });

    assert!(
        report
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "response_cache.invalid_ttl"),
        "ttl_seconds = 0 must produce a diagnostic"
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "response_cache.invalid_bypass_rate"),
        "bypass_rate out of range must produce a diagnostic"
    );
}

#[tokio::test]
async fn unknown_and_unavailable_backends_are_rejected_by_validation() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    register_adaptive_component().unwrap();

    let validate = |kind: &str| {
        let mut config = ResponseCacheConfig::default();
        config.backend.kind = kind.to_string();
        validate_plugin_config(&PluginConfig {
            components: vec![
                ComponentSpec::new(AdaptiveConfig {
                    response_cache: Some(config),
                    ..AdaptiveConfig::default()
                })
                .into(),
            ],
            ..PluginConfig::default()
        })
    };

    // An unrecognized backend kind is always rejected.
    let unknown = validate("sqlite");
    assert!(
        unknown
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "response_cache.unknown_backend"),
        "an unknown backend kind must be rejected: {:?}",
        unknown.diagnostics
    );

    // Without the redis-backend feature, selecting redis is rejected as unavailable.
    #[cfg(not(feature = "redis-backend"))]
    {
        let redis = validate("redis");
        assert!(
            redis
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "response_cache.backend_unavailable"),
            "redis without the redis-backend feature must be rejected: {:?}",
            redis.diagnostics
        );
    }
}

#[tokio::test]
async fn hit_preserves_usage_on_the_end_event_and_reports_savings_on_the_mark() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(ResponseCacheConfig::default()).await;

    // Capture every event so we can inspect the LLM end events + cache marks.
    let captured = Arc::new(StdMutex::new(Vec::<Event>::new()));
    let sink = Arc::clone(&captured);
    register_subscriber(
        "response_cache_event_capture",
        Arc::new(move |event: &Event| {
            sink.lock().unwrap().push(event.clone());
        }),
    )
    .unwrap();

    let calls = Arc::new(AtomicUsize::new(0));
    let provider = counting_provider(Arc::clone(&calls), sample_body());

    call(&provider, chat_request("end-event check")).await; // miss
    call(&provider, chat_request("end-event check")).await; // hit
    flush_subscribers().unwrap();

    let events = captured.lock().unwrap();
    let end_events: Vec<&Event> = events
        .iter()
        .filter(|event| {
            event.scope_type() == Some(ScopeType::Llm)
                && event.scope_category() == Some(ScopeCategory::End)
        })
        .collect();
    assert_eq!(end_events.len(), 2, "expected one LLM end event per call");

    // Transparency: a hit is shape-identical to a live call, so BOTH end events
    // carry usage. The cache never mutates the response to fake $0.
    for end_event in &end_events {
        let output = end_event.output().expect("llm end event has output");
        assert!(
            output.get("usage").is_some(),
            "both miss and hit end events must carry usage (no stripping)"
        );
    }

    // Savings ride on the `response_cache` hit mark instead.
    let hit_mark = events
        .iter()
        .find(|event| {
            event.name() == "response_cache"
                && event
                    .data()
                    .and_then(|data| data.get("status"))
                    .and_then(Json::as_str)
                    == Some("hit")
        })
        .expect("a response_cache hit mark should be emitted");
    let saved_tokens = hit_mark
        .metadata()
        .and_then(|metadata| metadata.get("nemo_relay.response_cache.saved_tokens"))
        .and_then(Json::as_u64);
    assert_eq!(
        saved_tokens,
        Some(1280),
        "the hit mark must report the avoided token count"
    );
    let ttl_ms = hit_mark
        .metadata()
        .and_then(|metadata| metadata.get("nemo_relay.response_cache.ttl_ms"))
        .and_then(Json::as_u64);
    assert!(
        ttl_ms.is_some_and(|ms| ms > 0),
        "the hit mark must carry the configured ttl_ms"
    );

    drop(events);
    deregister_subscriber("response_cache_event_capture").unwrap();
}

#[tokio::test]
async fn errors_are_not_cached() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(ResponseCacheConfig::default()).await;

    let calls = Arc::new(AtomicUsize::new(0));
    let error_body = json!({"error": {"message": "rate limited", "type": "rate_limit"}});
    let provider = counting_provider(Arc::clone(&calls), error_body);

    call(&provider, chat_request("will error")).await;
    call(&provider, chat_request("will error")).await;

    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "error-shaped responses must never be cached"
    );
}

#[tokio::test]
async fn error_null_success_bodies_are_cached() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(ResponseCacheConfig::default()).await;

    // Real OpenAI-Responses success bodies carry `"error": null`; a key-presence
    // check would silently disable caching for that whole surface.
    let calls = Arc::new(AtomicUsize::new(0));
    let body = json!({
        "id": "resp_1", "object": "response", "status": "completed", "error": null,
        "model": "gpt-4o",
        "output": [{"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello"}]}],
        "usage": {"input_tokens": 9, "output_tokens": 3, "total_tokens": 12}
    });
    let provider = counting_provider(Arc::clone(&calls), body);

    call(&provider, chat_request("responses success")).await;
    call(&provider, chat_request("responses success")).await;

    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "a success body with error: null must be cached and hit on repeat"
    );
}

#[tokio::test]
async fn non_final_status_bodies_are_not_cached() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(ResponseCacheConfig::default()).await;

    // An incomplete/failed body is not a complete, replayable answer.
    let calls = Arc::new(AtomicUsize::new(0));
    let body = json!({
        "id": "resp_1", "object": "response", "status": "incomplete", "error": null,
        "model": "gpt-4o",
        "output": [{"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "partial"}]}]
    });
    let provider = counting_provider(Arc::clone(&calls), body);

    call(&provider, chat_request("incomplete run")).await;
    call(&provider, chat_request("incomplete run")).await;

    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "a non-final status body must never be stored"
    );
}

#[tokio::test]
async fn miss_mark_is_emitted_even_when_the_provider_errors() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(ResponseCacheConfig::default()).await;

    let captured = Arc::new(StdMutex::new(Vec::<Event>::new()));
    let sink = Arc::clone(&captured);
    register_subscriber(
        "response_cache_error_capture",
        Arc::new(move |event: &Event| sink.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let calls = Arc::new(AtomicUsize::new(0));
    let provider = erroring_provider(Arc::clone(&calls));

    // The managed call surfaces the provider error to the caller...
    let result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("openai")
            .request(chat_request("will error"))
            .func(provider.clone())
            .model_name("gpt-4o")
            .build(),
    )
    .await;
    assert!(result.is_err(), "the provider error must surface");
    flush_subscribers().unwrap();

    // ...but the cache still recorded the miss decision (mark emitted before the call).
    let events = captured.lock().unwrap();
    let miss = events.iter().any(|event| {
        event.name() == "response_cache"
            && event
                .data()
                .and_then(|data| data.get("status"))
                .and_then(Json::as_str)
                == Some("miss")
    });
    assert!(
        miss,
        "a miss mark must be emitted even when the provider errors"
    );
    drop(events);
    deregister_subscriber("response_cache_error_capture").unwrap();
}

/// The text a strict streaming client would accumulate from replayed
/// provider-native chunks (OpenAI chat `delta.content` fragments and Anthropic
/// `text_delta` fragments).
fn replayed_text(chunks: &[Json]) -> String {
    chunks
        .iter()
        .filter_map(|chunk| {
            chunk
                .pointer("/choices/0/delta/content")
                .and_then(Json::as_str)
                .or_else(|| chunk.pointer("/delta/text").and_then(Json::as_str))
        })
        .collect()
}

/// OpenAI-chat streaming deltas the codec can assemble into one response.
fn openai_chat_stream_chunks() -> Vec<Json> {
    vec![
        json!({"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1_700_000_000, "model": "gpt-4o", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]}),
        json!({"id": "chatcmpl-1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "The answer "}, "finish_reason": null}]}),
        json!({"id": "chatcmpl-1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "is 42."}, "finish_reason": null}]}),
        json!({"id": "chatcmpl-1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}}),
    ]
}

#[tokio::test]
async fn streaming_repeat_is_a_hit_that_skips_the_provider_and_replays_the_aggregate() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    // Streaming aggregation infers its codec from the request surface.
    activate_cache(ResponseCacheConfig::default()).await;

    let calls = Arc::new(AtomicUsize::new(0));
    let chunks = openai_chat_stream_chunks();
    let provider = counting_stream_provider(Arc::clone(&calls), chunks.clone());

    let first = stream_call(&provider, chat_request("stream me")).await;
    let second = stream_call(&provider, chat_request("stream me")).await;

    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "provider should stream once; the repeat must be served from cache"
    );
    assert_eq!(
        first, chunks,
        "a miss must pass the live chunks through unchanged"
    );
    // The hit replays PROVIDER-NATIVE chunks a strict streaming client can parse
    // (not one aggregate-shaped frame), and they carry the stored answer + usage.
    assert!(
        second.len() > 1,
        "a hit must replay a native chunk sequence, got {second:?}"
    );
    assert_eq!(second[0]["object"], json!("chat.completion.chunk"));
    assert_eq!(replayed_text(&second), "The answer is 42.");
    let usage_chunk = second
        .iter()
        .rev()
        .find(|chunk| chunk.get("usage").is_some())
        .expect("the replay must carry a usage chunk");
    assert_eq!(usage_chunk["usage"]["total_tokens"], json!(14));
}

#[tokio::test]
async fn streaming_unrecognized_shape_without_a_codec_runs_live() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    // No codec, and a body the surface detector does not recognize (no
    // messages/system/input) -> nothing to aggregate -> each call runs live.
    activate_cache(ResponseCacheConfig::default()).await;

    let calls = Arc::new(AtomicUsize::new(0));
    let provider = counting_stream_provider(Arc::clone(&calls), openai_chat_stream_chunks());
    let unrecognized = || LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"model": "mystery", "prompt": "hi"}),
    };

    stream_call(&provider, unrecognized()).await;
    stream_call(&provider, unrecognized()).await;

    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "an unrecognized-shape stream with no codec must still bypass and run live"
    );
}

/// A minimal Anthropic Messages SSE sequence the anthropic codec assembles into
/// `content: [{type:text, text:"Hello, world."}]`.
fn anthropic_stream_chunks() -> Vec<Json> {
    vec![
        json!({"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-4-5", "content": [], "usage": {"input_tokens": 10, "output_tokens": 0}}}),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello, "}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "world."}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"input_tokens": 10, "output_tokens": 5}}),
        json!({"type": "message_stop"}),
    ]
}

/// Like [`stream_call`], but with a caller-chosen provider name (the value the
/// gateway derives from the route, e.g. `"anthropic.messages"`).
async fn stream_call_named(
    name: &str,
    provider: &LlmStreamExecutionNextFn,
    request: LlmRequest,
) -> Vec<Json> {
    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name(name.to_string())
            .request(request)
            .func(provider.clone())
            .collector(Box::new(|_chunk| Ok(())))
            .finalizer(Box::new(|| json!({"done": true})))
            .model_name("claude-haiku-4-5")
            .build(),
    )
    .await
    .unwrap();
    let mut collected = Vec::new();
    while let Some(item) = stream.next().await {
        collected.push(item.unwrap());
    }
    collected
}

#[tokio::test]
async fn no_codec_mis_inferred_empty_aggregate_is_not_cached() {
    // The correctness guard: a chat-shaped request infers the openai_chat codec,
    // but the provider streams NON-chat chunks (the misdetection case — e.g. a
    // system-less Anthropic body read as chat). The chat collector drops them and
    // finalizes an empty aggregate; the guard must refuse to cache it, so the
    // repeat runs live instead of serving a wrong empty response.
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(ResponseCacheConfig::default()).await;

    let calls = Arc::new(AtomicUsize::new(0));
    let foreign_chunks = vec![
        json!({"type": "content_block_delta", "delta": {"text": "The answer "}}),
        json!({"type": "content_block_delta", "delta": {"text": "is 42."}}),
    ];
    let provider = counting_stream_provider(Arc::clone(&calls), foreign_chunks);

    stream_call(&provider, chat_request("mis-infer")).await; // empty aggregate -> NOT stored
    stream_call(&provider, chat_request("mis-infer")).await; // must run live again

    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "a mis-inferred empty aggregate must not be cached; the repeat runs live"
    );
}

/// An Anthropic Messages SSE sequence that dies mid-answer with an in-band
/// `error` event (never a stream-level `Err`).
fn anthropic_stream_chunks_with_inband_error() -> Vec<Json> {
    vec![
        json!({"type": "message_start", "message": {"id": "msg_x", "type": "message",
            "role": "assistant", "content": [], "usage": {"input_tokens": 9, "output_tokens": 0}}}),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "partial ans"}}),
        json!({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}),
    ]
}

#[tokio::test]
async fn streaming_inband_error_is_not_cached() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(ResponseCacheConfig::default()).await;

    let calls = Arc::new(AtomicUsize::new(0));
    let provider = counting_stream_provider(
        Arc::clone(&calls),
        anthropic_stream_chunks_with_inband_error(),
    );

    // A stream that carries an in-band `error` event must NOT have its partial
    // aggregate cached: a later identical request would otherwise replay a
    // truncated answer as if it were complete.
    stream_call_named(
        "anthropic.messages",
        &provider,
        chat_request("erroring stream"),
    )
    .await;
    stream_call_named(
        "anthropic.messages",
        &provider,
        chat_request("erroring stream"),
    )
    .await;

    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "an erroring stream must not populate the cache; the repeat runs live"
    );
}

#[tokio::test]
async fn thinking_streams_are_not_cached() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(ResponseCacheConfig::default()).await;

    // The anthropic collector drops thinking/signature deltas, so a stored
    // thinking aggregate would replay a gutted, unsigned block.
    let chunks = vec![
        json!({"type": "message_start", "message": {"id": "msg_t", "type": "message",
            "role": "assistant", "content": [], "usage": {"input_tokens": 9, "output_tokens": 0}}}),
        json!({"type": "content_block_start", "index": 0,
            "content_block": {"type": "thinking", "thinking": ""}}),
        json!({"type": "content_block_delta", "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "step by step"}}),
        json!({"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}),
        json!({"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "answer"}}),
        json!({"type": "message_stop"}),
    ];
    let calls = Arc::new(AtomicUsize::new(0));
    let provider = counting_stream_provider(Arc::clone(&calls), chunks);

    stream_call_named("anthropic.messages", &provider, chat_request("think hard")).await;
    stream_call_named("anthropic.messages", &provider, chat_request("think hard")).await;

    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "a thinking stream must not be stored (its replay would be gutted)"
    );
}

#[tokio::test]
async fn refusal_only_streams_are_not_cached() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(ResponseCacheConfig::default()).await;

    // The chat collector has no refusal handling: a refusal-only stream
    // aggregates to a choice with no content and no tool calls, and a replay
    // would look empty where the live stream carried the refusal text.
    let chunks = vec![
        json!({"id": "chatcmpl-r", "object": "chat.completion.chunk", "created": 1_700_000_000,
            "model": "gpt-4o", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]}),
        json!({"id": "chatcmpl-r", "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"refusal": "I cannot help with that."}, "finish_reason": null}]}),
        json!({"id": "chatcmpl-r", "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
    ];
    let calls = Arc::new(AtomicUsize::new(0));
    let provider = counting_stream_provider(Arc::clone(&calls), chunks);

    stream_call(&provider, chat_request("refused")).await;
    stream_call(&provider, chat_request("refused")).await;

    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "a refusal-only stream must not be stored (its replay would be empty)"
    );
}

#[tokio::test]
async fn buffered_refusal_entry_is_not_replayed_to_a_streaming_caller() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(ResponseCacheConfig::default()).await;

    // The chunk synthesizer has no refusal delta, so replaying this buffered
    // body to a streaming caller would carry no content at all.
    let buffered_calls = Arc::new(AtomicUsize::new(0));
    let refusal_body = json!({
        "id": "chatcmpl-r", "object": "chat.completion", "created": 1_700_000_000,
        "model": "gpt-4o",
        "choices": [{"index": 0,
            "message": {"role": "assistant", "content": null,
                "refusal": "I cannot help with that."},
            "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}
    });
    let buffered_provider = counting_provider(Arc::clone(&buffered_calls), refusal_body);
    call(&buffered_provider, chat_request("refused, then streamed")).await;

    let stream_calls = Arc::new(AtomicUsize::new(0));
    let live_chunks = vec![
        json!({"id": "chatcmpl-r", "object": "chat.completion.chunk", "created": 1_700_000_000,
            "model": "gpt-4o", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]}),
        json!({"id": "chatcmpl-r", "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"refusal": "I cannot help with that."}, "finish_reason": null}]}),
        json!({"id": "chatcmpl-r", "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
    ];
    let stream_provider = counting_stream_provider(Arc::clone(&stream_calls), live_chunks);
    let streamed = stream_call(&stream_provider, chat_request("refused, then streamed")).await;

    assert_eq!(
        stream_calls.load(Ordering::SeqCst),
        1,
        "a stored body whose replay would drop the refusal must run the stream live"
    );
    assert!(
        streamed
            .iter()
            .any(|chunk| chunk.pointer("/choices/0/delta/refusal").is_some()),
        "the streaming caller must receive the refusal, not an empty replay: {streamed:?}"
    );

    // The entry itself stays in place: a buffered repeat is still a hit.
    call(&buffered_provider, chat_request("refused, then streamed")).await;
    assert_eq!(
        buffered_calls.load(Ordering::SeqCst),
        1,
        "the entry must keep serving buffered callers"
    );
}

#[tokio::test]
async fn misconfigured_keys_and_headers_are_rejected_by_validation() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    register_adaptive_component().unwrap();

    let validate = |config: ResponseCacheConfig| {
        let adaptive = AdaptiveConfig {
            response_cache: Some(config),
            ..AdaptiveConfig::default()
        };
        validate_plugin_config(&PluginConfig {
            components: vec![ComponentSpec::new(adaptive).into()],
            ..PluginConfig::default()
        })
    };

    let report = validate(ResponseCacheConfig {
        skip_keys: vec![
            "messages".to_string(),
            "system".to_string(),
            "prompt".to_string(),
        ],
        ..ResponseCacheConfig::default()
    });
    assert_eq!(
        report
            .diagnostics
            .iter()
            .filter(|d| d.code == "response_cache.reserved_skip_key")
            .count(),
        3,
        "dropping an answer-determining field must be rejected: {:?}",
        report.diagnostics
    );

    let report = validate(ResponseCacheConfig {
        header_allowlist: vec!["Authorization".to_string()],
        ..ResponseCacheConfig::default()
    });
    assert!(
        report
            .diagnostics
            .iter()
            .any(|d| d.code == "response_cache.auth_header_allowlisted"),
        "auth headers must never be allowlisted into keys: {:?}",
        report.diagnostics
    );

    let mut zeroed = ResponseCacheConfig::default();
    zeroed
        .backend
        .config
        .insert("max_bytes".to_string(), json!(0));
    let report = validate(zeroed);
    assert!(
        report
            .diagnostics
            .iter()
            .any(|d| d.code == "response_cache.invalid_backend_option"),
        "max_bytes: 0 silently disables the cache and must be rejected: {:?}",
        report.diagnostics
    );

    let mut mistyped = ResponseCacheConfig::default();
    mistyped
        .backend
        .config
        .insert("max_bytes".to_string(), json!("10"));
    let report = validate(mistyped);
    assert!(
        report
            .diagnostics
            .iter()
            .any(|d| d.code == "response_cache.invalid_backend_option"),
        "a wrong-typed budget silently vanishes and must be rejected: {:?}",
        report.diagnostics
    );
}

#[tokio::test]
async fn truncated_stream_without_a_terminal_event_is_not_cached() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(ResponseCacheConfig::default()).await;

    // A clean end-of-file after N deltas (proxy idle-timeout, upstream abort
    // without a finish event) is a truncated answer: every collector finalizes
    // it as a well-formed partial, so only the missing terminal event reveals
    // the truncation. It must not be stored and replayed as complete.
    let truncated: Vec<Json> = openai_chat_stream_chunks()
        .into_iter()
        .filter(|chunk| chunk["choices"][0]["finish_reason"].is_null())
        .collect();
    let calls = Arc::new(AtomicUsize::new(0));
    let provider = counting_stream_provider(Arc::clone(&calls), truncated);

    stream_call(&provider, chat_request("cut off")).await;
    stream_call(&provider, chat_request("cut off")).await;

    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "a stream with no terminal event must not populate the cache"
    );
}

#[tokio::test]
async fn no_codec_system_less_anthropic_uses_provider_hint_and_caches() {
    // A system-less Anthropic body is shape-ambiguous (reads as OpenAI Chat by
    // shape alone). The provider-name hint disambiguates it, so the correct codec
    // assembles a non-empty aggregate and the repeat is a hit.
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(ResponseCacheConfig::default()).await;

    let calls = Arc::new(AtomicUsize::new(0));
    let provider = counting_stream_provider(Arc::clone(&calls), anthropic_stream_chunks());
    let body = || LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "hi"}]}),
    };

    stream_call_named("anthropic.messages", &provider, body()).await; // miss -> stores
    let second = stream_call_named("anthropic.messages", &provider, body()).await;

    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "the anthropic provider hint must let the repeat hit the cache"
    );
    assert_eq!(
        second[0]["type"],
        json!("message_start"),
        "an anthropic replay must open with a native message_start event"
    );
    assert_eq!(
        replayed_text(&second),
        "Hello, world.",
        "the replayed native chunks carry the assembled anthropic answer"
    );
}

#[tokio::test]
async fn no_codec_buffered_and_streaming_share_one_store_entry() {
    // Buffered and streaming derive the same key from the auto-detected decode,
    // so an entry stored by one path is served to the other. Assert cross-replay
    // is a hit (not a re-run) in BOTH directions.
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(ResponseCacheConfig::default()).await;

    // Direction A: streaming miss stores the inferred-codec aggregate; a buffered
    // repeat of the same request is served from it (buffered provider skipped).
    let s_calls = Arc::new(AtomicUsize::new(0));
    let s_provider = counting_stream_provider(Arc::clone(&s_calls), openai_chat_stream_chunks());
    stream_call(&s_provider, chat_request("shared-A")).await;
    let b_calls = Arc::new(AtomicUsize::new(0));
    let b_provider = counting_provider(Arc::clone(&b_calls), sample_body());
    let buffered_hit = call(&b_provider, chat_request("shared-A")).await;
    assert_eq!(
        b_calls.load(Ordering::SeqCst),
        0,
        "a buffered repeat must hit the streamed entry, not run the provider"
    );
    assert_eq!(
        buffered_hit["choices"][0]["message"]["content"],
        json!("The answer is 42.")
    );

    // Direction B: buffered miss stores the raw response; a streaming repeat of a
    // (different) request is served from it, replayed as native chunks.
    let b2 = Arc::new(AtomicUsize::new(0));
    let b2_provider = counting_provider(Arc::clone(&b2), sample_body());
    call(&b2_provider, chat_request("shared-B")).await;
    let s2 = Arc::new(AtomicUsize::new(0));
    let s2_provider = counting_stream_provider(Arc::clone(&s2), openai_chat_stream_chunks());
    let streamed = stream_call(&s2_provider, chat_request("shared-B")).await;
    assert_eq!(
        s2.load(Ordering::SeqCst),
        0,
        "a streaming repeat must hit the buffered entry, not run the provider"
    );
    assert!(
        streamed.len() > 1,
        "the buffered entry is replayed as native streaming chunks"
    );
    assert_eq!(replayed_text(&streamed), "The answer is 42.");
}

#[tokio::test]
async fn cache_coexists_with_acg_execution_intercept() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    register_adaptive_component().unwrap();

    // Configure BOTH the ACG execution intercept (needs a state backend) and the
    // response cache on one adaptive component. The cache takes a lower priority
    // so it sits outermost and keys on the original (pre-ACG) request.
    let adaptive = AdaptiveConfig {
        state: Some(StateConfig {
            backend: BackendSpec::in_memory(),
        }),
        acg: Some(AcgComponentConfig::default()),
        response_cache: Some(ResponseCacheConfig {
            priority: 40,
            ..ResponseCacheConfig::default()
        }),
        ..AdaptiveConfig::default()
    };
    let report = initialize_plugins_exact(PluginConfig {
        components: vec![ComponentSpec::new(adaptive).into()],
        ..PluginConfig::default()
    })
    .await
    .unwrap();
    assert!(
        report.diagnostics.is_empty(),
        "unexpected diagnostics: {:?}",
        report.diagnostics
    );

    let calls = Arc::new(AtomicUsize::new(0));
    let provider = counting_provider(Arc::clone(&calls), sample_body());
    call(&provider, chat_request("co-configured")).await;
    call(&provider, chat_request("co-configured")).await;

    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "the cache must still hit when co-configured with the ACG execution intercept"
    );
}

/// Shared-cache test: a second, independent store instance (a stand-in for
/// another process / teammate) sees entries written by the first. Requires a
/// local Redis and `NEMO_RELAY_RUN_REDIS_TESTS=1`; otherwise it skips. Compile
/// with `--features redis-backend`.
#[cfg(feature = "redis-backend")]
#[tokio::test]
async fn redis_backend_shares_entries_across_store_instances() {
    use std::time::Duration;

    use nemo_relay_adaptive::response_cache::store::{CacheEntry, CacheStore, RedisCacheStore};

    let enabled = std::env::var("NEMO_RELAY_RUN_REDIS_TESTS")
        .ok()
        .is_some_and(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        });
    if !enabled {
        eprintln!(
            "skipping redis shared-cache test: set NEMO_RELAY_RUN_REDIS_TESTS=1 with a local Redis"
        );
        return;
    }

    let url = "redis://127.0.0.1:6379";
    let prefix = "nemo-relay:test-response-cache:";
    let writer = RedisCacheStore::new(url, prefix)
        .await
        .expect("connect redis (writer)");
    let reader = RedisCacheStore::new(url, prefix)
        .await
        .expect("connect redis (reader)");

    let key = "sha256:shared-cache-test-key";
    let _ = writer.delete(key).await;

    let entry = CacheEntry::new(
        json!({"answer": "shared"}),
        Duration::from_secs(60),
        key.to_string(),
        Some("gpt-4o".to_string()),
        Some("openai".to_string()),
    );
    writer
        .set(key, entry, Duration::from_secs(60))
        .await
        .expect("set");

    // The independent reader instance sees the writer's entry — team sharing.
    let got = reader.get(key).await.expect("get");
    assert!(
        got.is_some(),
        "a second store instance should see the shared entry"
    );
    assert_eq!(got.unwrap().response["answer"], json!("shared"));

    writer.delete(key).await.expect("delete");
    assert!(reader.get(key).await.expect("get").is_none());
}
