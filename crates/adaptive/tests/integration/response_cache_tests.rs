// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for the adaptive plugin's `response_cache` feature
//! (exact-match).

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::task::{Context, Poll};
use std::time::Duration;

use nemo_relay::api::event::{Event, ScopeCategory};
use nemo_relay::api::llm::{
    LlmCallExecuteParams, LlmRequest, LlmStreamCallExecuteParams, llm_call_execute,
    llm_stream_call_execute,
};
use nemo_relay::api::registry::{
    deregister_tool_execution_intercept, register_tool_execution_intercept,
};
use nemo_relay::api::runtime::{
    LlmExecutionNextFn, LlmJsonStream, LlmStreamExecutionNextFn, LlmStreamInner,
    NemoRelayContextState, ToolExecutionNextFn, global_context,
};
use nemo_relay::api::scope::ScopeType;
use nemo_relay::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use nemo_relay::api::tool::{ToolCallExecuteParams, tool_call_execute};
use nemo_relay::error::FlowError;
use nemo_relay::plugin::{
    DiagnosticLevel, PluginConfig, clear_plugin_configuration, initialize_plugins_exact,
    validate_plugin_config,
};
use nemo_relay_adaptive::plugin_component::{ComponentSpec, register_adaptive_component};
use nemo_relay_adaptive::{
    AcgComponentConfig, AdaptiveConfig, BackendSpec, ResponseCacheConfig, ResponseCacheKeyStrategy,
    StateConfig, ToolCacheConfig, ToolClass, ToolOverride,
};
use serde_json::{Value as Json, json};
use tokio::sync::Mutex;
use tokio_stream::{Stream, StreamExt};

#[path = "response_cache_common.rs"]
mod response_cache_common;
use response_cache_common::{activate_cache, call, chat_request};

static TEST_MUTEX: Mutex<()> = Mutex::const_new(());
const ROUTING_BACKEND_HEADER: &str = "x-nemo-relay-internal-dispatch-backend";

fn reset_global() {
    let _ = clear_plugin_configuration();
    let ctx = global_context();
    let mut state = ctx.write().unwrap();
    *state = NemoRelayContextState::new();
}

fn scoped_cache_config() -> ResponseCacheConfig {
    ResponseCacheConfig {
        namespace: "response-cache-integration-test".to_string(),
        ..ResponseCacheConfig::default()
    }
}

fn chat_request_with_token_cap(prompt: &str, token_cap: &str) -> LlmRequest {
    let mut request = chat_request(prompt);
    request
        .content
        .as_object_mut()
        .expect("chat request body must be an object")
        .insert(token_cap.to_string(), json!(64));
    request
}

fn chat_request_with_service_tier(prompt: &str, service_tier: &str) -> LlmRequest {
    let mut request = chat_request(prompt);
    request
        .content
        .as_object_mut()
        .expect("chat request body must be an object")
        .insert("service_tier".to_string(), json!(service_tier));
    request
}

fn chat_request_for_backend(prompt: &str, backend: &str) -> LlmRequest {
    let mut request = chat_request(prompt);
    request
        .headers
        .insert(ROUTING_BACKEND_HEADER.to_string(), json!(backend));
    request
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
            Ok(LlmJsonStream::new(tokio_stream::iter(
                chunks.into_iter().map(Ok::<Json, FlowError>),
            )))
        })
    })
}

struct CloseTrackingStream {
    stream: Pin<Box<dyn Stream<Item = Result<Json, FlowError>> + Send>>,
    close_calls: Arc<AtomicUsize>,
    close_error: Option<FlowError>,
    closed: bool,
}

impl Stream for CloseTrackingStream {
    type Item = Result<Json, FlowError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.closed {
            Poll::Ready(None)
        } else {
            self.stream.as_mut().poll_next(cx)
        }
    }
}

impl LlmStreamInner for CloseTrackingStream {
    fn close(
        mut self: Pin<&mut Self>,
    ) -> Pin<Box<dyn Future<Output = Result<(), FlowError>> + Send + '_>> {
        self.closed = true;
        self.close_calls.fetch_add(1, Ordering::SeqCst);
        let close_error = self.close_error.clone();
        Box::pin(async move { close_error.map_or(Ok(()), Err) })
    }
}

fn close_tracking_stream_provider(
    calls: Arc<AtomicUsize>,
    close_calls: Arc<AtomicUsize>,
    chunks: Vec<Json>,
    close_error: Option<FlowError>,
) -> LlmStreamExecutionNextFn {
    Arc::new(move |_req: LlmRequest| {
        let calls = Arc::clone(&calls);
        let close_calls = Arc::clone(&close_calls);
        let chunks = chunks.clone();
        let close_error = close_error.clone();
        Box::pin(async move {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(LlmJsonStream::from_closeable(CloseTrackingStream {
                stream: Box::pin(tokio_stream::iter(
                    chunks.into_iter().map(Ok::<Json, FlowError>),
                )),
                close_calls,
                close_error,
                closed: false,
            }))
        })
    })
}

async fn start_stream(provider: &LlmStreamExecutionNextFn, request: LlmRequest) -> LlmJsonStream {
    llm_stream_call_execute(
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
    .unwrap()
}

/// Runs one managed streaming call and drains it, returning the chunks the
/// caller observed (after the cache intercept and the managed pipeline).
async fn stream_call(provider: &LlmStreamExecutionNextFn, request: LlmRequest) -> Vec<Json> {
    let mut stream = start_stream(provider, request).await;
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
    activate_cache(scoped_cache_config()).await;

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
    activate_cache(scoped_cache_config()).await;

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
async fn routing_backends_use_independent_buffered_entries() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(scoped_cache_config()).await;

    let a_calls = Arc::new(AtomicUsize::new(0));
    let a_provider = counting_provider(Arc::clone(&a_calls), json!({"source": "backend-a"}));
    let b_calls = Arc::new(AtomicUsize::new(0));
    let b_provider = counting_provider(Arc::clone(&b_calls), json!({"source": "backend-b"}));

    let a = call(
        &a_provider,
        chat_request_for_backend("same routed prompt", "backend-a"),
    )
    .await;
    let b = call(
        &b_provider,
        chat_request_for_backend("same routed prompt", "backend-b"),
    )
    .await;
    let a_repeat = call(
        &a_provider,
        chat_request_for_backend("same routed prompt", "backend-a"),
    )
    .await;

    assert_eq!(a, json!({"source": "backend-a"}));
    assert_eq!(b, json!({"source": "backend-b"}));
    assert_eq!(a_repeat, a);
    assert_eq!(a_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        b_calls.load(Ordering::SeqCst),
        1,
        "a different selected backend must miss even with an identical body"
    );
}

#[tokio::test]
async fn buffered_openai_token_cap_spellings_do_not_share_entries() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(scoped_cache_config()).await;

    let legacy_calls = Arc::new(AtomicUsize::new(0));
    let legacy_provider =
        counting_provider(Arc::clone(&legacy_calls), json!({"source": "max_tokens"}));
    let current_calls = Arc::new(AtomicUsize::new(0));
    let current_provider = counting_provider(
        Arc::clone(&current_calls),
        json!({"source": "max_completion_tokens"}),
    );

    let legacy = call(
        &legacy_provider,
        chat_request_with_token_cap("same prompt", "max_tokens"),
    )
    .await;
    let legacy_repeat = call(
        &legacy_provider,
        chat_request_with_token_cap("same prompt", "max_tokens"),
    )
    .await;
    let current = call(
        &current_provider,
        chat_request_with_token_cap("same prompt", "max_completion_tokens"),
    )
    .await;

    assert_eq!(legacy, json!({"source": "max_tokens"}));
    assert_eq!(legacy_repeat, legacy);
    assert_eq!(current, json!({"source": "max_completion_tokens"}));
    assert_eq!(
        legacy_calls.load(Ordering::SeqCst),
        1,
        "the same spelling must remain cacheable"
    );
    assert_eq!(
        current_calls.load(Ordering::SeqCst),
        1,
        "the current spelling must miss rather than reuse the legacy request"
    );
}

#[tokio::test]
async fn buffered_service_tiers_do_not_share_entries() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(scoped_cache_config()).await;

    let default_calls = Arc::new(AtomicUsize::new(0));
    let default_provider = counting_provider(
        Arc::clone(&default_calls),
        json!({"source": "default", "service_tier": "default"}),
    );
    let priority_calls = Arc::new(AtomicUsize::new(0));
    let priority_provider = counting_provider(
        Arc::clone(&priority_calls),
        json!({"source": "priority", "service_tier": "priority"}),
    );

    let default = call(
        &default_provider,
        chat_request_with_service_tier("same prompt", "default"),
    )
    .await;
    let default_repeat = call(
        &default_provider,
        chat_request_with_service_tier("same prompt", "default"),
    )
    .await;
    let priority = call(
        &priority_provider,
        chat_request_with_service_tier("same prompt", "priority"),
    )
    .await;

    assert_eq!(
        default,
        json!({"source": "default", "service_tier": "default"})
    );
    assert_eq!(default_repeat, default);
    assert_eq!(
        priority,
        json!({"source": "priority", "service_tier": "priority"})
    );
    assert_eq!(
        default_calls.load(Ordering::SeqCst),
        1,
        "the same service tier must remain cacheable"
    );
    assert_eq!(
        priority_calls.load(Ordering::SeqCst),
        1,
        "a different service tier must miss rather than reuse the cached response"
    );
}

#[tokio::test]
async fn cosmetic_noise_still_hits_the_cache() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(scoped_cache_config()).await;

    let calls = Arc::new(AtomicUsize::new(0));
    let provider = counting_provider(Arc::clone(&calls), sample_body());

    // Same meaning, but with a streaming flag and a per-call user id that must
    // not affect the key, plus reordered top-level fields.
    let first = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.0,
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
            "model": "gpt-4o",
            "temperature": 0.0
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
    activate_cache(scoped_cache_config()).await;

    let calls = Arc::new(AtomicUsize::new(0));
    let provider = counting_provider(Arc::clone(&calls), sample_body());

    let stateful = || LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "stateful"}],
            "temperature": 0.0,
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
async fn nondeterministic_calls_emit_bypass_marks_and_run_live() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(ResponseCacheConfig {
        namespace: "determinism-test".to_string(),
        cache_nondeterministic: false,
        ..ResponseCacheConfig::default()
    })
    .await;

    let captured = Arc::new(StdMutex::new(Vec::<Event>::new()));
    let sink = Arc::clone(&captured);
    register_subscriber(
        "response_cache_nondeterministic_bypass_capture",
        Arc::new(move |event: &Event| sink.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let calls = Arc::new(AtomicUsize::new(0));
    let provider = counting_provider(Arc::clone(&calls), sample_body());
    let sampled = || LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "sampled"}],
            "temperature": 0.7
        }),
    };

    call(&provider, sampled()).await;
    call(&provider, sampled()).await;
    flush_subscribers().unwrap();

    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "nondeterministic calls must run live when caching them is disabled"
    );
    let bypasses = captured
        .lock()
        .unwrap()
        .iter()
        .filter(|event| {
            event.name() == "response_cache"
                && event
                    .data()
                    .and_then(|data| data.get("status"))
                    .and_then(Json::as_str)
                    == Some("bypass")
                && event
                    .metadata()
                    .and_then(|metadata| metadata.get("nemo_relay.response_cache.reason"))
                    .and_then(Json::as_str)
                    == Some("nondeterministic_temperature")
        })
        .count();
    assert_eq!(bypasses, 2, "each live call must emit its bypass decision");

    deregister_subscriber("response_cache_nondeterministic_bypass_capture").unwrap();
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
            key_strategy: ResponseCacheKeyStrategy::Unknown("semantic".to_string()),
            namespace: "invalid-config-test".to_string(),
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
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "response_cache.unsupported_key_strategy"),
        "an unsupported key strategy must produce a diagnostic"
    );
}

#[tokio::test]
async fn unknown_and_unavailable_backends_are_rejected_by_validation() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    register_adaptive_component().unwrap();

    let validate = |kind: &str| {
        let mut config = ResponseCacheConfig {
            namespace: "backend-validation-test".to_string(),
            ..ResponseCacheConfig::default()
        };
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
async fn response_cache_validation_diagnostics_identify_the_invalid_setting() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    register_adaptive_component().unwrap();

    let mut cache = ResponseCacheConfig {
        namespace: "diagnostic-contract-test".to_string(),
        key_strategy: ResponseCacheKeyStrategy::Unknown("semantic".to_string()),
        tools: Some(ToolCacheConfig {
            enabled: true,
            default: ToolClass {
                bypass_rate: Some(-0.01),
                ..ToolClass::default()
            },
            ..ToolCacheConfig::default()
        }),
        ..ResponseCacheConfig::default()
    };
    cache.backend.kind = "redis".to_string();

    let report = validate_plugin_config(&PluginConfig {
        components: vec![
            ComponentSpec::new(AdaptiveConfig {
                response_cache: Some(cache),
                ..AdaptiveConfig::default()
            })
            .into(),
        ],
        ..PluginConfig::default()
    });

    assert!(
        report.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "response_cache.unsupported_key_strategy"
                && diagnostic.level == DiagnosticLevel::Error
                && diagnostic.component.as_deref() == Some("response_cache")
                && diagnostic.field.as_deref() == Some("key_strategy")
        }),
        "an unsupported key strategy must identify its setting: {:?}",
        report.diagnostics
    );
    assert!(
        report.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "response_cache.tool_invalid_bypass_rate"
                && diagnostic.level == DiagnosticLevel::Error
                && diagnostic.field.as_deref() == Some("tools")
        }),
        "an invalid tool bypass rate must identify the tools section: {:?}",
        report.diagnostics
    );

    #[cfg(not(feature = "redis-backend"))]
    assert!(
        report.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "response_cache.backend_unavailable"
                && diagnostic.level == DiagnosticLevel::Error
                && diagnostic.field.as_deref() == Some("backend.kind")
        }),
        "redis must be rejected when its backend feature is not compiled: {:?}",
        report.diagnostics
    );

    #[cfg(feature = "redis-backend")]
    assert!(
        report.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "response_cache.missing_redis_url"
                && diagnostic.level == DiagnosticLevel::Error
                && diagnostic.field.as_deref() == Some("backend.config.url")
        }),
        "redis must identify a missing connection URL when its backend feature is compiled: {:?}",
        report.diagnostics
    );
}

#[tokio::test]
async fn hit_preserves_usage_on_the_end_event_and_reports_savings_on_the_mark() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(scoped_cache_config()).await;

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
async fn logical_strategy_reuses_across_reworded_tool_descriptions() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    // `logical` must be accepted by validation (activate_cache asserts no
    // diagnostics) and must reuse across a reworded tool description end-to-end.
    activate_cache(ResponseCacheConfig {
        namespace: "logical-key-integration-test".to_string(),
        key_strategy: ResponseCacheKeyStrategy::Logical,
        ..ResponseCacheConfig::default()
    })
    .await;

    let calls = Arc::new(AtomicUsize::new(0));
    let provider = counting_provider(Arc::clone(&calls), sample_body());

    let request_with_tool = |description: &str| LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            "messages": [{"role": "user", "content": "what is the weather?"}],
            "temperature": 0.0,
            "tools": [{"type": "function", "function": {
                "name": "get_weather",
                "description": description,
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }}]
        }),
    };

    call(&provider, request_with_tool("Get the weather for a city.")).await;
    call(
        &provider,
        request_with_tool("Look up the current weather (reworded)."),
    )
    .await;

    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "logical keying must serve the reworded-tool repeat from cache"
    );
}

#[tokio::test]
async fn errors_are_not_cached() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(scoped_cache_config()).await;

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
    activate_cache(scoped_cache_config()).await;

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
    activate_cache(scoped_cache_config()).await;

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
    activate_cache(scoped_cache_config()).await;

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

fn openai_chat_stream_chunks_for_backend(backend: &str) -> Vec<Json> {
    let mut chunks = openai_chat_stream_chunks();
    chunks[1]["choices"][0]["delta"]["content"] = json!(format!("served by {backend} "));
    chunks
}

fn openai_chat_stream_chunks_with_choice(choice: Json) -> Vec<Json> {
    vec![
        json!({"id": "chatcmpl-lossy", "object": "chat.completion.chunk",
            "created": 1_700_000_000, "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"role": "assistant"},
                "finish_reason": null}]}),
        json!({"id": "chatcmpl-lossy", "object": "chat.completion.chunk",
            "choices": [choice]}),
        json!({"id": "chatcmpl-lossy", "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
    ]
}

fn uncollected_openai_chat_choices() -> [(&'static str, Json); 3] {
    [
        (
            "choice.logprobs",
            json!({"index": 0, "delta": {"content": "answer"},
                "logprobs": {"content": [{"token": "answer", "logprob": -0.1}]},
                "finish_reason": null}),
        ),
        (
            "delta.refusal",
            json!({"index": 0,
                "delta": {"content": "fallback", "refusal": "I cannot help."},
                "finish_reason": null}),
        ),
        (
            "delta.function_call",
            json!({"index": 0,
                "delta": {"content": "fallback",
                    "function_call": {"name": "lookup", "arguments": "{}"}},
                "finish_reason": null}),
        ),
    ]
}

#[tokio::test]
async fn streaming_repeat_is_a_hit_that_skips_the_provider_and_replays_the_aggregate() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    // Streaming aggregation infers its codec from the request surface.
    activate_cache(scoped_cache_config()).await;

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
async fn routing_backends_use_independent_streaming_entries() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(scoped_cache_config()).await;

    let a_calls = Arc::new(AtomicUsize::new(0));
    let a_chunks = openai_chat_stream_chunks_for_backend("backend-a");
    let a_provider = counting_stream_provider(Arc::clone(&a_calls), a_chunks.clone());
    let b_calls = Arc::new(AtomicUsize::new(0));
    let b_chunks = openai_chat_stream_chunks_for_backend("backend-b");
    let b_provider = counting_stream_provider(Arc::clone(&b_calls), b_chunks.clone());

    let a = stream_call(
        &a_provider,
        chat_request_for_backend("same streamed route", "backend-a"),
    )
    .await;
    let b = stream_call(
        &b_provider,
        chat_request_for_backend("same streamed route", "backend-b"),
    )
    .await;
    let a_repeat = stream_call(
        &a_provider,
        chat_request_for_backend("same streamed route", "backend-a"),
    )
    .await;

    assert_eq!(a, a_chunks);
    assert_eq!(b, b_chunks);
    assert_eq!(replayed_text(&a_repeat), "served by backend-a is 42.");
    assert_eq!(a_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        b_calls.load(Ordering::SeqCst),
        1,
        "a different selected backend must stream live instead of replaying another route"
    );
}

#[tokio::test]
async fn streaming_openai_token_cap_spellings_do_not_share_entries() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(scoped_cache_config()).await;

    let legacy_calls = Arc::new(AtomicUsize::new(0));
    let legacy_chunks = openai_chat_stream_chunks();
    let legacy_provider =
        counting_stream_provider(Arc::clone(&legacy_calls), legacy_chunks.clone());
    let current_calls = Arc::new(AtomicUsize::new(0));
    let mut current_chunks = openai_chat_stream_chunks();
    current_chunks[1]["choices"][0]["delta"]["content"] = json!("A different ");
    let current_provider =
        counting_stream_provider(Arc::clone(&current_calls), current_chunks.clone());

    let legacy = stream_call(
        &legacy_provider,
        chat_request_with_token_cap("same streamed prompt", "max_tokens"),
    )
    .await;
    let legacy_repeat = stream_call(
        &legacy_provider,
        chat_request_with_token_cap("same streamed prompt", "max_tokens"),
    )
    .await;
    let current = stream_call(
        &current_provider,
        chat_request_with_token_cap("same streamed prompt", "max_completion_tokens"),
    )
    .await;

    assert_eq!(legacy, legacy_chunks);
    assert_eq!(replayed_text(&legacy_repeat), "The answer is 42.");
    assert_eq!(current, current_chunks);
    assert_eq!(
        legacy_calls.load(Ordering::SeqCst),
        1,
        "the same spelling must remain cacheable"
    );
    assert_eq!(
        current_calls.load(Ordering::SeqCst),
        1,
        "the current spelling must stream live instead of replaying the legacy entry"
    );
}

#[tokio::test]
async fn uncollected_openai_chat_fields_are_not_cached_for_streaming_callers() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(scoped_cache_config()).await;

    for (field, choice) in uncollected_openai_chat_choices() {
        let calls = Arc::new(AtomicUsize::new(0));
        let chunks = openai_chat_stream_chunks_with_choice(choice);
        let provider = counting_stream_provider(Arc::clone(&calls), chunks.clone());
        let prompt = format!("unsafe stream-to-stream: {field}");

        let first = stream_call(&provider, chat_request(&prompt)).await;
        let second = stream_call(&provider, chat_request(&prompt)).await;

        assert_eq!(first, chunks, "{field} must pass through unchanged");
        assert_eq!(second, chunks, "{field} must pass through unchanged");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            2,
            "{field} is erased by aggregation, so a streaming repeat must run live"
        );
    }
}

#[tokio::test]
async fn uncollected_openai_chat_fields_are_not_cached_for_buffered_callers() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(scoped_cache_config()).await;

    for (field, choice) in uncollected_openai_chat_choices() {
        let stream_calls = Arc::new(AtomicUsize::new(0));
        let chunks = openai_chat_stream_chunks_with_choice(choice);
        let stream_provider = counting_stream_provider(Arc::clone(&stream_calls), chunks.clone());
        let prompt = format!("unsafe stream-to-buffered: {field}");
        let streamed = stream_call(&stream_provider, chat_request(&prompt)).await;

        let buffered_calls = Arc::new(AtomicUsize::new(0));
        let buffered_provider =
            counting_provider(Arc::clone(&buffered_calls), json!({"source": field}));
        let response = call(&buffered_provider, chat_request(&prompt)).await;

        assert_eq!(streamed, chunks, "{field} must pass through unchanged");
        assert_eq!(stream_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            buffered_calls.load(Ordering::SeqCst),
            1,
            "{field} is erased by aggregation, so a buffered repeat must run live"
        );
        assert_eq!(response, json!({"source": field}));
    }
}

#[tokio::test]
async fn null_openai_chat_metadata_remains_cacheable() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(scoped_cache_config()).await;

    let chunks = openai_chat_stream_chunks_with_choice(json!({
        "index": 0,
        "delta": {"content": "answer", "refusal": null},
        "logprobs": null,
        "finish_reason": null
    }));
    let calls = Arc::new(AtomicUsize::new(0));
    let provider = counting_stream_provider(Arc::clone(&calls), chunks.clone());

    let first = stream_call(&provider, chat_request("null response metadata")).await;
    let second = stream_call(&provider, chat_request("null response metadata")).await;

    assert_eq!(first, chunks);
    assert_eq!(replayed_text(&second), "answer");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "null logprobs and refusal metadata must not disable caching"
    );
}

#[tokio::test]
async fn partially_consumed_stream_close_does_not_cache_and_closes_upstream_once() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(scoped_cache_config()).await;

    let calls = Arc::new(AtomicUsize::new(0));
    let close_calls = Arc::new(AtomicUsize::new(0));
    let provider = close_tracking_stream_provider(
        Arc::clone(&calls),
        Arc::clone(&close_calls),
        openai_chat_stream_chunks(),
        None,
    );

    let mut first = start_stream(&provider, chat_request("close early")).await;
    assert!(first.next().await.unwrap().is_ok());
    tokio::time::timeout(Duration::from_secs(2), async {
        while close_calls.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the tee should drain and close the short upstream stream");

    first.close().await.unwrap();
    first.close().await.unwrap();
    assert!(first.next().await.is_none());
    assert_eq!(close_calls.load(Ordering::SeqCst), 1);

    stream_call(&provider, chat_request("close early")).await;
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "closing before consumer-observed EOF must not populate the cache"
    );
    assert_eq!(close_calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn stream_cleanup_error_is_idempotent_and_prevents_caching() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(scoped_cache_config()).await;

    let calls = Arc::new(AtomicUsize::new(0));
    let close_calls = Arc::new(AtomicUsize::new(0));
    let provider = close_tracking_stream_provider(
        Arc::clone(&calls),
        Arc::clone(&close_calls),
        openai_chat_stream_chunks(),
        Some(FlowError::NotFound("cache tee cleanup failed".into())),
    );

    let mut first = start_stream(&provider, chat_request("cleanup failure")).await;
    assert!(first.next().await.unwrap().is_ok());
    for _ in 0..2 {
        let error = first.close().await.expect_err("close should fail");
        assert!(error.to_string().contains("cache tee cleanup failed"));
        assert!(matches!(error, FlowError::NotFound(_)));
    }
    assert!(first.next().await.is_none());
    assert_eq!(close_calls.load(Ordering::SeqCst), 1);

    let mut second = start_stream(&provider, chat_request("cleanup failure")).await;
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "an upstream cleanup failure must prevent the cache write"
    );
    assert!(second.close().await.is_err());
    assert_eq!(close_calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn naturally_completed_stream_closes_upstream_before_cache_commit() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(scoped_cache_config()).await;

    let calls = Arc::new(AtomicUsize::new(0));
    let close_calls = Arc::new(AtomicUsize::new(0));
    let provider = close_tracking_stream_provider(
        Arc::clone(&calls),
        Arc::clone(&close_calls),
        openai_chat_stream_chunks(),
        None,
    );

    stream_call(&provider, chat_request("natural completion")).await;
    assert_eq!(
        close_calls.load(Ordering::SeqCst),
        1,
        "upstream must already be closed once the consumer observes natural EOF"
    );

    stream_call(&provider, chat_request("natural completion")).await;
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(close_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn streaming_unrecognized_shape_without_a_codec_runs_live() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    // No codec, and a body the surface detector does not recognize (no
    // messages/system/input) -> nothing to aggregate -> each call runs live.
    activate_cache(scoped_cache_config()).await;

    let calls = Arc::new(AtomicUsize::new(0));
    let provider = counting_stream_provider(Arc::clone(&calls), openai_chat_stream_chunks());
    let unrecognized = || LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({"model": "mystery", "prompt": "hi", "temperature": 0.0}),
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
    activate_cache(scoped_cache_config()).await;

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
    activate_cache(scoped_cache_config()).await;

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
    activate_cache(scoped_cache_config()).await;

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
    activate_cache(scoped_cache_config()).await;

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
    activate_cache(scoped_cache_config()).await;

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
async fn unsafe_cache_scope_headers_and_backends_are_rejected_by_validation() {
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

    for namespace in ["", " \t"] {
        let report = validate(ResponseCacheConfig {
            namespace: namespace.to_string(),
            ..ResponseCacheConfig::default()
        });
        assert!(
            report.has_errors(),
            "an empty cache trust domain must fail validation: {:?}",
            report.diagnostics
        );
        assert!(
            report
                .diagnostics
                .iter()
                .any(|d| d.code == "response_cache.missing_namespace"),
            "an empty cache trust domain must be rejected: {:?}",
            report.diagnostics
        );
    }

    let report = validate(ResponseCacheConfig {
        namespace: "validation-test".to_string(),
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

    let report = validate(ResponseCacheConfig {
        namespace: "validation-test".to_string(),
        header_allowlist: vec!["anthropic-api-key".to_string()],
        ..ResponseCacheConfig::default()
    });
    assert!(
        report
            .diagnostics
            .iter()
            .any(|d| d.code == "response_cache.auth_header_allowlisted"),
        "provider auth headers must never be allowlisted into keys: {:?}",
        report.diagnostics
    );

    let mut zeroed = ResponseCacheConfig {
        namespace: "validation-test".to_string(),
        ..ResponseCacheConfig::default()
    };
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

    let mut mistyped = ResponseCacheConfig {
        namespace: "validation-test".to_string(),
        ..ResponseCacheConfig::default()
    };
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

    let mut mistyped_prefix = ResponseCacheConfig {
        namespace: "validation-test".to_string(),
        ..ResponseCacheConfig::default()
    };
    mistyped_prefix.backend.kind = "redis".to_string();
    mistyped_prefix
        .backend
        .config
        .insert("key_prefix".to_string(), json!(42));
    let report = validate(mistyped_prefix);
    assert!(
        report
            .diagnostics
            .iter()
            .any(|d| d.code == "response_cache.invalid_backend_option"),
        "a wrong-typed key_prefix silently becomes the shared default and must be rejected: {:?}",
        report.diagnostics
    );
}

#[tokio::test]
async fn truncated_stream_without_a_terminal_event_is_not_cached() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(scoped_cache_config()).await;

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
async fn multi_choice_stream_with_an_unfinished_choice_is_not_cached() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(scoped_cache_config()).await;

    // n=2: choice 0 finishes but choice 1 never carries a finish_reason before
    // the clean EOF — a truncation of choice 1. One finished choice must not
    // count as stream completion, or the partial aggregate would be stored and
    // served as the full answer (the streaming flag is not part of the key, so
    // a buffered repeat would hit it directly).
    let truncated = vec![
        json!({"id": "chatcmpl-2", "object": "chat.completion.chunk", "created": 1_700_000_000, "model": "gpt-4o", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "first"}, "finish_reason": null}]}),
        json!({"id": "chatcmpl-2", "object": "chat.completion.chunk", "choices": [{"index": 1, "delta": {"role": "assistant", "content": "second"}, "finish_reason": null}]}),
        json!({"id": "chatcmpl-2", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
    ];
    let stream_calls = Arc::new(AtomicUsize::new(0));
    let stream_provider = counting_stream_provider(Arc::clone(&stream_calls), truncated);
    stream_call(&stream_provider, chat_request("two choices")).await;

    let buffered_calls = Arc::new(AtomicUsize::new(0));
    let buffered_provider = counting_provider(Arc::clone(&buffered_calls), sample_body());
    call(&buffered_provider, chat_request("two choices")).await;

    assert_eq!(
        buffered_calls.load(Ordering::SeqCst),
        1,
        "a multi-choice stream with an unfinished choice must not populate the cache"
    );
}

#[tokio::test]
async fn no_codec_system_less_anthropic_uses_provider_hint_and_caches() {
    // A system-less Anthropic body is shape-ambiguous (reads as OpenAI Chat by
    // shape alone). The provider-name hint disambiguates it, so the correct codec
    // assembles a non-empty aggregate and the repeat is a hit.
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(scoped_cache_config()).await;

    let calls = Arc::new(AtomicUsize::new(0));
    let provider = counting_stream_provider(Arc::clone(&calls), anthropic_stream_chunks());
    let body = || LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.0
        }),
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
    activate_cache(scoped_cache_config()).await;

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
            namespace: "acg-coexistence-test".to_string(),
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

fn counting_tool(calls: Arc<AtomicUsize>, result: Json) -> ToolExecutionNextFn {
    Arc::new(move |_args: Json| {
        let calls = Arc::clone(&calls);
        let result = result.clone();
        Box::pin(async move {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(result.into())
        })
    })
}

async fn tool_call(name: &str, tool: &ToolExecutionNextFn, args: Json) -> Json {
    tool_call_execute(
        ToolCallExecuteParams::builder()
            .name(name)
            .args(args)
            .func(tool.clone())
            .build(),
    )
    .await
    .unwrap()
    .result
}

fn one_cacheable_class(members: &[&str]) -> ToolCacheConfig {
    let mut classes = std::collections::BTreeMap::new();
    classes.insert(
        "read_only".to_string(),
        ToolClass {
            cacheable: true,
            members: members.iter().map(|member| member.to_string()).collect(),
            ..ToolClass::default()
        },
    );
    ToolCacheConfig {
        enabled: true,
        classes,
        ..ToolCacheConfig::default()
    }
}

fn cache_with_tools(tools: ToolCacheConfig) -> ResponseCacheConfig {
    ResponseCacheConfig {
        namespace: "tool-cache-integration-test".to_string(),
        tools: Some(tools),
        ..ResponseCacheConfig::default()
    }
}

#[tokio::test]
async fn classified_tool_repeat_is_a_hit_that_skips_the_tool() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(cache_with_tools(one_cacheable_class(&["docs_lookup"]))).await;

    let calls = Arc::new(AtomicUsize::new(0));
    let tool = counting_tool(Arc::clone(&calls), json!({"doc": "the answer is 42"}));

    let first = tool_call("docs_lookup", &tool, json!({"q": "rust"})).await;
    let second = tool_call("docs_lookup", &tool, json!({"q": "rust"})).await;

    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "a classified-cacheable tool must run once; the repeat is served from cache"
    );
    assert_eq!(first, second, "a hit returns the stored result unchanged");
}

#[tokio::test]
async fn unclassified_tool_calls_emit_uncacheable_bypasses_and_run_live() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(cache_with_tools(one_cacheable_class(&["docs_lookup"]))).await;

    let captured = Arc::new(StdMutex::new(Vec::<Event>::new()));
    let sink = Arc::clone(&captured);
    let (bypass_tx, bypass_rx) = tokio::sync::mpsc::unbounded_channel();
    let bypass_rx = Arc::new(Mutex::new(bypass_rx));
    register_subscriber(
        "response_cache_unclassified_tool_capture",
        Arc::new(move |event: &Event| {
            if event.name() == "response_cache"
                && event
                    .data()
                    .and_then(|data| data.get("status"))
                    .and_then(Json::as_str)
                    == Some("bypass")
                && event
                    .metadata()
                    .and_then(|metadata| metadata.get("nemo_relay.response_cache.reason"))
                    .and_then(Json::as_str)
                    == Some("uncacheable")
                && event
                    .metadata()
                    .and_then(|metadata| metadata.get("nemo_relay.response_cache.surface"))
                    .and_then(Json::as_str)
                    == Some("tool")
            {
                let _ = bypass_tx.send(());
            }
            sink.lock().unwrap().push(event.clone());
        }),
    )
    .unwrap();

    let calls = Arc::new(AtomicUsize::new(0));
    let tool: ToolExecutionNextFn = {
        let calls = Arc::clone(&calls);
        let bypass_rx = Arc::clone(&bypass_rx);
        Arc::new(move |_args: Json| {
            let calls = Arc::clone(&calls);
            let bypass_rx = Arc::clone(&bypass_rx);
            Box::pin(async move {
                let mut bypass_rx = bypass_rx.lock().await;
                let bypass_seen =
                    tokio::time::timeout(Duration::from_secs(1), bypass_rx.recv()).await;
                assert!(
                    matches!(bypass_seen, Ok(Some(()))),
                    "the matching bypass mark must be queued before the live callback starts"
                );
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(json!({"written": "pizza"}).into())
            })
        })
    };

    tool_call("write_order", &tool, json!({"item": "pizza"})).await;
    tool_call("write_order", &tool, json!({"item": "pizza"})).await;
    flush_subscribers().unwrap();

    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "an unclassified tool must run live on every call"
    );
    let events = captured.lock().unwrap();
    let cache_marks: Vec<_> = events
        .iter()
        .filter(|event| event.name() == "response_cache")
        .map(|event| {
            (
                event
                    .data()
                    .and_then(|data| data.get("status"))
                    .and_then(Json::as_str),
                event
                    .metadata()
                    .and_then(|metadata| metadata.get("nemo_relay.response_cache.reason"))
                    .and_then(Json::as_str),
                event
                    .metadata()
                    .and_then(|metadata| metadata.get("nemo_relay.response_cache.surface"))
                    .and_then(Json::as_str),
            )
        })
        .collect();
    assert_eq!(
        cache_marks,
        vec![
            (Some("bypass"), Some("uncacheable"), Some("tool")),
            (Some("bypass"), Some("uncacheable"), Some("tool")),
        ],
        "each unclassified call must emit exactly one ordered bypass decision"
    );

    drop(events);
    deregister_subscriber("response_cache_unclassified_tool_capture").unwrap();
}

#[tokio::test]
async fn an_effectful_class_is_never_cached() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    let mut classes = std::collections::BTreeMap::new();
    classes.insert(
        "effectful".to_string(),
        ToolClass {
            cacheable: false,
            members: vec!["send_email".to_string()],
            ..ToolClass::default()
        },
    );
    activate_cache(cache_with_tools(ToolCacheConfig {
        enabled: true,
        classes,
        ..ToolCacheConfig::default()
    }))
    .await;

    let calls = Arc::new(AtomicUsize::new(0));
    let tool = counting_tool(Arc::clone(&calls), json!({"sent": true}));

    tool_call("send_email", &tool, json!({"to": "a@b.c"})).await;
    tool_call("send_email", &tool, json!({"to": "a@b.c"})).await;

    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "an effectful (cacheable=false) tool must run every time — a hit would skip the side effect"
    );
}

#[tokio::test]
async fn disabled_tools_section_does_not_cache() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    let mut tools = one_cacheable_class(&["docs_lookup"]);
    tools.enabled = false;
    activate_cache(cache_with_tools(tools)).await;

    let calls = Arc::new(AtomicUsize::new(0));
    let tool = counting_tool(Arc::clone(&calls), json!({"doc": "x"}));

    tool_call("docs_lookup", &tool, json!({"q": "rust"})).await;
    tool_call("docs_lookup", &tool, json!({"q": "rust"})).await;

    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "with tools.enabled = false the tool intercept is not installed"
    );
}

#[tokio::test]
async fn conventional_error_shaped_tool_results_are_not_cached_by_default() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(cache_with_tools(one_cacheable_class(&["lookup"]))).await;

    let calls = Arc::new(AtomicUsize::new(0));
    let tool = counting_tool(
        Arc::clone(&calls),
        json!({"type": "tool_result", "is_error": true, "content": "not found"}),
    );

    tool_call("lookup", &tool, json!({"q": "missing"})).await;
    tool_call("lookup", &tool, json!({"q": "missing"})).await;

    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "an Anthropic-style in-band tool error must run live again unless cache_errors is enabled"
    );
}

#[tokio::test]
async fn conventional_error_shaped_tool_results_can_be_cached_when_opted_in() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    let mut tools = one_cacheable_class(&["lookup"]);
    tools.cache_errors = true;
    activate_cache(cache_with_tools(tools)).await;

    let calls = Arc::new(AtomicUsize::new(0));
    let tool = counting_tool(
        Arc::clone(&calls),
        json!({"type": "tool_result", "is_error": true, "content": "not found"}),
    );

    tool_call("lookup", &tool, json!({"q": "missing"})).await;
    tool_call("lookup", &tool, json!({"q": "missing"})).await;

    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "cache_errors=true explicitly permits caching Anthropic-style in-band error results"
    );
}

#[tokio::test]
async fn tool_callback_errors_emit_misses_and_are_never_cached() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(cache_with_tools(one_cacheable_class(&["lookup"]))).await;

    let captured = Arc::new(StdMutex::new(Vec::<Event>::new()));
    let sink = Arc::clone(&captured);
    register_subscriber(
        "response_cache_tool_callback_error_capture",
        Arc::new(move |event: &Event| sink.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let calls = Arc::new(AtomicUsize::new(0));
    let tool: ToolExecutionNextFn = {
        let calls = Arc::clone(&calls);
        Arc::new(move |_args| {
            let calls = Arc::clone(&calls);
            Box::pin(async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Err(FlowError::Internal("tool unavailable".to_string()))
            })
        })
    };

    for _ in 0..2 {
        let error = tool_call_execute(
            ToolCallExecuteParams::builder()
                .name("lookup")
                .args(json!({"q": "missing"}))
                .func(tool.clone())
                .build(),
        )
        .await
        .expect_err("a tool callback error must reach the caller");
        assert!(matches!(error, FlowError::Internal(message) if message == "tool unavailable"));
    }
    assert_eq!(calls.load(Ordering::SeqCst), 2);

    flush_subscribers().unwrap();
    let misses = captured
        .lock()
        .unwrap()
        .iter()
        .filter(|event| {
            event.name() == "response_cache"
                && event
                    .data()
                    .and_then(|data| data.get("status"))
                    .and_then(Json::as_str)
                    == Some("miss")
                && event
                    .metadata()
                    .and_then(|metadata| metadata.get("nemo_relay.response_cache.surface"))
                    .and_then(Json::as_str)
                    == Some("tool")
        })
        .count();
    assert_eq!(misses, 2, "each failed call must still report a cache miss");
    deregister_subscriber("response_cache_tool_callback_error_capture").unwrap();
}

#[tokio::test]
async fn execution_intercepts_outside_the_cache_run_on_hits() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();

    let outer_runs = Arc::new(AtomicUsize::new(0));
    register_tool_execution_intercept(
        "response_cache_outer_tool_execution_test",
        40,
        Arc::new({
            let outer_runs = Arc::clone(&outer_runs);
            move |_name, args, next| {
                let outer_runs = Arc::clone(&outer_runs);
                Box::pin(async move {
                    outer_runs.fetch_add(1, Ordering::SeqCst);
                    next(args).await.map(Into::into)
                })
            }
        }),
    )
    .unwrap();
    activate_cache(cache_with_tools(one_cacheable_class(&["lookup"]))).await;

    let calls = Arc::new(AtomicUsize::new(0));
    let tool = counting_tool(Arc::clone(&calls), json!({"answer": "cached"}));
    tool_call("lookup", &tool, json!({"q": "relay"})).await;
    tool_call("lookup", &tool, json!({"q": "relay"})).await;

    assert_eq!(calls.load(Ordering::SeqCst), 1, "the second call must hit");
    assert_eq!(
        outer_runs.load(Ordering::SeqCst),
        2,
        "a lower-priority execution intercept wraps the cache and runs on hits"
    );
    deregister_tool_execution_intercept("response_cache_outer_tool_execution_test").unwrap();
}

#[tokio::test]
async fn tool_hit_emits_a_surface_tool_mark_with_saved_invocations() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(cache_with_tools(one_cacheable_class(&["docs_lookup"]))).await;

    let captured = Arc::new(StdMutex::new(Vec::<Event>::new()));
    let sink = Arc::clone(&captured);
    register_subscriber(
        "response_cache_tool_capture",
        Arc::new(move |event: &Event| sink.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let calls = Arc::new(AtomicUsize::new(0));
    let tool = counting_tool(Arc::clone(&calls), json!({"doc": "x"}));

    tool_call("docs_lookup", &tool, json!({"q": "rust"})).await; // miss
    tool_call("docs_lookup", &tool, json!({"q": "rust"})).await; // hit
    flush_subscribers().unwrap();

    let events = captured.lock().unwrap();
    let cache_statuses: Vec<_> = events
        .iter()
        .filter(|event| event.name() == "response_cache")
        .filter_map(|event| {
            event
                .data()
                .and_then(|data| data.get("status"))
                .and_then(Json::as_str)
        })
        .collect();
    assert_eq!(
        cache_statuses,
        vec!["miss", "hit"],
        "a cacheable repeat must emit one miss followed by one hit"
    );
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
        .expect("a response_cache tool hit mark should be emitted");
    let metadata = hit_mark.metadata().expect("hit mark has metadata");
    assert_eq!(
        metadata
            .get("nemo_relay.response_cache.surface")
            .and_then(Json::as_str),
        Some("tool"),
        "the tool hit mark must be tagged surface = tool"
    );
    assert_eq!(
        metadata
            .get("nemo_relay.response_cache.saved_invocations")
            .and_then(Json::as_u64),
        Some(1),
        "a tool hit reports one saved invocation"
    );
    let miss_mark = events
        .iter()
        .find(|event| {
            event.name() == "response_cache"
                && event
                    .data()
                    .and_then(|data| data.get("status"))
                    .and_then(Json::as_str)
                    == Some("miss")
        })
        .expect("a response_cache tool miss mark should be emitted");
    assert!(
        !miss_mark
            .metadata()
            .expect("miss mark has metadata")
            .as_object()
            .expect("miss mark metadata is an object")
            .contains_key("nemo_relay.response_cache.saved_invocations"),
        "a tool miss must not report saved invocations"
    );

    drop(events);
    deregister_subscriber("response_cache_tool_capture").unwrap();
}

#[tokio::test]
async fn invalid_tool_config_is_rejected_by_validation() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    register_adaptive_component().unwrap();

    let mut classes = std::collections::BTreeMap::new();
    classes.insert(
        "class_a".to_string(),
        ToolClass {
            cacheable: true,
            members: vec!["dup".to_string()],
            ..ToolClass::default()
        },
    );
    classes.insert(
        "class_b".to_string(),
        ToolClass {
            cacheable: true,
            ttl_seconds: Some(0),
            bypass_rate: Some(1.1),
            members: vec!["dup".to_string()],
            ..ToolClass::default()
        },
    );
    let mut overrides = std::collections::BTreeMap::new();
    overrides.insert(
        "docs_lookup".to_string(),
        ToolOverride {
            bypass_rate: Some(-0.1),
            ..ToolOverride::default()
        },
    );
    let adaptive = AdaptiveConfig {
        response_cache: Some(cache_with_tools(ToolCacheConfig {
            enabled: true,
            default: ToolClass {
                cacheable: true,
                members: vec!["safe_lookup".to_string()],
                ..ToolClass::default()
            },
            classes,
            overrides,
            ..ToolCacheConfig::default()
        })),
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
            .any(|diagnostic| diagnostic.code == "response_cache.tool_multiple_classes"),
        "a tool in multiple classes must be rejected: {:?}",
        report.diagnostics
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "response_cache.tool_invalid_ttl"),
        "a zero class TTL must be rejected: {:?}",
        report.diagnostics
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "response_cache.tool_invalid_bypass_rate"),
        "out-of-range class and override bypass rates must be rejected: {:?}",
        report.diagnostics
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "response_cache.tool_default_members"),
        "members on the default bucket must be rejected: {:?}",
        report.diagnostics
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "response_cache.tool_cacheable_default"),
        "a cacheable default bucket must be rejected: {:?}",
        report.diagnostics
    );
}

#[tokio::test]
async fn wildcard_member_validation_rules() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    register_adaptive_component().unwrap();

    let validate = |classes: std::collections::BTreeMap<String, ToolClass>| {
        let adaptive = AdaptiveConfig {
            response_cache: Some(cache_with_tools(ToolCacheConfig {
                enabled: true,
                classes,
                ..ToolCacheConfig::default()
            })),
            ..AdaptiveConfig::default()
        };
        validate_plugin_config(&PluginConfig {
            components: vec![ComponentSpec::new(adaptive).into()],
            ..PluginConfig::default()
        })
    };
    let cacheable_class = |members: &[&str]| ToolClass {
        cacheable: true,
        members: members.iter().map(|member| member.to_string()).collect(),
        ..ToolClass::default()
    };

    let mut classes = std::collections::BTreeMap::new();
    classes.insert("class_a".to_string(), cacheable_class(&["docs_*"]));
    classes.insert("class_b".to_string(), cacheable_class(&["docs_*"]));
    let report = validate(classes);
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "response_cache.tool_multiple_classes"),
        "an identical pattern in two classes must be rejected: {:?}",
        report.diagnostics
    );

    let mut classes = std::collections::BTreeMap::new();
    classes.insert(
        "read_only".to_string(),
        cacheable_class(&["docs_*", "docs_*"]),
    );
    let report = validate(classes);
    assert!(
        !report
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "response_cache.tool_multiple_classes"),
        "a repeated member inside one class is inert, not a cross-class conflict: {:?}",
        report.diagnostics
    );

    let mut classes = std::collections::BTreeMap::new();
    classes.insert("class_a".to_string(), cacheable_class(&["docs_*"]));
    classes.insert("class_b".to_string(), cacheable_class(&["*_lookup"]));
    let report = validate(classes);
    assert!(
        !report
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code.starts_with("response_cache.tool")),
        "distinct overlapping patterns must validate cleanly: {:?}",
        report.diagnostics
    );

    let mut classes = std::collections::BTreeMap::new();
    classes.insert("safe".to_string(), cacheable_class(&["*_email"]));
    classes.insert(
        "effectful".to_string(),
        ToolClass {
            cacheable: false,
            members: vec!["send_*".to_string()],
            ..ToolClass::default()
        },
    );
    let report = validate(classes);
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "response_cache.tool_conflicting_classes"),
        "opposite cacheability on overlapping wildcard classes must be rejected: {:?}",
        report.diagnostics
    );

    let mut classes = std::collections::BTreeMap::new();
    classes.insert("everything".to_string(), cacheable_class(&["*"]));
    let report = validate(classes);
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "response_cache.tool_catch_all_member"),
        "a cacheable '*' member must be rejected: {:?}",
        report.diagnostics
    );

    for invalid_pattern in ["**", "docs_*_lookup", "docs**"] {
        let mut classes = std::collections::BTreeMap::new();
        classes.insert("invalid".to_string(), cacheable_class(&[invalid_pattern]));
        let report = validate(classes);
        assert!(
            report
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "response_cache.tool_invalid_pattern"),
            "unsupported pattern '{invalid_pattern}' must be rejected: {:?}",
            report.diagnostics
        );
    }

    let mut overrides = std::collections::BTreeMap::new();
    overrides.insert(
        "*".to_string(),
        ToolOverride {
            cacheable: Some(true),
            ..ToolOverride::default()
        },
    );
    let adaptive = AdaptiveConfig {
        response_cache: Some(cache_with_tools(ToolCacheConfig {
            enabled: true,
            overrides,
            ..ToolCacheConfig::default()
        })),
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
            .any(|diagnostic| diagnostic.code == "response_cache.tool_catch_all_override"),
        "a cacheable '*' override must be rejected: {:?}",
        report.diagnostics
    );

    let mut overrides = std::collections::BTreeMap::new();
    overrides.insert(
        "docs_*_lookup".to_string(),
        ToolOverride {
            cacheable: Some(true),
            ..ToolOverride::default()
        },
    );
    let adaptive = AdaptiveConfig {
        response_cache: Some(cache_with_tools(ToolCacheConfig {
            enabled: true,
            overrides,
            ..ToolCacheConfig::default()
        })),
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
            .any(|diagnostic| diagnostic.code == "response_cache.tool_invalid_pattern"),
        "an unsupported override pattern must be rejected: {:?}",
        report.diagnostics
    );

    let mut overrides = std::collections::BTreeMap::new();
    overrides.insert(
        "*_email".to_string(),
        ToolOverride {
            cacheable: Some(true),
            ..ToolOverride::default()
        },
    );
    overrides.insert(
        "send_*".to_string(),
        ToolOverride {
            cacheable: Some(false),
            ..ToolOverride::default()
        },
    );
    let adaptive = AdaptiveConfig {
        response_cache: Some(cache_with_tools(ToolCacheConfig {
            enabled: true,
            overrides,
            ..ToolCacheConfig::default()
        })),
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
            .any(|diagnostic| diagnostic.code == "response_cache.tool_conflicting_overrides"),
        "opposite cacheability on overlapping wildcard overrides must be rejected: {:?}",
        report.diagnostics
    );

    let mut overrides = std::collections::BTreeMap::new();
    overrides.insert(
        "docs_*".to_string(),
        ToolOverride {
            cacheable: Some(false),
            ..ToolOverride::default()
        },
    );
    overrides.insert(
        "*_private".to_string(),
        ToolOverride {
            ttl_seconds: Some(60),
            ..ToolOverride::default()
        },
    );
    let adaptive = AdaptiveConfig {
        response_cache: Some(cache_with_tools(ToolCacheConfig {
            enabled: true,
            default: ToolClass {
                cacheable: true,
                ..ToolClass::default()
            },
            overrides,
            ..ToolCacheConfig::default()
        })),
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
            .any(|diagnostic| diagnostic.code == "response_cache.tool_conflicting_overrides"),
        "a wildcard override that inherits cacheability must not outrank an explicit deny: {:?}",
        report.diagnostics
    );
}

#[tokio::test]
async fn unknown_tool_field_warns_but_valid_class_names_do_not() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    register_adaptive_component().unwrap();

    let adaptive_json = json!({
        "response_cache": {
            "tools": {
                "enabled": true,
                "cache_errors": false,
                "classes": {
                    "read_only": {
                        "cacheable": true,
                        "tool_version": "class-v1",
                        "members": ["docs_lookup"],
                        "not_a_field": 7
                    }
                }
            }
        }
    });
    let component = nemo_relay::plugin::PluginComponentSpec {
        kind: "adaptive".to_string(),
        enabled: true,
        config: adaptive_json.as_object().unwrap().clone(),
    };
    let report = validate_plugin_config(&PluginConfig {
        components: vec![component],
        ..PluginConfig::default()
    });

    let unknown_field_diags: Vec<_> = report
        .diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.code == "adaptive.unknown_field")
        .collect();
    assert_eq!(
        unknown_field_diags.len(),
        1,
        "exactly one unknown-field warning (the bogus field), not the class name: {:?}",
        report.diagnostics
    );
    assert_eq!(
        unknown_field_diags[0].field.as_deref(),
        Some("not_a_field"),
        "the warning must point at the bogus field, never the class name"
    );
}
