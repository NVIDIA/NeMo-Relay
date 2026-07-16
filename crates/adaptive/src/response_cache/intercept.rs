// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The buffered and streaming LLM execution intercepts: cache decisions,
//! the streaming tee, and the storage rules.

use std::sync::Arc;

use nemo_relay::api::llm::LlmRequest;
use nemo_relay::api::runtime::{
    LlmExecutionFn, LlmExecutionNextFn, LlmJsonStream, LlmStreamExecutionFn,
    LlmStreamExecutionNextFn,
};
use nemo_relay::codec::resolve::{detect_request_surface_with_hint, streaming_codec};
use nemo_relay::codec::streaming::StreamingCodec;
use nemo_relay::error::Result as FlowResult;
use serde_json::Value as Json;
use std::cell::Cell;
use std::collections::BTreeSet;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use crate::config::ResponseCacheConfig;
use crate::response_cache::key::{KeyOutcome, build_cache_key};
use crate::response_cache::mark::{CacheMark, emit_cache_mark, savings_from};
use crate::response_cache::replay::{replay_aggregate, replay_is_lossy};
use crate::response_cache::store::{CacheEntry, CacheStore, now_unix_ms};

/// Bounded channel capacity for the streaming tee: it forwards live chunks to
/// the consumer while accumulating them, applying backpressure when the consumer
/// is slow.
const STREAM_TEE_CHANNEL_CAP: usize = 64;

/// Builds the buffered LLM execution intercept for the response cache.
///
/// Called from the adaptive runtime when the `response_cache` section is present.
pub(crate) fn make_intercept(
    store: Arc<dyn CacheStore>,
    config: Arc<ResponseCacheConfig>,
) -> LlmExecutionFn {
    Arc::new(
        move |provider: &str, request: LlmRequest, next: LlmExecutionNextFn| {
            let store = Arc::clone(&store);
            let config = Arc::clone(&config);
            let provider = provider.to_string();
            Box::pin(run_cache(provider, request, next, store, config))
        },
    )
}

/// Builds the streaming LLM execution intercept for the response cache.
///
/// On a miss it tees the live stream — forwarding chunks while feeding them to
/// the codec — and on natural completion stores the codec-assembled
/// **aggregate** response (the same shape a buffered call stores), so buffered
/// and streaming entries share one key. On a hit it replays that aggregate as
/// provider-native chunks. The codec is inferred from the request surface; only
/// a request whose surface can't be inferred runs live (uncached).
pub(crate) fn make_stream_intercept(
    store: Arc<dyn CacheStore>,
    config: Arc<ResponseCacheConfig>,
) -> LlmStreamExecutionFn {
    Arc::new(
        move |provider: &str, request: LlmRequest, next: LlmStreamExecutionNextFn| {
            let store = Arc::clone(&store);
            let config = Arc::clone(&config);
            let provider = provider.to_string();
            Box::pin(run_cache_stream(provider, request, next, store, config))
        },
    )
}

/// Core get-or-miss logic. Separated into a free `async fn` so the future type
/// is explicit and `Send`.
async fn run_cache(
    provider: String,
    request: LlmRequest,
    next: LlmExecutionNextFn,
    store: Arc<dyn CacheStore>,
    config: Arc<ResponseCacheConfig>,
) -> FlowResult<Json> {
    let backend = store.backend_kind();

    // Decision marks are emitted before `next()` (like the runtime's start
    // event) so every decision is recorded even when the provider then errors.
    let key = match build_cache_key(&provider, &request, &config) {
        KeyOutcome::Key(key) => key,
        KeyOutcome::Bypass(reason) => {
            emit_cache_mark(CacheMark::new("bypass", backend).reason(reason));
            return next(request).await;
        }
    };

    let model = request_model(&request);

    // Sampled bypass: re-run live to catch drift, refreshing the stored answer.
    if should_bypass(config.bypass_rate) {
        emit_cache_mark(
            CacheMark::new("bypass", backend)
                .reason("sampled")
                .key_hash(&key),
        );
        let response = next(request).await?;
        maybe_store(&store, &config, &key, &provider, model, &response).await;
        return Ok(response);
    }

    match store.get(&key).await {
        Ok(Some(entry)) => {
            let age_ms = now_unix_ms().saturating_sub(entry.created_unix_ms);
            let (saved_tokens, saved_cost) = savings_from(&entry);
            emit_cache_mark(
                CacheMark::new("hit", backend)
                    .key_hash(&key)
                    .age_ms(age_ms)
                    .ttl_ms(config.ttl().as_millis() as u64)
                    .savings(saved_tokens, saved_cost),
            );
            // A reuse must be shape-identical to a live call (usage intact);
            // savings are reported on the mark, never by mutating the body.
            Ok(entry.response.clone())
        }
        Ok(None) => {
            emit_cache_mark(
                CacheMark::new("miss", backend)
                    .key_hash(&key)
                    .ttl_ms(config.ttl().as_millis() as u64),
            );
            let response = next(request).await?;
            maybe_store(&store, &config, &key, &provider, model, &response).await;
            Ok(response)
        }
        Err(_) => {
            // Cache read failed: fail open as a live call and do not store.
            emit_cache_mark(
                CacheMark::new("miss", backend)
                    .reason("store_error")
                    .key_hash(&key),
            );
            next(request).await
        }
    }
}

/// Streaming counterpart of [`run_cache`]. Assembles the streamed chunks into a
/// single aggregate response (via the configured codec) and stores **that** — the
/// same shape a buffered call stores — so buffered and streaming entries share
/// one key. Replays the stored aggregate on a hit. Aggregation needs a codec,
/// inferred from the request surface; only an unrecognized surface runs live
/// (uncached). Fails open.
async fn run_cache_stream(
    provider: String,
    request: LlmRequest,
    next: LlmStreamExecutionNextFn,
    store: Arc<dyn CacheStore>,
    config: Arc<ResponseCacheConfig>,
) -> FlowResult<LlmJsonStream> {
    let backend = store.backend_kind();

    // Assembling streamed chunks into a stored response needs a streaming codec,
    // inferred via the shared request-surface detector (the observability/ACG
    // decode path), so gateway traffic caches with zero configuration. The
    // detector is hinted with the provider name so a system-less Anthropic
    // request is not misread as OpenAI Chat (an ambiguity core documents); the
    // inference is guarded against a mistake at store time (see
    // `tee_and_aggregate`).
    let codec = match detect_request_surface_with_hint(&request.content, Some(&provider)) {
        Some(surface) => streaming_codec(surface),
        None => {
            emit_cache_mark(CacheMark::new("bypass", backend).reason("stream_no_codec"));
            return next(request).await;
        }
    };

    // As in `run_cache`, the decision mark is emitted before `next()`.
    let key = match build_cache_key(&provider, &request, &config) {
        KeyOutcome::Key(key) => key,
        KeyOutcome::Bypass(reason) => {
            emit_cache_mark(CacheMark::new("bypass", backend).reason(reason));
            return next(request).await;
        }
    };

    let model = request_model(&request);

    // Sampled bypass: run live (and re-aggregate to refresh the stored answer).
    if should_bypass(config.bypass_rate) {
        emit_cache_mark(
            CacheMark::new("bypass", backend)
                .reason("sampled")
                .key_hash(&key),
        );
        let live = next(request).await?;
        return Ok(tee_and_aggregate(
            live, codec, store, config, key, provider, model,
        ));
    }

    match store.get(&key).await {
        Ok(Some(entry)) => {
            // An unfaithful chunk replay must not be served; the entry still
            // serves buffered callers, so run live without disturbing it.
            if replay_is_lossy(&entry.response) {
                emit_cache_mark(
                    CacheMark::new("miss", backend)
                        .reason("replay_lossy")
                        .key_hash(&key),
                );
                return next(request).await;
            }
            let age_ms = now_unix_ms().saturating_sub(entry.created_unix_ms);
            let (saved_tokens, saved_cost) = savings_from(&entry);
            emit_cache_mark(
                CacheMark::new("hit", backend)
                    .key_hash(&key)
                    .age_ms(age_ms)
                    .ttl_ms(config.ttl().as_millis() as u64)
                    .savings(saved_tokens, saved_cost),
            );
            // Replay the stored aggregate as provider-native chunks.
            Ok(replay_aggregate(entry.response.clone()))
        }
        Ok(None) => {
            emit_cache_mark(
                CacheMark::new("miss", backend)
                    .key_hash(&key)
                    .ttl_ms(config.ttl().as_millis() as u64),
            );
            let live = next(request).await?;
            Ok(tee_and_aggregate(
                live, codec, store, config, key, provider, model,
            ))
        }
        Err(_) => {
            // Cache read failed: fail open as a live stream and do not store.
            emit_cache_mark(
                CacheMark::new("miss", backend)
                    .reason("store_error")
                    .key_hash(&key),
            );
            next(request).await
        }
    }
}

/// Tees a live stream: forwards each chunk to the consumer while feeding it to the
/// codec's collector, and on natural, error-free completion stores the
/// codec-assembled **aggregate**. An upstream error, a chunk the codec rejects, or
/// a dropped consumer caches nothing. A content-empty aggregate is also not
/// stored — the codec was inferred from the request surface, so an empty result
/// signals a mis-inference (e.g. a system-less Anthropic request read as OpenAI
/// Chat, whose collector silently drops the foreign chunks), and caching it
/// would serve a wrong empty response on the repeat.
fn tee_and_aggregate(
    live: LlmJsonStream,
    codec: Box<dyn StreamingCodec>,
    store: Arc<dyn CacheStore>,
    config: Arc<ResponseCacheConfig>,
    key: String,
    provider: String,
    model: Option<String>,
) -> LlmJsonStream {
    let (tx, rx) = tokio::sync::mpsc::channel::<FlowResult<Json>>(STREAM_TEE_CHANNEL_CAP);
    tokio::spawn(async move {
        let mut collect = codec.collector();
        let mut live = live;
        let mut collector_failed = false;
        let mut completion = StreamCompletion::default();
        while let Some(item) = live.next().await {
            match &item {
                Ok(chunk) => {
                    // In-band provider errors never surface as stream-level Err.
                    if chunk_is_inband_error(chunk) || collect(chunk.clone()).is_err() {
                        collector_failed = true;
                    }
                    completion.observe(chunk);
                }
                Err(_) => {
                    // Upstream error is a failed call: forward it, never cache.
                    let _ = tx.send(item).await;
                    return;
                }
            }
            // Forward to the consumer; a send error means it was dropped.
            if tx.send(item).await.is_err() {
                return;
            }
        }
        // Store only protocol-complete streams: every collector finalizes a
        // clean truncation as a well-formed partial.
        if !collector_failed && completion.is_terminal() {
            let aggregate = codec.finalizer()();
            // Empty = mis-inferred surface; lossy = unfaithful replay.
            if aggregate_has_no_content(&aggregate) || aggregate_replay_lossy(&aggregate) {
                return;
            }
            maybe_store(&store, &config, &key, &provider, model, &aggregate).await;
        }
    });
    Box::pin(ReceiverStream::new(rx))
}

/// Streamed content the collectors assemble lossily: thinking blocks lose
/// their deltas/signature; refusal-only chat choices lose the refusal text.
fn aggregate_replay_lossy(aggregate: &Json) -> bool {
    if let Some(content) = aggregate.get("content").and_then(Json::as_array)
        && content.iter().any(|block| {
            matches!(
                block.get("type").and_then(Json::as_str),
                Some("thinking" | "redacted_thinking")
            )
        })
    {
        return true;
    }
    if let Some(choices) = aggregate.get("choices").and_then(Json::as_array)
        && !choices.is_empty()
        && choices.iter().all(|choice| {
            let message = choice.get("message");
            let content_empty = message
                .and_then(|message| message.get("content"))
                .is_none_or(|content| {
                    content.is_null() || content.as_str().is_some_and(str::is_empty)
                });
            let no_tool_calls = message
                .and_then(|message| message.get("tool_calls"))
                .and_then(Json::as_array)
                .is_none_or(|calls| calls.is_empty());
            content_empty && no_tool_calls
        })
    {
        return true;
    }
    false
}

/// Tracks stream completion (`response.incomplete` is deliberately excluded:
/// a capped answer must not replay as "the" answer). Chat streams interleave
/// per-choice chunks, so every choice that appeared must carry a
/// `finish_reason` — a clean close after only some choices finished is a
/// truncation that must not be cached.
#[derive(Default)]
struct StreamCompletion {
    saw_stop_event: bool,
    choices_seen: BTreeSet<u64>,
    choices_finished: BTreeSet<u64>,
}

impl StreamCompletion {
    fn observe(&mut self, chunk: &Json) {
        if let Some(choices) = chunk.get("choices").and_then(Json::as_array) {
            for choice in choices {
                let index = choice.get("index").and_then(Json::as_u64).unwrap_or(0);
                self.choices_seen.insert(index);
                if choice
                    .get("finish_reason")
                    .is_some_and(|reason| !reason.is_null())
                {
                    self.choices_finished.insert(index);
                }
            }
        }
        self.saw_stop_event |= matches!(
            chunk.get("type").and_then(Json::as_str),
            Some("message_stop" | "response.completed")
        );
    }

    fn is_terminal(&self) -> bool {
        self.saw_stop_event
            || (!self.choices_seen.is_empty() && self.choices_seen == self.choices_finished)
    }
}

/// Provider-native in-band error chunk; a false positive only skips a store.
fn chunk_is_inband_error(chunk: &Json) -> bool {
    if chunk.get("error").is_some_and(|error| !error.is_null()) {
        return true;
    }
    matches!(
        chunk.get("type").and_then(Json::as_str),
        Some("error" | "response.failed")
    )
}

/// True when a finalized streaming aggregate carries no response content in any
/// known provider shape (OpenAI Chat `choices`, OpenAI Responses `output`,
/// Anthropic `content`) — used to reject a mis-inferred codec's empty output.
fn aggregate_has_no_content(aggregate: &Json) -> bool {
    let empty_array = |key: &str| {
        aggregate
            .get(key)
            .and_then(Json::as_array)
            .is_none_or(|items| items.is_empty())
    };
    empty_array("choices") && empty_array("output") && empty_array("content")
}

async fn maybe_store(
    store: &Arc<dyn CacheStore>,
    config: &ResponseCacheConfig,
    key: &str,
    provider: &str,
    model: Option<String>,
    response: &Json,
) {
    // Failed calls are never cached.
    if is_error_response(response) {
        return;
    }
    let entry = CacheEntry::new(
        response.clone(),
        config.ttl(),
        key.to_string(),
        model,
        Some(provider.to_string()),
    );
    // Fail open: a store error must never break the live call.
    let _ = store.set(key, entry, config.ttl()).await;
}

fn request_model(request: &LlmRequest) -> Option<String> {
    request
        .content
        .get("model")
        .and_then(Json::as_str)
        .map(str::to_string)
}

/// Non-null `error` or non-final `status` = not a complete, replayable
/// answer. Must tolerate `error: null` — real Responses success bodies carry it.
fn is_error_response(response: &Json) -> bool {
    let Some(object) = response.as_object() else {
        return false;
    };
    if object.get("error").is_some_and(|error| !error.is_null()) {
        return true;
    }
    matches!(
        object.get("status").and_then(Json::as_str),
        Some("failed" | "cancelled" | "canceled" | "incomplete" | "in_progress" | "queued")
    )
}

thread_local! {
    static RNG_STATE: Cell<u64> = Cell::new(rng_seed());
}

fn rng_seed() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|delta| delta.as_nanos() as u64)
        .unwrap_or(0);
    (nanos ^ 0x9E37_79B9_7F4A_7C15) | 1
}

fn next_unit_f64() -> f64 {
    RNG_STATE.with(|cell| {
        let mut x = cell.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        cell.set(x);
        (x >> 11) as f64 / ((1u64 << 53) as f64)
    })
}

pub(crate) fn should_bypass(rate: f64) -> bool {
    if rate <= 0.0 {
        false
    } else if rate >= 1.0 {
        true
    } else {
        next_unit_f64() < rate
    }
}
