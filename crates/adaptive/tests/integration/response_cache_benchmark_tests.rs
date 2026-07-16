// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the cache-ON-vs-OFF savings benchmark
//! (`crates/adaptive`).
//!
//! These validate the accounting the benchmark relies on directly against the
//! real runtime: the `response_cache` marks partition every managed call into
//! `hits` / `misses` / `bypasses`; the actual provider call count equals
//! `total_requests - hits`; and the savings identity
//!
//! ```text
//! baseline_tokens == served_tokens + saved_tokens
//! ```
//!
//! holds, where `baseline_tokens` is the all-live cost (`total_requests *
//! per_call_total_tokens`), `served_tokens` is what the provider actually
//! served (accumulated only when the stub runs), and `saved_tokens` is summed
//! from the hit marks. Because the provider stub runs exactly on misses and
//! bypasses (never on hits), `served_tokens` auto-excludes reused answers.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

use nemo_relay::api::event::Event;
use nemo_relay::api::llm::LlmRequest;
use nemo_relay::api::runtime::{LlmExecutionNextFn, NemoRelayContextState, global_context};
use nemo_relay::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use nemo_relay::plugin::clear_plugin_configuration;
use nemo_relay_adaptive::ResponseCacheConfig;
use serde_json::{Value as Json, json};
use tokio::sync::Mutex;

#[path = "response_cache_common.rs"]
mod response_cache_common;
use response_cache_common::{activate_cache, call, chat_request};

static TEST_MUTEX: Mutex<()> = Mutex::const_new(());

/// The constant per-call token cost every response reports. Keeping this fixed
/// is what makes the savings identity hold: served-per-call ≡ saved-per-call.
const PER_CALL_TOTAL_TOKENS: u64 = 1280;

fn reset_global() {
    let _ = clear_plugin_configuration();
    let ctx = global_context();
    let mut state = ctx.write().unwrap();
    *state = NemoRelayContextState::new();
}

/// A provider stub that (a) counts how many times it actually runs and (b)
/// accumulates the `usage.total_tokens` it serves. The token add lives INSIDE
/// the closure, which the cache skips on a hit — so `served` naturally excludes
/// reused answers. `total_tokens` is read from the body itself, so the body is
/// the single source of truth for both what is served and what a hit reports as
/// saved.
fn counting_provider(
    calls: Arc<AtomicUsize>,
    served_tokens: Arc<AtomicU64>,
    body: Json,
) -> LlmExecutionNextFn {
    Arc::new(move |_req: LlmRequest| {
        let calls = Arc::clone(&calls);
        let served_tokens = Arc::clone(&served_tokens);
        let body = body.clone();
        Box::pin(async move {
            calls.fetch_add(1, Ordering::SeqCst);
            let tokens = body
                .pointer("/usage/total_tokens")
                .and_then(Json::as_u64)
                .unwrap_or(0);
            served_tokens.fetch_add(tokens, Ordering::SeqCst);
            Ok(body)
        })
    })
}

/// The canned response body, carrying deterministic token usage and cost. Its
/// `usage.total_tokens` is both what the stub accumulates as served and what a
/// hit mark reports as saved.
fn sample_body() -> Json {
    json!({
        "id": "resp_abc",
        "model": "gpt-4o",
        "choices": [{"message": {"role": "assistant", "content": "The answer is 42."}}],
        "usage": {
            "prompt_tokens": 1200,
            "completion_tokens": 80,
            "total_tokens": PER_CALL_TOTAL_TOKENS,
            "cost_usd": 0.0123
        }
    })
}

/// The response-cache config the benchmark tests use unless a test needs
/// otherwise: exact-match keying under a dedicated namespace, no sampled
/// bypass; keys are built from the auto-detected decode of each request.
fn bench_config() -> ResponseCacheConfig {
    ResponseCacheConfig {
        ttl_seconds: 3600,
        namespace: "bench_test".into(),
        bypass_rate: 0.0,
        ..Default::default()
    }
}

/// Tally of the `response_cache` marks observed during a workload, plus the
/// tokens the hit marks reported as saved. This mirrors exactly what the
/// benchmark accumulates from the mark stream.
#[derive(Debug, Default, Clone, Copy)]
struct CacheStats {
    hits: usize,
    misses: usize,
    bypasses: usize,
    saved_tokens: u64,
}

/// Registers a subscriber on the `response_cache` mark that folds every mark
/// into the shared [`CacheStats`]. Registration happens BEFORE the workload
/// because marks fire during the managed calls.
fn register_stats_subscriber(name: &str, stats: Arc<StdMutex<CacheStats>>) {
    register_subscriber(
        name,
        Arc::new(move |event: &Event| {
            if event.name() != "response_cache" {
                return;
            }
            let status = event
                .data()
                .and_then(|data| data.get("status"))
                .and_then(Json::as_str);
            let mut stats = stats.lock().unwrap();
            match status {
                Some("hit") => {
                    stats.hits += 1;
                    let saved = event
                        .metadata()
                        .and_then(|metadata| metadata.get("nemo_relay.response_cache.saved_tokens"))
                        .and_then(Json::as_u64)
                        .unwrap_or(0);
                    stats.saved_tokens += saved;
                }
                Some("miss") => stats.misses += 1,
                Some("bypass") => stats.bypasses += 1,
                _ => {}
            }
        }),
    )
    .unwrap();
}

/// A repeat workload: the first `distinct` prompts are unique, then the next
/// `repeats` prompts re-issue earlier ones (index `i % distinct`). Every request
/// uses the same provider (so served tokens accumulate on live calls only) and
/// the same shared stats subscriber.
async fn run_repeat_workload(provider: &LlmExecutionNextFn, distinct: usize, repeats: usize) {
    for i in 0..distinct {
        call(provider, chat_request(&format!("prompt #{i}"))).await;
    }
    for i in 0..repeats {
        let reused = i % distinct;
        call(provider, chat_request(&format!("prompt #{reused}"))).await;
    }
}

#[tokio::test]
async fn repeat_workload_yields_expected_hits_and_savings() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    activate_cache(bench_config()).await;

    let stats = Arc::new(StdMutex::new(CacheStats::default()));
    register_stats_subscriber("bench_repeat_savings", Arc::clone(&stats));

    let calls = Arc::new(AtomicUsize::new(0));
    let served_tokens = Arc::new(AtomicU64::new(0));
    let provider = counting_provider(
        Arc::clone(&calls),
        Arc::clone(&served_tokens),
        sample_body(),
    );

    // 4 distinct prompts, then 4 exact repeats of them -> exactly K = 4 hits.
    let distinct = 4;
    let repeats = 4;
    let total_requests = distinct + repeats;
    run_repeat_workload(&provider, distinct, repeats).await;
    flush_subscribers().unwrap();

    let stats = *stats.lock().unwrap();
    assert_eq!(
        stats.hits, repeats,
        "each of the K repeats must be served from cache as a hit"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        total_requests - repeats,
        "the provider must run exactly total_requests - hits times"
    );
    // Supporting invariants make a failure point at the cause, not just a total.
    assert_eq!(
        stats.hits + stats.misses,
        total_requests,
        "every request is either a hit or a miss (no bypasses here)"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        total_requests - stats.hits,
        "provider_calls_on == total_requests - hits"
    );
    assert!(
        stats.saved_tokens > 0,
        "hits must report a positive saved-token total"
    );
    assert_eq!(
        stats.saved_tokens,
        (repeats as u64) * PER_CALL_TOTAL_TOKENS,
        "saved_tokens == hits * per-call total_tokens"
    );
    assert_eq!(
        served_tokens.load(Ordering::SeqCst) + stats.saved_tokens,
        (total_requests as u64) * PER_CALL_TOTAL_TOKENS,
        "savings identity: served_tokens + saved_tokens == baseline_tokens"
    );

    deregister_subscriber("bench_repeat_savings").unwrap();
}

#[tokio::test]
async fn bypass_rate_one_disables_reuse() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    // bypass_rate = 1.0 forces every call live, even exact repeats.
    activate_cache(ResponseCacheConfig {
        bypass_rate: 1.0,
        ..bench_config()
    })
    .await;

    let stats = Arc::new(StdMutex::new(CacheStats::default()));
    register_stats_subscriber("bench_bypass", Arc::clone(&stats));

    let calls = Arc::new(AtomicUsize::new(0));
    let served_tokens = Arc::new(AtomicU64::new(0));
    let provider = counting_provider(
        Arc::clone(&calls),
        Arc::clone(&served_tokens),
        sample_body(),
    );

    // A workload full of repeats that would otherwise hit.
    let distinct = 2;
    let repeats = 4;
    let total_requests = distinct + repeats;
    run_repeat_workload(&provider, distinct, repeats).await;
    flush_subscribers().unwrap();

    let stats = *stats.lock().unwrap();
    // Load-bearing: no reuse, and the provider ran for every request.
    assert_eq!(
        stats.hits, 0,
        "bypass_rate = 1.0 must never serve a hit, even with repeats"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        total_requests,
        "every request must run live under full bypass"
    );
    assert_eq!(
        stats.bypasses, total_requests,
        "every request must be marked as a bypass"
    );
    assert_eq!(stats.saved_tokens, 0, "a bypass saves nothing");

    deregister_subscriber("bench_bypass").unwrap();
}

#[tokio::test]
async fn reinitialized_cache_starts_empty() {
    let _guard = TEST_MUTEX.lock().await;

    // Runs one repeat workload against a fresh cache and returns the observed
    // hit count. Each invocation resets the global state and re-activates the
    // cache, so `initialize_plugins_exact` builds a brand-new in-memory store.
    async fn run_once(subscriber: &str) -> usize {
        reset_global();
        activate_cache(bench_config()).await;

        let stats = Arc::new(StdMutex::new(CacheStats::default()));
        register_stats_subscriber(subscriber, Arc::clone(&stats));

        let calls = Arc::new(AtomicUsize::new(0));
        let served_tokens = Arc::new(AtomicU64::new(0));
        let provider = counting_provider(
            Arc::clone(&calls),
            Arc::clone(&served_tokens),
            sample_body(),
        );

        let distinct = 3;
        let repeats = 5;
        run_repeat_workload(&provider, distinct, repeats).await;
        flush_subscribers().unwrap();

        let hits = stats.lock().unwrap().hits;
        deregister_subscriber(subscriber).unwrap();
        hits
    }

    let first_run = run_once("bench_determinism_a").await;
    let second_run = run_once("bench_determinism_b").await;

    assert_eq!(
        first_run, second_run,
        "entries must not leak across plugin re-initialization: a leaked store \
         would let the second run hit on the first run's distinct prompts"
    );
}

#[tokio::test]
async fn warm_hits_stay_within_the_latency_budget() {
    let _guard = TEST_MUTEX.lock().await;
    reset_global();
    // Every warm hit exercises surface auto-detection — the default
    // configuration users actually run.
    activate_cache(ResponseCacheConfig {
        ttl_seconds: 3600,
        namespace: "bench_latency".into(),
        bypass_rate: 0.0,
        ..Default::default()
    })
    .await;

    let calls = Arc::new(AtomicUsize::new(0));
    let served = Arc::new(AtomicU64::new(0));
    let provider = counting_provider(Arc::clone(&calls), served, sample_body());
    let request = || chat_request("latency probe");

    // One miss warms the entry; every timed iteration must then be a hit.
    call(&provider, request()).await;

    const ITERATIONS: usize = 200;
    let mut timings_us: Vec<u128> = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let started = std::time::Instant::now();
        call(&provider, request()).await;
        timings_us.push(started.elapsed().as_micros());
    }

    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "every timed iteration must be served from cache"
    );

    // A warm hit is a key build plus a table lookup — around 0.2 ms measured.
    // The 2 ms median budget is deliberately generous so CI machines never
    // flake; at this fixture size it catches gross hit-path regressions — an
    // accidental provider call, lock contention, a blowup that scales with the
    // iteration count — not micro-costs like one extra clone of a small body.
    timings_us.sort_unstable();
    let p50 = timings_us[ITERATIONS / 2];
    assert!(
        p50 < 2_000,
        "warm-hit p50 must stay under the 2 ms budget, got {p50} µs"
    );
}
