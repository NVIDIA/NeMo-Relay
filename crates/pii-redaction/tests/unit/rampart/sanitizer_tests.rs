// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for Rampart sanitizer behavior and concurrency.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use super::*;

struct NameDetector;

impl DetectionModel for NameDetector {
    fn detect(&self, texts: &[&str]) -> Result<Vec<Detection>, DetectionError> {
        Ok(texts
            .iter()
            .enumerate()
            .filter_map(|(text_index, text)| {
                text.find("José").map(|start| Detection {
                    text_index,
                    start_utf8: start,
                    end_utf8: start + "José".len(),
                    label: "GIVEN_NAME".into(),
                    score: 0.99,
                })
            })
            .collect())
    }
}

struct FailingDetector;

impl DetectionModel for FailingDetector {
    fn detect(&self, _texts: &[&str]) -> Result<Vec<Detection>, DetectionError> {
        Err(PluginError::Internal("model failure".into()).into())
    }
}

struct PayloadLimitedDetector;

impl DetectionModel for PayloadLimitedDetector {
    fn detect(&self, _texts: &[&str]) -> Result<Vec<Detection>, DetectionError> {
        Err(DetectionError::PayloadLimit)
    }
}

struct PanickingDetector;

impl DetectionModel for PanickingDetector {
    fn detect(&self, _texts: &[&str]) -> Result<Vec<Detection>, DetectionError> {
        panic!("model panic")
    }
}

struct CountingDetector(Arc<AtomicUsize>);

impl DetectionModel for CountingDetector {
    fn detect(&self, _texts: &[&str]) -> Result<Vec<Detection>, DetectionError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(Vec::new())
    }
}

struct CountingNameDetector(Arc<AtomicUsize>);

impl DetectionModel for CountingNameDetector {
    fn detect(&self, texts: &[&str]) -> Result<Vec<Detection>, DetectionError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        NameDetector.detect(texts)
    }
}

struct StaticDetector(Vec<Detection>);

impl DetectionModel for StaticDetector {
    fn detect(&self, _texts: &[&str]) -> Result<Vec<Detection>, DetectionError> {
        Ok(self.0.clone())
    }
}

struct TransientDetector(Arc<AtomicUsize>);

impl DetectionModel for TransientDetector {
    fn detect(&self, texts: &[&str]) -> Result<Vec<Detection>, DetectionError> {
        if self.0.fetch_add(1, Ordering::Relaxed) == 0 {
            Err(PluginError::Internal("transient model failure".into()).into())
        } else {
            NameDetector.detect(texts)
        }
    }
}

struct BatchLimitedNameDetector(Arc<AtomicUsize>);

impl DetectionModel for BatchLimitedNameDetector {
    fn detect(&self, texts: &[&str]) -> Result<Vec<Detection>, DetectionError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        if texts.len() > 1 {
            Err(DetectionError::PayloadLimit)
        } else {
            NameDetector.detect(texts)
        }
    }
}

struct BlockingDetector {
    started: Arc<AtomicBool>,
    release: Arc<AtomicBool>,
    finished: Arc<AtomicBool>,
}

impl DetectionModel for BlockingDetector {
    fn detect(&self, _texts: &[&str]) -> Result<Vec<Detection>, DetectionError> {
        self.started.store(true, Ordering::Release);
        while !self.release.load(Ordering::Acquire) {
            std::thread::sleep(Duration::from_millis(1));
        }
        self.finished.store(true, Ordering::Release);
        Ok(Vec::new())
    }
}

struct CountingBlockingDetector {
    started: Arc<AtomicUsize>,
    release: Arc<AtomicBool>,
}

impl DetectionModel for CountingBlockingDetector {
    fn detect(&self, _texts: &[&str]) -> Result<Vec<Detection>, DetectionError> {
        self.started.fetch_add(1, Ordering::Release);
        while !self.release.load(Ordering::Acquire) {
            std::thread::sleep(Duration::from_millis(1));
        }
        Ok(Vec::new())
    }
}

fn sanitizer(detector: Arc<dyn DetectionModel>, patterns: Vec<&str>) -> RampartSanitizer {
    RampartSanitizer::new(
        RampartPiiConfig {
            model_path: "/tmp/rampart".into(),
            target_path_patterns: patterns.into_iter().map(str::to_string).collect(),
            ..RampartPiiConfig::default()
        },
        detector,
    )
    .unwrap()
}

fn trajectory_sanitizer(
    detector: Arc<dyn DetectionModel>,
    custom_mark_payload_policy: &str,
) -> RampartSanitizer {
    RampartSanitizer::new(
        RampartPiiConfig {
            model_path: "/tmp/rampart".into(),
            preset: Some("trajectory_context".into()),
            custom_mark_payload_policy: custom_mark_payload_policy.into(),
            ..RampartPiiConfig::default()
        },
        detector,
    )
    .unwrap()
}

fn tool_end_event(annotation: Json) -> Event {
    use nemo_relay::api::event::{
        BaseEvent, CategoryProfile, EventCategory, ScopeCategory, ScopeEvent,
    };

    Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .name("lookup_contact")
            .data(serde_json::json!({"contact": "José result"}))
            .build(),
        ScopeCategory::End,
        Default::default(),
        EventCategory::tool(),
        Some(CategoryProfile {
            tool_call_id: Some("provider-call-José".into()),
            tool_result_annotation: Some(annotation),
            ..CategoryProfile::default()
        }),
    ))
}

#[tokio::test(flavor = "current_thread")]
async fn trajectory_preset_sanitizes_multi_message_anthropic_request_without_projection() {
    use nemo_relay::api::runtime::LlmSanitizeRequestContext;

    let backend = trajectory_sanitizer(Arc::new(NameDetector), "preserve");
    let request = LlmRequest {
        headers: Map::from_iter([("x-user-context".into(), Json::String("José header".into()))]),
        content: serde_json::json!({
            "model": "claude-José",
            "system": "Help José safely",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Initial prompt from José"}]
                },
                {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": "toolu_José",
                        "name": "read_file",
                        "input": {"path": "/tmp/José.txt"}
                    }]
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "toolu_José",
                        "content": "The file belongs to José"
                    }]
                }
            ],
            "tools": [{
                "name": "read_file",
                "description": "Read files for José",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "José's file path"}
                    }
                }
            }]
        }),
    };

    let sanitized = llm_sanitize_request_callback(backend)(
        request,
        LlmSanitizeRequestContext::with_identity(LlmCodecIdentity::BuiltIn(
            BuiltinLlmCodec::AnthropicMessages,
        )),
    )
    .await
    .unwrap()
    .expect("trajectory content should remain observable after sanitization");

    assert_eq!(sanitized.headers["x-user-context"], "[REDACTED] header");
    assert_eq!(sanitized.content["model"], "claude-José");
    assert_eq!(sanitized.content["system"], "Help [REDACTED] safely");
    assert_eq!(
        sanitized.content["messages"][0]["content"][0]["text"],
        "Initial prompt from [REDACTED]"
    );
    assert_eq!(
        sanitized.content["messages"][1]["content"][0]["name"],
        "read_file"
    );
    assert_eq!(
        sanitized.content["messages"][1]["content"][0]["id"],
        "toolu_José"
    );
    assert_eq!(
        sanitized.content["messages"][1]["content"][0]["input"]["path"],
        "/tmp/[REDACTED].txt"
    );
    assert_eq!(
        sanitized.content["messages"][2]["content"][0]["tool_use_id"],
        "toolu_José"
    );
    assert_eq!(
        sanitized.content["messages"][2]["content"][0]["content"],
        "The file belongs to [REDACTED]"
    );
    assert_eq!(sanitized.content["tools"][0]["name"], "read_file");
    assert_eq!(
        sanitized.content["tools"][0]["description"],
        "Read files for [REDACTED]"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn trajectory_preset_sanitizes_root_scalar_tool_results() {
    let output = tool_sanitize_callback(trajectory_sanitizer(Arc::new(NameDetector), "preserve"))(
        "read_file".into(),
        Json::String("Owned by José".into()),
    )
    .await
    .unwrap();

    assert_eq!(output, "Owned by [REDACTED]");
}

#[tokio::test(flavor = "current_thread")]
async fn trajectory_preset_preserves_unknown_custom_marks_by_default() {
    use nemo_relay::api::event::{BaseEvent, EventCategory, MarkEvent};

    let calls = Arc::new(AtomicUsize::new(0));
    let event = Arc::new(Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("application.checkpoint")
            .data(serde_json::json!({"message": "José"}))
            .metadata(serde_json::json!({"owner": "José"}))
            .build(),
        Some(EventCategory::custom()),
        None,
    )));
    let fields = event.sanitize_fields();
    let output = event_sanitize_callback(
        trajectory_sanitizer(Arc::new(CountingDetector(Arc::clone(&calls))), "preserve"),
        None,
    )(Arc::clone(&event), fields.clone())
    .await
    .unwrap();

    assert_eq!(output, fields);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn trajectory_preset_can_inspect_all_unknown_custom_mark_strings() {
    use nemo_relay::api::event::{BaseEvent, EventCategory, MarkEvent};

    let event = Arc::new(Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("application.checkpoint")
            .data(serde_json::json!({"id": "José"}))
            .metadata(serde_json::json!({"owner": "José"}))
            .build(),
        Some(EventCategory::custom()),
        None,
    )));
    let output = event_sanitize_callback(
        trajectory_sanitizer(Arc::new(NameDetector), "redact_all_leaves"),
        None,
    )(Arc::clone(&event), event.sanitize_fields())
    .await
    .unwrap();

    assert_eq!(output.data.unwrap()["id"], "[REDACTED]");
    assert_eq!(output.metadata.unwrap()["owner"], "[REDACTED]");
}

#[test]
fn sanitizes_selected_utf8_spans_without_touching_metadata() {
    let sanitizer = sanitizer(
        Arc::new(NameDetector),
        vec!["/messages/*/content", "/message"],
    );
    let value = serde_json::json!({
        "messages": [{"content": "Hello José Rivera"}],
        "message": "José",
        "model": "model-José"
    });
    assert_eq!(
        sanitizer.sanitize_json(value).unwrap(),
        serde_json::json!({
            "messages": [{"content": "Hello [REDACTED] Rivera"}],
            "message": "[REDACTED]",
            "model": "model-José"
        })
    );
}

#[test]
fn content_cache_deduplicates_within_and_across_payloads() {
    let calls = Arc::new(AtomicUsize::new(0));
    let sanitizer = sanitizer(
        Arc::new(CountingNameDetector(Arc::clone(&calls))),
        vec!["/*"],
    );
    let payload = serde_json::json!({
        "first": "Hello José",
        "second": "Hello José"
    });

    let first = sanitizer.sanitize_json(payload.clone()).unwrap();
    let second = sanitizer.sanitize_json(payload).unwrap();

    assert_eq!(first["first"], "Hello [REDACTED]");
    assert_eq!(first["second"], "Hello [REDACTED]");
    assert_eq!(second, first);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn content_cache_evicts_to_its_entry_bound() {
    let mut cache = SanitizationCache::default();
    for index in 0..=MAX_CACHE_ENTRIES {
        cache.insert(
            text_cache_key(&index.to_string()),
            SanitizationDecision::Keep,
        );
    }

    assert_eq!(cache.entries.len(), MAX_CACHE_ENTRIES);
    assert_eq!(cache.order.len(), MAX_CACHE_ENTRIES);
    assert_eq!(cache.decision_bytes, MAX_CACHE_ENTRIES);
}

#[test]
fn content_cache_gives_recently_read_entries_a_second_chance() {
    let mut cache = SanitizationCache::default();
    let first = text_cache_key("first");
    let second = text_cache_key("second");
    cache.insert(first, SanitizationDecision::Keep);
    cache.insert(second, SanitizationDecision::Keep);
    for index in 2..MAX_CACHE_ENTRIES {
        cache.insert(
            text_cache_key(&index.to_string()),
            SanitizationDecision::Keep,
        );
    }

    assert!(cache.get(&first).is_some());
    cache.insert(text_cache_key("overflow"), SanitizationDecision::Keep);

    assert!(cache.entries.contains_key(&first));
    assert!(!cache.entries.contains_key(&second));
    assert_eq!(cache.entries.len(), MAX_CACHE_ENTRIES);
}

#[test]
fn content_cache_rejects_a_single_oversized_decision() {
    let mut cache = SanitizationCache::default();
    let range_count = MAX_CACHE_DECISION_BYTES / std::mem::size_of::<(usize, usize)>() + 1;
    let ranges = vec![(0, 1); range_count];

    cache.insert(
        text_cache_key("oversized"),
        SanitizationDecision::Redact(ranges.into()),
    );

    assert!(cache.entries.is_empty());
    assert!(cache.order.is_empty());
    assert_eq!(cache.decision_bytes, 0);
}

#[test]
fn payload_limited_batch_splits_without_dropping_the_envelope() {
    let calls = Arc::new(AtomicUsize::new(0));
    let sanitizer = sanitizer(
        Arc::new(BatchLimitedNameDetector(Arc::clone(&calls))),
        vec!["/*"],
    );

    let sanitized = sanitizer
        .sanitize_json(serde_json::json!({
            "first": "José one",
            "second": "José two",
            "metadata": 7
        }))
        .unwrap();

    assert_eq!(sanitized["first"], "[REDACTED] one");
    assert_eq!(sanitized["second"], "[REDACTED] two");
    assert_eq!(sanitized["metadata"], 7);
    assert_eq!(calls.load(Ordering::Relaxed), 3);
}

#[test]
fn exact_selectors_match_escaped_json_pointer_segments() {
    let sanitizer = RampartSanitizer::new(
        RampartPiiConfig {
            model_path: "/tmp/rampart".into(),
            target_paths: vec!["/a~1b/c~0d".into()],
            ..RampartPiiConfig::default()
        },
        Arc::new(NameDetector),
    )
    .unwrap();
    let value = serde_json::json!({
        "a/b": {"c~d": "Hello José"},
        "a": {"b": {"c~d": "José"}}
    });

    assert_eq!(
        sanitizer.sanitize_json(value).unwrap(),
        serde_json::json!({
            "a/b": {"c~d": "Hello [REDACTED]"},
            "a": {"b": {"c~d": "José"}}
        })
    );
}

#[test]
fn model_errors_fail_closed_only_for_selected_values() {
    let sanitizer = sanitizer(Arc::new(FailingDetector), vec!["/message"]);
    assert_eq!(
        sanitizer
            .sanitize_json(serde_json::json!({
                "message": "private",
                "metadata": "visible"
            }))
            .unwrap(),
        serde_json::json!({
            "message": FailClosedReason::SanitizerFailure.placeholder(),
            "metadata": "visible"
        })
    );
}

#[test]
fn malformed_detector_output_always_fails_closed() {
    let cases = [
        (
            "out-of-range text index",
            "private",
            vec![Detection {
                text_index: 1,
                start_utf8: 0,
                end_utf8: 1,
                label: "GIVEN_NAME".into(),
                score: 0.99,
            }],
        ),
        (
            "non-finite score",
            "private",
            vec![Detection {
                text_index: 0,
                start_utf8: 0,
                end_utf8: 1,
                label: "GIVEN_NAME".into(),
                score: f64::NAN,
            }],
        ),
        (
            "out-of-range score",
            "private",
            vec![Detection {
                text_index: 0,
                start_utf8: 0,
                end_utf8: 1,
                label: "GIVEN_NAME".into(),
                score: 1.01,
            }],
        ),
        (
            "empty span",
            "private",
            vec![Detection {
                text_index: 0,
                start_utf8: 1,
                end_utf8: 1,
                label: "GIVEN_NAME".into(),
                score: 0.99,
            }],
        ),
        (
            "span past input",
            "private",
            vec![Detection {
                text_index: 0,
                start_utf8: 0,
                end_utf8: 100,
                label: "GIVEN_NAME".into(),
                score: 0.99,
            }],
        ),
        (
            "non-UTF-8 boundary",
            "José",
            vec![Detection {
                text_index: 0,
                start_utf8: 4,
                end_utf8: 5,
                label: "GIVEN_NAME".into(),
                score: 0.99,
            }],
        ),
        (
            "overlapping spans",
            "private",
            vec![
                Detection {
                    text_index: 0,
                    start_utf8: 0,
                    end_utf8: 4,
                    label: "GIVEN_NAME".into(),
                    score: 0.99,
                },
                Detection {
                    text_index: 0,
                    start_utf8: 3,
                    end_utf8: 7,
                    label: "SURNAME".into(),
                    score: 0.99,
                },
            ],
        ),
    ];

    for (name, text, detections) in cases {
        let sanitizer = sanitizer(Arc::new(StaticDetector(detections)), vec!["/message"]);
        let sanitized = sanitizer
            .sanitize_json(serde_json::json!({"message": text, "metadata": "visible"}))
            .unwrap();

        assert_eq!(
            sanitized["message"],
            FailClosedReason::SanitizerFailure.placeholder(),
            "{name}"
        );
        assert_eq!(sanitized["metadata"], "visible", "{name}");
    }
}

#[test]
fn transient_model_failures_are_retried_and_successes_are_cached() {
    let calls = Arc::new(AtomicUsize::new(0));
    let sanitizer = sanitizer(
        Arc::new(TransientDetector(Arc::clone(&calls))),
        vec!["/message"],
    );
    let payload = serde_json::json!({"message": "Hello José"});

    let failed = sanitizer.sanitize_json(payload.clone()).unwrap();
    let retried = sanitizer.sanitize_json(payload.clone()).unwrap();
    let cached = sanitizer.sanitize_json(payload).unwrap();

    assert_eq!(
        failed["message"],
        FailClosedReason::SanitizerFailure.placeholder()
    );
    assert_eq!(retried["message"], "Hello [REDACTED]");
    assert_eq!(cached, retried);
    assert_eq!(calls.load(Ordering::Relaxed), 2);
}

#[test]
fn sparse_selected_field_above_16_kib_is_sanitized() {
    let sanitizer = sanitizer(Arc::new(NameDetector), vec!["/message"]);
    let message = format!("{}José", " ".repeat(16 * 1024));
    let sanitized = sanitizer
        .sanitize_json(serde_json::json!({
            "message": message,
            "metadata": "visible"
        }))
        .unwrap();
    let sanitized_message = sanitized["message"].as_str().unwrap();

    assert_eq!(sanitized_message.len(), 16 * 1024 + "[REDACTED]".len());
    assert!(sanitized_message.ends_with("[REDACTED]"));
    assert_eq!(sanitized["metadata"], "visible");
}

#[test]
fn selected_text_count_limit_omits_only_excess_unique_fields() {
    let calls = Arc::new(AtomicUsize::new(0));
    let sanitizer = sanitizer(Arc::new(CountingDetector(Arc::clone(&calls))), vec!["/*"]);
    let value = Json::Object(
        (0..=MAX_TEXTS_PER_PAYLOAD)
            .map(|index| {
                (
                    index.to_string(),
                    Json::String(format!("safe-value-{index}")),
                )
            })
            .collect(),
    );
    let sanitized = sanitizer.sanitize_json(value).unwrap();
    let omitted = sanitized
        .as_object()
        .unwrap()
        .values()
        .filter(|value| value.as_str() == Some(FailClosedReason::PayloadLimit.placeholder()))
        .count();

    assert_eq!(omitted, 1);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn selected_payload_byte_limit_has_an_exact_boundary() {
    let calls = Arc::new(AtomicUsize::new(0));
    let sanitizer = sanitizer(
        Arc::new(CountingDetector(Arc::clone(&calls))),
        vec!["/message"],
    );
    let exact = " ".repeat(MAX_PAYLOAD_TEXT_BYTES);
    let sanitized = sanitizer
        .sanitize_json(serde_json::json!({"message": exact}))
        .unwrap();
    assert_eq!(
        sanitized["message"].as_str().unwrap().len(),
        MAX_PAYLOAD_TEXT_BYTES
    );
    assert_eq!(calls.load(Ordering::Relaxed), 1);

    assert_eq!(
        sanitizer
            .sanitize_json(serde_json::json!({
                "message": " ".repeat(MAX_PAYLOAD_TEXT_BYTES + 1)
            }))
            .unwrap()["message"],
        FailClosedReason::PayloadLimit.placeholder()
    );
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn aggregate_limit_omits_only_the_field_beyond_the_budget() {
    let calls = Arc::new(AtomicUsize::new(0));
    let backend = sanitizer(
        Arc::new(CountingDetector(Arc::clone(&calls))),
        vec!["/messages/*/content"],
    );
    let output = tool_sanitize_callback(backend)(
        "tool".into(),
        serde_json::json!({
            "messages": [
                {"content": "first-private-value"},
                {"content": " ".repeat(MAX_PAYLOAD_TEXT_BYTES)}
            ],
            "metadata": "visible"
        }),
    )
    .await
    .unwrap();

    assert_eq!(output["messages"][0]["content"], "first-private-value");
    assert_eq!(
        output["messages"][1]["content"],
        FailClosedReason::PayloadLimit.placeholder()
    );
    assert_eq!(output["metadata"], "visible");
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn selected_payload_uses_one_detector_call() {
    let calls = Arc::new(AtomicUsize::new(0));
    let sanitizer = sanitizer(Arc::new(CountingDetector(Arc::clone(&calls))), vec!["/*"]);
    let value = Json::Object(
        (0..128)
            .map(|index| (index.to_string(), Json::String("safe".into())))
            .collect(),
    );
    assert_eq!(
        sanitizer
            .sanitize_json(value)
            .unwrap()
            .as_object()
            .unwrap()
            .len(),
        128
    );
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn async_callback_does_not_block_the_runtime_thread() {
    let started = Arc::new(AtomicBool::new(false));
    let release = Arc::new(AtomicBool::new(false));
    let finished = Arc::new(AtomicBool::new(false));
    let backend = sanitizer(
        Arc::new(BlockingDetector {
            started: Arc::clone(&started),
            release: Arc::clone(&release),
            finished,
        }),
        vec!["/message"],
    );
    let callback = tool_sanitize_callback(backend);
    let task = tokio::spawn(callback(
        "tool".into(),
        serde_json::json!({"message": "private"}),
    ));

    tokio::time::timeout(Duration::from_secs(1), async {
        while !started.load(Ordering::Acquire) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("blocking detector should start");

    let heartbeat = Arc::new(AtomicUsize::new(0));
    let heartbeat_task = {
        let heartbeat = Arc::clone(&heartbeat);
        tokio::spawn(async move {
            for _ in 0..4 {
                tokio::time::sleep(Duration::from_millis(1)).await;
                heartbeat.fetch_add(1, Ordering::Relaxed);
            }
        })
    };
    heartbeat_task.await.unwrap();
    assert_eq!(heartbeat.load(Ordering::Relaxed), 4);

    release.store(true, Ordering::Release);
    assert_eq!(
        task.await.unwrap().unwrap(),
        serde_json::json!({"message": "private"})
    );
}

#[test]
fn bounded_fanout_does_not_block_the_runtime_thread() {
    let worker_count = inference_worker_count();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .max_blocking_threads(1)
        .build()
        .unwrap();
    runtime.block_on(async {
        let started = Arc::new(AtomicUsize::new(0));
        let release = Arc::new(AtomicBool::new(false));
        let backend = sanitizer(
            Arc::new(CountingBlockingDetector {
                started: Arc::clone(&started),
                release: Arc::clone(&release),
            }),
            vec!["/message"],
        );
        let callback = tool_sanitize_callback(backend);
        let mut tasks = Vec::new();
        for index in 0..MAX_ADMITTED_INFERENCE {
            tasks.push(tokio::spawn(callback(
                format!("tool-{index}"),
                serde_json::json!({"message": "private"}),
            )));
        }
        tokio::time::timeout(Duration::from_secs(1), async {
            while started.load(Ordering::Acquire) != worker_count {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the bounded model slots should start");

        let heartbeat = tokio::spawn(async {
            for _ in 0..4 {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        });
        heartbeat
            .await
            .expect("the runtime should remain responsive during fanout");

        release.store(true, Ordering::Release);
        for task in tasks {
            assert_eq!(
                task.await.unwrap().unwrap(),
                serde_json::json!({"message": "private"})
            );
        }
        assert_eq!(started.load(Ordering::Acquire), worker_count);
    });
}

#[test]
fn dedicated_executor_ignores_saturated_tokio_blocking_pool() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .max_blocking_threads(1)
        .build()
        .unwrap();
    runtime.block_on(async {
        let blocker_started = Arc::new(AtomicBool::new(false));
        let blocker_release = Arc::new(AtomicBool::new(false));
        let started = Arc::clone(&blocker_started);
        let release = Arc::clone(&blocker_release);
        let blocker = tokio::task::spawn_blocking(move || {
            started.store(true, Ordering::Release);
            while !release.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
        });
        tokio::time::timeout(Duration::from_secs(1), async {
            while !blocker_started.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the Tokio blocking-pool fixture should start");

        let calls = Arc::new(AtomicUsize::new(0));
        let callback = tool_sanitize_callback(sanitizer(
            Arc::new(CountingDetector(Arc::clone(&calls))),
            vec!["/message"],
        ));
        let output = tokio::time::timeout(
            Duration::from_millis(250),
            callback(
                "dedicated-executor".into(),
                serde_json::json!({"message": "private"}),
            ),
        )
        .await
        .expect("Rampart must not queue behind Tokio's blocking pool")
        .unwrap();
        assert_eq!(output, serde_json::json!({"message": "private"}));
        assert_eq!(calls.load(Ordering::Acquire), 1);

        blocker_release.store(true, Ordering::Release);
        blocker.await.unwrap();
    });
}

#[tokio::test(flavor = "current_thread")]
async fn inference_worker_panics_fail_closed_for_every_surface() {
    use nemo_relay::api::event::{BaseEvent, MarkEvent};
    use nemo_relay::api::runtime::{LlmSanitizeRequestContext, LlmSanitizeResponseContext};

    let backend = sanitizer(Arc::new(PanickingDetector), vec!["/message"]);
    let tool = tool_sanitize_callback(backend.clone());
    assert_eq!(
        tool(
            "tool".into(),
            serde_json::json!({"message": "private", "metadata": "visible"}),
        )
        .await
        .unwrap(),
        Json::String(FailClosedReason::SanitizerFailure.placeholder().into())
    );

    let event = Arc::new(Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("mark")
            .data(serde_json::json!({"message": "private"}))
            .metadata(serde_json::json!({"message": "private"}))
            .build(),
        None,
        None,
    )));
    let fields = event.sanitize_fields();
    assert_eq!(
        event_sanitize_callback(backend.clone(), None)(event, fields)
            .await
            .unwrap(),
        EventSanitizeFields::default()
    );

    let request = LlmRequest {
        headers: Map::new(),
        content: serde_json::json!({"message": "private"}),
    };
    assert!(
        llm_sanitize_request_callback(backend.clone())(
            request,
            LlmSanitizeRequestContext::default(),
        )
        .await
        .unwrap()
        .is_none()
    );
    assert!(
        llm_sanitize_response_callback(backend)(
            serde_json::json!({"message": "private"}),
            LlmSanitizeResponseContext::default(),
        )
        .await
        .unwrap()
        .is_none()
    );
}

#[tokio::test(flavor = "current_thread")]
async fn model_window_limit_fails_closed_only_for_affected_fields() {
    use nemo_relay::api::event::{BaseEvent, MarkEvent};
    use nemo_relay::api::runtime::{LlmSanitizeRequestContext, LlmSanitizeResponseContext};

    let backend = sanitizer(Arc::new(PayloadLimitedDetector), vec!["/message"]);
    let private = "must-not-pass-through";
    let tool = tool_sanitize_callback(backend.clone())(
        "tool".into(),
        serde_json::json!({"message": private, "metadata": "visible"}),
    )
    .await
    .unwrap();
    assert_eq!(
        tool["message"],
        FailClosedReason::ModelWindowLimit.placeholder()
    );
    assert_eq!(tool["metadata"], "visible");

    let event = Arc::new(Event::Mark(MarkEvent::new(
        BaseEvent::builder()
            .name("mark")
            .data(serde_json::json!({"message": private}))
            .metadata(serde_json::json!({"message": private}))
            .build(),
        None,
        None,
    )));
    let event_fields =
        event_sanitize_callback(backend.clone(), None)(Arc::clone(&event), event.sanitize_fields())
            .await
            .unwrap();
    assert_eq!(
        event_fields.data.unwrap()["message"],
        FailClosedReason::ModelWindowLimit.placeholder()
    );
    assert_eq!(
        event_fields.metadata.unwrap()["message"],
        FailClosedReason::ModelWindowLimit.placeholder()
    );

    let request = LlmRequest {
        headers: Map::new(),
        content: serde_json::json!({"message": private}),
    };
    let request = llm_sanitize_request_callback(backend.clone())(
        request,
        LlmSanitizeRequestContext::default(),
    )
    .await
    .unwrap()
    .expect("field-level failure should preserve the request envelope");
    assert_eq!(
        request.content["message"],
        FailClosedReason::ModelWindowLimit.placeholder()
    );
    let response = llm_sanitize_response_callback(backend.clone())(
        serde_json::json!({"message": private}),
        LlmSanitizeResponseContext::default(),
    )
    .await
    .unwrap()
    .expect("field-level failure should preserve the response envelope");
    assert_eq!(
        response["message"],
        FailClosedReason::ModelWindowLimit.placeholder()
    );

    let codec = build_response_codec(ProviderSurface::OpenAIChat);
    let payload = serde_json::json!({
        "id": "chatcmpl-payload-limit",
        "model": "model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": private},
            "finish_reason": "stop"
        }]
    });
    let sanitized = backend
        .sanitize_response_with_codec(codec.as_ref(), ProviderSurface::OpenAIChat, payload)
        .unwrap();
    assert_eq!(
        sanitized["choices"][0]["message"]["content"],
        FailClosedReason::ModelWindowLimit.placeholder()
    );
}

#[test]
fn bounded_admission_times_out_before_spawning_more_blocking_work() {
    let worker_count = inference_worker_count();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .max_blocking_threads(1)
        .build()
        .unwrap();
    runtime.block_on(async {
        let started = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let finished = Arc::new(AtomicBool::new(false));
        let backend = sanitizer(
            Arc::new(BlockingDetector {
                started: Arc::clone(&started),
                release: Arc::clone(&release),
                finished,
            }),
            vec!["/message"],
        );
        let execution = Arc::clone(&backend.execution_admission);
        let callback = tool_sanitize_callback(backend);
        let mut active = Vec::new();
        for index in 0..worker_count {
            active.push(tokio::spawn(callback(
                format!("active-{index}"),
                serde_json::json!({"message": "private"}),
            )));
        }
        tokio::time::timeout(Duration::from_secs(1), async {
            while !started.load(Ordering::Acquire) || execution.available_permits() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("blocking detector should start");

        let contending = tokio::time::timeout(
            MAX_ADMISSION_WAIT + Duration::from_millis(100),
            callback(
                "contending".into(),
                serde_json::json!({"message": "private", "metadata": "visible"}),
            ),
        )
        .await
        .expect("contending sanitizer should respect the admission deadline")
        .unwrap();
        assert_eq!(
            contending,
            Json::String(FailClosedReason::SanitizerFailure.placeholder().into())
        );

        release.store(true, Ordering::Release);
        for task in active {
            assert_eq!(
                task.await.unwrap().unwrap(),
                serde_json::json!({"message": "private"})
            );
        }
    });
}

#[tokio::test(flavor = "current_thread")]
async fn cancelled_callback_keeps_admission_until_blocking_work_finishes() {
    let started = Arc::new(AtomicBool::new(false));
    let release = Arc::new(AtomicBool::new(false));
    let finished = Arc::new(AtomicBool::new(false));
    let backend = sanitizer(
        Arc::new(BlockingDetector {
            started: Arc::clone(&started),
            release: Arc::clone(&release),
            finished: Arc::clone(&finished),
        }),
        vec!["/message"],
    );
    let execution = Arc::clone(&backend.execution_admission);
    let worker_count = execution.available_permits();
    let callback = tool_sanitize_callback(backend);
    let active = tokio::spawn(callback(
        "active".into(),
        serde_json::json!({"message": "private"}),
    ));
    tokio::time::timeout(Duration::from_secs(1), async {
        while !started.load(Ordering::Acquire) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("blocking detector should start");
    assert_eq!(execution.available_permits(), worker_count - 1);

    active.abort();
    assert!(active.await.unwrap_err().is_cancelled());
    assert_eq!(
        execution.available_permits(),
        worker_count - 1,
        "cancelling the async caller must not release an in-flight model slot"
    );

    release.store(true, Ordering::Release);
    tokio::time::timeout(Duration::from_secs(1), async {
        while !finished.load(Ordering::Acquire) || execution.available_permits() != worker_count {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("detached inference work should finish");
    assert_eq!(
        callback(
            "recovered".into(),
            serde_json::json!({"message": "private"}),
        )
        .await
        .unwrap(),
        serde_json::json!({"message": "private"})
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelled_queued_callback_releases_capacity_without_running() {
    let started = Arc::new(AtomicUsize::new(0));
    let release = Arc::new(AtomicBool::new(false));
    let backend = sanitizer(
        Arc::new(CountingBlockingDetector {
            started: Arc::clone(&started),
            release: Arc::clone(&release),
        }),
        vec!["/message"],
    );
    let admission = Arc::clone(&backend.admission_capacity);
    let execution = Arc::clone(&backend.execution_admission);
    let worker_count = execution.available_permits();
    let callback = tool_sanitize_callback(backend);
    let mut active = Vec::new();
    for index in 0..worker_count {
        active.push(tokio::spawn(callback(
            format!("active-{index}"),
            serde_json::json!({"message": "private"}),
        )));
    }
    tokio::time::timeout(Duration::from_secs(1), async {
        while started.load(Ordering::Acquire) != worker_count || execution.available_permits() != 0
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("both model slots should start");

    let queued = tokio::spawn(callback(
        "queued".into(),
        serde_json::json!({"message": "private"}),
    ));
    tokio::time::timeout(Duration::from_millis(50), async {
        while admission.available_permits() != MAX_ADMITTED_INFERENCE - worker_count - 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("the queued callback should reserve bounded capacity");
    queued.abort();
    assert!(queued.await.unwrap_err().is_cancelled());
    assert_eq!(
        admission.available_permits(),
        MAX_ADMITTED_INFERENCE - worker_count
    );
    assert_eq!(started.load(Ordering::Acquire), worker_count);

    release.store(true, Ordering::Release);
    for task in active {
        task.await.unwrap().unwrap();
    }
}

#[tokio::test(flavor = "current_thread")]
async fn full_admission_queue_fails_closed_without_running() {
    let calls = Arc::new(AtomicUsize::new(0));
    let backend = sanitizer(
        Arc::new(CountingDetector(Arc::clone(&calls))),
        vec!["/message"],
    );
    let mut permits = Vec::new();
    for _ in 0..MAX_ADMITTED_INFERENCE {
        permits.push(
            Arc::clone(&backend.admission_capacity)
                .try_acquire_owned()
                .unwrap(),
        );
    }
    let output = tool_sanitize_callback(backend)(
        "queue-full".into(),
        serde_json::json!({"message": "private"}),
    )
    .await
    .unwrap();
    assert_eq!(
        output,
        Json::String(FailClosedReason::SanitizerFailure.placeholder().into())
    );
    assert_eq!(calls.load(Ordering::Acquire), 0);
    drop(permits);
}

#[tokio::test(flavor = "current_thread")]
async fn unselected_tool_payload_bypasses_full_admission_queue() {
    let calls = Arc::new(AtomicUsize::new(0));
    let backend = sanitizer(
        Arc::new(CountingDetector(Arc::clone(&calls))),
        vec!["/message"],
    );
    let mut permits = Vec::new();
    for _ in 0..MAX_ADMITTED_INFERENCE {
        permits.push(
            Arc::clone(&backend.admission_capacity)
                .try_acquire_owned()
                .unwrap(),
        );
    }
    let payload = serde_json::json!({"trace_id": "visible"});
    let output = tool_sanitize_callback(backend)("unselected".into(), payload.clone())
        .await
        .unwrap();
    assert_eq!(output, payload);
    assert_eq!(calls.load(Ordering::Acquire), 0);
    drop(permits);
}

#[tokio::test(flavor = "current_thread")]
async fn unselected_specialized_metadata_bypasses_full_admission_queue() {
    use nemo_relay::api::event::{
        BaseEvent, CategoryProfile, EventCategory, ScopeCategory, ScopeEvent,
    };

    let calls = Arc::new(AtomicUsize::new(0));
    let backend = sanitizer(
        Arc::new(CountingDetector(Arc::clone(&calls))),
        vec!["/message"],
    );
    let mut permits = Vec::new();
    for _ in 0..MAX_ADMITTED_INFERENCE {
        permits.push(
            Arc::clone(&backend.admission_capacity)
                .try_acquire_owned()
                .unwrap(),
        );
    }
    let event = Arc::new(Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .name("tool")
            .metadata(serde_json::json!({"trace_id": "visible"}))
            .build(),
        ScopeCategory::Start,
        Default::default(),
        EventCategory::tool(),
        Some(CategoryProfile::default()),
    )));
    let fields = event.sanitize_fields();
    let output =
        event_sanitize_callback(backend, Some((false, true)))(Arc::clone(&event), fields.clone())
            .await
            .unwrap();
    assert_eq!(output, fields);
    assert_eq!(calls.load(Ordering::Acquire), 0);
    drop(permits);
}

#[tokio::test(flavor = "current_thread")]
async fn configured_tool_output_sanitizes_annotation_as_an_independent_json_root() {
    let backend = sanitizer(Arc::new(NameDetector), vec!["/contact"]);
    let event = Arc::new(tool_end_event(serde_json::json!({
        "contact": "José",
        "outside": "José"
    })));
    let output = event_sanitize_callback(backend, Some((false, true)))(
        Arc::clone(&event),
        event.sanitize_fields(),
    )
    .await
    .unwrap();

    assert_eq!(output.data, event.sanitize_fields().data);
    let profile = output.category_profile.unwrap();
    assert_eq!(profile.tool_call_id.as_deref(), Some("provider-call-José"));
    assert_eq!(
        profile.tool_result_annotation.unwrap(),
        serde_json::json!({
            "contact": "[REDACTED]",
            "outside": "José"
        })
    );
}

#[tokio::test(flavor = "current_thread")]
async fn trajectory_tool_output_sanitizes_every_annotation_string() {
    let backend = trajectory_sanitizer(Arc::new(NameDetector), "preserve");
    let event = Arc::new(tool_end_event(serde_json::json!({
        "contact": "José",
        "nested": ["Owned by José"]
    })));
    let output = event_sanitize_callback(backend, Some((false, true)))(
        Arc::clone(&event),
        event.sanitize_fields(),
    )
    .await
    .unwrap();

    assert_eq!(output.data, event.sanitize_fields().data);
    let profile = output.category_profile.unwrap();
    assert_eq!(profile.tool_call_id.as_deref(), Some("provider-call-José"));
    assert_eq!(
        profile.tool_result_annotation.unwrap(),
        serde_json::json!({
            "contact": "[REDACTED]",
            "nested": ["Owned by [REDACTED]"]
        })
    );
}

#[tokio::test(flavor = "current_thread")]
async fn disabled_tool_output_preserves_annotation_without_inference() {
    let calls = Arc::new(AtomicUsize::new(0));
    let backend = sanitizer(
        Arc::new(CountingNameDetector(Arc::clone(&calls))),
        vec!["/contact"],
    );
    let event = Arc::new(tool_end_event(serde_json::json!({"contact": "José"})));
    let fields = event.sanitize_fields();
    let output = event_sanitize_callback(backend, Some((true, false)))(event, fields.clone())
        .await
        .unwrap();

    assert_eq!(output, fields);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn failed_tool_annotation_inference_replaces_only_the_annotation() {
    let backend = sanitizer(Arc::new(PanickingDetector), vec!["/contact"]);
    let event = Arc::new(tool_end_event(serde_json::json!({"contact": "José"})));
    let output = event_sanitize_callback(backend, Some((false, true)))(
        Arc::clone(&event),
        event.sanitize_fields(),
    )
    .await
    .unwrap();

    assert_eq!(output.data, event.sanitize_fields().data);
    let profile = output.category_profile.unwrap();
    assert_eq!(profile.tool_call_id.as_deref(), Some("provider-call-José"));
    assert_eq!(
        profile.tool_result_annotation.unwrap(),
        Json::String(FailClosedReason::SanitizerFailure.placeholder().into())
    );
}

#[tokio::test(flavor = "current_thread")]
async fn tool_annotation_model_limit_replaces_only_affected_fields() {
    let backend = sanitizer(Arc::new(PayloadLimitedDetector), vec!["/contact"]);
    let event = Arc::new(tool_end_event(serde_json::json!({
        "contact": "José",
        "outside": "visible"
    })));
    let output = event_sanitize_callback(backend, Some((false, true)))(
        Arc::clone(&event),
        event.sanitize_fields(),
    )
    .await
    .unwrap();

    assert_eq!(output.data, event.sanitize_fields().data);
    let profile = output.category_profile.unwrap();
    assert_eq!(profile.tool_call_id.as_deref(), Some("provider-call-José"));
    assert_eq!(
        profile.tool_result_annotation.unwrap(),
        serde_json::json!({
            "contact": FailClosedReason::ModelWindowLimit.placeholder(),
            "outside": "visible"
        })
    );
}

#[test]
fn empty_specialized_scope_does_not_require_admission() {
    use nemo_relay::api::event::{
        BaseEvent, CategoryProfile, EventCategory, ScopeCategory, ScopeEvent,
    };

    let event = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .name("tool")
            .data(serde_json::json!({"message": "[REDACTED]"}))
            .build(),
        ScopeCategory::Start,
        Default::default(),
        EventCategory::tool(),
        Some(CategoryProfile::default()),
    ));
    let fields = event.sanitize_fields();
    assert!(!event_has_candidate_fields(&event, &fields, false));
    assert_eq!(
        fail_closed_event_fields(&event, fields.clone(), None),
        fields,
        "specialized data already handled by the tool sanitizer must be preserved"
    );
}

#[test]
fn openai_chat_request_projection_preserves_provider_fields() {
    let sanitizer = sanitizer(Arc::new(NameDetector), vec!["/messages/*/content"]);
    let request = LlmRequest {
        headers: Map::from_iter([("x-vendor".into(), Json::String("José-header".into()))]),
        content: serde_json::json!({
            "model": "model-José",
            "messages": [{"role": "user", "content": "Hello José"}],
            "vendor_trace": "trace-José"
        }),
    };
    let codec = build_request_codec(ProviderSurface::OpenAIChat);
    let sanitized = sanitizer
        .sanitize_request_with_codec(codec.as_ref(), &request)
        .unwrap();

    assert_eq!(
        sanitized.content["messages"][0]["content"],
        "Hello [REDACTED]"
    );
    assert_eq!(sanitized.content["model"], "model-José");
    assert_eq!(sanitized.content["vendor_trace"], "trace-José");
    assert_eq!(sanitized.headers["x-vendor"], "José-header");
}

#[test]
fn openai_chat_response_projection_preserves_provider_fields() {
    let sanitizer = sanitizer(Arc::new(NameDetector), vec!["/message"]);
    let payload = serde_json::json!({
        "id": "chatcmpl-José",
        "model": "model-José",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello José"},
            "finish_reason": "stop"
        }],
        "vendor_trace": "trace-José"
    });
    let codec = build_response_codec(ProviderSurface::OpenAIChat);
    let sanitized = sanitizer
        .sanitize_response_with_codec(codec.as_ref(), ProviderSurface::OpenAIChat, payload)
        .unwrap();

    assert_eq!(
        sanitized["choices"][0]["message"]["content"],
        "Hello [REDACTED]"
    );
    assert_eq!(sanitized["id"], "chatcmpl-José");
    assert_eq!(sanitized["model"], "model-José");
    assert_eq!(sanitized["vendor_trace"], "trace-José");
}

#[test]
fn gemini_request_projection_preserves_provider_fields() {
    let sanitizer = sanitizer(Arc::new(NameDetector), vec!["/messages/0/content"]);
    let request = LlmRequest {
        headers: Map::new(),
        content: serde_json::json!({
            "contents": [{
                "role": "user",
                "parts": [{"text": "Hello José"}]
            }],
            "vendorTrace": "trace-José"
        }),
    };
    let codec = build_request_codec(ProviderSurface::GeminiGenerateContent);
    let sanitized = sanitizer
        .sanitize_request_with_codec(codec.as_ref(), &request)
        .unwrap();

    assert_eq!(
        sanitized.content["contents"][0]["parts"][0]["text"],
        "Hello [REDACTED]"
    );
    assert_eq!(sanitized.content["vendorTrace"], "trace-José");
}

#[test]
fn gemini_response_projection_preserves_provider_fields() {
    let sanitizer = sanitizer(Arc::new(NameDetector), vec!["/message"]);
    let payload = serde_json::json!({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{"text": "Hello José"}]
            },
            "finishReason": "STOP"
        }],
        "modelVersion": "model-José",
        "vendorTrace": "trace-José"
    });
    let codec = build_response_codec(ProviderSurface::GeminiGenerateContent);
    let sanitized = sanitizer
        .sanitize_response_with_codec(
            codec.as_ref(),
            ProviderSurface::GeminiGenerateContent,
            payload,
        )
        .unwrap();

    assert_eq!(
        sanitized["candidates"][0]["content"]["parts"][0]["text"],
        "Hello [REDACTED]"
    );
    assert_eq!(sanitized["modelVersion"], "model-José");
    assert_eq!(sanitized["vendorTrace"], "trace-José");
}

#[tokio::test(flavor = "current_thread")]
async fn explicit_normalized_oci_request_and_response_fail_closed_before_inference() {
    use nemo_relay::api::runtime::{LlmSanitizeRequestContext, LlmSanitizeResponseContext};

    let calls = Arc::new(AtomicUsize::new(0));
    let backend = RampartSanitizer::new(
        RampartPiiConfig {
            model_path: "/tmp/rampart".into(),
            target_paths: vec!["/message".into()],
            target_path_patterns: vec!["/messages/*/content".into()],
            ..RampartPiiConfig::default()
        },
        Arc::new(CountingNameDetector(Arc::clone(&calls))),
    )
    .unwrap();
    let request = LlmRequest {
        headers: Map::new(),
        content: serde_json::json!({
            "chatRequest": {
                "apiFormat": "GENERIC",
                "messages": [{
                    "role": "USER",
                    "content": [{"type": "TEXT", "text": "Hello José"}]
                }]
            }
        }),
    };
    let response = serde_json::json!({
        "chatResponse": {
            "apiFormat": "GENERIC",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "ASSISTANT",
                    "content": [{"type": "TEXT", "text": "Hello José"}]
                },
                "finishReason": "stop"
            }]
        }
    });

    let sanitized_request = llm_sanitize_request_callback(backend.clone())(
        request,
        LlmSanitizeRequestContext::for_request_codec(Some(build_request_codec(
            ProviderSurface::OCIGenAI,
        ))),
    )
    .await
    .unwrap();
    let sanitized_response = llm_sanitize_response_callback(backend)(
        response,
        LlmSanitizeResponseContext::for_response_codec(Some(build_response_codec(
            ProviderSurface::OCIGenAI,
        ))),
    )
    .await
    .unwrap();

    assert!(sanitized_request.is_none());
    assert!(sanitized_response.is_none());
    assert_eq!(
        calls.load(Ordering::Relaxed),
        0,
        "unsupported normalized OCI projection must be rejected before model inference"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn trajectory_preset_sanitizes_raw_oci_shapes_without_projection() {
    use nemo_relay::api::runtime::{LlmSanitizeRequestContext, LlmSanitizeResponseContext};

    let backend = trajectory_sanitizer(Arc::new(NameDetector), "preserve");
    let request = LlmRequest {
        headers: Map::new(),
        content: serde_json::json!({
            "modelId": "unchanged-model",
            "vendorEnvelope": {"revision": 7},
            "chatRequest": {
                "apiFormat": "GENERIC",
                "messages": [{
                    "role": "USER",
                    "content": [
                        {"type": "TEXT", "text": "First message from José"},
                        {"type": "TEXT", "text": "Second message from José"}
                    ]
                }]
            }
        }),
    };
    let response = serde_json::json!({
        "modelId": "unchanged-model",
        "vendorEnvelope": {"revision": 9},
        "chatResponse": {
            "apiFormat": "GENERIC",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "ASSISTANT",
                        "content": [
                            {"type": "TEXT", "text": "First reply to José"},
                            {"type": "TEXT", "text": "Second reply to José"}
                        ]
                    },
                    "finishReason": "stop"
                },
                {
                    "index": 1,
                    "message": {
                        "role": "ASSISTANT",
                        "content": [{"type": "TEXT", "text": "Alternate reply to José"}]
                    },
                    "finishReason": "stop"
                }
            ]
        }
    });

    let sanitized_request = llm_sanitize_request_callback(backend.clone())(
        request,
        LlmSanitizeRequestContext::with_identity(LlmCodecIdentity::BuiltIn(
            BuiltinLlmCodec::OCIGenAI,
        )),
    )
    .await
    .unwrap()
    .expect("trajectory OCI request must remain observable");
    let sanitized_response = llm_sanitize_response_callback(backend)(
        response,
        LlmSanitizeResponseContext::with_identity(LlmCodecIdentity::BuiltIn(
            BuiltinLlmCodec::OCIGenAI,
        )),
    )
    .await
    .unwrap()
    .expect("trajectory OCI response must remain observable");

    assert_eq!(sanitized_request.content["modelId"], "unchanged-model");
    assert_eq!(sanitized_request.content["vendorEnvelope"]["revision"], 7);
    let request_parts = sanitized_request.content["chatRequest"]["messages"][0]["content"]
        .as_array()
        .expect("OCI request parts remain an array");
    assert_eq!(request_parts.len(), 2);
    assert_eq!(request_parts[0]["text"], "First message from [REDACTED]");
    assert_eq!(request_parts[1]["text"], "Second message from [REDACTED]");

    assert_eq!(sanitized_response["modelId"], "unchanged-model");
    assert_eq!(sanitized_response["vendorEnvelope"]["revision"], 9);
    let choices = sanitized_response["chatResponse"]["choices"]
        .as_array()
        .expect("OCI response choices remain an array");
    assert_eq!(choices.len(), 2);
    assert_eq!(
        choices[0]["message"]["content"].as_array().unwrap().len(),
        2
    );
    assert_eq!(
        choices[0]["message"]["content"][0]["text"],
        "First reply to [REDACTED]"
    );
    assert_eq!(
        choices[0]["message"]["content"][1]["text"],
        "Second reply to [REDACTED]"
    );
    assert_eq!(
        choices[1]["message"]["content"][0]["text"],
        "Alternate reply to [REDACTED]"
    );
}

#[test]
fn gemini_response_projection_omits_multiple_candidates_for_normalized_targets() {
    let sanitizer = sanitizer(Arc::new(NameDetector), vec!["/*"]);
    let payload = serde_json::json!({
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "Hello José"}]
                },
                "finishReason": "STOP"
            },
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "Private José"}]
                },
                "finishReason": "STOP"
            }
        ],
        "modelVersion": "gemini"
    });
    let codec = build_response_codec(ProviderSurface::GeminiGenerateContent);

    assert_eq!(
        sanitizer.sanitize_response_with_codec(
            codec.as_ref(),
            ProviderSurface::GeminiGenerateContent,
            payload,
        ),
        Err(SanitizeError::Codec)
    );
}

#[test]
fn openai_chat_response_projection_omits_multiple_choices_for_choice_targets() {
    let exact = RampartSanitizer::new(
        RampartPiiConfig {
            model_path: "/tmp/rampart".into(),
            target_paths: vec!["/message".into()],
            ..RampartPiiConfig::default()
        },
        Arc::new(NameDetector),
    )
    .unwrap();
    let wildcard = sanitizer(Arc::new(NameDetector), vec!["/*"]);
    let payload = serde_json::json!({
        "id": "chatcmpl-multi",
        "model": "model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello José"},
                "finish_reason": "stop"
            },
            {
                "index": 1,
                "message": {"role": "assistant", "content": "Private José"},
                "finish_reason": "stop"
            }
        ]
    });
    let codec = build_response_codec(ProviderSurface::OpenAIChat);

    assert_eq!(
        exact.sanitize_response_with_codec(
            codec.as_ref(),
            ProviderSurface::OpenAIChat,
            payload.clone(),
        ),
        Err(SanitizeError::Codec)
    );
    assert_eq!(
        wildcard
            .sanitize_response_with_codec(codec.as_ref(), ProviderSurface::OpenAIChat, payload,),
        Err(SanitizeError::Codec)
    );
}

#[test]
fn openai_chat_response_projection_keeps_multiple_choices_for_response_targets() {
    let sanitizer = sanitizer(Arc::new(NameDetector), vec!["/model"]);
    let payload = serde_json::json!({
        "id": "chatcmpl-multi",
        "model": "model-José",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "first"},
                "finish_reason": "stop"
            },
            {
                "index": 1,
                "message": {"role": "assistant", "content": "second"},
                "finish_reason": "stop"
            }
        ]
    });
    let codec = build_response_codec(ProviderSurface::OpenAIChat);
    let sanitized = sanitizer
        .sanitize_response_with_codec(codec.as_ref(), ProviderSurface::OpenAIChat, payload)
        .unwrap();

    assert_eq!(sanitized["model"], "model-[REDACTED]");
    assert_eq!(sanitized["choices"][1]["message"]["content"], "second");
}
