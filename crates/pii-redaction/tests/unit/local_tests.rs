// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicUsize, Ordering};

use nemo_relay::plugin::{deregister_local_model_provider, register_local_model_provider_tracked};
use serde_json::json;

use super::*;

struct ProviderGuard {
    name: &'static str,
    registration_id: u64,
}

impl Drop for ProviderGuard {
    fn drop(&mut self) {
        let _ = deregister_local_model_provider(self.name, self.registration_id);
    }
}

fn backend(
    name: &'static str,
    callback: impl Fn(Json, Duration) -> PluginResult<Json> + Send + Sync + 'static,
) -> (ProviderGuard, CompiledLocalBackend) {
    let registration_id = register_local_model_provider_tracked(name, Arc::new(callback)).unwrap();
    let backend = CompiledLocalBackend::new(
        LocalBackendConfig {
            backend: Some(name.into()),
            ..LocalBackendConfig::default()
        },
        None,
    )
    .unwrap();
    (
        ProviderGuard {
            name,
            registration_id,
        },
        backend,
    )
}

fn alice_detector(request: Json, _timeout: Duration) -> PluginResult<Json> {
    let mut detections = Vec::new();
    for item in request["texts"]
        .as_array()
        .expect("provider request should contain texts")
    {
        let text_id = item["id"].as_u64().expect("text id should be an integer");
        let text = item["text"].as_str().expect("text should be a string");
        for (start, value) in text.match_indices("Alice") {
            detections.push(json!({
                "text_id": text_id,
                "start_utf8": start,
                "end_utf8": start + value.len(),
                "label": "given_name",
                "score": 0.99
            }));
        }
    }
    Ok(json!({"version": 1, "detections": detections}))
}

#[test]
fn applies_non_overlapping_utf8_byte_spans() {
    let (_provider, backend) = backend("local-test-utf8", |_, _| {
        Ok(json!({
            "version": 1,
            "detections": [
                {
                    "text_id": 0,
                    "start_utf8": 0,
                    "end_utf8": 5,
                    "label": "given_name",
                    "score": 0.99
                },
                {
                    "text_id": 0,
                    "start_utf8": 6,
                    "end_utf8": 12,
                    "label": "surname",
                    "score": 0.98
                }
            ]
        }))
    });

    assert_eq!(
        backend.sanitize_json(json!({"text": "José Rivera"})),
        json!({"text": "[REDACTED] [REDACTED]"})
    );
}

#[test]
fn malformed_or_overlapping_spans_fail_closed_for_the_batch() {
    let (_provider, backend) = backend("local-test-overlap", |_, _| {
        Ok(json!({
            "version": 1,
            "detections": [
                {
                    "text_id": 0,
                    "start_utf8": 0,
                    "end_utf8": 4,
                    "label": "name",
                    "score": 0.9
                },
                {
                    "text_id": 0,
                    "start_utf8": 3,
                    "end_utf8": 6,
                    "label": "name",
                    "score": 0.9
                }
            ]
        }))
    });

    assert_eq!(
        backend.sanitize_json(json!({"first": "secret", "second": "safe"})),
        json!({"first": "[REDACTED]", "second": "[REDACTED]"})
    );
}

#[test]
fn provider_errors_fail_closed_without_changing_unselected_paths() {
    let registration_id = register_local_model_provider_tracked(
        "local-test-failure",
        Arc::new(|_, _| Err(PluginError::RegistrationFailed("boom".into()))),
    )
    .unwrap();
    let _provider = ProviderGuard {
        name: "local-test-failure",
        registration_id,
    };
    let backend = CompiledLocalBackend::new(
        LocalBackendConfig {
            backend: Some("local-test-failure".into()),
            target_paths: vec!["/selected".into()],
            replacement: Some("[PRIVATE]".into()),
            ..LocalBackendConfig::default()
        },
        None,
    )
    .unwrap();

    assert_eq!(
        backend.sanitize_json(json!({
            "selected": "secret",
            "unselected": "preserve"
        })),
        json!({
            "selected": "[PRIVATE]",
            "unselected": "preserve"
        })
    );
}

#[test]
fn batches_provider_requests_and_preserves_no_detection_values() {
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&calls);
    let (_provider, backend) = backend("local-test-batching", move |request, _| {
        observed.fetch_add(1, Ordering::SeqCst);
        assert!(request["texts"].as_array().unwrap().len() <= MAX_BATCH_ITEMS);
        Ok(json!({"version": 1, "detections": []}))
    });
    let values = (0..(MAX_BATCH_ITEMS + 1))
        .map(|index| Json::String(format!("value-{index}")))
        .collect();

    let sanitized = backend.sanitize_json(Json::Array(values));

    assert_eq!(sanitized[0], "value-0");
    assert_eq!(
        sanitized[MAX_BATCH_ITEMS],
        format!("value-{MAX_BATCH_ITEMS}")
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn latency_budget_applies_to_the_entire_payload() {
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&calls);
    let registration_id = register_local_model_provider_tracked(
        "local-test-total-deadline",
        Arc::new(move |_, timeout| {
            observed.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(timeout + Duration::from_millis(5));
            Err(PluginError::RegistrationFailed("timed out".into()))
        }),
    )
    .unwrap();
    let _provider = ProviderGuard {
        name: "local-test-total-deadline",
        registration_id,
    };
    let backend = CompiledLocalBackend::new(
        LocalBackendConfig {
            backend: Some("local-test-total-deadline".into()),
            max_latency_ms: Some(10),
            ..LocalBackendConfig::default()
        },
        None,
    )
    .unwrap();
    let values = (0..(MAX_BATCH_ITEMS + 1))
        .map(|index| Json::String(format!("value-{index}")))
        .collect();

    let sanitized = backend.sanitize_json(Json::Array(values));

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(
        sanitized
            .as_array()
            .unwrap()
            .iter()
            .all(|value| value == "[REDACTED]")
    );
}

#[test]
fn oversized_text_is_redacted_without_calling_the_provider() {
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&calls);
    let (_provider, backend) = backend("local-test-oversized", move |_, _| {
        observed.fetch_add(1, Ordering::SeqCst);
        Ok(json!({"version": 1, "detections": []}))
    });

    assert_eq!(
        backend.sanitize_json(Json::String("x".repeat(MAX_TEXT_BYTES + 1))),
        Json::String("[REDACTED]".into())
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn oversized_text_does_not_shift_later_provider_results() {
    let (_provider, backend) = backend("local-test-oversized-middle", |_, _| {
        Ok(json!({"version": 1, "detections": []}))
    });

    assert_eq!(
        backend.sanitize_json(json!(["first", "x".repeat(MAX_TEXT_BYTES + 1), "third"])),
        json!(["first", "[REDACTED]", "third"])
    );
}

#[test]
fn payload_count_limit_fails_closed_without_unbounded_provider_calls() {
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&calls);
    let (_provider, backend) = backend("local-test-count-limit", move |_, _| {
        observed.fetch_add(1, Ordering::SeqCst);
        Ok(json!({"version": 1, "detections": []}))
    });
    let values = (0..(MAX_TEXTS_PER_PAYLOAD + 1))
        .map(|index| Json::String(format!("value-{index}")))
        .collect();

    let sanitized = backend.sanitize_json(Json::Array(values));

    assert_eq!(sanitized[0], "value-0");
    assert_eq!(sanitized[MAX_TEXTS_PER_PAYLOAD], "[REDACTED]");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        MAX_TEXTS_PER_PAYLOAD.div_ceil(MAX_BATCH_ITEMS)
    );
}

#[test]
fn payload_byte_limit_fails_closed_after_the_bounded_prefix() {
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&calls);
    let (_provider, backend) = backend("local-test-byte-limit", move |_, _| {
        observed.fetch_add(1, Ordering::SeqCst);
        Ok(json!({"version": 1, "detections": []}))
    });
    let accepted = MAX_PAYLOAD_TEXT_BYTES / MAX_TEXT_BYTES;
    let values = (0..=accepted)
        .map(|_| Json::String("x".repeat(MAX_TEXT_BYTES)))
        .collect();

    let sanitized = backend.sanitize_json(Json::Array(values));

    assert_eq!(sanitized[accepted - 1], "x".repeat(MAX_TEXT_BYTES));
    assert_eq!(sanitized[accepted], "[REDACTED]");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        MAX_PAYLOAD_TEXT_BYTES.div_ceil(MAX_BATCH_BYTES)
    );
}

#[test]
fn non_utf8_boundary_detection_fails_closed() {
    let (_provider, backend) = backend("local-test-utf8-boundary", |_, _| {
        Ok(json!({
            "version": 1,
            "detections": [{
                "text_id": 0,
                "start_utf8": 1,
                "end_utf8": 2,
                "label": "invalid",
                "score": 1.0
            }]
        }))
    });

    assert_eq!(
        backend.sanitize_json(json!("é")),
        Json::String("[REDACTED]".into())
    );
}

#[test]
fn local_policy_rejects_malformed_or_unbounded_values() {
    let registration_id = register_local_model_provider_tracked(
        "local-test-policy-bounds",
        Arc::new(|request, _| Ok(request)),
    )
    .unwrap();
    let _provider = ProviderGuard {
        name: "local-test-policy-bounds",
        registration_id,
    };

    for (config, expected) in [
        (
            LocalBackendConfig {
                backend: Some("x".repeat(MAX_LOCAL_MODEL_PROVIDER_VALUE_BYTES + 1)),
                ..LocalBackendConfig::default()
            },
            "local.backend",
        ),
        (
            LocalBackendConfig {
                backend: Some("local-test-policy-bounds".into()),
                target_paths: vec!["message".into()],
                ..LocalBackendConfig::default()
            },
            "valid JSON pointer",
        ),
        (
            LocalBackendConfig {
                backend: Some("local-test-policy-bounds".into()),
                target_paths: vec!["/bad~escape".into()],
                ..LocalBackendConfig::default()
            },
            "valid JSON pointer",
        ),
        (
            LocalBackendConfig {
                backend: Some("local-test-policy-bounds".into()),
                target_path_patterns: vec!["messages/*/content".into()],
                ..LocalBackendConfig::default()
            },
            "valid JSON-pointer pattern",
        ),
        (
            LocalBackendConfig {
                backend: Some("local-test-policy-bounds".into()),
                replacement: Some("x".repeat(MAX_LOCAL_MODEL_REPLACEMENT_BYTES + 1)),
                ..LocalBackendConfig::default()
            },
            "local.replacement",
        ),
        (
            LocalBackendConfig {
                backend: Some("local-test-policy-bounds".into()),
                model_id: Some("x".repeat(MAX_LOCAL_MODEL_PROVIDER_VALUE_BYTES + 1)),
                ..LocalBackendConfig::default()
            },
            "local.model_id",
        ),
        (
            LocalBackendConfig {
                backend: Some("local-test-policy-bounds".into()),
                min_score: Some(f64::NAN),
                ..LocalBackendConfig::default()
            },
            "local.min_score",
        ),
        (
            LocalBackendConfig {
                backend: Some("local-test-policy-bounds".into()),
                excluded_labels: vec!["NAME".into(), "NAME".into()],
                ..LocalBackendConfig::default()
            },
            "local.excluded_labels",
        ),
    ] {
        let error = CompiledLocalBackend::new(config, None)
            .err()
            .expect("invalid local policy should fail");
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[test]
fn local_policy_accepts_root_and_escaped_json_pointers() {
    assert!(is_valid_json_pointer(""));
    assert!(is_valid_json_pointer("/nested/a~1b/~0value"));
}

#[test]
fn target_path_patterns_match_one_segment_without_widening_exact_paths() {
    let registration_id =
        register_local_model_provider_tracked("local-test-path-patterns", Arc::new(alice_detector))
            .unwrap();
    let _provider = ProviderGuard {
        name: "local-test-path-patterns",
        registration_id,
    };
    let backend = CompiledLocalBackend::new(
        LocalBackendConfig {
            backend: Some("local-test-path-patterns".into()),
            target_paths: vec!["/exact".into()],
            target_path_patterns: vec!["/messages/*/content".into()],
            ..LocalBackendConfig::default()
        },
        None,
    )
    .unwrap();

    assert_eq!(
        backend.sanitize_json(json!({
            "exact": "Alice",
            "messages": [
                {"content": "Alice", "name": "Alice"},
                {"content": "Alice"}
            ],
            "nested": {"messages": [{"content": "Alice"}]}
        })),
        json!({
            "exact": "[REDACTED]",
            "messages": [
                {"content": "[REDACTED]", "name": "Alice"},
                {"content": "[REDACTED]"}
            ],
            "nested": {"messages": [{"content": "Alice"}]}
        })
    );
}

#[test]
fn request_codec_classifies_only_normalized_content_patterns() {
    let registration_id = register_local_model_provider_tracked(
        "local-test-openai-request",
        Arc::new(alice_detector),
    )
    .unwrap();
    let _provider = ProviderGuard {
        name: "local-test-openai-request",
        registration_id,
    };
    let backend = CompiledLocalBackend::new(
        LocalBackendConfig {
            backend: Some("local-test-openai-request".into()),
            target_path_patterns: vec![
                "/messages/*/content".into(),
                "/messages/*/content/*/text".into(),
            ],
            ..LocalBackendConfig::default()
        },
        Some("openai_chat".into()),
    )
    .unwrap();
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "Alice-model",
            "trace_id": "Alice-trace",
            "messages": [
                {"role": "system", "content": "Keep this policy"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Email Alice"},
                        {"type": "image_url", "image_url": {"url": "https://Alice.invalid"}}
                    ]
                }
            ]
        }),
    };
    let codec = backend
        .request_codec
        .as_ref()
        .expect("configured request codec should exist");
    let annotated = codec
        .decode(&request)
        .expect("OpenAI request should decode");
    let sanitized_annotated =
        sanitize_serializable(&backend, annotated).expect("annotated request should sanitize");
    codec
        .encode(&sanitized_annotated, &request)
        .expect("sanitized OpenAI request should encode");

    let sanitized = llm_sanitize_request_callback(backend)(request);

    assert_eq!(sanitized.content["model"], "Alice-model");
    assert_eq!(sanitized.content["trace_id"], "Alice-trace");
    assert_eq!(
        sanitized.content["messages"][0]["content"],
        "Keep this policy"
    );
    assert_eq!(
        sanitized.content["messages"][1]["content"][0]["text"],
        "Email [REDACTED]"
    );
    assert_eq!(
        sanitized.content["messages"][1]["content"][1]["image_url"]["url"],
        "https://Alice.invalid"
    );
}

#[test]
fn response_codec_classifies_message_content_without_touching_identity_fields() {
    let registration_id = register_local_model_provider_tracked(
        "local-test-openai-response",
        Arc::new(alice_detector),
    )
    .unwrap();
    let _provider = ProviderGuard {
        name: "local-test-openai-response",
        registration_id,
    };
    let backend = CompiledLocalBackend::new(
        LocalBackendConfig {
            backend: Some("local-test-openai-response".into()),
            target_path_patterns: vec!["/message".into(), "/message/*/text".into()],
            ..LocalBackendConfig::default()
        },
        Some("openai_chat".into()),
    )
    .unwrap();
    let response = json!({
        "id": "Alice-response",
        "model": "Alice-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello Alice"},
            "finish_reason": "stop"
        }],
        "vendor_trace": "Alice-trace"
    });

    let sanitized = backend
        .sanitize_response_with_codec(response)
        .expect("configured codec should sanitize the response");

    assert_eq!(sanitized["id"], "Alice-response");
    assert_eq!(sanitized["model"], "Alice-model");
    assert_eq!(sanitized["vendor_trace"], "Alice-trace");
    assert_eq!(
        sanitized["choices"][0]["message"]["content"],
        "Hello [REDACTED]"
    );
}

#[test]
fn request_codec_failure_replaces_the_observable_body() {
    let registration_id = register_local_model_provider_tracked(
        "local-test-invalid-openai-request",
        Arc::new(alice_detector),
    )
    .unwrap();
    let _provider = ProviderGuard {
        name: "local-test-invalid-openai-request",
        registration_id,
    };
    let backend = CompiledLocalBackend::new(
        LocalBackendConfig {
            backend: Some("local-test-invalid-openai-request".into()),
            target_path_patterns: vec!["/messages/*/content".into()],
            replacement: Some("[PRIVATE]".into()),
            ..LocalBackendConfig::default()
        },
        Some("openai_chat".into()),
    )
    .unwrap();
    let request = LlmRequest {
        headers: serde_json::Map::from_iter([(
            "x-provider-id".into(),
            Json::String("preserve-header".into()),
        )]),
        content: json!({
            "messages": "Alice cannot be decoded as an OpenAI message list",
            "vendor_trace": "Alice-trace"
        }),
    };

    let sanitized = llm_sanitize_request_callback(backend)(request);

    assert_eq!(sanitized.content, json!("[PRIVATE]"));
    assert_eq!(sanitized.headers["x-provider-id"], "preserve-header");
}

#[test]
fn request_codec_ambiguous_multi_message_edit_fails_closed() {
    let registration_id = register_local_model_provider_tracked(
        "local-test-ambiguous-openai-request",
        Arc::new(alice_detector),
    )
    .unwrap();
    let _provider = ProviderGuard {
        name: "local-test-ambiguous-openai-request",
        registration_id,
    };
    let backend = CompiledLocalBackend::new(
        LocalBackendConfig {
            backend: Some("local-test-ambiguous-openai-request".into()),
            target_path_patterns: vec!["/messages/*/content".into()],
            replacement: Some("[PRIVATE]".into()),
            ..LocalBackendConfig::default()
        },
        Some("openai_chat".into()),
    )
    .unwrap();
    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "messages": [
                {"role": "system", "content": "Alice owns this policy"},
                {"role": "user", "content": "Email Alice"}
            ]
        }),
    };

    let sanitized = llm_sanitize_request_callback(backend)(request);

    assert_eq!(sanitized.content, json!("[PRIVATE]"));
}

#[test]
fn response_codec_failure_replaces_the_observable_payload() {
    let registration_id = register_local_model_provider_tracked(
        "local-test-invalid-openai-response",
        Arc::new(alice_detector),
    )
    .unwrap();
    let _provider = ProviderGuard {
        name: "local-test-invalid-openai-response",
        registration_id,
    };
    let backend = CompiledLocalBackend::new(
        LocalBackendConfig {
            backend: Some("local-test-invalid-openai-response".into()),
            target_path_patterns: vec!["/message".into()],
            replacement: Some("[PRIVATE]".into()),
            ..LocalBackendConfig::default()
        },
        Some("openai_chat".into()),
    )
    .unwrap();
    let response = json!({
        "choices": "Alice cannot be decoded as an OpenAI response list",
        "vendor_trace": "Alice-trace"
    });

    let sanitized = llm_sanitize_response_callback(backend)(response);

    assert_eq!(sanitized, json!("[PRIVATE]"));
}

#[test]
fn host_policy_applies_score_threshold_and_label_exclusions() {
    let registration_id = register_local_model_provider_tracked(
        "local-test-detection-policy",
        Arc::new(|_, _| {
            Ok(json!({
                "version": 1,
                "detections": [
                    {
                        "text_id": 0,
                        "start_utf8": 0,
                        "end_utf8": 5,
                        "label": "LOW_SCORE",
                        "score": 0.49
                    },
                    {
                        "text_id": 0,
                        "start_utf8": 6,
                        "end_utf8": 11,
                        "label": "PRESERVE",
                        "score": 0.99
                    },
                    {
                        "text_id": 0,
                        "start_utf8": 12,
                        "end_utf8": 17,
                        "label": "REDACT",
                        "score": 0.99
                    }
                ]
            }))
        }),
    )
    .unwrap();
    let _provider = ProviderGuard {
        name: "local-test-detection-policy",
        registration_id,
    };
    let backend = CompiledLocalBackend::new(
        LocalBackendConfig {
            backend: Some("local-test-detection-policy".into()),
            min_score: Some(0.5),
            excluded_labels: vec!["PRESERVE".into()],
            ..LocalBackendConfig::default()
        },
        None,
    )
    .unwrap();

    assert_eq!(
        backend.sanitize_json(json!("first next1 final")),
        json!("first next1 [REDACTED]")
    );
}

#[test]
fn validates_filtered_detections_before_applying_host_policy() {
    let registration_id = register_local_model_provider_tracked(
        "local-test-filtered-invalid-span",
        Arc::new(|_, _| {
            Ok(json!({
                "version": 1,
                "detections": [{
                    "text_id": 0,
                    "start_utf8": 0,
                    "end_utf8": 999,
                    "label": "LOW_SCORE",
                    "score": 0.1
                }]
            }))
        }),
    )
    .unwrap();
    let _provider = ProviderGuard {
        name: "local-test-filtered-invalid-span",
        registration_id,
    };
    let backend = CompiledLocalBackend::new(
        LocalBackendConfig {
            backend: Some("local-test-filtered-invalid-span".into()),
            min_score: Some(0.5),
            ..LocalBackendConfig::default()
        },
        None,
    )
    .unwrap();

    assert_eq!(
        backend.sanitize_json(json!("preserve without detections")),
        json!("[REDACTED]")
    );
}

#[test]
fn enforces_detection_limit_for_each_text() {
    let registration_id = register_local_model_provider_tracked(
        "local-test-per-text-detection-limit",
        Arc::new(|_, _| {
            let detections = (0..=MAX_DETECTIONS_PER_TEXT)
                .map(|index| {
                    json!({
                        "text_id": 0,
                        "start_utf8": index,
                        "end_utf8": index + 1,
                        "label": "NAME",
                        "score": 0.9
                    })
                })
                .collect::<Vec<_>>();
            Ok(json!({"version": 1, "detections": detections}))
        }),
    )
    .unwrap();
    let _provider = ProviderGuard {
        name: "local-test-per-text-detection-limit",
        registration_id,
    };
    let backend = CompiledLocalBackend::new(
        LocalBackendConfig {
            backend: Some("local-test-per-text-detection-limit".into()),
            ..LocalBackendConfig::default()
        },
        None,
    )
    .unwrap();

    assert_eq!(
        backend.sanitize_json(json!(["x".repeat(MAX_DETECTIONS_PER_TEXT + 1), "second"])),
        json!(["[REDACTED]", "[REDACTED]"])
    );
}
