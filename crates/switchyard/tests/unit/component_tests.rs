// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use futures_util::{StreamExt, stream};
use nemo_relay::api::llm::LlmRequest;
use nemo_relay::api::runtime::{LlmExecutionNextFn, LlmJsonStream, LlmStreamExecutionNextFn};
use nemo_relay::error::{FlowError, UpstreamFailure, UpstreamFailureClass};
use serde_json::{Map, Value as Json, json};
use switchyard_libsy::{LlmResponse, Response as LibsyResponse, Step};
use switchyard_protocol::text_response;

use super::*;

fn binding(protocol: WireProtocol, model: &str, weight: f64) -> TargetBinding {
    TargetBinding {
        model: model.into(),
        protocol,
        endpoint: protocol.endpoint().into(),
        base_url: "https://provider.example.com/v1".into(),
        weight,
        headers: BTreeMap::new(),
        header_env: BTreeMap::new(),
    }
}

fn chat_config() -> SwitchyardConfig {
    SwitchyardConfig {
        algorithm: AlgorithmConfig::Random { seed: Some(42) },
        targets: BTreeMap::from([(
            "chat".into(),
            binding(WireProtocol::OpenaiChat, "provider/chat", 1.0),
        )]),
        default_targets: ProtocolDefaults {
            openai_chat: "chat".into(),
            ..ProtocolDefaults::default()
        },
        enabled_inbound_profiles: BTreeSet::from([WireProtocol::OpenaiChat]),
        ..SwitchyardConfig::default()
    }
}

fn chat_request() -> LlmRequest {
    LlmRequest {
        headers: Map::from_iter([
            (
                "x-nemo-relay-session-id".into(),
                Json::String("session-1".into()),
            ),
            (
                "authorization".into(),
                Json::String("Bearer caller-secret".into()),
            ),
        ]),
        content: json!({
            "model": "caller/model",
            "messages": [{"role": "user", "content": "hello"}],
            "provider_extension": {"preserve": true}
        }),
    }
}

fn chat_response(model: &str, text: &str) -> Json {
    json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "system_fingerprint": "fp_exact",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2
        }
    })
}

fn chat_chunk(model: &str, text: &str) -> Json {
    json!({
        "id": "chatcmpl-stream",
        "object": "chat.completion.chunk",
        "model": model,
        "system_fingerprint": "fp_stream_exact",
        "choices": [{
            "index": 0,
            "delta": {"content": text},
            "finish_reason": null
        }]
    })
}

fn provider_error(class: UpstreamFailureClass, status: Option<u16>) -> FlowError {
    FlowError::Upstream(UpstreamFailure {
        status,
        body: "provider failure".into(),
        headers: BTreeMap::new(),
        class,
    })
}

#[test]
fn version_one_configuration_has_a_migration_error() {
    let error = parse_config(&Map::from_iter([("version".into(), json!(1))])).unwrap_err();
    assert!(error.contains("removed switchyard-server Decision API"));
    assert!(error.contains("version = 2"));
}

#[test]
fn random_configuration_rejects_invalid_targets_and_weights() {
    let mut config = chat_config();
    config.targets.get_mut("chat").unwrap().endpoint = "/wrong".into();
    assert!(validate_config(&config).is_err());

    let mut config = chat_config();
    config.targets.get_mut("chat").unwrap().weight = f64::NAN;
    assert!(validate_config(&config).is_err());

    let mut config = chat_config();
    config.targets.get_mut("chat").unwrap().weight = 0.0;
    assert!(validate_config(&config).is_err());
}

#[tokio::test]
async fn run_stream_requires_relay_to_fulfill_the_provider_call() {
    let runtime = SwitchyardRuntime::new(chat_config()).unwrap();
    let request = runtime
        .libsy_request(WireProtocol::OpenaiChat, &chat_request(), false)
        .unwrap();
    let mut steps = runtime
        .algorithm
        .clone()
        .run_stream(LibsyContext::default(), request);

    let decision = steps.next().await.unwrap().unwrap();
    assert!(matches!(decision, Step::Decision(_)));

    let call = steps.next().await.unwrap().unwrap();
    let Step::CallLlm(call) = call else {
        panic!("expected Relay-hosted CallLlm step");
    };
    assert_eq!(call.get_decision().selected_model(), "chat");
    call.respond(Ok(LibsyResponse {
        llm_response: LlmResponse::Agg(text_response(Some("provider/chat".into()), "hello")),
        metadata: None,
    }))
    .unwrap();

    let returned = steps.next().await.unwrap().unwrap();
    assert!(matches!(returned, Step::ReturnToAgent(_)));
}

#[tokio::test]
async fn buffered_dispatch_uses_relay_binding_and_preserves_same_protocol_json() {
    let runtime = SwitchyardRuntime::new(chat_config()).unwrap();
    let captured = Arc::new(Mutex::new(Vec::new()));
    let capture = Arc::clone(&captured);
    let expected = chat_response("provider/chat", "routed");
    let provider_response = expected.clone();
    let next: LlmExecutionNextFn = Arc::new(move |request| {
        let capture = Arc::clone(&capture);
        let response = provider_response.clone();
        Box::pin(async move {
            capture.lock().unwrap().push(request);
            Ok(response)
        })
    });

    let response = runtime
        .execute_buffered("openai.chat_completions", chat_request(), next)
        .await
        .unwrap();
    assert_eq!(response, expected);

    let requests = captured.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].content["model"], "provider/chat");
    assert_eq!(requests[0].content["stream"], false);
    assert_eq!(
        requests[0].content["provider_extension"],
        json!({"preserve": true})
    );
    assert_eq!(
        requests[0].headers[INTERNAL_DISPATCH_URL_HEADER],
        "https://provider.example.com/v1/chat/completions"
    );
    assert!(!requests[0].headers.contains_key("authorization"));
    assert!(!requests[0].headers.contains_key("x-nemo-relay-session-id"));
}

#[tokio::test]
async fn cross_protocol_buffered_route_uses_switchyard_translation_both_ways() {
    let mut config = chat_config();
    config.targets.get_mut("chat").unwrap().weight = 0.0;
    config.targets.insert(
        "anthropic".into(),
        binding(WireProtocol::AnthropicMessages, "provider/anthropic", 1.0),
    );
    let runtime = SwitchyardRuntime::new(config).unwrap();
    let next: LlmExecutionNextFn = Arc::new(move |request| {
        Box::pin(async move {
            assert_eq!(request.content["model"], "provider/anthropic");
            assert_eq!(request.content["messages"][0]["role"], "user");
            assert!(request.content.get("input").is_none());
            Ok(json!({
                "id": "msg-test",
                "type": "message",
                "role": "assistant",
                "model": "provider/anthropic",
                "content": [{"type": "text", "text": "translated"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 2, "output_tokens": 1}
            }))
        })
    });

    let response = runtime
        .execute_buffered("openai.chat_completions", chat_request(), next)
        .await
        .unwrap();
    assert_eq!(response["choices"][0]["message"]["content"], "translated");
}

#[tokio::test]
async fn deterministic_seed_reproduces_the_same_target_sequence() {
    fn config() -> SwitchyardConfig {
        let mut config = chat_config();
        config.targets.insert(
            "other".into(),
            binding(WireProtocol::OpenaiChat, "provider/other", 1.0),
        );
        config
    }

    async fn sequence(runtime: &SwitchyardRuntime) -> Vec<String> {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let capture = Arc::clone(&captured);
        let next: LlmExecutionNextFn = Arc::new(move |request| {
            let capture = Arc::clone(&capture);
            Box::pin(async move {
                let model = request.content["model"].as_str().unwrap().to_string();
                capture.lock().unwrap().push(model.clone());
                Ok(chat_response(&model, "ok"))
            })
        });
        for _ in 0..12 {
            runtime
                .execute_buffered("openai.chat_completions", chat_request(), next.clone())
                .await
                .unwrap();
        }
        captured.lock().unwrap().clone()
    }

    let first = SwitchyardRuntime::new(config()).unwrap();
    let second = SwitchyardRuntime::new(config()).unwrap();
    let first_sequence = sequence(&first).await;
    let second_sequence = sequence(&second).await;
    assert_eq!(first_sequence, second_sequence);
    assert!(first_sequence.contains(&"provider/chat".to_string()));
    assert!(first_sequence.contains(&"provider/other".to_string()));
}

#[tokio::test]
async fn retryable_provider_failure_starts_a_fresh_libsy_run() {
    let mut config = chat_config();
    config.max_retries = 1;
    let runtime = SwitchyardRuntime::new(config).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let capture = Arc::clone(&calls);
    let next: LlmExecutionNextFn = Arc::new(move |request| {
        let capture = Arc::clone(&capture);
        Box::pin(async move {
            if capture.fetch_add(1, Ordering::SeqCst) == 0 {
                return Err(provider_error(
                    UpstreamFailureClass::RetryableStatus,
                    Some(503),
                ));
            }
            Ok(chat_response(
                request.content["model"].as_str().unwrap(),
                "retried",
            ))
        })
    });

    let response = runtime
        .execute_buffered("openai.chat_completions", chat_request(), next)
        .await
        .unwrap();
    assert_eq!(response["choices"][0]["message"]["content"], "retried");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn non_retryable_failure_dispatches_trusted_fallback_exactly_once() {
    let mut config = chat_config();
    config.targets.insert(
        "fallback".into(),
        binding(WireProtocol::OpenaiChat, "provider/fallback", 0.0),
    );
    config.default_targets.openai_chat = "fallback".into();
    let runtime = SwitchyardRuntime::new(config).unwrap();
    let models = Arc::new(Mutex::new(Vec::new()));
    let capture = Arc::clone(&models);
    let next: LlmExecutionNextFn = Arc::new(move |request| {
        let capture = Arc::clone(&capture);
        Box::pin(async move {
            let model = request.content["model"].as_str().unwrap().to_string();
            capture.lock().unwrap().push(model.clone());
            if model == "provider/chat" {
                return Err(provider_error(
                    UpstreamFailureClass::InvalidRequest,
                    Some(400),
                ));
            }
            Ok(chat_response(&model, "fallback"))
        })
    });

    let response = runtime
        .execute_buffered("openai.chat_completions", chat_request(), next)
        .await
        .unwrap();
    assert_eq!(response["choices"][0]["message"]["content"], "fallback");
    assert_eq!(
        *models.lock().unwrap(),
        vec!["provider/chat", "provider/fallback"]
    );
}

#[tokio::test]
async fn concurrent_runs_dispatch_independently() {
    let runtime = Arc::new(SwitchyardRuntime::new(chat_config()).unwrap());
    let calls = Arc::new(AtomicUsize::new(0));
    let next: LlmExecutionNextFn = {
        let calls = Arc::clone(&calls);
        Arc::new(move |request| {
            let calls = Arc::clone(&calls);
            Box::pin(async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(chat_response(
                    request.content["model"].as_str().unwrap(),
                    "ok",
                ))
            })
        })
    };
    let runs = (0..16).map(|_| {
        let runtime = Arc::clone(&runtime);
        let next = next.clone();
        tokio::spawn(async move {
            runtime
                .execute_buffered("openai.chat_completions", chat_request(), next)
                .await
        })
    });
    for result in futures_util::future::join_all(runs).await {
        assert!(result.unwrap().is_ok());
    }
    assert_eq!(calls.load(Ordering::SeqCst), 16);
}

#[tokio::test]
async fn streaming_prefetch_retries_before_commit() {
    let mut config = chat_config();
    config.max_retries = 1;
    let runtime = SwitchyardRuntime::new(config).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let capture = Arc::clone(&calls);
    let next: LlmStreamExecutionNextFn = Arc::new(move |_| {
        let capture = Arc::clone(&capture);
        Box::pin(async move {
            if capture.fetch_add(1, Ordering::SeqCst) == 0 {
                return Err(provider_error(UpstreamFailureClass::Connection, None));
            }
            Ok(LlmJsonStream::new(stream::iter([Ok(chat_chunk(
                "provider/chat",
                "retried",
            ))])))
        })
    });

    let output = runtime
        .execute_stream("openai.chat_completions", chat_request(), next)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(output.iter().all(Result::is_ok));
}

#[tokio::test]
async fn first_provider_error_event_retries_before_commit() {
    let mut config = chat_config();
    config.max_retries = 1;
    let runtime = SwitchyardRuntime::new(config).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let capture = Arc::clone(&calls);
    let next: LlmStreamExecutionNextFn = Arc::new(move |_| {
        let capture = Arc::clone(&capture);
        Box::pin(async move {
            if capture.fetch_add(1, Ordering::SeqCst) == 0 {
                return Ok(LlmJsonStream::new(stream::iter([Ok(json!({
                    "error": {"message": "temporary provider failure"}
                }))])));
            }
            Ok(LlmJsonStream::new(stream::iter([Ok(chat_chunk(
                "provider/chat",
                "retried",
            ))])))
        })
    });

    let output = runtime
        .execute_stream("openai.chat_completions", chat_request(), next)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(output.iter().all(Result::is_ok));
}

#[tokio::test]
async fn streaming_late_error_is_propagated_without_retry() {
    let runtime = SwitchyardRuntime::new(chat_config()).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let capture = Arc::clone(&calls);
    let next: LlmStreamExecutionNextFn = Arc::new(move |_| {
        let capture = Arc::clone(&capture);
        Box::pin(async move {
            capture.fetch_add(1, Ordering::SeqCst);
            Ok(LlmJsonStream::new(stream::iter([
                Ok(chat_chunk("provider/chat", "partial")),
                Err(provider_error(UpstreamFailureClass::Connection, None)),
            ])))
        })
    });

    let output = runtime
        .execute_stream("openai.chat_completions", chat_request(), next)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(output.first().is_some_and(Result::is_ok));
    assert!(output.last().is_some_and(Result::is_err));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn late_provider_error_event_is_propagated_without_retry() {
    let runtime = SwitchyardRuntime::new(chat_config()).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let capture = Arc::clone(&calls);
    let next: LlmStreamExecutionNextFn = Arc::new(move |_| {
        let capture = Arc::clone(&capture);
        Box::pin(async move {
            capture.fetch_add(1, Ordering::SeqCst);
            Ok(LlmJsonStream::new(stream::iter([
                Ok(chat_chunk("provider/chat", "partial")),
                Ok(json!({"error": {"message": "late provider failure"}})),
            ])))
        })
    });

    let output = runtime
        .execute_stream("openai.chat_completions", chat_request(), next)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(output.first().is_some_and(Result::is_ok));
    assert!(output.last().is_some_and(Result::is_err));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn cross_protocol_stream_uses_switchyard_translation_both_ways() {
    let mut config = chat_config();
    config.targets.get_mut("chat").unwrap().weight = 0.0;
    config.targets.insert(
        "anthropic".into(),
        binding(WireProtocol::AnthropicMessages, "provider/anthropic", 1.0),
    );
    let runtime = SwitchyardRuntime::new(config).unwrap();
    let next: LlmStreamExecutionNextFn = Arc::new(move |request| {
        Box::pin(async move {
            assert_eq!(request.content["model"], "provider/anthropic");
            Ok(LlmJsonStream::new(stream::iter([
                Ok(json!({
                    "type": "message_start",
                    "message": {
                        "id": "msg-stream",
                        "type": "message",
                        "role": "assistant",
                        "model": "provider/anthropic",
                        "content": [],
                        "usage": {"input_tokens": 2, "output_tokens": 0}
                    }
                })),
                Ok(json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""}
                })),
                Ok(json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "translated"}
                })),
                Ok(json!({"type": "content_block_stop", "index": 0})),
                Ok(json!({
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 1}
                })),
                Ok(json!({"type": "message_stop"})),
            ])))
        })
    });

    let output = runtime
        .execute_stream("openai.chat_completions", chat_request(), next)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(output.iter().all(Result::is_ok));
    assert!(
        output
            .iter()
            .filter_map(|result| result.as_ref().ok())
            .any(|event| {
                event["choices"][0]["delta"]["content"]
                    .as_str()
                    .is_some_and(|text| text == "translated")
            })
    );
}

#[tokio::test]
async fn same_protocol_stream_round_trip_preserves_raw_provider_event() {
    let runtime = SwitchyardRuntime::new(chat_config()).unwrap();
    let raw = chat_chunk("provider/chat", "exact");
    let provider_chunk = raw.clone();
    let next: LlmStreamExecutionNextFn = Arc::new(move |_| {
        let chunk = provider_chunk.clone();
        Box::pin(async move { Ok(LlmJsonStream::new(stream::iter([Ok(chunk)]))) })
    });

    let output = runtime
        .execute_stream("openai.chat_completions", chat_request(), next)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].as_ref().ok(), Some(&raw));
}

#[test]
fn plugin_source_does_not_use_relay_translation_codecs() {
    let component = include_str!("../../src/component.rs");
    let translation = include_str!("../../src/translation.rs");
    let stream_translation = include_str!("../../src/stream_translation.rs");
    for source in [component, translation, stream_translation] {
        assert!(!source.contains("nemo_relay::codec"));
    }
}
