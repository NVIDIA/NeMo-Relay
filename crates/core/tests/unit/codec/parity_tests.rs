// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cross-provider codec parity tests for the NeMo Relay core crate.
//!
//! Each test builds the same logical scenario in all three built-in provider
//! schemas (OpenAI Chat Completions, Anthropic Messages, OpenAI Responses) and
//! asserts that detection plus normalization produce agreeing output. Where
//! the schemas legitimately diverge, the divergence is asserted explicitly:
//! the asymmetry is part of the parity contract, and a change here means one
//! codec drifted from the others.

use serde_json::json;

use super::model_pricing::pricing_test_mutex;
use super::request::{GenerationParams, Message, MessageContent};
use super::resolve::{normalize_request, normalize_request_with_hint, normalize_response};
use super::response::{
    AnnotatedLlmResponse, ApiSpecificResponse, CostEstimate, CostSource, FinishReason,
    PricingCatalog, PricingResolver, Usage, reset_active_pricing_resolver,
    set_active_pricing_resolver,
};
use crate::api::llm::LlmRequest;
use crate::json::Json;

// -------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------

struct ResetPricingResolverGuard;

impl Drop for ResetPricingResolverGuard {
    fn drop(&mut self) {
        let _ = reset_active_pricing_resolver();
    }
}

fn install_parity_pricing(model_id: &str) {
    let catalog = PricingCatalog::from_json_str(
        &json!({
            "version": 1,
            "entries": [
                {
                    "provider": "test",
                    "model_id": model_id,
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

fn req(content: Json) -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content,
    }
}

fn decode(raw: &Json) -> AnnotatedLlmResponse {
    normalize_response(raw).unwrap_or_else(|| panic!("response should detect and decode: {raw}"))
}

/// The same logical response (one assistant text message) in each schema.
fn chat_text_response(model: &str) -> Json {
    json!({
        "id": "chatcmpl-parity",
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop"
        }]
    })
}

fn anthropic_text_response(model: &str) -> Json {
    json!({
        "id": "msg_parity",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn"
    })
}

fn responses_text_response(model: &str) -> Json {
    json!({
        "id": "resp_parity",
        "model": model,
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hello"}]
        }]
    })
}

/// The same logical usage (1000 prompt / 500 completion / 200 cache-read)
/// expressed in each schema's native usage shape. `extra_usage` merges extra
/// schema-specific usage keys into the payload.
fn chat_response_with_usage(model: &str, extra_usage: Json) -> Json {
    let mut raw = chat_text_response(model);
    let mut usage = json!({
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "total_tokens": 1500,
        "prompt_tokens_details": {"cached_tokens": 200}
    });
    merge_object(&mut usage, extra_usage);
    raw.as_object_mut().unwrap().insert("usage".into(), usage);
    raw
}

fn anthropic_response_with_usage(model: &str, extra_usage: Json) -> Json {
    let mut raw = anthropic_text_response(model);
    // Anthropic reports no total_tokens; the codec computes prompt + completion.
    let mut usage = json!({
        "input_tokens": 1000,
        "output_tokens": 500,
        "cache_read_input_tokens": 200
    });
    merge_object(&mut usage, extra_usage);
    raw.as_object_mut().unwrap().insert("usage".into(), usage);
    raw
}

fn responses_response_with_usage(model: &str, extra_usage: Json) -> Json {
    let mut raw = responses_text_response(model);
    let mut usage = json!({
        "input_tokens": 1000,
        "output_tokens": 500,
        "total_tokens": 1500,
        "input_tokens_details": {"cached_tokens": 200}
    });
    merge_object(&mut usage, extra_usage);
    raw.as_object_mut().unwrap().insert("usage".into(), usage);
    raw
}

fn merge_object(target: &mut Json, extra: Json) {
    if let (Some(target), Json::Object(extra)) = (target.as_object_mut(), extra) {
        target.extend(extra);
    }
}

// ===================================================================
// Response parity: model name and message text
// ===================================================================

#[test]
fn test_response_model_name_and_text_parity() {
    let chat = decode(&chat_text_response("parity-shared-model"));
    let anthropic = decode(&anthropic_text_response("parity-shared-model"));
    let responses = decode(&responses_text_response("parity-shared-model"));

    for decoded in [&chat, &anthropic, &responses] {
        assert_eq!(decoded.model.as_deref(), Some("parity-shared-model"));
        assert_eq!(decoded.response_text(), Some("hello"));
        assert_eq!(decoded.finish_reason, Some(FinishReason::Complete));
    }

    // Response IDs keep the provider-native value because IDs are
    // provider-scoped identifiers; only the field mapping is normalized.
    assert_eq!(chat.id.as_deref(), Some("chatcmpl-parity"));
    assert_eq!(anthropic.id.as_deref(), Some("msg_parity"));
    assert_eq!(responses.id.as_deref(), Some("resp_parity"));
}

// ===================================================================
// Response parity: finish reasons
// ===================================================================

#[test]
fn test_finish_reason_complete_parity() {
    let raws = [
        json!({"choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}]}),
        json!({"type": "message", "content": [{"type": "text", "text": "x"}], "stop_reason": "end_turn"}),
        json!({"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "x"}]}]}),
    ];
    for raw in &raws {
        assert_eq!(
            decode(raw).finish_reason,
            Some(FinishReason::Complete),
            "expected Complete for {raw}",
        );
    }
}

#[test]
fn test_finish_reason_length_parity() {
    let raws = [
        json!({"choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "length"}]}),
        json!({"type": "message", "content": [{"type": "text", "text": "x"}], "stop_reason": "max_tokens"}),
        json!({
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": []
        }),
    ];
    for raw in &raws {
        assert_eq!(
            decode(raw).finish_reason,
            Some(FinishReason::Length),
            "expected Length for {raw}",
        );
    }
}

#[test]
fn test_finish_reason_tool_use_parity_and_responses_divergence() {
    let chat = decode(&json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_parity_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"NYC\"}"}
                }]
            },
            "finish_reason": "tool_calls"
        }]
    }));
    let anthropic = decode(&json!({
        "type": "message",
        "content": [{
            "type": "tool_use",
            "id": "call_parity_1",
            "name": "get_weather",
            "input": {"city": "NYC"}
        }],
        "stop_reason": "tool_use"
    }));
    let responses = decode(&json!({
        "status": "completed",
        "output": [{
            "type": "function_call",
            "call_id": "call_parity_1",
            "name": "get_weather",
            "arguments": "{\"city\":\"NYC\"}"
        }]
    }));

    assert_eq!(chat.finish_reason, Some(FinishReason::ToolUse));
    assert_eq!(anthropic.finish_reason, Some(FinishReason::ToolUse));
    // The Responses API has no distinct tool-use terminal status: a
    // function_call turn still ends with status "completed", so the
    // normalized finish reason is Complete and tool-call presence must be
    // read from `tool_calls` instead.
    assert_eq!(responses.finish_reason, Some(FinishReason::Complete));
    assert!(responses.has_tool_calls());
}

// ===================================================================
// Response parity: tool calls
// ===================================================================

#[test]
fn test_response_tool_call_parity() {
    // One logical tool invocation. The id lives in a schema-specific field
    // (chat `tool_calls[].id`, Anthropic `tool_use.id`, Responses `call_id`),
    // and arguments arrive as a JSON string in the OpenAI schemas but as
    // parsed JSON in Anthropic's `input`. Normalization erases all of that.
    let chat = decode(&json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_parity_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"city\":\"NYC\",\"units\":\"c\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    }));
    let anthropic = decode(&json!({
        "type": "message",
        "content": [{
            "type": "tool_use",
            "id": "call_parity_1",
            "name": "get_weather",
            "input": {"city": "NYC", "units": "c"}
        }],
        "stop_reason": "tool_use"
    }));
    // The Responses item-level `id` is ignored; the cross-provider
    // correlation id is `call_id`.
    let responses = decode(&json!({
        "status": "completed",
        "output": [{
            "type": "function_call",
            "id": "fc_item_1",
            "call_id": "call_parity_1",
            "name": "get_weather",
            "arguments": "{\"city\":\"NYC\",\"units\":\"c\"}"
        }]
    }));

    let chat_calls = chat.tool_calls.expect("chat tool calls");
    let anthropic_calls = anthropic.tool_calls.expect("anthropic tool calls");
    let responses_calls = responses.tool_calls.expect("responses tool calls");

    assert_eq!(chat_calls, anthropic_calls);
    assert_eq!(chat_calls, responses_calls);
    assert_eq!(chat_calls.len(), 1);
    assert_eq!(chat_calls[0].id, "call_parity_1");
    assert_eq!(chat_calls[0].name, "get_weather");
    assert_eq!(
        chat_calls[0].arguments,
        json!({"city": "NYC", "units": "c"})
    );
    assert!(chat_calls[0].arguments.is_object());
}

// ===================================================================
// Response parity: usage
// ===================================================================

#[test]
fn test_response_usage_parity() {
    // The model name is unique to this test and never appears in any pricing
    // catalog, so estimation deterministically yields no cost.
    let chat = decode(&chat_response_with_usage("parity-usage-model", json!({})));
    let anthropic = decode(&anthropic_response_with_usage(
        "parity-usage-model",
        json!({}),
    ));
    let responses = decode(&responses_response_with_usage(
        "parity-usage-model",
        json!({}),
    ));

    let expected = Usage {
        prompt_tokens: Some(1000),
        completion_tokens: Some(500),
        total_tokens: Some(1500),
        cache_read_tokens: Some(200),
        cache_write_tokens: None,
        cost: None,
    };
    assert_eq!(chat.usage, Some(expected.clone()));
    // Anthropic supplies no total_tokens on the wire; the codec computes
    // prompt + completion so the normalized usage still matches the others.
    assert_eq!(anthropic.usage, Some(expected.clone()));
    assert_eq!(responses.usage, Some(expected));
}

#[test]
fn test_response_usage_schema_specific_extras() {
    let chat = decode(&chat_response_with_usage("parity-usage-model", json!({})));
    let anthropic = decode(&anthropic_response_with_usage(
        "parity-usage-model",
        json!({"cache_creation_input_tokens": 64}),
    ));
    let responses = decode(&responses_response_with_usage(
        "parity-usage-model",
        json!({"output_tokens_details": {"reasoning_tokens": 128}}),
    ));

    // Only Anthropic reports prompt-cache writes, so cache_write_tokens is
    // populated for Anthropic alone.
    assert_eq!(
        anthropic.usage.as_ref().unwrap().cache_write_tokens,
        Some(64)
    );
    assert_eq!(chat.usage.as_ref().unwrap().cache_write_tokens, None);
    assert_eq!(responses.usage.as_ref().unwrap().cache_write_tokens, None);

    // Only the Responses schema reports reasoning tokens; the normalized
    // Usage has no reasoning slot, so the value is preserved in the
    // schema-specific api_specific payload.
    match responses.api_specific.as_ref().unwrap() {
        ApiSpecificResponse::OpenAIResponses {
            output_tokens_details,
            ..
        } => {
            assert_eq!(
                output_tokens_details,
                &Some(json!({"reasoning_tokens": 128}))
            );
        }
        other => panic!("expected OpenAIResponses api_specific, got {other:?}"),
    }
    assert!(matches!(
        chat.api_specific,
        Some(ApiSpecificResponse::OpenAIChat { .. })
    ));
    assert!(matches!(
        anthropic.api_specific,
        Some(ApiSpecificResponse::AnthropicMessages { .. })
    ));

    // The shared usage facts still agree despite the schema-specific extras.
    for decoded in [&chat, &anthropic, &responses] {
        let usage = decoded.usage.as_ref().unwrap();
        assert_eq!(usage.prompt_tokens, Some(1000));
        assert_eq!(usage.completion_tokens, Some(500));
        assert_eq!(usage.total_tokens, Some(1500));
        assert_eq!(usage.cache_read_tokens, Some(200));
    }
}

// ===================================================================
// Response parity: cost
// ===================================================================

#[test]
fn test_provider_reported_cost_object_parity() {
    // Provider-reported cost always wins over estimation, so this test does
    // not depend on the active pricing resolver.
    let cost = json!({"cost": {
        "total": 0.0123,
        "input": 0.004,
        "output": 0.0083,
        "currency": "USD"
    }});
    let chat = decode(&chat_response_with_usage(
        "parity-reported-model",
        cost.clone(),
    ));
    let anthropic = decode(&anthropic_response_with_usage(
        "parity-reported-model",
        cost.clone(),
    ));
    let responses = decode(&responses_response_with_usage(
        "parity-reported-model",
        cost,
    ));

    let expected = CostEstimate {
        total: Some(0.0123),
        currency: "USD".to_string(),
        input: Some(0.004),
        output: Some(0.0083),
        cache_read: None,
        cache_write: None,
        source: CostSource::ProviderReported,
        pricing_provider: None,
        pricing_model: None,
        pricing_as_of: None,
        pricing_source: None,
    };
    assert_eq!(chat.usage.unwrap().cost, Some(expected.clone()));
    assert_eq!(anthropic.usage.unwrap().cost, Some(expected.clone()));
    assert_eq!(responses.usage.unwrap().cost, Some(expected));
}

#[test]
fn test_provider_reported_scalar_cost_parity() {
    // The legacy scalar `cost_usd` is accepted by all three schemas and is
    // always interpreted as a USD total.
    let cost = json!({"cost_usd": 0.5});
    let decoded = [
        decode(&chat_response_with_usage(
            "parity-reported-model",
            cost.clone(),
        )),
        decode(&anthropic_response_with_usage(
            "parity-reported-model",
            cost.clone(),
        )),
        decode(&responses_response_with_usage(
            "parity-reported-model",
            cost,
        )),
    ];
    for response in decoded {
        let cost = response.usage.unwrap().cost.expect("scalar reported cost");
        assert_eq!(cost.total, Some(0.5));
        assert_eq!(cost.currency, "USD");
        assert_eq!(cost.source, CostSource::ProviderReported);
        assert_eq!(cost.input, None);
        assert_eq!(cost.output, None);
    }
}

#[test]
fn test_estimated_cost_parity_for_identical_model_and_usage() {
    let _pricing_guard = pricing_test_mutex()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    install_parity_pricing("parity-priced-model");
    let _reset_guard = ResetPricingResolverGuard;

    // Each codec infers its own default provider ("openai" vs "anthropic"),
    // but pricing lookup falls back to the model-only key, so an identical
    // model + normalized usage must estimate identically everywhere.
    let chat = decode(&chat_response_with_usage("parity-priced-model", json!({})));
    let anthropic = decode(&anthropic_response_with_usage(
        "parity-priced-model",
        json!({}),
    ));
    let responses = decode(&responses_response_with_usage(
        "parity-priced-model",
        json!({}),
    ));

    let chat_cost = chat.usage.unwrap().cost.expect("chat estimated cost");
    let anthropic_cost = anthropic
        .usage
        .unwrap()
        .cost
        .expect("anthropic estimated cost");
    let responses_cost = responses
        .usage
        .unwrap()
        .cost
        .expect("responses estimated cost");

    assert_eq!(chat_cost, anthropic_cost);
    assert_eq!(chat_cost, responses_cost);
    // 800 billable prompt (1000 - 200 cached) * 0.15/M
    //   + 500 completion * 0.60/M + 200 cache-read * 0.075/M
    let total = chat_cost.total.expect("estimated total");
    assert!(
        (total - 0.000_435).abs() < 1e-9,
        "unexpected estimated total: {total}"
    );
    assert_eq!(chat_cost.currency, "USD");
    assert_eq!(chat_cost.source, CostSource::ModelPricing);
    assert_eq!(chat_cost.pricing_provider.as_deref(), Some("test"));
    assert_eq!(
        chat_cost.pricing_model.as_deref(),
        Some("parity-priced-model")
    );
}

// ===================================================================
// Request parity: detection and hint hardening
// ===================================================================

#[test]
fn test_request_hint_never_overrides_strong_signals() {
    // A wrong hint must not reroute a body whose shape is unambiguous.
    let responses_request = req(json!({
        "model": "gpt-parity",
        "instructions": "You are terse.",
        "input": "Summarize the docs.",
        "max_output_tokens": 64
    }));
    let hinted = normalize_request_with_hint(&responses_request, Some("anthropic"))
        .expect("responses request decodes despite wrong hint");
    assert_eq!(
        hinted,
        normalize_request(&responses_request).expect("responses request decodes"),
    );
    // Responses-only field proves the Responses codec handled the body.
    assert_eq!(hinted.max_output_tokens, Some(64));

    let anthropic_request = req(json!({
        "model": "claude-parity",
        "system": "You are terse.",
        "messages": [{"role": "user", "content": "Summarize the docs."}],
        "stop_sequences": ["END"]
    }));
    let hinted = normalize_request_with_hint(&anthropic_request, Some("openai.chat"))
        .expect("anthropic request decodes despite wrong hint");
    assert_eq!(
        hinted,
        normalize_request(&anthropic_request).expect("anthropic request decodes"),
    );
    // stop_sequences normalized into params (not left in extra) proves the
    // Anthropic codec handled the body.
    let stop = hinted
        .params
        .as_ref()
        .and_then(|params| params.stop.as_ref())
        .expect("anthropic stop_sequences normalized");
    assert_eq!(stop, &vec!["END".to_string()]);
    assert!(!hinted.extra.contains_key("stop_sequences"));
}

#[test]
fn test_request_unknown_hint_matches_hintless_normalization() {
    // Unrecognized hint strings are ignored: full decoded output (not just
    // surface detection) must match the hintless path for every schema.
    let bodies = [
        json!({"model": "m", "instructions": "sys", "input": "hi"}),
        json!({"model": "m", "system": "sys", "messages": [{"role": "user", "content": "hi"}]}),
        json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}),
    ];
    for body in &bodies {
        let request = req(body.clone());
        let baseline = normalize_request(&request).expect("canonical body decodes");
        for hint in ["gemini", "not-a-provider", "anthropic.count_tokens"] {
            assert_eq!(
                normalize_request_with_hint(&request, Some(hint)).as_ref(),
                Some(&baseline),
                "hint {hint:?} must not change normalization for {body}",
            );
        }
    }
}

#[test]
fn test_request_hint_none_equals_normalize_request() {
    // normalize_request_with_hint(None) is the same full decode as
    // normalize_request, for every canonical body shape.
    let bodies = [
        json!({"model": "m", "instructions": "sys", "input": "hi"}),
        json!({"model": "m", "system": "sys", "messages": [{"role": "user", "content": "hi"}]}),
        json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}),
    ];
    for body in &bodies {
        let request = req(body.clone());
        assert_eq!(
            normalize_request_with_hint(&request, None),
            normalize_request(&request),
            "hint=None must equal normalize_request for {body}",
        );
    }
}

// ===================================================================
// Request parity: normalization of an equivalent request
// ===================================================================

#[test]
fn test_request_normalization_parity() {
    let chat = normalize_request(&req(json!({
        "model": "parity-request-model",
        "messages": [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Summarize the docs."}
        ],
        "temperature": 0.5,
        "max_tokens": 256,
        "stop": ["END"]
    })))
    .expect("chat request decodes");

    let anthropic = normalize_request(&req(json!({
        "model": "parity-request-model",
        "system": "You are terse.",
        "messages": [{"role": "user", "content": "Summarize the docs."}],
        "temperature": 0.5,
        "max_tokens": 256,
        "stop_sequences": ["END"]
    })))
    .expect("anthropic request decodes");

    let responses = normalize_request(&req(json!({
        "model": "parity-request-model",
        "instructions": "You are terse.",
        "input": "Summarize the docs.",
        "temperature": 0.5,
        "max_output_tokens": 256
    })))
    .expect("responses request decodes");

    let expected_messages = vec![
        Message::System {
            content: MessageContent::Text("You are terse.".to_string()),
            name: None,
        },
        Message::User {
            content: MessageContent::Text("Summarize the docs.".to_string()),
            name: None,
        },
    ];
    for decoded in [&chat, &anthropic, &responses] {
        assert_eq!(decoded.messages, expected_messages);
        assert_eq!(decoded.model.as_deref(), Some("parity-request-model"));
        assert_eq!(decoded.system_prompt(), Some("You are terse."));
        assert_eq!(decoded.last_user_message(), Some("Summarize the docs."));
    }

    // Chat `max_tokens`/`stop` and Anthropic `max_tokens`/`stop_sequences`
    // normalize into identical GenerationParams.
    let expected_params = GenerationParams {
        temperature: Some(0.5),
        max_tokens: Some(256),
        top_p: None,
        stop: Some(vec!["END".to_string()]),
    };
    assert_eq!(chat.params, Some(expected_params.clone()));
    assert_eq!(anthropic.params, Some(expected_params));

    // Responses schema deviations: `max_output_tokens` maps into
    // params.max_tokens like the other schemas but is also preserved on the
    // Responses-only field, and the schema has no stop sequences, so
    // params.stop stays None.
    assert_eq!(
        responses.params,
        Some(GenerationParams {
            temperature: Some(0.5),
            max_tokens: Some(256),
            top_p: None,
            stop: None,
        })
    );
    assert_eq!(responses.max_output_tokens, Some(256));
    assert_eq!(chat.max_output_tokens, None);
    assert_eq!(anthropic.max_output_tokens, None);
}
