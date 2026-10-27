// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cross-provider codec parity tests for the NeMo Relay core crate.
//!
//! Each test builds the same logical scenario in the three OpenAI/Anthropic
//! built-in provider schemas (OpenAI Chat Completions, Anthropic Messages,
//! OpenAI Responses) and asserts that detection plus normalization produce
//! agreeing output. Gemini generateContent-specific semantics (contents array,
//! functionResponse role split, thinking tokens, etc.) are covered in
//! gemini_generate_content_tests.rs. Where the schemas legitimately diverge,
//! the divergence is asserted explicitly:
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

#[test]
fn owned_encoding_reuses_payloads_across_provider_codecs() {
    use super::anthropic::AnthropicMessagesCodec;
    use super::gemini_generate_content::GeminiGenerateContentCodec;
    use super::oci_genai::OCIGenAIChatCodec;
    use super::openai_responses::OpenAIResponsesCodec;
    use super::traits::LlmCodec;

    let payload = "A".repeat(1024 * 1024);
    let mut cases: Vec<(&dyn LlmCodec, Json, &str, &str)> = vec![
        (
            &OpenAIResponsesCodec,
            json!({"model": "gpt-4.1", "input": [{"role": "user", "content": [
                {"type": "input_image", "image_url": payload}
            ]}]}),
            "/input/0/content/0/image_url",
            "/temperature",
        ),
        (
            &AnthropicMessagesCodec,
            json!({"model": "claude-sonnet-4", "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": payload}}
            ]}]}),
            "/messages/0/content/0/source/data",
            "/temperature",
        ),
        (
            &GeminiGenerateContentCodec,
            json!({"contents": [{"role": "user", "parts": [
                {"inlineData": {"mimeType": "image/jpeg", "data": payload}}
            ]}]}),
            "/contents/0/parts/0/inlineData/data",
            "/generationConfig/temperature",
        ),
    ];
    // OCI accepts both an envelope and a bare body, with multiple API formats.
    for api_format in ["GENERIC", "COHERE", "COHEREV2"] {
        let (body, payload_path) = if api_format == "COHERE" {
            (
                json!({"apiFormat": api_format, "message": payload}),
                "/message",
            )
        } else {
            (
                json!({"apiFormat": api_format, "messages": [{"role": "USER", "content": [
                    {"type": "IMAGE", "imageUrl": {"url": payload}}
                ]}]}),
                "/messages/0/content/0/imageUrl/url",
            )
        };
        cases.push((
            &OCIGenAIChatCodec,
            body.clone(),
            payload_path,
            "/temperature",
        ));
        cases.push((
            &OCIGenAIChatCodec,
            json!({"compartmentId": "test", "servingMode": {"servingType": "ON_DEMAND", "modelId": "test"}, "chatRequest": body}),
            if api_format == "COHERE" { "/chatRequest/message" } else { "/chatRequest/messages/0/content/0/imageUrl/url" },
            "/chatRequest/temperature",
        ));
    }

    for (codec, content, payload_path, temperature_path) in cases {
        for change_temperature in [false, true] {
            let mut original = req(content.clone());
            original.headers.insert("x-test".into(), json!("preserve"));
            let payload_pointer = original
                .content
                .pointer(payload_path)
                .unwrap()
                .as_str()
                .unwrap()
                .as_ptr();
            let baseline = codec.decode(&original).unwrap();
            let mut annotated = baseline.clone();
            if change_temperature {
                annotated
                    .params
                    .get_or_insert_with(GenerationParams::default)
                    .temperature = Some(0.25);
            }
            let expected = codec
                .encode_with_baseline(&annotated, &original, &baseline)
                .unwrap();

            let encoded = codec
                .encode_owned_with_baseline(&annotated, original, &baseline)
                .unwrap();

            assert_eq!(encoded, expected);
            assert_eq!(encoded.headers["x-test"], json!("preserve"));
            if change_temperature {
                assert_eq!(
                    encoded.content.pointer(temperature_path),
                    Some(&json!(0.25))
                );
            } else {
                assert_eq!(encoded.content, content);
            }
            assert_eq!(
                encoded
                    .content
                    .pointer(payload_path)
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .as_ptr(),
                payload_pointer,
                "owned encoding copied the payload at {payload_path}"
            );
        }
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

    let openai_expected = Usage {
        prompt_tokens: Some(1000),
        completion_tokens: Some(500),
        total_tokens: Some(1500),
        cache_read_tokens: Some(200),
        cache_write_tokens: None,
        uncached_input_tokens: Some(800),
        cost: None,
    };
    assert_eq!(chat.usage, Some(openai_expected.clone()));
    assert_eq!(responses.usage, Some(openai_expected));

    // Anthropic supplies no total_tokens on the wire; the codec computes
    // prompt + completion. Unlike OpenAI, its input count excludes separately
    // reported cache reads, so the authoritative uncached count is 1,000.
    assert_eq!(
        anthropic.usage,
        Some(Usage {
            prompt_tokens: Some(1000),
            completion_tokens: Some(500),
            total_tokens: Some(1500),
            cache_read_tokens: Some(200),
            cache_write_tokens: None,
            uncached_input_tokens: Some(1000),
            cost: None,
        })
    );
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
    // model + request must estimate identically everywhere.
    // Anthropic counts cache reads separately from input tokens.
    let chat = decode(&chat_response_with_usage("parity-priced-model", json!({})));
    let anthropic = decode(&anthropic_response_with_usage(
        "parity-priced-model",
        json!({"input_tokens": 800}),
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
        for hint in [
            "gemini_generate_content",
            "not-a-provider",
            "anthropic.count_tokens",
        ] {
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
    assert_eq!(chat.messages, expected_messages);
    let expected_user_messages = vec![Message::User {
        content: MessageContent::Text("Summarize the docs.".to_string()),
        name: None,
    }];
    assert_eq!(anthropic.messages, expected_user_messages);
    assert_eq!(responses.messages, expected_user_messages);
    assert_eq!(
        anthropic.instructions,
        Some(MessageContent::Text("You are terse.".to_string()))
    );
    assert_eq!(anthropic.instructions, responses.instructions);
    for decoded in [&chat, &anthropic, &responses] {
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

#[test]
fn baseline_patching_keeps_raw_array_fields_with_reordered_logical_items() {
    let original = json!([
        {"type": "text", "text": "a", "provider_marker": "A"},
        {"type": "text", "text": "b", "provider_marker": "B"},
        {"type": "text", "text": "c", "provider_marker": "C"}
    ]);
    let baseline = json!([
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
        {"type": "text", "text": "c"}
    ]);
    let edited = json!([
        {"type": "text", "text": "b"},
        {"type": "text", "text": "a"},
        {"type": "text", "text": "c"}
    ]);

    assert_eq!(
        super::patch_changed_json(&original, &baseline, &edited).unwrap(),
        json!([
            {"type": "text", "text": "b", "provider_marker": "B"},
            {"type": "text", "text": "a", "provider_marker": "A"},
            {"type": "text", "text": "c", "provider_marker": "C"}
        ])
    );
}

#[test]
fn baseline_patching_does_not_pair_reordered_deletion_with_unrelated_insertion() {
    let original = json!([
        {"type": "text", "text": "a", "provider_marker": "A"},
        {"type": "text", "text": "b", "provider_marker": "B"}
    ]);
    let baseline = json!([
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"}
    ]);
    let edited = json!([
        {"type": "text", "text": "new"},
        {"type": "text", "text": "a"}
    ]);

    assert_eq!(
        super::patch_changed_json(&original, &baseline, &edited).unwrap(),
        json!([
            {"type": "text", "text": "new"},
            {"type": "text", "text": "a", "provider_marker": "A"}
        ])
    );
}

#[test]
fn baseline_patching_handles_insert_delete_and_duplicate_reorders() {
    let duplicate_a = json!({"type": "text", "text": "a"});
    let b = json!({"type": "text", "text": "b"});
    let original = json!([
        {"type": "text", "text": "a", "provider_marker": "A1"},
        {"type": "text", "text": "a", "provider_marker": "A2"},
        {"type": "text", "text": "b", "provider_marker": "B"}
    ]);
    let baseline = Json::Array(vec![duplicate_a.clone(), duplicate_a.clone(), b.clone()]);
    let edited = Json::Array(vec![
        duplicate_a.clone(),
        b,
        duplicate_a,
        json!({"type": "text", "text": "new"}),
    ]);

    assert_eq!(
        super::patch_changed_json(&original, &baseline, &edited).unwrap(),
        json!([
            {"type": "text", "text": "a", "provider_marker": "A1"},
            {"type": "text", "text": "b", "provider_marker": "B"},
            {"type": "text", "text": "a", "provider_marker": "A2"},
            {"type": "text", "text": "new"}
        ])
    );
}

#[test]
fn baseline_patching_accepts_multiple_replacements_without_provenance() {
    let original = json!([
        {"type": "text", "text": "a", "provider_marker": "A"},
        {"type": "text", "text": "b", "provider_marker": "B"}
    ]);
    let baseline = json!([
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"}
    ]);
    let edited = json!([
        {"type": "text", "text": "b-edited"},
        {"type": "text", "text": "a-edited"}
    ]);

    assert_eq!(
        super::patch_changed_json(&original, &baseline, &edited).unwrap(),
        edited
    );
}

// ===================================================================
// OCI GenAI parity: the fourth built-in surface agrees with the others
// ===================================================================

/// The same logical response (one assistant text message) in the OCI GenAI
/// GENERIC `ChatResult` schema.
fn oci_text_response(model: &str) -> Json {
    json!({
        "modelId": model,
        "chatResponse": {
            "apiFormat": "GENERIC",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "ASSISTANT",
                    "content": [{"type": "TEXT", "text": "hello"}]
                },
                "finishReason": "stop"
            }]
        }
    })
}

fn oci_response_with_usage(model: &str, extra_usage: Json) -> Json {
    let mut raw = oci_text_response(model);
    // OCI reports no cache token counters; only the three basic counters.
    let mut usage = json!({
        "promptTokens": 1000,
        "completionTokens": 500,
        "totalTokens": 1500
    });
    merge_object(&mut usage, extra_usage);
    raw["chatResponse"]
        .as_object_mut()
        .unwrap()
        .insert("usage".into(), usage);
    raw
}

#[test]
fn test_oci_response_model_name_and_text_parity() {
    let chat = decode(&chat_text_response("parity-shared-model"));
    let oci = decode(&oci_text_response("parity-shared-model"));

    assert_eq!(oci.model, chat.model);
    assert_eq!(oci.response_text(), chat.response_text());
    assert_eq!(oci.finish_reason, chat.finish_reason);
    // OCI ChatResult payloads carry no response id; the other schemas do.
    assert_eq!(oci.id, None);
}

#[test]
fn test_oci_finish_reason_parity() {
    // Complete: GENERIC "stop" and COHERE "COMPLETE" agree with the others.
    let generic_stop = json!({
        "chatResponse": {
            "apiFormat": "GENERIC",
            "choices": [{"message": {"role": "ASSISTANT", "content": []}, "finishReason": "stop"}]
        }
    });
    let cohere_complete = json!({
        "chatResponse": {"apiFormat": "COHERE", "text": "x", "finishReason": "COMPLETE"}
    });
    for raw in [&generic_stop, &cohere_complete] {
        assert_eq!(
            decode(raw).finish_reason,
            Some(FinishReason::Complete),
            "expected Complete for {raw}",
        );
    }

    // Length: GENERIC "length" and COHERE "MAX_TOKENS" agree with the others.
    let generic_length = json!({
        "chatResponse": {
            "apiFormat": "GENERIC",
            "choices": [{"message": {"role": "ASSISTANT", "content": []}, "finishReason": "length"}]
        }
    });
    let cohere_max_tokens = json!({
        "chatResponse": {"apiFormat": "COHERE", "text": "x", "finishReason": "MAX_TOKENS"}
    });
    for raw in [&generic_length, &cohere_max_tokens] {
        assert_eq!(
            decode(raw).finish_reason,
            Some(FinishReason::Length),
            "expected Length for {raw}",
        );
    }
}

#[test]
fn test_oci_response_tool_call_parity() {
    // The same logical tool invocation as test_response_tool_call_parity, in
    // the flat OCI GENERIC shape (string-encoded arguments like OpenAI Chat).
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
    let oci = decode(&json!({
        "chatResponse": {
            "apiFormat": "GENERIC",
            "choices": [{
                "message": {
                    "role": "ASSISTANT",
                    "content": [],
                    "toolCalls": [{
                        "id": "call_parity_1",
                        "type": "FUNCTION",
                        "name": "get_weather",
                        "arguments": "{\"city\":\"NYC\",\"units\":\"c\"}"
                    }]
                },
                "finishReason": "tool_calls"
            }]
        }
    }));

    assert_eq!(oci.finish_reason, Some(FinishReason::ToolUse));
    assert_eq!(oci.tool_calls, chat.tool_calls);
    let calls = oci.tool_calls.expect("oci tool calls");
    assert_eq!(calls[0].arguments, json!({"city": "NYC", "units": "c"}));
    assert!(calls[0].arguments.is_object());
}

#[test]
fn test_oci_response_usage_parity() {
    let chat = decode(&chat_response_with_usage("parity-usage-model", json!({})));
    let oci = decode(&oci_response_with_usage("parity-usage-model", json!({})));

    let chat_usage = chat.usage.unwrap();
    let oci_usage = oci.usage.unwrap();
    assert_eq!(oci_usage.prompt_tokens, chat_usage.prompt_tokens);
    assert_eq!(oci_usage.completion_tokens, chat_usage.completion_tokens);
    assert_eq!(oci_usage.total_tokens, chat_usage.total_tokens);
    // Divergence: OCI has no prompt-cache counters, so cache_read_tokens is
    // None where the OpenAI Chat fixture reports 200 cached tokens.
    assert_eq!(oci_usage.cache_read_tokens, None);
    assert_eq!(chat_usage.cache_read_tokens, Some(200));
    assert!(matches!(
        oci.api_specific,
        Some(ApiSpecificResponse::OCIGenAI { .. })
    ));
}

#[test]
fn test_oci_request_normalization_parity() {
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

    let oci = normalize_request(&req(json!({
        "compartmentId": "ocid1.compartment.oc1..parity",
        "servingMode": {"servingType": "ON_DEMAND", "modelId": "parity-request-model"},
        "chatRequest": {
            "apiFormat": "GENERIC",
            "messages": [
                {"role": "SYSTEM", "content": [{"type": "TEXT", "text": "You are terse."}]},
                {"role": "USER", "content": [{"type": "TEXT", "text": "Summarize the docs."}]}
            ],
            "temperature": 0.5,
            "maxTokens": 256,
            "stop": ["END"]
        }
    })))
    .expect("oci request decodes");

    // UPPERCASE roles and TEXT content-part lists normalize to the same
    // messages OpenAI Chat produces; the model comes from servingMode.
    assert_eq!(oci.messages, chat.messages);
    assert_eq!(oci.params, chat.params);
    assert_eq!(oci.model, chat.model);
    assert_eq!(oci.system_prompt(), chat.system_prompt());
    assert_eq!(oci.last_user_message(), chat.last_user_message());
}

#[test]
fn test_oci_request_hint_never_overrides_strong_signals() {
    // The "oci" hint must not reroute unambiguous non-OCI bodies.
    let chat_request = req(json!({
        "model": "gpt-parity",
        "messages": [{"role": "user", "content": "hi"}]
    }));
    let hinted = normalize_request_with_hint(&chat_request, Some("oci"))
        .expect("chat request decodes despite oci hint");
    assert_eq!(
        hinted,
        normalize_request(&chat_request).expect("chat request decodes"),
    );
    // And the reverse: a strong OCI envelope decodes as OCI even with a
    // wrong hint.
    let oci_request = req(json!({
        "compartmentId": "ocid1.compartment.oc1..parity",
        "servingMode": {"servingType": "ON_DEMAND", "modelId": "m"},
        "chatRequest": {
            "apiFormat": "GENERIC",
            "messages": [{"role": "USER", "content": [{"type": "TEXT", "text": "hi"}]}]
        }
    }));
    let hinted = normalize_request_with_hint(&oci_request, Some("anthropic"))
        .expect("oci request decodes despite wrong hint");
    assert_eq!(
        hinted,
        normalize_request(&oci_request).expect("oci request decodes"),
    );
    assert!(matches!(
        hinted.api_specific,
        Some(super::request::ApiSpecificRequest::OCIGenAI { .. })
    ));
}

#[test]
fn container_edits_preserve_matches_and_replace_unmatched_values() {
    let original = json!({"items": [
        {"role": "tool", "tool_call_id": "a", "content": "old", "vendor": "A"},
        {"role": "tool", "tool_call_id": "b", "content": "old", "vendor": "B"},
        {"role": "tool", "tool_call_id": "c", "content": "old", "vendor": "C"}
    ], "mapping": {"a": 1, "b": 2, "vendor": true}, "untouched": null});
    let baseline = json!({"items": [
        {"role": "tool", "tool_call_id": "a", "content": "old"},
        {"role": "tool", "tool_call_id": "b", "content": "old"},
        {"role": "tool", "tool_call_id": "c", "content": "old"}
    ], "mapping": {"a": 1, "b": 2}, "untouched": null});
    let edited = json!({"items": [
        {"role": "tool", "tool_call_id": "c", "content": "blocked"},
        {"role": "tool", "tool_call_id": "b", "content": "blocked"},
        {"role": "tool", "tool_call_id": "d", "content": "new"}
    ], "mapping": {"a": [1, 2], "c": null}, "untouched": null});
    assert_eq!(
        super::patch_changed_json(&original, &baseline, &edited).unwrap(),
        json!({
            "items": [
                {"role": "tool", "tool_call_id": "c", "content": "blocked", "vendor": "C"},
                {"role": "tool", "tool_call_id": "b", "content": "blocked", "vendor": "B"},
                {"role": "tool", "tool_call_id": "d", "content": "new"}
            ], "mapping": {"a": [1, 2], "c": null, "vendor": true}, "untouched": null
        })
    );
    for replacement in [
        json!([]),
        json!([null, {"id": "arbitrary"}, [1, 2]]),
        json!(null),
    ] {
        assert_eq!(
            super::patch_changed_json(&original["items"], &baseline["items"], &replacement)
                .unwrap(),
            replacement
        );
    }
}

#[test]
fn duplicate_or_changed_tool_ids_do_not_inherit_unrelated_fields() {
    let original = json!([
        {"role": "tool", "tool_call_id": "a", "content": "old", "vendor": "A"},
        {"role": "tool", "tool_call_id": "a", "content": "old", "vendor": "B"}
    ]);
    let baseline = json!([
        {"role": "tool", "tool_call_id": "a", "content": "old"},
        {"role": "tool", "tool_call_id": "a", "content": "old"}
    ]);
    for edited in [
        json!([
            {"role": "tool", "tool_call_id": "a", "content": "one"},
            {"role": "tool", "tool_call_id": "a", "content": "two"}
        ]),
        json!([{ "role": "tool", "tool_call_id": "new", "content": "new" }]),
    ] {
        assert_eq!(
            super::patch_changed_json(&original, &baseline, &edited).unwrap(),
            edited
        );
    }
}

#[test]
fn all_builtin_codecs_accept_multiple_message_replacements() {
    use super::traits::LlmCodec;
    let cases: Vec<(Box<dyn LlmCodec>, Json)> = vec![
        (
            Box::new(super::openai_chat::OpenAIChatCodec),
            json!({"model": "test", "messages": [
                {"role": "user", "content": "a", "vendor": "A"}, {"role": "user", "content": "b", "vendor": "B"}
            ]}),
        ),
        (
            Box::new(super::openai_responses::OpenAIResponsesCodec),
            json!({"model": "test", "input": [
                {"role": "user", "content": "a", "vendor": "A"}, {"role": "user", "content": "b", "vendor": "B"}
            ]}),
        ),
        (
            Box::new(super::anthropic::AnthropicMessagesCodec),
            json!({"model": "test", "max_tokens": 10, "messages": [
                {"role": "user", "content": "a", "vendor": "A"}, {"role": "user", "content": "b", "vendor": "B"}
            ]}),
        ),
        (
            Box::new(super::gemini_generate_content::GeminiGenerateContentCodec),
            json!({"contents": [
                {"role": "user", "parts": [{"text": "a"}], "vendor": "A"},
                {"role": "user", "parts": [{"text": "b"}], "vendor": "B"}
            ]}),
        ),
        (
            Box::new(super::oci_genai::OCIGenAIChatCodec),
            json!({"apiFormat": "GENERIC", "messages": [
                {"role": "USER", "content": [{"type": "TEXT", "text": "a"}], "vendor": "A"},
                {"role": "USER", "content": [{"type": "TEXT", "text": "b"}], "vendor": "B"}
            ]}),
        ),
        (
            Box::new(super::oci_genai::OCIGenAIChatCodec),
            json!({"apiFormat": "COHERE", "chatHistory": [
            {"role": "USER", "message": "a", "vendor": "A"}, {"role": "USER", "message": "b", "vendor": "B"}
        ], "message": "current"}),
        ),
    ];
    for (codec, content) in cases {
        let original = req(content);
        let mut annotated = codec.decode(&original).unwrap();
        let baseline = annotated.messages.clone();
        annotated.messages = ["new one", "new two", "inserted"]
            .into_iter()
            .map(|text| Message::User {
                content: MessageContent::Text(text.into()),
                name: None,
            })
            .collect();
        let encoded = codec.encode(&annotated, &original).unwrap();
        assert_eq!(codec.decode(&encoded).unwrap().messages, annotated.messages);
        assert!(!encoded.content.to_string().contains("vendor"));
        // Exact matches retain their original fields after reordering.
        annotated.messages = baseline.clone();
        annotated.messages.swap(0, 1);
        let reordered = codec.encode(&annotated, &original).unwrap();
        assert_eq!(
            codec.decode(&reordered).unwrap().messages,
            annotated.messages
        );
        assert!(reordered.content.to_string().contains("vendor"));
    }
}

#[test]
fn multiple_identified_results_keep_provider_fields_after_reordering() {
    use super::traits::LlmCodec;
    let cases: Vec<(Box<dyn LlmCodec>, Json)> = vec![
        (
            Box::new(super::openai_chat::OpenAIChatCodec),
            json!({"messages": [
                {"role": "tool", "tool_call_id": "a", "content": "old", "vendor": "A"},
                {"role": "tool", "tool_call_id": "b", "content": "old", "vendor": "B"}
            ]}),
        ),
        (
            Box::new(super::openai_responses::OpenAIResponsesCodec),
            json!({"input": [
                {"type": "function_call_output", "call_id": "a", "output": "old", "vendor": "A"},
                {"type": "function_call_output", "call_id": "b", "output": "old", "vendor": "B"}
            ]}),
        ),
        (
            Box::new(super::gemini_generate_content::GeminiGenerateContentCodec),
            json!({"contents": [
                {"role": "user", "vendor_wrapper": true, "parts": [
                    {"functionResponse": {"id": "a", "name": "first", "response": {"old": true}, "vendor": "A"}, "vendor_part": "A"},
                    {"functionResponse": {"id": "b", "name": "second", "response": {"old": true}, "vendor": "B"}, "vendor_part": "B"}
                ]}
            ]}),
        ),
    ];
    for (codec, content) in cases {
        let original = req(content);
        let mut annotated = codec.decode(&original).unwrap();
        for message in &mut annotated.messages {
            match message {
                Message::Tool { content, .. } => *content = MessageContent::Text("blocked".into()),
                Message::ToolResultItem { output, .. } => *output = json!("blocked"),
                Message::ProviderNative { value, .. } => value["content"] = json!("blocked"),
                other => panic!("unexpected result: {other:?}"),
            }
        }
        annotated.messages.swap(0, 1);
        let encoded = codec.encode(&annotated, &original).unwrap();
        let text = encoded.content.to_string();
        assert!(text.contains("\"vendor\":\"A\""), "{text}");
        assert!(text.contains("\"vendor\":\"B\""), "{text}");
        let decoded = codec.decode(&encoded).unwrap();
        let id = |message: &Message| match message {
            Message::Tool { tool_call_id, .. } => tool_call_id.clone(),
            Message::ToolResultItem { call_id, .. } => call_id.clone(),
            Message::ProviderNative { value, .. } => {
                value["tool_call_id"].as_str().unwrap().to_string()
            }
            _ => panic!("expected result"),
        };
        assert_eq!(
            decoded.messages.iter().map(id).collect::<Vec<_>>(),
            vec!["b", "a"]
        );
        if original.content.get("contents").is_some() {
            assert_eq!(encoded.content["contents"][0]["vendor_wrapper"], true);
            assert_eq!(
                encoded.content["contents"][0]["parts"][0]["vendor_part"],
                "B"
            );
        }
    }
}

#[test]
fn anthropic_multiple_nested_result_edits_keep_fields() {
    use super::request::ContentPart;
    use super::traits::LlmCodec;
    let codec = super::anthropic::AnthropicMessagesCodec;
    let original = req(json!({"model": "test", "max_tokens": 10, "messages": [
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "a", "content": "old", "vendor": "A"},
            {"type": "tool_result", "tool_use_id": "b", "content": "old", "vendor": "B"}
        ]}
    ]}));
    let mut annotated = codec.decode(&original).unwrap();
    let Message::User {
        content: MessageContent::Parts(parts),
        ..
    } = &mut annotated.messages[0]
    else {
        panic!("expected parts")
    };
    for part in parts.iter_mut() {
        let ContentPart::ToolResult { content, .. } = part else {
            panic!("expected tool result")
        };
        *content = json!("blocked");
    }
    parts.swap(0, 1);
    let encoded = codec.encode(&annotated, &original).unwrap();
    assert_eq!(
        encoded.content["messages"][0]["content"],
        json!([
            {"type": "tool_result", "tool_use_id": "b", "content": "blocked", "vendor": "B"},
            {"type": "tool_result", "tool_use_id": "a", "content": "blocked", "vendor": "A"}
        ])
    );
}
