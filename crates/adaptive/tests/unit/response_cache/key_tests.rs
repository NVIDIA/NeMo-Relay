// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for response-cache keying in the NeMo Relay adaptive crate.

use super::*;
use crate::acg::canonicalize::{canonicalize_value, sha256_hex};

#[test]
fn fingerprint_matches_canonicalize_then_hash() {
    // `fingerprint` streams canonical bytes into the hasher; it must produce
    // byte-for-byte the same digest as materializing the canonical string
    // first, or every existing key would silently change.
    let doc = json!({
        "z": {"nested": [3, 1.5, -0.0, 1e21]},
        "a": "höllo \u{1F600} wörld",
        "codec": null,
        "headers": {},
        "body": {"messages": [{"role": "user", "content": "hi"}]}
    });
    assert_eq!(
        fingerprint(&doc).unwrap(),
        sha256_hex(&canonicalize_value(&doc).unwrap()),
    );
}

fn request(content: Json) -> LlmRequest {
    LlmRequest {
        headers: Map::new(),
        content,
    }
}

fn key_of(provider: &str, request: &LlmRequest, config: &ResponseCacheConfig) -> String {
    match build_cache_key(provider, request, config) {
        KeyOutcome::Key(key) => key,
        other => panic!("expected a key, got {other:?}"),
    }
}

#[test]
fn field_order_and_whitespace_do_not_change_the_key() {
    let config = ResponseCacheConfig::default();
    let first = request(
        json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "tool_choice": "auto"}),
    );
    let second = request(
        json!({"tool_choice": "auto", "messages": [{"content": "hi", "role": "user"}], "model": "m"}),
    );
    assert_eq!(
        key_of("openai", &first, &config),
        key_of("openai", &second, &config)
    );
}

#[test]
fn skiplist_fields_do_not_change_the_key() {
    let config = ResponseCacheConfig::default();
    let base = request(json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}));
    let noisy = request(json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
        "user": "abc",
        "metadata": {"trace": "xyz"}
    }));
    assert_eq!(
        key_of("openai", &base, &config),
        key_of("openai", &noisy, &config)
    );
}

#[test]
fn namespace_and_provider_separate_keys() {
    let request = request(json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]}));
    let ns_a = ResponseCacheConfig {
        namespace: "a".to_string(),
        ..ResponseCacheConfig::default()
    };
    let ns_b = ResponseCacheConfig {
        namespace: "b".to_string(),
        ..ResponseCacheConfig::default()
    };
    assert_ne!(
        key_of("openai", &request, &ns_a),
        key_of("openai", &request, &ns_b)
    );
    // Same namespace, different provider/family also separates.
    assert_ne!(
        key_of("openai", &request, &ns_a),
        key_of("anthropic", &request, &ns_a)
    );
}

#[test]
fn random_tool_call_ids_are_normalized_to_one_key() {
    let config = ResponseCacheConfig::default();
    let make = |call_id: &str| {
        request(json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": call_id, "content": "sunny"}
            ]
        }))
    };
    assert_eq!(
        key_of("openai", &make("call_RANDOM_1"), &config),
        key_of("openai", &make("call_RANDOM_2"), &config),
        "random tool-call ids must not change the key"
    );
}

#[test]
fn raw_params_objects_do_not_collide_with_typed_caps() {
    // A top-level `params` object lands in the flattened `extra` and
    // overwrites the typed field on serialization — the token caps would
    // vanish from the key.
    let config = ResponseCacheConfig::default();
    let make = |cap: u64| {
        request(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "write"}],
            "max_tokens": cap,
            "params": {"vendor": "x"}
        }))
    };
    assert_ne!(
        key_of("openai", &make(1), &config),
        key_of("openai", &make(100), &config),
        "a raw params object must not erase the token cap from the key"
    );
}

#[test]
fn wrong_typed_generation_scalars_do_not_collide() {
    let config = ResponseCacheConfig::default();
    let plain = request(json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}]
    }));
    // A string temperature is dropped by the typed extraction and excluded
    // from `extra`; it must not key like the temperature-less request.
    let string_temp = request(json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": "0.9"
    }));
    assert_ne!(
        key_of("openai", &plain, &config),
        key_of("openai", &string_temp, &config),
        "a wrong-typed temperature must separate keys"
    );
    // Same for a float token cap (as_u64 yields None).
    let float_cap = request(json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100.0
    }));
    assert_ne!(
        key_of("openai", &plain, &config),
        key_of("openai", &float_cap, &config),
        "a float token cap must separate keys"
    );
    // A string parallel_tool_calls is dropped by as_bool.
    let string_parallel = request(json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "parallel_tool_calls": "false"
    }));
    assert_ne!(
        key_of("openai", &plain, &config),
        key_of("openai", &string_parallel, &config),
        "a wrong-typed parallel_tool_calls must separate keys"
    );
    // And a float top_logprobs (as_u64 yields None).
    let float_logprobs = request(json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "top_logprobs": 5.0
    }));
    assert_ne!(
        key_of("openai", &plain, &config),
        key_of("openai", &float_logprobs, &config),
        "a float top_logprobs must separate keys"
    );
}

#[test]
fn unmodeled_tool_choice_fields_do_not_collide() {
    let config = ResponseCacheConfig::default();
    let make = |strict: bool| {
        request(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "call it"}],
            "tool_choice": {"type": "function",
                            "function": {"name": "docs_lookup", "strict": strict}}
        }))
    };
    assert_ne!(
        key_of("openai", &make(true), &config),
        key_of("openai", &make(false), &config),
        "unmodeled tool_choice fields must separate keys"
    );
}

#[test]
fn stateful_conversation_and_default_store_bypass() {
    let config = ResponseCacheConfig::default();
    // Server-side conversation state.
    let with_conversation = request(json!({
        "model": "gpt-4o", "input": "summarize", "store": false,
        "conversation": "conv_1"
    }));
    assert_eq!(
        build_cache_key("openai", &with_conversation, &config),
        KeyOutcome::Bypass("stateful_conversation")
    );
    // Responses persists by default: no explicit `store: false` opt-out
    // means the call is stateful.
    let default_store = request(json!({"model": "gpt-4o", "input": "hello"}));
    assert_eq!(
        build_cache_key("openai", &default_store, &config),
        KeyOutcome::Bypass("stateful_store")
    );
    let opted_out = request(json!({"model": "gpt-4o", "input": "hello", "store": false}));
    assert!(matches!(
        build_cache_key("openai", &opted_out, &config),
        KeyOutcome::Key(_)
    ));
    // A `prompt` object is a Responses prompt-template reference, so the
    // call persists by default too; only the explicit opt-out is stateless.
    let template = request(json!({
        "model": "gpt-4o",
        "prompt": {"id": "pmpt_1", "variables": {"tone": "formal"}}
    }));
    assert_eq!(
        build_cache_key("openai", &template, &config),
        KeyOutcome::Bypass("stateful_store")
    );
    let template_opted_out = request(json!({
        "model": "gpt-4o",
        "prompt": {"id": "pmpt_1", "variables": {"tone": "formal"}},
        "store": false
    }));
    assert!(matches!(
        build_cache_key("openai", &template_opted_out, &config),
        KeyOutcome::Key(_)
    ));
}

#[test]
fn null_bodies_bypass_the_cache() {
    // The gateway parses unparseable upstream bodies to `null`; every such
    // request would share one key, so they are never cacheable.
    assert_eq!(
        build_cache_key(
            "openai",
            &request(Json::Null),
            &ResponseCacheConfig::default()
        ),
        KeyOutcome::Bypass("unparseable_body")
    );
}

#[test]
fn unrepresentable_integers_bypass_the_cache() {
    // 9007199254740995 and 9007199254740996 are distinct ids but the same
    // f64, so without the bypass they canonicalize to one key.
    let config = ResponseCacheConfig::default();
    let make = |id: u64| {
        request(json!({
            "model": "claude-sonnet-4",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "look up the record"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "lookup",
                     "input": {"record_id": id}}
                ]}
            ]
        }))
    };
    for id in [9_007_199_254_740_995_u64, 9_007_199_254_740_996] {
        assert_eq!(
            build_cache_key("anthropic", &make(id), &config),
            KeyOutcome::Bypass("unrepresentable_number"),
            "id {id} lies beyond 2^53 and must not be trusted in a key"
        );
    }
}

#[test]
fn stateful_responses_calls_bypass() {
    let config = ResponseCacheConfig::default();
    let with_store = request(json!({"model": "m", "messages": [], "store": true}));
    assert_eq!(
        build_cache_key("openai", &with_store, &config),
        KeyOutcome::Bypass("stateful_store")
    );
    let with_prev =
        request(json!({"model": "m", "messages": [], "previous_response_id": "resp_1"}));
    assert_eq!(
        build_cache_key("openai", &with_prev, &config),
        KeyOutcome::Bypass("stateful_previous_response_id")
    );
    // A truthy non-boolean `store` must still bypass (it is otherwise stripped
    // from the key), while `store: false` stays cacheable.
    let with_truthy = request(json!({"model": "m", "messages": [], "store": "true"}));
    assert_eq!(
        build_cache_key("openai", &with_truthy, &config),
        KeyOutcome::Bypass("stateful_store")
    );
    let not_stored = request(json!({"model": "m", "messages": [], "store": false}));
    assert!(matches!(
        build_cache_key("openai", &not_stored, &config),
        KeyOutcome::Key(_)
    ));
}

#[test]
fn nondeterministic_calls_bypass_only_when_disabled() {
    let sampled = request(json!({"model": "m", "messages": [], "temperature": 0.7}));
    let skip = ResponseCacheConfig {
        cache_nondeterministic: false,
        ..ResponseCacheConfig::default()
    };
    assert_eq!(
        build_cache_key("openai", &sampled, &skip),
        KeyOutcome::Bypass("nondeterministic_temperature")
    );
    // Absent temperature: providers default to positive sampling.
    let absent = request(json!({"model": "m", "messages": []}));
    assert_eq!(
        build_cache_key("openai", &absent, &skip),
        KeyOutcome::Bypass("nondeterministic_temperature")
    );
    // Explicitly pinned deterministic stays cacheable.
    let pinned = request(json!({"model": "m", "messages": [], "temperature": 0.0}));
    assert!(matches!(
        build_cache_key("openai", &pinned, &skip),
        KeyOutcome::Key(_)
    ));
    // Default keeps caching temperature > 0 calls.
    assert!(matches!(
        build_cache_key("openai", &sampled, &ResponseCacheConfig::default()),
        KeyOutcome::Key(_)
    ));
}

#[test]
fn chat_shaped_requests_key_on_the_detected_decode() {
    // A request the OpenAI-chat codec can decode: detection must pick the
    // chat surface and the keyed body must be the decode, not the raw body
    // — a silent decode regression cannot hide behind identical keys.
    let request = request(json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0
    }));
    let (body, effective_codec) = resolved_body("openai", &request);
    assert_eq!(effective_codec, Some("openai_chat"));
    assert_ne!(
        body, request.content,
        "the keyed body must be the normalized decode, not the raw body"
    );
}

#[test]
fn undetectable_shape_falls_back_to_raw_keying() {
    // No `messages`/`input`/`system` top-level key: no surface detects, so
    // the raw body is fingerprinted — still a usable, stable key.
    let config = ResponseCacheConfig::default();
    let request = request(json!({"model": "m", "prompt": "hi"}));
    let (body, effective_codec) = resolved_body("openai", &request);
    assert_eq!(effective_codec, None, "nothing must detect this shape");
    assert_eq!(body, request.content, "the raw body is keyed as-is");
    let first = key_of("openai", &request, &config);
    let second = key_of("openai", &request, &config);
    assert_eq!(first, second, "raw-fallback keys must be stable");
}

#[test]
fn dual_token_caps_do_not_collide() {
    // With both caps present the decode keeps one; requests differing in
    // the other must not merge.
    let config = ResponseCacheConfig::default();
    let low = request(json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "write a story"}],
        "max_completion_tokens": 100,
        "max_tokens": 1
    }));
    let high = request(json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "write a story"}],
        "max_completion_tokens": 100,
        "max_tokens": 9999
    }));
    assert_ne!(
        key_of("openai", &low, &config),
        key_of("openai", &high, &config),
        "requests carrying both token caps must key on both"
    );
}

#[test]
fn anthropic_system_block_metadata_does_not_collide() {
    // System content blocks are flattened to their text on decode; block
    // fields beyond the provider cache hint must not vanish from the key.
    let config = ResponseCacheConfig::default();
    let make = |marker: u64| {
        request(json!({
            "model": "claude-sonnet-4",
            "max_tokens": 64,
            "system": [{"type": "text", "text": "Use policy X", "priority": marker}],
            "messages": [{"role": "user", "content": "Answer"}]
        }))
    };
    assert_ne!(
        key_of("anthropic", &make(1), &config),
        key_of("anthropic", &make(2), &config),
        "system-block metadata must separate keys"
    );
}

#[test]
fn unmodeled_tool_fields_do_not_collide() {
    // `FunctionDefinition` has no unknown-field catch-all, so a field like
    // OpenAI's `function.strict` is silently dropped on decode; requests
    // differing in it must not merge.
    let config = ResponseCacheConfig::default();
    let make = |strict: bool| {
        request(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "look it up"}],
            "tools": [{"type": "function", "function": {
                "name": "docs_lookup",
                "description": "Look up docs",
                "parameters": {"type": "object", "properties": {}},
                "strict": strict
            }}]
        }))
    };
    assert_ne!(
        key_of("openai", &make(true), &config),
        key_of("openai", &make(false), &config),
        "unmodeled tool fields must separate keys"
    );
}

#[test]
fn cleanly_modeled_tools_still_key_on_the_decode() {
    // The tools round-trip guard must not disable normalized keying for
    // tools the normalized types represent fully.
    let request = request(json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "look it up"}],
        "tools": [{"type": "function", "function": {
            "name": "docs_lookup",
            "description": "Look up docs",
            "parameters": {"type": "object", "properties": {}}
        }}]
    }));
    let (body, effective_codec) = resolved_body("openai", &request);
    assert_eq!(
        effective_codec,
        Some("openai_chat"),
        "cleanly modeled tools must keep the decode"
    );
    // This minimal fixture round-trips byte-identically — which is exactly
    // what the tools guard verifies before trusting the decode.
    assert_eq!(body["tools"], request.content["tools"]);
}

#[test]
fn system_less_anthropic_body_uses_the_provider_hint() {
    // An Anthropic request without a top-level `system` is shape-identical
    // to OpenAI Chat; the provider-name hint must resolve it.
    let request = request(json!({
        "model": "claude-sonnet-4",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}]
    }));
    let (_, hinted) = resolved_body("anthropic", &request);
    assert_eq!(
        hinted,
        Some("anthropic_messages"),
        "the hint must detect the anthropic surface for a system-less body"
    );
    let (_, unhinted) = resolved_body("openai", &request);
    assert_eq!(
        unhinted,
        Some("openai_chat"),
        "without the anthropic hint the same shape reads as chat"
    );
}

#[test]
fn unmodeled_message_fields_do_not_collide() {
    // The normalized message types are closed, so an assistant field they
    // do not model — the deprecated `function_call`, `refusal` — decodes
    // to nothing; conversations differing only there must not merge.
    let config = ResponseCacheConfig::default();
    let make = |arguments: &str| {
        request(json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "assistant", "content": null,
                 "function_call": {"name": "lookup", "arguments": arguments}},
                {"role": "user", "content": "Continue"}
            ]
        }))
    };
    assert_ne!(
        key_of("openai", &make("{\"q\":\"alpha\"}"), &config),
        key_of("openai", &make("{\"q\":\"beta\"}"), &config),
        "unmodeled message fields must separate keys"
    );
}

#[test]
fn non_array_stop_forms_do_not_collide_with_a_stopless_request() {
    // Only an array of strings decodes faithfully; every other `stop`
    // form is silently dropped and must stay raw-keyed.
    let config = ResponseCacheConfig::default();
    let without = request(json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "count: one END two"}]
    }));
    for stop in [json!("END"), json!(["END", 7]), json!({}), json!(7)] {
        let with_stop = request(json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "count: one END two"}],
            "stop": stop
        }));
        assert_ne!(
            key_of("openai", &with_stop, &config),
            key_of("openai", &without, &config),
            "a malformed stop ({stop}) must not share a key with a stopless request"
        );
    }
}

#[test]
fn null_text_system_block_does_not_collide_with_no_system() {
    // A `text: null` block decodes to no system prompt at all; it must
    // not share a key with a request that genuinely has no system.
    let config = ResponseCacheConfig::default();
    let malformed = request(json!({
        "model": "claude-sonnet-4",
        "max_tokens": 64,
        "system": [{"type": "text", "text": null}],
        "messages": [{"role": "user", "content": "Answer"}]
    }));
    let clean = request(json!({
        "model": "claude-sonnet-4",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Answer"}]
    }));
    assert_ne!(
        key_of("anthropic", &malformed, &config),
        key_of("anthropic", &clean, &config),
        "a null-text system block must not key like an absent system"
    );
}
