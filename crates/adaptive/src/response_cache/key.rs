// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact-match cache-key derivation.
//!
//! The LLM execution intercept receives the *raw* [`LlmRequest`] (the decoded
//! `AnnotatedLlmRequest` is only handed to *request* intercepts), so the plugin
//! decodes the request itself and keys on the normalized form: provider-shaped
//! differences that mean the same thing collapse to one key. The provider
//! surface is auto-detected from the request shape (hinted by the provider
//! name); there is nothing to configure. Only when detection or decode fails is
//! the raw body fingerprinted (the fallback). Either way RFC
//! 8785 canonicalization removes field-order/whitespace noise, an always-on
//! skip-list drops volatile/identity fields, tool-call IDs are normalized, and
//! only allowlisted headers fold in.

use nemo_relay::api::llm::LlmRequest;
use nemo_relay::codec::request::AnnotatedLlmRequest;
use nemo_relay::codec::resolve::{
    ProviderSurface, detect_request_surface_with_hint, request_codec,
};
use serde_json::{Map, Value as Json, json};
use sha2::{Digest, Sha256};

use crate::config::ResponseCacheConfig;
use crate::response_cache::store::CACHE_SCHEMA_VERSION;

/// Top-level request-body keys that never affect the answer and are always
/// dropped before fingerprinting (IDs, routing, bookkeeping, streaming flag).
pub const DEFAULT_SKIP_KEYS: &[&str] = &["stream", "user", "metadata", "service_tier", "store"];

/// Result of deriving a cache key for a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyOutcome {
    /// A usable exact-match key fingerprint (`"sha256:…"`).
    Key(String),
    /// The request is intentionally not cacheable; the reason is a short,
    /// stable label suitable for telemetry.
    Bypass(&'static str),
}

/// Derives the cache key for a request, or decides it must bypass the cache.
///
/// The request is decoded to the normalized `AnnotatedLlmRequest` and keyed on
/// that (so provider-shaped differences that mean the same thing collapse to
/// one key); the surface is auto-detected from the request shape. Only when
/// detection or decode fails is the raw request body fingerprinted. Either way the answer-determining fields are
/// kept, noise is dropped, tool-call IDs are normalized, and only allowlisted
/// headers fold in.
pub fn build_cache_key(
    provider: &str,
    request: &LlmRequest,
    config: &ResponseCacheConfig,
) -> KeyOutcome {
    // Unparseable bodies arrive as null; they would all share one key.
    if request.content.is_null() {
        return KeyOutcome::Bypass("unparseable_body");
    }
    // Cacheability gates run on the RAW request, so they are correct regardless
    // of which codec (if any) decodes the body — a chat codec may park `store`
    // in `extra` rather than the typed field, so we must not rely on the decode.
    if let Some(object) = request.content.as_object() {
        // Any present, non-`false` `store` opts into server-side persistence —
        // bypass even a malformed non-boolean rather than risk caching a stateful
        // call (whose result is otherwise keyed with `store` stripped).
        if object
            .get("store")
            .is_some_and(|value| !matches!(value, Json::Bool(false) | Json::Null))
        {
            return KeyOutcome::Bypass("stateful_store");
        }
        if object.contains_key("previous_response_id") {
            return KeyOutcome::Bypass("stateful_previous_response_id");
        }
        // Server-side conversation state the key cannot see.
        if object.contains_key("conversation") || object.contains_key("container") {
            return KeyOutcome::Bypass("stateful_conversation");
        }
        // Responses persists by default; only an explicit opt-out is stateless.
        // A `prompt` object is the Responses prompt-template reference; a bare
        // string `prompt` is a completions body with no server-side state.
        if (object.contains_key("input")
            || object.contains_key("instructions")
            || object.get("prompt").is_some_and(Json::is_object))
            && !object
                .get("store")
                .is_some_and(|store| store == &Json::Bool(false))
        {
            return KeyOutcome::Bypass("stateful_store");
        }
    }
    // Toggle off = explicit temperature 0 only; absent defaults to sampling.
    if !config.cache_nondeterministic
        && request_temperature(&request.content).is_none_or(|temperature| temperature > 0.0)
    {
        return KeyOutcome::Bypass("nondeterministic_temperature");
    }

    // Body to fingerprint: the decoded/normalized form when a surface resolves
    // and decode succeeds, otherwise the raw request body.
    let (mut body, effective_codec) = resolved_body(provider, request);

    if let Some(object) = body.as_object_mut() {
        // `AnnotatedLlmRequest.extra` is `#[serde(flatten)]`, so provider fields
        // also land at top level — one skip pass covers both.
        for key in DEFAULT_SKIP_KEYS {
            object.remove(*key);
        }
        for key in &config.skip_keys {
            object.remove(key);
        }
        normalize_tool_call_ids(object);
    }

    let headers = allowlisted_headers(&request.headers, &config.header_allowlist);

    let key_doc = json!({
        "v": CACHE_SCHEMA_VERSION,
        "ns": config.namespace,
        "provider": provider,
        "strategy": config.key_strategy,
        "codec": effective_codec,
        "body": body,
        "headers": headers,
    });
    if contains_unrepresentable_int(&key_doc) {
        return KeyOutcome::Bypass("unrepresentable_number");
    }

    match fingerprint(&key_doc) {
        Some(key) => KeyOutcome::Key(key),
        None => KeyOutcome::Bypass("canonicalization_failed"),
    }
}

/// True when any integer in `value` lies outside ±2^53: RFC 8785 serializes
/// numbers as f64, so distinct integers beyond that range can canonicalize to
/// the same bytes and must never share a trusted key.
fn contains_unrepresentable_int(value: &Json) -> bool {
    const MAX_EXACT: u64 = 1 << 53;
    match value {
        Json::Number(number) => {
            if let Some(unsigned) = number.as_u64() {
                unsigned > MAX_EXACT
            } else if let Some(signed) = number.as_i64() {
                signed.unsigned_abs() > MAX_EXACT
            } else {
                false
            }
        }
        Json::Array(items) => items.iter().any(contains_unrepresentable_int),
        Json::Object(map) => map.values().any(contains_unrepresentable_int),
        _ => false,
    }
}

/// Canonicalizes straight into SHA-256; byte-identical output to
/// `sha256_hex(&canonicalize_value(doc)?)` (proven by test).
fn fingerprint<T: serde::Serialize>(doc: &T) -> Option<String> {
    let mut hasher = Sha256::new();
    serde_json_canonicalizer::to_writer(doc, &mut HashWriter(&mut hasher)).ok()?;
    let digest = hasher.finalize();
    let mut out = String::with_capacity(7 + 64);
    out.push_str("sha256:");
    for byte in digest {
        out.push(char::from(HEX[(byte >> 4) as usize]));
        out.push(char::from(HEX[(byte & 0x0f) as usize]));
    }
    Some(out)
}

const HEX: [u8; 16] = *b"0123456789abcdef";

/// Feeds canonical bytes straight into the hasher (`digest` 0.11 no longer
/// implements `io::Write` for hashers).
struct HashWriter<'a>(&'a mut Sha256);

impl std::io::Write for HashWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.update(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// The body to fingerprint plus the codec that actually produced it.
///
/// The surface is auto-detected from the request shape, hinted by the provider
/// name — the same detector the streaming path uses. `(raw clone, None)` when
/// nothing detects or decodes.
fn resolved_body(provider: &str, request: &LlmRequest) -> (Json, Option<&'static str>) {
    let surface = detect_request_surface_with_hint(&request.content, Some(provider));
    let decoded = surface.and_then(|surface| {
        let annotated = decode_surface(surface, request)?;
        let body = serde_json::to_value(&annotated).ok()?;
        Some((body, surface.codec_name()))
    });
    match decoded {
        Some((body, name)) => (body, Some(name)),
        None => (request.content.clone(), None),
    }
}

/// Decodes the raw request via the surface's codec. Returns `None` on a decode
/// failure (the caller falls back to raw-body fingerprinting).
fn decode_surface(surface: ProviderSurface, request: &LlmRequest) -> Option<AnnotatedLlmRequest> {
    // Known-lossy shapes stay raw-keyed: a fallback only ever costs a miss.
    if lossy_request_shape(surface, &request.content) {
        return None;
    }
    let annotated = request_codec(surface).decode(request).ok()?;
    // The chat/anthropic codecs parse `messages` with `unwrap_or_default()`, so a
    // message carrying content the normalized type cannot represent (Anthropic
    // `tool_use`/`tool_result`/`image`, OpenAI `input_audio`/`file`) collapses the
    // WHOLE array to empty — silently dropping answer-affecting content from the
    // key, so two such requests would collide. If the raw request carried messages
    // but the decode produced none, reject the decode so the caller falls back to
    // raw-body fingerprinting, which keeps distinct requests distinct (lower
    // normalization, but never a wrong reuse).
    if annotated.messages.is_empty() && raw_has_messages(request) {
        return None;
    }
    // The closed tool types drop unmodeled fields (`function.strict`,
    // server-tool settings); trust the decode only if it round-trips.
    if let Some(raw_tools) = request.content.get("tools") {
        let decoded_tools = serde_json::to_value(&annotated.tools).ok()?;
        if decoded_tools != *raw_tools {
            return None;
        }
    }
    // Same round-trip rule for `tool_choice`.
    if let Some(raw_tool_choice) = request.content.get("tool_choice") {
        let decoded_tool_choice = serde_json::to_value(&annotated.tool_choice).ok()?;
        if decoded_tool_choice != *raw_tool_choice {
            return None;
        }
    }
    // Closed message types drop `function_call`/`refusal`; legitimately
    // restructured shapes (synthesized system) just key raw — a dedup loss only.
    if let Some(raw_conversation) = request
        .content
        .get("messages")
        .or_else(|| request.content.get("input"))
    {
        let decoded_messages = serde_json::to_value(&annotated.messages).ok()?;
        if decoded_messages != *raw_conversation {
            return None;
        }
    }
    Some(annotated)
}

/// Raw shapes the built-in codecs are known to decode lossily — answer-affecting
/// fields the normalized types drop or merge.
fn lossy_request_shape(surface: ProviderSurface, content: &Json) -> bool {
    let Some(object) = content.as_object() else {
        return false;
    };
    // Shapes the decode silently drops: a raw `params` object overwrites the
    // typed field via the flattened extra; wrong-typed scalars vanish entirely.
    if object.contains_key("params") {
        return true;
    }
    let non_number = |key: &str| {
        object
            .get(key)
            .is_some_and(|value| value.as_f64().is_none())
    };
    let non_u64 = |key: &str| {
        object
            .get(key)
            .is_some_and(|value| value.as_u64().is_none())
    };
    let non_bool = |key: &str| {
        object
            .get(key)
            .is_some_and(|value| value.as_bool().is_none())
    };
    if non_number("temperature")
        || non_number("top_p")
        || non_u64("max_tokens")
        || non_u64("max_completion_tokens")
        || non_u64("max_output_tokens")
        || non_u64("top_logprobs")
        || non_bool("parallel_tool_calls")
    {
        return true;
    }
    match surface {
        // `stop` is modeled only as an array of strings — any other present
        // form (a bare string, a mixed array, an object) is dropped on decode.
        // And when both token caps are present the decode keeps one.
        ProviderSurface::OpenAIChat => {
            object
                .get("stop")
                .is_some_and(|stop| !is_string_array(stop))
                || (object.contains_key("max_tokens")
                    && object.contains_key("max_completion_tokens"))
        }
        // Same for `stop_sequences`; and system content blocks are flattened
        // to their text on decode, so any non-text block, non-string text, or
        // block metadata beyond the provider cache hint is lost.
        ProviderSurface::AnthropicMessages => {
            object
                .get("stop_sequences")
                .is_some_and(|stop| !is_string_array(stop))
                || object
                    .get("system")
                    .and_then(Json::as_array)
                    .is_some_and(|blocks| blocks.iter().any(lossy_system_block))
        }
        ProviderSurface::OpenAIResponses => false,
    }
}

/// Whether `value` is an array whose items are all strings — the only `stop` /
/// `stop_sequences` form the normalized types keep faithfully.
fn is_string_array(value: &Json) -> bool {
    value
        .as_array()
        .is_some_and(|items| items.iter().all(Json::is_string))
}

/// Whether an Anthropic `system` content block carries anything the decode's
/// text-flattening would lose.
fn lossy_system_block(block: &Json) -> bool {
    let Some(fields) = block.as_object() else {
        return true;
    };
    fields.get("type").and_then(Json::as_str) != Some("text")
        || !fields.get("text").is_some_and(Json::is_string)
        || fields
            .keys()
            .any(|key| !matches!(key.as_str(), "type" | "text" | "cache_control"))
}

/// Whether the raw request body carries a non-empty `messages` (chat) or `input`
/// (Responses) array — the content a decode is expected to preserve.
fn raw_has_messages(request: &LlmRequest) -> bool {
    let non_empty_array = |key: &str| {
        request
            .content
            .get(key)
            .and_then(Json::as_array)
            .is_some_and(|items| !items.is_empty())
    };
    non_empty_array("messages") || non_empty_array("input")
}

fn request_temperature(content: &Json) -> Option<f64> {
    content
        .get("temperature")
        .and_then(Json::as_f64)
        .or_else(|| {
            content
                .pointer("/params/temperature")
                .and_then(Json::as_f64)
        })
}

/// Keeps only allowlisted headers, matched case-insensitively and emitted with
/// lowercased names so header-case noise does not change the key.
fn allowlisted_headers(headers: &Map<String, Json>, allowlist: &[String]) -> Map<String, Json> {
    let mut kept = Map::new();
    for name in allowlist {
        for (header_name, value) in headers {
            if header_name.eq_ignore_ascii_case(name) {
                kept.insert(header_name.to_ascii_lowercase(), value.clone());
            }
        }
    }
    kept
}

/// Renumbers tool-call IDs consistently so random IDs do not cause misses.
///
/// Tool calls and their results carry random IDs; what matters is which result
/// pairs with which call, not the literal value. We map IDs to `tcid_0`,
/// `tcid_1`, … in first-seen order across `messages`, rewriting both
/// `tool_calls[].id` and `tool_call_id`. Shapes without these keys are left
/// untouched (this only affects hit-rate, never correctness).
fn normalize_tool_call_ids(body: &mut Map<String, Json>) {
    let Some(messages) = body.get_mut("messages").and_then(Json::as_array_mut) else {
        return;
    };

    let mut mapping: Map<String, Json> = Map::new();
    for message in messages.iter_mut() {
        let Some(object) = message.as_object_mut() else {
            continue;
        };
        if let Some(tool_calls) = object.get_mut("tool_calls").and_then(Json::as_array_mut) {
            for call in tool_calls.iter_mut() {
                if let Some(id_value) = call.get_mut("id") {
                    rewrite_id(id_value, &mut mapping);
                }
            }
        }
        if let Some(id_value) = object.get_mut("tool_call_id") {
            rewrite_id(id_value, &mut mapping);
        }
    }
}

/// Rewrites a single JSON string id in place to its stable, first-seen mapping.
fn rewrite_id(id_value: &mut Json, mapping: &mut Map<String, Json>) {
    let Some(original) = id_value.as_str().map(str::to_string) else {
        return;
    };
    let stable = match mapping.get(&original) {
        Some(Json::String(existing)) => existing.clone(),
        _ => {
            let stable = format!("tcid_{}", mapping.len());
            mapping.insert(original, Json::String(stable.clone()));
            stable
        }
    };
    *id_value = Json::String(stable);
}

#[cfg(test)]
#[path = "../../tests/unit/response_cache/key_tests.rs"]
mod tests;
