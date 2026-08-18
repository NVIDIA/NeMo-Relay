// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in tool-result cache.
//!
//! A hit suppresses the real call, so caching is off by default and must be
//! enabled only for tools that are read-only and stable for the configured TTL.
//! Key and store failures fail open to the real call.

use std::sync::Arc;
use std::time::Duration;

use nemo_relay::api::runtime::{ToolExecutionFn, ToolExecutionNextFn};
use nemo_relay::api::tool::{ToolExecutionInterceptOutcome, ToolExecutionResult};
use nemo_relay::error::Result as FlowResult;
use serde_json::{Value as Json, json};

use crate::config::ResponseCacheConfig;
use crate::response_cache::config::{ToolCacheConfig, ToolClass, ToolOverride};
use crate::response_cache::intercept::should_bypass;
use crate::response_cache::key::{KeyOutcome, build_tool_cache_key};
use crate::response_cache::mark::{
    CacheMark, CacheMarkStatus, CacheReason, CacheSurface, emit_cache_mark,
};
use crate::response_cache::store::{CacheEntry, CacheStore, now_unix_ms};

const TOOL_RESULT_CACHE_ENTRY_SCHEMA: &str = "nemo.relay.ResponseCacheToolResult@1";

#[derive(Debug, Clone, PartialEq)]
struct ResolvedToolPolicy {
    cacheable: bool,
    ttl: Duration,
    bypass_rate: f64,
    arg_skip: Vec<String>,
    tool_version: Option<String>,
}

fn resolve_policy(
    tool_name: &str,
    response_cache: &ResponseCacheConfig,
    tools: &ToolCacheConfig,
) -> ResolvedToolPolicy {
    let class: &ToolClass = resolve_class(tool_name, tools).unwrap_or(&tools.default);
    let over: Option<&ToolOverride> = resolve_override(tool_name, tools);

    let cacheable = over
        .and_then(|over| over.cacheable)
        .unwrap_or(class.cacheable);

    let ttl_seconds = over
        .and_then(|over| over.ttl_seconds)
        .or(class.ttl_seconds)
        .unwrap_or(response_cache.ttl_seconds);

    let bypass_rate = over
        .and_then(|over| over.bypass_rate)
        .or(class.bypass_rate)
        .unwrap_or(response_cache.bypass_rate);

    let arg_skip = match over.and_then(|over| over.arg_skip.clone()) {
        Some(list) => list,
        None => class.arg_skip.clone(),
    };

    let tool_version = over
        .and_then(|over| over.tool_version.clone())
        .or_else(|| class.tool_version.clone());

    ResolvedToolPolicy {
        cacheable,
        ttl: Duration::from_secs(ttl_seconds),
        bypass_rate,
        arg_skip,
        tool_version,
    }
}

fn resolve_class<'a>(tool_name: &str, tools: &'a ToolCacheConfig) -> Option<&'a ToolClass> {
    for class in tools.classes.values() {
        if class
            .members
            .iter()
            .any(|member| !member.contains('*') && member == tool_name)
        {
            return Some(class);
        }
    }
    best_wildcard_match(
        tools.classes.values().flat_map(|class| {
            class
                .members
                .iter()
                .map(move |member| (member.as_str(), class))
        }),
        tool_name,
    )
}

fn resolve_override<'a>(tool_name: &str, tools: &'a ToolCacheConfig) -> Option<&'a ToolOverride> {
    if let Some(over) = tools.overrides.get(tool_name) {
        return Some(over);
    }
    best_wildcard_match(
        tools
            .overrides
            .iter()
            .map(|(key, over)| (key.as_str(), over)),
        tool_name,
    )
}

fn best_wildcard_match<'a, T>(
    candidates: impl Iterator<Item = (&'a str, &'a T)>,
    name: &str,
) -> Option<&'a T> {
    let mut best = None;
    for (pattern, candidate) in candidates {
        if !pattern.contains('*') || !wildcard_match(pattern, name) {
            continue;
        }
        let rank = wildcard_rank(pattern);
        if best.as_ref().is_none_or(|(_, current)| rank > *current) {
            best = Some((candidate, rank));
        }
    }
    best.map(|(candidate, _)| candidate)
}

type WildcardRank<'a> = (usize, std::cmp::Reverse<usize>, std::cmp::Reverse<&'a str>);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WildcardPattern<'a> {
    Prefix(&'a str),
    Suffix(&'a str),
    Contains(&'a str),
}

fn parse_wildcard_pattern(pattern: &str) -> Option<WildcardPattern<'_>> {
    if pattern == "*" {
        return Some(WildcardPattern::Contains(""));
    }
    if let Some(contains) = pattern
        .strip_prefix('*')
        .and_then(|pattern| pattern.strip_suffix('*'))
    {
        return (!contains.is_empty() && !contains.contains('*'))
            .then_some(WildcardPattern::Contains(contains));
    }
    if let Some(suffix) = pattern.strip_prefix('*') {
        return (!suffix.is_empty() && !suffix.contains('*'))
            .then_some(WildcardPattern::Suffix(suffix));
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        return (!prefix.is_empty() && !prefix.contains('*'))
            .then_some(WildcardPattern::Prefix(prefix));
    }
    None
}

pub(crate) fn is_supported_tool_pattern(pattern: &str) -> bool {
    !pattern.contains('*') || parse_wildcard_pattern(pattern).is_some()
}

/// Returns the deterministic specificity order for a wildcard pattern.
///
/// Literal and wildcard counts are Unicode-character based. The final
/// lexicographic component only breaks otherwise equal ranks.
fn wildcard_rank(pattern: &str) -> WildcardRank<'_> {
    let stars = pattern
        .chars()
        .filter(|character| *character == '*')
        .count();
    (
        pattern.chars().count() - stars,
        std::cmp::Reverse(stars),
        std::cmp::Reverse(pattern),
    )
}

/// Returns whether two supported wildcard patterns can match a common tool name.
pub(crate) fn wildcard_patterns_overlap(left: &str, right: &str) -> bool {
    let (Some(left), Some(right)) = (parse_wildcard_pattern(left), parse_wildcard_pattern(right))
    else {
        return false;
    };
    match (left, right) {
        (WildcardPattern::Prefix(left), WildcardPattern::Prefix(right)) => {
            left.starts_with(right) || right.starts_with(left)
        }
        (WildcardPattern::Suffix(left), WildcardPattern::Suffix(right)) => {
            left.ends_with(right) || right.ends_with(left)
        }
        _ => true,
    }
}

fn wildcard_match(pattern: &str, name: &str) -> bool {
    match parse_wildcard_pattern(pattern) {
        Some(WildcardPattern::Prefix(prefix)) => name.starts_with(prefix),
        Some(WildcardPattern::Suffix(suffix)) => name.ends_with(suffix),
        Some(WildcardPattern::Contains(contains)) => name.contains(contains),
        None => !pattern.contains('*') && pattern == name,
    }
}

pub(crate) fn make_tool_intercept(
    store: Arc<dyn CacheStore>,
    response_cache: Arc<ResponseCacheConfig>,
    tools: Arc<ToolCacheConfig>,
) -> ToolExecutionFn {
    Arc::new(move |name: &str, args: Json, next: ToolExecutionNextFn| {
        let store = Arc::clone(&store);
        let response_cache = Arc::clone(&response_cache);
        let tools = Arc::clone(&tools);
        let name = name.to_string();
        Box::pin(run_tool_cache(
            name,
            args,
            next,
            store,
            response_cache,
            tools,
        ))
    })
}

async fn run_tool_cache(
    name: String,
    args: Json,
    next: ToolExecutionNextFn,
    store: Arc<dyn CacheStore>,
    response_cache: Arc<ResponseCacheConfig>,
    tools: Arc<ToolCacheConfig>,
) -> FlowResult<ToolExecutionInterceptOutcome> {
    let policy = resolve_policy(&name, &response_cache, &tools);

    if !policy.cacheable {
        return next(args).await.map(Into::into);
    }

    let backend = store.backend_kind();

    let key = match build_tool_cache_key(
        &response_cache.namespace,
        &name,
        policy.tool_version.as_deref(),
        &args,
        &policy.arg_skip,
        tools.cache_errors,
    ) {
        KeyOutcome::Key(key) => key,
        KeyOutcome::Bypass(reason) => {
            emit_cache_mark(
                CacheMark::new(CacheMarkStatus::Bypass, backend)
                    .surface(CacheSurface::Tool)
                    .reason(reason),
            );
            return next(args).await.map(Into::into);
        }
    };

    if should_bypass(policy.bypass_rate) {
        emit_cache_mark(
            CacheMark::new(CacheMarkStatus::Bypass, backend)
                .surface(CacheSurface::Tool)
                .reason(CacheReason::Sampled)
                .key_hash(&key),
        );
        let result = next(args).await?;
        store_tool_result(&store, &key, policy.ttl, &result, tools.cache_errors).await;
        return Ok(result.into());
    }

    match store.get(&key).await {
        Ok(Some(entry))
            if !tools.cache_errors
                && is_error_shaped_tool_result(&decode_tool_result(&entry.response).result) =>
        {
            // A prior Relay version could have stored a snake_case `is_error`
            // result under this same policy. Never replay it after error
            // caching is disabled; a successful live result replaces it.
            emit_cache_mark(
                CacheMark::new(CacheMarkStatus::Bypass, backend)
                    .surface(CacheSurface::Tool)
                    .reason(CacheReason::CachedError)
                    .key_hash(&key),
            );
            let result = next(args).await?;
            store_tool_result(&store, &key, policy.ttl, &result, tools.cache_errors).await;
            Ok(result.into())
        }
        Ok(Some(entry)) => {
            let result = decode_tool_result(&entry.response);
            let age_ms = now_unix_ms().saturating_sub(entry.created_unix_ms);
            emit_cache_mark(
                CacheMark::new(CacheMarkStatus::Hit, backend)
                    .surface(CacheSurface::Tool)
                    .key_hash(&key)
                    .age_ms(age_ms)
                    .ttl_ms(policy.ttl.as_millis() as u64)
                    .saved_invocations(1),
            );
            Ok(result.into())
        }
        Ok(None) => {
            emit_cache_mark(
                CacheMark::new(CacheMarkStatus::Miss, backend)
                    .surface(CacheSurface::Tool)
                    .key_hash(&key)
                    .ttl_ms(policy.ttl.as_millis() as u64),
            );
            let result = next(args).await?;
            store_tool_result(&store, &key, policy.ttl, &result, tools.cache_errors).await;
            Ok(result.into())
        }
        Err(_) => {
            emit_cache_mark(
                CacheMark::new(CacheMarkStatus::Miss, backend)
                    .surface(CacheSurface::Tool)
                    .reason(CacheReason::StoreError)
                    .key_hash(&key),
            );
            next(args).await.map(Into::into)
        }
    }
}

async fn store_tool_result(
    store: &Arc<dyn CacheStore>,
    key: &str,
    ttl: Duration,
    result: &ToolExecutionResult,
    cache_errors: bool,
) {
    if !cache_errors && is_error_shaped_tool_result(&result.result) {
        return;
    }
    let entry = CacheEntry::new(encode_tool_result(result), ttl, key.to_string(), None, None);
    let _ = store.set(key, entry, ttl).await;
}

fn encode_tool_result(result: &ToolExecutionResult) -> Json {
    json!({
        "$schema": TOOL_RESULT_CACHE_ENTRY_SCHEMA,
        "result": result.result,
        "annotation": result.annotation,
    })
}

fn decode_tool_result(value: &Json) -> ToolExecutionResult {
    let Some(object) = value.as_object() else {
        return value.clone().into();
    };
    if object.get("$schema").and_then(Json::as_str) != Some(TOOL_RESULT_CACHE_ENTRY_SCHEMA) {
        return value.clone().into();
    }
    let Some(result) = object.get("result") else {
        return value.clone().into();
    };
    let mut decoded = ToolExecutionResult::new(result.clone());
    if let Some(annotation) = object.get("annotation").filter(|value| !value.is_null()) {
        decoded = decoded.with_annotation(annotation.clone());
    }
    decoded
}

/// A tool result has no universal provider envelope. Treat only the explicit,
/// widely used in-band error signals as failures by default; applications that
/// use these fields for stable data can opt into caching them.
fn is_error_shaped_tool_result(result: &Json) -> bool {
    let Some(object) = result.as_object() else {
        return false;
    };
    object.get("error").is_some_and(|error| !error.is_null())
        || object.get("isError").and_then(Json::as_bool) == Some(true)
        || object.get("is_error").and_then(Json::as_bool) == Some(true)
}

#[cfg(test)]
#[path = "../../tests/unit/response_cache/tool_policy_tests.rs"]
mod policy_tests;

#[cfg(test)]
#[path = "../../tests/unit/response_cache/tool_tests.rs"]
mod coverage_tests;
