// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in tool-result cache.
//!
//! A hit suppresses the real call, so caching is off by default and must be
//! enabled only for tools that are read-only and stable for the configured TTL.
//! Key and store failures fail open to the real call.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use nemo_relay::api::runtime::{ToolExecutionFn, ToolExecutionNextFn};
use nemo_relay::api::tool::ToolExecutionInterceptOutcome;
use nemo_relay::error::Result as FlowResult;
use serde_json::Value as Json;

use crate::config::ResponseCacheConfig;
use crate::response_cache::config::{ToolCacheConfig, ToolClass, ToolOverride};
use crate::response_cache::intercept::should_bypass;
use crate::response_cache::key::{KeyOutcome, build_tool_cache_key};
use crate::response_cache::mark::{CacheMark, CacheSurface, emit_cache_mark};
use crate::response_cache::store::{CacheEntry, CacheStore, now_unix_ms};

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

    let tool_version = over.and_then(|over| over.tool_version.clone());

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

/// Returns whether two `*` patterns can match at least one common tool name.
///
/// This evaluates the product of the two wildcard automata, so it is exact for
/// this deliberately small pattern language without constructing a sample name.
pub(crate) fn wildcard_patterns_overlap(left: &str, right: &str) -> bool {
    let left: Vec<char> = left.chars().collect();
    let right: Vec<char> = right.chars().collect();
    let mut pending = vec![(0, 0)];
    let mut visited = HashSet::new();

    while let Some((left_index, right_index)) = pending.pop() {
        if !visited.insert((left_index, right_index)) {
            continue;
        }
        if left_index == left.len() && right_index == right.len() {
            return true;
        }

        if left.get(left_index) == Some(&'*') {
            pending.push((left_index + 1, right_index));
        }
        if right.get(right_index) == Some(&'*') {
            pending.push((left_index, right_index + 1));
        }

        let (Some(left_character), Some(right_character)) =
            (left.get(left_index), right.get(right_index))
        else {
            continue;
        };
        if *left_character == '*' || *right_character == '*' || left_character == right_character {
            pending.push((
                left_index + usize::from(*left_character != '*'),
                right_index + usize::from(*right_character != '*'),
            ));
        }
    }

    false
}

fn wildcard_match(pattern: &str, name: &str) -> bool {
    if !pattern.contains('*') {
        return pattern == name;
    }
    let segments: Vec<&str> = pattern.split('*').collect();
    let (first, rest) = segments
        .split_first()
        .expect("split always yields a segment");
    if !name.starts_with(first) {
        return false;
    }
    let mut cursor = first.len();
    let (last, middles) = rest
        .split_last()
        .expect("a starred pattern splits into at least two segments");
    for segment in middles {
        match name[cursor..].find(segment) {
            Some(position) => cursor += position + segment.len(),
            None => return false,
        }
    }
    name.len() >= cursor + last.len() && name.ends_with(last)
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
                CacheMark::new("bypass", backend)
                    .surface(CacheSurface::Tool)
                    .reason(reason),
            );
            return next(args).await.map(Into::into);
        }
    };

    if should_bypass(policy.bypass_rate) {
        emit_cache_mark(
            CacheMark::new("bypass", backend)
                .surface(CacheSurface::Tool)
                .reason("sampled")
                .key_hash(&key),
        );
        let result = next(args).await?;
        store_tool_result(&store, &key, policy.ttl, &result, tools.cache_errors).await;
        return Ok(result.into());
    }

    match store.get(&key).await {
        Ok(Some(entry)) if !tools.cache_errors && is_error_shaped_tool_result(&entry.response) => {
            // A prior Relay version could have stored a snake_case `is_error`
            // result under this same policy. Never replay it after error
            // caching is disabled; a successful live result replaces it.
            emit_cache_mark(
                CacheMark::new("bypass", backend)
                    .surface(CacheSurface::Tool)
                    .reason("cached_error")
                    .key_hash(&key),
            );
            let result = next(args).await?;
            store_tool_result(&store, &key, policy.ttl, &result, tools.cache_errors).await;
            Ok(result.into())
        }
        Ok(Some(entry)) => {
            let age_ms = now_unix_ms().saturating_sub(entry.created_unix_ms);
            emit_cache_mark(
                CacheMark::new("hit", backend)
                    .surface(CacheSurface::Tool)
                    .key_hash(&key)
                    .age_ms(age_ms)
                    .ttl_ms(policy.ttl.as_millis() as u64)
                    .saved_invocations(1),
            );
            Ok(entry.response.clone().into())
        }
        Ok(None) => {
            emit_cache_mark(
                CacheMark::new("miss", backend)
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
                CacheMark::new("miss", backend)
                    .surface(CacheSurface::Tool)
                    .reason("store_error")
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
    result: &Json,
    cache_errors: bool,
) {
    if !cache_errors && is_error_shaped_tool_result(result) {
        return;
    }
    let entry = CacheEntry::new(result.clone(), ttl, key.to_string(), None, None);
    let _ = store.set(key, entry, ttl).await;
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
mod tests {
    use super::*;
    use crate::response_cache::config::ToolClass;
    use std::collections::BTreeMap;

    fn response_cache(ttl_seconds: u64, bypass_rate: f64) -> ResponseCacheConfig {
        ResponseCacheConfig {
            ttl_seconds,
            bypass_rate,
            ..ResponseCacheConfig::default()
        }
    }

    fn class(cacheable: bool, members: &[&str]) -> ToolClass {
        ToolClass {
            cacheable,
            members: members.iter().map(|member| member.to_string()).collect(),
            ..ToolClass::default()
        }
    }

    #[test]
    fn unclassified_tool_falls_into_the_default_bucket_uncached() {
        let tools = ToolCacheConfig::default();
        let policy = resolve_policy("anything", &response_cache(3600, 0.0), &tools);
        assert!(
            !policy.cacheable,
            "an unknown tool must default to not cached"
        );
    }

    #[test]
    fn class_membership_makes_a_tool_cacheable() {
        let mut classes = BTreeMap::new();
        classes.insert("read_only".to_string(), class(true, &["docs_lookup"]));
        classes.insert(
            "volatile".to_string(),
            ToolClass {
                cacheable: true,
                ttl_seconds: Some(300),
                bypass_rate: Some(0.2),
                members: vec!["get_weather".to_string()],
                ..ToolClass::default()
            },
        );
        let tools = ToolCacheConfig {
            classes,
            ..ToolCacheConfig::default()
        };
        let policy = resolve_policy("docs_lookup", &response_cache(3600, 0.0), &tools);
        assert!(policy.cacheable);
        assert_eq!(policy.ttl, Duration::from_secs(3600));
        assert_eq!(policy.bypass_rate, 0.0);
        let policy = resolve_policy("get_weather", &response_cache(3600, 0.0), &tools);
        assert_eq!(policy.ttl, Duration::from_secs(300));
        assert_eq!(policy.bypass_rate, 0.2);
    }

    #[test]
    fn per_tool_override_wins_over_its_class() {
        let mut classes = BTreeMap::new();
        classes.insert(
            "read_only".to_string(),
            ToolClass {
                cacheable: true,
                arg_skip: vec!["request_id".to_string()],
                members: vec!["docs_lookup".to_string()],
                ..ToolClass::default()
            },
        );
        let mut overrides = BTreeMap::new();
        overrides.insert(
            "docs_lookup".to_string(),
            ToolOverride {
                cacheable: Some(false),
                ttl_seconds: Some(30),
                bypass_rate: Some(0.25),
                tool_version: Some("v2".to_string()),
                ..ToolOverride::default()
            },
        );
        let tools = ToolCacheConfig {
            classes,
            overrides,
            ..ToolCacheConfig::default()
        };
        let policy = resolve_policy("docs_lookup", &response_cache(3600, 0.0), &tools);
        assert!(!policy.cacheable, "override cacheable=false must win");
        assert_eq!(policy.ttl, Duration::from_secs(30));
        assert_eq!(policy.bypass_rate, 0.25);
        assert_eq!(policy.tool_version.as_deref(), Some("v2"));
        assert_eq!(policy.arg_skip, vec!["request_id".to_string()]);
    }

    #[test]
    fn override_arg_skip_replaces_the_class_list() {
        let mut classes = BTreeMap::new();
        classes.insert(
            "read_only".to_string(),
            ToolClass {
                cacheable: true,
                arg_skip: vec!["session_id".to_string()],
                members: vec!["lookup".to_string()],
                ..ToolClass::default()
            },
        );
        let mut overrides = BTreeMap::new();
        overrides.insert(
            "lookup".to_string(),
            ToolOverride {
                arg_skip: Some(vec![]),
                ..ToolOverride::default()
            },
        );
        let tools = ToolCacheConfig {
            classes,
            overrides,
            ..ToolCacheConfig::default()
        };
        let policy = resolve_policy("lookup", &response_cache(3600, 0.0), &tools);
        assert!(
            policy.arg_skip.is_empty(),
            "an override arg_skip (even empty) replaces the class list"
        );
    }

    #[test]
    fn default_bucket_can_be_flipped_on_for_broad_coverage() {
        let tools = ToolCacheConfig {
            default: ToolClass {
                cacheable: true,
                ttl_seconds: Some(60),
                bypass_rate: Some(0.5),
                ..ToolClass::default()
            },
            ..ToolCacheConfig::default()
        };
        let policy = resolve_policy("unknown_tool", &response_cache(3600, 0.0), &tools);
        assert!(
            policy.cacheable,
            "default cacheable=true covers unknown tools"
        );
        assert_eq!(policy.ttl, Duration::from_secs(60));
        assert_eq!(policy.bypass_rate, 0.5);
    }

    #[test]
    fn wildcard_match_table() {
        let cases = [
            ("*", "", true),
            ("*", "anything", true),
            ("docs_*", "docs_lookup", true),
            ("docs_*", "docs_", true),
            ("docs_*", "doc_lookup", false),
            ("*_price", "stock_price", true),
            ("*_price", "price", false),
            ("get_*_price", "get_stock_price", true),
            ("get_*_price", "get_price", false),
            ("a*a", "a", false),
            ("a*a", "aa", true),
            ("a*a", "aba", true),
            ("a*b*c", "abc", true),
            ("a*b*c", "acb", false),
            ("Docs_*", "docs_lookup", false), // case-sensitive
            ("abc*", "abc*", true),           // no escaping: '*' matches itself via the span
        ];
        for (pattern, name, expected) in cases {
            assert_eq!(
                wildcard_match(pattern, name),
                expected,
                "wildcard_match({pattern:?}, {name:?})"
            );
        }
    }

    #[test]
    fn wildcard_overlap_table() {
        let cases = [
            ("*_email", "send_*", true),
            ("delete_*", "*_record", true),
            ("docs_*", "send_*", false),
            ("a*b*c", "a*c", true),
            ("é*", "*é", true),
            ("foo*", "bar*", false),
        ];
        for (left, right, expected) in cases {
            assert_eq!(
                wildcard_patterns_overlap(left, right),
                expected,
                "wildcard_patterns_overlap({left:?}, {right:?})"
            );
        }
    }

    #[test]
    fn wildcard_rank_counts_unicode_characters_not_utf8_bytes() {
        assert_eq!(wildcard_rank("*é*").0, 1);
        assert_eq!(wildcard_rank("*éé*").0, 2);
        assert_eq!(wildcard_rank("*💡*").0, 1);
    }

    #[test]
    fn wildcard_member_classifies_a_matching_tool() {
        let mut classes = BTreeMap::new();
        classes.insert("read_only".to_string(), class(true, &["docs_*"]));
        let tools = ToolCacheConfig {
            classes,
            ..ToolCacheConfig::default()
        };
        assert!(resolve_policy("docs_lookup", &response_cache(3600, 0.0), &tools).cacheable);
        assert!(
            !resolve_policy("send_email", &response_cache(3600, 0.0), &tools).cacheable,
            "a non-matching tool still falls through to default"
        );
    }

    #[test]
    fn exact_member_beats_any_wildcard_match() {
        let mut classes = BTreeMap::new();
        classes.insert("a_wildcards".to_string(), class(true, &["docs_*"]));
        classes.insert("b_exact".to_string(), class(false, &["docs_lookup"]));
        let tools = ToolCacheConfig {
            classes,
            ..ToolCacheConfig::default()
        };
        let policy = resolve_policy("docs_lookup", &response_cache(3600, 0.0), &tools);
        assert!(
            !policy.cacheable,
            "the exact member's class must win over a matching wildcard"
        );
    }

    #[test]
    fn most_specific_wildcard_wins() {
        let mut classes = BTreeMap::new();
        classes.insert("a_catch_all".to_string(), class(false, &["*"]));
        classes.insert("b_docs".to_string(), class(true, &["docs_*"]));
        let tools = ToolCacheConfig {
            classes,
            ..ToolCacheConfig::default()
        };
        assert!(
            resolve_policy("docs_lookup", &response_cache(3600, 0.0), &tools).cacheable,
            "the pattern with more literal characters must win"
        );
        assert!(!resolve_policy("send_email", &response_cache(3600, 0.0), &tools).cacheable);
    }

    #[test]
    fn equal_literals_fewer_stars_then_smaller_pattern_break_ties() {
        let mut classes = BTreeMap::new();
        classes.insert("two_stars".to_string(), class(false, &["a*b*"]));
        classes.insert("one_star".to_string(), class(true, &["ab*"]));
        let tools = ToolCacheConfig {
            classes,
            ..ToolCacheConfig::default()
        };
        assert!(
            resolve_policy("ab", &response_cache(3600, 0.0), &tools).cacheable,
            "with equal literal counts the pattern with fewer stars must win"
        );

        let mut classes = BTreeMap::new();
        classes.insert("suffix".to_string(), class(false, &["*x"]));
        classes.insert("prefix".to_string(), class(true, &["x*"]));
        let tools = ToolCacheConfig {
            classes,
            ..ToolCacheConfig::default()
        };
        assert!(
            !resolve_policy("x", &response_cache(3600, 0.0), &tools).cacheable,
            "'*x' sorts before 'x*', so the suffix class must win the tie"
        );
    }

    #[test]
    fn override_patterns_apply_with_exact_keys_winning() {
        let mut classes = BTreeMap::new();
        classes.insert("read_only".to_string(), class(true, &["docs_*"]));
        let mut overrides = BTreeMap::new();
        overrides.insert(
            "docs_secret_*".to_string(),
            ToolOverride {
                cacheable: Some(false),
                ..ToolOverride::default()
            },
        );
        overrides.insert(
            "docs_secret_audit".to_string(),
            ToolOverride {
                cacheable: Some(true),
                ..ToolOverride::default()
            },
        );
        let tools = ToolCacheConfig {
            classes,
            overrides,
            ..ToolCacheConfig::default()
        };
        let cacheable =
            |name: &str| resolve_policy(name, &response_cache(3600, 0.0), &tools).cacheable;
        assert!(
            !cacheable("docs_secret_dump"),
            "a pattern override must apply to the tools it matches"
        );
        assert!(
            cacheable("docs_secret_audit"),
            "an exact override key must win over a matching pattern"
        );
        assert!(
            cacheable("docs_lookup"),
            "tools no override matches keep their class policy"
        );
        let mut overrides = BTreeMap::new();
        overrides.insert(
            "docs_*".to_string(),
            ToolOverride {
                cacheable: Some(false),
                ..ToolOverride::default()
            },
        );
        let mut classes = BTreeMap::new();
        classes.insert("read_only".to_string(), class(true, &["docs_*"]));
        let tools = ToolCacheConfig {
            classes,
            overrides,
            ..ToolCacheConfig::default()
        };
        assert!(
            !resolve_policy("docs_*", &response_cache(3600, 0.0), &tools).cacheable,
            "the literal name `docs_*` resolves its exact entry"
        );
        assert!(!resolve_policy("docs_lookup", &response_cache(3600, 0.0), &tools).cacheable);
    }

    #[test]
    fn most_specific_override_pattern_wins() {
        let mut classes = BTreeMap::new();
        classes.insert("read_only".to_string(), class(true, &["docs_*"]));
        let mut overrides = BTreeMap::new();
        overrides.insert(
            "docs_*".to_string(),
            ToolOverride {
                cacheable: Some(true),
                ..ToolOverride::default()
            },
        );
        overrides.insert(
            "docs_secret_*".to_string(),
            ToolOverride {
                cacheable: Some(false),
                ..ToolOverride::default()
            },
        );
        let tools = ToolCacheConfig {
            classes,
            overrides,
            ..ToolCacheConfig::default()
        };
        assert!(
            !resolve_policy("docs_secret_dump", &response_cache(3600, 0.0), &tools).cacheable,
            "`docs_secret_*` (more literal characters) must beat `docs_*`"
        );
    }
}

#[cfg(test)]
#[path = "../../tests/unit/response_cache/tool_policy_tests.rs"]
mod policy_tests;

#[cfg(test)]
#[path = "../../tests/unit/response_cache/tool_tests.rs"]
mod coverage_tests;
