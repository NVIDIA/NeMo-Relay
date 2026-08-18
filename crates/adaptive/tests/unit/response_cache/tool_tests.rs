// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error-classification tests for the tool-result response cache.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use nemo_relay::api::runtime::ToolExecutionNextFn;
use nemo_relay::api::tool::ToolExecutionResult;

use super::*;
use crate::config::ResponseCacheConfig;
use crate::response_cache::config::{ToolCacheConfig, ToolClass, ToolOverride};
use crate::response_cache::store::{CacheEntry, CacheStore, InMemoryCacheStore};

#[test]
fn wildcard_matching_handles_literals_and_missing_middle_segments() {
    assert!(wildcard_match("docs_lookup", "docs_lookup"));
    assert!(!wildcard_match("docs_lookup", "docs_search"));
    assert!(!wildcard_match("a*b*c", "axc"));
}

#[derive(Default)]
struct FailingGetStore {
    get_calls: AtomicUsize,
    set_calls: AtomicUsize,
}

impl CacheStore for FailingGetStore {
    fn get<'a>(
        &'a self,
        _key: &'a str,
    ) -> crate::response_cache::store::BoxCacheFuture<'a, Option<Arc<CacheEntry>>> {
        self.get_calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async {
            Err(crate::error::AdaptiveError::Storage(
                "cache read unavailable".to_string(),
            ))
        })
    }

    fn set<'a>(
        &'a self,
        _key: &'a str,
        _entry: CacheEntry,
        _ttl: Duration,
    ) -> crate::response_cache::store::BoxCacheFuture<'a, ()> {
        self.set_calls.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Ok(()) })
    }

    fn health<'a>(&'a self) -> crate::response_cache::store::BoxCacheFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }

    fn backend_kind(&self) -> &'static str {
        "failing_test"
    }
}

fn cache_config() -> Arc<ResponseCacheConfig> {
    Arc::new(ResponseCacheConfig {
        namespace: "tool-cache-unit-tests".to_string(),
        ttl_seconds: 60,
        ..ResponseCacheConfig::default()
    })
}

fn counting_next(calls: Arc<AtomicUsize>, result: Json) -> ToolExecutionNextFn {
    Arc::new(move |_args| {
        let calls = Arc::clone(&calls);
        let result = result.clone();
        Box::pin(async move {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(result.into())
        })
    })
}

fn versioned_tools(class_version: &str, override_version: Option<&str>) -> Arc<ToolCacheConfig> {
    let classes = std::collections::BTreeMap::from([(
        "read_only".to_string(),
        ToolClass {
            cacheable: true,
            tool_version: Some(class_version.to_string()),
            members: vec!["docs_lookup".to_string()],
            ..ToolClass::default()
        },
    )]);
    let overrides = override_version
        .map(|version| {
            std::collections::BTreeMap::from([(
                "docs_lookup".to_string(),
                ToolOverride {
                    tool_version: Some(version.to_string()),
                    ..ToolOverride::default()
                },
            )])
        })
        .unwrap_or_default();
    Arc::new(ToolCacheConfig {
        enabled: true,
        classes,
        overrides,
        ..ToolCacheConfig::default()
    })
}

#[test]
fn conventional_tool_error_detection_is_deliberately_narrow() {
    assert!(is_error_shaped_tool_result(&serde_json::json!({
        "error": "upstream unavailable"
    })));
    assert!(is_error_shaped_tool_result(&serde_json::json!({
        "isError": true
    })));
    assert!(is_error_shaped_tool_result(&serde_json::json!({
        "is_error": true
    })));
    assert!(!is_error_shaped_tool_result(&serde_json::json!({
        "error": null
    })));
    assert!(!is_error_shaped_tool_result(&serde_json::json!({
        "status": "failed"
    })));
    assert!(!is_error_shaped_tool_result(&serde_json::json!("error")));
}

#[tokio::test]
async fn tool_cache_read_error_fails_open_without_writing() {
    let store = Arc::new(FailingGetStore::default());
    let calls = Arc::new(AtomicUsize::new(0));
    let next = counting_next(
        Arc::clone(&calls),
        Json::String("live tool result".to_string()),
    );
    let response_cache = cache_config();
    let tools = Arc::new(ToolCacheConfig {
        enabled: true,
        default: ToolClass {
            cacheable: true,
            ..ToolClass::default()
        },
        ..ToolCacheConfig::default()
    });

    let outcome = run_tool_cache(
        "docs_lookup".to_string(),
        serde_json::json!({"query": "response cache"}),
        next,
        store.clone(),
        response_cache,
        tools,
    )
    .await
    .expect("a cache read failure must not fail the tool call");

    assert_eq!(outcome.result, Json::String("live tool result".to_string()));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(store.get_calls.load(Ordering::SeqCst), 1);
    assert_eq!(store.set_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn stale_error_entries_are_not_replayed_when_error_caching_is_disabled() {
    let store: Arc<dyn CacheStore> = Arc::new(InMemoryCacheStore::new(1 << 20));
    let response_cache = Arc::new(ResponseCacheConfig {
        namespace: "tool-cache-stale-error-test".to_string(),
        ..ResponseCacheConfig::default()
    });
    let tools = Arc::new(ToolCacheConfig {
        enabled: true,
        default: ToolClass {
            cacheable: true,
            ..ToolClass::default()
        },
        ..ToolCacheConfig::default()
    });
    let args = serde_json::json!({"query": "relay"});
    let key = match build_tool_cache_key(
        &response_cache.namespace,
        "docs_lookup",
        None,
        &args,
        &[],
        false,
    ) {
        KeyOutcome::Key(key) => key,
        other => panic!("expected a cache key, got {other:?}"),
    };
    let ttl = Duration::from_secs(60);
    store
        .set(
            &key,
            CacheEntry::new(
                serde_json::json!({"is_error": true, "content": "stale"}),
                ttl,
                key.clone(),
                None,
                None,
            ),
            ttl,
        )
        .await
        .unwrap();

    let calls = Arc::new(AtomicUsize::new(0));
    let next: ToolExecutionNextFn = Arc::new({
        let calls = Arc::clone(&calls);
        move |_args| {
            let calls = Arc::clone(&calls);
            Box::pin(async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(serde_json::json!({"answer": "fresh"}).into())
            })
        }
    });
    let result = run_tool_cache(
        "docs_lookup".to_string(),
        args.clone(),
        Arc::clone(&next),
        Arc::clone(&store),
        Arc::clone(&response_cache),
        Arc::clone(&tools),
    )
    .await
    .unwrap();

    assert_eq!(result.result, serde_json::json!({"answer": "fresh"}));
    let hit = run_tool_cache(
        "docs_lookup".to_string(),
        args,
        next,
        store,
        response_cache,
        tools,
    )
    .await
    .unwrap();
    assert_eq!(hit.result, serde_json::json!({"answer": "fresh"}));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn cached_tool_results_preserve_annotations() {
    let store: Arc<dyn CacheStore> = Arc::new(InMemoryCacheStore::new(1 << 20));
    let response_cache = cache_config();
    let tools = Arc::new(ToolCacheConfig {
        enabled: true,
        default: ToolClass {
            cacheable: true,
            ..ToolClass::default()
        },
        ..ToolCacheConfig::default()
    });
    let calls = Arc::new(AtomicUsize::new(0));
    let next: ToolExecutionNextFn = Arc::new({
        let calls = Arc::clone(&calls);
        move |_args| {
            let calls = Arc::clone(&calls);
            Box::pin(async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(ToolExecutionResult::annotated(
                    serde_json::json!({"answer": "cached"}),
                    serde_json::json!({"source": "tool"}),
                ))
            })
        }
    });
    let args = serde_json::json!({"query": "relay"});

    let first = run_tool_cache(
        "docs_lookup".to_string(),
        args.clone(),
        Arc::clone(&next),
        Arc::clone(&store),
        Arc::clone(&response_cache),
        Arc::clone(&tools),
    )
    .await
    .unwrap();
    let hit = run_tool_cache(
        "docs_lookup".to_string(),
        args,
        next,
        store,
        response_cache,
        tools,
    )
    .await
    .unwrap();

    assert_eq!(first, hit);
    assert_eq!(hit.annotation, Some(serde_json::json!({"source": "tool"})));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn cache_error_policy_partitions_tool_keys() {
    let store = Arc::new(InMemoryCacheStore::new(1 << 20));
    let response_cache = cache_config();
    let opt_in_tools = Arc::new(ToolCacheConfig {
        enabled: true,
        cache_errors: true,
        default: ToolClass {
            cacheable: true,
            ..ToolClass::default()
        },
        ..ToolCacheConfig::default()
    });
    let default_tools = Arc::new(ToolCacheConfig {
        enabled: true,
        default: ToolClass {
            cacheable: true,
            ..ToolClass::default()
        },
        ..ToolCacheConfig::default()
    });
    let calls = Arc::new(AtomicUsize::new(0));
    let args = serde_json::json!({"query": "relay"});

    let error = run_tool_cache(
        "docs_lookup".to_string(),
        args.clone(),
        counting_next(
            Arc::clone(&calls),
            serde_json::json!({"error": "temporary outage"}),
        ),
        store.clone(),
        Arc::clone(&response_cache),
        opt_in_tools,
    )
    .await
    .unwrap();
    assert_eq!(
        error.result,
        serde_json::json!({"error": "temporary outage"})
    );

    let success = run_tool_cache(
        "docs_lookup".to_string(),
        args.clone(),
        counting_next(Arc::clone(&calls), serde_json::json!({"answer": "fresh"})),
        store.clone(),
        Arc::clone(&response_cache),
        Arc::clone(&default_tools),
    )
    .await
    .unwrap();
    assert_eq!(success.result, serde_json::json!({"answer": "fresh"}));

    let hit = run_tool_cache(
        "docs_lookup".to_string(),
        args,
        counting_next(
            Arc::clone(&calls),
            serde_json::json!({"answer": "unexpected"}),
        ),
        store,
        response_cache,
        default_tools,
    )
    .await
    .unwrap();
    assert_eq!(hit.result, serde_json::json!({"answer": "fresh"}));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn class_tool_version_partitions_keys_unless_an_override_replaces_it() {
    let store = Arc::new(InMemoryCacheStore::new(1 << 20));
    let response_cache = cache_config();
    let calls = Arc::new(AtomicUsize::new(0));
    let args = serde_json::json!({"query": "relay"});

    let class_v1 = run_tool_cache(
        "docs_lookup".to_string(),
        args.clone(),
        counting_next(
            Arc::clone(&calls),
            serde_json::json!({"version": "class-v1"}),
        ),
        store.clone(),
        Arc::clone(&response_cache),
        versioned_tools("class-v1", None),
    )
    .await
    .unwrap();
    let class_v2 = run_tool_cache(
        "docs_lookup".to_string(),
        args.clone(),
        counting_next(
            Arc::clone(&calls),
            serde_json::json!({"version": "class-v2"}),
        ),
        store.clone(),
        Arc::clone(&response_cache),
        versioned_tools("class-v2", None),
    )
    .await
    .unwrap();
    assert_eq!(class_v1.result, serde_json::json!({"version": "class-v1"}));
    assert_eq!(class_v2.result, serde_json::json!({"version": "class-v2"}));

    let overridden = run_tool_cache(
        "docs_lookup".to_string(),
        args.clone(),
        counting_next(
            Arc::clone(&calls),
            serde_json::json!({"version": "override"}),
        ),
        store.clone(),
        Arc::clone(&response_cache),
        versioned_tools("class-v1", Some("tool-v1")),
    )
    .await
    .unwrap();
    let override_hit = run_tool_cache(
        "docs_lookup".to_string(),
        args,
        counting_next(
            Arc::clone(&calls),
            serde_json::json!({"version": "unexpected"}),
        ),
        store,
        response_cache,
        versioned_tools("class-v2", Some("tool-v1")),
    )
    .await
    .unwrap();
    assert_eq!(
        overridden.result,
        serde_json::json!({"version": "override"})
    );
    assert_eq!(
        override_hit.result,
        serde_json::json!({"version": "override"})
    );
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn per_tool_override_ttl_reaches_the_stored_entry() {
    let store = Arc::new(InMemoryCacheStore::new(1 << 20));
    let response_cache = cache_config();
    let classes = std::collections::BTreeMap::from([(
        "read_only".to_string(),
        ToolClass {
            cacheable: true,
            ttl_seconds: Some(17),
            members: vec!["docs_lookup".to_string()],
            ..ToolClass::default()
        },
    )]);
    let overrides = std::collections::BTreeMap::from([(
        "docs_lookup".to_string(),
        ToolOverride {
            ttl_seconds: Some(23),
            ..ToolOverride::default()
        },
    )]);
    let tools = Arc::new(ToolCacheConfig {
        enabled: true,
        classes,
        overrides,
        ..ToolCacheConfig::default()
    });
    let args = serde_json::json!({"query": "relay"});

    run_tool_cache(
        "docs_lookup".to_string(),
        args.clone(),
        counting_next(
            Arc::new(AtomicUsize::new(0)),
            serde_json::json!({"answer": "cached"}),
        ),
        store.clone(),
        Arc::clone(&response_cache),
        tools,
    )
    .await
    .unwrap();

    let key = match build_tool_cache_key(
        &response_cache.namespace,
        "docs_lookup",
        None,
        &args,
        &[],
        false,
    ) {
        KeyOutcome::Key(key) => key,
        other => panic!("expected tool key, got {other:?}"),
    };
    let entry = store
        .get(&key)
        .await
        .unwrap()
        .expect("the successful result should be stored");
    assert_eq!(
        entry.expires_unix_ms - entry.created_unix_ms,
        Duration::from_secs(23).as_millis() as u64,
        "the override TTL, not the class or parent TTL, controls the stored entry"
    );
}
