// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for response-cache storage backends in the NeMo Relay adaptive crate.

use super::*;
use serde_json::json;

fn entry(key: &str, created: u64, expires: u64) -> CacheEntry {
    CacheEntry {
        response: json!({ "answer": key }),
        created_unix_ms: created,
        expires_unix_ms: expires,
        key_hash: key.to_string(),
        model_name: None,
        provider_name: None,
    }
}

const BIG: usize = 1 << 20; // 1 MiB — never evicts in these tests

#[test]
fn ttl_arithmetic_is_milliseconds() {
    // Pin the seconds->milliseconds conversion: a units regression would
    // expire entries 1000x early or late.
    let entry = CacheEntry::new(
        json!({"ok": true}),
        Duration::from_secs(60),
        "sha256:t".to_string(),
        None,
        None,
    );
    assert_eq!(entry.expires_unix_ms - entry.created_unix_ms, 60_000);
}

#[tokio::test]
async fn same_key_replacement_swaps_content_and_accounting() {
    let store = InMemoryCacheStore::new(BIG);
    let small = entry("a", 100, u64::MAX);
    let small_size = entry_size(&small);
    store.set("k", small, Duration::MAX).await.unwrap();

    let big = CacheEntry {
        response: json!({ "answer": "a much longer replacement body".repeat(4) }),
        ..entry("a", 200, u64::MAX)
    };
    let big_size = entry_size(&big);
    assert_ne!(small_size, big_size);
    store.set("k", big, Duration::MAX).await.unwrap();

    let got = store.get("k").await.unwrap().expect("entry present");
    assert_eq!(
        got.response["answer"],
        json!("a much longer replacement body".repeat(4)),
        "a same-key set must serve the replacement content"
    );
    assert_eq!(
        store.total_bytes(),
        big_size,
        "replacement must swap the accounted size, not add to it"
    );
}

#[tokio::test]
async fn expired_entry_reads_as_absent_and_is_reaped() {
    let store = InMemoryCacheStore::new(BIG);
    // expires_unix_ms = 1 (1ms after the epoch) is firmly in the past.
    store
        .set("k", entry("k", 0, 1), Duration::from_secs(0))
        .await
        .unwrap();
    assert!(
        store.get("k").await.unwrap().is_none(),
        "expired entry must read as absent"
    );
    assert!(store.is_empty(), "reading an expired entry should reap it");
    assert_eq!(store.total_bytes(), 0, "reaping must reclaim bytes");
}

#[tokio::test]
async fn eviction_drops_the_oldest_entry_when_over_the_byte_budget() {
    // Budget holds exactly two of these entries; the third forces eviction.
    let one = entry("a", 100, u64::MAX);
    let size = entry_size(&one);
    let store = InMemoryCacheStore::new(size * 2);

    store.set("a", one, Duration::MAX).await.unwrap();
    store
        .set("b", entry("b", 200, u64::MAX), Duration::MAX)
        .await
        .unwrap();
    // Third insert exceeds max_bytes -> evict oldest (created = 100 -> "a").
    store
        .set("c", entry("c", 300, u64::MAX), Duration::MAX)
        .await
        .unwrap();

    assert!(
        store.get("a").await.unwrap().is_none(),
        "oldest entry should be evicted once over the byte budget"
    );
    assert!(store.get("b").await.unwrap().is_some());
    assert!(store.get("c").await.unwrap().is_some());
    assert!(
        store.total_bytes() <= size * 2,
        "must stay within the budget"
    );
}

#[tokio::test]
async fn a_refreshed_entry_is_not_evicted_by_its_stale_queue_node() {
    // "a" is inserted first, then refreshed; its original queue node is
    // stale. Eviction must skip that node and drop the true oldest ("b"),
    // not the refreshed "a".
    let one = entry("a", 100, u64::MAX);
    let size = entry_size(&one);
    let store = InMemoryCacheStore::new(size * 2);

    store.set("a", one, Duration::MAX).await.unwrap();
    store
        .set("b", entry("b", 200, u64::MAX), Duration::MAX)
        .await
        .unwrap();
    store
        .set("a", entry("a", 300, u64::MAX), Duration::MAX)
        .await
        .unwrap();
    store
        .set("c", entry("c", 400, u64::MAX), Duration::MAX)
        .await
        .unwrap();

    assert!(
        store.get("b").await.unwrap().is_none(),
        "the oldest live entry must be evicted"
    );
    assert!(
        store.get("a").await.unwrap().is_some(),
        "a refreshed entry must not be evicted through its stale node"
    );
    assert!(store.get("c").await.unwrap().is_some());
}

#[tokio::test]
async fn an_entry_larger_than_the_budget_is_not_cached_and_keeps_existing_entries() {
    let small = entry("a", 100, u64::MAX);
    let budget = entry_size(&small);
    let store = InMemoryCacheStore::new(budget);
    store.set("a", small, Duration::MAX).await.unwrap();

    // An entry whose size alone exceeds max_bytes must be skipped — not stored
    // (breaching the cap) and not flushing the cache to make room it can't use.
    let oversized = CacheEntry::new(
        json!({ "blob": "x".repeat(budget * 10 + 100) }),
        Duration::MAX,
        "b".to_string(),
        None,
        None,
    );
    store.set("b", oversized, Duration::MAX).await.unwrap();

    assert!(
        store.get("b").await.unwrap().is_none(),
        "an oversized entry must not be cached"
    );
    assert!(
        store.get("a").await.unwrap().is_some(),
        "an oversized set must not flush existing entries"
    );
    assert!(store.total_bytes() <= budget, "the byte budget must hold");

    // A fresher answer too large to store must still invalidate the stale
    // one — otherwise the old answer keeps serving after a newer one exists.
    let refresh = CacheEntry::new(
        json!({ "blob": "x".repeat(budget * 10 + 100) }),
        Duration::MAX,
        "a".to_string(),
        None,
        None,
    );
    store.set("a", refresh, Duration::MAX).await.unwrap();
    assert!(
        store.get("a").await.unwrap().is_none(),
        "the stale entry must not outlive a fresher, unstorable answer"
    );
    assert_eq!(store.total_bytes(), 0);
}
