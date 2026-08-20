// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for response-cache storage backends in the NeMo Relay adaptive crate.

use super::*;
use serde_json::json;

#[cfg(feature = "redis-backend")]
use std::io::{Read, Write};
#[cfg(feature = "redis-backend")]
use std::net::{TcpListener, TcpStream};
#[cfg(feature = "redis-backend")]
use std::thread;
#[cfg(feature = "redis-backend")]
use std::time::Duration;

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

#[cfg(feature = "redis-backend")]
fn read_redis_command(stream: &mut TcpStream) -> Vec<u8> {
    fn read_line(stream: &mut TcpStream, request: &mut Vec<u8>) -> String {
        let start = request.len();
        loop {
            let mut byte = [0_u8; 1];
            stream.read_exact(&mut byte).expect("read RESP command");
            request.push(byte[0]);
            if request.ends_with(b"\r\n") {
                return std::str::from_utf8(&request[start..request.len() - 2])
                    .expect("RESP command must be UTF-8")
                    .to_string();
            }
        }
    }

    let mut request = Vec::new();
    let count = read_line(stream, &mut request)
        .strip_prefix('*')
        .expect("RESP command array")
        .parse::<usize>()
        .expect("RESP command count");
    for _ in 0..count {
        let length = read_line(stream, &mut request)
            .strip_prefix('$')
            .expect("RESP bulk string")
            .parse::<usize>()
            .expect("RESP bulk string length");
        let mut argument = vec![0_u8; length + 2];
        stream
            .read_exact(&mut argument)
            .expect("RESP bulk string value");
        assert!(argument.ends_with(b"\r\n"));
        request.extend(argument);
    }
    request
}

#[cfg(feature = "redis-backend")]
fn start_redis_test_server(response: Vec<u8>) -> (String, thread::JoinHandle<Vec<u8>>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind test Redis peer");
    let url = format!(
        "redis://{}/",
        listener.local_addr().expect("test Redis address")
    );
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept Redis client");
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("set Redis test peer read timeout");
        // redis-rs identifies itself with two `CLIENT SETINFO` commands before
        // it allows normal commands on a new connection.
        for _ in 0..2 {
            let setup = read_redis_command(&mut stream);
            assert!(setup.windows(6).any(|window| window == b"CLIENT"));
            stream
                .write_all(b"+OK\r\n")
                .expect("acknowledge Redis client setup");
        }
        let command = read_redis_command(&mut stream);
        stream.write_all(&response).expect("write Redis response");
        stream.flush().expect("flush Redis response");
        command
    });
    (url, server)
}

#[cfg(feature = "redis-backend")]
#[tokio::test(start_paused = true)]
async fn redis_initialization_times_out_for_a_silent_peer() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("redis://{}/", listener.local_addr().unwrap());
    let started = tokio::time::Instant::now();

    let err = match RedisCacheStore::new(&url, "test:").await {
        Ok(_) => panic!("a silent Redis peer must not complete initialization"),
        Err(err) => err,
    };

    assert!(
        matches!(
            err,
            AdaptiveError::Storage(ref message)
                if message == "redis CONNECT: timed out after 2s"
        ),
        "initialization must use the response-cache Redis deadline: {err}"
    );
    assert_eq!(
        tokio::time::Instant::now().duration_since(started),
        REDIS_OP_TIMEOUT
    );
}

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

#[tokio::test]
async fn repeated_replacement_compacts_stale_insertion_order_nodes() {
    let store = InMemoryCacheStore::new(BIG);
    for generation in 0..70 {
        store
            .set(
                "stable-key",
                entry("stable-key", generation, u64::MAX),
                Duration::MAX,
            )
            .await
            .unwrap();
    }

    let guard = store.inner.lock().unwrap();
    assert_eq!(guard.map.len(), 1);
    assert_eq!(guard.order.len(), 4);
    assert_eq!(guard.next_generation, 70);
}

#[test]
fn evicting_an_empty_queue_is_a_noop() {
    // An eviction loop can reach an empty queue after stale nodes have been
    // skipped. It must report that nothing was removed rather than underflowing
    // the byte accounting.
    let mut inner = Inner::default();
    assert!(!evict_oldest(&mut inner));
    assert!(inner.map.is_empty());
    assert_eq!(inner.total_bytes, 0);
}

#[tokio::test]
async fn an_unknown_backend_is_rejected_before_initialization() {
    let mut config = ResponseCacheConfig::default();
    config.backend.kind = "not-a-cache".to_string();

    let error = match build_store(&config).await {
        Ok(_) => panic!("an unknown response-cache backend must be rejected"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        AdaptiveError::InvalidConfig(message)
            if message == "response_cache: unknown backend kind 'not-a-cache'"
    ));
}

#[cfg(feature = "redis-backend")]
#[tokio::test]
async fn redis_backend_requires_a_url_before_connecting() {
    // This validates configuration locally and never attempts a network
    // connection, so it remains deterministic in the unit-test suite.
    let mut config = ResponseCacheConfig::default();
    config.backend.kind = "redis".to_string();

    let error = match build_store(&config).await {
        Ok(_) => panic!("a Redis backend without a URL must be rejected"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        AdaptiveError::InvalidConfig(message)
            if message == "response_cache: redis backend requires backend.config.url"
    ));
}

#[cfg(feature = "redis-backend")]
#[tokio::test]
async fn redis_get_treats_an_entry_past_its_own_expiry_as_a_miss() {
    // Redis can retain a value briefly longer than the response-cache TTL.
    // The entry stamp remains authoritative, so a stale serialized entry must
    // not be served even when Redis returns it.
    let expired = entry("expired", 0, 1);
    let encoded = serde_json::to_vec(&expired).expect("serialize cache entry");
    let mut response = format!("${}\r\n", encoded.len()).into_bytes();
    response.extend(encoded);
    response.extend(b"\r\n");
    let (url, server) = start_redis_test_server(response);

    let store = RedisCacheStore::new(&url, "response-cache:")
        .await
        .expect("connect test Redis peer");
    assert!(
        store.get("expired").await.expect("Redis GET").is_none(),
        "an entry whose embedded expiry elapsed must be a miss"
    );

    let command = server.join().expect("test Redis server");
    assert!(command.windows(3).any(|window| window == b"GET"));
    assert!(
        command
            .windows(b"response-cache:expired".len())
            .any(|window| window == b"response-cache:expired")
    );
}

#[cfg(feature = "redis-backend")]
#[tokio::test]
async fn configured_redis_backend_pings_and_reports_its_kind() {
    // This minimal RESP peer validates the configured store's operational
    // health path without relying on a host Redis service.
    let (url, server) = start_redis_test_server(b"+PONG\r\n".to_vec());
    let mut config = ResponseCacheConfig::default();
    config.backend.kind = "redis".to_string();
    config
        .backend
        .config
        .insert("url".to_string(), Json::String(url));

    let store = build_store(&config)
        .await
        .expect("configured Redis backend builds");
    assert_eq!(store.backend_kind(), "redis");
    store.health().await.expect("Redis PING succeeds");

    let command = server.join().expect("test Redis server");
    assert!(command.windows(4).any(|window| window == b"PING"));
}
