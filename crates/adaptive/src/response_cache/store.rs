// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! General-purpose cache store with expiry for the response cache.
//!
//! This is intentionally separate from the adaptive learning-data storage
//! ([`crate::storage`]), which is shaped for specific learning records rather
//! than a "store any value with a TTL" cache. The trait keeps the same
//! object-safe, boxed-future style; the Redis backend lives behind the
//! `redis-backend` feature.

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

use crate::config::ResponseCacheConfig;
use crate::error::{AdaptiveError, Result};

/// Boxed, `Send` future returned by [`CacheStore`] operations.
pub type BoxCacheFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + Send + 'a>>;

/// Cache schema version, folded into every cache key by the key derivation.
///
/// Bump this to invalidate every previously stored entry when the entry shape
/// or the key derivation changes in an incompatible way: old entries become
/// unreachable under the new keys.
#[doc(hidden)]
pub const CACHE_SCHEMA_VERSION: u32 = 1;

/// Wall-clock milliseconds since the Unix epoch.
pub fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|delta| delta.as_millis() as u64)
        .unwrap_or(0)
}

/// One cached response together with the metadata needed to expire it and
/// report savings on a reuse.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// The saved answer, returned **unchanged** on a hit (usage intact, so a
    /// reuse is shape-identical to a live call). Its token counts are read to
    /// report savings on the cache-hit mark — the response is never mutated.
    pub response: Json,
    /// When the entry was stored.
    pub created_unix_ms: u64,
    /// When the entry expires (`created + ttl`).
    pub expires_unix_ms: u64,
    /// The full cache key fingerprint (`"sha256:…"`).
    pub key_hash: String,
    /// Optional model name recorded for diagnostics.
    pub model_name: Option<String>,
    /// Optional provider/family name recorded for diagnostics.
    pub provider_name: Option<String>,
}

impl CacheEntry {
    /// Builds an entry that expires `ttl` from now.
    pub fn new(
        response: Json,
        ttl: Duration,
        key_hash: String,
        model_name: Option<String>,
        provider_name: Option<String>,
    ) -> Self {
        let created = now_unix_ms();
        Self {
            response,
            created_unix_ms: created,
            expires_unix_ms: created
                .saturating_add(u64::try_from(ttl.as_millis()).unwrap_or(u64::MAX)),
            key_hash,
            model_name,
            provider_name,
        }
    }

    /// Whether the entry is expired relative to `now_ms`.
    pub fn is_expired(&self, now_ms: u64) -> bool {
        now_ms >= self.expires_unix_ms
    }
}

/// A general "store a value with an expiry" cache.
///
/// Implementations must be cheap to clone behind an [`std::sync::Arc`] and must
/// not block: the in-memory backend locks only for the duration of a
/// synchronous map operation and never holds the lock across an `await`.
pub trait CacheStore: Send + Sync + 'static {
    /// Looks up an entry. Returns `Ok(None)` when the key is absent or expired.
    /// Entries come back shared behind an [`Arc`] so a lookup never deep-clones
    /// the stored response (the in-memory backend holds its lock O(1)).
    fn get<'a>(&'a self, key: &'a str) -> BoxCacheFuture<'a, Option<Arc<CacheEntry>>>;
    /// Stores an entry under `key`. `ttl` is supplied for backends with native
    /// expiry (e.g. Redis `SETEX`); the in-memory backend uses
    /// [`CacheEntry::expires_unix_ms`].
    fn set<'a>(&'a self, key: &'a str, entry: CacheEntry, ttl: Duration) -> BoxCacheFuture<'a, ()>;
    /// Cheap reachability check surfaced by `doctor`/validation.
    fn health<'a>(&'a self) -> BoxCacheFuture<'a, ()>;
    /// Backend kind label for telemetry (e.g. `"in_memory"`).
    fn backend_kind(&self) -> &'static str;
}

/// One stored entry plus the byte size it accounts for. The entry is shared
/// behind an [`Arc`] so `get` hands out a reference instead of a deep clone.
struct StoredEntry {
    entry: Arc<CacheEntry>,
    size: usize,
    /// Matches this entry's node in [`Inner::order`]; a node whose generation
    /// no longer matches is stale (the key was replaced or reaped) and is
    /// skipped on pop.
    generation: u64,
}

/// Locked map plus a running byte total, guarded together so the budget never
/// drifts from the contents. `order` is an insertion-order queue so eviction
/// pops the oldest entry in O(1) instead of scanning the map; stale nodes
/// (replaced or reaped keys) are skipped lazily and compacted when the queue
/// outgrows the map.
#[derive(Default)]
struct Inner {
    map: HashMap<String, StoredEntry>,
    total_bytes: usize,
    order: VecDeque<(u64, String)>,
    next_generation: u64,
}

/// Process-local cache backed by a locked map.
///
/// Drops entries lazily when they are read after expiry, and is bounded by a
/// **total-bytes** budget (`max_bytes`) so a few large completions cannot OOM
/// the process. Eviction is oldest-first by `created_unix_ms`. This is the
/// single-process path; a shared deployment uses the Redis-backed
/// [`CacheStore`].
pub struct InMemoryCacheStore {
    inner: Mutex<Inner>,
    max_bytes: usize,
}

/// Approximate resident size: response bytes + metadata strings (key hash is
/// held twice) + fixed slot overhead. An OOM guard, not exact accounting.
fn entry_size(entry: &CacheEntry) -> usize {
    const ENTRY_OVERHEAD: usize = 256;
    serde_json::to_vec(&entry.response)
        .map(|bytes| bytes.len())
        .unwrap_or(0)
        + entry.key_hash.len() * 2
        + entry.model_name.as_deref().map_or(0, str::len)
        + entry.provider_name.as_deref().map_or(0, str::len)
        + ENTRY_OVERHEAD
}

impl InMemoryCacheStore {
    /// Creates a store bounded by `max_bytes` (minimum 1).
    pub fn new(max_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(Inner::default()),
            max_bytes: max_bytes.max(1),
        }
    }

    /// Current number of stored entries (including not-yet-reaped expired ones).
    pub fn len(&self) -> usize {
        self.inner.lock().map(|guard| guard.map.len()).unwrap_or(0)
    }

    /// Whether the store currently holds no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Current total accounted bytes.
    pub fn total_bytes(&self) -> usize {
        self.inner
            .lock()
            .map(|guard| guard.total_bytes)
            .unwrap_or(0)
    }
}

/// Removes the oldest entry (smallest `created_unix_ms`), updating the byte
/// total. Returns whether an entry was removed.
fn evict_oldest(inner: &mut Inner) -> bool {
    while let Some((generation, key)) = inner.order.pop_front() {
        let live = inner
            .map
            .get(&key)
            .is_some_and(|stored| stored.generation == generation);
        if !live {
            continue;
        }
        if let Some(stored) = inner.map.remove(&key) {
            inner.total_bytes = inner.total_bytes.saturating_sub(stored.size);
        }
        return true;
    }
    false
}

impl CacheStore for InMemoryCacheStore {
    fn get<'a>(&'a self, key: &'a str) -> BoxCacheFuture<'a, Option<Arc<CacheEntry>>> {
        Box::pin(async move {
            let now = now_unix_ms();
            let mut guard = self
                .inner
                .lock()
                .map_err(|err| AdaptiveError::Internal(format!("cache lock poisoned: {err}")))?;
            match guard.map.get(key) {
                Some(stored) if !stored.entry.is_expired(now) => {
                    Ok(Some(Arc::clone(&stored.entry)))
                }
                Some(_) => {
                    if let Some(stored) = guard.map.remove(key) {
                        guard.total_bytes = guard.total_bytes.saturating_sub(stored.size);
                    }
                    Ok(None)
                }
                None => Ok(None),
            }
        })
    }

    fn set<'a>(
        &'a self,
        key: &'a str,
        entry: CacheEntry,
        _ttl: Duration,
    ) -> BoxCacheFuture<'a, ()> {
        Box::pin(async move {
            let size = entry_size(&entry);
            let mut guard = self
                .inner
                .lock()
                .map_err(|err| AdaptiveError::Internal(format!("cache lock poisoned: {err}")))?;
            // Never store past the whole budget, but still drop the stale
            // entry — a fresher answer exists even if it cannot be stored.
            if size > self.max_bytes {
                if let Some(previous) = guard.map.remove(key) {
                    guard.total_bytes = guard.total_bytes.saturating_sub(previous.size);
                }
                return Ok(());
            }

            // Replacing an existing key frees its old size first.
            if let Some(previous) = guard.map.remove(key) {
                guard.total_bytes = guard.total_bytes.saturating_sub(previous.size);
            }

            // Evict oldest-first until the new entry fits the byte budget
            // (always keep room for the entry itself).
            while !guard.map.is_empty()
                && guard.total_bytes + size > self.max_bytes
                && evict_oldest(&mut guard)
            {}

            guard.total_bytes += size;
            let generation = guard.next_generation;
            guard.next_generation += 1;
            guard.map.insert(
                key.to_string(),
                StoredEntry {
                    entry: Arc::new(entry),
                    size,
                    generation,
                },
            );
            guard.order.push_back((generation, key.to_string()));
            // Replacing keys leaves stale nodes behind; compact before the
            // queue can outgrow the live map by more than a constant factor.
            let inner = &mut *guard;
            if inner.order.len() > inner.map.len() * 2 + 64 {
                let map = &inner.map;
                let order = std::mem::take(&mut inner.order);
                inner.order = order
                    .into_iter()
                    .filter(|(generation, key)| {
                        map.get(key)
                            .is_some_and(|stored| stored.generation == *generation)
                    })
                    .collect();
            }
            Ok(())
        })
    }

    fn health<'a>(&'a self) -> BoxCacheFuture<'a, ()> {
        Box::pin(async move {
            // Touch the lock so a poisoned mutex surfaces as unhealthy.
            self.inner
                .lock()
                .map(|_| ())
                .map_err(|err| AdaptiveError::Internal(format!("cache lock poisoned: {err}")))
        })
    }

    fn backend_kind(&self) -> &'static str {
        "in_memory"
    }
}

/// Redis-backed [`CacheStore`] for sharing one cache across processes / a team.
///
/// Uses Redis's native expiry (`SET … PX <ttl_ms>`) so entries drop themselves,
/// and a configurable key prefix. The connection is a [`ConnectionManager`]
/// (cheap to clone, auto-reconnecting), cloned before each await so the store
/// never blocks. Mirrors the connection pattern of [`crate::redis::RedisBackend`].
#[cfg(feature = "redis-backend")]
pub struct RedisCacheStore {
    conn: redis::aio::ConnectionManager,
    key_prefix: String,
}

#[cfg(feature = "redis-backend")]
impl RedisCacheStore {
    /// Connects to Redis and returns a new store.
    ///
    /// # Errors
    /// Returns [`AdaptiveError::Storage`] when the client or connection fails.
    pub async fn new(url: &str, key_prefix: impl Into<String>) -> Result<Self> {
        let (_client, conn) = crate::redis::connect(url).await?;
        Ok(Self {
            conn,
            key_prefix: key_prefix.into(),
        })
    }

    /// Removes an entry if present. Kept off [`CacheStore`]: the runtime never
    /// deletes (entries expire via TTL); this exists for manual cleanup of a
    /// shared Redis.
    pub async fn delete(&self, key: &str) -> Result<()> {
        with_redis_deadline("DEL", async move {
            let mut conn = self.conn.clone();
            let full_key = self.full_key(key);
            redis::cmd("DEL")
                .arg(&full_key)
                .exec_async(&mut conn)
                .await
                .map_err(|err| AdaptiveError::Storage(format!("redis DEL: {err}")))?;
            Ok(())
        })
        .await
    }

    fn full_key(&self, key: &str) -> String {
        format!("{}{}", self.key_prefix, key)
    }
}

/// A hung Redis peer must degrade to fail-open, not block the request.
#[cfg(feature = "redis-backend")]
const REDIS_OP_TIMEOUT: Duration = Duration::from_secs(2);

#[cfg(feature = "redis-backend")]
async fn with_redis_deadline<T>(
    operation: &'static str,
    future: impl Future<Output = Result<T>>,
) -> Result<T> {
    match tokio::time::timeout(REDIS_OP_TIMEOUT, future).await {
        Ok(result) => result,
        Err(_) => Err(AdaptiveError::Storage(format!(
            "redis {operation}: timed out after {REDIS_OP_TIMEOUT:?}"
        ))),
    }
}

#[cfg(feature = "redis-backend")]
impl CacheStore for RedisCacheStore {
    fn get<'a>(&'a self, key: &'a str) -> BoxCacheFuture<'a, Option<Arc<CacheEntry>>> {
        Box::pin(with_redis_deadline("GET", async move {
            let mut conn = self.conn.clone();
            let full_key = self.full_key(key);
            let maybe_json: Option<String> = redis::cmd("GET")
                .arg(&full_key)
                .query_async(&mut conn)
                .await
                .map_err(|err| AdaptiveError::Storage(format!("redis GET: {err}")))?;
            // Redis already dropped the key if it expired, so any value present
            // is live.
            match maybe_json {
                Some(json) => {
                    let entry: CacheEntry =
                        serde_json::from_str(&json).map_err(AdaptiveError::Serialization)?;
                    // The entry's own stamp is authoritative over Redis PX.
                    if entry.is_expired(now_unix_ms()) {
                        return Ok(None);
                    }
                    Ok(Some(Arc::new(entry)))
                }
                None => Ok(None),
            }
        }))
    }

    fn set<'a>(&'a self, key: &'a str, entry: CacheEntry, ttl: Duration) -> BoxCacheFuture<'a, ()> {
        Box::pin(with_redis_deadline("SET", async move {
            let mut conn = self.conn.clone();
            let full_key = self.full_key(key);
            let json = serde_json::to_string(&entry).map_err(AdaptiveError::Serialization)?;
            let ttl_ms = u64::try_from(ttl.as_millis()).unwrap_or(u64::MAX).max(1); // PX requires a positive TTL
            redis::cmd("SET")
                .arg(&full_key)
                .arg(json)
                .arg("PX")
                .arg(ttl_ms)
                .exec_async(&mut conn)
                .await
                .map_err(|err| AdaptiveError::Storage(format!("redis SET: {err}")))?;
            Ok(())
        }))
    }

    fn health<'a>(&'a self) -> BoxCacheFuture<'a, ()> {
        Box::pin(with_redis_deadline("PING", async move {
            let mut conn = self.conn.clone();
            redis::cmd("PING")
                .exec_async(&mut conn)
                .await
                .map_err(|err| AdaptiveError::Storage(format!("redis PING: {err}")))?;
            Ok(())
        }))
    }

    fn backend_kind(&self) -> &'static str {
        "redis"
    }
}

/// Builds the configured backend and runs its reachability check.
///
/// Intended for `nemo-relay doctor` / health surfaces: it constructs a
/// throwaway store from the response-cache config and calls
/// [`CacheStore::health`], returning the backend kind label on success or a
/// human-readable error.
pub async fn check_backend_health(
    config: &ResponseCacheConfig,
) -> std::result::Result<String, String> {
    let store = build_store(config).await.map_err(|err| err.to_string())?;
    store.health().await.map_err(|err| err.to_string())?;
    Ok(store.backend_kind().to_string())
}

/// Builds the response-cache backend from config.
///
/// Returns the boxed [`CacheStore`] used by the intercept and the `doctor`
/// health check. Redis support is gated behind the `redis-backend` feature.
pub(crate) async fn build_store(config: &ResponseCacheConfig) -> Result<Arc<dyn CacheStore>> {
    match config.backend.kind.as_str() {
        "in_memory" => Ok(Arc::new(InMemoryCacheStore::new(
            config.backend.max_bytes(),
        ))),
        "redis" => build_redis_store(config).await,
        other => Err(AdaptiveError::InvalidConfig(format!(
            "response_cache: unknown backend kind '{other}'"
        ))),
    }
}

#[cfg(feature = "redis-backend")]
async fn build_redis_store(config: &ResponseCacheConfig) -> Result<Arc<dyn CacheStore>> {
    use crate::response_cache::store::RedisCacheStore;

    let url = config
        .backend
        .config
        .get("url")
        .and_then(Json::as_str)
        .ok_or_else(|| {
            AdaptiveError::InvalidConfig(
                "response_cache: redis backend requires backend.config.url".to_string(),
            )
        })?;
    let key_prefix = config
        .backend
        .config
        .get("key_prefix")
        .and_then(Json::as_str)
        .unwrap_or("nemo-relay:llm-cache:");
    let store = RedisCacheStore::new(url, key_prefix)
        .await
        .map_err(|err| AdaptiveError::Storage(format!("response_cache: {err}")))?;
    Ok(Arc::new(store))
}

#[cfg(not(feature = "redis-backend"))]
async fn build_redis_store(_config: &ResponseCacheConfig) -> Result<Arc<dyn CacheStore>> {
    Err(AdaptiveError::InvalidConfig(
        "response_cache: backend.kind = \"redis\" requires building with the 'redis-backend' \
         feature"
            .to_string(),
    ))
}

#[cfg(test)]
mod tests {
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
}
