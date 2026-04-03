// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Declarative proxy configuration with buffered settings and lazy creation.
//!
//! This module provides a high-level API for configuring and managing a
//! [`NexusProxy`] via simple setter functions. Settings are buffered in a
//! [`ProxyConfig`] and the proxy is materialized lazily when [`ensure_proxy()`]
//! is called.
//!
//! The [`ProxyManager`] struct is stored in the global context's extensions map
//! under the key [`PROXY_EXTENSION_KEY`] (`"proxy"`). All public functions in
//! this module operate on that singleton.
//!
//! # Example
//!
//! ```rust,no_run
//! use nvidia_nat_nexus_proxy::manager::*;
//! use nvidia_nat_nexus_proxy::storage::InMemoryBackend;
//! use nvidia_nat_nexus_proxy::AnyBackend;
//!
//! // Buffer configuration
//! set_use_proxy(true);
//! set_proxy_backend(Box::new(InMemoryBackend::new()) as AnyBackend);
//! set_dynamo_intercept(true);
//!
//! // Lazily create and register the proxy
//! // await ensure_proxy().unwrap();
//! ```

use nvidia_nat_nexus_core::global_context;

use crate::context_helpers::resolve_agent_id;
use crate::error::{ProxyError, Result};
use crate::learner::LatencySensitivityLearner;
use crate::proxy::NexusProxy;
use crate::storage::{AnyBackend, InMemoryBackend};
use crate::trie::SensitivityConfig;

/// Key used to store the [`ProxyManager`] in the global context extensions map.
pub const PROXY_EXTENSION_KEY: &str = "proxy";

/// Buffered proxy configuration. Setters modify this; [`ensure_proxy()`] materializes from it.
#[derive(Default)]
pub struct ProxyConfig {
    /// Whether the proxy should be active.
    pub enabled: bool,
    /// The storage backend to use. If `None`, defaults to [`InMemoryBackend`] at ensure time.
    pub backend: Option<AnyBackend>,
    /// Sensitivity scoring configuration with 4-signal weights.
    pub sensitivity_config: SensitivityConfig,
    /// Whether to register the DynamoIntercept for AgentHints injection.
    pub dynamo_intercept: bool,
}

/// Manages the declarative proxy lifecycle.
///
/// Holds the buffered [`ProxyConfig`], an optional materialized [`NexusProxy`],
/// and a flag tracking whether the proxy is currently registered with the Nexus
/// runtime.
pub struct ProxyManager {
    pub(crate) config: ProxyConfig,
    pub(crate) proxy: Option<NexusProxy>,
    pub(crate) registered: bool,
}

impl ProxyManager {
    /// Creates a new ProxyManager with default configuration.
    pub fn new() -> Self {
        Self {
            config: ProxyConfig::default(),
            proxy: None,
            registered: false,
        }
    }
}

impl Default for ProxyManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Ensures a [`ProxyManager`] exists in the global context extensions map.
/// Creates one with default configuration if absent.
fn ensure_manager_exists() {
    let ctx = global_context();
    let mut guard = ctx.write().unwrap_or_else(|e| e.into_inner());
    if guard
        .get_extension::<ProxyManager>(PROXY_EXTENSION_KEY)
        .is_none()
    {
        guard.set_extension(PROXY_EXTENSION_KEY, ProxyManager::new());
    }
}

/// Acquires a read lock on the global context and calls `f` with a reference
/// to the [`ProxyManager`]. Panics if the ProxyManager is not in the extensions map.
#[cfg(test)]
fn with_manager<F, R>(f: F) -> R
where
    F: FnOnce(&ProxyManager) -> R,
{
    let ctx = global_context();
    let guard = ctx.read().unwrap_or_else(|e| e.into_inner());
    let mgr = guard
        .get_extension::<ProxyManager>(PROXY_EXTENSION_KEY)
        .expect("ProxyManager not found in extensions map; call ensure_manager_exists first");
    f(mgr)
}

/// Acquires a write lock on the global context and calls `f` with a mutable
/// reference to the [`ProxyManager`].
fn with_manager_mut<F, R>(f: F) -> R
where
    F: FnOnce(&mut ProxyManager) -> R,
{
    let ctx = global_context();
    let mut guard = ctx.write().unwrap_or_else(|e| e.into_inner());
    let mgr = guard
        .get_extension_mut::<ProxyManager>(PROXY_EXTENSION_KEY)
        .expect("ProxyManager not found in extensions map; call ensure_manager_exists first");
    f(mgr)
}

// ---------------------------------------------------------------------------
// Public API (7 functions)
// ---------------------------------------------------------------------------

/// Buffers the intent to enable or disable the proxy.
///
/// If setting to `false` and a proxy is currently registered, the proxy
/// is deregistered and removed immediately. The global context lock is
/// released before calling deregister to avoid re-entrant lock acquisition.
pub fn set_use_proxy(enabled: bool) {
    ensure_manager_exists();
    // Extract proxy outside the lock if we need to deregister
    let mut proxy_to_deregister: Option<NexusProxy> = None;
    with_manager_mut(|mgr| {
        mgr.config.enabled = enabled;
        if !enabled && mgr.registered {
            proxy_to_deregister = mgr.proxy.take();
            mgr.registered = false;
        }
    });
    // Deregister outside the lock (deregister acquires global_context write lock internally)
    if let Some(ref mut proxy) = proxy_to_deregister {
        if let Err(e) = proxy.deregister() {
            eprintln!("nexus-proxy: deregister failed during set_use_proxy(false): {e}");
        }
    }
}

/// Buffers the storage backend choice.
///
/// The backend is consumed when [`ensure_proxy()`] materializes the proxy.
/// Calling this after `ensure_proxy()` has no effect on an already-running proxy.
pub fn set_proxy_backend(backend: AnyBackend) {
    ensure_manager_exists();
    with_manager_mut(|mgr| {
        mgr.config.backend = Some(backend);
    });
}

/// Buffers the sensitivity scoring configuration.
///
/// The config is consumed when [`ensure_proxy()`] materializes the proxy.
pub fn set_proxy_sensitivity(config: SensitivityConfig) {
    ensure_manager_exists();
    with_manager_mut(|mgr| {
        mgr.config.sensitivity_config = config;
    });
}

/// Buffers the DynamoIntercept opt-in flag.
///
/// When enabled, the proxy will register a DynamoIntercept that injects
/// AgentHints into LLM requests at `nvext.agent_hints`.
pub fn set_dynamo_intercept(enabled: bool) {
    ensure_manager_exists();
    with_manager_mut(|mgr| {
        mgr.config.dynamo_intercept = enabled;
    });
}

/// Creates and registers a [`NexusProxy`] from the buffered configuration.
///
/// This is the critical function that materializes the proxy. It:
/// 1. Reads buffered config from the ProxyManager
/// 2. Resolves the agent ID via cascading fallback
/// 3. **Releases the global context lock** before any async operations
/// 4. Builds and registers the NexusProxy
/// 5. Re-acquires the lock to store the proxy
///
/// If the proxy is already registered, this is a no-op (idempotent).
/// If `set_use_proxy(true)` has not been called, it is implicitly enabled.
///
/// # Concurrency
///
/// This function is not safe to call concurrently from multiple threads.
/// In practice it is always called from Python (GIL) or single-threaded init.
///
/// # Errors
///
/// Returns an error if the proxy builder fails or registration fails.
pub async fn ensure_proxy() -> Result<()> {
    ensure_manager_exists();
    // Step 1: Read config under lock, then release.
    // The lock must be released before proxy.register() because registration
    // acquires the global_context write lock internally.
    let (sensitivity_config, backend, dynamo_intercept, agent_id) = {
        let ctx = global_context();
        let mut guard = ctx.write().unwrap_or_else(|e| e.into_inner());
        let mgr = guard
            .get_extension_mut::<ProxyManager>(PROXY_EXTENSION_KEY)
            .expect("ProxyManager must exist after ensure_manager_exists");

        if mgr.registered {
            return Ok(()); // Already registered, no-op
        }

        // Implicitly enable if not already
        mgr.config.enabled = true;

        let sensitivity_config = mgr.config.sensitivity_config.clone();
        let backend = mgr
            .config
            .backend
            .take()
            .unwrap_or_else(|| Box::new(InMemoryBackend::new()) as AnyBackend);
        let dynamo_intercept = mgr.config.dynamo_intercept;
        let agent_id = resolve_agent_id().unwrap_or_else(|| {
            eprintln!("nexus-proxy: no agent_id resolved, falling back to \"default-agent\"");
            "default-agent".to_string()
        });

        (sensitivity_config, backend, dynamo_intercept, agent_id)
    };
    // Lock is released here

    // Step 2: Build and register proxy WITHOUT holding the lock
    let learner = LatencySensitivityLearner::new(&agent_id, sensitivity_config);

    let mut proxy = NexusProxy::builder()
        .agent_id(&agent_id)
        .backend(backend)
        .dynamo_intercept(dynamo_intercept)
        .learner(Box::new(learner))
        .build()
        .map_err(|e| ProxyError::Internal(format!("proxy build failed: {e}")))?;

    proxy.register().await?;

    // Step 3: Re-acquire lock and store the proxy
    {
        let ctx = global_context();
        let mut guard = ctx.write().unwrap_or_else(|e| e.into_inner());
        let mgr = guard
            .get_extension_mut::<ProxyManager>(PROXY_EXTENSION_KEY)
            .expect("ProxyManager must exist after ensure_manager_exists");
        mgr.proxy = Some(proxy);
        mgr.registered = true;
    }

    Ok(())
}

/// Deregisters and removes the proxy.
///
/// Safe to call when no proxy is active (returns `Ok(())`).
/// The global context lock is released before calling deregister to avoid
/// re-entrant lock acquisition.
///
/// # Errors
///
/// Returns an error if deregistration fails.
pub fn teardown_proxy() -> Result<()> {
    ensure_manager_exists();
    // Extract proxy outside the lock
    let mut proxy_to_deregister: Option<NexusProxy> = None;
    with_manager_mut(|mgr| {
        proxy_to_deregister = mgr.proxy.take();
        mgr.registered = false;
    });
    // Deregister outside the lock (deregister acquires global_context write lock internally)
    if let Some(ref mut proxy) = proxy_to_deregister {
        proxy.deregister()?;
    }
    Ok(())
}

/// Returns whether a proxy is currently created and registered.
///
/// Returns `false` if no [`ProxyManager`] exists in the extensions map.
pub fn proxy_active() -> bool {
    let ctx = global_context();
    let guard = match ctx.read() {
        Ok(g) => g,
        Err(_) => return false,
    };
    match guard.get_extension::<ProxyManager>(PROXY_EXTENSION_KEY) {
        Some(mgr) => mgr.registered,
        None => false,
    }
}

#[cfg(test)]
#[allow(clippy::await_holding_lock)]
mod tests {
    use super::*;
    use crate::storage::InMemoryBackend;
    use crate::trie::SensitivityConfig;
    // NOTE: These tests MUST be run with --test-threads=1 since they share the
    // global context singleton. The cleanup() helper uses unwrap_or_else(into_inner)
    // to handle poisoned locks gracefully.

    /// Helper to clean up ProxyManager from extensions after each test.
    /// Handles poisoned RwLock gracefully using `write().unwrap_or_else()`.
    fn cleanup() {
        // Extract proxy outside the lock to avoid re-entrant deadlock.
        // deregister() acquires global_context write lock, so we can't hold it.
        let mut proxy_to_drop: Option<NexusProxy> = None;
        {
            let ctx = global_context();
            let mut guard = ctx.write().unwrap_or_else(|e| e.into_inner());
            if let Some(mgr) = guard.get_extension_mut::<ProxyManager>(PROXY_EXTENSION_KEY) {
                proxy_to_drop = mgr.proxy.take();
                mgr.registered = false;
            }
            guard.remove_extension(PROXY_EXTENSION_KEY);
        }
        // Now deregister (and drop) outside the lock
        if let Some(ref mut proxy) = proxy_to_drop {
            let _ = proxy.deregister();
        }
    }

    // All manager tests share global state — serialize with a mutex to prevent
    // poisoned-lock cascades when tests run in parallel.
    static TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    // ---- Unit tests (no async, no proxy registration) ----

    #[test]
    fn test_proxy_config_default() {
        // Pure unit test, no global state
        let config = ProxyConfig::default();
        assert!(!config.enabled);
        assert!(config.backend.is_none());
        assert!(!config.dynamo_intercept);
        assert!((config.sensitivity_config.w_critical - 0.5).abs() < f64::EPSILON);
        assert!((config.sensitivity_config.w_fanout - 0.3).abs() < f64::EPSILON);
        assert!((config.sensitivity_config.w_position - 0.2).abs() < f64::EPSILON);
        assert!((config.sensitivity_config.w_parallel - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_set_use_proxy_creates_manager() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        cleanup();
        set_use_proxy(true);
        let enabled = with_manager(|mgr| mgr.config.enabled);
        assert!(enabled);
        cleanup();
    }

    #[test]
    fn test_set_proxy_backend_stores_backend() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        cleanup();
        set_proxy_backend(Box::new(InMemoryBackend::new()) as AnyBackend);
        let has_backend = with_manager(|mgr| mgr.config.backend.is_some());
        assert!(has_backend);
        cleanup();
    }

    #[test]
    fn test_set_proxy_sensitivity_stores_config() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        cleanup();
        let custom = SensitivityConfig {
            sensitivity_scale: 10,
            w_critical: 0.5,
            w_fanout: 0.3,
            w_position: 0.2,
            w_parallel: 0.0,
        };
        set_proxy_sensitivity(custom);
        let scale = with_manager(|mgr| mgr.config.sensitivity_config.sensitivity_scale);
        assert_eq!(scale, 10);
        cleanup();
    }

    #[test]
    fn test_set_dynamo_intercept_stores_flag() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        cleanup();
        set_dynamo_intercept(true);
        let enabled = with_manager(|mgr| mgr.config.dynamo_intercept);
        assert!(enabled);
        cleanup();
    }

    #[test]
    fn test_proxy_active_false_before_ensure() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        cleanup();
        set_use_proxy(true);
        assert!(!proxy_active());
        cleanup();
    }

    #[test]
    fn test_proxy_active_false_when_no_manager() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        cleanup();
        assert!(!proxy_active());
    }

    #[test]
    fn test_teardown_when_no_proxy_is_ok() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        cleanup();
        set_use_proxy(true);
        let result = teardown_proxy();
        assert!(result.is_ok());
        cleanup();
    }

    #[test]
    fn test_defaults_inmemory_backend_and_nat_matching_weights() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        cleanup();
        ensure_manager_exists();
        with_manager(|mgr| {
            assert!(mgr.config.backend.is_none());
            assert_eq!(mgr.config.sensitivity_config.sensitivity_scale, 5);
            assert!((mgr.config.sensitivity_config.w_critical - 0.5).abs() < f64::EPSILON);
            assert!((mgr.config.sensitivity_config.w_fanout - 0.3).abs() < f64::EPSILON);
            assert!((mgr.config.sensitivity_config.w_position - 0.2).abs() < f64::EPSILON);
            assert!((mgr.config.sensitivity_config.w_parallel - 0.0).abs() < f64::EPSILON);
        });
        cleanup();
    }

    // ---- Async tests (proxy registration/teardown) ----

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_ensure_proxy_creates_and_registers() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        cleanup();
        set_use_proxy(true);
        set_proxy_backend(Box::new(InMemoryBackend::new()) as AnyBackend);

        let result = ensure_proxy().await;
        assert!(result.is_ok(), "ensure_proxy failed: {:?}", result.err());
        assert!(proxy_active());
        cleanup();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_teardown_proxy_deregisters() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        cleanup();
        set_use_proxy(true);
        set_proxy_backend(Box::new(InMemoryBackend::new()) as AnyBackend);

        ensure_proxy().await.unwrap();
        assert!(proxy_active());

        teardown_proxy().unwrap();
        assert!(!proxy_active());
        cleanup();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_double_ensure_is_idempotent() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        cleanup();
        set_use_proxy(true);
        set_proxy_backend(Box::new(InMemoryBackend::new()) as AnyBackend);

        ensure_proxy().await.unwrap();
        assert!(proxy_active());

        let result = ensure_proxy().await;
        assert!(result.is_ok());
        assert!(proxy_active());
        cleanup();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_ensure_without_set_use_proxy_implicitly_enables() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        cleanup();
        let result = ensure_proxy().await;
        assert!(result.is_ok());
        assert!(proxy_active());
        cleanup();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_set_use_proxy_false_after_ensure_does_teardown() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        cleanup();
        set_use_proxy(true);
        ensure_proxy().await.unwrap();
        assert!(proxy_active());

        set_use_proxy(false);
        assert!(!proxy_active());
        cleanup();
    }
}
