// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The central orchestrating struct for the nexus-proxy crate.
//!
//! [`NexusProxy`] ties together the agent identity, storage backend, hot cache,
//! and async telemetry channel. It is constructed via [`NexusProxyBuilder`] which
//! enforces required fields (`agent_id`, `backend`) at build time.

use std::fmt;
use std::sync::{Arc, RwLock};

use nvidia_nat_nexus_core::{
    nat_nexus_deregister_llm_request_intercept, nat_nexus_deregister_subscriber,
    nat_nexus_deregister_tool_execution_intercept, nat_nexus_register_llm_request_intercept,
    nat_nexus_register_subscriber, nat_nexus_register_tool_execution_intercept, Event,
};

use crate::drain::drain_task;
use crate::dynamo_intercept::DynamoIntercept;
use crate::error::{ProxyError, Result};
use crate::intercepts::create_tool_execution_intercept;
use crate::learner::Learner;
use crate::storage::{AnyBackend, StorageBackendDyn};
use crate::subscriber::create_subscriber;
use crate::types::HotCache;

/// The central proxy struct that wires Nexus event subscribers and intercepts
/// to a storage backend via an async telemetry channel.
///
/// `NexusProxy` holds the agent identity, storage backend (as
/// `Arc<dyn StorageBackendDyn + Send + Sync>`), hot cache, and channel
/// endpoints. Fields are `pub(crate)` so that sibling modules (subscriber,
/// intercepts, drain) can access them directly.
///
/// Construct via [`NexusProxyBuilder`]:
/// ```rust,no_run
/// # use nvidia_nat_nexus_proxy::proxy::{NexusProxy, NexusProxyBuilder};
/// # use nvidia_nat_nexus_proxy::storage::InMemoryBackend;
/// let proxy = NexusProxy::builder()
///     .agent_id("my-agent")
///     .backend(Box::new(InMemoryBackend::new()))
///     .build()
///     .expect("build should succeed");
/// ```
pub struct NexusProxy {
    /// Identifier of the agent this proxy is associated with.
    pub(crate) agent_id: String,

    /// The storage backend, shared via `Arc` so the drain task and proxy
    /// can both hold a reference.
    pub(crate) backend: Arc<dyn StorageBackendDyn + Send + Sync>,

    /// The hot cache holding the current [`ExecutionPlan`], prediction trie,
    /// and pre-computed hints. Intercepts read from this; the drain task writes to it.
    pub(crate) hot_cache: Arc<RwLock<HotCache>>,

    /// Sender side of the async telemetry channel.
    /// The event subscriber clones events into this channel.
    pub(crate) event_tx: tokio::sync::mpsc::UnboundedSender<Event>,

    /// Receiver side of the async telemetry channel.
    /// Ownership is transferred to the drain task when `register()` is called.
    pub(crate) event_rx: Option<tokio::sync::mpsc::UnboundedReceiver<Event>>,

    /// Handle to the background drain task, if spawned.
    pub(crate) drain_handle: Option<tokio::task::JoinHandle<()>>,

    /// Name used to register the event subscriber with the Nexus runtime.
    pub(crate) subscriber_name: String,

    /// Name used to register the tool execution intercept with the Nexus runtime.
    pub(crate) tool_intercept_name: String,

    /// Name used to register the LLM request intercept (DynamoIntercept), if enabled.
    pub(crate) dynamo_intercept_name: Option<String>,

    /// Priority for the LLM request intercept.
    pub(crate) llm_intercept_priority: i32,

    /// Priority for the tool execution intercept.
    pub(crate) tool_intercept_priority: i32,

    /// Learner pipeline invoked by the drain task after each completed run.
    /// Ownership is transferred to the drain task when `register()` is called.
    pub(crate) learners: Option<Vec<Box<dyn Learner>>>,
}

impl fmt::Debug for NexusProxy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NexusProxy")
            .field("agent_id", &self.agent_id)
            .field("subscriber_name", &self.subscriber_name)
            .field("tool_intercept_name", &self.tool_intercept_name)
            .field("dynamo_intercept_name", &self.dynamo_intercept_name)
            .field("llm_intercept_priority", &self.llm_intercept_priority)
            .field("tool_intercept_priority", &self.tool_intercept_priority)
            .finish_non_exhaustive()
    }
}

impl NexusProxy {
    /// Returns the agent identifier.
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    /// Returns a reference to the hot cache.
    pub fn hot_cache(&self) -> &Arc<RwLock<HotCache>> {
        &self.hot_cache
    }

    /// Returns a reference to the storage backend.
    pub fn backend(&self) -> &Arc<dyn StorageBackendDyn + Send + Sync> {
        &self.backend
    }

    /// Creates a new [`NexusProxyBuilder`].
    pub fn builder() -> NexusProxyBuilder {
        NexusProxyBuilder::new()
    }

    /// Wires the event subscriber and both intercepts with the Nexus global context.
    ///
    /// This method:
    /// 1. Seeds the hot cache from `backend.load_plan_dyn()` (so intercepts have data immediately)
    /// 2. Spawns the background drain task (takes ownership of event_rx)
    /// 3. Registers the event subscriber
    /// 4. Registers the LLM request intercept
    /// 5. Registers the tool execution intercept
    ///
    /// If any registration fails, previously registered components are cleaned up.
    pub async fn register(&mut self) -> Result<()> {
        // 1. Seed hot cache from backend (non-fatal if it fails)
        match self.backend.load_plan_dyn(&self.agent_id).await {
            Ok(plan) => {
                if let Ok(mut guard) = self.hot_cache.write() {
                    guard.plan = plan;
                }
            }
            Err(e) => {
                // Non-fatal: intercepts will work without cached plan
                eprintln!("nexus-proxy: hot cache seeding failed: {e}");
            }
        }

        // 2. Spawn drain task (takes ownership of receiver)
        let rx = self
            .event_rx
            .take()
            .ok_or_else(|| ProxyError::Internal("register() already called".into()))?;
        let learners = self.learners.take().unwrap_or_default();
        let drain_handle = {
            let backend: Arc<dyn StorageBackendDyn + Send + Sync> = Arc::clone(&self.backend);
            let cache = Arc::clone(&self.hot_cache);
            let aid = self.agent_id.clone();
            tokio::spawn(async move {
                drain_task(rx, backend, cache, aid, learners).await;
            })
        };
        self.drain_handle = Some(drain_handle);

        // 3. Register subscriber
        if let Err(e) = nat_nexus_register_subscriber(
            &self.subscriber_name,
            create_subscriber(self.event_tx.clone()),
        ) {
            // Abort the drain task so it doesn't leak
            if let Some(handle) = self.drain_handle.take() {
                handle.abort();
            }
            return Err(ProxyError::RegistrationFailed(format!("subscriber: {e}")));
        }

        // 4. Register tool execution intercept
        if let Err(e) = nat_nexus_register_tool_execution_intercept(
            &self.tool_intercept_name,
            self.tool_intercept_priority,
            create_tool_execution_intercept(Arc::clone(&self.hot_cache)),
        ) {
            // Cleanup: deregister subscriber
            let _ = nat_nexus_deregister_subscriber(&self.subscriber_name);
            return Err(ProxyError::RegistrationFailed(format!(
                "tool intercept: {e}"
            )));
        }

        // 5. Register DynamoIntercept (LLM request intercept) if enabled
        if let Some(ref name) = self.dynamo_intercept_name {
            let dynamo = DynamoIntercept::new(Arc::clone(&self.hot_cache), self.agent_id.clone());
            if let Err(e) = nat_nexus_register_llm_request_intercept(
                name,
                self.llm_intercept_priority,
                false, // break_chain: false -- don't prevent lower-priority intercepts
                dynamo.into_request_fn(),
            ) {
                // Cleanup: deregister subscriber + tool intercept
                let _ = nat_nexus_deregister_subscriber(&self.subscriber_name);
                let _ = nat_nexus_deregister_tool_execution_intercept(&self.tool_intercept_name);
                return Err(ProxyError::RegistrationFailed(format!(
                    "dynamo request intercept: {e}"
                )));
            }
        }

        Ok(())
    }

    /// Removes the subscriber and both intercepts from the Nexus global context.
    ///
    /// Safe to call multiple times -- deregistering a non-existent name returns `Ok(false)`.
    pub fn deregister(&mut self) -> Result<()> {
        let _ = nat_nexus_deregister_subscriber(&self.subscriber_name);
        let _ = nat_nexus_deregister_tool_execution_intercept(&self.tool_intercept_name);
        if let Some(ref name) = self.dynamo_intercept_name {
            let _ = nat_nexus_deregister_llm_request_intercept(name);
        }

        // Abort drain task if still running
        if let Some(handle) = self.drain_handle.take() {
            handle.abort();
        }

        Ok(())
    }

    /// Deregisters from Nexus and waits for the drain task to finish processing
    /// any buffered events.
    pub async fn shutdown(mut self) -> Result<()> {
        // Deregister to stop receiving new events
        let _ = nat_nexus_deregister_subscriber(&self.subscriber_name);
        let _ = nat_nexus_deregister_tool_execution_intercept(&self.tool_intercept_name);
        if let Some(ref name) = self.dynamo_intercept_name {
            let _ = nat_nexus_deregister_llm_request_intercept(name);
        }

        // Drop the sender to signal drain task to finish
        // Replace event_tx with a new sender that has no receiver, effectively
        // dropping the original sender.
        let (dead_tx, _) = tokio::sync::mpsc::unbounded_channel::<Event>();
        let old_tx = std::mem::replace(&mut self.event_tx, dead_tx);
        drop(old_tx);

        // Wait for drain task to complete
        if let Some(handle) = self.drain_handle.take() {
            let _ = handle.await;
        }

        Ok(())
    }
}

impl Drop for NexusProxy {
    fn drop(&mut self) {
        let _ = self.deregister();
    }
}

/// Builder for [`NexusProxy`].
///
/// Enforces that `agent_id` and `backend` are provided before construction.
/// Intercept priorities default to 100.
pub struct NexusProxyBuilder {
    agent_id: Option<String>,
    backend: Option<AnyBackend>,
    llm_intercept_priority: i32,
    tool_intercept_priority: i32,
    enable_dynamo_intercept: bool,
    learners: Vec<Box<dyn Learner>>,
}

impl NexusProxyBuilder {
    /// Creates a new builder with default intercept priorities of 100.
    pub fn new() -> Self {
        Self {
            agent_id: None,
            backend: None,
            llm_intercept_priority: 100,
            tool_intercept_priority: 100,
            enable_dynamo_intercept: false,
            learners: vec![],
        }
    }

    /// Sets the agent identifier (required).
    pub fn agent_id(mut self, id: impl Into<String>) -> Self {
        self.agent_id = Some(id.into());
        self
    }

    /// Sets the storage backend (required).
    ///
    /// Accepts an [`AnyBackend`] (`Box<dyn StorageBackendDyn + Send + Sync>`).
    pub fn backend(mut self, backend: AnyBackend) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Sets the priority for the LLM request intercept (default: 100).
    pub fn llm_intercept_priority(mut self, priority: i32) -> Self {
        self.llm_intercept_priority = priority;
        self
    }

    /// Sets the priority for the tool execution intercept (default: 100).
    pub fn tool_intercept_priority(mut self, priority: i32) -> Self {
        self.tool_intercept_priority = priority;
        self
    }

    /// Enables the DynamoIntercept, which injects AgentHints into LLM requests
    /// as a request intercept. Disabled by default.
    pub fn dynamo_intercept(mut self, enable: bool) -> Self {
        self.enable_dynamo_intercept = enable;
        self
    }

    /// Adds a learner to the pipeline (optional, can be called multiple times).
    pub fn learner(mut self, learner: Box<dyn Learner>) -> Self {
        self.learners.push(learner);
        self
    }

    /// Builds the [`NexusProxy`].
    ///
    /// Returns an error if `agent_id` or `backend` has not been set.
    /// Creates the async telemetry channel and initializes the hot cache
    /// to `None`. Does NOT spawn the drain task or seed the hot cache --
    /// those happen during `register()`.
    pub fn build(self) -> Result<NexusProxy> {
        let agent_id = self.agent_id.ok_or_else(|| {
            ProxyError::Internal("agent_id is required but was not set".to_string())
        })?;

        let backend = self.backend.ok_or_else(|| {
            ProxyError::Internal("backend is required but was not set".to_string())
        })?;

        // Convert Box<dyn StorageBackendDyn + Send + Sync> to Arc<dyn StorageBackendDyn + Send + Sync>
        let backend: Arc<dyn StorageBackendDyn + Send + Sync> = Arc::from(backend);

        let (event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel();

        let subscriber_name = format!("nexus_proxy_{agent_id}_subscriber");
        let tool_intercept_name = format!("nexus_proxy_{agent_id}_tool_execution");
        let dynamo_intercept_name = if self.enable_dynamo_intercept {
            Some(format!("nexus_proxy_{agent_id}_dynamo_request"))
        } else {
            None
        };

        Ok(NexusProxy {
            agent_id,
            backend,
            hot_cache: Arc::new(RwLock::new(HotCache {
                plan: None,
                trie: None,
                agent_hints_default: None,
            })),
            event_tx,
            event_rx: Some(event_rx),
            drain_handle: None,
            subscriber_name,
            tool_intercept_name,
            dynamo_intercept_name,
            llm_intercept_priority: self.llm_intercept_priority,
            tool_intercept_priority: self.tool_intercept_priority,
            learners: if self.learners.is_empty() {
                None
            } else {
                Some(self.learners)
            },
        })
    }
}

impl Default for NexusProxyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::InMemoryBackend;

    #[test]
    fn test_builder_missing_agent_id() {
        let result = NexusProxy::builder()
            .backend(Box::new(InMemoryBackend::new()))
            .build();

        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("agent_id"), "expected 'agent_id' in: {msg}");
    }

    #[test]
    fn test_builder_missing_backend() {
        let result = NexusProxy::builder().agent_id("x").build();

        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("backend"), "expected 'backend' in: {msg}");
    }

    #[test]
    fn test_builder_success() {
        let result = NexusProxy::builder()
            .agent_id("test-agent")
            .backend(Box::new(InMemoryBackend::new()))
            .build();

        assert!(result.is_ok());
        let proxy = result.unwrap();
        assert_eq!(proxy.agent_id(), "test-agent");
    }

    #[test]
    fn test_builder_custom_priorities() {
        let proxy = NexusProxy::builder()
            .agent_id("test-agent")
            .backend(Box::new(InMemoryBackend::new()))
            .llm_intercept_priority(50)
            .tool_intercept_priority(75)
            .build()
            .unwrap();

        assert_eq!(proxy.llm_intercept_priority, 50);
        assert_eq!(proxy.tool_intercept_priority, 75);
    }

    #[test]
    fn test_naming_convention() {
        let proxy = NexusProxy::builder()
            .agent_id("myagent")
            .backend(Box::new(InMemoryBackend::new()))
            .build()
            .unwrap();

        assert_eq!(proxy.subscriber_name, "nexus_proxy_myagent_subscriber");
        assert_eq!(
            proxy.tool_intercept_name,
            "nexus_proxy_myagent_tool_execution"
        );
    }

    #[test]
    fn test_hot_cache_initially_none() {
        let proxy = NexusProxy::builder()
            .agent_id("test-agent")
            .backend(Box::new(InMemoryBackend::new()))
            .build()
            .unwrap();

        let guard = proxy.hot_cache().read().unwrap();
        assert!(guard.plan.is_none());
    }

    #[test]
    fn test_proxy_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<NexusProxy>();
    }

    #[test]
    fn test_builder_dynamo_intercept_default_disabled() {
        let proxy = NexusProxy::builder()
            .agent_id("dynamo-test")
            .backend(Box::new(InMemoryBackend::new()))
            .build()
            .unwrap();
        assert!(proxy.dynamo_intercept_name.is_none());
    }

    #[test]
    fn test_builder_dynamo_intercept_enabled() {
        let proxy = NexusProxy::builder()
            .agent_id("dynamo-test")
            .backend(Box::new(InMemoryBackend::new()))
            .dynamo_intercept(true)
            .build()
            .unwrap();
        assert_eq!(
            proxy.dynamo_intercept_name,
            Some("nexus_proxy_dynamo-test_dynamo_request".to_string())
        );
    }
}
