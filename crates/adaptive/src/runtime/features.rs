// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{
    Arc, RwLock,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;

use crate::acg::CacheRequestFacts;
use nemo_relay::api::event::Event;
use nemo_relay::api::registry::{
    scope_deregister_llm_request_intercept, scope_register_llm_request_intercept,
};
use nemo_relay::api::runtime::{
    EventSubscriberFn, LlmExecutionFn, LlmRequestInterceptFn, LlmStreamExecutionFn, ToolExecutionFn,
};
use nemo_relay::codec::request::AnnotatedLlmRequest;
use nemo_relay::plugin::{
    ConfigReport, DiagnosticLevel, PluginError, PluginRegistration as ComponentRegistration,
    PluginRegistrationContext as HostedRegistrationContext, rollback_registrations,
};
use uuid::Uuid;

use crate::acg_component::{
    build_provider_plugin, create_acg_llm_execution_intercept, create_acg_llm_request_intercept,
    create_acg_llm_stream_execution_intercept, load_persisted_acg_state,
};
use crate::acg_learner::AcgLearner;
use crate::adaptive_hints_intercept::AdaptiveHintsIntercept;
use crate::cache_diagnostics::{self, CacheDiagnosticsTracker};
use crate::config::{
    AcgComponentConfig, AdaptiveConfig, AdaptiveHintsComponentConfig, TelemetryComponentConfig,
    ToolParallelismComponentConfig,
};
use crate::context_helpers::resolve_agent_id;
use crate::error::{AdaptiveError, Result};
use crate::intercepts::create_tool_execution_intercept_with_mode;
use crate::learner::latency::LatencySensitivityLearner;
use crate::learner::traits::Learner;
use crate::runtime::backend::build_backend;
use crate::runtime::validation::validate_config;
use crate::storage::traits::StorageBackendDyn;
use crate::subscriber::create_subscriber_with_counter;
use crate::tool_parallelism_learner::ToolParallelismLearner;
use crate::types::cache::HotCache;

/// Hosted adaptive runtime that registers NeMo Relay plugin components.
///
/// This type validates configuration, builds the configured storage backend,
/// registers intercepts and subscribers, and maintains the hot cache used by
/// adaptive features on the request path.
pub struct AdaptiveRuntime {
    config: AdaptiveConfig,
    report: ConfigReport,
    registered_agent_id: Option<String>,
    backend: Option<Arc<dyn StorageBackendDyn + Send + Sync>>,
    hot_cache: Arc<RwLock<HotCache>>,
    cache_diagnostics_tracker: Arc<RwLock<CacheDiagnosticsTracker>>,
    pending_events: Arc<AtomicUsize>,
    event_tx: tokio::sync::mpsc::UnboundedSender<Event>,
    event_rx: Option<tokio::sync::mpsc::UnboundedReceiver<Event>>,
    drain_handle: Option<tokio::task::JoinHandle<()>>,
    registered: bool,
    runtime_id: Uuid,
    bound_scopes: Arc<RwLock<HashSet<Uuid>>>,
    registrations: Vec<ComponentRegistration>,
}

impl fmt::Debug for AdaptiveRuntime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AdaptiveRuntime")
            .field("runtime_id", &self.runtime_id)
            .field("registered", &self.registered)
            .finish_non_exhaustive()
    }
}

struct RegistrationContext<'a> {
    runtime: &'a mut AdaptiveRuntime,
    registrations: HostedRegistrationContext,
}

impl<'a> RegistrationContext<'a> {
    fn new(runtime: &'a mut AdaptiveRuntime) -> Self {
        Self {
            runtime,
            registrations: HostedRegistrationContext::new(),
        }
    }

    fn register_subscriber(&mut self, name: &str, callback: EventSubscriberFn) -> Result<()> {
        self.registrations
            .register_subscriber(name, callback)
            .map_err(Into::into)
    }

    fn register_llm_request_intercept(
        &mut self,
        name: &str,
        priority: i32,
        break_chain: bool,
        callback: LlmRequestInterceptFn,
    ) -> Result<()> {
        self.registrations
            .register_llm_request_intercept(name, priority, break_chain, callback)
            .map_err(Into::into)
    }

    fn register_llm_execution_intercept(
        &mut self,
        name: &str,
        priority: i32,
        callback: LlmExecutionFn,
    ) -> Result<()> {
        self.registrations
            .register_llm_execution_intercept(name, priority, callback)
            .map_err(Into::into)
    }

    fn register_llm_stream_execution_intercept(
        &mut self,
        name: &str,
        priority: i32,
        callback: LlmStreamExecutionFn,
    ) -> Result<()> {
        self.registrations
            .register_llm_stream_execution_intercept(name, priority, callback)
            .map_err(Into::into)
    }

    fn register_tool_execution_intercept(
        &mut self,
        name: &str,
        priority: i32,
        callback: ToolExecutionFn,
    ) -> Result<()> {
        self.registrations
            .register_tool_execution_intercept(name, priority, callback)
            .map_err(Into::into)
    }

    fn take_event_receiver(&mut self) -> Result<tokio::sync::mpsc::UnboundedReceiver<Event>> {
        self.runtime
            .event_rx
            .take()
            .ok_or_else(|| AdaptiveError::Internal("telemetry already registered".into()))
    }

    fn set_drain_task(&mut self, handle: tokio::task::JoinHandle<()>) {
        self.runtime.drain_handle = Some(handle);
    }

    fn finish(self) -> Vec<ComponentRegistration> {
        self.registrations.into_registrations()
    }
}

trait AdaptiveFeature: Send + Sync + 'static {
    fn register<'a>(
        &'a mut self,
        ctx: &'a mut RegistrationContext<'_>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;
}

impl AdaptiveRuntime {
    /// Create a new adaptive runtime from configuration.
    ///
    /// # Parameters
    /// - `config`: Adaptive runtime configuration to validate and apply.
    ///
    /// # Returns
    /// A [`Result`] containing a new [`AdaptiveRuntime`].
    ///
    /// # Errors
    /// Returns [`AdaptiveError::InvalidConfig`] when validation reports errors,
    /// or any backend-construction error produced while building the configured
    /// state backend.
    pub async fn new(config: AdaptiveConfig) -> Result<Self> {
        let report = validate_config(&config);
        if report.has_errors() {
            let joined = report
                .diagnostics
                .iter()
                .filter(|diagnostic| diagnostic.level == DiagnosticLevel::Error)
                .map(|diagnostic| diagnostic.message.clone())
                .collect::<Vec<_>>()
                .join("; ");
            return Err(AdaptiveError::InvalidConfig(joined));
        }

        let backend = match config.state.as_ref() {
            Some(state) => Some(build_backend(&state.backend).await?),
            None => None,
        };
        let (event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel();

        Ok(Self {
            config,
            report,
            registered_agent_id: None,
            backend,
            hot_cache: Arc::new(RwLock::new(HotCache {
                plan: None,
                trie: None,
                agent_hints_default: None,
                acg_profiles: std::collections::HashMap::new(),
                acg_profile_observation_counts: std::collections::HashMap::new(),
                acg_stability: None,
                acg_observation_count: 0,
            })),
            cache_diagnostics_tracker: Arc::new(RwLock::new(CacheDiagnosticsTracker::default())),
            pending_events: Arc::new(AtomicUsize::new(0)),
            event_tx,
            event_rx: Some(event_rx),
            drain_handle: None,
            registered: false,
            runtime_id: Uuid::now_v7(),
            bound_scopes: Arc::new(RwLock::new(HashSet::new())),
            registrations: vec![],
        })
    }

    /// Validate an adaptive runtime configuration without constructing a runtime.
    ///
    /// # Parameters
    /// - `config`: Configuration to validate.
    ///
    /// # Returns
    /// A [`ConfigReport`] containing validation diagnostics.
    pub fn validate_config(config: &AdaptiveConfig) -> ConfigReport {
        validate_config(config)
    }

    /// Return the configuration report captured during construction.
    ///
    /// # Returns
    /// The [`ConfigReport`] associated with this runtime.
    pub fn report(&self) -> &ConfigReport {
        &self.report
    }

    /// Block until the telemetry drain has processed all pending events.
    ///
    /// # Notes
    /// This method performs a simple polling wait and is intended for tests,
    /// shutdown paths, or other coordination points.
    pub fn wait_for_idle(&self) {
        loop {
            if self.pending_events.load(Ordering::SeqCst) == 0 {
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[must_use]
    /// Build cache-diagnostics facts for an annotated request.
    ///
    /// # Parameters
    /// - `agent_id`: Agent identifier associated with the request.
    /// - `provider`: Logical provider name associated with the request.
    /// - `annotated_request`: Annotated request to analyze.
    ///
    /// # Returns
    /// `Some(CacheRequestFacts)` when enough hot-cache state is available to
    /// derive them and `None` otherwise.
    pub fn build_cache_request_facts(
        &self,
        agent_id: &str,
        provider: &str,
        annotated_request: &AnnotatedLlmRequest,
    ) -> Option<CacheRequestFacts> {
        cache_diagnostics::build_cache_request_facts(
            agent_id,
            provider,
            annotated_request,
            &self.hot_cache,
            &self.cache_diagnostics_tracker,
        )
    }

    fn acg_scope_registration_name(&self, scope_uuid: Uuid) -> String {
        format!(
            "adaptive_{}_acg_scope_request_{scope_uuid}",
            self.runtime_id
        )
    }

    /// Bind the runtime's ACG request rewrite to an active scope.
    ///
    /// External framework integrations can bind the runtime to a session scope
    /// and then invoke ``nemo_relay.llm.request_intercepts(...)`` explicitly at
    /// the provider boundary. Once any scope is bound, this runtime's hosted
    /// ACG execution intercept becomes pass-through so external frameworks do
    /// not double-translate requests.
    ///
    /// # Errors
    /// Returns an error when the runtime is not yet registered, when ACG is
    /// not configured for this runtime, or when the scope-local request
    /// intercept cannot be constructed or registered.
    pub fn bind_scope(&mut self, scope_uuid: Uuid) -> Result<()> {
        if !self.registered {
            return Err(AdaptiveError::RegistrationFailed(
                "adaptive runtime must be registered before binding ACG request intercepts".into(),
            ));
        }

        let agent_id = self.registered_agent_id.as_deref().ok_or_else(|| {
            AdaptiveError::Internal("adaptive runtime missing registered agent id".into())
        })?;
        let acg_config = self.config.acg.as_ref().ok_or_else(|| {
            AdaptiveError::InvalidConfig(
                "adaptive runtime does not enable scope-bound ACG request intercepts".into(),
            )
        })?;
        if self
            .bound_scopes
            .read()
            .map_err(|error| AdaptiveError::Internal(error.to_string()))?
            .contains(&scope_uuid)
        {
            return Ok(());
        }

        let provider = acg_config.provider.clone();
        let priority = acg_config.priority;
        let plugin = build_provider_plugin(&provider)?;
        let registration_name = self.acg_scope_registration_name(scope_uuid);
        scope_register_llm_request_intercept(
            &scope_uuid,
            &registration_name,
            priority,
            false,
            create_acg_llm_request_intercept(
                self.hot_cache.clone(),
                agent_id.to_string(),
                provider.clone(),
                plugin,
            ),
        )
        .map_err(|error| {
            AdaptiveError::RegistrationFailed(format!(
                "scope-bound ACG llm request intercept: {error}"
            ))
        })?;

        self.bound_scopes
            .write()
            .map_err(|error| AdaptiveError::Internal(error.to_string()))?
            .insert(scope_uuid);

        let bound_scopes = self.bound_scopes.clone();
        self.registrations.push(ComponentRegistration::new(
            "adaptive_scope",
            registration_name.clone(),
            Box::new(move || {
                if let Ok(mut guard) = bound_scopes.write() {
                    guard.remove(&scope_uuid);
                }
                scope_deregister_llm_request_intercept(&scope_uuid, &registration_name)
                    .map(|_| ())
                    .map_err(|error| {
                        PluginError::RegistrationFailed(format!(
                            "scope-bound ACG llm request intercept deregistration failed: {error}"
                        ))
                    })
            }),
        ));

        Ok(())
    }
    /// Register all configured adaptive features with the shared runtime.
    ///
    /// # Returns
    /// A [`Result`] that is `Ok(())` when registration succeeds.
    ///
    /// # Errors
    /// Returns any error raised while seeding state or registering features.
    pub async fn register(&mut self) -> Result<()> {
        if self.registered {
            return Ok(());
        }

        let agent_id = self.agent_id();
        self.registered_agent_id = Some(agent_id.clone());
        Self::seed_hot_cache(self.backend.clone(), self.hot_cache.clone(), &agent_id).await;

        if self.config.acg.is_some()
            && let Some(backend) = self.backend.as_ref()
            && let Err(_) =
                load_persisted_acg_state(&agent_id, backend.as_ref(), &self.hot_cache).await
        {
            log::warn!(
                target: "nemo_relay.runtime",
                event = "adaptive_acg_cache_seed_failed";
                "Adaptive runtime could not seed the ACG hot cache"
            );
        }

        let mut pending = self.pending_features(&agent_id);

        for feature in &mut pending {
            self.register_feature(feature).await?;
        }

        self.registered = true;
        Ok(())
    }

    fn agent_id(&self) -> String {
        self.config
            .agent_id
            .clone()
            .or_else(resolve_agent_id)
            .unwrap_or_else(|| "default-agent".to_string())
    }

    async fn seed_hot_cache(
        backend: Option<Arc<dyn StorageBackendDyn + Send + Sync>>,
        hot_cache: Arc<RwLock<HotCache>>,
        agent_id: &str,
    ) {
        let Some(backend) = backend else {
            return;
        };

        match backend.load_plan_dyn(agent_id).await {
            Ok(plan) => {
                if let Ok(mut guard) = hot_cache.write() {
                    guard.plan = plan;
                }
            }
            Err(_) => log::warn!(
                target: "nemo_relay.runtime",
                event = "adaptive_hot_cache_seed_failed";
                "Adaptive runtime could not seed the hot cache"
            ),
        }
    }

    fn pending_features(&self, agent_id: &str) -> Vec<Box<dyn AdaptiveFeature>> {
        let mut pending: Vec<Box<dyn AdaptiveFeature>> = vec![];
        if let Some(config) = self.config.telemetry.clone()
            && self.backend.is_some()
        {
            pending.push(Box::new(TelemetryFeature::new(
                config,
                agent_id.to_string(),
                self.runtime_id,
                self.config.acg.clone(),
            )));
        }
        if let Some(config) = self.config.adaptive_hints.clone() {
            pending.push(Box::new(AdaptiveHintsFeature::new(
                config,
                self.hot_cache.clone(),
                agent_id.to_string(),
                self.runtime_id,
            )));
        }
        if let Some(config) = self.config.tool_parallelism.clone() {
            pending.push(Box::new(ToolParallelismFeature::new(
                config,
                self.hot_cache.clone(),
                self.runtime_id,
            )));
        }
        if let Some(config) = self.config.acg.clone()
            && self.backend.is_some()
        {
            pending.push(Box::new(AcgFeature::new(
                config,
                self.hot_cache.clone(),
                self.bound_scopes.clone(),
                agent_id.to_string(),
                self.runtime_id,
            )));
        }
        pending
    }

    async fn register_feature(&mut self, feature: &mut Box<dyn AdaptiveFeature>) -> Result<()> {
        let mut ctx = RegistrationContext::new(self);
        if let Err(error) = feature.register(&mut ctx).await {
            let mut just_registered = ctx.finish();
            rollback_registrations(&mut just_registered);
            rollback_registrations(&mut self.registrations);
            if let Some(handle) = self.drain_handle.take() {
                handle.abort();
            }
            self.registered = false;
            return Err(error);
        }

        let completed = ctx.finish();
        self.registrations.extend(completed);
        Ok(())
    }

    /// Deregister all previously registered adaptive features.
    ///
    /// # Returns
    /// A [`Result`] that is `Ok(())` after registrations have been rolled back.
    ///
    /// # Errors
    /// Returns any rollback error surfaced by the hosted plugin system.
    pub fn deregister(&mut self) -> Result<()> {
        rollback_registrations(&mut self.registrations);
        if let Ok(mut guard) = self.bound_scopes.write() {
            guard.clear();
        }
        if let Some(handle) = self.drain_handle.take() {
            handle.abort();
        }
        self.registered = false;
        Ok(())
    }

    /// Deregister the runtime and consume it.
    ///
    /// # Returns
    /// A [`Result`] that is `Ok(())` when shutdown completes.
    ///
    /// # Errors
    /// Propagates any error returned by [`Self::deregister`].
    pub async fn shutdown(mut self) -> Result<()> {
        self.deregister()
    }
}

impl Drop for AdaptiveRuntime {
    fn drop(&mut self) {
        let _ = self.deregister();
    }
}

struct TelemetryFeature {
    agent_id: String,
    subscriber_name: String,
    learners: Vec<Box<dyn Learner>>,
}

impl TelemetryFeature {
    fn new(
        config: TelemetryComponentConfig,
        agent_id: String,
        runtime_id: Uuid,
        acg_config: Option<AcgComponentConfig>,
    ) -> Self {
        let subscriber_name = config
            .subscriber_name
            .unwrap_or_else(|| format!("adaptive_{runtime_id}_subscriber"));
        Self {
            learners: build_learners(&agent_id, &config.learners, acg_config.as_ref()),
            agent_id,
            subscriber_name,
        }
    }
}

impl AdaptiveFeature for TelemetryFeature {
    fn register<'a>(
        &'a mut self,
        ctx: &'a mut RegistrationContext<'_>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let backend = ctx.runtime.backend.as_ref().cloned().ok_or_else(|| {
                AdaptiveError::InvalidConfig("telemetry requires state backend".into())
            })?;
            let rx = ctx.take_event_receiver()?;
            let cache = ctx.runtime.hot_cache.clone();
            let agent_id = self.agent_id.clone();
            let learners = std::mem::take(&mut self.learners);
            let pending_events = ctx.runtime.pending_events.clone();
            ctx.set_drain_task(tokio::spawn(async move {
                crate::drain::drain_task_with_counter(
                    rx,
                    backend,
                    cache,
                    pending_events,
                    agent_id,
                    learners,
                )
                .await;
            }));
            ctx.register_subscriber(
                &self.subscriber_name,
                create_subscriber_with_counter(
                    ctx.runtime.event_tx.clone(),
                    ctx.runtime.pending_events.clone(),
                ),
            )
        })
    }
}

struct AdaptiveHintsFeature {
    name: String,
    priority: i32,
    break_chain: bool,
    hot_cache: Arc<RwLock<HotCache>>,
    agent_id: String,
}

impl AdaptiveHintsFeature {
    fn new(
        config: AdaptiveHintsComponentConfig,
        hot_cache: Arc<RwLock<HotCache>>,
        agent_id: String,
        runtime_id: Uuid,
    ) -> Self {
        Self {
            name: format!("adaptive_{runtime_id}_adaptive_hints_request"),
            priority: config.priority,
            break_chain: config.break_chain,
            hot_cache,
            agent_id,
        }
    }
}

impl AdaptiveFeature for AdaptiveHintsFeature {
    fn register<'a>(
        &'a mut self,
        ctx: &'a mut RegistrationContext<'_>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let adaptive_hints =
                AdaptiveHintsIntercept::new(self.hot_cache.clone(), self.agent_id.clone());
            ctx.register_llm_request_intercept(
                &self.name,
                self.priority,
                self.break_chain,
                adaptive_hints.into_request_fn(),
            )
        })
    }
}

struct ToolParallelismFeature {
    name: String,
    priority: i32,
    hot_cache: Arc<RwLock<HotCache>>,
    mode: String,
}

impl ToolParallelismFeature {
    fn new(
        config: ToolParallelismComponentConfig,
        hot_cache: Arc<RwLock<HotCache>>,
        runtime_id: Uuid,
    ) -> Self {
        Self {
            name: format!("adaptive_{runtime_id}_tool_execution"),
            priority: config.priority,
            hot_cache,
            mode: config.mode,
        }
    }
}

impl AdaptiveFeature for ToolParallelismFeature {
    fn register<'a>(
        &'a mut self,
        ctx: &'a mut RegistrationContext<'_>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            ctx.register_tool_execution_intercept(
                &self.name,
                self.priority,
                create_tool_execution_intercept_with_mode(
                    self.hot_cache.clone(),
                    self.mode.clone(),
                ),
            )
        })
    }
}

struct AcgFeature {
    execution_name: String,
    stream_name: String,
    priority: i32,
    hot_cache: Arc<RwLock<HotCache>>,
    bound_scopes: Arc<RwLock<HashSet<Uuid>>>,
    agent_id: String,
    provider: String,
}

impl AcgFeature {
    fn new(
        config: AcgComponentConfig,
        hot_cache: Arc<RwLock<HotCache>>,
        bound_scopes: Arc<RwLock<HashSet<Uuid>>>,
        agent_id: String,
        runtime_id: Uuid,
    ) -> Self {
        Self {
            execution_name: format!("adaptive_{runtime_id}_acg_llm_execution"),
            stream_name: format!("adaptive_{runtime_id}_acg_llm_stream_execution"),
            priority: config.priority,
            hot_cache,
            bound_scopes,
            agent_id,
            provider: config.provider,
        }
    }
}

impl AdaptiveFeature for AcgFeature {
    fn register<'a>(
        &'a mut self,
        ctx: &'a mut RegistrationContext<'_>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let plugin = build_provider_plugin(&self.provider)?;
            let execution_intercept = create_acg_llm_execution_intercept(
                self.hot_cache.clone(),
                self.agent_id.clone(),
                self.provider.clone(),
                plugin.clone(),
            );
            let bound_scopes = self.bound_scopes.clone();
            ctx.register_llm_execution_intercept(
                &self.execution_name,
                self.priority,
                Arc::new(move |name, request, next| {
                    let execution_intercept = execution_intercept.clone();
                    let bound_scopes = bound_scopes.clone();
                    let name = name.to_string();
                    Box::pin(async move {
                        let has_bound_scopes = bound_scopes
                            .read()
                            .map(|guard| !guard.is_empty())
                            .unwrap_or(false);
                        if has_bound_scopes {
                            return next(request).await;
                        }
                        execution_intercept(&name, request, next).await
                    })
                }),
            )?;
            let stream_intercept = create_acg_llm_stream_execution_intercept(
                self.hot_cache.clone(),
                self.agent_id.clone(),
                self.provider.clone(),
                plugin,
            );
            let bound_scopes = self.bound_scopes.clone();
            ctx.register_llm_stream_execution_intercept(
                &self.stream_name,
                self.priority,
                Arc::new(move |name, request, next| {
                    let stream_intercept = stream_intercept.clone();
                    let bound_scopes = bound_scopes.clone();
                    let name = name.to_string();
                    Box::pin(async move {
                        let has_bound_scopes = bound_scopes
                            .read()
                            .map(|guard| !guard.is_empty())
                            .unwrap_or(false);
                        if has_bound_scopes {
                            return next(request).await;
                        }
                        stream_intercept(&name, request, next).await
                    })
                }),
            )
        })
    }
}

fn build_learners(
    agent_id: &str,
    learners: &[String],
    acg_config: Option<&AcgComponentConfig>,
) -> Vec<Box<dyn Learner>> {
    let mut built: Vec<Box<dyn Learner>> = vec![];
    for learner in learners {
        match learner.as_str() {
            "latency_sensitivity" => built.push(Box::new(LatencySensitivityLearner::new(
                agent_id,
                crate::trie::builder::SensitivityConfig::default(),
            ))),
            "tool_parallelism" => built.push(Box::new(ToolParallelismLearner::new(agent_id))),
            "acg" => {
                if let Some(config) = acg_config {
                    built.push(Box::new(AcgLearner::new(
                        agent_id,
                        config.observation_window,
                        config.stability_thresholds.clone(),
                    )));
                }
            }
            _ => {}
        }
    }
    built
}

#[cfg(test)]
#[path = "../../tests/unit/runtime_features_tests.rs"]
mod tests;
