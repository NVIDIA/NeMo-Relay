// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Internal config-driven adaptive runtime.

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};

use nemo_flow::{
    ConfigDiagnostic, ConfigPolicy, ConfigReport, DiagnosticLevel, Event, EventSubscriberFn,
    LlmRequestInterceptFn, PluginRegistration as ComponentRegistration,
    PluginRegistrationContext as HostedRegistrationContext, ToolExecutionFn, UnsupportedBehavior,
    rollback_registrations,
};
use uuid::Uuid;

use crate::adaptive_hints_intercept::AdaptiveHintsIntercept;
use crate::config::{
    AdaptiveConfig, AdaptiveHintsComponentConfig, BackendSpec, TelemetryComponentConfig,
    ToolParallelismComponentConfig,
};
use crate::context_helpers::resolve_agent_id;
use crate::drain::drain_task;
use crate::error::{AdaptiveError, Result};
use crate::intercepts::create_tool_execution_intercept;
use crate::learner::{LatencySensitivityLearner, Learner};
#[cfg(feature = "redis-backend")]
use crate::redis::RedisBackend;
use crate::storage::{InMemoryBackend, StorageBackendDyn};
use crate::subscriber::create_subscriber;
use crate::types::HotCache;

pub struct AdaptiveRuntime {
    config: AdaptiveConfig,
    backend: Option<Arc<dyn StorageBackendDyn + Send + Sync>>,
    hot_cache: Arc<RwLock<HotCache>>,
    event_tx: tokio::sync::mpsc::UnboundedSender<Event>,
    event_rx: Option<tokio::sync::mpsc::UnboundedReceiver<Event>>,
    drain_handle: Option<tokio::task::JoinHandle<()>>,
    registered: bool,
    runtime_id: Uuid,
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
    pub async fn new(config: AdaptiveConfig) -> Result<Self> {
        let report = Self::validate_config(&config);
        if report.has_errors() {
            let joined = report
                .diagnostics
                .iter()
                .filter(|d| d.level == DiagnosticLevel::Error)
                .map(|d| d.message.clone())
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
            backend,
            hot_cache: Arc::new(RwLock::new(HotCache {
                plan: None,
                trie: None,
                agent_hints_default: None,
            })),
            event_tx,
            event_rx: Some(event_rx),
            drain_handle: None,
            registered: false,
            runtime_id: Uuid::new_v4(),
            registrations: vec![],
        })
    }

    pub fn validate_config(config: &AdaptiveConfig) -> ConfigReport {
        let mut report = ConfigReport::default();

        if config.version != 1 {
            push_policy_diag(
                &mut report.diagnostics,
                config.policy.unsupported_value,
                "adaptive.unsupported_config_version",
                None,
                Some("version".to_string()),
                format!("adaptive config version {} is unsupported", config.version),
            );
        }

        if let Some(state) = &config.state {
            validate_backend(&mut report, &config.policy, &state.backend);
        }

        if config.telemetry.is_some() && config.state.is_none() {
            report.diagnostics.push(ConfigDiagnostic {
                level: DiagnosticLevel::Warning,
                code: "adaptive.section_disabled_missing_state".to_string(),
                component: Some("telemetry".to_string()),
                field: None,
                message: "telemetry requires state backend and will be disabled".to_string(),
            });
        }

        if let Some(tool_parallelism) = &config.tool_parallelism
            && tool_parallelism.mode != "observe_only"
            && tool_parallelism.mode != "inject_hints"
            && tool_parallelism.mode != "schedule"
        {
            push_policy_diag(
                &mut report.diagnostics,
                config.policy.unsupported_value,
                "adaptive.unsupported_value",
                Some("tool_parallelism".to_string()),
                Some("mode".to_string()),
                format!(
                    "tool_parallelism mode '{}' is unsupported; expected observe_only, inject_hints, or schedule",
                    tool_parallelism.mode
                ),
            );
        }

        report
    }

    pub async fn register(&mut self) -> Result<()> {
        if self.registered {
            return Ok(());
        }

        let agent_id = self
            .config
            .agent_id
            .clone()
            .or_else(resolve_agent_id)
            .unwrap_or_else(|| "default-agent".to_string());

        if let Some(ref backend) = self.backend {
            match backend.load_plan_dyn(&agent_id).await {
                Ok(plan) => {
                    if let Ok(mut guard) = self.hot_cache.write() {
                        guard.plan = plan;
                    }
                }
                Err(err) => eprintln!("nemo-flow-adaptive: hot cache seeding failed: {err}"),
            }
        }

        let mut pending: Vec<Box<dyn AdaptiveFeature>> = vec![];
        if let Some(config) = self.config.telemetry.clone()
            && self.backend.is_some()
        {
            pending.push(Box::new(TelemetryFeature::new(
                config,
                agent_id.clone(),
                self.runtime_id,
            )));
        }
        if let Some(config) = self.config.adaptive_hints.clone() {
            pending.push(Box::new(AdaptiveHintsFeature::new(
                config,
                self.hot_cache.clone(),
                agent_id.clone(),
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

        for feature in &mut pending {
            let mut ctx = RegistrationContext::new(self);
            if let Err(err) = feature.register(&mut ctx).await {
                let mut just_registered = ctx.finish();
                rollback_registrations(&mut just_registered);
                rollback_registrations(&mut self.registrations);
                if let Some(handle) = self.drain_handle.take() {
                    handle.abort();
                }
                self.registered = false;
                return Err(err);
            }
            let completed = ctx.finish();
            self.registrations.extend(completed);
        }

        self.registered = true;
        Ok(())
    }

    pub fn deregister(&mut self) -> Result<()> {
        rollback_registrations(&mut self.registrations);
        if let Some(handle) = self.drain_handle.take() {
            handle.abort();
        }
        self.registered = false;
        Ok(())
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
    fn new(config: TelemetryComponentConfig, agent_id: String, runtime_id: Uuid) -> Self {
        let subscriber_name = config
            .subscriber_name
            .unwrap_or_else(|| format!("adaptive_{runtime_id}_subscriber"));
        Self {
            learners: build_learners(&agent_id, &config.learners),
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
            ctx.set_drain_task(tokio::spawn(async move {
                drain_task(rx, backend, cache, agent_id, learners).await;
            }));
            ctx.register_subscriber(
                &self.subscriber_name,
                create_subscriber(ctx.runtime.event_tx.clone()),
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
                create_tool_execution_intercept(self.hot_cache.clone()),
            )
        })
    }
}

fn build_learners(agent_id: &str, learners: &[String]) -> Vec<Box<dyn Learner>> {
    let mut built: Vec<Box<dyn Learner>> = vec![];
    for learner in learners {
        if learner == "latency_sensitivity" {
            built.push(Box::new(LatencySensitivityLearner::new(
                agent_id,
                crate::trie::SensitivityConfig::default(),
            )));
        }
    }
    built
}

fn validate_backend(report: &mut ConfigReport, policy: &ConfigPolicy, backend: &BackendSpec) {
    let kind = backend.kind.as_str();
    match kind {
        "in_memory" => {}
        "redis" => {}
        _ => {
            push_policy_diag(
                &mut report.diagnostics,
                policy.unknown_component,
                "adaptive.unknown_backend",
                Some(kind.to_string()),
                None,
                format!("backend kind '{kind}' is unsupported"),
            );
        }
    }
}

fn push_policy_diag(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    behavior: UnsupportedBehavior,
    code: &str,
    component: Option<String>,
    field: Option<String>,
    message: String,
) {
    let level = match behavior {
        UnsupportedBehavior::Ignore => return,
        UnsupportedBehavior::Warn => DiagnosticLevel::Warning,
        UnsupportedBehavior::Error => DiagnosticLevel::Error,
    };

    diagnostics.push(ConfigDiagnostic {
        level,
        code: code.to_string(),
        component,
        field,
        message,
    });
}

async fn build_backend(backend: &BackendSpec) -> Result<Arc<dyn StorageBackendDyn + Send + Sync>> {
    match backend.kind.as_str() {
        "in_memory" => Ok(Arc::new(InMemoryBackend::new())),
        #[cfg(feature = "redis-backend")]
        "redis" => {
            let url = backend
                .config
                .get("url")
                .and_then(|v| v.as_str())
                .ok_or_else(|| AdaptiveError::InvalidConfig("redis backend missing url".into()))?;
            let key_prefix = backend
                .config
                .get("key_prefix")
                .and_then(|v| v.as_str())
                .unwrap_or("nemo_flow:");
            Ok(Arc::new(
                RedisBackend::new(url, key_prefix)
                    .await
                    .map_err(|e| AdaptiveError::Storage(e.to_string()))?,
            ))
        }
        #[cfg(not(feature = "redis-backend"))]
        "redis" => Err(AdaptiveError::InvalidConfig(
            "redis backend is not enabled in this build".into(),
        )),
        other => Err(AdaptiveError::InvalidConfig(format!(
            "unsupported backend '{other}'"
        ))),
    }
}
