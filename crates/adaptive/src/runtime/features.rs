// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};

use nemo_flow::context::callbacks::EventSubscriberFn;
use nemo_flow::context::callbacks::{LlmRequestInterceptFn, ToolExecutionFn};
use nemo_flow::plugin::{
    ConfigReport, DiagnosticLevel, PluginRegistration as ComponentRegistration,
    PluginRegistrationContext as HostedRegistrationContext, rollback_registrations,
};
use nemo_flow::types::event::Event;
use uuid::Uuid;

use crate::adaptive_hints_intercept::AdaptiveHintsIntercept;
use crate::config::{
    AdaptiveConfig, AdaptiveHintsComponentConfig, TelemetryComponentConfig,
    ToolParallelismComponentConfig,
};
use crate::context_helpers::resolve_agent_id;
use crate::drain::drain_task;
use crate::error::{AdaptiveError, Result};
use crate::intercepts::create_tool_execution_intercept;
use crate::learner::latency::LatencySensitivityLearner;
use crate::learner::traits::Learner;
use crate::runtime::backend::build_backend;
use crate::runtime::validation::validate_config;
use crate::storage::traits::StorageBackendDyn;
use crate::subscriber::create_subscriber;
use crate::types::cache::HotCache;

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
            runtime_id: Uuid::now_v7(),
            registrations: vec![],
        })
    }

    pub fn validate_config(config: &AdaptiveConfig) -> ConfigReport {
        validate_config(config)
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
                Err(error) => eprintln!("nemo-flow-adaptive: hot cache seeding failed: {error}"),
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
                crate::trie::builder::SensitivityConfig::default(),
            )));
        }
    }
    built
}

#[cfg(test)]
#[path = "../../tests/unit/runtime_features_tests.rs"]
mod tests;
