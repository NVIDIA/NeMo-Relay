// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamic config-driven optimizer runtime and component registries.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, LazyLock, RwLock};

use nvidia_nat_nexus_core::{
    nat_nexus_deregister_llm_execution_intercept, nat_nexus_deregister_llm_request_intercept,
    nat_nexus_deregister_llm_stream_execution_intercept, nat_nexus_deregister_subscriber,
    nat_nexus_deregister_tool_execution_intercept, nat_nexus_deregister_tool_request_intercept,
    nat_nexus_register_llm_execution_intercept, nat_nexus_register_llm_request_intercept,
    nat_nexus_register_llm_stream_execution_intercept, nat_nexus_register_subscriber,
    nat_nexus_register_tool_execution_intercept, nat_nexus_register_tool_request_intercept, Event,
    EventSubscriberFn, LlmExecutionFn, LlmRequestInterceptFn, LlmStreamExecutionFn,
    ToolExecutionFn, ToolInterceptFn,
};
use serde::Deserialize;
use serde_json::{Map, Value as Json};
use uuid::Uuid;

use crate::config::{
    BackendSpec, ComponentSpec, ConfigDiagnostic, ConfigPolicy, ConfigReport, DiagnosticLevel,
    DynamoHintsComponentConfig, OptimizerConfig, TelemetryComponentConfig,
    ToolParallelismComponentConfig, UnsupportedBehavior,
};
use crate::context_helpers::resolve_agent_id;
use crate::drain::drain_task;
use crate::dynamo_intercept::DynamoIntercept;
use crate::error::{OptimizerError, Result};
use crate::intercepts::create_tool_execution_intercept;
use crate::learner::{LatencySensitivityLearner, Learner};
#[cfg(feature = "redis-backend")]
use crate::redis::RedisBackend;
use crate::storage::{InMemoryBackend, StorageBackendDyn};
use crate::subscriber::create_subscriber;
use crate::types::HotCache;

type FactoryMap = HashMap<String, Arc<dyn OptimizerComponentFactory>>;
type HostedPluginMap = HashMap<String, Arc<dyn HostedPluginHandler>>;

static COMPONENT_FACTORIES: LazyLock<RwLock<FactoryMap>> = LazyLock::new(|| {
    let mut factories: FactoryMap = HashMap::new();
    let builtins: [Arc<dyn OptimizerComponentFactory>; 4] = [
        Arc::new(TelemetryFactory),
        Arc::new(DynamoHintsFactory),
        Arc::new(ToolParallelismFactory),
        Arc::new(ExternalComponentFactory),
    ];
    for factory in builtins {
        factories.insert(factory.kind().to_string(), factory);
    }
    RwLock::new(factories)
});

static HOSTED_PLUGIN_HANDLERS: LazyLock<RwLock<HostedPluginMap>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

pub struct OptimizerRuntime {
    config: OptimizerConfig,
    report: ConfigReport,
    backend: Option<Arc<dyn StorageBackendDyn + Send + Sync>>,
    hot_cache: Arc<RwLock<HotCache>>,
    event_tx: tokio::sync::mpsc::UnboundedSender<Event>,
    event_rx: Option<tokio::sync::mpsc::UnboundedReceiver<Event>>,
    drain_handle: Option<tokio::task::JoinHandle<()>>,
    registered: bool,
    runtime_id: Uuid,
    registrations: Vec<ComponentRegistration>,
}

impl fmt::Debug for OptimizerRuntime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OptimizerRuntime")
            .field("runtime_id", &self.runtime_id)
            .field("registered", &self.registered)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ValidationContext {
    pub has_state: bool,
}

#[derive(Clone)]
pub struct BuildContext {
    pub agent_id: String,
    pub backend: Option<Arc<dyn StorageBackendDyn + Send + Sync>>,
    pub hot_cache: Arc<RwLock<HotCache>>,
    pub event_tx: tokio::sync::mpsc::UnboundedSender<Event>,
    runtime_id: Uuid,
}

pub struct RegistrationContext<'a> {
    runtime: &'a mut OptimizerRuntime,
    registrations: Vec<ComponentRegistration>,
}

impl<'a> RegistrationContext<'a> {
    fn new(runtime: &'a mut OptimizerRuntime) -> Self {
        Self {
            runtime,
            registrations: vec![],
        }
    }

    pub fn register_subscriber(&mut self, name: &str, callback: EventSubscriberFn) -> Result<()> {
        register_subscriber_impl(&mut self.registrations, "component", name, callback)
    }

    pub fn register_llm_request_intercept(
        &mut self,
        name: &str,
        priority: i32,
        break_chain: bool,
        callback: LlmRequestInterceptFn,
    ) -> Result<()> {
        register_llm_request_intercept_impl(
            &mut self.registrations,
            "component",
            name,
            priority,
            break_chain,
            callback,
        )
    }

    pub fn register_llm_execution_intercept(
        &mut self,
        name: &str,
        priority: i32,
        callback: LlmExecutionFn,
    ) -> Result<()> {
        register_llm_execution_intercept_impl(
            &mut self.registrations,
            "component",
            name,
            priority,
            callback,
        )
    }

    pub fn register_llm_stream_execution_intercept(
        &mut self,
        name: &str,
        priority: i32,
        callback: LlmStreamExecutionFn,
    ) -> Result<()> {
        register_llm_stream_execution_intercept_impl(
            &mut self.registrations,
            "component",
            name,
            priority,
            callback,
        )
    }

    pub fn register_tool_request_intercept(
        &mut self,
        name: &str,
        priority: i32,
        break_chain: bool,
        callback: ToolInterceptFn,
    ) -> Result<()> {
        register_tool_request_intercept_impl(
            &mut self.registrations,
            "component",
            name,
            priority,
            break_chain,
            callback,
        )
    }

    pub fn register_tool_execution_intercept(
        &mut self,
        name: &str,
        priority: i32,
        callback: ToolExecutionFn,
    ) -> Result<()> {
        register_tool_execution_intercept_impl(
            &mut self.registrations,
            "component",
            name,
            priority,
            callback,
        )
    }

    pub fn take_event_receiver(&mut self) -> Result<tokio::sync::mpsc::UnboundedReceiver<Event>> {
        self.runtime
            .event_rx
            .take()
            .ok_or_else(|| OptimizerError::Internal("telemetry already registered".into()))
    }

    pub fn set_drain_task(&mut self, handle: tokio::task::JoinHandle<()>) {
        self.runtime.drain_handle = Some(handle);
    }

    fn finish(self) -> Vec<ComponentRegistration> {
        self.registrations
    }
}

pub struct HostedRegistrationContext {
    registrations: Vec<ComponentRegistration>,
}

impl HostedRegistrationContext {
    fn new() -> Self {
        Self {
            registrations: vec![],
        }
    }

    pub fn register_subscriber(&mut self, name: &str, callback: EventSubscriberFn) -> Result<()> {
        register_subscriber_impl(
            &mut self.registrations,
            "external_component",
            name,
            callback,
        )
    }

    pub fn register_llm_request_intercept(
        &mut self,
        name: &str,
        priority: i32,
        break_chain: bool,
        callback: LlmRequestInterceptFn,
    ) -> Result<()> {
        register_llm_request_intercept_impl(
            &mut self.registrations,
            "external_component",
            name,
            priority,
            break_chain,
            callback,
        )
    }

    pub fn register_llm_execution_intercept(
        &mut self,
        name: &str,
        priority: i32,
        callback: LlmExecutionFn,
    ) -> Result<()> {
        register_llm_execution_intercept_impl(
            &mut self.registrations,
            "external_component",
            name,
            priority,
            callback,
        )
    }

    pub fn register_llm_stream_execution_intercept(
        &mut self,
        name: &str,
        priority: i32,
        callback: LlmStreamExecutionFn,
    ) -> Result<()> {
        register_llm_stream_execution_intercept_impl(
            &mut self.registrations,
            "external_component",
            name,
            priority,
            callback,
        )
    }

    pub fn register_tool_request_intercept(
        &mut self,
        name: &str,
        priority: i32,
        break_chain: bool,
        callback: ToolInterceptFn,
    ) -> Result<()> {
        register_tool_request_intercept_impl(
            &mut self.registrations,
            "external_component",
            name,
            priority,
            break_chain,
            callback,
        )
    }

    pub fn register_tool_execution_intercept(
        &mut self,
        name: &str,
        priority: i32,
        callback: ToolExecutionFn,
    ) -> Result<()> {
        register_tool_execution_intercept_impl(
            &mut self.registrations,
            "external_component",
            name,
            priority,
            callback,
        )
    }

    pub fn add_registration(&mut self, registration: ComponentRegistration) {
        self.registrations.push(registration);
    }

    pub fn extend_registrations(&mut self, registrations: Vec<ComponentRegistration>) {
        self.registrations.extend(registrations);
    }

    fn finish(self) -> Vec<ComponentRegistration> {
        self.registrations
    }
}

pub struct ComponentRegistration {
    pub kind: String,
    pub name: String,
    deregister: Box<dyn FnMut() -> Result<()> + Send>,
}

impl fmt::Debug for ComponentRegistration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComponentRegistration")
            .field("kind", &self.kind)
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl ComponentRegistration {
    pub fn new(
        kind: impl Into<String>,
        name: impl Into<String>,
        deregister: Box<dyn FnMut() -> Result<()> + Send>,
    ) -> Self {
        Self {
            kind: kind.into(),
            name: name.into(),
            deregister,
        }
    }
}

pub trait OptimizerComponentFactory: Send + Sync + 'static {
    fn kind(&self) -> &'static str;

    fn allows_multiple_instances(&self) -> bool {
        false
    }

    fn requires_state(&self, _spec: &ComponentSpec) -> bool {
        false
    }

    fn validate(
        &self,
        spec: &ComponentSpec,
        policy: &ConfigPolicy,
        ctx: &ValidationContext,
    ) -> Vec<ConfigDiagnostic>;

    fn build(
        &self,
        spec: &ComponentSpec,
        ctx: &BuildContext,
    ) -> Result<Box<dyn OptimizerComponent>>;
}

pub trait OptimizerComponent: Send + Sync + 'static {
    fn kind(&self) -> &'static str;

    fn register<'a>(
        &'a mut self,
        ctx: &'a mut RegistrationContext<'_>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;
}

pub trait HostedPluginHandler: Send + Sync + 'static {
    fn plugin_kind(&self) -> &str;

    fn validate(
        &self,
        instance_id: &str,
        plugin_config: &Map<String, Json>,
    ) -> Vec<ConfigDiagnostic>;

    fn register(
        &self,
        instance_id: &str,
        plugin_config: &Map<String, Json>,
        ctx: &mut HostedRegistrationContext,
    ) -> Result<()>;
}

pub fn register_component_factory(factory: Arc<dyn OptimizerComponentFactory>) -> Result<()> {
    let mut guard = COMPONENT_FACTORIES
        .write()
        .map_err(|e| OptimizerError::Internal(format!("component registry lock poisoned: {e}")))?;
    let kind = factory.kind().to_string();
    if guard.contains_key(&kind) {
        return Err(OptimizerError::RegistrationFailed(format!(
            "component factory '{kind}' is already registered"
        )));
    }
    guard.insert(kind, factory);
    Ok(())
}

pub fn deregister_component_factory(kind: &str) -> bool {
    if is_builtin_component_kind(kind) {
        return false;
    }
    COMPONENT_FACTORIES
        .write()
        .ok()
        .and_then(|mut guard| guard.remove(kind))
        .is_some()
}

pub fn list_component_kinds() -> Vec<String> {
    let mut kinds = COMPONENT_FACTORIES
        .read()
        .map(|guard| guard.keys().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    kinds.sort();
    kinds
}

pub fn register_hosted_plugin_handler(handler: Arc<dyn HostedPluginHandler>) -> Result<()> {
    let mut guard = HOSTED_PLUGIN_HANDLERS.write().map_err(|e| {
        OptimizerError::Internal(format!("hosted plugin registry lock poisoned: {e}"))
    })?;
    let plugin_kind = handler.plugin_kind().to_string();
    if guard.contains_key(&plugin_kind) {
        return Err(OptimizerError::RegistrationFailed(format!(
            "hosted plugin handler '{plugin_kind}' is already registered"
        )));
    }
    guard.insert(plugin_kind, handler);
    Ok(())
}

pub fn deregister_hosted_plugin_handler(plugin_kind: &str) -> bool {
    HOSTED_PLUGIN_HANDLERS
        .write()
        .ok()
        .and_then(|mut guard| guard.remove(plugin_kind))
        .is_some()
}

pub fn list_hosted_plugin_kinds() -> Vec<String> {
    let mut kinds = HOSTED_PLUGIN_HANDLERS
        .read()
        .map(|guard| guard.keys().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    kinds.sort();
    kinds
}

impl OptimizerRuntime {
    pub async fn new(config: OptimizerConfig) -> Result<Self> {
        let report = Self::validate_config(&config);
        if report.has_errors() {
            let joined = report
                .diagnostics
                .iter()
                .filter(|d| d.level == DiagnosticLevel::Error)
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
                .join("; ");
            return Err(OptimizerError::InvalidConfig(joined));
        }

        let backend = match config.state.as_ref() {
            Some(state) => Some(build_backend(&state.backend).await?),
            None => None,
        };
        let (event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel();

        Ok(Self {
            config,
            report,
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

    pub fn validate_config(config: &OptimizerConfig) -> ConfigReport {
        let mut report = ConfigReport::default();

        if config.version != 1 {
            push_policy_diag(
                &mut report.diagnostics,
                config.policy.unsupported_value,
                "optimizer.unsupported_config_version",
                None,
                Some("version".to_string()),
                format!("optimizer config version {} is unsupported", config.version),
            );
        }

        if let Some(state) = &config.state {
            validate_backend(&mut report, &config.policy, &state.backend);
        }

        validate_component_multiplicity(&mut report, config);

        let ctx = ValidationContext {
            has_state: config.state.is_some(),
        };
        for spec in &config.components {
            let Some(factory) = component_factory(&spec.kind) else {
                push_policy_diag(
                    &mut report.diagnostics,
                    config.policy.unknown_component,
                    "optimizer.unknown_component",
                    Some(spec.kind.clone()),
                    None,
                    format!("component kind '{}' is unsupported", spec.kind),
                );
                continue;
            };

            report
                .diagnostics
                .extend(factory.validate(spec, &config.policy, &ctx));
        }

        report
    }

    pub fn report(&self) -> &ConfigReport {
        &self.report
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
                Err(e) => eprintln!("nexus-optimizer: hot cache seeding failed: {e}"),
            }
        }

        let build_ctx = BuildContext {
            agent_id: agent_id.clone(),
            backend: self.backend.clone(),
            hot_cache: self.hot_cache.clone(),
            event_tx: self.event_tx.clone(),
            runtime_id: self.runtime_id,
        };

        let mut pending = vec![];
        for spec in self.config.components.iter().filter(|spec| spec.enabled) {
            let Some(factory) = component_factory(&spec.kind) else {
                continue;
            };
            if factory.requires_state(spec) && self.backend.is_none() {
                continue;
            }
            match factory.build(spec, &build_ctx) {
                Ok(component) => pending.push(component),
                Err(OptimizerError::NotFound(_)) => continue,
                Err(err) => return Err(err),
            }
        }

        for component in &mut pending {
            let mut ctx = RegistrationContext::new(self);
            if let Err(err) = component.register(&mut ctx).await {
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

    pub async fn shutdown(mut self) -> Result<()> {
        self.deregister()?;
        let (dead_tx, _) = tokio::sync::mpsc::unbounded_channel::<Event>();
        let old_tx = std::mem::replace(&mut self.event_tx, dead_tx);
        drop(old_tx);
        if let Some(handle) = self.drain_handle.take() {
            let _ = handle.await;
        }
        Ok(())
    }
}

impl Drop for OptimizerRuntime {
    fn drop(&mut self) {
        let _ = self.deregister();
    }
}

struct TelemetryFactory;

impl OptimizerComponentFactory for TelemetryFactory {
    fn kind(&self) -> &'static str {
        "telemetry"
    }

    fn requires_state(&self, _spec: &ComponentSpec) -> bool {
        true
    }

    fn validate(
        &self,
        spec: &ComponentSpec,
        policy: &ConfigPolicy,
        ctx: &ValidationContext,
    ) -> Vec<ConfigDiagnostic> {
        let mut diagnostics = vec![];
        validate_unknown_fields(
            &mut diagnostics,
            policy,
            Some(spec.kind.clone()),
            &spec.config,
            &["subscriber_name", "learners"],
        );
        if !ctx.has_state {
            diagnostics.push(ConfigDiagnostic {
                level: DiagnosticLevel::Warning,
                code: "optimizer.component_disabled_missing_state".to_string(),
                component: Some(spec.kind.clone()),
                field: None,
                message: "telemetry component requires state backend and will be disabled"
                    .to_string(),
            });
        }
        diagnostics
    }

    fn build(
        &self,
        spec: &ComponentSpec,
        ctx: &BuildContext,
    ) -> Result<Box<dyn OptimizerComponent>> {
        let config = parse_component_config::<TelemetryComponentConfig>(&spec.config, self.kind())?;
        let subscriber_name = config
            .subscriber_name
            .unwrap_or_else(|| format!("nexus_optimizer_{}_subscriber", ctx.runtime_id));
        Ok(Box::new(TelemetryComponent {
            agent_id: ctx.agent_id.clone(),
            subscriber_name,
            learners: build_learners(&ctx.agent_id, &config.learners),
        }))
    }
}

struct TelemetryComponent {
    agent_id: String,
    subscriber_name: String,
    learners: Vec<Box<dyn Learner>>,
}

impl OptimizerComponent for TelemetryComponent {
    fn kind(&self) -> &'static str {
        "telemetry"
    }

    fn register<'a>(
        &'a mut self,
        ctx: &'a mut RegistrationContext<'_>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let backend = ctx.runtime.backend.as_ref().cloned().ok_or_else(|| {
                OptimizerError::InvalidConfig("telemetry requires state backend".into())
            })?;
            let rx = ctx.take_event_receiver()?;
            let cache = ctx.runtime.hot_cache.clone();
            let aid = self.agent_id.clone();
            let learners = std::mem::take(&mut self.learners);
            ctx.set_drain_task(tokio::spawn(async move {
                drain_task(rx, backend, cache, aid, learners).await;
            }));
            ctx.register_subscriber(
                &self.subscriber_name,
                create_subscriber(ctx.runtime.event_tx.clone()),
            )
        })
    }
}

struct DynamoHintsFactory;

impl OptimizerComponentFactory for DynamoHintsFactory {
    fn kind(&self) -> &'static str {
        "dynamo_hints"
    }

    fn validate(
        &self,
        spec: &ComponentSpec,
        policy: &ConfigPolicy,
        _ctx: &ValidationContext,
    ) -> Vec<ConfigDiagnostic> {
        let mut diagnostics = vec![];
        validate_unknown_fields(
            &mut diagnostics,
            policy,
            Some(spec.kind.clone()),
            &spec.config,
            &[
                "priority",
                "break_chain",
                "inject_header",
                "inject_body_path",
            ],
        );
        diagnostics
    }

    fn build(
        &self,
        spec: &ComponentSpec,
        ctx: &BuildContext,
    ) -> Result<Box<dyn OptimizerComponent>> {
        let config =
            parse_component_config::<DynamoHintsComponentConfig>(&spec.config, self.kind())?;
        Ok(Box::new(DynamoHintsComponent {
            name: format!("nexus_optimizer_{}_dynamo_request", ctx.runtime_id),
            priority: config.priority,
            break_chain: config.break_chain,
            hot_cache: ctx.hot_cache.clone(),
            agent_id: ctx.agent_id.clone(),
        }))
    }
}

struct DynamoHintsComponent {
    name: String,
    priority: i32,
    break_chain: bool,
    hot_cache: Arc<RwLock<HotCache>>,
    agent_id: String,
}

impl OptimizerComponent for DynamoHintsComponent {
    fn kind(&self) -> &'static str {
        "dynamo_hints"
    }

    fn register<'a>(
        &'a mut self,
        ctx: &'a mut RegistrationContext<'_>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let dynamo = DynamoIntercept::new(self.hot_cache.clone(), self.agent_id.clone());
            ctx.register_llm_request_intercept(
                &self.name,
                self.priority,
                self.break_chain,
                dynamo.into_request_fn(),
            )
        })
    }
}

struct ToolParallelismFactory;

impl OptimizerComponentFactory for ToolParallelismFactory {
    fn kind(&self) -> &'static str {
        "tool_parallelism"
    }

    fn validate(
        &self,
        spec: &ComponentSpec,
        policy: &ConfigPolicy,
        _ctx: &ValidationContext,
    ) -> Vec<ConfigDiagnostic> {
        let mut diagnostics = vec![];
        validate_unknown_fields(
            &mut diagnostics,
            policy,
            Some(spec.kind.clone()),
            &spec.config,
            &["priority", "mode"],
        );
        if let Some(mode) = spec.config.get("mode").and_then(|v| v.as_str()) {
            if mode != "observe_only" && mode != "inject_hints" && mode != "schedule" {
                push_policy_diag(
                    &mut diagnostics,
                    policy.unsupported_value,
                    "optimizer.unsupported_value",
                    Some(spec.kind.clone()),
                    Some("mode".to_string()),
                    format!(
                        "tool_parallelism mode '{mode}' is unsupported; expected observe_only, inject_hints, or schedule"
                    ),
                );
            }
        }
        diagnostics
    }

    fn build(
        &self,
        spec: &ComponentSpec,
        ctx: &BuildContext,
    ) -> Result<Box<dyn OptimizerComponent>> {
        let config =
            parse_component_config::<ToolParallelismComponentConfig>(&spec.config, self.kind())?;
        Ok(Box::new(ToolParallelismComponent {
            name: format!("nexus_optimizer_{}_tool_execution", ctx.runtime_id),
            priority: config.priority,
            hot_cache: ctx.hot_cache.clone(),
        }))
    }
}

struct ToolParallelismComponent {
    name: String,
    priority: i32,
    hot_cache: Arc<RwLock<HotCache>>,
}

impl OptimizerComponent for ToolParallelismComponent {
    fn kind(&self) -> &'static str {
        "tool_parallelism"
    }

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

#[derive(Debug, Deserialize)]
struct ExternalComponentConfig {
    plugin_kind: String,
    instance_id: String,
    #[serde(default)]
    plugin_config: Map<String, Json>,
}

struct ExternalComponentFactory;

impl OptimizerComponentFactory for ExternalComponentFactory {
    fn kind(&self) -> &'static str {
        "external_component"
    }

    fn allows_multiple_instances(&self) -> bool {
        true
    }

    fn validate(
        &self,
        spec: &ComponentSpec,
        policy: &ConfigPolicy,
        _ctx: &ValidationContext,
    ) -> Vec<ConfigDiagnostic> {
        let mut diagnostics = vec![];
        validate_unknown_fields(
            &mut diagnostics,
            policy,
            Some(spec.kind.clone()),
            &spec.config,
            &["plugin_kind", "instance_id", "plugin_config"],
        );

        let config =
            match parse_component_config::<ExternalComponentConfig>(&spec.config, self.kind()) {
                Ok(config) => config,
                Err(err) => {
                    diagnostics.push(ConfigDiagnostic {
                        level: DiagnosticLevel::Error,
                        code: "optimizer.invalid_external_component_config".to_string(),
                        component: Some(spec.kind.clone()),
                        field: None,
                        message: err.to_string(),
                    });
                    return diagnostics;
                }
            };

        let Some(handler) = hosted_plugin_handler(&config.plugin_kind) else {
            push_policy_diag(
                &mut diagnostics,
                policy.unknown_component,
                "optimizer.unknown_plugin_kind",
                Some(spec.kind.clone()),
                Some("plugin_kind".to_string()),
                format!(
                    "external component references unknown plugin kind '{}'",
                    config.plugin_kind
                ),
            );
            return diagnostics;
        };

        diagnostics.extend(handler.validate(&config.instance_id, &config.plugin_config));
        diagnostics
    }

    fn build(
        &self,
        spec: &ComponentSpec,
        _ctx: &BuildContext,
    ) -> Result<Box<dyn OptimizerComponent>> {
        let config = parse_component_config::<ExternalComponentConfig>(&spec.config, self.kind())?;
        let Some(handler) = hosted_plugin_handler(&config.plugin_kind) else {
            return Err(OptimizerError::NotFound(format!(
                "hosted plugin '{}' is not registered",
                config.plugin_kind
            )));
        };
        Ok(Box::new(ExternalComponent {
            instance_id: config.instance_id,
            plugin_config: config.plugin_config,
            handler,
        }))
    }
}

struct ExternalComponent {
    instance_id: String,
    plugin_config: Map<String, Json>,
    handler: Arc<dyn HostedPluginHandler>,
}

impl OptimizerComponent for ExternalComponent {
    fn kind(&self) -> &'static str {
        "external_component"
    }

    fn register<'a>(
        &'a mut self,
        ctx: &'a mut RegistrationContext<'_>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let mut hosted_ctx = HostedRegistrationContext::new();
            if let Err(err) =
                self.handler
                    .register(&self.instance_id, &self.plugin_config, &mut hosted_ctx)
            {
                let mut pending = hosted_ctx.finish();
                rollback_registrations(&mut pending);
                return Err(err);
            }
            ctx.registrations.extend(hosted_ctx.finish());
            Ok(())
        })
    }
}

fn component_factory(kind: &str) -> Option<Arc<dyn OptimizerComponentFactory>> {
    COMPONENT_FACTORIES
        .read()
        .ok()
        .and_then(|guard| guard.get(kind).cloned())
}

fn hosted_plugin_handler(plugin_kind: &str) -> Option<Arc<dyn HostedPluginHandler>> {
    HOSTED_PLUGIN_HANDLERS
        .read()
        .ok()
        .and_then(|guard| guard.get(plugin_kind).cloned())
}

fn is_builtin_component_kind(kind: &str) -> bool {
    matches!(
        kind,
        "telemetry" | "dynamo_hints" | "tool_parallelism" | "external_component"
    )
}

fn validate_component_multiplicity(report: &mut ConfigReport, config: &OptimizerConfig) {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for spec in &config.components {
        *counts.entry(spec.kind.as_str()).or_default() += 1;
    }

    let mut emitted = HashSet::new();
    for spec in &config.components {
        let count = counts.get(spec.kind.as_str()).copied().unwrap_or_default();
        if count <= 1 || !emitted.insert(spec.kind.clone()) {
            continue;
        }

        let allows_multiple = component_factory(&spec.kind)
            .map(|factory| factory.allows_multiple_instances())
            .unwrap_or(false);
        if !allows_multiple {
            report.diagnostics.push(ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: "optimizer.duplicate_component".to_string(),
                component: Some(spec.kind.clone()),
                field: None,
                message: format!("component kind '{}' may only appear once", spec.kind),
            });
        }
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

fn parse_component_config<T>(config: &Map<String, Json>, component: &str) -> Result<T>
where
    T: for<'de> Deserialize<'de>,
{
    serde_json::from_value(Json::Object(config.clone())).map_err(|e| {
        OptimizerError::InvalidConfig(format!("invalid config for component '{component}': {e}"))
    })
}

fn validate_backend(report: &mut ConfigReport, policy: &ConfigPolicy, backend: &BackendSpec) {
    let kind = backend.kind.as_str();
    let known_fields: &[&str] = match kind {
        "in_memory" => &[],
        "redis" => &["url", "key_prefix"],
        _ => {
            push_policy_diag(
                &mut report.diagnostics,
                policy.unknown_component,
                "optimizer.unknown_backend",
                Some(kind.to_string()),
                None,
                format!("backend kind '{kind}' is unsupported"),
            );
            return;
        }
    };

    let mut diagnostics = vec![];
    validate_unknown_fields(
        &mut diagnostics,
        policy,
        Some(kind.to_string()),
        &backend.config,
        known_fields,
    );
    report.diagnostics.extend(diagnostics);
}

fn validate_unknown_fields(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    component: Option<String>,
    config: &Map<String, Json>,
    known_fields: &[&str],
) {
    let known: HashSet<&str> = known_fields.iter().copied().collect();
    for field in config.keys() {
        if !known.contains(field.as_str()) {
            push_policy_diag(
                diagnostics,
                policy.unknown_field,
                "optimizer.unknown_field",
                component.clone(),
                Some(field.clone()),
                format!(
                    "field '{}' is not recognized for component/backend '{}'",
                    field,
                    component.as_deref().unwrap_or("unknown")
                ),
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

async fn build_backend(spec: &BackendSpec) -> Result<Arc<dyn StorageBackendDyn + Send + Sync>> {
    match spec.kind.as_str() {
        "in_memory" => Ok(Arc::new(InMemoryBackend::default())),
        #[cfg(feature = "redis-backend")]
        "redis" => {
            let url = spec
                .config
                .get("url")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    OptimizerError::InvalidConfig(
                        "redis backend requires string field 'url'".into(),
                    )
                })?;
            let key_prefix = spec
                .config
                .get("key_prefix")
                .and_then(|v| v.as_str())
                .unwrap_or("nexus:");
            let backend = RedisBackend::new(url, key_prefix.to_string()).await?;
            Ok(Arc::new(backend))
        }
        other => Err(OptimizerError::InvalidConfig(format!(
            "unsupported backend kind '{other}'"
        ))),
    }
}

fn register_subscriber_impl(
    registrations: &mut Vec<ComponentRegistration>,
    kind: &str,
    name: &str,
    callback: EventSubscriberFn,
) -> Result<()> {
    nat_nexus_register_subscriber(name, callback)
        .map_err(|e| OptimizerError::RegistrationFailed(format!("subscriber: {e}")))?;

    let name_owned = name.to_string();
    registrations.push(ComponentRegistration::new(
        kind.to_string(),
        name_owned.clone(),
        Box::new(move || {
            nat_nexus_deregister_subscriber(&name_owned)
                .map(|_| ())
                .map_err(|e| {
                    OptimizerError::RegistrationFailed(format!(
                        "subscriber deregistration failed: {e}"
                    ))
                })
        }),
    ));
    Ok(())
}

fn register_llm_request_intercept_impl(
    registrations: &mut Vec<ComponentRegistration>,
    kind: &str,
    name: &str,
    priority: i32,
    break_chain: bool,
    callback: LlmRequestInterceptFn,
) -> Result<()> {
    nat_nexus_register_llm_request_intercept(name, priority, break_chain, callback).map_err(
        |e| OptimizerError::RegistrationFailed(format!("dynamo request intercept: {e}")),
    )?;

    let name_owned = name.to_string();
    registrations.push(ComponentRegistration::new(
        kind.to_string(),
        name_owned.clone(),
        Box::new(move || {
            nat_nexus_deregister_llm_request_intercept(&name_owned)
                .map(|_| ())
                .map_err(|e| {
                    OptimizerError::RegistrationFailed(format!(
                        "llm request intercept deregistration failed: {e}"
                    ))
                })
        }),
    ));
    Ok(())
}

fn register_llm_execution_intercept_impl(
    registrations: &mut Vec<ComponentRegistration>,
    kind: &str,
    name: &str,
    priority: i32,
    callback: LlmExecutionFn,
) -> Result<()> {
    nat_nexus_register_llm_execution_intercept(name, priority, callback)
        .map_err(|e| OptimizerError::RegistrationFailed(format!("llm execution intercept: {e}")))?;

    let name_owned = name.to_string();
    registrations.push(ComponentRegistration::new(
        kind.to_string(),
        name_owned.clone(),
        Box::new(move || {
            nat_nexus_deregister_llm_execution_intercept(&name_owned)
                .map(|_| ())
                .map_err(|e| {
                    OptimizerError::RegistrationFailed(format!(
                        "llm execution intercept deregistration failed: {e}"
                    ))
                })
        }),
    ));
    Ok(())
}

fn register_llm_stream_execution_intercept_impl(
    registrations: &mut Vec<ComponentRegistration>,
    kind: &str,
    name: &str,
    priority: i32,
    callback: LlmStreamExecutionFn,
) -> Result<()> {
    nat_nexus_register_llm_stream_execution_intercept(name, priority, callback).map_err(|e| {
        OptimizerError::RegistrationFailed(format!("llm stream execution intercept: {e}"))
    })?;

    let name_owned = name.to_string();
    registrations.push(ComponentRegistration::new(
        kind.to_string(),
        name_owned.clone(),
        Box::new(move || {
            nat_nexus_deregister_llm_stream_execution_intercept(&name_owned)
                .map(|_| ())
                .map_err(|e| {
                    OptimizerError::RegistrationFailed(format!(
                        "llm stream execution intercept deregistration failed: {e}"
                    ))
                })
        }),
    ));
    Ok(())
}

fn register_tool_request_intercept_impl(
    registrations: &mut Vec<ComponentRegistration>,
    kind: &str,
    name: &str,
    priority: i32,
    break_chain: bool,
    callback: ToolInterceptFn,
) -> Result<()> {
    nat_nexus_register_tool_request_intercept(name, priority, break_chain, callback)
        .map_err(|e| OptimizerError::RegistrationFailed(format!("tool request intercept: {e}")))?;

    let name_owned = name.to_string();
    registrations.push(ComponentRegistration::new(
        kind.to_string(),
        name_owned.clone(),
        Box::new(move || {
            nat_nexus_deregister_tool_request_intercept(&name_owned)
                .map(|_| ())
                .map_err(|e| {
                    OptimizerError::RegistrationFailed(format!(
                        "tool request intercept deregistration failed: {e}"
                    ))
                })
        }),
    ));
    Ok(())
}

fn register_tool_execution_intercept_impl(
    registrations: &mut Vec<ComponentRegistration>,
    kind: &str,
    name: &str,
    priority: i32,
    callback: ToolExecutionFn,
) -> Result<()> {
    nat_nexus_register_tool_execution_intercept(name, priority, callback)
        .map_err(|e| OptimizerError::RegistrationFailed(format!("tool intercept: {e}")))?;

    let name_owned = name.to_string();
    registrations.push(ComponentRegistration::new(
        kind.to_string(),
        name_owned.clone(),
        Box::new(move || {
            nat_nexus_deregister_tool_execution_intercept(&name_owned)
                .map(|_| ())
                .map_err(|e| {
                    OptimizerError::RegistrationFailed(format!(
                        "tool execution intercept deregistration failed: {e}"
                    ))
                })
        }),
    ));
    Ok(())
}

fn rollback_registrations(registrations: &mut Vec<ComponentRegistration>) {
    for registration in registrations.iter_mut().rev() {
        let _ = (registration.deregister)();
    }
    registrations.clear();
}
