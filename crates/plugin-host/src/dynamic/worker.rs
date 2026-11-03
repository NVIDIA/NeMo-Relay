// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! gRPC worker dynamic plugin loader and host-side proxy adapter.

#[cfg(test)]
use crate::HostPluginRegistrationContext;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use chrono::{DateTime, Utc};
use futures_util::FutureExt;
use nemo_relay_diagnostics::{
    FailureClass as DiagnosticFailureClass, PluginOperationFailureCode as DiagnosticOperation,
    PluginPhase as DiagnosticPhase,
};
use nemo_relay_worker_proto::DOWNSTREAM_CONTINUATION_METADATA;
use nemo_relay_worker_proto::conversions::{self, WireValue};
use nemo_relay_worker_proto::nvidia_worker_v1 as wire;
use nemo_relay_worker_proto::nvidia_worker_v1::WORKER_PROTOCOL;
use nemo_relay_worker_proto::nvidia_worker_v1::plugin_worker_service_client::PluginWorkerServiceClient as PluginWorkerClient;
use nemo_relay_worker_proto::nvidia_worker_v1::relay_host_runtime_service_server::{
    RelayHostRuntimeService as RelayHostRuntime,
    RelayHostRuntimeServiceServer as RelayHostRuntimeServer,
};
use nemo_relay_worker_proto::nvidia_worker_v1::*;
use nemo_relay_worker_proto::{
    CallbackMessageLimitService, DEFAULT_WORKER_MAX_MESSAGE_BYTES,
    WORKER_MAX_MESSAGE_BYTES_BY_NAME_ENV, WORKER_MAX_MESSAGE_BYTES_ENV,
    WorkerCallbackMessageLimits,
};
type JsonEnvelope = wire::JsonValue;
type StreamChunk = wire::InvokeStreamResponse;
type LogLevel = wire::LogSeverity;
type ProtoRuntimeDiagnostic = wire::RuntimeDiagnostic;
type ProtoLlmCodecIdentity = wire::LlmCodecReference;
type ProtoLlmSanitizeRequestContext = wire::LlmCodecContext;
type ProtoLlmSanitizeResponseContext = wire::LlmCodecContext;
type ProtoToolExecutionInterceptOutcome = wire::ToolExecutionInterceptOutcome;
#[path = "worker_transport.rs"]
mod transport;
use serde_json::{Map, Value as Json};
use sha2::{Digest, Sha256};
use tokio::runtime::{Builder as RuntimeBuilder, Runtime};
use tokio::sync::{mpsc, oneshot, watch};
use tokio_stream::{Stream, StreamExt};
use tonic::transport::{Channel, Endpoint, Server};
use tonic::{Request, Response, Status};
use transport::{
    ExecutionCodecContext as ProtoLlmExecutionCodecContext, LlmInvocation, PreparedInvocation,
    ToolInvocation,
};
use uuid::Uuid;

#[cfg(unix)]
use hyper_util::rt::TokioIo;
#[cfg(not(unix))]
use std::net::{SocketAddr, TcpListener};
#[cfg(unix)]
use std::os::unix::net::UnixListener as StdUnixListener;
#[cfg(not(unix))]
use tokio::net::TcpListener as TokioTcpListener;
#[cfg(unix)]
use tokio::net::{UnixListener, UnixStream};
#[cfg(not(unix))]
use tokio_stream::wrappers::TcpListenerStream;
#[cfg(unix)]
use tokio_stream::wrappers::UnixListenerStream;
#[cfg(unix)]
use tower::service_fn;

use crate::{
    ConfigDiagnostic, DiagnosticLevel, Plugin, PluginError, active_runtime_diagnostics_snapshot,
    deregister_plugin_registration_checked, register_plugin_tracked,
};
use nemo_relay_runtime::api::event::{Event, EventSanitizeFields};
use nemo_relay_runtime::api::llm::LlmRequest;
use nemo_relay_runtime::api::registry::{
    RuntimeRegistrationKind, deregister_conditional_middleware_guardrail,
    list_runtime_registrations, register_conditional_middleware_guardrail,
};
use nemo_relay_runtime::api::runtime::subscriber_dispatcher::{
    PublicationBuffer, capture_nested_publication_buffer, with_nested_publication_buffer,
};
use nemo_relay_runtime::api::runtime::{
    EventMetadataInjectorFn, EventSanitizeFn, LlmCodecIdentity, LlmExecutionContext,
    LlmExecutionNextFn, LlmJsonStream, LlmSanitizeRequestContext, LlmSanitizeResponseContext,
    LlmStreamExecutionNextFn, LlmStreamInner, MiddlewareContinuationContext, ToolExecutionContext,
    ToolExecutionNextFn, current_scope_stack, with_scope_stack,
};
use nemo_relay_runtime::api::scope::{
    EmitMarkEventParams, PopScopeParams, PushScopeParams, ScopeAttributes, ScopeHandle, ScopeType,
    event as emit_scope_mark, pop_scope, push_scope,
};
use nemo_relay_runtime::api::tool::{ToolExecutionInterceptOutcome, ToolExecutionResult};
use nemo_relay_types::codec::request::{ANNOTATED_LLM_REQUEST_SCHEMA, AnnotatedLlmRequest};
use nemo_relay_types::codec::response::AnnotatedLlmResponse;
use nemo_relay_types::codec::traits::{LlmCodec, LlmResponseCodec};
use nemo_relay_types::error::{FlowError, Result as FlowResult};

use super::trust::DynamicPluginLoadTrust;
use super::{
    DynamicPluginKind, DynamicPluginManifest, DynamicPluginManifestLoad,
    DynamicPluginTeardownOutcome, WorkerRuntime, deregister_tracked_registrations_checked,
    validate_annotated_request_consumer_compatibility, validate_dynamic_plugin_relay_compatibility,
    validate_llm_execution_context_compatibility, validate_tool_execution_context_compatibility,
};

const JSON_SCHEMA: &str = "nemo.relay.Json@1";
const LLM_REQUEST_SCHEMA: &str = "nemo.relay.LlmRequest@1";
const WORKER_STARTUP_TIMEOUT: Duration = Duration::from_secs(10);
const WORKER_RPC_TIMEOUT: Duration = Duration::from_secs(30);
const WORKER_CONNECT_RETRY: Duration = Duration::from_millis(25);
const MANAGED_ENVIRONMENTS_DIR: &str = ".dynamic-plugin-environments";
const PYTHON_WORKER_BOOTSTRAP: &str = r#"
import asyncio
import importlib
import inspect
import sys

target = sys.argv[1]
module_name, separator, function_name = target.partition(":")
if not separator or not module_name or not function_name:
    raise SystemExit("Python worker entrypoint must be 'module:function'")

entrypoint = getattr(importlib.import_module(module_name), function_name)
result = entrypoint()
if inspect.isawaitable(result):
    asyncio.run(result)
"#;

/// Worker plugin load request derived from host dynamic-plugin state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerPluginLoadSpec {
    /// Expected plugin id.
    pub plugin_id: String,
    /// Path to the authored `relay-plugin.toml`.
    pub manifest_ref: String,
    /// Relay-managed per-plugin runtime environment.
    ///
    /// This is required for Python workers and ignored by other worker runtimes.
    pub environment_ref: Option<String>,
    /// Resolved dynamic plugin config passed to the worker.
    pub config: Map<String, Json>,
}

/// Owns gRPC worker processes registered into the plugin registry.
///
/// Dropping this value deregisters worker plugin kinds and shuts down worker
/// processes. Clear active plugin configuration before dropping it so runtime
/// callbacks cannot outlive the worker activation.
pub struct WorkerPluginActivation {
    plugins: Vec<Arc<WorkerPluginInstance>>,
    plugin_registrations: Vec<(String, u64)>,
}

impl WorkerPluginActivation {
    /// Returns `true` when no worker plugins were loaded.
    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }

    /// Consumes the activation; deregistration runs from `Drop`.
    pub fn clear(self) {}

    pub(crate) fn deregister_plugin_kinds_checked(&mut self) -> DynamicPluginTeardownOutcome {
        deregister_tracked_registrations_checked(&mut self.plugin_registrations, "worker")
    }

    pub(crate) fn shutdown_plugins_checked(&self) -> DynamicPluginTeardownOutcome {
        let mut outcome = DynamicPluginTeardownOutcome::success();
        for plugin in self.plugins.iter().rev() {
            outcome.merge(plugin.shutdown_checked());
        }
        outcome
    }
}

impl Drop for WorkerPluginActivation {
    fn drop(&mut self) {
        for (plugin_kind, registration_id) in self.plugin_registrations.iter().rev() {
            let _ = deregister_plugin_registration_checked(plugin_kind, *registration_id);
        }
    }
}

/// Loads gRPC worker plugins and registers their plugin kinds.
///
/// The returned activation must be kept alive until after active plugin
/// configuration has been cleared.
pub fn load_worker_plugins<I>(specs: I) -> crate::Result<WorkerPluginActivation>
where
    I: IntoIterator<Item = WorkerPluginLoadSpec>,
{
    load_worker_plugins_with_diagnostics(
        specs,
        nemo_relay_diagnostics::Diagnostics::process_default(),
    )
}

/// Load plugin workers using the runtime owner's selected diagnostic context.
/// Diagnostic admission never controls whether an execution activation succeeds.
pub fn load_worker_plugins_with_diagnostics<I>(
    specs: I,
    diagnostics: &nemo_relay_diagnostics::Diagnostics,
) -> crate::Result<WorkerPluginActivation>
where
    I: IntoIterator<Item = WorkerPluginLoadSpec>,
{
    load_worker_plugins_inner(specs, false, &HashMap::new(), &mut Vec::new(), diagnostics)
}

/// Prepare workers without calling Register until their host component activates.
pub(super) fn prepare_worker_plugins_with_trust<I>(
    specs: I,
    trust: &HashMap<String, DynamicPluginLoadTrust>,
    skipped: &mut Vec<super::DynamicPluginValidationReport>,
    diagnostics: &nemo_relay_diagnostics::Diagnostics,
) -> crate::Result<WorkerPluginActivation>
where
    I: IntoIterator<Item = WorkerPluginLoadSpec>,
{
    load_worker_plugins_inner(specs, true, trust, skipped, diagnostics)
}

fn load_worker_plugins_inner<I>(
    specs: I,
    defer_registration: bool,
    trust: &HashMap<String, DynamicPluginLoadTrust>,
    skipped: &mut Vec<super::DynamicPluginValidationReport>,
    diagnostics: &nemo_relay_diagnostics::Diagnostics,
) -> crate::Result<WorkerPluginActivation>
where
    I: IntoIterator<Item = WorkerPluginLoadSpec>,
{
    let mut activation = WorkerPluginActivation {
        plugins: Vec::new(),
        plugin_registrations: Vec::new(),
    };
    for spec in specs {
        let manifest = match trust.get(&spec.plugin_id) {
            Some(trust) => match trust.manifest_for_load(&spec.manifest_ref, skipped)? {
                Some(manifest) => Some(manifest),
                None => continue,
            },
            None => None,
        };
        let instance = load_one_worker_plugin(&spec, defer_registration, manifest, diagnostics)?;
        let plugin_kind = instance.plugin_kind.clone();
        let registration_id = register_plugin_tracked(Arc::new(WorkerPluginAdapter {
            plugin_kind: plugin_kind.clone(),
            allows_multiple_components: instance.allows_multiple_components,
            instance: instance.clone(),
        }))?;
        activation.plugins.push(instance);
        activation
            .plugin_registrations
            .push((plugin_kind, registration_id));
    }
    Ok(activation)
}

struct WorkerPluginAdapter {
    plugin_kind: String,
    allows_multiple_components: bool,
    instance: Arc<WorkerPluginInstance>,
}

impl Plugin for WorkerPluginAdapter {
    fn plugin_kind(&self) -> &str {
        &self.plugin_kind
    }

    fn allows_multiple_components(&self) -> bool {
        self.allows_multiple_components
    }

    fn validate(&self, plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        if plugin_config != &self.instance.config {
            return vec![worker_error_diagnostic(
                &self.plugin_kind,
                "plugin.worker_config_mismatch",
                "worker plugin config changed after dynamic activation; reload the worker activation",
            )];
        }
        self.instance.validation_diagnostics.clone()
    }

    fn register<'a>(
        &'a self,
        plugin_config: &Map<String, Json>,
        ctx: &'a mut dyn crate::PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = crate::Result<()>> + Send + 'a>> {
        let config_matches = plugin_config == &self.instance.config;
        Box::pin(async move {
            if !config_matches {
                return Err(PluginError::RegistrationFailed(
                    "worker plugin config changed after dynamic activation; reload the worker activation"
                        .into(),
                ));
            }
            self.instance.prepare_registrations().await?;
            self.instance.install_registrations(ctx)
        })
    }
}

struct WorkerPluginInstance {
    callback_timeouts: super::WorkerCallbackTimeouts,
    plugin_kind: String,
    allows_multiple_components: bool,
    config: Map<String, Json>,
    validation_diagnostics: Vec<ConfigDiagnostic>,
    relay_compat: String,
    registrations: tokio::sync::OnceCell<Vec<Registration>>,
    runtime: OwnedWorkerRuntime,
    client: PluginWorkerClient<Channel>,
    host_state: Arc<WorkerHostRuntimeState>,
    shutdown: Mutex<Option<oneshot::Sender<()>>>,
    process: Mutex<Option<Child>>,
    activation_dir: PathBuf,
    teardown_started: AtomicBool,
}

impl Drop for WorkerPluginInstance {
    fn drop(&mut self) {
        let outcome = self.shutdown_checked();
        if !outcome.errors.is_empty() {
            log::error!(
                target: "nemo_relay.worker",
                event = "worker_cleanup_failed",
                plugin_id = self.plugin_kind.as_str(),
                failure_count = outcome.errors.len(),
                safe_to_unload = outcome.safe_to_unload;
                "Worker plugin cleanup failed during drop"
            );
        }
    }
}

impl WorkerPluginInstance {
    fn shutdown_checked(&self) -> DynamicPluginTeardownOutcome {
        let mut outcome = DynamicPluginTeardownOutcome::success();
        if self.teardown_started.swap(true, Ordering::AcqRel) {
            return outcome;
        }

        log::info!(
            target: "nemo_relay.worker",
            event = "worker_stopping",
            plugin_id = self.plugin_kind.as_str();
            "Worker plugin is stopping"
        );

        if let Err(error) = self.host_state.close_scope_admission() {
            outcome.record_error(
                format!("worker scope admission shutdown failed: {error}"),
                false,
            );
        }
        self.host_state.cleanup_conditional_middleware_guardrails();

        let mut client =
            message_limited_client(&self.client, &self.host_state.message_limits, None);
        let request = ShutdownRequest {
            credentials: Some(wire::Credentials {
                activation_id: self.host_state.activation_id.clone(),
                auth_token: self.host_state.auth_token.clone(),
            }),

            reason: "plugin activation cleared".into(),
        };
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            block_on_runtime(self.runtime.runtime(), async move {
                worker_rpc(client.shutdown(worker_rpc_request(request)))
                    .await
                    .inspect_err(|error| {
                        self.host_state.observer.rpc_failure(
                            DiagnosticOperation::WorkerShutdown,
                            error,
                            DiagnosticPhase::Shutdown,
                        );
                    })
            })
        })) {
            Ok(Ok(_)) => {}
            Ok(Err(error)) => outcome.record_error(
                format!(
                    "worker plugin '{}' shutdown RPC failed: {error}",
                    self.plugin_kind
                ),
                true,
            ),
            Err(payload) => outcome.record_error(
                format!(
                    "worker plugin '{}' shutdown RPC panicked: {}",
                    self.plugin_kind,
                    panic_payload_message(payload.as_ref())
                ),
                true,
            ),
        }

        if let Err(error) =
            block_on_runtime(self.runtime.runtime(), self.host_state.drain_owned_scopes())
        {
            outcome.record_error(format!("worker owned scope cleanup failed: {error}"), false);
        }
        if let Some(diagnostics) = &self.host_state.diagnostics {
            diagnostics.revoke();
        }
        match self.shutdown.lock() {
            Ok(mut shutdown) => {
                if let Some(sender) = shutdown.take()
                    && sender.send(()).is_err()
                {
                    outcome.record_error(
                        format!(
                            "worker plugin '{}' host runtime shutdown channel was closed",
                            self.plugin_kind
                        ),
                        true,
                    );
                }
            }
            Err(error) => outcome.record_error(
                format!(
                    "worker plugin '{}' host runtime shutdown lock poisoned: {error}",
                    self.plugin_kind
                ),
                true,
            ),
        }

        let exit_errors = outcome.errors.len();
        self.stop_process_checked(&mut outcome);
        if outcome.errors.len() > exit_errors {
            self.host_state.observer.failure(
                DiagnosticOperation::WorkerExit,
                DiagnosticFailureClass::Unknown,
                DiagnosticPhase::Shutdown,
            );
        } else {
            self.host_state.observer.success(
                nemo_relay_diagnostics::SuccessCode::PluginWorkerProcessCleaned,
                DiagnosticPhase::Shutdown,
            );
        }
        if outcome.safe_to_unload
            && let Err(error) = std::fs::remove_dir_all(&self.activation_dir)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            outcome.record_error(
                format!(
                    "worker plugin '{}' activation directory cleanup failed for '{}': {error}",
                    self.plugin_kind,
                    self.activation_dir.display()
                ),
                true,
            );
        }
        if outcome.errors.is_empty() {
            log::info!(
                target: "nemo_relay.worker",
                event = "worker_stopped",
                plugin_id = self.plugin_kind.as_str();
                "Worker plugin stopped"
            );
        }
        if !outcome.safe_to_unload {
            self.teardown_started.store(false, Ordering::Release);
        }
        outcome
    }

    fn stop_process_checked(&self, outcome: &mut DynamicPluginTeardownOutcome) {
        let mut process = match self.process.lock() {
            Ok(process) => process,
            Err(error) => {
                outcome.record_error(
                    format!(
                        "worker plugin '{}' process lock poisoned: {error}",
                        self.plugin_kind
                    ),
                    false,
                );
                return;
            }
        };
        let Some(child) = process.as_mut() else {
            return;
        };
        match child.try_wait() {
            Ok(Some(_)) => {
                process.take();
            }
            Ok(None) => {
                if let Err(kill_error) = child.kill() {
                    match child.try_wait() {
                        Ok(Some(_)) => {
                            process.take();
                            outcome.record_error(
                                format!(
                                    "worker plugin '{}' process kill failed after exit: {kill_error}",
                                    self.plugin_kind
                                ),
                                true,
                            );
                        }
                        Ok(None) | Err(_) => outcome.record_error(
                            format!(
                                "worker plugin '{}' process kill failed: {kill_error}",
                                self.plugin_kind
                            ),
                            false,
                        ),
                    }
                    return;
                }
                match child.wait() {
                    Ok(_) => {
                        process.take();
                    }
                    Err(error) => outcome.record_error(
                        format!(
                            "worker plugin '{}' process wait failed after kill: {error}",
                            self.plugin_kind
                        ),
                        false,
                    ),
                }
            }
            Err(error) => outcome.record_error(
                format!(
                    "worker plugin '{}' process status check failed: {error}",
                    self.plugin_kind
                ),
                false,
            ),
        }
    }
}

fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("unknown panic payload")
}

/// Starts and validates a worker, then stores its settings for this activation.
fn load_one_worker_plugin(
    spec: &WorkerPluginLoadSpec,
    defer_registration: bool,
    verified_manifest: Option<DynamicPluginManifest>,
    diagnostics: &nemo_relay_diagnostics::Diagnostics,
) -> crate::Result<Arc<WorkerPluginInstance>> {
    use nemo_relay_diagnostics::{
        FailureClass, PluginExecution, PluginOperationFailureCode as Operation, PluginPhase,
    };
    let activation_uuid = Uuid::now_v7();
    let observer = super::diagnostics::PluginDiagnostics::new(
        diagnostics,
        &spec.plugin_id,
        PluginExecution::WorkerProtocol,
        activation_uuid,
    );
    log::info!(
        target: "nemo_relay.worker",
        event = "worker_starting",
        plugin_id = spec.plugin_id.as_str();
        "Worker plugin is starting"
    );
    let (manifest, manifest_ref) = match verified_manifest {
        Some(manifest) => (manifest, spec.manifest_ref.clone()),
        None => DynamicPluginManifest::load_from_path(&spec.manifest_ref).inspect_err(|_| {
            observer.failure(
                Operation::WorkerValidation,
                FailureClass::Unknown,
                PluginPhase::Validation,
            );
        })?,
    };
    let (relay_compat, runtime, entrypoint, callback_timeouts, default_message_bytes) = (|| {
        if manifest.plugin.id.trim() != spec.plugin_id {
            return Err(PluginError::InvalidConfig(format!(
                "dynamic plugin manifest id '{}' does not match expected id '{}'",
                manifest.plugin.id, spec.plugin_id
            )));
        }
        if manifest.plugin.kind != DynamicPluginKind::Worker {
            return Err(PluginError::InvalidConfig(format!(
                "dynamic plugin '{}' is kind {}; worker loader only supports worker",
                spec.plugin_id, manifest.plugin.kind
            )));
        }
        validate_relay_compatibility(manifest.compat.relay.as_deref())?;
        let relay_compat = manifest
            .compat
            .relay
            .as_deref()
            .expect("validated worker manifest must declare compat.relay")
            .to_string();
        let DynamicPluginManifestLoad::Worker(load) = &manifest.load else {
            unreachable!("validated worker manifest must carry worker load contract");
        };
        let runtime = load
            .runtime
            .ok_or_else(|| PluginError::InvalidConfig("load.runtime is required".into()))?;
        let entrypoint = load
            .entrypoint
            .as_deref()
            .ok_or_else(|| PluginError::InvalidConfig("load.entrypoint is required".into()))?
            .to_string();
        let callback_timeouts = load.callback_timeouts.clone().unwrap_or_default();
        let authored = load.max_message_bytes.clone().unwrap_or_default();
        let default_message_bytes = WorkerCallbackMessageLimits {
            default_bytes: authored
                .default_bytes
                .map_or(DEFAULT_WORKER_MAX_MESSAGE_BYTES, |v| v.get() as usize),
            by_name: authored
                .by_name
                .into_iter()
                .map(|(k, v)| (k, v.get() as usize))
                .collect(),
        };
        Ok((
            relay_compat,
            runtime,
            entrypoint,
            callback_timeouts,
            default_message_bytes,
        ))
    })()
    .inspect_err(|_| {
        observer.failure(
            Operation::WorkerValidation,
            FailureClass::Unknown,
            PluginPhase::Validation,
        );
    })?;

    let max_message_bytes = default_message_bytes
        .from_env_with_defaults()
        .map_err(PluginError::InvalidConfig)
        .inspect_err(|_| {
            observer.failure(
                Operation::WorkerValidation,
                FailureClass::Unknown,
                PluginPhase::Validation,
            );
        })?;
    let activation_id = activation_uuid.to_string();
    let auth_token = Uuid::now_v7().to_string();
    let activation_dir = std::env::temp_dir().join(format!("nmrw-{}", activation_uuid.simple()));
    std::fs::create_dir_all(&activation_dir)
        .map_err(|err| PluginError::Internal(format!("worker activation directory: {err}")))
        .inspect_err(|_| {
            observer.failure(
                Operation::WorkerLaunch,
                FailureClass::Unknown,
                PluginPhase::Activation,
            );
        })?;
    let mut activation_dir_guard = ActivationDirGuard::new(activation_dir.clone());
    let runtime_handle = OwnedWorkerRuntime::new(
        RuntimeBuilder::new_multi_thread()
            .enable_all()
            .thread_name("nemo-relay-worker-host")
            .build()
            .map_err(|err| PluginError::Internal(format!("worker runtime: {err}")))
            .inspect_err(|_| {
                observer.failure(
                    Operation::WorkerLaunch,
                    FailureClass::Unknown,
                    PluginPhase::Activation,
                );
            })?,
    );
    let WorkerEndpoints {
        host_server,
        host_advertise,
        worker_advertise,
        worker_connect,
        worker_endpoint_file,
    } = WorkerEndpoints::new(&activation_dir).inspect_err(|_| {
        observer.failure(
            Operation::WorkerTransport,
            FailureClass::Unknown,
            PluginPhase::Activation,
        );
    })?;
    let mut host_state = WorkerHostRuntimeState::new(activation_id.clone(), auth_token.clone());
    host_state.message_limits = max_message_bytes.clone();
    host_state.diagnostics = Some(Arc::new(diagnostic_transport::WorkerDiagnostics::new(
        diagnostics.clone(),
        activation_uuid,
    )));
    host_state.observer = observer.clone();
    let host_state = Arc::new(host_state);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    runtime_handle.runtime().spawn(serve_host_runtime(
        host_server,
        host_state.clone(),
        shutdown_rx,
        max_message_bytes.clone(),
    ));

    let manifest_path = PathBuf::from(&manifest_ref);
    let mut child = ChildGuard::new(
        spawn_worker_process(WorkerProcessLaunch {
            runtime,
            manifest_path: &manifest_path,
            environment_ref: spec.environment_ref.as_deref(),
            plugin_id: &spec.plugin_id,
            entrypoint: &entrypoint,
            forward_env: match &manifest.load {
                DynamicPluginManifestLoad::Worker(load) => &load.forward_env,
                _ => unreachable!("validated worker manifest"),
            },
            activation_id: &activation_id,
            auth_token: &auth_token,
            host_endpoint: &host_advertise,
            worker_endpoint: &worker_advertise,
            worker_endpoint_file: worker_endpoint_file.as_deref(),
            max_message_bytes: &max_message_bytes,
        })
        .inspect_err(|_| {
            observer.failure(
                Operation::WorkerLaunch,
                FailureClass::Unknown,
                PluginPhase::Activation,
            )
        })?,
    );
    observer.success(
        nemo_relay_diagnostics::SuccessCode::PluginWorkerStarted,
        PluginPhase::Activation,
    );
    log::info!(
        target: "nemo_relay.worker",
        event = "worker_started",
        plugin_id = spec.plugin_id.as_str(),
        pid = child
            .child
            .as_ref()
            .expect("worker child should remain guarded")
            .id();
        "Worker plugin process started"
    );
    let client = block_on_runtime(
        runtime_handle.runtime(),
        connect_worker_with_retry(&worker_connect, &spec.plugin_id),
    )
    .inspect_err(|_| {
        observer.failure(
            Operation::WorkerTransport,
            FailureClass::Unknown,
            PluginPhase::Activation,
        )
    })?;

    let health = block_on_runtime(
        runtime_handle.runtime(),
        worker_rpc(
            message_limited_client(&client, &max_message_bytes, None).health(worker_rpc_request(
                HealthRequest {
                    credentials: Some(wire::Credentials {
                        activation_id: activation_id.clone(),
                        auth_token: auth_token.clone(),
                    }),
                },
            )),
        ),
    )
    .map_err(|err| {
        observer.rpc_failure(Operation::WorkerHandshake, &err, PluginPhase::Activation);
        PluginError::RegistrationFailed(format!("worker health check failed: {err}"))
    })?;
    let health = health.into_inner();
    nemo_relay_worker_proto::validation::validate(&health)
        .map_err(PluginError::RegistrationFailed)?;
    if health.status != wire::WorkerHealthStatus::Ready as i32 {
        observer.failure(
            Operation::WorkerHandshake,
            FailureClass::Unknown,
            PluginPhase::Activation,
        );
        let message = format!("worker plugin health check failed: {}", health.message);
        return Err(PluginError::RegistrationFailed(message));
    }

    let handshake = block_on_runtime(
        runtime_handle.runtime(),
        worker_rpc(
            message_limited_client(&client, &max_message_bytes, None).handshake(
                worker_rpc_request(HandshakeRequest {
                    credentials: Some(wire::Credentials {
                        activation_id: activation_id.clone(),
                        auth_token: auth_token.clone(),
                    }),
                    plugin_id: spec.plugin_id.clone(),
                    relay_version: env!("CARGO_PKG_VERSION").into(),
                    protocol: Some(wire::ProtocolIdentity {
                        name: WORKER_PROTOCOL.into(),
                        version: wire::CONTRACT_VERSION.into(),
                    }),
                    host_endpoint: host_advertise.clone(),
                }),
            ),
        ),
    )
    .map_err(|err| {
        if err.code() == tonic::Code::FailedPrecondition {
            observer.failure(
                Operation::WorkerHandshake,
                FailureClass::Incompatible,
                PluginPhase::Activation,
            );
        } else {
            observer.rpc_failure(Operation::WorkerHandshake, &err, PluginPhase::Activation);
        }
        PluginError::RegistrationFailed(format!("worker handshake failed: {err}"))
    })?;
    let handshake = handshake.into_inner();
    nemo_relay_worker_proto::validation::validate(&handshake)
        .map_err(PluginError::RegistrationFailed)?;
    validate_worker_handshake(&spec.plugin_id, &handshake).inspect_err(|_| {
        observer.failure(
            Operation::WorkerHandshake,
            FailureClass::Incompatible,
            PluginPhase::Activation,
        );
    })?;
    observer.success(
        nemo_relay_diagnostics::SuccessCode::PluginWorkerHandshakeCompleted,
        PluginPhase::Activation,
    );
    host_state.set_gate_callback(WorkerGateCallback {
        callback_timeouts: callback_timeouts.clone(),
        plugin_kind: spec.plugin_id.clone(),
        runtime: runtime_handle.runtime().handle().clone(),
        client: client.clone(),
        host_state: Arc::downgrade(&host_state),
    });

    let config = Json::Object(spec.config.clone());
    let validate = block_on_runtime(
        runtime_handle.runtime(),
        worker_rpc(
            message_limited_client(&client, &max_message_bytes, None).validate(worker_rpc_request(
                ValidateRequest {
                    credentials: Some(wire::Credentials {
                        activation_id: activation_id.clone(),
                        auth_token: auth_token.clone(),
                    }),
                    plugin_id: spec.plugin_id.clone(),
                    config: Some(encode_json(JSON_SCHEMA, &config)),
                },
            )),
        ),
    )
    .map_err(|err| {
        observer.rpc_failure(Operation::WorkerValidation, &err, PluginPhase::Validation);
        PluginError::RegistrationFailed(format!("worker validation RPC failed: {err}"))
    })?;
    let validate = validate.into_inner();
    nemo_relay_worker_proto::validation::validate(&validate)
        .map_err(PluginError::RegistrationFailed)?;
    let validation_diagnostics = match validate.result {
        Some(wire::validate_response::Result::Diagnostics(v)) => v
            .entries
            .into_iter()
            .map(|v| {
                conversions::decode(v).map_err(|e| PluginError::RegistrationFailed(e.to_string()))
            })
            .collect::<crate::Result<Vec<ConfigDiagnostic>>>()?,
        Some(wire::validate_response::Result::Error(e)) => {
            return Err(worker_error_to_plugin(e, "worker validation failed"));
        }
        None => {
            return Err(PluginError::RegistrationFailed(
                "missing worker validation result".into(),
            ));
        }
    };
    if diagnostics_have_errors(&validation_diagnostics) {
        observer.failure(
            Operation::WorkerValidation,
            FailureClass::Unknown,
            PluginPhase::Validation,
        );
    } else {
        observer.success(
            nemo_relay_diagnostics::SuccessCode::PluginWorkerValidated,
            PluginPhase::Validation,
        );
    }

    let instance = Arc::new(WorkerPluginInstance {
        callback_timeouts,
        plugin_kind: spec.plugin_id.clone(),
        allows_multiple_components: handshake.allows_multiple_components,
        config: spec.config.clone(),
        validation_diagnostics,
        relay_compat,
        registrations: tokio::sync::OnceCell::new(),
        runtime: runtime_handle,
        client,
        host_state,
        shutdown: Mutex::new(Some(shutdown_tx)),
        process: Mutex::new(Some(child.take())),
        activation_dir: activation_dir_guard.keep(),
        teardown_started: AtomicBool::new(false),
    });
    if !defer_registration && !diagnostics_have_errors(&instance.validation_diagnostics) {
        block_on_runtime(instance.runtime.runtime(), instance.prepare_registrations())?;
    }
    Ok(instance)
}

enum HostRuntimeServer {
    #[cfg(unix)]
    Unix(StdUnixListener),
    #[cfg(not(unix))]
    Tcp(TcpListener),
}

#[derive(Clone)]
enum WorkerConnectEndpoint {
    #[cfg(unix)]
    Unix(PathBuf),
    #[cfg(not(unix))]
    Tcp(String),
    #[cfg(not(unix))]
    Announced(PathBuf),
}

struct WorkerEndpoints {
    host_server: HostRuntimeServer,
    host_advertise: String,
    worker_advertise: String,
    worker_connect: WorkerConnectEndpoint,
    worker_endpoint_file: Option<PathBuf>,
}

impl WorkerEndpoints {
    fn new(activation_dir: &Path) -> crate::Result<Self> {
        #[cfg(not(unix))]
        let _ = activation_dir;

        #[cfg(unix)]
        {
            let host_socket = activation_dir.join("host.sock");
            let worker_socket = activation_dir.join("worker.sock");
            let _ = std::fs::remove_file(&host_socket);
            let host_listener = StdUnixListener::bind(&host_socket).map_err(|err| {
                PluginError::RegistrationFailed(format!(
                    "failed to bind worker host runtime socket '{}': {err}",
                    host_socket.display()
                ))
            })?;
            host_listener.set_nonblocking(true).map_err(|err| {
                PluginError::RegistrationFailed(format!(
                    "failed to configure worker host runtime socket '{}': {err}",
                    host_socket.display()
                ))
            })?;
            Ok(Self {
                host_server: HostRuntimeServer::Unix(host_listener),
                host_advertise: unix_endpoint_display(&host_socket),
                worker_advertise: unix_endpoint_display(&worker_socket),
                worker_connect: WorkerConnectEndpoint::Unix(worker_socket),
                worker_endpoint_file: None,
            })
        }

        #[cfg(not(unix))]
        {
            let (host_listener, host_addr) = bind_loopback_listener()?;
            let worker_endpoint_file = activation_dir.join("worker-endpoint");
            Ok(Self {
                host_server: HostRuntimeServer::Tcp(host_listener),
                host_advertise: format!("http://{host_addr}"),
                worker_advertise: "tcp://127.0.0.1:0".into(),
                worker_connect: WorkerConnectEndpoint::Announced(worker_endpoint_file.clone()),
                worker_endpoint_file: Some(worker_endpoint_file),
            })
        }
    }
}

/// Serves bounded execution RPCs without changing the separate diagnostic caps.
async fn serve_host_runtime(
    endpoint: HostRuntimeServer,
    state: Arc<WorkerHostRuntimeState>,
    shutdown: oneshot::Receiver<()>,
    max_message_bytes: WorkerCallbackMessageLimits,
) {
    let diagnostics = diagnostic_limits::BoundedService::new(
        nemo_relay_worker_proto::nvidia_worker_v1::relay_diagnostics_service_server::RelayDiagnosticsServiceServer::new(
            diagnostic_transport::Service {
                state: state.clone(),
            },
        )
        .max_decoding_message_size(diagnostic_transport::REQUEST_BYTES)
        .max_encoding_message_size(diagnostic_transport::RESPONSE_BYTES),
    );
    let service = CallbackMessageLimitService::new(
        RelayHostRuntimeServer::new(WorkerHostRuntimeService { state }),
        max_message_bytes,
        |server, limit| {
            server
                .max_decoding_message_size(limit)
                .max_encoding_message_size(limit)
        },
    );
    let result = match endpoint {
        #[cfg(unix)]
        HostRuntimeServer::Unix(listener) => {
            let listener = match UnixListener::from_std(listener) {
                Ok(listener) => listener,
                Err(_) => {
                    log::error!(
                        target: "nemo_relay.worker",
                        event = "worker_host_runtime_failed",
                        transport = "unix";
                        "Worker host runtime failed to attach its socket"
                    );
                    return;
                }
            };
            Server::builder()
                .add_service(diagnostics)
                .add_service(service)
                .serve_with_incoming_shutdown(UnixListenerStream::new(listener), async {
                    let _ = shutdown.await;
                })
                .await
        }
        #[cfg(not(unix))]
        HostRuntimeServer::Tcp(listener) => {
            let listener = match TokioTcpListener::from_std(listener) {
                Ok(listener) => listener,
                Err(_) => {
                    log::error!(
                        target: "nemo_relay.worker",
                        event = "worker_host_runtime_failed",
                        transport = "tcp";
                        "Worker host runtime failed to attach its endpoint"
                    );
                    return;
                }
            };
            Server::builder()
                .add_service(diagnostics)
                .add_service(service)
                .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                    let _ = shutdown.await;
                })
                .await
        }
    };
    if result.is_err() {
        log::error!(
            target: "nemo_relay.worker",
            event = "worker_host_runtime_failed",
            reason = "server_stopped";
            "Worker host runtime server failed"
        );
    }
}

async fn connect_worker_with_retry(
    endpoint: &WorkerConnectEndpoint,
    plugin_id: &str,
) -> crate::Result<PluginWorkerClient<Channel>> {
    let start = std::time::Instant::now();
    let mut attempts = 0_u64;
    let mut retry_logged = false;
    loop {
        let connect_endpoint = match resolve_worker_connect_endpoint(endpoint) {
            Ok(Some(endpoint)) => endpoint,
            Ok(None) if start.elapsed() < WORKER_STARTUP_TIMEOUT => {
                tokio::time::sleep(WORKER_CONNECT_RETRY).await;
                continue;
            }
            Ok(None) => {
                let message = format!(
                    "worker did not announce endpoint within {}s",
                    WORKER_STARTUP_TIMEOUT.as_secs()
                );
                return Err(PluginError::RegistrationFailed(message));
            }
            Err(err) => return Err(err),
        };
        match connect_worker(&connect_endpoint).await {
            Ok(client) => {
                if attempts > 0 {
                    log::info!(
                        target: "nemo_relay.worker",
                        event = "worker_connection_recovered",
                        plugin_id = plugin_id,
                        attempt_count = attempts + 1;
                        "Worker plugin connection recovered"
                    );
                }
                return Ok(client);
            }
            Err(_) if start.elapsed() < WORKER_STARTUP_TIMEOUT => {
                attempts = attempts.saturating_add(1);
                if !retry_logged {
                    log::warn!(
                        target: "nemo_relay.worker",
                        event = "worker_connection_retrying",
                        plugin_id = plugin_id;
                        "Worker plugin connection failed; retrying"
                    );
                    retry_logged = true;
                }
                tokio::time::sleep(WORKER_CONNECT_RETRY).await;
            }
            Err(err) => {
                let message = format!(
                    "worker did not start within {}s: {err}",
                    WORKER_STARTUP_TIMEOUT.as_secs()
                );
                return Err(PluginError::RegistrationFailed(message));
            }
        }
    }
}

#[cfg(not(unix))]
fn normalize_worker_tcp_endpoint(endpoint: &str) -> crate::Result<String> {
    let endpoint = endpoint.trim();
    if let Some(authority) = endpoint.strip_prefix("tcp://") {
        if authority.is_empty() {
            return Err(PluginError::RegistrationFailed(
                "worker announced an empty TCP endpoint".into(),
            ));
        }
        return Ok(format!("http://{authority}"));
    }
    if endpoint.starts_with("http://") {
        return Ok(endpoint.to_owned());
    }
    Err(PluginError::RegistrationFailed(format!(
        "worker announced unsupported endpoint '{endpoint}'"
    )))
}

fn resolve_worker_connect_endpoint(
    endpoint: &WorkerConnectEndpoint,
) -> crate::Result<Option<WorkerConnectEndpoint>> {
    match endpoint {
        #[cfg(unix)]
        WorkerConnectEndpoint::Unix(path) => Ok(Some(WorkerConnectEndpoint::Unix(path.clone()))),
        #[cfg(not(unix))]
        WorkerConnectEndpoint::Tcp(endpoint) => Ok(Some(WorkerConnectEndpoint::Tcp(
            normalize_worker_tcp_endpoint(endpoint)?,
        ))),
        #[cfg(not(unix))]
        WorkerConnectEndpoint::Announced(path) => {
            let endpoint = match std::fs::read_to_string(path) {
                Ok(endpoint) => endpoint,
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                Err(err) => {
                    return Err(PluginError::RegistrationFailed(format!(
                        "failed to read worker endpoint file '{}': {err}",
                        path.display()
                    )));
                }
            };
            let endpoint = endpoint.trim();
            if endpoint.is_empty() {
                // The worker may have created or truncated the announcement file immediately
                // before publishing its endpoint. Treat that transient state like a missing file
                // so the bounded startup loop polls again.
                return Ok(None);
            }
            Ok(Some(WorkerConnectEndpoint::Tcp(
                normalize_worker_tcp_endpoint(endpoint)?,
            )))
        }
    }
}

async fn connect_worker(
    endpoint: &WorkerConnectEndpoint,
) -> crate::Result<PluginWorkerClient<Channel>> {
    match endpoint {
        #[cfg(unix)]
        WorkerConnectEndpoint::Unix(socket) => {
            let path = Arc::new(socket.to_path_buf());
            let endpoint = Endpoint::try_from("http://[::]:50051")
                .map_err(|err| PluginError::Internal(format!("invalid worker endpoint: {err}")))?;
            let channel = endpoint
                .connect_with_connector(service_fn(move |_| {
                    let path = path.clone();
                    async move { UnixStream::connect(&*path).await.map(TokioIo::new) }
                }))
                .await
                .map_err(|err| {
                    PluginError::RegistrationFailed(format!(
                        "failed to connect to worker socket '{}': {err}",
                        socket.display()
                    ))
                })?;
            Ok(PluginWorkerClient::new(channel))
        }
        #[cfg(not(unix))]
        WorkerConnectEndpoint::Tcp(endpoint) => {
            let channel = Endpoint::from_shared(endpoint.clone())
                .map_err(|err| PluginError::Internal(format!("invalid worker endpoint: {err}")))?
                .connect()
                .await
                .map_err(|err| {
                    PluginError::RegistrationFailed(format!(
                        "failed to connect to worker endpoint '{endpoint}': {err}"
                    ))
                })?;
            Ok(PluginWorkerClient::new(channel))
        }
        #[cfg(not(unix))]
        WorkerConnectEndpoint::Announced(path) => Err(PluginError::Internal(format!(
            "worker endpoint file '{}' was not resolved before connect",
            path.display()
        ))),
    }
}

struct WorkerProcessLaunch<'a> {
    forward_env: &'a [String],
    runtime: WorkerRuntime,
    manifest_path: &'a Path,
    environment_ref: Option<&'a str>,
    plugin_id: &'a str,
    entrypoint: &'a str,
    activation_id: &'a str,
    auth_token: &'a str,
    host_endpoint: &'a str,
    worker_endpoint: &'a str,
    worker_endpoint_file: Option<&'a Path>,
    max_message_bytes: &'a WorkerCallbackMessageLimits,
}

/// Launches a worker with sanitized environment and explicit host-owned settings.
fn spawn_worker_process(spec: WorkerProcessLaunch<'_>) -> crate::Result<Child> {
    let manifest_dir = spec
        .manifest_path
        .parent()
        .unwrap_or_else(|| Path::new("."));
    let (mut command, command_display) = match spec.runtime {
        WorkerRuntime::Python => {
            let python = resolve_python_executable(spec.plugin_id, spec.environment_ref)?;
            if !python.is_file() {
                return Err(PluginError::RegistrationFailed(format!(
                    "configured Python worker environment interpreter '{}' does not exist",
                    python.display()
                )));
            }
            let mut command = Command::new(python);
            clear_host_python_environment(&mut command);
            command
                .arg("-c")
                .arg(PYTHON_WORKER_BOOTSTRAP)
                .arg(spec.entrypoint);
            (command, spec.entrypoint.to_string())
        }
        WorkerRuntime::Rust | WorkerRuntime::Command => {
            let entrypoint = resolve_manifest_relative_path(spec.manifest_path, spec.entrypoint);
            let command_display = entrypoint.display().to_string();
            (Command::new(entrypoint), command_display)
        }
    };
    minimize_worker_environment(&mut command, spec.forward_env);
    command
        .current_dir(manifest_dir)
        .env_remove("NEMO_RELAY_WORKER_ENDPOINT_FILE")
        .env("NEMO_RELAY_WORKER_ID", spec.activation_id)
        .env("NEMO_RELAY_PLUGIN_ID", spec.plugin_id)
        .env("NEMO_RELAY_WORKER_SOCKET", spec.worker_endpoint)
        .env("NEMO_RELAY_HOST_SOCKET", spec.host_endpoint)
        .env("NEMO_RELAY_WORKER_TOKEN", spec.auth_token)
        .env(
            WORKER_MAX_MESSAGE_BYTES_ENV,
            spec.max_message_bytes.default_bytes.to_string(),
        )
        .env(
            WORKER_MAX_MESSAGE_BYTES_BY_NAME_ENV,
            serde_json::to_string(&spec.max_message_bytes.by_name)
                .map_err(|e| PluginError::Internal(e.to_string()))?,
        )
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit());
    if let Some(path) = spec.worker_endpoint_file {
        command.env("NEMO_RELAY_WORKER_ENDPOINT_FILE", path);
    }
    command.spawn().map_err(|err| {
        PluginError::RegistrationFailed(format!(
            "failed to spawn {} worker '{}': {err}",
            spec.runtime, command_display
        ))
    })
}

fn minimize_worker_environment(command: &mut Command, forward_env: &[String]) {
    const ALLOWLIST: &[&str] = &[
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "TMPDIR",
        "TEMP",
        "TMP",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
    ];
    command.env_clear();
    for (name, value) in std::env::vars_os() {
        let Some(name_str) = name.to_str() else {
            continue;
        };
        // Windows environment variable names are case-insensitive.
        #[cfg(windows)]
        let name_str = name_str.to_ascii_uppercase();
        let matches = |pattern: &str| {
            #[cfg(windows)]
            let pattern = pattern.to_ascii_uppercase();
            match pattern.strip_suffix('*') {
                Some(prefix) => name_str.starts_with(prefix),
                None => name_str == pattern,
            }
        };
        if ALLOWLIST.iter().any(|pattern| matches(pattern))
            || forward_env.iter().any(|pattern| matches(pattern))
        {
            command.env(name, value);
        }
    }
}

fn resolve_python_executable(
    plugin_id: &str,
    environment_ref: Option<&str>,
) -> crate::Result<PathBuf> {
    let environment_ref = environment_ref.ok_or_else(|| {
        PluginError::InvalidConfig(
            "Python worker activation requires a lifecycle-managed environment_ref; run `nemo-relay plugins add <path>`"
                .into(),
        )
    })?;
    let environment = Path::new(environment_ref);
    let expected_name = Sha256::digest(plugin_id.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let is_managed_path = environment.is_absolute()
        && environment
            .file_name()
            .is_some_and(|name| name == std::ffi::OsStr::new(&expected_name))
        && environment
            .parent()
            .and_then(Path::file_name)
            .is_some_and(|name| name == MANAGED_ENVIRONMENTS_DIR);
    if !is_managed_path {
        return Err(PluginError::InvalidConfig(format!(
            "Python worker environment_ref '{}' is not the lifecycle-managed path for plugin '{plugin_id}'",
            environment.display()
        )));
    }
    if std::fs::symlink_metadata(environment)
        .is_ok_and(|metadata| metadata.file_type().is_symlink())
    {
        return Err(PluginError::InvalidConfig(format!(
            "Python worker environment_ref '{}' must not be a symbolic link",
            environment.display()
        )));
    }
    Ok(if cfg!(windows) {
        environment.join("Scripts").join("python.exe")
    } else {
        environment.join("bin").join("python")
    })
}

fn clear_host_python_environment(command: &mut Command) {
    for key in ["PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"] {
        command.env_remove(key);
    }
}

impl WorkerPluginInstance {
    // Register needs the preceding components' live runtime registrations.
    // Cache its plan once per worker so multiple components do not repeat setup.
    async fn prepare_registrations(&self) -> crate::Result<()> {
        self.registrations
            .get_or_try_init(|| async {
                let result = self.request_registration_plan().await;
                match &result {
                    Ok(_) => self.host_state.observer.success(
                        nemo_relay_diagnostics::SuccessCode::PluginWorkerRegistered,
                        DiagnosticPhase::Activation,
                    ),
                    Err(_) => self.host_state.observer.failure(
                        DiagnosticOperation::WorkerRegistration,
                        DiagnosticFailureClass::Unknown,
                        DiagnosticPhase::Activation,
                    ),
                }
                result
            })
            .await?;
        Ok(())
    }

    async fn request_registration_plan(&self) -> crate::Result<Vec<Registration>> {
        let mut client =
            message_limited_client(&self.client, &self.host_state.message_limits, None);
        let register = worker_rpc(client.register(worker_rpc_request(RegisterRequest {
            credentials: Some(wire::Credentials {
                activation_id: self.host_state.activation_id.clone(),
                auth_token: self.host_state.auth_token.clone(),
            }),
            plugin_id: self.plugin_kind.clone(),

            config: Some(encode_json(JSON_SCHEMA, &Json::Object(self.config.clone()))),
        })))
        .await
        .map_err(|err| {
            PluginError::RegistrationFailed(format!("worker registration RPC failed: {err}"))
        })?
        .into_inner();
        nemo_relay_worker_proto::validation::validate(&register)
            .map_err(PluginError::RegistrationFailed)?;
        let register = match register.result {
            Some(wire::register_response::Result::Plan(v)) => v,
            Some(wire::register_response::Result::Error(e)) => {
                return Err(worker_error_to_plugin(e, "worker registration failed"));
            }
            None => {
                return Err(PluginError::RegistrationFailed(
                    "missing registration plan".into(),
                ));
            }
        };
        validate_registration_plan(&self.plugin_kind, &register)?;
        let registrations = register.registrations;
        if registrations.iter().any(|registration| {
            RegistrationSurface::try_from(registration.surface)
                .is_ok_and(|surface| surface == RegistrationSurface::LlmRequestIntercept)
        }) {
            validate_annotated_request_consumer_compatibility(
                &self.relay_compat,
                &self.plugin_kind,
            )?;
        }
        if registrations.iter().any(|registration| {
            RegistrationSurface::try_from(registration.surface)
                .is_ok_and(|surface| surface == RegistrationSurface::ToolExecutionIntercept)
        }) {
            validate_tool_execution_context_compatibility(&self.relay_compat, &self.plugin_kind)?;
        }
        if registrations.iter().any(|registration| {
            RegistrationSurface::try_from(registration.surface).is_ok_and(|surface| {
                matches!(
                    surface,
                    RegistrationSurface::LlmExecutionIntercept
                        | RegistrationSurface::LlmStreamExecutionIntercept
                )
            })
        }) {
            validate_llm_execution_context_compatibility(&self.relay_compat, &self.plugin_kind)?;
        }
        let initial_gates = register.conditional_middleware_guardrails;
        for gate in initial_gates {
            let kinds = gate
                .kinds
                .into_iter()
                .map(|kind| {
                    RegistrationSurface::try_from(kind)
                        .map_err(|_| {
                            PluginError::RegistrationFailed(format!(
                                "worker plugin '{}' returned an unknown conditional middleware guardrail kind",
                                self.plugin_kind
                            ))
                        })
                        .and_then(|surface| {
                            runtime_registration_kind_from_surface(surface).map_err(|status| {
                                PluginError::RegistrationFailed(status.message().to_string())
                            })
                        })
                })
                .collect::<crate::Result<BTreeSet<_>>>()?;
            if let Err(error) = self
                .host_state
                .register_owned_conditional_middleware_guardrail(
                    gate.name,
                    kinds,
                    gate.registration_name,
                    gate.reason,
                    gate.callback,
                )
            {
                self.host_state.cleanup_conditional_middleware_guardrails();
                return Err(PluginError::RegistrationFailed(format!(
                    "worker initial conditional middleware guardrail failed: {error}"
                )));
            }
        }

        log::info!(
            target: "nemo_relay.worker",
            event = "worker_connected",
            plugin_id = self.plugin_kind.as_str();
            "Worker plugin connected and registered"
        );
        Ok(registrations)
    }

    fn install_registrations(
        &self,
        ctx: &mut dyn crate::PluginRegistrationContext,
    ) -> crate::Result<()> {
        for registration in self.registrations.get().ok_or_else(|| {
            PluginError::Internal("worker registration plan is not prepared".into())
        })? {
            let surface = RegistrationSurface::try_from(registration.surface).map_err(|_| {
                PluginError::RegistrationFailed(format!(
                    "worker plugin '{}' returned unsupported registration surface {}",
                    self.plugin_kind, registration.surface
                ))
            })?;
            match surface {
                RegistrationSurface::Subscriber => {
                    self.install_subscriber_registration(ctx, &registration.local_name)?
                }
                RegistrationSurface::EventMetadataInjector => self
                    .install_event_metadata_injector_registration(
                        ctx,
                        &registration.local_name,
                        registration.priority,
                    )?,
                RegistrationSurface::MarkSanitizeGuardrail
                | RegistrationSurface::ScopeSanitizeStartGuardrail
                | RegistrationSurface::ScopeSanitizeEndGuardrail => self
                    .install_event_sanitize_registration(
                        ctx,
                        &registration.local_name,
                        registration.priority,
                        surface,
                    )?,
                RegistrationSurface::ToolSanitizeRequestGuardrail
                | RegistrationSurface::ToolSanitizeResponseGuardrail
                | RegistrationSurface::ToolConditionalExecutionGuardrail
                | RegistrationSurface::ToolRequestIntercept
                | RegistrationSurface::ToolExecutionIntercept => {
                    self.install_tool_registration(ctx, registration, surface)?
                }
                RegistrationSurface::LlmSanitizeRequestGuardrail
                | RegistrationSurface::LlmSanitizeResponseGuardrail
                | RegistrationSurface::LlmConditionalExecutionGuardrail
                | RegistrationSurface::LlmRequestIntercept
                | RegistrationSurface::LlmExecutionIntercept
                | RegistrationSurface::LlmStreamExecutionIntercept => {
                    self.install_llm_registration(ctx, registration, surface)?
                }
                RegistrationSurface::ConditionalMiddlewareGuardrail
                | RegistrationSurface::Unspecified => {
                    return Err(PluginError::RegistrationFailed(format!(
                        "worker plugin '{}' returned unspecified registration surface",
                        self.plugin_kind
                    )));
                }
            }
        }
        Ok(())
    }

    fn install_subscriber_registration(
        &self,
        ctx: &mut dyn crate::PluginRegistrationContext,
        name: &str,
    ) -> crate::Result<()> {
        let instance = Arc::new(self.clone_for_callback());
        let callback_name = name.to_owned();
        ctx.register_subscriber(
            name,
            Arc::new(move |event| {
                if instance.invoke_subscriber(&callback_name, event).is_err() {
                    instance.log_callback_fallback(&callback_name, RegistrationSurface::Subscriber);
                }
            }),
        )
    }

    fn install_event_metadata_injector_registration(
        &self,
        ctx: &mut dyn crate::PluginRegistrationContext,
        name: &str,
        priority: i32,
    ) -> crate::Result<()> {
        let instance = Arc::new(self.clone_for_callback());
        let callback_name = name.to_owned();
        let callback: EventMetadataInjectorFn = Arc::new(move |event| {
            let instance = instance.clone();
            let callback_name = callback_name.clone();
            Box::pin(async move {
                instance
                    .invoke_event_metadata_injector(&callback_name, &event)
                    .await
            })
        });
        ctx.register_event_metadata_injector(name, priority, callback)
    }

    fn install_event_sanitize_registration(
        &self,
        ctx: &mut dyn crate::PluginRegistrationContext,
        name: &str,
        priority: i32,
        surface: RegistrationSurface,
    ) -> crate::Result<()> {
        let instance = Arc::new(self.clone_for_callback());
        let callback_name = name.to_owned();
        let callback: EventSanitizeFn =
            Arc::new(move |event: Arc<Event>, _fields: EventSanitizeFields| {
                let instance = instance.clone();
                let callback_name = callback_name.clone();
                Box::pin(async move {
                    instance
                        .invoke_event_sanitize(&callback_name, surface, &event)
                        .await
                })
            });
        match surface {
            RegistrationSurface::MarkSanitizeGuardrail => {
                ctx.register_mark_sanitize_guardrail(name, priority, callback)
            }
            RegistrationSurface::ScopeSanitizeStartGuardrail => {
                ctx.register_scope_sanitize_start_guardrail(name, priority, callback)
            }
            RegistrationSurface::ScopeSanitizeEndGuardrail => {
                ctx.register_scope_sanitize_end_guardrail(name, priority, callback)
            }
            _ => unreachable!("event sanitizer surface was pre-filtered"),
        }
    }

    fn install_tool_registration(
        &self,
        ctx: &mut dyn crate::PluginRegistrationContext,
        registration: &Registration,
        surface: RegistrationSurface,
    ) -> crate::Result<()> {
        let name = registration.local_name.as_str();
        let priority = registration.priority;
        let instance = Arc::new(self.clone_for_callback());
        let callback_name = name.to_owned();
        match surface {
            RegistrationSurface::ToolSanitizeRequestGuardrail => ctx
                .register_tool_sanitize_request_guardrail(
                    name,
                    priority,
                    Arc::new(move |tool_name, value| {
                        let instance = instance.clone();
                        let callback_name = callback_name.clone();
                        Box::pin(async move {
                            instance
                                .invoke_tool_json(&callback_name, surface, &tool_name, value, None)
                                .await
                        })
                    }),
                ),
            RegistrationSurface::ToolSanitizeResponseGuardrail => ctx
                .register_tool_sanitize_response_guardrail(
                    name,
                    priority,
                    Arc::new(move |tool_name, value| {
                        let instance = instance.clone();
                        let callback_name = callback_name.clone();
                        Box::pin(async move {
                            instance
                                .invoke_tool_json(&callback_name, surface, &tool_name, value, None)
                                .await
                        })
                    }),
                ),
            RegistrationSurface::ToolConditionalExecutionGuardrail => ctx
                .register_tool_conditional_execution_guardrail(
                    name,
                    priority,
                    Arc::new(move |tool_name, value| {
                        let instance = instance.clone();
                        let callback_name = callback_name.clone();
                        Box::pin(async move {
                            instance
                                .invoke_tool_guardrail(&callback_name, &tool_name, value)
                                .await
                        })
                    }),
                ),
            RegistrationSurface::ToolRequestIntercept => ctx.register_tool_request_intercept(
                name,
                priority,
                registration.break_chain,
                Arc::new(move |tool_name, value| {
                    let instance = instance.clone();
                    let callback_name = callback_name.clone();
                    Box::pin(async move {
                        instance
                            .invoke_tool_json(&callback_name, surface, &tool_name, value, None)
                            .await
                    })
                }),
            ),
            RegistrationSurface::ToolExecutionIntercept => ctx.register_tool_execution_intercept(
                name,
                priority,
                Arc::new(move |context: ToolExecutionContext, next| {
                    let instance = instance.clone();
                    let callback_name = callback_name.clone();
                    let tool_name = context.tool_name().to_owned();
                    let tool_call_id = context.tool_call_id().map(str::to_owned);
                    let value = context.into_args();
                    Box::pin(async move {
                        instance
                            .invoke_tool_execution(
                                &callback_name,
                                &tool_name,
                                value,
                                tool_call_id.as_deref(),
                                next,
                            )
                            .await
                    })
                }),
            ),
            _ => Err(PluginError::RegistrationFailed(format!(
                "worker plugin '{}' cannot install registration surface {} as a tool callback",
                self.plugin_kind,
                surface.as_str_name()
            ))),
        }
    }

    fn install_llm_registration(
        &self,
        ctx: &mut dyn crate::PluginRegistrationContext,
        registration: &Registration,
        surface: RegistrationSurface,
    ) -> crate::Result<()> {
        let name = registration.local_name.as_str();
        let priority = registration.priority;
        let instance = Arc::new(self.clone_for_callback());
        let callback_name = name.to_owned();
        match surface {
            RegistrationSurface::LlmSanitizeRequestGuardrail => ctx
                .register_llm_sanitize_request_guardrail(
                    name,
                    priority,
                    Arc::new(move |request, context| {
                        let instance = instance.clone();
                        let callback_name = callback_name.clone();
                        Box::pin(async move {
                            instance
                                .invoke_llm_sanitize_request(&callback_name, request, context)
                                .await
                        })
                    }),
                ),
            RegistrationSurface::LlmSanitizeResponseGuardrail => ctx
                .register_llm_sanitize_response_guardrail(
                    name,
                    priority,
                    Arc::new(move |value, context| {
                        let instance = instance.clone();
                        let callback_name = callback_name.clone();
                        Box::pin(async move {
                            instance
                                .invoke_llm_sanitize_response(&callback_name, value, context)
                                .await
                        })
                    }),
                ),
            RegistrationSurface::LlmConditionalExecutionGuardrail => ctx
                .register_llm_conditional_execution_guardrail(
                    name,
                    priority,
                    Arc::new(move |request| {
                        let instance = instance.clone();
                        let callback_name = callback_name.clone();
                        Box::pin(async move {
                            instance.invoke_llm_guardrail(&callback_name, request).await
                        })
                    }),
                ),
            RegistrationSurface::LlmRequestIntercept => ctx.register_llm_request_intercept(
                name,
                priority,
                registration.break_chain,
                Arc::new(move |model_name, request, annotated| {
                    let instance = instance.clone();
                    let callback_name = callback_name.clone();
                    Box::pin(async move {
                        instance
                            .invoke_llm_request_intercept(
                                &callback_name,
                                &model_name,
                                request,
                                annotated,
                            )
                            .await
                    })
                }),
            ),
            RegistrationSurface::LlmExecutionIntercept => ctx.register_llm_execution_intercept(
                name,
                priority,
                Arc::new(move |model_name, request, context, next| {
                    let instance = instance.clone();
                    let callback_name = callback_name.clone();
                    let model_name = model_name.to_owned();
                    Box::pin(async move {
                        instance
                            .invoke_llm_execution(
                                &callback_name,
                                &model_name,
                                request,
                                context,
                                next,
                            )
                            .await
                    })
                }),
            ),
            RegistrationSurface::LlmStreamExecutionIntercept => ctx
                .register_llm_stream_execution_intercept(
                    name,
                    priority,
                    Arc::new(move |model_name, request, context, next| {
                        let instance = instance.clone();
                        let callback_name = callback_name.clone();
                        let model_name = model_name.to_owned();
                        Box::pin(async move {
                            instance
                                .invoke_llm_stream_execution(
                                    &callback_name,
                                    &model_name,
                                    request,
                                    context,
                                    next,
                                )
                                .await
                        })
                    }),
                ),
            _ => Err(PluginError::RegistrationFailed(format!(
                "worker plugin '{}' cannot install registration surface {} as an LLM callback",
                self.plugin_kind,
                surface.as_str_name()
            ))),
        }
    }

    /// Shares the worker connection and copies its callback deadline settings.
    fn clone_for_callback(&self) -> WorkerPluginCallback {
        WorkerPluginCallback {
            callback_timeouts: self.callback_timeouts.clone(),
            plugin_kind: self.plugin_kind.clone(),
            activation_id: self.host_state.activation_id.clone(),
            runtime: self.runtime.handle(),
            client: self.client.clone(),
            host_state: self.host_state.clone(),
        }
    }
}

#[cfg(not(unix))]
fn bind_loopback_listener() -> crate::Result<(TcpListener, SocketAddr)> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).map_err(|err| {
        PluginError::RegistrationFailed(format!(
            "failed to bind worker host runtime endpoint: {err}"
        ))
    })?;
    listener.set_nonblocking(true).map_err(|err| {
        PluginError::RegistrationFailed(format!(
            "failed to configure worker host runtime endpoint: {err}"
        ))
    })?;
    let addr = listener.local_addr().map_err(|err| {
        PluginError::RegistrationFailed(format!(
            "failed to inspect worker host runtime endpoint: {err}"
        ))
    })?;
    Ok((listener, addr))
}

#[derive(Clone)]
struct WorkerPluginCallback {
    callback_timeouts: super::WorkerCallbackTimeouts,
    plugin_kind: String,
    activation_id: String,
    runtime: tokio::runtime::Handle,
    client: PluginWorkerClient<Channel>,
    host_state: Arc<WorkerHostRuntimeState>,
}

#[derive(Clone)]
struct WorkerGateCallback {
    callback_timeouts: super::WorkerCallbackTimeouts,
    plugin_kind: String,
    runtime: tokio::runtime::Handle,
    client: PluginWorkerClient<Channel>,
    host_state: std::sync::Weak<WorkerHostRuntimeState>,
}

impl WorkerGateCallback {
    /// Calls a worker-owned conditional guardrail while its activation is available.
    fn invoke(
        &self,
        callback_name: &str,
        kinds: &BTreeSet<RuntimeRegistrationKind>,
        registration_name: &str,
    ) -> FlowResult<Option<String>> {
        let host_state = self
            .host_state
            .upgrade()
            .ok_or_else(|| FlowError::NotFound("worker activation is shutting down".into()))?;
        WorkerPluginCallback {
            callback_timeouts: self.callback_timeouts.clone(),
            plugin_kind: self.plugin_kind.clone(),
            activation_id: host_state.activation_id.clone(),
            runtime: self.runtime.clone(),
            client: self.client.clone(),
            host_state,
        }
        .invoke_conditional_middleware(callback_name, kinds, registration_name)
    }
}

impl WorkerPluginCallback {
    fn log_callback_fallback(&self, callback_name: &str, surface: RegistrationSurface) {
        log::warn!(
            target: "nemo_relay.worker",
            event = "worker_callback_fallback",
            plugin_id = self.plugin_kind.as_str(),
            callback = callback_name,
            surface = surface.as_str_name();
            "Worker plugin callback failed; Relay used the safe fallback"
        );
    }
}

struct WorkerInvocationGuard {
    runtime: tokio::runtime::Handle,
    client: PluginWorkerClient<Channel>,
    host_state: Arc<WorkerHostRuntimeState>,
    activation_id: String,
    auth_token: String,
    invocation_id: String,
    continuation_id: String,
    scope_stack_id: String,
    scope_lifetime: Option<Arc<ScopeStackLifetime>>,
    cancel_on_drop: bool,
    cleaned: bool,
}

impl WorkerInvocationGuard {
    fn new(callback: &WorkerPluginCallback, request: &PreparedInvocation) -> Self {
        Self {
            runtime: callback.runtime.clone(),
            client: callback.client.clone(),
            host_state: callback.host_state.clone(),
            activation_id: request.activation_id.clone(),
            auth_token: request.auth_token.clone(),
            invocation_id: request.invocation_id.clone(),
            continuation_id: request.continuation_id.clone(),
            scope_stack_id: request
                .scope
                .as_ref()
                .map(|scope| scope.scope_stack_id.clone())
                .unwrap_or_default(),
            scope_lifetime: request.scope.as_ref().and_then(|scope| {
                callback
                    .host_state
                    .scope_stacks
                    .lock()
                    .ok()
                    .and_then(|stacks| {
                        stacks
                            .get(&scope.scope_stack_id)
                            .map(|stored| stored.lifetime.clone())
                    })
            }),
            cancel_on_drop: true,
            cleaned: false,
        }
    }

    fn cancellation(
        &mut self,
        reason: impl Into<String>,
    ) -> Option<(PluginWorkerClient<Channel>, CancelInvocationRequest)> {
        if !self.cancel_on_drop {
            return None;
        }
        self.cancel_on_drop = false;
        // Revoke inspection before awaiting worker cancellation.
        self.host_state
            .cancel_scope_stack_snapshot(&self.scope_stack_id);
        Some((
            message_limited_client(&self.client, &self.host_state.message_limits, None),
            CancelInvocationRequest {
                credentials: Some(wire::Credentials {
                    activation_id: self.activation_id.clone(),
                    auth_token: self.auth_token.clone(),
                }),
                invocation_id: self.invocation_id.clone(),
                reason: reason.into(),
            },
        ))
    }

    fn cancel(&mut self, reason: impl Into<String>) {
        if let Some((mut client, request)) = self.cancellation(reason) {
            let observer = self.host_state.observer.clone();
            self.runtime.spawn(async move {
                let _ = worker_rpc(client.cancel_invocation(worker_rpc_request(request)))
                    .await
                    .inspect_err(|error| {
                        observer.rpc_failure(
                            DiagnosticOperation::WorkerCancellation,
                            error,
                            DiagnosticPhase::Runtime,
                        );
                    });
            });
        }
    }

    async fn cancel_and_wait(&mut self, reason: impl Into<String>) -> FlowResult<()> {
        if let Some((mut client, request)) = self.cancellation(reason) {
            worker_rpc(client.cancel_invocation(worker_rpc_request(request)))
                .await
                .inspect_err(|error| {
                    self.host_state.observer.rpc_failure(
                        DiagnosticOperation::WorkerCancellation,
                        error,
                        DiagnosticPhase::Runtime,
                    );
                })
                .map_err(|error| {
                    FlowError::Internal(format!("worker stream cleanup failed: {error}"))
                })?;
        }
        Ok(())
    }

    fn finish(&mut self) {
        self.cancel_on_drop = false;
        self.cleanup();
    }

    async fn finish_and_wait(&mut self) -> FlowResult<()> {
        self.cancel_on_drop = false;
        match self.start_cleanup() {
            Some(result) => result.await.map_err(|_| {
                FlowError::Internal("worker continuation cleanup task interrupted".into())
            })?,
            None => Ok(()),
        }
    }

    fn cleanup(&mut self) {
        let _ = self.start_cleanup();
    }

    fn start_cleanup(&mut self) -> Option<oneshot::Receiver<FlowResult<()>>> {
        if self.cleaned {
            return None;
        }
        self.cleaned = true;
        let cleanup = self.host_state.revoke_continuation(&self.continuation_id);
        let state = self.host_state.clone();
        let scope = self.scope_stack_id.clone();
        let lifetime = self.scope_lifetime.clone();
        let runtime = self.runtime.clone();
        let (tx, rx) = oneshot::channel();
        if cleanup
            .as_ref()
            .is_some_and(|cleanup| cleanup.borrow().active > 0)
        {
            self.runtime.spawn(async move {
                let result = wait_for_continuation_cleanup(cleanup.expect("active cleanup")).await;
                state.revoke_scope_stack(&scope, &runtime);
                let scopes = wait_for_scope_cleanup(lifetime).await;
                if let Err(Err(error)) = tx.send(result.and(scopes)) {
                    log::warn!(target: "nemo_relay.worker", event = "worker_continuation_cleanup_failed", scope_stack_id = scope.as_str(); "Worker continuation cleanup failed: {error}");
                }
            });
        } else {
            let result = cleanup
                .and_then(|cleanup| cleanup.borrow().error.clone())
                .map_or(Ok(()), |error| Err(FlowError::Internal(error)));
            state.revoke_scope_stack(&scope, &runtime);
            if lifetime
                .as_ref()
                .is_some_and(|lifetime| lifetime.complete.borrow().is_none())
            {
                self.runtime.spawn(async move {
                    let scopes = wait_for_scope_cleanup(lifetime).await;
                    let _ = tx.send(result.and(scopes));
                });
            } else {
                let scopes = lifetime
                    .as_ref()
                    .and_then(|lifetime| lifetime.complete.borrow().clone())
                    .unwrap_or(Ok(()))
                    .map_err(FlowError::Internal);
                let _ = tx.send(result.and(scopes));
            }
        }
        Some(rx)
    }
}

impl Drop for WorkerInvocationGuard {
    fn drop(&mut self) {
        self.cancel("host caller cancelled the worker invocation");
        self.cleanup();
    }
}

async fn wait_for_scope_cleanup(lifetime: Option<Arc<ScopeStackLifetime>>) -> FlowResult<()> {
    if let Some(lifetime) = lifetime {
        let mut complete = lifetime.complete.subscribe();
        let result = tokio::time::timeout(WORKER_RPC_TIMEOUT, complete.wait_for(Option::is_some))
            .await
            .map_err(|_| FlowError::Internal("worker scope cleanup timed out".into()))?
            .map_err(|_| FlowError::Internal("worker scope cleanup interrupted".into()))?
            .clone()
            .expect("scope completion checked above");
        result.map_err(FlowError::Internal)?;
    }
    Ok(())
}

type WorkerStreamCloseResult = std::result::Result<(), String>;

struct WorkerStreamCompletionSignal(watch::Sender<Option<WorkerStreamCloseResult>>);

impl Drop for WorkerStreamCompletionSignal {
    fn drop(&mut self) {
        self.0.send_if_modified(|outcome| {
            if outcome.is_some() {
                return false;
            }
            *outcome = Some(Err(
                "worker stream task ended before cleanup completed".into()
            ));
            true
        });
    }
}

struct WorkerForwardedLlmStream {
    receiver: Option<tokio_stream::wrappers::ReceiverStream<FlowResult<Json>>>,
    completion: watch::Receiver<Option<WorkerStreamCloseResult>>,
    close_result: Option<FlowResult<()>>,
}

impl Stream for WorkerForwardedLlmStream {
    type Item = FlowResult<Json>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        match this.receiver.as_mut() {
            Some(receiver) => Pin::new(receiver).poll_next(cx),
            None => Poll::Ready(None),
        }
    }
}

impl LlmStreamInner for WorkerForwardedLlmStream {
    fn terminalize(self: Pin<&mut Self>) {
        self.get_mut().receiver.take();
    }

    fn close(self: Pin<&mut Self>) -> Pin<Box<dyn Future<Output = FlowResult<()>> + Send + '_>> {
        let this = self.get_mut();
        // Disconnect before waiting so the forwarder requests worker cancellation.
        if let Some(receiver) = this.receiver.take() {
            let mut receiver = receiver.into_inner();
            receiver.close();
            while let Ok(item) = receiver.try_recv() {
                if let Err(error) = item {
                    this.close_result.get_or_insert(Err(error));
                }
            }
        }
        let mut completion = this.completion.clone();
        Box::pin(async move {
            let outcome = completion
                .wait_for(Option::is_some)
                .await
                .map_err(|_| {
                    FlowError::Internal("worker stream task ended before cleanup completed".into())
                })?
                .clone()
                .expect("completion checked above")
                .map_err(FlowError::Internal);
            if this.close_result.is_none() {
                this.close_result = Some(outcome);
            }
            this.close_result
                .clone()
                .expect("close result stored above")
        })
    }
}

impl WorkerPluginCallback {
    fn invoke_conditional_middleware(
        &self,
        callback_name: &str,
        kinds: &BTreeSet<RuntimeRegistrationKind>,
        registration_name: &str,
    ) -> FlowResult<Option<String>> {
        let request = self.base_request(
            callback_name,
            RegistrationSurface::ConditionalMiddlewareGuardrail,
            None,
            Some(invoke_request_payload::Payload::ConditionalMiddleware(
                ConditionalMiddlewareInvocation {
                    kinds: kinds
                        .iter()
                        .copied()
                        .map(registration_surface_from_kind)
                        .map(|surface| surface as i32)
                        .collect(),
                    registration_name: registration_name.into(),
                },
            )),
        )?;
        guardrail_from_invoke_response(self.invoke_blocking(request)?)
    }

    fn invoke_subscriber(&self, registration_name: &str, event: &Event) -> FlowResult<()> {
        let request = self.base_request(
            registration_name,
            RegistrationSurface::Subscriber,
            None,
            Some(invoke_request_payload_event(event)?),
        )?;
        let response = self.invoke_blocking(request)?;
        match response.result {
            Some(invoke_response_result::Result::Completed(_)) => Ok(()),
            Some(invoke_response_result::Result::Error(error)) => Err(worker_error_to_flow(error)),
            _ => Err(FlowError::Internal(
                "worker subscriber returned unexpected result".into(),
            )),
        }
    }

    async fn invoke_event_metadata_injector(
        &self,
        registration_name: &str,
        event: &Event,
    ) -> FlowResult<BTreeMap<String, Json>> {
        let request = self.base_request(
            registration_name,
            RegistrationSurface::EventMetadataInjector,
            None,
            Some(invoke_request_payload_event(event)?),
        )?;
        let value = json_from_invoke_response(self.invoke_async(request).await?)?;
        let additions = serde_json::from_value::<BTreeMap<String, Json>>(value).map_err(|err| {
            FlowError::Internal(format!(
                "worker returned invalid Event metadata additions: {err}"
            ))
        })?;
        Ok(additions)
    }

    async fn invoke_event_sanitize(
        &self,
        registration_name: &str,
        surface: RegistrationSurface,
        event: &Event,
    ) -> FlowResult<EventSanitizeFields> {
        let request = self.base_request(
            registration_name,
            surface,
            None,
            Some(invoke_request_payload_event(event)?),
        )?;
        match self.invoke_async(request).await?.result {
            Some(wire::invoke_response::Result::EventSanitize(v)) => decode_wire(&v),
            Some(wire::invoke_response::Result::Error(e)) => Err(worker_error_to_flow(e)),
            _ => Err(FlowError::InvalidArgument(
                "unexpected event sanitizer result".into(),
            )),
        }
    }

    async fn invoke_tool_json(
        &self,
        registration_name: &str,
        surface: RegistrationSurface,
        tool_name: &str,
        value: Json,
        continuation_id: Option<String>,
    ) -> FlowResult<Json> {
        let request = self.base_request(
            registration_name,
            surface,
            continuation_id,
            Some(invoke_request_payload_tool(tool_name, value, None)),
        )?;
        json_from_invoke_response(self.invoke_async(request).await?)
    }

    async fn invoke_tool_guardrail(
        &self,
        registration_name: &str,
        tool_name: &str,
        value: Json,
    ) -> FlowResult<Option<String>> {
        let request = self.base_request(
            registration_name,
            RegistrationSurface::ToolConditionalExecutionGuardrail,
            None,
            Some(invoke_request_payload_tool(tool_name, value, None)),
        )?;
        guardrail_from_invoke_response(self.invoke_async(request).await?)
    }

    async fn invoke_tool_execution(
        &self,
        registration_name: &str,
        tool_name: &str,
        value: Json,
        tool_call_id: Option<&str>,
        next: ToolExecutionNextFn,
    ) -> FlowResult<ToolExecutionInterceptOutcome> {
        let continuation_id = self
            .host_state
            .insert_continuation(Continuation::tool(next))?;
        let request = self.base_request(
            registration_name,
            RegistrationSurface::ToolExecutionIntercept,
            Some(continuation_id),
            Some(invoke_request_payload_tool(tool_name, value, tool_call_id)),
        )?;
        let response = self.invoke_async(request).await?;
        match response.result {
            Some(invoke_response_result::Result::ToolExecutionOutcome(outcome)) => {
                tool_execution_intercept_outcome_from_proto(outcome)
            }
            Some(invoke_response_result::Result::Error(error)) => Err(worker_error_to_flow(error)),
            _ => Err(FlowError::Internal(
                "worker tool execution intercept returned unexpected result".into(),
            )),
        }
    }

    async fn invoke_llm_sanitize_request(
        &self,
        registration_name: &str,
        request: LlmRequest,
        context: LlmSanitizeRequestContext,
    ) -> FlowResult<Option<LlmRequest>> {
        let mut invoke = self.base_request(
            registration_name,
            RegistrationSurface::LlmSanitizeRequestGuardrail,
            None,
            Some(invoke_request_payload_llm_context(
                "",
                Some(request),
                None,
                None,
                llm_invocation::SanitizeContext::RequestSanitizeContext(
                    ProtoLlmSanitizeRequestContext {
                        codec: Some(codec_identity_to_proto(context.codec())),
                        capability_id: None,
                    },
                ),
            )?),
        )?;
        let capability = context
            .resolve_codec()
            .map(|codec| -> FlowResult<WorkerCodecCapabilityGuard> {
                let capability = self
                    .host_state
                    .issue_request_codec(&invoke.invocation_id, codec)?;
                let Some(invoke_request_payload::Payload::Llm(llm)) = invoke.payload.as_mut()
                else {
                    unreachable!("LLM sanitizer invocation must have an LLM payload");
                };
                let Some(llm_invocation::SanitizeContext::RequestSanitizeContext(context)) =
                    llm.sanitize_context.as_mut()
                else {
                    unreachable!("request sanitizer invocation must have a request context");
                };
                context.capability_id = Some(capability.id().into());
                Ok(capability)
            })
            .transpose();
        let _capability = self.cleanup_after_setup_error(&invoke, capability)?;
        let response = self.invoke_async(invoke).await;
        optional_json_from_invoke_response(response?)?
            .map(serde_json::from_value)
            .transpose()
            .map_err(|err| {
                FlowError::Internal(format!("worker returned invalid LLM request: {err}"))
            })
    }

    async fn invoke_llm_sanitize_response(
        &self,
        registration_name: &str,
        response: Json,
        context: LlmSanitizeResponseContext,
    ) -> FlowResult<Option<Json>> {
        let mut invoke = self.base_request(
            registration_name,
            RegistrationSurface::LlmSanitizeResponseGuardrail,
            None,
            Some(invoke_request_payload_llm_context(
                "",
                None,
                None,
                Some(response),
                llm_invocation::SanitizeContext::ResponseSanitizeContext(
                    ProtoLlmSanitizeResponseContext {
                        codec: Some(codec_identity_to_proto(context.codec())),
                        capability_id: None,
                    },
                ),
            )?),
        )?;
        let capability = context
            .resolve_codec()
            .map(|codec| -> FlowResult<WorkerCodecCapabilityGuard> {
                let capability = self
                    .host_state
                    .issue_response_codec(&invoke.invocation_id, codec)?;
                let Some(invoke_request_payload::Payload::Llm(llm)) = invoke.payload.as_mut()
                else {
                    unreachable!("LLM sanitizer invocation must have an LLM payload");
                };
                let Some(llm_invocation::SanitizeContext::ResponseSanitizeContext(context)) =
                    llm.sanitize_context.as_mut()
                else {
                    unreachable!("response sanitizer invocation must have a response context");
                };
                context.capability_id = Some(capability.id().into());
                Ok(capability)
            })
            .transpose();
        let _capability = self.cleanup_after_setup_error(&invoke, capability)?;
        let response = self.invoke_async(invoke).await;
        optional_json_from_invoke_response(response?)
    }

    async fn invoke_llm_guardrail(
        &self,
        registration_name: &str,
        request: LlmRequest,
    ) -> FlowResult<Option<String>> {
        let invoke = self.base_request(
            registration_name,
            RegistrationSurface::LlmConditionalExecutionGuardrail,
            None,
            Some(invoke_request_payload_llm("", Some(request), None, None)?),
        )?;
        guardrail_from_invoke_response(self.invoke_async(invoke).await?)
    }

    async fn invoke_llm_request_intercept(
        &self,
        registration_name: &str,
        model_name: &str,
        request: LlmRequest,
        annotated: Option<AnnotatedLlmRequest>,
    ) -> FlowResult<nemo_relay_runtime::api::llm::LlmRequestInterceptOutcome> {
        let invoke = self.base_request(
            registration_name,
            RegistrationSurface::LlmRequestIntercept,
            None,
            Some(invoke_request_payload_llm(
                model_name,
                Some(request),
                annotated,
                None,
            )?),
        )?;
        let response = self.invoke_async(invoke).await?;
        match response.result {
            Some(invoke_response_result::Result::LlmRequestOutcome(outcome)) => {
                decode_wire(&outcome)
            }
            Some(invoke_response_result::Result::Error(error)) => Err(worker_error_to_flow(error)),
            _ => Err(FlowError::Internal(
                "worker LLM request intercept returned unexpected result".into(),
            )),
        }
    }

    async fn invoke_llm_execution(
        &self,
        registration_name: &str,
        model_name: &str,
        request: LlmRequest,
        execution_context: LlmExecutionContext,
        next: LlmExecutionNextFn,
    ) -> FlowResult<Json> {
        let continuation_id = self
            .host_state
            .insert_continuation(Continuation::llm(next))?;
        let mut invoke = self.base_request(
            registration_name,
            RegistrationSurface::LlmExecutionIntercept,
            Some(continuation_id),
            Some(invoke_request_payload_llm(
                model_name,
                Some(request),
                None,
                None,
            )?),
        )?;
        let codec_capabilities = self.attach_llm_execution_codec_context(
            &mut invoke,
            &execution_context,
            WorkerLlmExecutionMode::CompleteResponse,
        );
        let _codec_capabilities = self.cleanup_after_setup_error(&invoke, codec_capabilities)?;
        json_from_invoke_response(self.invoke_async(invoke).await?)
    }

    async fn invoke_llm_stream_execution(
        &self,
        registration_name: &str,
        model_name: &str,
        request: LlmRequest,
        execution_context: LlmExecutionContext,
        next: LlmStreamExecutionNextFn,
    ) -> FlowResult<LlmJsonStream> {
        let continuation_id = self
            .host_state
            .insert_continuation(Continuation::llm_stream(next))?;
        let mut invoke = self.base_request(
            registration_name,
            RegistrationSurface::LlmStreamExecutionIntercept,
            Some(continuation_id.clone()),
            Some(invoke_request_payload_llm(
                model_name,
                Some(request),
                None,
                None,
            )?),
        )?;
        let codec_capabilities = self.attach_llm_execution_codec_context(
            &mut invoke,
            &execution_context,
            WorkerLlmExecutionMode::Streaming,
        );
        let codec_capabilities = self.cleanup_after_setup_error(&invoke, codec_capabilities)?;
        let mut client = message_limited_client(
            &self.client,
            &self.host_state.message_limits,
            Some(registration_name),
        );
        let mut guard = WorkerInvocationGuard::new(self, &invoke);
        let (tx, rx) = mpsc::channel(16);
        let (next_ready_tx, next_ready_rx) = oneshot::channel();
        let (completion_tx, completion_rx) = watch::channel(None);
        let plugin_id = self.plugin_kind.clone();
        let observer = self.host_state.observer.clone();
        self.runtime.spawn(async move {
            let completion = WorkerStreamCompletionSignal(completion_tx);
            let outcome = async move {
                let _codec_capabilities = codec_capabilities;
                let result = tokio::select! {
                    result = client.invoke_stream(worker_stream_rpc_request(invoke)?) => result,
                    _ = tx.closed() => {
                        guard.cancel_and_wait("host stopped consuming the worker stream").await?;
                        guard.finish_and_wait().await?;
                        return Ok(());
                    }
                };
                let _ = next_ready_tx.send(());
                match result {
                    Ok(response) => {
                        let mut stream = response.into_inner();
                        loop {
                            let item = tokio::select! {
                                item = stream.next() => item,
                                _ = tx.closed() => {
                                    guard.cancel_and_wait("host stopped consuming the worker stream").await?;
                                    break;
                                }
                            };
                            let Some(item) = item else {
                                break;
                            };
                            let result = match item {
                                Ok(chunk) => {
                                    if let Some(stream_chunk_item::Item::Error(error)) = &chunk.item
                                        && is_worker_callback_failure(error, &guard.continuation_id)
                                    {
                                        observer.failure(DiagnosticOperation::WorkerCallback, DiagnosticFailureClass::Unknown, DiagnosticPhase::Runtime);
                                    }
                                    json_from_stream_chunk(chunk)
                                        .map_err(|error| super::http_error::reported_by(error, &plugin_id, "worker"))
                                }
                                Err(err) => {
                                    if !is_downstream_status(&err, &guard.continuation_id) {
                                        observer.rpc_failure(DiagnosticOperation::WorkerTransport, &err, DiagnosticPhase::Runtime);
                                    }
                                    Err(FlowError::Internal(format!("worker stream transport failed: {err}")))
                                }
                            };
                            let terminal = result.is_err();
                            if tx.send(result).await.is_err() {
                                guard.cancel_and_wait("host stopped consuming the worker stream").await?;
                                break;
                            }
                            if terminal {
                                // Error delivery can precede producer cleanup on the worker.
                                guard.cancel_and_wait("worker stream failed").await?;
                                break;
                            }
                        }
                    }
                    Err(err) => {
                        if !is_downstream_status(&err, &guard.continuation_id) {
                            observer.rpc_failure(DiagnosticOperation::WorkerTransport, &err, DiagnosticPhase::Runtime);
                        }
                        let reason = if err.code() == tonic::Code::DeadlineExceeded {
                            "worker stream invocation timed out"
                        } else {
                            "worker stream transport failed"
                        };
                        let _ = tx
                            .send(Err(worker_status_to_flow(
                                "worker stream invoke failed",
                                err,
                            )))
                            .await;
                        guard.cancel_and_wait(reason).await?;
                    }
                }
                guard.finish_and_wait().await?;
                Ok::<(), FlowError>(())
            }
            .await;
            completion.0.send_replace(Some(outcome.map_err(|error| error.to_string())));
        });
        next_ready_rx.await.map_err(|_| {
            FlowError::Internal(
                "worker stream invocation ended before the downstream stream opened".into(),
            )
        })?;
        Ok(LlmJsonStream::from_closeable(WorkerForwardedLlmStream {
            receiver: Some(tokio_stream::wrappers::ReceiverStream::new(rx)),
            completion: completion_rx,
            close_result: None,
        }))
    }

    fn attach_llm_execution_codec_context(
        &self,
        invoke: &mut PreparedInvocation,
        context: &LlmExecutionContext,
        mode: WorkerLlmExecutionMode,
    ) -> FlowResult<WorkerCodecCapabilityGuards> {
        match (mode, context.response_codec()) {
            (WorkerLlmExecutionMode::CompleteResponse, None) => {
                return Err(FlowError::InvalidArgument(
                    "complete-response execution requires a response codec context".into(),
                ));
            }
            (WorkerLlmExecutionMode::Streaming, Some(_)) => {
                return Err(FlowError::InvalidArgument(
                    "streaming execution cannot use a response codec context".into(),
                ));
            }
            _ => {}
        }

        let request_guard = context
            .request_codec()
            .resolve_codec()
            .map(|codec| {
                self.host_state
                    .issue_request_codec(&invoke.invocation_id, codec)
            })
            .transpose()?;
        let request = ProtoLlmSanitizeRequestContext {
            codec: Some(codec_identity_to_proto(context.request_codec().codec())),
            capability_id: request_guard.as_ref().map(|guard| guard.id().into()),
        };

        let (response, response_guard) = match context.response_codec() {
            Some(response_context) => {
                let guard = response_context
                    .resolve_codec()
                    .map(|codec| {
                        self.host_state
                            .issue_response_codec(&invoke.invocation_id, codec)
                    })
                    .transpose()?;
                let response = ProtoLlmSanitizeResponseContext {
                    codec: Some(codec_identity_to_proto(response_context.codec())),
                    capability_id: guard.as_ref().map(|guard| guard.id().into()),
                };
                (Some(response), guard)
            }
            None => (None, None),
        };

        let Some(invoke_request_payload::Payload::Llm(llm)) = invoke.payload.as_mut() else {
            unreachable!("LLM execution invocation must have an LLM payload");
        };
        llm.execution_codec_context = Some(Box::new(ProtoLlmExecutionCodecContext {
            request: Some(request),
            response,
        }));
        Ok(WorkerCodecCapabilityGuards {
            _request: request_guard,
            _response: response_guard,
        })
    }

    fn cleanup_after_setup_error<T>(
        &self,
        invoke: &PreparedInvocation,
        result: FlowResult<T>,
    ) -> FlowResult<T> {
        result.inspect_err(|_| {
            let mut guard = WorkerInvocationGuard::new(self, invoke);
            guard.finish();
        })
    }

    fn base_request(
        &self,
        registration_name: &str,
        surface: RegistrationSurface,
        continuation_id: Option<String>,
        payload: Option<invoke_request_payload::Payload>,
    ) -> FlowResult<PreparedInvocation> {
        let scope_stack_id = match self.host_state.insert_invocation_scope_stack(
            current_scope_stack(),
            capture_nested_publication_buffer(),
        ) {
            Ok(scope_stack_id) => scope_stack_id,
            Err(error) => {
                if let Some(continuation_id) = continuation_id.as_deref() {
                    self.host_state.remove_continuation(continuation_id);
                }
                return Err(error);
            }
        };
        Ok(PreparedInvocation {
            activation_id: self.activation_id.clone(),
            auth_token: self.host_state.auth_token.clone(),
            invocation_id: Uuid::now_v7().to_string(),
            registration_name: registration_name.into(),
            surface: surface as i32,
            continuation_id: continuation_id.unwrap_or_default(),
            scope: Some(ScopeContext {
                scope_stack_id,
                parent_scope_id: String::new(),
            }),
            payload,
        })
    }

    fn invoke_blocking(&self, request: PreparedInvocation) -> FlowResult<InvokeResponse> {
        block_on_handle(&self.runtime, self.invoke_async(request))
    }

    /// Runs a callback with its configured deadline unless the caller controls cancellation.
    async fn invoke_async(&self, request: PreparedInvocation) -> FlowResult<InvokeResponse> {
        let callback_name = request.registration_name.clone();
        let surface = request.surface;
        // Execution intercepts may await long-running tools or model calls through
        // next(). Their caller owns cancellation; short callbacks retain an RPC deadline.
        let timeout = match RegistrationSurface::try_from(surface) {
            Ok(
                RegistrationSurface::ToolExecutionIntercept
                | RegistrationSurface::LlmExecutionIntercept,
            ) => None,
            _ => Some(
                self.callback_timeouts
                    .by_name
                    .get(&callback_name)
                    .copied()
                    .or(self.callback_timeouts.default_ms)
                    .map(|millis| Duration::from_millis(millis.get()))
                    .unwrap_or(WORKER_RPC_TIMEOUT),
            ),
        };
        let result = self.invoke_async_with_timeout(request, timeout).await;
        if let Err(error) = &result {
            let surface_name = RegistrationSurface::try_from(surface)
                .map(|surface| surface.as_str_name())
                .unwrap_or("UNKNOWN");
            log::warn!(
                target: "nemo_relay.worker",
                event = "worker_callback_failed",
                plugin_id = self.plugin_kind.as_str(),
                callback = callback_name.as_str(),
                surface = surface_name;
                "Worker plugin callback failed: {error}"
            );
        }
        result
    }

    /// Invokes a worker callback with an optional deadline.
    /// On timeout, requests cancellation and reports the surface and callback name.
    async fn invoke_async_with_timeout(
        &self,
        request: PreparedInvocation,
        timeout: Option<Duration>,
    ) -> FlowResult<InvokeResponse> {
        let mut guard = WorkerInvocationGuard::new(self, &request);
        let callback_name = request.registration_name.clone();
        let surface = RegistrationSurface::try_from(request.surface).ok();
        let surface_name = RegistrationSurface::try_from(request.surface)
            .map(|surface| surface.as_str_name())
            .unwrap_or("UNKNOWN");
        let mut client = message_limited_client(
            &self.client,
            &self.host_state.message_limits,
            Some(&callback_name),
        );
        let response = client.invoke(worker_callback_rpc_request(request)?);
        let result = match timeout {
            Some(timeout) => worker_rpc_with_timeout(timeout, response).await,
            None => response.await,
        };
        if result
            .as_ref()
            .is_err_and(|err| err.code() == tonic::Code::DeadlineExceeded)
        {
            guard.cancel("worker invocation timed out");
        }
        guard.finish_and_wait().await?;
        let result = result.and_then(|response| {
            nemo_relay_worker_proto::validation::validate(response.get_ref())
                .map_err(Status::data_loss)?;
            validate_callback_result(surface, response.get_ref())?;
            Ok(response)
        });
        result
            .map(|response| {
                let response = response.into_inner();
                let failed = match &response.result {
                    Some(invoke_response_result::Result::Error(error)) => is_worker_callback_failure(error, &guard.continuation_id),
                    _ => false,
                };
                if failed { self.host_state.observer.failure(DiagnosticOperation::WorkerCallback, DiagnosticFailureClass::Unknown, DiagnosticPhase::Runtime); }
                report_worker_response(response, &self.plugin_kind)
            })
            .map_err(|err| {
                self.host_state.observer.rpc_failure(DiagnosticOperation::WorkerTransport, &err, DiagnosticPhase::Runtime);
                if err.code() == tonic::Code::DeadlineExceeded {
                    FlowError::Internal(format!(
                        "worker invocation timed out (surface={surface_name}, callback={callback_name}): {err}"
                    ))
                } else {
                    worker_status_to_flow("worker invoke failed", err)
                }
            })
    }
}

struct OwnedWorkerRuntime {
    runtime: Option<Runtime>,
}

impl OwnedWorkerRuntime {
    fn new(runtime: Runtime) -> Self {
        Self {
            runtime: Some(runtime),
        }
    }

    fn runtime(&self) -> &Runtime {
        self.runtime
            .as_ref()
            .expect("worker runtime accessed after drop")
    }

    fn handle(&self) -> tokio::runtime::Handle {
        self.runtime().handle().clone()
    }
}

impl Drop for OwnedWorkerRuntime {
    fn drop(&mut self) {
        let Some(runtime) = self.runtime.take() else {
            return;
        };
        if tokio::runtime::Handle::try_current().is_ok() {
            std::thread::scope(|scope| {
                scope
                    .spawn(move || drop(runtime))
                    .join()
                    .expect("worker runtime drop thread panicked");
            });
        } else {
            drop(runtime);
        }
    }
}

struct ChildGuard {
    child: Option<Child>,
}

impl ChildGuard {
    fn new(child: Child) -> Self {
        Self { child: Some(child) }
    }

    fn take(&mut self) -> Child {
        self.child.take().expect("worker child already taken")
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

struct ActivationDirGuard {
    path: Option<PathBuf>,
}

impl ActivationDirGuard {
    fn new(path: PathBuf) -> Self {
        Self { path: Some(path) }
    }

    fn keep(&mut self) -> PathBuf {
        self.path
            .take()
            .expect("worker activation directory already taken")
    }
}

impl Drop for ActivationDirGuard {
    fn drop(&mut self) {
        if let Some(path) = self.path.take() {
            let _ = std::fs::remove_dir_all(path);
        }
    }
}

fn worker_rpc_request<T>(message: T) -> Request<T> {
    Request::new(message)
}

fn worker_callback_rpc_request(
    message: PreparedInvocation,
) -> FlowResult<Request<wire::InvokeRequest>> {
    let name = message.registration_name.clone();
    let message = message.into_wire()?;
    nemo_relay_worker_proto::validation::validate(&message).map_err(FlowError::InvalidArgument)?;
    Ok(nemo_relay_worker_proto::callback_request(message, &name))
}

async fn worker_rpc<T, F>(future: F) -> Result<Response<T>, Status>
where
    F: Future<Output = Result<Response<T>, Status>>,
{
    worker_rpc_with_timeout(WORKER_RPC_TIMEOUT, future).await
}

async fn worker_rpc_with_timeout<T, F>(timeout: Duration, future: F) -> Result<Response<T>, Status>
where
    F: Future<Output = Result<Response<T>, Status>>,
{
    match tokio::time::timeout(timeout, future).await {
        Ok(result) => result,
        Err(_) => Err(Status::deadline_exceeded(format!(
            "worker RPC timed out after {}ms",
            timeout.as_millis()
        ))),
    }
}

fn block_on_runtime<F>(runtime: &Runtime, future: F) -> F::Output
where
    F: Future + Send,
    F::Output: Send,
{
    if tokio::runtime::Handle::try_current().is_ok() {
        std::thread::scope(|scope| {
            scope
                .spawn(|| runtime.block_on(future))
                .join()
                .expect("worker runtime blocking thread panicked")
        })
    } else {
        runtime.block_on(future)
    }
}

fn block_on_handle<F>(handle: &tokio::runtime::Handle, future: F) -> F::Output
where
    F: Future + Send,
    F::Output: Send,
{
    if tokio::runtime::Handle::try_current().is_ok() {
        let handle = handle.clone();
        std::thread::scope(|scope| {
            scope
                .spawn(move || handle.block_on(future))
                .join()
                .expect("worker callback blocking thread panicked")
        })
    } else {
        handle.block_on(future)
    }
}

#[path = "worker_diagnostic_transport.rs"]
mod diagnostic_limits;
#[path = "worker_diagnostics.rs"]
mod diagnostic_transport;

fn message_limited_client(
    client: &PluginWorkerClient<Channel>,
    limits: &WorkerCallbackMessageLimits,
    callback: Option<&str>,
) -> PluginWorkerClient<Channel> {
    let limit = limits.for_callback(callback.unwrap_or_default());
    client
        .clone()
        .max_decoding_message_size(limit)
        .max_encoding_message_size(limit)
}

struct WorkerHostRuntimeState {
    message_limits: WorkerCallbackMessageLimits,
    observer: super::diagnostics::PluginDiagnostics,
    diagnostics: Option<Arc<diagnostic_transport::WorkerDiagnostics>>,
    activation_id: String,
    auth_token: String,
    scope_stacks: Mutex<HashMap<String, StoredScopeStack>>,
    scope_accepting: AtomicBool,
    scope_activity: watch::Sender<usize>,
    scope_cleanup_error: Mutex<Option<String>>,
    snapshot_cancelled_bindings: Mutex<std::collections::HashSet<String>>,
    pending_scope_cleanups: Mutex<Vec<PendingScopeCleanup>>,
    scope_stack_cleanups: Mutex<Vec<nemo_relay_runtime::api::runtime::ScopeStackHandle>>,
    scope_stack_cleanup_complete: Condvar,
    scope_handles: Mutex<HashMap<String, StoredScopeHandle>>,
    continuations: Mutex<HashMap<String, StoredContinuation>>,
    codecs: Mutex<HashMap<String, WorkerCodecCapability>>,
    gates_active: AtomicBool,
    conditional_middleware_guardrails: Mutex<HashMap<String, WorkerOwnedGate>>,
    gate_callback: Mutex<Option<WorkerGateCallback>>,
}

struct WorkerOwnedGate {
    local_name: String,
    qualified_name: String,
}

struct WorkerCodecCapability {
    invocation_id: String,
    direction: WorkerCodecDirection,
}

struct WorkerCodecCapabilityGuard {
    host_state: Arc<WorkerHostRuntimeState>,
    capability_id: String,
}

struct WorkerCodecCapabilityGuards {
    _request: Option<WorkerCodecCapabilityGuard>,
    _response: Option<WorkerCodecCapabilityGuard>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum WorkerLlmExecutionMode {
    CompleteResponse,
    Streaming,
}

impl WorkerCodecCapabilityGuard {
    fn id(&self) -> &str {
        &self.capability_id
    }
}

impl Drop for WorkerCodecCapabilityGuard {
    fn drop(&mut self) {
        self.host_state.remove_codec(&self.capability_id);
    }
}

enum WorkerCodecDirection {
    Request(Arc<dyn LlmCodec>),
    Response(Arc<dyn LlmResponseCodec>),
}

struct ScopeActivityGuard(watch::Sender<usize>);
impl ScopeActivityGuard {
    fn new(activity: &watch::Sender<usize>) -> Self {
        activity.send_modify(|active| *active += 1);
        Self(activity.clone())
    }
}
impl Drop for ScopeActivityGuard {
    fn drop(&mut self) {
        self.0.send_modify(|active| *active -= 1);
    }
}

type ScopeCleanupResult = std::result::Result<(), String>;

struct ScopeStackLifetime {
    revoked: AtomicBool,
    pushes: watch::Sender<usize>,
    complete: watch::Sender<Option<ScopeCleanupResult>>,
}
impl Default for ScopeStackLifetime {
    fn default() -> Self {
        Self {
            revoked: AtomicBool::new(false),
            pushes: watch::channel(0).0,
            complete: watch::channel(None).0,
        }
    }
}

struct ScopePushAdmission {
    binding: Option<Arc<ScopeStackLifetime>>,
    _binding_activity: Option<ScopeActivityGuard>,
    _activation_activity: ScopeActivityGuard,
}

struct StoredScopeStack {
    lifetime: Arc<ScopeStackLifetime>,
    handle: nemo_relay_runtime::api::runtime::ScopeStackHandle,
    publication_buffer: Option<PublicationBuffer>,
    continuation_context: Option<MiddlewareContinuationContext>,
    invocation_base_depth: Option<usize>,
}

struct ScopeUnwindPlan {
    handle: nemo_relay_runtime::api::runtime::ScopeStackHandle,
    base_depth: usize,
    continuation_context: Option<MiddlewareContinuationContext>,
    publication_buffer: Option<PublicationBuffer>,
}

struct PendingScopeCleanup {
    handle: nemo_relay_runtime::api::runtime::ScopeStackHandle,
    base_depth: usize,
}

#[derive(Clone)]
struct StoredScopeHandle {
    handle: ScopeHandle,
    scope_stack_id: String,
    context: StoredInvocationContext,
    pop_active: Arc<AtomicBool>,
}

struct ScopePopReservation(Arc<AtomicBool>);

impl Drop for ScopePopReservation {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

fn has_active_scope_stack_alias(
    stacks: &HashMap<String, StoredScopeStack>,
    handle: &nemo_relay_runtime::api::runtime::ScopeStackHandle,
) -> bool {
    stacks.values().any(|candidate| {
        candidate.invocation_base_depth.is_some() && Arc::ptr_eq(&candidate.handle, handle)
    })
}

struct ScopeStackCleanupGuard<'a> {
    state: &'a WorkerHostRuntimeState,
    handle: nemo_relay_runtime::api::runtime::ScopeStackHandle,
}

impl Drop for ScopeStackCleanupGuard<'_> {
    fn drop(&mut self) {
        if let Ok(mut cleanups) = self.state.scope_stack_cleanups.lock() {
            cleanups.retain(|handle| !Arc::ptr_eq(handle, &self.handle));
            self.state.scope_stack_cleanup_complete.notify_all();
        }
    }
}

impl WorkerHostRuntimeState {
    fn new(activation_id: String, auth_token: String) -> Self {
        Self {
            message_limits: WorkerCallbackMessageLimits::default(),
            diagnostics: None,
            observer: super::diagnostics::PluginDiagnostics::default(),
            activation_id,
            auth_token,
            scope_stacks: Mutex::new(HashMap::new()),
            scope_accepting: AtomicBool::new(true),
            scope_activity: watch::channel(0).0,
            scope_cleanup_error: Mutex::new(None),
            snapshot_cancelled_bindings: Mutex::new(std::collections::HashSet::new()),
            pending_scope_cleanups: Mutex::new(Vec::new()),
            scope_stack_cleanups: Mutex::new(Vec::new()),
            scope_stack_cleanup_complete: Condvar::new(),
            scope_handles: Mutex::new(HashMap::new()),
            continuations: Mutex::new(HashMap::new()),
            codecs: Mutex::new(HashMap::new()),
            gates_active: AtomicBool::new(true),
            conditional_middleware_guardrails: Mutex::new(HashMap::new()),
            gate_callback: Mutex::new(None),
        }
    }

    fn set_gate_callback(&self, callback: WorkerGateCallback) {
        *self
            .gate_callback
            .lock()
            .expect("worker gate callback lock") = Some(callback);
    }

    fn cleanup_conditional_middleware_guardrails(&self) {
        self.gates_active.store(false, Ordering::Release);
        let mut gates = match self.conditional_middleware_guardrails.lock() {
            Ok(gates) => gates,
            Err(error) => {
                log::error!(
                    target: "nemo_relay.worker",
                    event = "worker_gate_ownership_lock_poisoned";
                    "Worker gate ownership lock was poisoned during cleanup; recovering owned gates"
                );
                error.into_inner()
            }
        };
        let names = gates
            .drain()
            .map(|(_, gate)| gate.qualified_name)
            .collect::<Vec<_>>();
        drop(gates);
        for name in names {
            if let Err(error) = deregister_conditional_middleware_guardrail(&name) {
                log::error!(
                    target: "nemo_relay.worker",
                    event = "worker_gate_cleanup_failed",
                    gate = name.as_str();
                    "Worker-owned conditional middleware guardrail cleanup failed: {error}"
                );
            }
        }
    }

    fn register_owned_conditional_middleware_guardrail(
        &self,
        name: String,
        kinds: BTreeSet<RuntimeRegistrationKind>,
        registration_name: String,
        reason: String,
        callback: bool,
    ) -> FlowResult<String> {
        let mut owned_gates = self
            .conditional_middleware_guardrails
            .lock()
            .map_err(|error| {
                FlowError::Internal(format!("gate ownership lock poisoned: {error}"))
            })?;
        if !self.gates_active.load(Ordering::Acquire) {
            return Err(FlowError::NotFound(
                "worker activation is shutting down".into(),
            ));
        }
        if owned_gates.values().any(|gate| gate.local_name == name) {
            return Err(FlowError::AlreadyExists(format!(
                "conditional middleware guardrail '{name}' already exists for this activation"
            )));
        }
        let handle = format!("gate-{}", Uuid::now_v7());
        let qualified_name = format!(
            "__nemo_relay_worker_gate__{}__{}__{}",
            self.activation_id, name, handle
        );
        let callback_name = name.clone();
        let gate_callback = self
            .gate_callback
            .lock()
            .map_err(|error| FlowError::Internal(error.to_string()))?
            .clone();
        register_conditional_middleware_guardrail(
            &qualified_name,
            kinds,
            &registration_name,
            Arc::new(move |kinds, registration_name| {
                if !callback {
                    return Some(reason.clone());
                }
                let Some(invoker) = &gate_callback else {
                    return None;
                };
                match invoker.invoke(&callback_name, kinds, registration_name) {
                    Ok(reason) => reason,
                    Err(error) => {
                        log::warn!(
                            target: "nemo_relay.worker",
                            event = "worker_conditional_middleware_guardrail_failed",
                            callback = callback_name.as_str(),
                            registration_name = registration_name;
                            "Worker conditional middleware guardrail failed open: {error}"
                        );
                        None
                    }
                }
            }),
        )?;
        owned_gates.insert(
            handle.clone(),
            WorkerOwnedGate {
                local_name: name,
                qualified_name,
            },
        );
        Ok(handle)
    }

    fn issue_request_codec(
        self: &Arc<Self>,
        invocation_id: &str,
        codec: Arc<dyn LlmCodec>,
    ) -> FlowResult<WorkerCodecCapabilityGuard> {
        self.issue_codec(invocation_id, WorkerCodecDirection::Request(codec))
    }

    fn issue_response_codec(
        self: &Arc<Self>,
        invocation_id: &str,
        codec: Arc<dyn LlmResponseCodec>,
    ) -> FlowResult<WorkerCodecCapabilityGuard> {
        self.issue_codec(invocation_id, WorkerCodecDirection::Response(codec))
    }

    fn issue_codec(
        self: &Arc<Self>,
        invocation_id: &str,
        direction: WorkerCodecDirection,
    ) -> FlowResult<WorkerCodecCapabilityGuard> {
        let id = format!("codec-{}", Uuid::now_v7());
        let mut codecs = self
            .codecs
            .lock()
            .map_err(|error| FlowError::Internal(format!("codec lock poisoned: {error}")))?;
        codecs.insert(
            id.clone(),
            WorkerCodecCapability {
                invocation_id: invocation_id.to_owned(),
                direction,
            },
        );
        Ok(WorkerCodecCapabilityGuard {
            host_state: Arc::clone(self),
            capability_id: id,
        })
    }

    fn remove_codec(&self, id: &str) {
        if let Ok(mut codecs) = self.codecs.lock() {
            codecs.remove(id);
        }
    }

    fn request_codec(&self, id: &str, invocation_id: &str) -> Result<Arc<dyn LlmCodec>, Status> {
        let codecs = self
            .codecs
            .lock()
            .map_err(|err| Status::internal(format!("codec lock poisoned: {err}")))?;
        let capability = codecs
            .get(id)
            .ok_or_else(|| Status::not_found("codec capability is unavailable"))?;
        if capability.invocation_id != invocation_id {
            return Err(Status::permission_denied(
                "codec capability does not belong to this invocation",
            ));
        }
        match &capability.direction {
            WorkerCodecDirection::Request(codec) => Ok(codec.clone()),
            WorkerCodecDirection::Response(_) => Err(Status::invalid_argument(
                "codec capability is response-only",
            )),
        }
    }

    fn response_codec(
        &self,
        id: &str,
        invocation_id: &str,
    ) -> Result<Arc<dyn LlmResponseCodec>, Status> {
        let codecs = self
            .codecs
            .lock()
            .map_err(|err| Status::internal(format!("codec lock poisoned: {err}")))?;
        let capability = codecs
            .get(id)
            .ok_or_else(|| Status::not_found("codec capability is unavailable"))?;
        if capability.invocation_id != invocation_id {
            return Err(Status::permission_denied(
                "codec capability does not belong to this invocation",
            ));
        }
        match &capability.direction {
            WorkerCodecDirection::Response(codec) => Ok(codec.clone()),
            WorkerCodecDirection::Request(_) => {
                Err(Status::invalid_argument("codec capability is request-only"))
            }
        }
    }

    fn authorize(&self, activation_id: &str, token: &str) -> Result<(), Status> {
        if activation_id != self.activation_id || token != self.auth_token {
            return Err(Status::unauthenticated("invalid worker host token"));
        }
        Ok(())
    }

    fn insert_invocation_scope_stack(
        &self,
        source_stack: nemo_relay_runtime::api::runtime::ScopeStackHandle,
        publication_buffer: Option<PublicationBuffer>,
    ) -> FlowResult<String> {
        let mut stacks = self
            .scope_stacks
            .lock()
            .map_err(|error| FlowError::Internal(format!("scope stack lock poisoned: {error}")))?;
        loop {
            if !self.scope_accepting.load(Ordering::Acquire) {
                return Err(FlowError::NotFound(
                    "worker activation is shutting down".into(),
                ));
            }
            let cleanups = self.scope_stack_cleanups.lock().map_err(|error| {
                FlowError::Internal(format!("scope cleanup lock poisoned: {error}"))
            })?;
            if !cleanups
                .iter()
                .any(|handle| Arc::ptr_eq(handle, &source_stack))
            {
                break;
            }
            drop(stacks);
            let guard = self
                .scope_stack_cleanup_complete
                .wait(cleanups)
                .map_err(|error| {
                    FlowError::Internal(format!("scope cleanup lock poisoned: {error}"))
                })?;
            drop(guard);
            stacks = self.scope_stacks.lock().map_err(|error| {
                FlowError::Internal(format!("scope stack lock poisoned: {error}"))
            })?;
        }

        let continuation_context =
            MiddlewareContinuationContext::capture().isolated_with_scope_stack(&source_stack)?;
        let stack = continuation_context.scope_stack();
        let id = format!("invoke-{}", Uuid::now_v7());
        let stack_guard = stack
            .read()
            .map_err(|error| FlowError::Internal(format!("scope stack lock poisoned: {error}")))?;
        let invocation_base_depth = stack_guard.scopes().len();
        drop(stack_guard);
        stacks.insert(
            id.clone(),
            StoredScopeStack {
                lifetime: Arc::default(),
                handle: stack,
                publication_buffer,
                continuation_context: Some(continuation_context),
                invocation_base_depth: Some(invocation_base_depth),
            },
        );
        Ok(id)
    }

    fn cancel_scope_stack_snapshot(&self, scope_stack_identifier: &str) {
        if scope_stack_identifier.is_empty() {
            return;
        }
        // Recover the marker set only to preserve revocation during cleanup.
        // Inspection reports a poisoned lock as an explicit error.
        self.snapshot_cancelled_bindings
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(scope_stack_identifier.to_owned());
    }

    fn admit_scope_push(
        &self,
        scope: Option<&ScopeContext>,
    ) -> Result<(Option<StoredInvocationContext>, ScopePushAdmission), Status> {
        let stacks = self
            .scope_stacks
            .lock()
            .map_err(|error| Status::internal(format!("scope stack lock poisoned: {error}")))?;
        if !self.scope_accepting.load(Ordering::Acquire) {
            return Err(Status::failed_precondition(
                "worker activation is shutting down",
            ));
        }
        let cancelled = self.snapshot_cancelled_bindings.lock().map_err(|error| {
            Status::internal(format!("scope cancellation lock poisoned: {error}"))
        })?;
        if scope.is_some_and(|scope| cancelled.contains(&scope.scope_stack_id)) {
            return Err(Status::cancelled("scope stack binding is cancelled"));
        }
        let context = scope
            .map(|scope| self.select_scope_context(&stacks, scope))
            .transpose()?;
        let binding = scope.map(|scope| stacks[&scope.scope_stack_id].lifetime.clone());
        let admission = ScopePushAdmission {
            _binding_activity: binding
                .as_ref()
                .map(|binding| ScopeActivityGuard::new(&binding.pushes)),
            binding,
            _activation_activity: ScopeActivityGuard::new(&self.scope_activity),
        };
        Ok((context, admission))
    }

    fn record_scope_cleanup(&self, lifetime: &ScopeStackLifetime, result: FlowResult<()>) {
        let result = result.map_err(|error| error.to_string());
        if let Err(error) = &result {
            self.scope_cleanup_error
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .get_or_insert_with(|| error.clone());
        }
        lifetime.complete.send_replace(Some(result));
    }

    fn revoke_scope_stack(self: &Arc<Self>, id: &str, runtime: &tokio::runtime::Handle) {
        let (stored, unwind, activity) = {
            let Ok(mut stacks) = self.scope_stacks.lock() else {
                return;
            };
            let Some(stored) = stacks.remove(id) else {
                return;
            };
            stored.lifetime.revoked.store(true, Ordering::Release);
            self.snapshot_cancelled_bindings
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .remove(id);
            let activity = ScopeActivityGuard::new(&self.scope_activity);
            let unwind = match self.prepare_scope_unwind(&stacks, &stored) {
                Ok(unwind) => unwind,
                Err(error) => {
                    self.record_scope_cleanup(&stored.lifetime, Err(error));
                    return;
                }
            };
            (stored, unwind, activity)
        };
        let state = self.clone();
        let id = id.to_owned();
        let lifetime = stored.lifetime.clone();
        let cleanup_runtime = runtime.clone();
        runtime.spawn(async move {
            let mut pushes = lifetime.pushes.subscribe();
            let _ = pushes.wait_for(|active| *active == 0).await;
            cleanup_runtime.spawn_blocking(move || {
                let _activity = activity;
                let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    state.cleanup_scope_stack(&id, stored, unwind)
                }))
                .unwrap_or_else(|payload| {
                    Err(FlowError::Internal(format!(
                        "worker scope cleanup panicked: {}",
                        panic_payload_message(payload.as_ref())
                    )))
                });
                state.record_scope_cleanup(&lifetime, result);
            });
        });
    }

    fn close_scope_admission(&self) -> FlowResult<()> {
        let _stacks = self
            .scope_stacks
            .lock()
            .map_err(|error| FlowError::Internal(format!("scope stack lock poisoned: {error}")))?;
        self.scope_accepting.store(false, Ordering::Release);
        Ok(())
    }

    async fn drain_owned_scopes(self: &Arc<Self>) -> FlowResult<()> {
        let deadline = tokio::time::Instant::now() + WORKER_RPC_TIMEOUT;
        let ids = self
            .scope_stacks
            .lock()
            .map_err(|error| FlowError::Internal(format!("scope stack lock poisoned: {error}")))?
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        for id in ids {
            self.revoke_scope_stack(&id, &tokio::runtime::Handle::current());
        }
        let mut activity = self.scope_activity.subscribe();
        tokio::time::timeout_at(deadline, activity.wait_for(|active| *active == 0))
            .await
            .map_err(|_| FlowError::Internal("worker scope cleanup timed out".into()))?
            .map_err(|_| FlowError::Internal("worker scope cleanup interrupted".into()))?;
        let state = self.clone();
        let unbound_activity = ScopeActivityGuard::new(&self.scope_activity);
        tokio::time::timeout_at(
            deadline,
            tokio::task::spawn_blocking(move || {
                let _activity = unbound_activity;
                let result =
                    std::panic::catch_unwind(AssertUnwindSafe(|| state.cleanup_unbound_scopes()))
                        .unwrap_or_else(|payload| {
                            Err(FlowError::Internal(format!(
                                "worker unbound scope cleanup panicked: {}",
                                panic_payload_message(payload.as_ref())
                            )))
                        });
                if let Err(error) = &result {
                    state
                        .scope_cleanup_error
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .get_or_insert_with(|| error.to_string());
                }
                result
            }),
        )
        .await
        .map_err(|_| FlowError::Internal("worker scope cleanup timed out".into()))?
        .map_err(|error| {
            FlowError::Internal(format!("worker unbound scope cleanup failed: {error}"))
        })??;
        if let Some(error) = self
            .scope_cleanup_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
        {
            return Err(FlowError::Internal(error));
        }
        Ok(())
    }

    fn pop_owned_scope(handle: &StoredScopeHandle) -> FlowResult<()> {
        match handle.context.run(|| {
            pop_scope(
                PopScopeParams::builder()
                    .handle_uuid(&handle.handle.uuid)
                    .build(),
            )
        }) {
            Ok(()) => Ok(()),
            Err(error) => {
                let stack = handle.context.scope_stack.read().map_err(|lock| {
                    FlowError::Internal(format!("scope stack lock poisoned: {lock}"))
                })?;
                if stack.find(&handle.handle.uuid).is_none() {
                    Ok(())
                } else {
                    Err(error)
                }
            }
        }
    }

    fn cleanup_unbound_scopes(&self) -> FlowResult<()> {
        let mut handles = Vec::new();
        self.scope_handles
            .lock()
            .map_err(|error| FlowError::Internal(format!("scope handle lock poisoned: {error}")))?
            .retain(|_, handle| {
                if handle.scope_stack_id.is_empty() {
                    handles.push(handle.clone());
                    false
                } else {
                    true
                }
            });
        handles.sort_by_key(|handle| std::cmp::Reverse(handle.handle.uuid));
        let mut failure = None;
        for handle in handles {
            if let Err(error) = Self::pop_owned_scope(&handle) {
                self.scope_handles
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .insert(format!("scope-{}", handle.handle.uuid), handle);
                failure.get_or_insert(error);
            }
        }
        failure.map_or(Ok(()), Err)
    }

    #[cfg(test)]
    fn cleanup_invocation_scope_stack(&self, id: &str) {
        let (stored, unwind) = {
            let Ok(mut stacks) = self.scope_stacks.lock() else {
                return;
            };
            let Some(stored) = stacks.remove(id) else {
                return;
            };
            stored.lifetime.revoked.store(true, Ordering::Release);
            self.snapshot_cancelled_bindings
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .remove(id);
            let unwind = match self.prepare_scope_unwind(&stacks, &stored) {
                Ok(unwind) => unwind,
                Err(error) => {
                    self.record_scope_cleanup(&stored.lifetime, Err(error));
                    return;
                }
            };
            (stored, unwind)
        };
        let lifetime = stored.lifetime.clone();
        let result = self.cleanup_scope_stack(id, stored, unwind);
        self.record_scope_cleanup(&lifetime, result);
    }

    fn cleanup_scope_stack(
        &self,
        id: &str,
        stored: StoredScopeStack,
        unwind: Option<ScopeUnwindPlan>,
    ) -> FlowResult<()> {
        let original = stored.invocation_base_depth.map(|_| stored.handle.clone());
        let mut owned = Vec::new();
        self.scope_handles
            .lock()
            .map_err(|error| FlowError::Internal(format!("scope handle lock poisoned: {error}")))?
            .retain(|_, handle| {
                if handle.scope_stack_id == id {
                    owned.push(handle.clone());
                    false
                } else {
                    true
                }
            });
        owned.sort_by_key(|handle| std::cmp::Reverse(handle.handle.uuid));
        let mut failure = None;
        for handle in owned {
            if original
                .as_ref()
                .is_some_and(|stack| Arc::ptr_eq(stack, &handle.context.scope_stack))
            {
                continue;
            }
            if let Err(error) = Self::pop_owned_scope(&handle) {
                self.scope_handles
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .insert(format!("scope-{}", handle.handle.uuid), handle);
                failure.get_or_insert(error);
            }
        }
        if let Some(unwind) = unwind {
            self.scope_stack_cleanups
                .lock()
                .map_err(|error| {
                    FlowError::Internal(format!("scope cleanup lock poisoned: {error}"))
                })?
                .push(unwind.handle.clone());
            let _cleanup = ScopeStackCleanupGuard {
                state: self,
                handle: unwind.handle.clone(),
            };
            if let Err(error) = Self::unwind_scope_stack(
                &unwind.handle,
                unwind.base_depth,
                unwind.continuation_context,
                unwind.publication_buffer,
            ) {
                failure.get_or_insert(error);
            }
        }
        failure.map_or(Ok(()), Err)
    }

    fn prepare_scope_unwind(
        &self,
        stacks: &HashMap<String, StoredScopeStack>,
        stored: &StoredScopeStack,
    ) -> FlowResult<Option<ScopeUnwindPlan>> {
        let mut base_depth = stored.invocation_base_depth.unwrap_or(1);
        let mut pending = self.pending_scope_cleanups.lock().map_err(|error| {
            FlowError::Internal(format!("pending scope cleanup lock poisoned: {error}"))
        })?;
        if has_active_scope_stack_alias(stacks, &stored.handle) {
            pending.push(PendingScopeCleanup {
                handle: stored.handle.clone(),
                base_depth,
            });
            return Ok(None);
        }
        pending.retain(|cleanup| {
            if Arc::ptr_eq(&cleanup.handle, &stored.handle) {
                base_depth = base_depth.min(cleanup.base_depth);
                false
            } else {
                true
            }
        });
        Ok(Some(ScopeUnwindPlan {
            handle: stored.handle.clone(),
            base_depth,
            continuation_context: stored.continuation_context.clone(),
            publication_buffer: stored.publication_buffer.clone(),
        }))
    }

    fn unwind_scope_stack(
        stack: &nemo_relay_runtime::api::runtime::ScopeStackHandle,
        base_depth: usize,
        continuation_context: Option<MiddlewareContinuationContext>,
        publication_buffer: Option<PublicationBuffer>,
    ) -> FlowResult<()> {
        let unwind = || {
            let mut failure = None;
            loop {
                let top_uuid = {
                    let stack = stack.read().map_err(|error| {
                        FlowError::Internal(format!("scope stack lock poisoned: {error}"))
                    })?;
                    if stack.scopes().len() <= base_depth {
                        return failure.map_or(Ok(()), Err);
                    }
                    stack.top().uuid
                };
                if let Err(error) = with_scope_stack(stack.clone(), || {
                    pop_scope(PopScopeParams::builder().handle_uuid(&top_uuid).build())
                }) {
                    failure.get_or_insert(error);
                    stack
                        .write()
                        .map_err(|error| {
                            FlowError::Internal(format!("scope stack lock poisoned: {error}"))
                        })?
                        .remove(&top_uuid)?;
                }
            }
        };
        match continuation_context {
            Some(context) => context.run_sync(unwind),
            None => with_nested_publication_buffer(publication_buffer, || {
                with_scope_stack(stack.clone(), unwind)
            }),
        }
    }

    /// Store a callback and own the signal that keeps its calls active.
    fn insert_continuation(&self, continuation: Continuation) -> FlowResult<String> {
        let id = format!("next-{}", Uuid::now_v7());
        let mut continuations = self
            .continuations
            .lock()
            .map_err(|err| FlowError::Internal(format!("continuation lock poisoned: {err}")))?;
        let (lifetime, _) = watch::channel(());
        continuations.insert(
            id.clone(),
            StoredContinuation {
                continuation,
                lifetime,
                cleanup: watch::channel(ContinuationCleanupState::default()).0,
            },
        );
        Ok(id)
    }

    /// Remove a callback and cancel any provider work still using it.
    fn remove_continuation(&self, id: &str) {
        self.revoke_continuation(id);
    }

    fn revoke_continuation(&self, id: &str) -> Option<watch::Receiver<ContinuationCleanupState>> {
        self.continuations
            .lock()
            .ok()?
            .remove(id)
            .map(|stored| stored.cleanup.subscribe())
    }

    /// Acquire work ownership under the registry lock so revocation cannot miss it.
    fn continuation(
        &self,
        id: &str,
    ) -> Result<(Continuation, watch::Receiver<()>, ContinuationWorkGuard), Status> {
        let continuations = self
            .continuations
            .lock()
            .map_err(|err| Status::internal(format!("continuation lock poisoned: {err}")))?;
        let stored = continuations
            .get(id)
            .ok_or_else(|| Status::not_found("continuation not found"))?;
        stored.cleanup.send_modify(|state| state.active += 1);
        Ok((
            stored.continuation.clone(),
            stored.lifetime.subscribe(),
            ContinuationWorkGuard(stored.cleanup.clone()),
        ))
    }

    #[cfg(test)]
    fn stack(
        &self,
        id: &str,
    ) -> Result<Option<nemo_relay_runtime::api::runtime::ScopeStackHandle>, Status> {
        if id.is_empty() {
            return Ok(None);
        }
        self.scope_stacks
            .lock()
            .map_err(|err| Status::internal(format!("scope stack lock poisoned: {err}")))?
            .get(id)
            .map(|stored| stored.handle.clone())
            .map(Some)
            .ok_or_else(|| Status::not_found("scope stack not found"))
    }

    fn selected_scope_context(
        &self,
        scope: Option<&ScopeContext>,
    ) -> Result<Option<StoredInvocationContext>, Status> {
        let Some(scope) = scope else {
            return Ok(None);
        };
        let stacks = self
            .scope_stacks
            .lock()
            .map_err(|error| Status::internal(format!("scope stack lock poisoned: {error}")))?;
        let cancelled = self.snapshot_cancelled_bindings.lock().map_err(|error| {
            Status::internal(format!("snapshot cancellation lock poisoned: {error}"))
        })?;
        if cancelled.contains(&scope.scope_stack_id) {
            return Err(Status::cancelled("scope stack binding is cancelled"));
        }
        self.select_scope_context(&stacks, scope).map(Some)
    }

    fn select_scope_context(
        &self,
        stacks: &HashMap<String, StoredScopeStack>,
        scope: &ScopeContext,
    ) -> Result<StoredInvocationContext, Status> {
        let stored = stacks.get(&scope.scope_stack_id).ok_or_else(|| {
            Status::not_found("scope stack binding is expired, dropped, or foreign")
        })?;
        if stored.lifetime.revoked.load(Ordering::Acquire) {
            return Err(Status::not_found("scope stack binding is revoked"));
        }
        let context = StoredInvocationContext {
            scope_stack: stored.handle.clone(),
            publication_buffer: stored.publication_buffer.clone(),
            continuation_context: stored.continuation_context.clone(),
        };
        if scope.parent_scope_id.is_empty() {
            return Ok(context);
        }
        let handles = self
            .scope_handles
            .lock()
            .map_err(|error| Status::internal(format!("scope handle lock poisoned: {error}")))?;
        let parent = handles
            .get(&scope.parent_scope_id)
            .ok_or_else(|| Status::not_found("parent scope binding is unavailable"))?;
        if parent.scope_stack_id != scope.scope_stack_id {
            return Err(Status::permission_denied(
                "parent scope belongs to another binding",
            ));
        }
        if parent.pop_active.load(Ordering::Acquire) {
            return Err(Status::failed_precondition("parent scope is closing"));
        }
        let prefix = nemo_relay_runtime::api::runtime::snapshot_scope_stack_through_scope(
            &parent.context.scope_stack,
            &parent.handle.uuid,
        )
        .map_err(status_from_flow)?
        .ok_or_else(|| Status::not_found("parent scope binding is expired"))?;
        let mut selected = parent.context.clone();
        selected.continuation_context = selected
            .continuation_context
            .as_ref()
            .map(|context| context.isolated_with_scope_stack(&prefix))
            .transpose()
            .map_err(status_from_flow)?;
        selected.scope_stack = selected
            .continuation_context
            .as_ref()
            .map(MiddlewareContinuationContext::scope_stack)
            .unwrap_or(prefix);
        Ok(selected)
    }

    #[cfg(test)]
    fn invocation_context(&self, id: &str) -> Result<Option<StoredInvocationContext>, Status> {
        if id.is_empty() {
            return Ok(None);
        }
        self.scope_stacks
            .lock()
            .map_err(|err| Status::internal(format!("scope stack lock poisoned: {err}")))?
            .get(id)
            .map(|stored| StoredInvocationContext {
                scope_stack: stored.handle.clone(),
                publication_buffer: stored.publication_buffer.clone(),
                continuation_context: stored.continuation_context.clone(),
            })
            .map(Some)
            .ok_or_else(|| Status::not_found("scope stack not found"))
    }
}

#[derive(Clone)]
struct StoredInvocationContext {
    scope_stack: nemo_relay_runtime::api::runtime::ScopeStackHandle,
    publication_buffer: Option<PublicationBuffer>,
    continuation_context: Option<MiddlewareContinuationContext>,
}

impl StoredInvocationContext {
    fn run<T>(&self, callback: impl FnOnce() -> T) -> T {
        match &self.continuation_context {
            Some(context) => context.run_sync(callback),
            None => with_nested_publication_buffer(self.publication_buffer.clone(), || {
                with_scope_stack(self.scope_stack.clone(), callback)
            }),
        }
    }
}

/// Keep the cancellation sender in the registry, separate from active calls.
///
/// Removing this entry closes all receivers, including those held by provider calls.
struct StoredContinuation {
    continuation: Continuation,
    lifetime: watch::Sender<()>,
    cleanup: watch::Sender<ContinuationCleanupState>,
}

#[derive(Clone, Default)]
struct ContinuationCleanupState {
    active: usize,
    error: Option<String>,
}

struct ContinuationWorkGuard(watch::Sender<ContinuationCleanupState>);

impl ContinuationWorkGuard {
    fn failed_cleanup(&self, error: &FlowError) {
        self.0.send_modify(|state| {
            state.error.get_or_insert_with(|| error.to_string());
        });
    }
}

impl Drop for ContinuationWorkGuard {
    fn drop(&mut self) {
        self.0.send_modify(|state| state.active -= 1);
    }
}

async fn wait_for_continuation_cleanup(
    mut cleanup: watch::Receiver<ContinuationCleanupState>,
) -> FlowResult<()> {
    let state = tokio::time::timeout(
        WORKER_RPC_TIMEOUT,
        cleanup.wait_for(|state| state.active == 0),
    )
    .await
    .map_err(|_| FlowError::Internal("worker downstream continuation cleanup timed out".into()))?
    .map_err(|_| FlowError::Internal("worker downstream continuation cleanup interrupted".into()))?
    .clone();
    state
        .error
        .map_or(Ok(()), |error| Err(FlowError::Internal(error)))
}

/// Poll an operation until it finishes or its invocation ends.
async fn while_continuation_active<T>(
    lifetime: &mut watch::Receiver<()>,
    future: impl Future<Output = T>,
) -> Result<T, Status> {
    tokio::select! {
        biased;
        _ = lifetime.changed() => Err(Status::cancelled("worker invocation has ended")),
        result = future => Ok(result),
    }
}

#[derive(Clone)]
enum Continuation {
    Tool {
        next: ToolExecutionNextFn,
        context: MiddlewareContinuationContext,
    },
    Llm {
        next: LlmExecutionNextFn,
        context: MiddlewareContinuationContext,
    },
    LlmStream {
        next: LlmStreamExecutionNextFn,
        context: MiddlewareContinuationContext,
    },
}

impl Continuation {
    fn tool(next: ToolExecutionNextFn) -> Self {
        Self::Tool {
            next,
            context: MiddlewareContinuationContext::capture(),
        }
    }

    fn llm(next: LlmExecutionNextFn) -> Self {
        Self::Llm {
            next,
            context: MiddlewareContinuationContext::capture(),
        }
    }

    fn llm_stream(next: LlmStreamExecutionNextFn) -> Self {
        Self::LlmStream {
            next,
            context: MiddlewareContinuationContext::capture(),
        }
    }
}

struct WorkerHostRuntimeService {
    state: Arc<WorkerHostRuntimeState>,
}

fn runtime_registration_kind_from_surface(
    surface: RegistrationSurface,
) -> Result<RuntimeRegistrationKind, Status> {
    match surface {
        RegistrationSurface::Subscriber => Ok(RuntimeRegistrationKind::Subscriber),
        RegistrationSurface::EventMetadataInjector => {
            Ok(RuntimeRegistrationKind::EventMetadataInjector)
        }
        RegistrationSurface::MarkSanitizeGuardrail => {
            Ok(RuntimeRegistrationKind::MarkSanitizeGuardrail)
        }
        RegistrationSurface::ScopeSanitizeStartGuardrail => {
            Ok(RuntimeRegistrationKind::ScopeSanitizeStartGuardrail)
        }
        RegistrationSurface::ScopeSanitizeEndGuardrail => {
            Ok(RuntimeRegistrationKind::ScopeSanitizeEndGuardrail)
        }
        RegistrationSurface::ToolSanitizeRequestGuardrail => {
            Ok(RuntimeRegistrationKind::ToolSanitizeRequestGuardrail)
        }
        RegistrationSurface::ToolSanitizeResponseGuardrail => {
            Ok(RuntimeRegistrationKind::ToolSanitizeResponseGuardrail)
        }
        RegistrationSurface::ToolConditionalExecutionGuardrail => {
            Ok(RuntimeRegistrationKind::ToolConditionalExecutionGuardrail)
        }
        RegistrationSurface::ToolRequestIntercept => {
            Ok(RuntimeRegistrationKind::ToolRequestIntercept)
        }
        RegistrationSurface::ToolExecutionIntercept => {
            Ok(RuntimeRegistrationKind::ToolExecutionIntercept)
        }
        RegistrationSurface::LlmSanitizeRequestGuardrail => {
            Ok(RuntimeRegistrationKind::LlmSanitizeRequestGuardrail)
        }
        RegistrationSurface::LlmSanitizeResponseGuardrail => {
            Ok(RuntimeRegistrationKind::LlmSanitizeResponseGuardrail)
        }
        RegistrationSurface::LlmConditionalExecutionGuardrail => {
            Ok(RuntimeRegistrationKind::LlmConditionalExecutionGuardrail)
        }
        RegistrationSurface::LlmRequestIntercept => {
            Ok(RuntimeRegistrationKind::LlmRequestIntercept)
        }
        RegistrationSurface::LlmExecutionIntercept => {
            Ok(RuntimeRegistrationKind::LlmExecutionIntercept)
        }
        RegistrationSurface::LlmStreamExecutionIntercept => {
            Ok(RuntimeRegistrationKind::LlmStreamExecutionIntercept)
        }
        RegistrationSurface::ConditionalMiddlewareGuardrail | RegistrationSurface::Unspecified => {
            Err(Status::invalid_argument(
                "surface is not a runtime registration kind",
            ))
        }
    }
}

fn registration_surface_from_kind(kind: RuntimeRegistrationKind) -> RegistrationSurface {
    match kind {
        RuntimeRegistrationKind::Subscriber => RegistrationSurface::Subscriber,
        RuntimeRegistrationKind::EventMetadataInjector => {
            RegistrationSurface::EventMetadataInjector
        }
        RuntimeRegistrationKind::MarkSanitizeGuardrail => {
            RegistrationSurface::MarkSanitizeGuardrail
        }
        RuntimeRegistrationKind::ScopeSanitizeStartGuardrail => {
            RegistrationSurface::ScopeSanitizeStartGuardrail
        }
        RuntimeRegistrationKind::ScopeSanitizeEndGuardrail => {
            RegistrationSurface::ScopeSanitizeEndGuardrail
        }
        RuntimeRegistrationKind::ToolSanitizeRequestGuardrail => {
            RegistrationSurface::ToolSanitizeRequestGuardrail
        }
        RuntimeRegistrationKind::ToolSanitizeResponseGuardrail => {
            RegistrationSurface::ToolSanitizeResponseGuardrail
        }
        RuntimeRegistrationKind::ToolConditionalExecutionGuardrail => {
            RegistrationSurface::ToolConditionalExecutionGuardrail
        }
        RuntimeRegistrationKind::ToolRequestIntercept => RegistrationSurface::ToolRequestIntercept,
        RuntimeRegistrationKind::ToolExecutionIntercept => {
            RegistrationSurface::ToolExecutionIntercept
        }
        RuntimeRegistrationKind::LlmSanitizeRequestGuardrail => {
            RegistrationSurface::LlmSanitizeRequestGuardrail
        }
        RuntimeRegistrationKind::LlmSanitizeResponseGuardrail => {
            RegistrationSurface::LlmSanitizeResponseGuardrail
        }
        RuntimeRegistrationKind::LlmConditionalExecutionGuardrail => {
            RegistrationSurface::LlmConditionalExecutionGuardrail
        }
        RuntimeRegistrationKind::LlmRequestIntercept => RegistrationSurface::LlmRequestIntercept,
        RuntimeRegistrationKind::LlmExecutionIntercept => {
            RegistrationSurface::LlmExecutionIntercept
        }
        RuntimeRegistrationKind::LlmStreamExecutionIntercept => {
            RegistrationSurface::LlmStreamExecutionIntercept
        }
    }
}

#[tonic::async_trait]
impl RelayHostRuntime for WorkerHostRuntimeService {
    async fn get_current_scope_stack_snapshot(
        &self,
        request: Request<
            nemo_relay_worker_proto::nvidia_worker_v1::GetCurrentScopeStackSnapshotRequest,
        >,
    ) -> Result<
        Response<nemo_relay_worker_proto::nvidia_worker_v1::GetCurrentScopeStackSnapshotResponse>,
        Status,
    > {
        let request = request.into_inner();
        let credentials = request
            .credentials
            .as_ref()
            .ok_or_else(|| Status::unauthenticated("missing credentials"))?;
        self.state
            .authorize(&credentials.activation_id, &credentials.auth_token)?;
        nemo_relay_worker_proto::validation::validate(&request)
            .map_err(Status::invalid_argument)?;
        let snapshot = match request.scope {
            None => None,
            Some(scope) => {
                if scope.scope_stack_id.trim().is_empty() {
                    return Err(Status::invalid_argument(
                        "snapshot scope_stack_id must not be empty",
                    ));
                }
                // Keep ownership stable until the detached snapshot is complete.
                let stacks = self.state.scope_stacks.lock().map_err(|error| {
                    Status::internal(format!("scope stack lock poisoned: {error}"))
                })?;
                let cancelled = self
                    .state
                    .snapshot_cancelled_bindings
                    .lock()
                    .map_err(|error| {
                        Status::internal(format!("snapshot cancellation lock poisoned: {error}"))
                    })?;
                if cancelled.contains(&scope.scope_stack_id) {
                    return Err(Status::cancelled("scope stack binding is cancelled"));
                }
                let selected = self.state.select_scope_context(&stacks, &scope)?;
                Some(
                    nemo_relay_runtime::api::runtime::scope_stack_snapshot(&selected.scope_stack)
                        .map_err(status_from_flow)?,
                )
            }
        };
        let snapshot = snapshot.map(|snapshot| {
            nemo_relay_worker_proto::nvidia_worker_v1::ScopeStackSnapshot {
                scopes: snapshot
                    .scopes
                    .into_iter()
                    .map(
                        |scope| nemo_relay_worker_proto::nvidia_worker_v1::ScopeStackEntry {
                            scope_id: scope.scope_id.to_string(),
                            scope_type: snapshot_scope_type_to_proto(scope.scope_type),
                        },
                    )
                    .collect(),
            }
        });
        Ok(Response::new(
            nemo_relay_worker_proto::nvidia_worker_v1::GetCurrentScopeStackSnapshotResponse {
                result: Some(match snapshot {
                    Some(v) => wire::get_current_scope_stack_snapshot_response::Result::Snapshot(v),
                    None => wire::get_current_scope_stack_snapshot_response::Result::NoContext(()),
                }),
            },
        ))
    }

    async fn log(&self, request: Request<LogRequest>) -> Result<Response<LogResponse>, Status> {
        let request = request.into_inner();
        let credentials = request
            .credentials
            .as_ref()
            .ok_or_else(|| Status::unauthenticated("missing credentials"))?;
        self.state
            .authorize(&credentials.activation_id, &credentials.auth_token)?;
        nemo_relay_worker_proto::validation::validate(&request)
            .map_err(Status::invalid_argument)?;
        let level = match LogLevel::try_from(request.level) {
            Ok(LogLevel::Error) => log::Level::Error,
            Ok(LogLevel::Warn) => log::Level::Warn,
            Ok(LogLevel::Info) => log::Level::Info,
            Ok(LogLevel::Debug) => log::Level::Debug,
            Ok(LogLevel::Trace) => log::Level::Trace,
            Ok(LogLevel::Unspecified) | Err(_) => {
                return Err(Status::invalid_argument("log level must be specified"));
            }
        };
        let target = worker_log_target(&request.target)?;
        let fields = worker_log_fields(request.fields)?;
        let args = format_args!("{}", request.message);
        let record = log::Record::builder()
            .args(args)
            .level(level)
            .target(&target)
            .key_values(&fields)
            .build();
        log::logger().log(&record);
        Ok(Response::new(host_ack(Ok(()))?))
    }

    async fn list_runtime_registrations(
        &self,
        request: Request<ListRuntimeRegistrationsRequest>,
    ) -> Result<Response<ListRuntimeRegistrationsResponse>, Status> {
        let request = request.into_inner();
        let credentials = request
            .credentials
            .as_ref()
            .ok_or_else(|| Status::unauthenticated("missing credentials"))?;
        self.state
            .authorize(&credentials.activation_id, &credentials.auth_token)?;
        nemo_relay_worker_proto::validation::validate(&request)
            .map_err(Status::invalid_argument)?;
        let kinds = request
            .kinds
            .into_iter()
            .map(|kind| {
                RegistrationSurface::try_from(kind)
                    .map_err(|_| Status::invalid_argument("unknown runtime registration kind"))
                    .and_then(runtime_registration_kind_from_surface)
            })
            .collect::<Result<BTreeSet<_>, _>>()?;
        let selected = (!kinds.is_empty()).then_some(&kinds);
        let registrations = list_runtime_registrations(selected).map_err(status_from_flow)?;
        Ok(Response::new(ListRuntimeRegistrationsResponse {
            registrations: registrations
                .into_iter()
                .map(|v| conversions::encode(v).map_err(|e| Status::internal(e.to_string())))
                .collect::<Result<_, _>>()?,
        }))
    }

    async fn register_conditional_middleware_guardrail(
        &self,
        request: Request<RegisterConditionalMiddlewareGuardrailRequest>,
    ) -> Result<Response<RegisterConditionalMiddlewareGuardrailResponse>, Status> {
        let request = request.into_inner();
        let credentials = request
            .credentials
            .as_ref()
            .ok_or_else(|| Status::unauthenticated("missing credentials"))?;
        self.state
            .authorize(&credentials.activation_id, &credentials.auth_token)?;
        nemo_relay_worker_proto::validation::validate(&request)
            .map_err(Status::invalid_argument)?;
        let request = request
            .registration
            .ok_or_else(|| Status::invalid_argument("missing gate registration"))?;
        let kinds = request
            .kinds
            .into_iter()
            .map(|kind| {
                RegistrationSurface::try_from(kind)
                    .map_err(|_| Status::invalid_argument("unknown runtime registration kind"))
                    .and_then(runtime_registration_kind_from_surface)
            })
            .collect::<Result<BTreeSet<_>, _>>()?;
        let handle = self
            .state
            .register_owned_conditional_middleware_guardrail(
                request.name,
                kinds,
                request.registration_name,
                request.reason,
                request.callback,
            )
            .map_err(status_from_flow)?;
        Ok(Response::new(
            RegisterConditionalMiddlewareGuardrailResponse { handle },
        ))
    }

    async fn deregister_conditional_middleware_guardrail(
        &self,
        request: Request<DeregisterConditionalMiddlewareGuardrailRequest>,
    ) -> Result<Response<DeregisterConditionalMiddlewareGuardrailResponse>, Status> {
        let request = request.into_inner();
        let credentials = request
            .credentials
            .as_ref()
            .ok_or_else(|| Status::unauthenticated("missing credentials"))?;
        self.state
            .authorize(&credentials.activation_id, &credentials.auth_token)?;
        nemo_relay_worker_proto::validation::validate(&request)
            .map_err(Status::invalid_argument)?;
        let mut owned_gates = self
            .state
            .conditional_middleware_guardrails
            .lock()
            .map_err(|error| Status::internal(format!("gate ownership lock poisoned: {error}")))?;
        let removed = match owned_gates.get(&request.handle) {
            Some(gate) => {
                let removed = deregister_conditional_middleware_guardrail(&gate.qualified_name)
                    .map_err(status_from_flow)?;
                if removed {
                    owned_gates.remove(&request.handle);
                }
                removed
            }
            None => false,
        };
        Ok(Response::new(
            DeregisterConditionalMiddlewareGuardrailResponse { removed },
        ))
    }

    async fn emit_mark(
        &self,
        request: Request<EmitMarkRequest>,
    ) -> Result<Response<EmitMarkResponse>, Status> {
        let request = request.into_inner();
        let credentials = request
            .credentials
            .as_ref()
            .ok_or_else(|| Status::unauthenticated("missing credentials"))?;
        self.state
            .authorize(&credentials.activation_id, &credentials.auth_token)?;
        nemo_relay_worker_proto::validation::validate(&request)
            .map_err(Status::invalid_argument)?;
        let mark: nemo_relay_types::api::event::PendingMarkSpec = conversions::decode(
            request
                .mark
                .ok_or_else(|| Status::invalid_argument("missing mark"))?,
        )
        .map_err(|e| Status::invalid_argument(e.to_string()))?;
        self.with_stack(request.scope.as_ref(), || {
            emit_scope_mark(
                EmitMarkEventParams::builder()
                    .name(&mark.name)
                    .data_opt(mark.data)
                    .metadata_opt(mark.metadata)
                    .data_schema_opt(mark.data_schema)
                    .severity_opt(mark.severity)
                    .category_opt(mark.category)
                    .category_profile_opt(mark.category_profile)
                    .build(),
            )
        })?;
        Ok(Response::new(EmitMarkResponse::default()))
    }

    async fn get_runtime_diagnostics(
        &self,
        request: Request<GetRuntimeDiagnosticsRequest>,
    ) -> Result<Response<GetRuntimeDiagnosticsResponse>, Status> {
        let request = request.into_inner();
        let credentials = request
            .credentials
            .as_ref()
            .ok_or_else(|| Status::unauthenticated("missing credentials"))?;
        self.state
            .authorize(&credentials.activation_id, &credentials.auth_token)?;
        nemo_relay_worker_proto::validation::validate(&request)
            .map_err(Status::invalid_argument)?;
        Ok(Response::new(GetRuntimeDiagnosticsResponse {
            entries: active_runtime_diagnostics_snapshot()
                .into_iter()
                .map(|diagnostic| ProtoRuntimeDiagnostic {
                    code: diagnostic.code,
                    message: diagnostic.message,
                    count: diagnostic.count,
                })
                .collect(),
        }))
    }

    async fn push_scope(
        &self,
        request: Request<PushScopeRequest>,
    ) -> Result<Response<PushScopeResponse>, Status> {
        let request = request.into_inner();
        let credentials = request
            .credentials
            .as_ref()
            .ok_or_else(|| Status::unauthenticated("missing credentials"))?;
        self.state
            .authorize(&credentials.activation_id, &credentials.auth_token)?;
        nemo_relay_worker_proto::validation::validate(&request)
            .map_err(Status::invalid_argument)?;
        let (context, admission) = self.state.admit_scope_push(request.scope.as_ref())?;
        let push = || {
            push_scope(
                PushScopeParams::builder()
                    .name(&request.name)
                    .scope_type(proto_scope_type(request.scope_type))
                    .attributes(ScopeAttributes::empty())
                    .data_opt(optional_envelope_to_json(request.data)?)
                    .metadata_opt(optional_envelope_to_json(request.metadata)?)
                    .input_opt(optional_envelope_to_json(request.input)?)
                    .timestamp_opt(optional_worker_timestamp(request.time)?)
                    .build(),
            )
        };
        let result = match &context {
            Some(context) => context.run(push),
            None => push(),
        };
        match result {
            Ok(handle) => {
                let id = format!("scope-{}", handle.uuid);
                let scope_stack_id = request
                    .scope
                    .as_ref()
                    .map(|scope| scope.scope_stack_id.clone())
                    .unwrap_or_default();
                let stored = StoredScopeHandle {
                    handle,
                    scope_stack_id: scope_stack_id.clone(),
                    context: context.unwrap_or_else(|| StoredInvocationContext {
                        scope_stack: current_scope_stack(),
                        publication_buffer: capture_nested_publication_buffer(),
                        continuation_context: Some(MiddlewareContinuationContext::capture()),
                    }),
                    pop_active: Arc::new(AtomicBool::new(false)),
                };
                let commit = (|| {
                    let stacks = self.state.scope_stacks.lock().map_err(|error| {
                        Status::internal(format!("scope stack lock poisoned: {error}"))
                    })?;
                    if !self.state.scope_accepting.load(Ordering::Acquire) {
                        return Err(Status::failed_precondition(
                            "worker activation is shutting down",
                        ));
                    }
                    if let Some(binding) = &admission.binding
                        && (binding.revoked.load(Ordering::Acquire)
                            || stacks
                                .get(&scope_stack_id)
                                .is_none_or(|stack| !Arc::ptr_eq(&stack.lifetime, binding)))
                    {
                        return Err(Status::not_found(
                            "scope stack binding was revoked during scope creation",
                        ));
                    }
                    let mut handles = self.state.scope_handles.lock().map_err(|error| {
                        Status::internal(format!("scope handle lock poisoned: {error}"))
                    })?;
                    if let Some(scope) = request
                        .scope
                        .as_ref()
                        .filter(|scope| !scope.parent_scope_id.is_empty())
                        && handles
                            .get(&scope.parent_scope_id)
                            .is_none_or(|parent| parent.pop_active.load(Ordering::Acquire))
                    {
                        return Err(Status::not_found(
                            "parent scope binding expired during scope creation",
                        ));
                    }
                    handles.insert(id.clone(), stored.clone());
                    Ok(())
                })();
                if let Err(error) = commit {
                    let rollback = stored.context.run(|| {
                        pop_scope(
                            PopScopeParams::builder()
                                .handle_uuid(&stored.handle.uuid)
                                .build(),
                        )
                    });
                    if rollback.is_err() {
                        // The revoked binding's deferred unwind owns any scope
                        // temporarily hidden below another admitted push.
                        self.state
                            .scope_handles
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .insert(id, stored);
                    }
                    return Err(error);
                }
                Ok(Response::new(PushScopeResponse {
                    scope_handle_id: id,
                }))
            }
            Err(err) => Err(status_from_flow(err)),
        }
    }

    async fn pop_scope(
        &self,
        request: Request<PopScopeRequest>,
    ) -> Result<Response<PopScopeResponse>, Status> {
        let request = request.into_inner();
        let credentials = request
            .credentials
            .as_ref()
            .ok_or_else(|| Status::unauthenticated("missing credentials"))?;
        self.state
            .authorize(&credentials.activation_id, &credentials.auth_token)?;
        nemo_relay_worker_proto::validation::validate(&request)
            .map_err(Status::invalid_argument)?;
        let timestamp = match optional_worker_timestamp(request.time) {
            Ok(timestamp) => timestamp,
            Err(err) => return Ok(Response::new(host_ack(Err(err))?)),
        };
        let output = optional_envelope_to_json(request.output).map_err(status_from_flow)?;
        let metadata = optional_envelope_to_json(request.metadata).map_err(status_from_flow)?;
        let handle = {
            let handles =
                self.state.scope_handles.lock().map_err(|err| {
                    Status::internal(format!("scope handle lock poisoned: {err}"))
                })?;
            let handle = handles
                .get(&request.scope_handle_id)
                .ok_or_else(|| Status::not_found("scope handle not found"))?;
            if handle.pop_active.swap(true, Ordering::AcqRel) {
                return Err(Status::failed_precondition("scope is already closing"));
            }
            handle.clone()
        };
        let _reservation = ScopePopReservation(handle.pop_active.clone());
        if !handle.scope_stack_id.is_empty() {
            self.state.selected_scope_context(Some(&ScopeContext {
                scope_stack_id: handle.scope_stack_id.clone(),
                parent_scope_id: String::new(),
            }))?;
        }
        let result = handle.context.run(|| {
            pop_scope(
                PopScopeParams::builder()
                    .handle_uuid(&handle.handle.uuid)
                    .output_opt(output)
                    .metadata_opt(metadata)
                    .timestamp_opt(timestamp)
                    .build(),
            )
        });
        if result.is_ok() {
            self.state
                .scope_handles
                .lock()
                .map_err(|err| Status::internal(format!("scope handle lock poisoned: {err}")))?
                .remove(&request.scope_handle_id);
        }
        Ok(Response::new(host_ack(result)?))
    }

    async fn create_scope_stack(
        &self,
        request: Request<CreateScopeStackRequest>,
    ) -> Result<Response<CreateScopeStackResponse>, Status> {
        let request = request.into_inner();
        let credentials = request
            .credentials
            .as_ref()
            .ok_or_else(|| Status::unauthenticated("missing credentials"))?;
        self.state
            .authorize(&credentials.activation_id, &credentials.auth_token)?;
        nemo_relay_worker_proto::validation::validate(&request)
            .map_err(Status::invalid_argument)?;
        let id = format!("stack-{}", Uuid::now_v7());
        let mut stacks = self
            .state
            .scope_stacks
            .lock()
            .map_err(|err| Status::internal(format!("scope stack lock poisoned: {err}")))?;
        if !self.state.scope_accepting.load(Ordering::Acquire) {
            return Err(Status::failed_precondition(
                "worker activation is shutting down",
            ));
        }
        stacks.insert(
            id.clone(),
            StoredScopeStack {
                lifetime: Arc::default(),
                handle: nemo_relay_runtime::api::runtime::create_scope_stack(),
                publication_buffer: None,
                continuation_context: None,
                invocation_base_depth: None,
            },
        );
        Ok(Response::new(CreateScopeStackResponse {
            scope_stack_id: id,
        }))
    }

    async fn drop_scope_stack(
        &self,
        request: Request<DropScopeStackRequest>,
    ) -> Result<Response<DropScopeStackResponse>, Status> {
        let request = request.into_inner();
        let credentials = request
            .credentials
            .as_ref()
            .ok_or_else(|| Status::unauthenticated("missing credentials"))?;
        self.state
            .authorize(&credentials.activation_id, &credentials.auth_token)?;
        nemo_relay_worker_proto::validation::validate(&request)
            .map_err(Status::invalid_argument)?;
        drop(
            self.state
                .scope_stacks
                .lock()
                .map_err(|error| Status::internal(format!("scope stack lock poisoned: {error}")))?,
        );
        self.state
            .revoke_scope_stack(&request.scope_stack_id, &tokio::runtime::Handle::current());
        Ok(Response::new(DropScopeStackResponse {}))
    }

    /// Run a tool call and drop its future if the worker invocation ends.
    async fn tool_next(
        &self,
        request: Request<ToolNextRequest>,
    ) -> Result<Response<ToolNextResponse>, Status> {
        let request = request.into_inner();
        let credentials = request
            .credentials
            .as_ref()
            .ok_or_else(|| Status::unauthenticated("missing credentials"))?;
        self.state
            .authorize(&credentials.activation_id, &credentials.auth_token)?;
        nemo_relay_worker_proto::validation::validate(&request)
            .map_err(Status::invalid_argument)?;
        let (continuation, mut lifetime, _work) =
            self.state.continuation(&request.continuation_id)?;
        let Continuation::Tool { next, context } = continuation else {
            return Err(Status::invalid_argument(
                "continuation is not a tool continuation",
            ));
        };
        let context = self.isolated_continuation_context(&context, request.scope.as_ref())?;
        let value =
            required_envelope(request.value, "tool next value").map_err(status_from_flow)?;
        let value = decode_wire::<Json>(&value)
            .map_err(|err| Status::invalid_argument(format!("invalid tool next JSON: {err}")))?;
        let result = while_continuation_active(
            &mut lifetime,
            AssertUnwindSafe(context.invoke(move || next(value))).catch_unwind(),
        )
        .await?
        .unwrap_or_else(|payload| {
            Err(FlowError::Internal(format!(
                "worker tool continuation panicked: {}",
                panic_payload_message(payload.as_ref())
            )))
        });
        let downstream_failed = result.is_err();
        let mut response = tool_execution_result_response(result);
        if downstream_failed
            && let Some(wire::tool_next_response::Result::Error(error)) = &mut response.result
        {
            error.downstream_continuation_id = Some(request.continuation_id);
        }
        Ok(Response::new(response))
    }

    /// Run an LLM call and drop its future if the worker invocation ends.
    async fn llm_next(
        &self,
        request: Request<LlmNextRequest>,
    ) -> Result<Response<LlmNextResponse>, Status> {
        let request = request.into_inner();
        let credentials = request
            .credentials
            .as_ref()
            .ok_or_else(|| Status::unauthenticated("missing credentials"))?;
        self.state
            .authorize(&credentials.activation_id, &credentials.auth_token)?;
        nemo_relay_worker_proto::validation::validate(&request)
            .map_err(Status::invalid_argument)?;
        let (continuation, mut lifetime, _work) =
            self.state.continuation(&request.continuation_id)?;
        let Continuation::Llm { next, context } = continuation else {
            return Err(Status::invalid_argument(
                "continuation is not an LLM continuation",
            ));
        };
        let context = self.isolated_continuation_context(&context, request.scope.as_ref())?;
        let continuation_id = request.continuation_id;
        let request =
            required_envelope(request.request, "llm next request").map_err(status_from_flow)?;
        let request = decode_wire::<LlmRequest>(&request)
            .map_err(|err| Status::invalid_argument(format!("invalid LLM next request: {err}")))?;
        let result = while_continuation_active(
            &mut lifetime,
            AssertUnwindSafe(context.invoke(move || next(request))).catch_unwind(),
        )
        .await?
        .unwrap_or_else(|payload| {
            Err(FlowError::Internal(format!(
                "worker LLM continuation panicked: {}",
                panic_payload_message(payload.as_ref())
            )))
        });
        let mut response: wire::LlmNextResponse = json_result(result);
        if let Some(wire::llm_next_response::Result::Error(error)) = &mut response.result {
            error.downstream_continuation_id = Some(continuation_id);
        }
        Ok(Response::new(response))
    }

    type LlmStreamNextStream = Pin<
        Box<dyn tokio_stream::Stream<Item = Result<wire::LlmStreamNextResponse, Status>> + Send>,
    >;

    /// Open and forward a provider stream while its worker invocation is active.
    async fn llm_stream_next(
        &self,
        request: Request<LlmStreamNextRequest>,
    ) -> Result<Response<Self::LlmStreamNextStream>, Status> {
        let request = request.into_inner();
        let credentials = request
            .credentials
            .as_ref()
            .ok_or_else(|| Status::unauthenticated("missing credentials"))?;
        self.state
            .authorize(&credentials.activation_id, &credentials.auth_token)?;
        nemo_relay_worker_proto::validation::validate(&request)
            .map_err(Status::invalid_argument)?;
        let (continuation, mut lifetime, work) =
            self.state.continuation(&request.continuation_id)?;
        let Continuation::LlmStream { next, context } = continuation else {
            return Err(Status::invalid_argument(
                "continuation is not an LLM stream continuation",
            ));
        };
        let context = self.isolated_continuation_context(&context, request.scope.as_ref())?;
        let continuation_id = request.continuation_id;
        let request = required_envelope(request.request, "llm stream next request")
            .map_err(status_from_flow)?;
        let request = decode_wire::<LlmRequest>(&request).map_err(|err| {
            Status::invalid_argument(format!("invalid LLM stream next request: {err}"))
        })?;
        let stream = while_continuation_active(
            &mut lifetime,
            AssertUnwindSafe(context.invoke(move || next(request))).catch_unwind(),
        )
        .await?
        .map_err(|payload| {
            downstream_status(
                Status::internal(format!(
                    "worker stream continuation panicked: {}",
                    panic_payload_message(payload.as_ref())
                )),
                &continuation_id,
            )
        })?
        .map_err(|error| downstream_status(status_from_flow(error), &continuation_id))?;
        let (tx, rx) = mpsc::channel(16);
        tokio::spawn(async move {
            context
                .run(async move {
                    let _work = &work;
                    let mut stream = stream;
                    let mut terminal_error = None;
                    loop {
                        let item = tokio::select! {
                            biased;
                            _ = lifetime.changed() => break,
                            _ = tx.closed() => break,
                            item = AssertUnwindSafe(stream.next()).catch_unwind() => item,
                        };
                        match item {
                            Ok(Some(Ok(value))) => {
                                if !matches!(
                                    while_continuation_active(&mut lifetime, tx.send(Ok(value)))
                                        .await,
                                    Ok(Ok(()))
                                ) {
                                    break;
                                }
                            }
                            Ok(Some(Err(error))) => {
                                terminal_error = Some(error);
                                break;
                            }
                            Ok(None) => break,
                            Err(payload) => {
                                terminal_error = Some(FlowError::Internal(format!(
                                    "worker stream continuation panicked: {}",
                                    panic_payload_message(payload.as_ref())
                                )));
                                break;
                            }
                        }
                    }
                    // Cancellation revokes further provider polling, but cleanup must
                    // still run in the continuation context and has its own deadline.
                    let cleanup = tokio::time::timeout(
                        WORKER_RPC_TIMEOUT,
                        AssertUnwindSafe(async { stream.close().await }).catch_unwind(),
                    )
                    .await;
                    let cleanup = match cleanup {
                        Ok(Ok(result)) => result,
                        Ok(Err(payload)) => Err(FlowError::Internal(format!(
                            "worker downstream stream cleanup panicked: {}",
                            panic_payload_message(payload.as_ref())
                        ))),
                        Err(_) => Err(FlowError::Internal(
                            "worker downstream stream cleanup timed out".into(),
                        )),
                    };
                    if let Err(error) = cleanup {
                        work.failed_cleanup(&error);
                        if terminal_error.is_some()
                            || tx.is_closed()
                            || lifetime.has_changed().is_err()
                        {
                            log::warn!(
                                target: "nemo_relay.worker",
                                event = "worker_downstream_stream_cleanup_failed";
                                "Worker downstream stream cleanup failed: {error}"
                            );
                        } else {
                            terminal_error = Some(error);
                        }
                    }
                    if let Some(error) = terminal_error {
                        let _ = while_continuation_active(&mut lifetime, tx.send(Err(error))).await;
                    }
                })
                .await;
        });
        let mapped = tokio_stream::wrappers::ReceiverStream::new(rx).map(move |item| match item {
            Ok(value) => Ok(wire::LlmStreamNextResponse {
                item: Some(wire::llm_stream_next_response::Item::Value(encode_json(
                    JSON_SCHEMA,
                    &value,
                ))),
            }),
            Err(err) => {
                let mut error = flow_error_to_worker(err);
                error.downstream_continuation_id = Some(continuation_id.clone());
                Ok(wire::LlmStreamNextResponse {
                    item: Some(wire::llm_stream_next_response::Item::Error(error)),
                })
            }
        });
        Ok(Response::new(Box::pin(mapped)))
    }

    async fn decode_llm_codec_request(
        &self,
        request: Request<DecodeLlmCodecRequestRequest>,
    ) -> Result<Response<DecodeLlmCodecRequestResponse>, Status> {
        let request = request.into_inner();
        let credentials = request
            .credentials
            .as_ref()
            .ok_or_else(|| Status::unauthenticated("missing credentials"))?;
        self.state
            .authorize(&credentials.activation_id, &credentials.auth_token)?;
        nemo_relay_worker_proto::validation::validate(&request)
            .map_err(Status::invalid_argument)?;
        let codec = self
            .state
            .request_codec(&request.codec_capability_id, &request.invocation_id)?;
        let request = required_envelope(request.request, "codec request")
            .and_then(|value| {
                decode_wire::<LlmRequest>(&value)
                    .map_err(|err| FlowError::Internal(err.to_string()))
            })
            .map_err(status_from_flow)?;
        Ok(Response::new(typed_json_result(
            ANNOTATED_LLM_REQUEST_SCHEMA,
            codec.decode(&request),
        )))
    }

    async fn encode_llm_codec_request(
        &self,
        request: Request<EncodeLlmCodecRequestRequest>,
    ) -> Result<Response<EncodeLlmCodecRequestResponse>, Status> {
        let request = request.into_inner();
        let credentials = request
            .credentials
            .as_ref()
            .ok_or_else(|| Status::unauthenticated("missing credentials"))?;
        self.state
            .authorize(&credentials.activation_id, &credentials.auth_token)?;
        nemo_relay_worker_proto::validation::validate(&request)
            .map_err(Status::invalid_argument)?;
        let codec = self
            .state
            .request_codec(&request.codec_capability_id, &request.invocation_id)?;
        let annotated = required_envelope(request.annotated_request, "annotated codec request")
            .and_then(|value| {
                decode_wire::<AnnotatedLlmRequest>(&value)
                    .map_err(|err| FlowError::Internal(err.to_string()))
            })
            .map_err(status_from_flow)?;
        let original = required_envelope(request.original_request, "original codec request")
            .and_then(|value| {
                decode_wire::<LlmRequest>(&value)
                    .map_err(|err| FlowError::Internal(err.to_string()))
            })
            .map_err(status_from_flow)?;
        Ok(Response::new(typed_json_result(
            LLM_REQUEST_SCHEMA,
            codec.encode(&annotated, &original),
        )))
    }

    async fn decode_llm_codec_response(
        &self,
        request: Request<DecodeLlmCodecResponseRequest>,
    ) -> Result<Response<DecodeLlmCodecResponseResponse>, Status> {
        let request = request.into_inner();
        let credentials = request
            .credentials
            .as_ref()
            .ok_or_else(|| Status::unauthenticated("missing credentials"))?;
        self.state
            .authorize(&credentials.activation_id, &credentials.auth_token)?;
        nemo_relay_worker_proto::validation::validate(&request)
            .map_err(Status::invalid_argument)?;
        let codec = self
            .state
            .response_codec(&request.codec_capability_id, &request.invocation_id)?;
        let response = required_envelope(request.response, "codec response")
            .and_then(|value| {
                decode_wire::<Json>(&value).map_err(|err| FlowError::Internal(err.to_string()))
            })
            .map_err(status_from_flow)?;
        Ok(Response::new(typed_json_result(
            "",
            codec.decode_response(&response),
        )))
    }
}

fn worker_log_target(target: &str) -> Result<String, Status> {
    if target.is_empty() {
        return Ok("nemo_relay.plugin.worker".into());
    }
    if target.len() > 256
        || target
            .bytes()
            .any(|byte| byte == 0 || byte.is_ascii_control())
    {
        return Err(Status::invalid_argument("invalid plugin log target"));
    }
    Ok(format!("nemo_relay.plugin.worker.{target}"))
}

struct WorkerLogFields(Map<String, Json>);

impl log::kv::Source for WorkerLogFields {
    fn visit<'kvs>(
        &'kvs self,
        visitor: &mut dyn log::kv::VisitSource<'kvs>,
    ) -> Result<(), log::kv::Error> {
        for (key, value) in &self.0 {
            visitor.visit_pair(
                log::kv::Key::from_str(key),
                log::kv::Value::from_serde(value),
            )?;
        }
        Ok(())
    }
}

fn worker_log_fields(fields: Option<wire::JsonObject>) -> Result<WorkerLogFields, Status> {
    let Some(fields) = fields else {
        return Ok(WorkerLogFields(Map::new()));
    };
    match decode_value(&fields).map_err(|error| Status::invalid_argument(error.to_string()))? {
        Json::Object(fields) => Ok(WorkerLogFields(fields)),
        _ => Err(Status::invalid_argument(
            "plugin log fields must be a JSON object",
        )),
    }
}

impl WorkerHostRuntimeService {
    fn isolated_continuation_context(
        &self,
        context: &MiddlewareContinuationContext,
        scope: Option<&ScopeContext>,
    ) -> Result<MiddlewareContinuationContext, Status> {
        let Some(selected) = self.state.selected_scope_context(scope)? else {
            return context.isolated().map_err(status_from_flow);
        };
        context
            .isolated_with_scope_stack(&selected.scope_stack)
            .map_err(status_from_flow)
    }

    fn with_stack<T>(
        &self,
        scope: Option<&ScopeContext>,
        f: impl FnOnce() -> FlowResult<T>,
    ) -> Result<T, Status> {
        match self.state.selected_scope_context(scope)? {
            Some(context) => context.run(f),
            None => f(),
        }
        .map_err(status_from_flow)
    }
}

mod invoke_request_payload {
    pub(crate) use super::transport::Payload;
}

mod llm_invocation {
    pub(crate) use super::transport::SanitizeContext;
}

mod invoke_response_result {
    pub(crate) use nemo_relay_worker_proto::nvidia_worker_v1::invoke_response::Result;
}

mod stream_chunk_item {
    pub(crate) use nemo_relay_worker_proto::nvidia_worker_v1::invoke_stream_response::Item;
}

fn invoke_request_payload_event(event: &Event) -> FlowResult<invoke_request_payload::Payload> {
    Ok(invoke_request_payload::Payload::Event(Box::new(
        encode_wire(event)?,
    )))
}

fn invoke_request_payload_tool(
    tool_name: &str,
    value: Json,
    tool_call_id: Option<&str>,
) -> invoke_request_payload::Payload {
    invoke_request_payload::Payload::Tool(ToolInvocation {
        tool_name: tool_name.into(),
        value: Some(encode_json(JSON_SCHEMA, &value)),
        tool_call_id: tool_call_id.map(Into::into),
    })
}

fn invoke_request_payload_llm(
    model_name: &str,
    request: Option<LlmRequest>,
    annotated_request: Option<AnnotatedLlmRequest>,
    response: Option<Json>,
) -> FlowResult<invoke_request_payload::Payload> {
    Ok(invoke_request_payload::Payload::Llm(Box::new(
        LlmInvocation {
            model_name: model_name.into(),
            request: request.as_ref().map(encode_wire).transpose()?,
            annotated_request: annotated_request.as_ref().map(encode_wire).transpose()?,
            response: response
                .as_ref()
                .map(|value| encode_json(JSON_SCHEMA, value)),
            sanitize_context: None,
            execution_codec_context: None,
        },
    )))
}

fn invoke_request_payload_llm_context(
    model_name: &str,
    request: Option<LlmRequest>,
    annotated_request: Option<AnnotatedLlmRequest>,
    response: Option<Json>,
    context: impl Into<llm_invocation::SanitizeContext>,
) -> FlowResult<invoke_request_payload::Payload> {
    let mut payload = invoke_request_payload_llm(model_name, request, annotated_request, response)?;
    if let invoke_request_payload::Payload::Llm(llm) = &mut payload {
        llm.sanitize_context = Some(context.into());
    }
    Ok(payload)
}

fn codec_identity_to_proto(identity: &LlmCodecIdentity) -> ProtoLlmCodecIdentity {
    conversions::encode(identity.clone()).expect("valid codec identity")
}

fn encode_wire<D: Clone + WireValue<W>, W>(value: &D) -> FlowResult<W> {
    conversions::encode(value.clone())
        .map_err(|error| FlowError::InvalidArgument(error.to_string()))
}

fn encode_json(_schema: &str, value: &Json) -> wire::JsonValue {
    // serde_json::Value cannot contain non-finite numbers or invalid JSON.
    wire::JsonValue {
        json: serde_json::to_vec(value).expect("JSON values serialize"),
    }
}

// Message validation cannot relate an InvokeResponse to its originating request.
// Keep that pairing explicit before any callback-specific decoder applies a value.
fn validate_callback_result(
    surface: Option<RegistrationSurface>,
    response: &InvokeResponse,
) -> Result<(), Status> {
    use RegistrationSurface as S;
    use wire::invoke_response::Result as R;
    let valid = matches!(
        (surface, response.result.as_ref()),
        (Some(_), Some(R::Error(_)))
            | (Some(S::Subscriber), Some(R::Completed(_)))
            | (
                Some(S::EventMetadataInjector),
                Some(R::MetadataAdditions(_))
            )
            | (
                Some(
                    S::MarkSanitizeGuardrail
                        | S::ScopeSanitizeStartGuardrail
                        | S::ScopeSanitizeEndGuardrail
                ),
                Some(R::EventSanitize(_))
            )
            | (
                Some(
                    S::ToolSanitizeRequestGuardrail
                        | S::ToolSanitizeResponseGuardrail
                        | S::ToolRequestIntercept
                ),
                Some(R::ToolValue(_))
            )
            | (
                Some(
                    S::ToolConditionalExecutionGuardrail
                        | S::LlmConditionalExecutionGuardrail
                        | S::ConditionalMiddlewareGuardrail
                ),
                Some(R::Guardrail(_))
            )
            | (
                Some(S::ToolExecutionIntercept),
                Some(R::ToolExecutionOutcome(_))
            )
            | (
                Some(S::LlmSanitizeRequestGuardrail),
                Some(R::LlmRequest(_) | R::Unchanged(_))
            )
            | (
                Some(S::LlmSanitizeResponseGuardrail),
                Some(R::LlmResponse(_) | R::Unchanged(_))
            )
            | (Some(S::LlmRequestIntercept), Some(R::LlmRequestOutcome(_)))
            | (Some(S::LlmExecutionIntercept), Some(R::LlmResponse(_)))
    );
    if valid {
        Ok(())
    } else {
        Err(Status::data_loss(format!(
            "unexpected callback result alternative for {}",
            surface
                .map(|value| value.as_str_name())
                .unwrap_or("UNKNOWN")
        )))
    }
}

fn json_from_invoke_response(response: InvokeResponse) -> FlowResult<Json> {
    match response.result {
        Some(
            wire::invoke_response::Result::ToolValue(v)
            | wire::invoke_response::Result::LlmResponse(v),
        ) => decode_wire(&v),
        Some(wire::invoke_response::Result::MetadataAdditions(v)) => decode_wire(&v),
        Some(wire::invoke_response::Result::LlmRequest(v)) => {
            serde_json::to_value(decode_wire::<LlmRequest>(&v)?)
                .map_err(|e| FlowError::Internal(e.to_string()))
        }
        Some(wire::invoke_response::Result::Error(e)) => Err(worker_error_to_flow(e)),
        _ => Err(FlowError::InvalidArgument(
            "unexpected callback result alternative".into(),
        )),
    }
}

fn optional_json_from_invoke_response(response: InvokeResponse) -> FlowResult<Option<Json>> {
    if matches!(
        response.result,
        Some(wire::invoke_response::Result::Unchanged(_))
    ) {
        Ok(None)
    } else {
        json_from_invoke_response(response).map(Some)
    }
}

fn guardrail_from_invoke_response(response: InvokeResponse) -> FlowResult<Option<String>> {
    match response.result {
        Some(wire::invoke_response::Result::Guardrail(v)) => match v.decision {
            Some(wire::guardrail_decision::Decision::Allow(_)) => Ok(None),
            Some(wire::guardrail_decision::Decision::BlockReason(reason)) if !reason.is_empty() => {
                Ok(Some(reason))
            }
            _ => Err(FlowError::InvalidArgument(
                "missing guardrail decision".into(),
            )),
        },
        Some(wire::invoke_response::Result::Error(e)) => Err(worker_error_to_flow(e)),
        _ => Err(FlowError::InvalidArgument(
            "unexpected guardrail result".into(),
        )),
    }
}

fn json_from_stream_chunk(chunk: StreamChunk) -> FlowResult<Json> {
    nemo_relay_worker_proto::validation::validate(&chunk).map_err(FlowError::InvalidArgument)?;
    match chunk.item {
        Some(stream_chunk_item::Item::Value(value)) => decode_wire::<Json>(&value)
            .map_err(|err| FlowError::Internal(format!("invalid worker stream chunk: {err}"))),
        Some(stream_chunk_item::Item::Error(error)) => Err(worker_error_to_flow(error)),
        None => Err(FlowError::Internal("worker stream chunk was empty".into())),
    }
}

fn required_envelope<W>(value: Option<W>, field: &str) -> FlowResult<W> {
    value.ok_or_else(|| FlowError::InvalidArgument(format!("missing {field}")))
}

fn optional_envelope_to_json(value: Option<JsonEnvelope>) -> FlowResult<Option<Json>> {
    value
        .map(|value| {
            decode_wire::<Json>(&value)
                .map_err(|err| FlowError::Internal(format!("invalid JSON envelope: {err}")))
        })
        .transpose()
}

fn optional_worker_timestamp(
    value: Option<prost_types::Timestamp>,
) -> FlowResult<Option<DateTime<Utc>>> {
    value
        .map(|v| conversions::decode(v).map_err(|e| FlowError::InvalidArgument(e.to_string())))
        .transpose()
}

fn host_ack<R: Default>(result: FlowResult<()>) -> Result<R, Status> {
    result.map(|()| R::default()).map_err(status_from_flow)
}

fn json_result<R: RpcResponse<Json>>(result: FlowResult<Json>) -> R {
    R::build(result)
}

fn typed_json_result<D, R: RpcResponse<D>>(_schema: &str, result: FlowResult<D>) -> R {
    R::build(result)
}

fn tool_execution_result_response(
    result: FlowResult<ToolExecutionResult>,
) -> wire::ToolNextResponse {
    wire::ToolNextResponse::build(result)
}

fn tool_execution_intercept_outcome_from_proto(
    value: ProtoToolExecutionInterceptOutcome,
) -> FlowResult<ToolExecutionInterceptOutcome> {
    conversions::decode(value).map_err(|e| FlowError::InvalidArgument(e.to_string()))
}

fn flow_error_to_worker(err: FlowError) -> WorkerError {
    WorkerError {
        code: "host.runtime_error".into(),
        message: super::http_error::to_message(&err),
        retryable: false,
        downstream_continuation_id: None,
        http_status: match &err {
            FlowError::Upstream(failure) => failure.status.map(u32::from),
            _ => None,
        },
    }
}

fn worker_error_to_flow(error: WorkerError) -> FlowError {
    let mut error = error;
    if error.code != "worker.cancelled"
        && let Some(status) = error.http_status.and_then(|v| u16::try_from(v).ok())
        && nemo_relay_types::plugin::parse_http_error_message(&error.message).is_none()
        && let Ok(message) = nemo_relay_types::plugin::http_error_message(status, &error.message)
    {
        error.message = message;
    }
    if error.code == "worker.cancelled" {
        FlowError::Internal(format!("worker invocation cancelled: {}", error.message))
    } else if let Some(failure) = super::http_error::upstream_from_message(&error.message) {
        failure
    } else {
        FlowError::Internal(format!("{}: {}", error.code, error.message))
    }
}

fn worker_status_to_flow(context: &str, error: Status) -> FlowError {
    match error.code() {
        tonic::Code::DeadlineExceeded => {
            FlowError::Internal(format!("worker invocation timed out: {error}"))
        }
        tonic::Code::Cancelled => {
            FlowError::Internal(format!("worker invocation cancelled: {error}"))
        }
        _ => FlowError::Internal(format!("{context}: {error}")),
    }
}

fn worker_error_to_plugin(error: WorkerError, fallback: &str) -> PluginError {
    let message = if error.message.is_empty() {
        fallback.to_string()
    } else {
        format!("{}: {}", error.code, error.message)
    };
    PluginError::RegistrationFailed(message)
}

fn status_from_flow(err: FlowError) -> Status {
    if let FlowError::InvalidArgument(message) = &err {
        return Status::invalid_argument(message.clone());
    }
    Status::internal(err.to_string())
}

fn snapshot_scope_type_to_proto(scope_type: ScopeType) -> i32 {
    use nemo_relay_worker_proto::nvidia_worker_v1::ScopeType as ProtoScopeType;
    (match scope_type {
        ScopeType::Agent => ProtoScopeType::Agent,
        ScopeType::Function => ProtoScopeType::Function,
        ScopeType::Tool => ProtoScopeType::Tool,
        ScopeType::Llm => ProtoScopeType::Llm,
        ScopeType::Retriever => ProtoScopeType::Retriever,
        ScopeType::Embedder => ProtoScopeType::Embedder,
        ScopeType::Reranker => ProtoScopeType::Reranker,
        ScopeType::Guardrail => ProtoScopeType::Guardrail,
        ScopeType::Evaluator => ProtoScopeType::Evaluator,
        ScopeType::Custom => ProtoScopeType::Custom,
        ScopeType::Unknown => ProtoScopeType::Unknown,
    }) as i32
}

fn proto_scope_type(scope_type: i32) -> ScopeType {
    match nemo_relay_worker_proto::nvidia_worker_v1::ScopeType::try_from(scope_type) {
        Ok(nemo_relay_worker_proto::nvidia_worker_v1::ScopeType::Agent) => ScopeType::Agent,
        Ok(nemo_relay_worker_proto::nvidia_worker_v1::ScopeType::Function) => ScopeType::Function,
        Ok(nemo_relay_worker_proto::nvidia_worker_v1::ScopeType::Tool) => ScopeType::Tool,
        Ok(nemo_relay_worker_proto::nvidia_worker_v1::ScopeType::Llm) => ScopeType::Llm,
        Ok(nemo_relay_worker_proto::nvidia_worker_v1::ScopeType::Retriever) => ScopeType::Retriever,
        Ok(nemo_relay_worker_proto::nvidia_worker_v1::ScopeType::Embedder) => ScopeType::Embedder,
        Ok(nemo_relay_worker_proto::nvidia_worker_v1::ScopeType::Reranker) => ScopeType::Reranker,
        Ok(nemo_relay_worker_proto::nvidia_worker_v1::ScopeType::Guardrail) => ScopeType::Guardrail,
        Ok(nemo_relay_worker_proto::nvidia_worker_v1::ScopeType::Evaluator) => ScopeType::Evaluator,
        Ok(nemo_relay_worker_proto::nvidia_worker_v1::ScopeType::Custom) => ScopeType::Custom,
        Ok(nemo_relay_worker_proto::nvidia_worker_v1::ScopeType::Unknown) => ScopeType::Unknown,
        _ => ScopeType::Custom,
    }
}

fn validate_registration_plan(
    plugin_id: &str,
    response: &wire::RegistrationPlan,
) -> crate::Result<()> {
    for registration in &response.registrations {
        if registration.local_name.trim().is_empty() {
            return Err(PluginError::RegistrationFailed(format!(
                "worker plugin '{plugin_id}' returned a registration with empty local_name"
            )));
        }
        let surface = RegistrationSurface::try_from(registration.surface).map_err(|_| {
            PluginError::RegistrationFailed(format!(
                "worker plugin '{plugin_id}' returned unsupported registration surface {}",
                registration.surface
            ))
        })?;
        if surface == RegistrationSurface::Unspecified {
            return Err(PluginError::RegistrationFailed(format!(
                "worker plugin '{plugin_id}' returned unspecified registration surface"
            )));
        }
    }
    let mut gate_names = std::collections::HashSet::new();
    for gate in &response.conditional_middleware_guardrails {
        if gate.name.trim().is_empty()
            || gate.registration_name.trim().is_empty()
            || gate.kinds.is_empty()
        {
            return Err(PluginError::RegistrationFailed(format!(
                "worker plugin '{plugin_id}' returned an invalid conditional middleware guardrail"
            )));
        }
        if !gate_names.insert(gate.name.as_str()) {
            return Err(PluginError::RegistrationFailed(format!(
                "worker plugin '{plugin_id}' returned duplicate conditional middleware guardrail '{}'",
                gate.name
            )));
        }
    }
    Ok(())
}

fn diagnostics_have_errors(diagnostics: &[ConfigDiagnostic]) -> bool {
    diagnostics
        .iter()
        .any(|diagnostic| diagnostic.level == DiagnosticLevel::Error)
}

fn worker_error_diagnostic(plugin_kind: &str, code: &str, message: &str) -> ConfigDiagnostic {
    ConfigDiagnostic {
        level: DiagnosticLevel::Error,
        code: code.into(),
        component: Some(plugin_kind.into()),
        field: None,
        message: message.into(),
    }
}

fn validate_worker_handshake(plugin_id: &str, handshake: &HandshakeResponse) -> crate::Result<()> {
    let identity = handshake
        .identity
        .as_ref()
        .ok_or_else(|| PluginError::RegistrationFailed("missing worker identity".into()))?;
    let protocol = identity.protocol.as_ref().ok_or_else(|| {
        PluginError::RegistrationFailed("missing worker protocol identity".into())
    })?;
    if identity.plugin_id != plugin_id
        || identity.plugin_kind != plugin_id
        || protocol.name != WORKER_PROTOCOL
        || protocol.version != wire::CONTRACT_VERSION
    {
        return Err(PluginError::RegistrationFailed(
            "incompatible worker identity".into(),
        ));
    }
    Ok(())
}

fn validate_relay_compatibility(relay: Option<&str>) -> crate::Result<()> {
    validate_dynamic_plugin_relay_compatibility(relay, "worker")
}

fn resolve_manifest_relative_path(manifest_path: &Path, value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        manifest_path
            .parent()
            .map(|parent| parent.join(&path))
            .unwrap_or(path)
    }
}

#[cfg(unix)]
fn unix_endpoint_display(path: &Path) -> String {
    format!("unix://{}", path.display())
}

#[cfg(test)]
#[path = "../../tests/unit/dynamic_worker_tests.rs"]
mod tests;

// Attribution follows the returned error, never a mutable "last next failed" flag.
// Existing continuation IDs are bounded host-issued identities; no error ledger is retained.
fn is_propagated_continuation(origin: Option<&str>, continuation_id: &str) -> bool {
    !continuation_id.is_empty() && origin == Some(continuation_id)
}

fn is_worker_callback_failure(error: &WorkerError, continuation_id: &str) -> bool {
    error.code != "worker.cancelled"
        && !is_propagated_continuation(error.downstream_continuation_id.as_deref(), continuation_id)
}

fn is_downstream_status(status: &Status, continuation_id: &str) -> bool {
    is_propagated_continuation(
        status
            .metadata()
            .get(DOWNSTREAM_CONTINUATION_METADATA)
            .and_then(|value| value.to_str().ok()),
        continuation_id,
    )
}

fn downstream_status(mut status: Status, continuation_id: &str) -> Status {
    // IDs originate in insert_continuation and are ASCII, including the UUID.
    if let Ok(value) = continuation_id.parse() {
        status
            .metadata_mut()
            .insert(DOWNSTREAM_CONTINUATION_METADATA, value);
    }
    status
}

fn report_worker_response(mut response: InvokeResponse, plugin_id: &str) -> InvokeResponse {
    let error = match response.result.as_mut() {
        Some(invoke_response_result::Result::Error(error)) => Some(error),
        _ => None,
    };
    if let Some(error) = error
        && error.code != "worker.cancelled"
        && let Some(failure) = super::http_error::upstream_from_message(&error.message)
    {
        error.message = super::http_error::to_message(&super::http_error::reported_by(
            failure, plugin_id, "worker",
        ));
    }
    response
}

fn worker_stream_rpc_request(
    message: PreparedInvocation,
) -> FlowResult<Request<wire::InvokeStreamRequest>> {
    let name = message.registration_name.clone();
    let message = message.into_stream_wire()?;
    nemo_relay_worker_proto::validation::validate(&message).map_err(FlowError::InvalidArgument)?;
    Ok(nemo_relay_worker_proto::callback_request(message, &name))
}
trait DecodeTo<D> {
    fn decode_to(&self) -> FlowResult<D>;
}
impl<D: WireValue<W>, W: Clone> DecodeTo<D> for W {
    fn decode_to(&self) -> FlowResult<D> {
        conversions::decode(self.clone()).map_err(|e| FlowError::InvalidArgument(e.to_string()))
    }
}
fn decode_wire<D>(value: &impl DecodeTo<D>) -> FlowResult<D> {
    value.decode_to()
}
fn decode_value<D>(value: &impl DecodeTo<D>) -> FlowResult<D> {
    value.decode_to()
}

trait RpcResponse<D> {
    fn build(result: FlowResult<D>) -> Self;
}
macro_rules! rpc_response {
    ($ty:ty,$module:ident,$domain:ty) => {
        impl RpcResponse<$domain> for $ty {
            fn build(result: FlowResult<$domain>) -> Self {
                Self {
                    result: Some(
                        match result.and_then(|v| {
                            conversions::encode(v).map_err(|e| FlowError::Internal(e.to_string()))
                        }) {
                            Ok(v) => wire::$module::Result::Value(v),
                            Err(e) => wire::$module::Result::Error(flow_error_to_worker(e)),
                        },
                    ),
                }
            }
        }
    };
}
rpc_response!(
    wire::ToolNextResponse,
    tool_next_response,
    ToolExecutionResult
);
rpc_response!(wire::LlmNextResponse, llm_next_response, Json);
rpc_response!(
    wire::DecodeLlmCodecRequestResponse,
    decode_llm_codec_request_response,
    AnnotatedLlmRequest
);
rpc_response!(
    wire::EncodeLlmCodecRequestResponse,
    encode_llm_codec_request_response,
    LlmRequest
);
rpc_response!(
    wire::DecodeLlmCodecResponseResponse,
    decode_llm_codec_response_response,
    AnnotatedLlmResponse
);
