// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! gRPC worker dynamic plugin loader and host-side proxy adapter.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use futures_util::FutureExt;
use nemo_relay_worker_proto::v1::plugin_worker_client::PluginWorkerClient;
use nemo_relay_worker_proto::v1::relay_host_runtime_server::{
    RelayHostRuntime, RelayHostRuntimeServer,
};
use nemo_relay_worker_proto::v1::{
    CancelInvocationRequest, ConditionalMiddlewareInvocation, CreateScopeStackRequest,
    CreateScopeStackResponse, DeregisterConditionalMiddlewareGuardrailRequest,
    DeregisterConditionalMiddlewareGuardrailResponse, DropScopeStackRequest, EmitMarkRequest,
    GetRuntimeDiagnosticsRequest, GetRuntimeDiagnosticsResponse, GuardrailResult, HandshakeRequest,
    HandshakeResponse, HealthRequest, HostAck, InvokeRequest, InvokeResponse, JsonEnvelope,
    JsonResult, ListRuntimeRegistrationsRequest, ListRuntimeRegistrationsResponse,
    LlmCodecDecodeRequest, LlmCodecDecodeResponse, LlmCodecEncodeRequest,
    LlmCodecIdentity as ProtoLlmCodecIdentity, LlmCodecKind, LlmInvocation, LlmNextRequest,
    LlmSanitizeRequestContext as ProtoLlmSanitizeRequestContext,
    LlmSanitizeResponseContext as ProtoLlmSanitizeResponseContext, LlmStreamNextRequest,
    PopScopeRequest, PushScopeRequest, PushScopeResponse,
    RegisterConditionalMiddlewareGuardrailRequest, RegisterConditionalMiddlewareGuardrailResponse,
    RegisterRequest, RegisterResponse, Registration, RegistrationSurface,
    RuntimeDiagnostic as ProtoRuntimeDiagnostic,
    RuntimeRegistrationIdentity as ProtoRuntimeRegistrationIdentity,
    RuntimeRegistrationOwner as ProtoRuntimeRegistrationOwner,
    RuntimeRegistrationOwnerKind as ProtoRuntimeRegistrationOwnerKind, ScopeContext,
    ShutdownRequest, StreamChunk,
    ToolExecutionInterceptOutcome as ProtoToolExecutionInterceptOutcome,
    ToolExecutionResult as ProtoToolExecutionResult, ToolExecutionResultResponse, ToolInvocation,
    ToolNextRequest, ValidateRequest, WorkerError,
};
use nemo_relay_worker_proto::{
    WORKER_PROTOCOL_GRPC_V1, decode_json_envelope, decode_json_value, json_envelope, json_value,
};
use serde_json::{Map, Value as Json};
use sha2::{Digest, Sha256};
use tokio::runtime::{Builder as RuntimeBuilder, Runtime};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::StreamExt;
use tonic::transport::{Channel, Endpoint, Server};
use tonic::{Request, Response, Status};
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

use crate::api::event::{DataSchema, Event, EventCategory, EventSanitizeFields, LogSeverity};
use crate::api::llm::{LLM_REQUEST_INTERCEPT_OUTCOME_SCHEMA, LlmRequest};
use crate::api::registry::{
    RuntimeRegistrationKind, RuntimeRegistrationOwnerKind,
    deregister_conditional_middleware_guardrail, list_runtime_registrations,
    register_conditional_middleware_guardrail,
};
use crate::api::runtime::subscriber_dispatcher::{
    PublicationBuffer, capture_nested_publication_buffer, with_nested_publication_buffer,
};
use crate::api::runtime::{
    EventMetadataInjectorFn, EventSanitizeFn, LlmCodecIdentity, LlmExecutionNextFn, LlmJsonStream,
    LlmSanitizeRequestContext, LlmSanitizeResponseContext, LlmStreamExecutionNextFn,
    MiddlewareContinuationContext, ToolExecutionNextFn, current_scope_stack, with_scope_stack,
};
use crate::api::scope::{
    EmitMarkEventParams, PopScopeParams, PushScopeParams, ScopeAttributes, ScopeHandle, ScopeType,
    event as emit_scope_mark, pop_scope, push_scope,
};
use crate::api::tool::{ToolExecutionInterceptOutcome, ToolExecutionResult};
use crate::codec::request::{ANNOTATED_LLM_REQUEST_SCHEMA, AnnotatedLlmRequest};
use crate::codec::traits::{LlmCodec, LlmResponseCodec};
use crate::error::{FlowError, Result as FlowResult};
use crate::plugin::{
    ConfigDiagnostic, DiagnosticLevel, Plugin, PluginError, PluginRegistrationContext,
    active_runtime_diagnostics_snapshot, deregister_plugin_registration_checked,
    register_plugin_tracked,
};

use super::{
    DynamicPluginKind, DynamicPluginManifest, DynamicPluginManifestLoad,
    DynamicPluginTeardownOutcome, WorkerRuntime, deregister_tracked_registrations_checked,
    validate_annotated_request_consumer_compatibility, validate_dynamic_plugin_relay_compatibility,
};

const JSON_SCHEMA: &str = "nemo.relay.Json@1";
const DATA_SCHEMA_SCHEMA: &str = "nemo.relay.DataSchema@1";
const EVENT_SCHEMA: &str = "nemo.relay.Event@1";
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
pub fn load_worker_plugins<I>(specs: I) -> crate::plugin::Result<WorkerPluginActivation>
where
    I: IntoIterator<Item = WorkerPluginLoadSpec>,
{
    let mut activation = WorkerPluginActivation {
        plugins: Vec::new(),
        plugin_registrations: Vec::new(),
    };
    for spec in specs {
        let instance = load_one_worker_plugin(&spec)?;
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
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = crate::plugin::Result<()>> + Send + 'a>> {
        let config_matches = plugin_config == &self.instance.config;
        Box::pin(async move {
            if !config_matches {
                return Err(PluginError::RegistrationFailed(
                    "worker plugin config changed after dynamic activation; reload the worker activation"
                        .into(),
                ));
            }
            self.instance.install_registrations(ctx)
        })
    }
}

struct WorkerPluginInstance {
    plugin_kind: String,
    allows_multiple_components: bool,
    config: Map<String, Json>,
    validation_diagnostics: Vec<ConfigDiagnostic>,
    registrations: Vec<Registration>,
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

        self.host_state.cleanup_conditional_middleware_guardrails();

        let mut client = self.client.clone();
        let request = ShutdownRequest {
            activation_id: self.host_state.activation_id.clone(),
            auth_token: self.host_state.auth_token.clone(),
            reason: "plugin activation cleared".into(),
        };
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            block_on_runtime(self.runtime.runtime(), async move {
                worker_rpc(client.shutdown(worker_rpc_request(request))).await
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

        self.stop_process_checked(&mut outcome);
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

fn load_one_worker_plugin(
    spec: &WorkerPluginLoadSpec,
) -> crate::plugin::Result<Arc<WorkerPluginInstance>> {
    log::info!(
        target: "nemo_relay.worker",
        event = "worker_starting",
        plugin_id = spec.plugin_id.as_str();
        "Worker plugin is starting"
    );
    let (manifest, manifest_ref) = DynamicPluginManifest::load_from_path(&spec.manifest_ref)?;
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
        .ok_or_else(|| PluginError::InvalidConfig("load.entrypoint is required".into()))?;

    let activation_uuid = Uuid::now_v7();
    let activation_id = activation_uuid.to_string();
    let auth_token = Uuid::now_v7().to_string();
    let activation_dir = std::env::temp_dir().join(format!("nmrw-{}", activation_uuid.simple()));
    std::fs::create_dir_all(&activation_dir)
        .map_err(|err| PluginError::Internal(format!("worker activation directory: {err}")))?;
    let mut activation_dir_guard = ActivationDirGuard::new(activation_dir.clone());
    let runtime_handle = OwnedWorkerRuntime::new(
        RuntimeBuilder::new_multi_thread()
            .enable_all()
            .thread_name("nemo-relay-worker-host")
            .build()
            .map_err(|err| PluginError::Internal(format!("worker runtime: {err}")))?,
    );
    let WorkerEndpoints {
        host_server,
        host_advertise,
        worker_advertise,
        worker_connect,
        worker_endpoint_file,
    } = WorkerEndpoints::new(&activation_dir)?;
    let host_state = Arc::new(WorkerHostRuntimeState::new(
        activation_id.clone(),
        auth_token.clone(),
    ));
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    runtime_handle.runtime().spawn(serve_host_runtime(
        host_server,
        host_state.clone(),
        shutdown_rx,
    ));

    let manifest_path = PathBuf::from(&manifest_ref);
    let mut child = ChildGuard::new(spawn_worker_process(WorkerProcessLaunch {
        runtime,
        manifest_path: &manifest_path,
        environment_ref: spec.environment_ref.as_deref(),
        plugin_id: &spec.plugin_id,
        entrypoint,
        activation_id: &activation_id,
        auth_token: &auth_token,
        host_endpoint: &host_advertise,
        worker_endpoint: &worker_advertise,
        worker_endpoint_file: worker_endpoint_file.as_deref(),
    })?);
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
    let mut client = block_on_runtime(
        runtime_handle.runtime(),
        connect_worker_with_retry(&worker_connect, &spec.plugin_id),
    )?;

    let health = block_on_runtime(
        runtime_handle.runtime(),
        worker_rpc(client.health(worker_rpc_request(HealthRequest {
            activation_id: activation_id.clone(),
            auth_token: auth_token.clone(),
        }))),
    )
    .map_err(|err| PluginError::RegistrationFailed(format!("worker health check failed: {err}")))?;
    let health = health.into_inner();
    if !health.ok {
        let message = format!("worker plugin health check failed: {}", health.message);
        return Err(PluginError::RegistrationFailed(message));
    }

    let handshake = block_on_runtime(
        runtime_handle.runtime(),
        worker_rpc(client.handshake(worker_rpc_request(HandshakeRequest {
            activation_id: activation_id.clone(),
            plugin_id: spec.plugin_id.clone(),
            relay_version: env!("CARGO_PKG_VERSION").into(),
            worker_protocol: WORKER_PROTOCOL_GRPC_V1.into(),
            auth_token: auth_token.clone(),
            host_endpoint: host_advertise.clone(),
        }))),
    )
    .map_err(|err| PluginError::RegistrationFailed(format!("worker handshake failed: {err}")))?;
    let handshake = handshake.into_inner();
    validate_worker_handshake(&spec.plugin_id, &handshake)?;
    host_state.set_gate_callback(WorkerGateCallback {
        plugin_kind: spec.plugin_id.clone(),
        runtime: runtime_handle.runtime().handle().clone(),
        client: client.clone(),
        host_state: Arc::downgrade(&host_state),
    });

    let config = Json::Object(spec.config.clone());
    let validate = block_on_runtime(
        runtime_handle.runtime(),
        worker_rpc(client.validate(worker_rpc_request(ValidateRequest {
            activation_id: activation_id.clone(),
            plugin_id: spec.plugin_id.clone(),
            auth_token: auth_token.clone(),
            config: Some(json_envelope(JSON_SCHEMA, &config)?),
        }))),
    )
    .map_err(|err| {
        PluginError::RegistrationFailed(format!("worker validation RPC failed: {err}"))
    })?;
    let validate = validate.into_inner();
    if let Some(error) = validate.error {
        return Err(worker_error_to_plugin(error, "worker validation failed"));
    }
    let validation_diagnostics = match validate.diagnostics {
        Some(diagnostics) => decode_json_envelope::<Vec<ConfigDiagnostic>>(&diagnostics)
            .map_err(PluginError::Serialization)?,
        None => Vec::new(),
    };

    let (registrations, initial_gates) = if diagnostics_have_errors(&validation_diagnostics) {
        (Vec::new(), Vec::new())
    } else {
        let register = block_on_runtime(
            runtime_handle.runtime(),
            worker_rpc(client.register(worker_rpc_request(RegisterRequest {
                activation_id: activation_id.clone(),
                plugin_id: spec.plugin_id.clone(),
                auth_token: auth_token.clone(),
                config: Some(json_envelope(JSON_SCHEMA, &config)?),
            }))),
        )
        .map_err(|err| {
            PluginError::RegistrationFailed(format!("worker registration RPC failed: {err}"))
        })?;
        let register = register.into_inner();
        if let Some(error) = register.error {
            return Err(worker_error_to_plugin(error, "worker registration failed"));
        }
        validate_registration_plan(&spec.plugin_id, &register)?;
        (
            register.registrations,
            register.conditional_middleware_guardrails,
        )
    };
    for gate in initial_gates {
        let kinds = gate
            .kinds
            .into_iter()
            .map(|kind| {
                RegistrationSurface::try_from(kind)
                    .map_err(|_| {
                        PluginError::RegistrationFailed(format!(
                            "worker plugin '{}' returned an unknown conditional middleware guardrail kind",
                            spec.plugin_id
                        ))
                    })
                    .and_then(|surface| {
                        runtime_registration_kind_from_surface(surface).map_err(|status| {
                            PluginError::RegistrationFailed(status.message().to_string())
                        })
                    })
            })
            .collect::<crate::plugin::Result<BTreeSet<_>>>()?;
        if let Err(error) = host_state.register_owned_conditional_middleware_guardrail(
            gate.name,
            kinds,
            gate.registration_name,
            gate.reason,
            gate.callback,
        ) {
            host_state.cleanup_conditional_middleware_guardrails();
            return Err(PluginError::RegistrationFailed(format!(
                "worker initial conditional middleware guardrail failed: {error}"
            )));
        }
    }
    if registrations.iter().any(|registration| {
        RegistrationSurface::try_from(registration.surface)
            .is_ok_and(|surface| surface == RegistrationSurface::LlmRequestIntercept)
    }) {
        validate_annotated_request_consumer_compatibility(&relay_compat, &spec.plugin_id)?;
    }

    log::info!(
        target: "nemo_relay.worker",
        event = "worker_connected",
        plugin_id = spec.plugin_id.as_str();
        "Worker plugin connected and registered"
    );
    Ok(Arc::new(WorkerPluginInstance {
        plugin_kind: spec.plugin_id.clone(),
        allows_multiple_components: handshake.allows_multiple_components,
        config: spec.config.clone(),
        validation_diagnostics,
        registrations,
        runtime: runtime_handle,
        client,
        host_state,
        shutdown: Mutex::new(Some(shutdown_tx)),
        process: Mutex::new(Some(child.take())),
        activation_dir: activation_dir_guard.keep(),
        teardown_started: AtomicBool::new(false),
    }))
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
    fn new(activation_dir: &Path) -> crate::plugin::Result<Self> {
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

async fn serve_host_runtime(
    endpoint: HostRuntimeServer,
    state: Arc<WorkerHostRuntimeState>,
    shutdown: oneshot::Receiver<()>,
) {
    let service = RelayHostRuntimeServer::new(WorkerHostRuntimeService { state });
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
) -> crate::plugin::Result<PluginWorkerClient<Channel>> {
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
fn normalize_worker_tcp_endpoint(endpoint: &str) -> crate::plugin::Result<String> {
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
) -> crate::plugin::Result<Option<WorkerConnectEndpoint>> {
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
) -> crate::plugin::Result<PluginWorkerClient<Channel>> {
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
}

fn spawn_worker_process(spec: WorkerProcessLaunch<'_>) -> crate::plugin::Result<Child> {
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
    minimize_worker_environment(&mut command);
    command
        .current_dir(manifest_dir)
        .env("NEMO_RELAY_WORKER_ID", spec.activation_id)
        .env("NEMO_RELAY_PLUGIN_ID", spec.plugin_id)
        .env("NEMO_RELAY_WORKER_SOCKET", spec.worker_endpoint)
        .env("NEMO_RELAY_HOST_SOCKET", spec.host_endpoint)
        .env("NEMO_RELAY_WORKER_TOKEN", spec.auth_token)
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

fn minimize_worker_environment(command: &mut Command) {
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
    let retained = ALLOWLIST
        .iter()
        .filter_map(|name| std::env::var_os(name).map(|value| (*name, value)))
        .collect::<Vec<_>>();
    command.env_clear();
    command.envs(retained);
}

fn resolve_python_executable(
    plugin_id: &str,
    environment_ref: Option<&str>,
) -> crate::plugin::Result<PathBuf> {
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
    fn install_registrations(
        &self,
        ctx: &mut PluginRegistrationContext,
    ) -> crate::plugin::Result<()> {
        for registration in &self.registrations {
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
        ctx: &mut PluginRegistrationContext,
        name: &str,
    ) -> crate::plugin::Result<()> {
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
        ctx: &mut PluginRegistrationContext,
        name: &str,
        priority: i32,
    ) -> crate::plugin::Result<()> {
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
        ctx: &mut PluginRegistrationContext,
        name: &str,
        priority: i32,
        surface: RegistrationSurface,
    ) -> crate::plugin::Result<()> {
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
        ctx: &mut PluginRegistrationContext,
        registration: &Registration,
        surface: RegistrationSurface,
    ) -> crate::plugin::Result<()> {
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
                Arc::new(move |tool_name, value, next| {
                    let instance = instance.clone();
                    let callback_name = callback_name.clone();
                    let tool_name = tool_name.to_owned();
                    Box::pin(async move {
                        instance
                            .invoke_tool_execution(&callback_name, &tool_name, value, next)
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
        ctx: &mut PluginRegistrationContext,
        registration: &Registration,
        surface: RegistrationSurface,
    ) -> crate::plugin::Result<()> {
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
                Arc::new(move |model_name, request, next| {
                    let instance = instance.clone();
                    let callback_name = callback_name.clone();
                    let model_name = model_name.to_owned();
                    Box::pin(async move {
                        instance
                            .invoke_llm_execution(&callback_name, &model_name, request, next)
                            .await
                    })
                }),
            ),
            RegistrationSurface::LlmStreamExecutionIntercept => ctx
                .register_llm_stream_execution_intercept(
                    name,
                    priority,
                    Arc::new(move |model_name, request, next| {
                        let instance = instance.clone();
                        let callback_name = callback_name.clone();
                        let model_name = model_name.to_owned();
                        Box::pin(async move {
                            instance
                                .invoke_llm_stream_execution(
                                    &callback_name,
                                    &model_name,
                                    request,
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

    fn clone_for_callback(&self) -> WorkerPluginCallback {
        WorkerPluginCallback {
            plugin_kind: self.plugin_kind.clone(),
            activation_id: self.host_state.activation_id.clone(),
            runtime: self.runtime.handle(),
            client: self.client.clone(),
            host_state: self.host_state.clone(),
        }
    }
}

#[cfg(not(unix))]
fn bind_loopback_listener() -> crate::plugin::Result<(TcpListener, SocketAddr)> {
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
    plugin_kind: String,
    activation_id: String,
    runtime: tokio::runtime::Handle,
    client: PluginWorkerClient<Channel>,
    host_state: Arc<WorkerHostRuntimeState>,
}

#[derive(Clone)]
struct WorkerGateCallback {
    plugin_kind: String,
    runtime: tokio::runtime::Handle,
    client: PluginWorkerClient<Channel>,
    host_state: std::sync::Weak<WorkerHostRuntimeState>,
}

impl WorkerGateCallback {
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
    cancel_on_drop: bool,
    cleaned: bool,
}

impl WorkerInvocationGuard {
    fn new(callback: &WorkerPluginCallback, request: &InvokeRequest) -> Self {
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
            cancel_on_drop: true,
            cleaned: false,
        }
    }

    fn cancel(&mut self, reason: impl Into<String>) {
        if !self.cancel_on_drop {
            return;
        }
        self.cancel_on_drop = false;
        let mut client = self.client.clone();
        let request = CancelInvocationRequest {
            activation_id: self.activation_id.clone(),
            invocation_id: self.invocation_id.clone(),
            auth_token: self.auth_token.clone(),
            reason: reason.into(),
        };
        self.runtime.spawn(async move {
            let _ = worker_rpc(client.cancel_invocation(worker_rpc_request(request))).await;
        });
    }

    fn finish(&mut self) {
        self.cancel_on_drop = false;
        self.cleanup();
    }

    fn cleanup(&mut self) {
        if self.cleaned {
            return;
        }
        self.cleaned = true;
        if !self.continuation_id.is_empty() {
            self.host_state.remove_continuation(&self.continuation_id);
        }
        if !self.scope_stack_id.is_empty() {
            self.host_state
                .cleanup_invocation_scope_stack(&self.scope_stack_id);
        }
    }
}

impl Drop for WorkerInvocationGuard {
    fn drop(&mut self) {
        self.cancel("host caller cancelled the worker invocation");
        self.cleanup();
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
        );
        guardrail_from_invoke_response(self.invoke_blocking(request)?)
    }

    fn invoke_subscriber(&self, registration_name: &str, event: &Event) -> FlowResult<()> {
        let request = self.base_request(
            registration_name,
            RegistrationSurface::Subscriber,
            None,
            Some(invoke_request_payload_event(event)),
        );
        let response = self.invoke_blocking(request)?;
        match response.result {
            Some(invoke_response_result::Result::Empty(_)) | None => Ok(()),
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
            Some(invoke_request_payload_event(event)),
        );
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
            Some(invoke_request_payload_event(event)),
        );
        let value = json_from_invoke_response(self.invoke_async(request).await?)?;
        serde_json::from_value(value).map_err(|err| {
            FlowError::Internal(format!(
                "worker returned invalid event sanitize fields: {err}"
            ))
        })
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
            Some(invoke_request_payload_tool(tool_name, value)),
        );
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
            Some(invoke_request_payload_tool(tool_name, value)),
        );
        guardrail_from_invoke_response(self.invoke_async(request).await?)
    }

    async fn invoke_tool_execution(
        &self,
        registration_name: &str,
        tool_name: &str,
        value: Json,
        next: ToolExecutionNextFn,
    ) -> FlowResult<ToolExecutionInterceptOutcome> {
        let continuation_id = self
            .host_state
            .insert_continuation(Continuation::tool(next))?;
        let request = self.base_request(
            registration_name,
            RegistrationSurface::ToolExecutionIntercept,
            Some(continuation_id),
            Some(invoke_request_payload_tool(tool_name, value)),
        );
        let response = self.invoke_async(request).await?;
        match response.result {
            Some(invoke_response_result::Result::ToolExecution(result)) => {
                let outcome = result.outcome.ok_or_else(|| {
                    FlowError::Internal("worker tool execution intercept outcome is missing".into())
                })?;
                tool_execution_intercept_outcome_from_proto(outcome).map_err(|err| {
                    FlowError::Internal(format!(
                        "worker returned invalid tool execution intercept outcome: {err}"
                    ))
                })
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
                        codec_capability_id: None,
                    },
                ),
            )),
        );
        let capability_id = context.resolve_codec().map(|codec| {
            let capability_id = self
                .host_state
                .insert_request_codec(&invoke.invocation_id, codec);
            let Some(invoke_request_payload::Payload::Llm(llm)) = invoke.payload.as_mut() else {
                unreachable!("LLM sanitizer invocation must have an LLM payload");
            };
            let Some(llm_invocation::SanitizeContext::RequestSanitizeContext(context)) =
                llm.sanitize_context.as_mut()
            else {
                unreachable!("request sanitizer invocation must have a request context");
            };
            context.codec_capability_id = Some(capability_id.clone());
            capability_id
        });
        let _capability_guard = capability_id.as_ref().map(|capability_id| {
            WorkerCodecCapabilityGuard::new(Arc::clone(&self.host_state), capability_id.clone())
        });
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
                        codec_capability_id: None,
                    },
                ),
            )),
        );
        let capability_id = context.resolve_codec().map(|codec| {
            let capability_id = self
                .host_state
                .insert_response_codec(&invoke.invocation_id, codec);
            let Some(invoke_request_payload::Payload::Llm(llm)) = invoke.payload.as_mut() else {
                unreachable!("LLM sanitizer invocation must have an LLM payload");
            };
            let Some(llm_invocation::SanitizeContext::ResponseSanitizeContext(context)) =
                llm.sanitize_context.as_mut()
            else {
                unreachable!("response sanitizer invocation must have a response context");
            };
            context.codec_capability_id = Some(capability_id.clone());
            capability_id
        });
        let _capability_guard = capability_id.as_ref().map(|capability_id| {
            WorkerCodecCapabilityGuard::new(Arc::clone(&self.host_state), capability_id.clone())
        });
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
            Some(invoke_request_payload_llm("", Some(request), None, None)),
        );
        guardrail_from_invoke_response(self.invoke_async(invoke).await?)
    }

    async fn invoke_llm_request_intercept(
        &self,
        registration_name: &str,
        model_name: &str,
        request: LlmRequest,
        annotated: Option<AnnotatedLlmRequest>,
    ) -> FlowResult<crate::api::llm::LlmRequestInterceptOutcome> {
        let invoke = self.base_request(
            registration_name,
            RegistrationSurface::LlmRequestIntercept,
            None,
            Some(invoke_request_payload_llm(
                model_name,
                Some(request),
                annotated,
                None,
            )),
        );
        let response = self.invoke_async(invoke).await?;
        match response.result {
            Some(invoke_response_result::Result::LlmRequest(result)) => {
                let outcome = required_envelope(result.outcome, "llm request intercept outcome")?;
                if outcome.schema != LLM_REQUEST_INTERCEPT_OUTCOME_SCHEMA {
                    return Err(FlowError::Internal(format!(
                        "worker returned unsupported LLM request intercept outcome schema: {}",
                        outcome.schema
                    )));
                }
                decode_json_envelope(&outcome).map_err(|err| {
                    FlowError::Internal(format!(
                        "worker returned invalid LLM request intercept outcome: {err}"
                    ))
                })
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
        next: LlmExecutionNextFn,
    ) -> FlowResult<Json> {
        let continuation_id = self
            .host_state
            .insert_continuation(Continuation::llm(next))?;
        let invoke = self.base_request(
            registration_name,
            RegistrationSurface::LlmExecutionIntercept,
            Some(continuation_id),
            Some(invoke_request_payload_llm(
                model_name,
                Some(request),
                None,
                None,
            )),
        );
        json_from_invoke_response(self.invoke_async(invoke).await?)
    }

    async fn invoke_llm_stream_execution(
        &self,
        registration_name: &str,
        model_name: &str,
        request: LlmRequest,
        next: LlmStreamExecutionNextFn,
    ) -> FlowResult<LlmJsonStream> {
        let continuation_id = self
            .host_state
            .insert_continuation(Continuation::llm_stream(next))?;
        let invoke = self.base_request(
            registration_name,
            RegistrationSurface::LlmStreamExecutionIntercept,
            Some(continuation_id.clone()),
            Some(invoke_request_payload_llm(
                model_name,
                Some(request),
                None,
                None,
            )),
        );
        let mut client = self.client.clone();
        let mut guard = WorkerInvocationGuard::new(self, &invoke);
        let (tx, rx) = mpsc::channel(16);
        let (next_ready_tx, next_ready_rx) = oneshot::channel();
        self.runtime.spawn(async move {
            let result = tokio::select! {
                result = worker_rpc(client.invoke_stream(worker_rpc_request(invoke))) => result,
                _ = tx.closed() => {
                    guard.cancel("host stopped consuming the worker stream");
                    guard.finish();
                    return;
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
                                guard.cancel("host stopped consuming the worker stream");
                                break;
                            }
                        };
                        let Some(item) = item else {
                            break;
                        };
                        let result = match item {
                            Ok(chunk) => json_from_stream_chunk(chunk),
                            Err(err) => Err(FlowError::Internal(format!(
                                "worker stream transport failed: {err}"
                            ))),
                        };
                        if tx.send(result).await.is_err() {
                            guard.cancel("host stopped consuming the worker stream");
                            break;
                        }
                    }
                }
                Err(err) => {
                    let reason = if err.code() == tonic::Code::DeadlineExceeded {
                        "worker stream invocation timed out"
                    } else {
                        "worker stream transport failed"
                    };
                    guard.cancel(reason);
                    let _ = tx
                        .send(Err(worker_status_to_flow(
                            "worker stream invoke failed",
                            err,
                        )))
                        .await;
                }
            }
            guard.finish();
        });
        next_ready_rx.await.map_err(|_| {
            FlowError::Internal(
                "worker stream invocation ended before the downstream stream opened".into(),
            )
        })?;
        Ok(LlmJsonStream::new(
            tokio_stream::wrappers::ReceiverStream::new(rx),
        ))
    }

    fn base_request(
        &self,
        registration_name: &str,
        surface: RegistrationSurface,
        continuation_id: Option<String>,
        payload: Option<invoke_request_payload::Payload>,
    ) -> InvokeRequest {
        let scope_stack_id = self.host_state.insert_invocation_scope_stack(
            current_scope_stack(),
            capture_nested_publication_buffer(),
        );
        InvokeRequest {
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
        }
    }

    fn invoke_blocking(&self, request: InvokeRequest) -> FlowResult<InvokeResponse> {
        block_on_handle(&self.runtime, self.invoke_async(request))
    }

    async fn invoke_async(&self, request: InvokeRequest) -> FlowResult<InvokeResponse> {
        let callback_name = request.registration_name.clone();
        let surface = request.surface;
        let result = self
            .invoke_async_with_timeout(request, WORKER_RPC_TIMEOUT)
            .await;
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

    async fn invoke_async_with_timeout(
        &self,
        request: InvokeRequest,
        timeout: Duration,
    ) -> FlowResult<InvokeResponse> {
        let mut guard = WorkerInvocationGuard::new(self, &request);
        let mut client = self.client.clone();
        let result =
            worker_rpc_with_timeout(timeout, client.invoke(worker_rpc_request(request))).await;
        if result
            .as_ref()
            .is_err_and(|err| err.code() == tonic::Code::DeadlineExceeded)
        {
            guard.cancel("worker invocation timed out");
        }
        guard.finish();
        result
            .map(|response| response.into_inner())
            .map_err(|err| worker_status_to_flow("worker invoke failed", err))
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

struct WorkerHostRuntimeState {
    activation_id: String,
    auth_token: String,
    scope_stacks: Mutex<HashMap<String, StoredScopeStack>>,
    pending_scope_cleanups: Mutex<Vec<PendingScopeCleanup>>,
    scope_stack_cleanups: Mutex<Vec<crate::api::runtime::ScopeStackHandle>>,
    scope_stack_cleanup_complete: Condvar,
    scope_handles: Mutex<HashMap<String, StoredScopeHandle>>,
    continuations: Mutex<HashMap<String, Continuation>>,
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

impl WorkerCodecCapabilityGuard {
    fn new(host_state: Arc<WorkerHostRuntimeState>, capability_id: String) -> Self {
        Self {
            host_state,
            capability_id,
        }
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

struct StoredScopeStack {
    handle: crate::api::runtime::ScopeStackHandle,
    publication_buffer: Option<PublicationBuffer>,
    invocation_base_depth: Option<usize>,
}

struct PendingScopeCleanup {
    handle: crate::api::runtime::ScopeStackHandle,
    base_depth: usize,
}

struct StoredScopeHandle {
    handle: ScopeHandle,
    scope_stack_id: String,
}

fn has_active_scope_stack_alias(
    stacks: &HashMap<String, StoredScopeStack>,
    handle: &crate::api::runtime::ScopeStackHandle,
) -> bool {
    stacks.values().any(|candidate| {
        candidate.invocation_base_depth.is_some() && Arc::ptr_eq(&candidate.handle, handle)
    })
}

struct ScopeStackCleanupGuard<'a> {
    state: &'a WorkerHostRuntimeState,
    handle: crate::api::runtime::ScopeStackHandle,
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
            activation_id,
            auth_token,
            scope_stacks: Mutex::new(HashMap::new()),
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

    fn insert_request_codec(&self, invocation_id: &str, codec: Arc<dyn LlmCodec>) -> String {
        self.insert_codec(invocation_id, WorkerCodecDirection::Request(codec))
    }

    fn insert_response_codec(
        &self,
        invocation_id: &str,
        codec: Arc<dyn LlmResponseCodec>,
    ) -> String {
        self.insert_codec(invocation_id, WorkerCodecDirection::Response(codec))
    }

    fn insert_codec(&self, invocation_id: &str, direction: WorkerCodecDirection) -> String {
        let id = format!("codec-{}", Uuid::now_v7());
        if let Ok(mut codecs) = self.codecs.lock() {
            codecs.insert(
                id.clone(),
                WorkerCodecCapability {
                    invocation_id: invocation_id.to_owned(),
                    direction,
                },
            );
        }
        id
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
            return Err(Status::permission_denied("invalid worker host token"));
        }
        Ok(())
    }

    fn insert_invocation_scope_stack(
        &self,
        stack: crate::api::runtime::ScopeStackHandle,
        publication_buffer: Option<PublicationBuffer>,
    ) -> String {
        let id = format!("invoke-{}", Uuid::now_v7());
        let Ok(mut stacks) = self.scope_stacks.lock() else {
            return id;
        };
        loop {
            let Ok(cleanups) = self.scope_stack_cleanups.lock() else {
                return id;
            };
            if !cleanups.iter().any(|handle| Arc::ptr_eq(handle, &stack)) {
                break;
            }
            drop(stacks);
            let Ok(guard) = self.scope_stack_cleanup_complete.wait(cleanups) else {
                return id;
            };
            drop(guard);
            let Ok(guard) = self.scope_stacks.lock() else {
                return id;
            };
            stacks = guard;
        }
        let Ok(stack_guard) = stack.read() else {
            return id;
        };
        let invocation_base_depth = stack_guard.scopes().len();
        drop(stack_guard);
        stacks.insert(
            id.clone(),
            StoredScopeStack {
                handle: stack,
                publication_buffer,
                invocation_base_depth: Some(invocation_base_depth),
            },
        );
        id
    }

    fn cleanup_invocation_scope_stack(&self, id: &str) {
        let unwind = self.take_invocation_scope_cleanup(id);
        if let Some((handle, base_depth)) = unwind {
            let _cleanup = ScopeStackCleanupGuard {
                state: self,
                handle: handle.clone(),
            };
            Self::unwind_scope_stack(&handle, base_depth);
        }
        if let Ok(mut handles) = self.scope_handles.lock() {
            handles.retain(|_, handle| handle.scope_stack_id != id);
        }
    }

    fn take_invocation_scope_cleanup(
        &self,
        id: &str,
    ) -> Option<(crate::api::runtime::ScopeStackHandle, usize)> {
        let Ok(mut stacks) = self.scope_stacks.lock() else {
            return None;
        };
        let stored = stacks.remove(id)?;
        let mut base_depth = stored.invocation_base_depth?;
        let Ok(mut pending) = self.pending_scope_cleanups.lock() else {
            return None;
        };
        if has_active_scope_stack_alias(&stacks, &stored.handle) {
            pending.push(PendingScopeCleanup {
                handle: stored.handle,
                base_depth,
            });
            return None;
        }
        pending.retain(|cleanup| {
            if Arc::ptr_eq(&cleanup.handle, &stored.handle) {
                base_depth = base_depth.min(cleanup.base_depth);
                false
            } else {
                true
            }
        });
        let Ok(mut cleanups) = self.scope_stack_cleanups.lock() else {
            return None;
        };
        cleanups.push(stored.handle.clone());
        Some((stored.handle, base_depth))
    }

    fn unwind_scope_stack(stack: &crate::api::runtime::ScopeStackHandle, base_depth: usize) {
        loop {
            let top_uuid = {
                let Ok(stack) = stack.read() else {
                    return;
                };
                if stack.scopes().len() <= base_depth {
                    return;
                }
                stack.top().uuid
            };
            let popped = with_scope_stack(stack.clone(), || {
                pop_scope(PopScopeParams::builder().handle_uuid(&top_uuid).build())
            })
            .is_ok();
            if popped {
                continue;
            }
            let Ok(mut stack) = stack.write() else {
                return;
            };
            if stack.remove(&top_uuid).is_err() {
                return;
            }
        }
    }

    fn insert_continuation(&self, continuation: Continuation) -> FlowResult<String> {
        let id = format!("next-{}", Uuid::now_v7());
        let mut continuations = self
            .continuations
            .lock()
            .map_err(|err| FlowError::Internal(format!("continuation lock poisoned: {err}")))?;
        continuations.insert(id.clone(), continuation);
        Ok(id)
    }

    fn remove_continuation(&self, id: &str) {
        if let Ok(mut continuations) = self.continuations.lock() {
            continuations.remove(id);
        }
    }

    fn continuation(&self, id: &str) -> Result<Continuation, Status> {
        self.continuations
            .lock()
            .map_err(|err| Status::internal(format!("continuation lock poisoned: {err}")))?
            .get(id)
            .cloned()
            .ok_or_else(|| Status::not_found("continuation not found"))
    }

    #[cfg(test)]
    fn stack(&self, id: &str) -> Result<Option<crate::api::runtime::ScopeStackHandle>, Status> {
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
            })
            .map(Some)
            .ok_or_else(|| Status::not_found("scope stack not found"))
    }
}

#[derive(Clone)]
struct StoredInvocationContext {
    scope_stack: crate::api::runtime::ScopeStackHandle,
    publication_buffer: Option<PublicationBuffer>,
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
    async fn list_runtime_registrations(
        &self,
        request: Request<ListRuntimeRegistrationsRequest>,
    ) -> Result<Response<ListRuntimeRegistrationsResponse>, Status> {
        let request = request.into_inner();
        self.state
            .authorize(&request.activation_id, &request.auth_token)?;
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
                .map(|registration| ProtoRuntimeRegistrationIdentity {
                    kind: registration_surface_from_kind(registration.kind) as i32,
                    local_name: registration.local_name,
                    effective_name: registration.effective_name,
                    owner: Some(ProtoRuntimeRegistrationOwner {
                        kind: match registration.owner.kind {
                            RuntimeRegistrationOwnerKind::Core => {
                                ProtoRuntimeRegistrationOwnerKind::Core as i32
                            }
                            RuntimeRegistrationOwnerKind::GlobalApi => {
                                ProtoRuntimeRegistrationOwnerKind::GlobalApi as i32
                            }
                            RuntimeRegistrationOwnerKind::Plugin => {
                                ProtoRuntimeRegistrationOwnerKind::Plugin as i32
                            }
                        },
                        plugin_kind: registration.owner.plugin_kind,
                        component_ordinal: registration.owner.component_ordinal,
                    }),
                })
                .collect(),
            error: None,
        }))
    }

    async fn register_conditional_middleware_guardrail(
        &self,
        request: Request<RegisterConditionalMiddlewareGuardrailRequest>,
    ) -> Result<Response<RegisterConditionalMiddlewareGuardrailResponse>, Status> {
        let request = request.into_inner();
        self.state
            .authorize(&request.activation_id, &request.auth_token)?;
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
            RegisterConditionalMiddlewareGuardrailResponse {
                handle,
                error: None,
            },
        ))
    }

    async fn deregister_conditional_middleware_guardrail(
        &self,
        request: Request<DeregisterConditionalMiddlewareGuardrailRequest>,
    ) -> Result<Response<DeregisterConditionalMiddlewareGuardrailResponse>, Status> {
        let request = request.into_inner();
        self.state
            .authorize(&request.activation_id, &request.auth_token)?;
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
            DeregisterConditionalMiddlewareGuardrailResponse {
                removed,
                error: None,
            },
        ))
    }

    async fn emit_mark(
        &self,
        request: Request<EmitMarkRequest>,
    ) -> Result<Response<HostAck>, Status> {
        let EmitMarkRequest {
            activation_id,
            auth_token,
            scope,
            name,
            data,
            metadata,
            data_schema,
            severity,
            category,
        } = request.into_inner();
        self.state.authorize(&activation_id, &auth_token)?;
        let data_schema = optional_typed_envelope::<DataSchema>(
            data_schema,
            "mark data_schema",
            DATA_SCHEMA_SCHEMA,
        );
        let severity = optional_log_severity(&severity);
        let category = (!category.is_empty()).then(|| EventCategory::new(category));
        let result = self.with_stack(scope.as_ref(), || {
            emit_scope_mark(
                EmitMarkEventParams::builder()
                    .name(&name)
                    .data_opt(optional_envelope_to_json(data)?)
                    .metadata_opt(optional_envelope_to_json(metadata)?)
                    .data_schema_opt(data_schema?)
                    .severity_opt(severity?)
                    .category_opt(category)
                    .build(),
            )
        });
        Ok(Response::new(host_ack(result)))
    }

    async fn get_runtime_diagnostics(
        &self,
        request: Request<GetRuntimeDiagnosticsRequest>,
    ) -> Result<Response<GetRuntimeDiagnosticsResponse>, Status> {
        let request = request.into_inner();
        self.state
            .authorize(&request.activation_id, &request.auth_token)?;
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
        self.state
            .authorize(&request.activation_id, &request.auth_token)?;
        let result = self.with_stack(request.scope.as_ref(), || {
            push_scope(
                PushScopeParams::builder()
                    .name(&request.name)
                    .scope_type(proto_scope_type(request.scope_type))
                    .attributes(ScopeAttributes::empty())
                    .data_opt(optional_envelope_to_json(request.data)?)
                    .metadata_opt(optional_envelope_to_json(request.metadata)?)
                    .input_opt(optional_envelope_to_json(request.input)?)
                    .build(),
            )
        });
        match result {
            Ok(handle) => {
                let id = format!("scope-{}", handle.uuid);
                let scope_stack_id = request
                    .scope
                    .as_ref()
                    .map(|scope| scope.scope_stack_id.clone())
                    .unwrap_or_default();
                self.state
                    .scope_handles
                    .lock()
                    .map_err(|err| Status::internal(format!("scope handle lock poisoned: {err}")))?
                    .insert(
                        id.clone(),
                        StoredScopeHandle {
                            handle,
                            scope_stack_id,
                        },
                    );
                Ok(Response::new(PushScopeResponse {
                    scope_handle_id: id,
                    error: None,
                }))
            }
            Err(err) => Ok(Response::new(PushScopeResponse {
                scope_handle_id: String::new(),
                error: Some(flow_error_to_worker(err)),
            })),
        }
    }

    async fn pop_scope(
        &self,
        request: Request<PopScopeRequest>,
    ) -> Result<Response<HostAck>, Status> {
        let request = request.into_inner();
        self.state
            .authorize(&request.activation_id, &request.auth_token)?;
        let handle = self
            .state
            .scope_handles
            .lock()
            .map_err(|err| Status::internal(format!("scope handle lock poisoned: {err}")))?
            .remove(&request.scope_handle_id)
            .ok_or_else(|| Status::not_found("scope handle not found"))?;
        let output = optional_envelope_to_json(request.output).map_err(status_from_flow)?;
        let metadata = optional_envelope_to_json(request.metadata).map_err(status_from_flow)?;
        let pop = || {
            pop_scope(
                PopScopeParams::builder()
                    .handle_uuid(&handle.handle.uuid)
                    .output_opt(output)
                    .metadata_opt(metadata)
                    .build(),
            )
        };
        let result = if handle.scope_stack_id.is_empty() {
            pop()
        } else if let Some(context) = self.state.invocation_context(&handle.scope_stack_id)? {
            with_nested_publication_buffer(context.publication_buffer, || {
                with_scope_stack(context.scope_stack, pop)
            })
        } else {
            pop()
        };
        Ok(Response::new(host_ack(result)))
    }

    async fn create_scope_stack(
        &self,
        request: Request<CreateScopeStackRequest>,
    ) -> Result<Response<CreateScopeStackResponse>, Status> {
        let request = request.into_inner();
        self.state
            .authorize(&request.activation_id, &request.auth_token)?;
        let id = format!("stack-{}", Uuid::now_v7());
        self.state
            .scope_stacks
            .lock()
            .map_err(|err| Status::internal(format!("scope stack lock poisoned: {err}")))?
            .insert(
                id.clone(),
                StoredScopeStack {
                    handle: crate::api::runtime::create_scope_stack(),
                    publication_buffer: None,
                    invocation_base_depth: None,
                },
            );
        Ok(Response::new(CreateScopeStackResponse {
            scope_stack_id: id,
            error: None,
        }))
    }

    async fn drop_scope_stack(
        &self,
        request: Request<DropScopeStackRequest>,
    ) -> Result<Response<HostAck>, Status> {
        let request = request.into_inner();
        self.state
            .authorize(&request.activation_id, &request.auth_token)?;
        self.state
            .scope_stacks
            .lock()
            .map_err(|err| Status::internal(format!("scope stack lock poisoned: {err}")))?
            .remove(&request.scope_stack_id);
        Ok(Response::new(HostAck {
            ok: true,
            error: None,
        }))
    }

    async fn tool_next(
        &self,
        request: Request<ToolNextRequest>,
    ) -> Result<Response<ToolExecutionResultResponse>, Status> {
        let request = request.into_inner();
        self.state
            .authorize(&request.activation_id, &request.auth_token)?;
        let continuation = self.state.continuation(&request.continuation_id)?;
        let Continuation::Tool { next, context } = continuation else {
            return Err(Status::invalid_argument(
                "continuation is not a tool continuation",
            ));
        };
        let context = self.isolated_continuation_context(&context, request.scope.as_ref())?;
        let value =
            required_envelope(request.value, "tool next value").map_err(status_from_flow)?;
        let value = decode_json_envelope::<Json>(&value)
            .map_err(|err| Status::invalid_argument(format!("invalid tool next JSON: {err}")))?;
        let result = AssertUnwindSafe(context.invoke(move || next(value)))
            .catch_unwind()
            .await
            .unwrap_or_else(|payload| {
                Err(FlowError::Internal(format!(
                    "worker tool continuation panicked: {}",
                    panic_payload_message(payload.as_ref())
                )))
            });
        Ok(Response::new(tool_execution_result_response(result)))
    }

    async fn llm_next(
        &self,
        request: Request<LlmNextRequest>,
    ) -> Result<Response<JsonResult>, Status> {
        let request = request.into_inner();
        self.state
            .authorize(&request.activation_id, &request.auth_token)?;
        let continuation = self.state.continuation(&request.continuation_id)?;
        let Continuation::Llm { next, context } = continuation else {
            return Err(Status::invalid_argument(
                "continuation is not an LLM continuation",
            ));
        };
        let context = self.isolated_continuation_context(&context, request.scope.as_ref())?;
        let request =
            required_envelope(request.request, "llm next request").map_err(status_from_flow)?;
        let request = decode_json_envelope::<LlmRequest>(&request)
            .map_err(|err| Status::invalid_argument(format!("invalid LLM next request: {err}")))?;
        let result = AssertUnwindSafe(context.invoke(move || next(request)))
            .catch_unwind()
            .await
            .unwrap_or_else(|payload| {
                Err(FlowError::Internal(format!(
                    "worker LLM continuation panicked: {}",
                    panic_payload_message(payload.as_ref())
                )))
            });
        Ok(Response::new(json_result(result)))
    }

    type LlmStreamNextStream =
        Pin<Box<dyn tokio_stream::Stream<Item = Result<StreamChunk, Status>> + Send>>;

    async fn llm_stream_next(
        &self,
        request: Request<LlmStreamNextRequest>,
    ) -> Result<Response<Self::LlmStreamNextStream>, Status> {
        let request = request.into_inner();
        self.state
            .authorize(&request.activation_id, &request.auth_token)?;
        let continuation = self.state.continuation(&request.continuation_id)?;
        let Continuation::LlmStream { next, context } = continuation else {
            return Err(Status::invalid_argument(
                "continuation is not an LLM stream continuation",
            ));
        };
        let context = self.isolated_continuation_context(&context, request.scope.as_ref())?;
        let request = required_envelope(request.request, "llm stream next request")
            .map_err(status_from_flow)?;
        let request = decode_json_envelope::<LlmRequest>(&request).map_err(|err| {
            Status::invalid_argument(format!("invalid LLM stream next request: {err}"))
        })?;
        let stream = AssertUnwindSafe(context.invoke(move || next(request)))
            .catch_unwind()
            .await
            .map_err(|payload| {
                Status::internal(format!(
                    "worker stream continuation panicked: {}",
                    panic_payload_message(payload.as_ref())
                ))
            })?
            .map_err(status_from_flow)?;
        let (tx, rx) = mpsc::channel(16);
        tokio::spawn(async move {
            context
                .run(async move {
                    let mut stream = stream;
                    loop {
                        tokio::select! {
                            biased;
                            _ = tx.closed() => break,
                            item = AssertUnwindSafe(stream.next()).catch_unwind() => {
                                match item {
                                    Ok(Some(item)) => {
                                        if tx.send(item).await.is_err() {
                                            break;
                                        }
                                    }
                                    Ok(None) => break,
                                    Err(payload) => {
                                        let _ = tx
                                            .send(Err(FlowError::Internal(format!(
                                                "worker stream continuation panicked: {}",
                                                panic_payload_message(payload.as_ref())
                                            ))))
                                            .await;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                })
                .await;
        });
        let mapped = tokio_stream::wrappers::ReceiverStream::new(rx).map(|item| match item {
            Ok(value) => Ok(StreamChunk {
                item: Some(stream_chunk_item::Item::Value(json_envelope_infallible(
                    JSON_SCHEMA,
                    &value,
                ))),
            }),
            Err(err) => Ok(StreamChunk {
                item: Some(stream_chunk_item::Item::Error(flow_error_to_worker(err))),
            }),
        });
        Ok(Response::new(Box::pin(mapped)))
    }

    async fn decode_llm_codec_request(
        &self,
        request: Request<LlmCodecDecodeRequest>,
    ) -> Result<Response<JsonResult>, Status> {
        let request = request.into_inner();
        self.state
            .authorize(&request.activation_id, &request.auth_token)?;
        let codec = self
            .state
            .request_codec(&request.codec_capability_id, &request.invocation_id)?;
        let request = required_envelope(request.request, "codec request")
            .and_then(|value| {
                decode_json_envelope::<LlmRequest>(&value)
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
        request: Request<LlmCodecEncodeRequest>,
    ) -> Result<Response<JsonResult>, Status> {
        let request = request.into_inner();
        self.state
            .authorize(&request.activation_id, &request.auth_token)?;
        let codec = self
            .state
            .request_codec(&request.codec_capability_id, &request.invocation_id)?;
        let annotated = required_envelope(request.annotated_request, "annotated codec request")
            .and_then(|value| {
                decode_json_envelope::<AnnotatedLlmRequest>(&value)
                    .map_err(|err| FlowError::Internal(err.to_string()))
            })
            .map_err(status_from_flow)?;
        let original = required_envelope(request.original_request, "original codec request")
            .and_then(|value| {
                decode_json_envelope::<LlmRequest>(&value)
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
        request: Request<LlmCodecDecodeResponse>,
    ) -> Result<Response<JsonResult>, Status> {
        let request = request.into_inner();
        self.state
            .authorize(&request.activation_id, &request.auth_token)?;
        let codec = self
            .state
            .response_codec(&request.codec_capability_id, &request.invocation_id)?;
        let response = required_envelope(request.response, "codec response")
            .and_then(|value| {
                decode_json_envelope::<Json>(&value)
                    .map_err(|err| FlowError::Internal(err.to_string()))
            })
            .map_err(status_from_flow)?;
        Ok(Response::new(json_result(
            codec.decode_response(&response).and_then(|value| {
                serde_json::to_value(value).map_err(|err| FlowError::Internal(err.to_string()))
            }),
        )))
    }
}

impl WorkerHostRuntimeService {
    fn isolated_continuation_context(
        &self,
        context: &MiddlewareContinuationContext,
        scope: Option<&ScopeContext>,
    ) -> Result<MiddlewareContinuationContext, Status> {
        let Some(stack_id) = scope
            .map(|scope| scope.scope_stack_id.as_str())
            .filter(|stack_id| !stack_id.is_empty())
        else {
            return context.isolated().map_err(status_from_flow);
        };
        let selected = self
            .state
            .invocation_context(stack_id)?
            .ok_or_else(|| Status::not_found("scope stack not found"))?;
        context
            .isolated_with_scope_stack(&selected.scope_stack)
            .map_err(status_from_flow)
    }

    fn with_stack<T>(
        &self,
        scope: Option<&ScopeContext>,
        f: impl FnOnce() -> FlowResult<T>,
    ) -> FlowResult<T> {
        let Some(stack_id) = scope.map(|scope| scope.scope_stack_id.as_str()) else {
            return f();
        };
        let Some(context) = self
            .state
            .invocation_context(stack_id)
            .map_err(|err| FlowError::Internal(err.to_string()))?
        else {
            return f();
        };
        with_nested_publication_buffer(context.publication_buffer, || {
            with_scope_stack(context.scope_stack, f)
        })
    }
}

mod invoke_request_payload {
    pub(crate) use nemo_relay_worker_proto::v1::invoke_request::Payload;
}

mod llm_invocation {
    pub(crate) use nemo_relay_worker_proto::v1::llm_invocation::SanitizeContext;
}

mod invoke_response_result {
    pub(crate) use nemo_relay_worker_proto::v1::invoke_response::Result;
}

mod stream_chunk_item {
    pub(crate) use nemo_relay_worker_proto::v1::stream_chunk::Item;
}

fn invoke_request_payload_event(event: &Event) -> invoke_request_payload::Payload {
    invoke_request_payload::Payload::Event(json_envelope_infallible(EVENT_SCHEMA, event))
}

fn invoke_request_payload_tool(tool_name: &str, value: Json) -> invoke_request_payload::Payload {
    invoke_request_payload::Payload::Tool(ToolInvocation {
        tool_name: tool_name.into(),
        value: Some(json_envelope_infallible(JSON_SCHEMA, &value)),
    })
}

fn invoke_request_payload_llm(
    model_name: &str,
    request: Option<LlmRequest>,
    annotated_request: Option<AnnotatedLlmRequest>,
    response: Option<Json>,
) -> invoke_request_payload::Payload {
    invoke_request_payload::Payload::Llm(LlmInvocation {
        model_name: model_name.into(),
        request: request
            .as_ref()
            .map(|request| json_envelope_infallible(LLM_REQUEST_SCHEMA, request)),
        annotated_request: annotated_request
            .as_ref()
            .map(|request| json_envelope_infallible(ANNOTATED_LLM_REQUEST_SCHEMA, request)),
        response: response
            .as_ref()
            .map(|response| json_envelope_infallible(JSON_SCHEMA, response)),
        sanitize_context: None,
    })
}

fn invoke_request_payload_llm_context(
    model_name: &str,
    request: Option<LlmRequest>,
    annotated_request: Option<AnnotatedLlmRequest>,
    response: Option<Json>,
    context: impl Into<llm_invocation::SanitizeContext>,
) -> invoke_request_payload::Payload {
    invoke_request_payload::Payload::Llm(LlmInvocation {
        model_name: model_name.into(),
        request: request
            .as_ref()
            .map(|request| json_envelope_infallible(LLM_REQUEST_SCHEMA, request)),
        annotated_request: annotated_request
            .as_ref()
            .map(|request| json_envelope_infallible(ANNOTATED_LLM_REQUEST_SCHEMA, request)),
        response: response
            .as_ref()
            .map(|response| json_envelope_infallible(JSON_SCHEMA, response)),
        sanitize_context: Some(context.into()),
    })
}

fn codec_identity_to_proto(identity: &LlmCodecIdentity) -> ProtoLlmCodecIdentity {
    let (kind, id) = match identity {
        LlmCodecIdentity::None => (LlmCodecKind::Unspecified as i32, None),
        LlmCodecIdentity::BuiltIn(codec) => {
            (LlmCodecKind::Builtin as i32, Some(codec.id().to_owned()))
        }
        LlmCodecIdentity::Runtime(id) => (LlmCodecKind::Runtime as i32, Some(id.clone())),
        LlmCodecIdentity::Opaque => (LlmCodecKind::Opaque as i32, None),
    };
    ProtoLlmCodecIdentity { kind, id }
}

fn json_envelope_infallible<T: serde::Serialize>(schema: &str, value: &T) -> JsonEnvelope {
    json_envelope(schema, value).expect("Relay DTO JSON serialization should be infallible")
}

fn json_from_invoke_response(response: InvokeResponse) -> FlowResult<Json> {
    match response.result {
        Some(invoke_response_result::Result::Json(result)) => {
            if let Some(error) = result.error {
                return Err(worker_error_to_flow(error));
            }
            let envelope = required_envelope(result.value, "worker JSON result")?;
            decode_json_envelope::<Json>(&envelope).map_err(|err| {
                FlowError::Internal(format!("worker returned invalid JSON result: {err}"))
            })
        }
        Some(invoke_response_result::Result::Error(error)) => Err(worker_error_to_flow(error)),
        _ => Err(FlowError::Internal(
            "worker returned unexpected invoke result".into(),
        )),
    }
}

fn optional_json_from_invoke_response(response: InvokeResponse) -> FlowResult<Option<Json>> {
    match response.result {
        Some(invoke_response_result::Result::Empty(_)) => Ok(None),
        Some(invoke_response_result::Result::Json(result)) => {
            if let Some(error) = result.error {
                return Err(worker_error_to_flow(error));
            }
            let envelope = required_envelope(result.value, "worker JSON result")?;
            decode_json_envelope::<Json>(&envelope)
                .map(Some)
                .map_err(|err| {
                    FlowError::Internal(format!("worker returned invalid JSON result: {err}"))
                })
        }
        Some(invoke_response_result::Result::Error(error)) => Err(worker_error_to_flow(error)),
        _ => Err(FlowError::Internal(
            "worker returned unexpected LLM sanitizer result".into(),
        )),
    }
}

fn guardrail_from_invoke_response(response: InvokeResponse) -> FlowResult<Option<String>> {
    match response.result {
        Some(invoke_response_result::Result::Guardrail(GuardrailResult { block_reason })) => {
            Ok((!block_reason.is_empty()).then_some(block_reason))
        }
        Some(invoke_response_result::Result::Error(error)) => Err(worker_error_to_flow(error)),
        _ => Err(FlowError::Internal(
            "worker guardrail returned unexpected invoke result".into(),
        )),
    }
}

fn json_from_stream_chunk(chunk: StreamChunk) -> FlowResult<Json> {
    match chunk.item {
        Some(stream_chunk_item::Item::Value(value)) => decode_json_envelope::<Json>(&value)
            .map_err(|err| FlowError::Internal(format!("invalid worker stream chunk: {err}"))),
        Some(stream_chunk_item::Item::Error(error)) => Err(worker_error_to_flow(error)),
        None => Err(FlowError::Internal("worker stream chunk was empty".into())),
    }
}

fn required_envelope(value: Option<JsonEnvelope>, field: &str) -> FlowResult<JsonEnvelope> {
    value.ok_or_else(|| FlowError::Internal(format!("{field} is missing")))
}

fn optional_envelope_to_json(value: Option<JsonEnvelope>) -> FlowResult<Option<Json>> {
    value
        .map(|value| {
            decode_json_envelope::<Json>(&value)
                .map_err(|err| FlowError::Internal(format!("invalid JSON envelope: {err}")))
        })
        .transpose()
}

fn optional_typed_envelope<T: serde::de::DeserializeOwned>(
    value: Option<JsonEnvelope>,
    field: &str,
    expected_schema: &str,
) -> FlowResult<Option<T>> {
    value
        .map(|value| {
            if value.schema != expected_schema {
                return Err(FlowError::InvalidArgument(format!(
                    "{field} has schema {:?}; expected {expected_schema:?}",
                    value.schema
                )));
            }
            decode_json_envelope::<T>(&value).map_err(|err| {
                FlowError::InvalidArgument(format!("{field} has an invalid value: {err}"))
            })
        })
        .transpose()
}

fn optional_log_severity(value: &str) -> FlowResult<Option<LogSeverity>> {
    if value.is_empty() {
        return Ok(None);
    }
    serde_json::from_value(Json::String(value.to_owned()))
        .map(Some)
        .map_err(|err| FlowError::InvalidArgument(format!("mark severity is invalid: {err}")))
}

fn host_ack(result: FlowResult<()>) -> HostAck {
    match result {
        Ok(()) => HostAck {
            ok: true,
            error: None,
        },
        Err(err) => HostAck {
            ok: false,
            error: Some(flow_error_to_worker(err)),
        },
    }
}

fn json_result(result: FlowResult<Json>) -> JsonResult {
    match result {
        Ok(value) => JsonResult {
            value: Some(json_envelope_infallible(JSON_SCHEMA, &value)),
            error: None,
        },
        Err(err) => JsonResult {
            value: None,
            error: Some(flow_error_to_worker(err)),
        },
    }
}

fn typed_json_result<T: serde::Serialize>(schema: &str, result: FlowResult<T>) -> JsonResult {
    match result {
        Ok(value) => JsonResult {
            value: Some(json_envelope_infallible(schema, &value)),
            error: None,
        },
        Err(err) => JsonResult {
            value: None,
            error: Some(flow_error_to_worker(err)),
        },
    }
}

fn tool_execution_result_response(
    result: FlowResult<ToolExecutionResult>,
) -> ToolExecutionResultResponse {
    match result {
        Ok(value) => match tool_execution_result_to_proto(value) {
            Ok(value) => ToolExecutionResultResponse {
                value: Some(value),
                error: None,
            },
            Err(err) => ToolExecutionResultResponse {
                value: None,
                error: Some(flow_error_to_worker(FlowError::Internal(format!(
                    "failed to encode tool execution result: {err}"
                )))),
            },
        },
        Err(err) => ToolExecutionResultResponse {
            value: None,
            error: Some(flow_error_to_worker(err)),
        },
    }
}

fn tool_execution_result_to_proto(
    value: ToolExecutionResult,
) -> std::result::Result<ProtoToolExecutionResult, serde_json::Error> {
    Ok(ProtoToolExecutionResult {
        result: Some(json_value(&value.result)?),
        annotation: value
            .annotation
            .as_ref()
            .filter(|value| !value.is_null())
            .map(json_value)
            .transpose()?,
    })
}

fn tool_execution_intercept_outcome_from_proto(
    value: ProtoToolExecutionInterceptOutcome,
) -> FlowResult<ToolExecutionInterceptOutcome> {
    let result = value.result.ok_or_else(|| {
        FlowError::Internal("worker tool execution intercept outcome result is missing".into())
    })?;
    let result = decode_json_value(&result).map_err(|err| {
        FlowError::Internal(format!("invalid worker tool execution result JSON: {err}"))
    })?;
    let annotation = value
        .annotation
        .as_ref()
        .map(decode_json_value)
        .transpose()
        .map_err(|err| {
            FlowError::Internal(format!(
                "invalid worker tool execution annotation JSON: {err}"
            ))
        })?
        .filter(|value: &Json| !value.is_null());
    let pending_marks = value
        .pending_marks
        .as_ref()
        .map(decode_json_value)
        .transpose()
        .map_err(|err| FlowError::Internal(format!("invalid worker pending marks JSON: {err}")))?
        .unwrap_or_default();
    Ok(ToolExecutionInterceptOutcome {
        result,
        annotation,
        pending_marks,
    })
}

fn flow_error_to_worker(err: FlowError) -> WorkerError {
    WorkerError {
        code: "host.runtime_error".into(),
        message: err.to_string(),
        retryable: false,
    }
}

fn worker_error_to_flow(error: WorkerError) -> FlowError {
    if error.code == "worker.cancelled" {
        FlowError::Internal(format!("worker invocation cancelled: {}", error.message))
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
    Status::internal(err.to_string())
}

fn proto_scope_type(scope_type: i32) -> ScopeType {
    match nemo_relay_worker_proto::v1::ScopeType::try_from(scope_type) {
        Ok(nemo_relay_worker_proto::v1::ScopeType::Agent) => ScopeType::Agent,
        Ok(nemo_relay_worker_proto::v1::ScopeType::Function) => ScopeType::Function,
        Ok(nemo_relay_worker_proto::v1::ScopeType::Tool) => ScopeType::Tool,
        Ok(nemo_relay_worker_proto::v1::ScopeType::Llm) => ScopeType::Llm,
        Ok(nemo_relay_worker_proto::v1::ScopeType::Retriever) => ScopeType::Retriever,
        Ok(nemo_relay_worker_proto::v1::ScopeType::Embedder) => ScopeType::Embedder,
        Ok(nemo_relay_worker_proto::v1::ScopeType::Reranker) => ScopeType::Reranker,
        Ok(nemo_relay_worker_proto::v1::ScopeType::Guardrail) => ScopeType::Guardrail,
        Ok(nemo_relay_worker_proto::v1::ScopeType::Evaluator) => ScopeType::Evaluator,
        Ok(nemo_relay_worker_proto::v1::ScopeType::Custom) => ScopeType::Custom,
        Ok(nemo_relay_worker_proto::v1::ScopeType::Unknown) => ScopeType::Unknown,
        _ => ScopeType::Custom,
    }
}

fn validate_registration_plan(
    plugin_id: &str,
    response: &RegisterResponse,
) -> crate::plugin::Result<()> {
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

fn validate_worker_handshake(
    expected_plugin_id: &str,
    handshake: &HandshakeResponse,
) -> crate::plugin::Result<()> {
    if handshake.plugin_id != expected_plugin_id || handshake.plugin_kind != expected_plugin_id {
        return Err(PluginError::InvalidConfig(format!(
            "worker plugin returned id '{}' kind '{}' but manifest id is '{}'",
            handshake.plugin_id, handshake.plugin_kind, expected_plugin_id
        )));
    }
    if handshake.worker_protocol != WORKER_PROTOCOL_GRPC_V1 {
        return Err(PluginError::InvalidConfig(format!(
            "unsupported worker_protocol '{}'",
            handshake.worker_protocol
        )));
    }
    Ok(())
}

fn validate_relay_compatibility(relay: Option<&str>) -> crate::plugin::Result<()> {
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
#[path = "../../../tests/unit/dynamic_worker_tests.rs"]
mod tests;
