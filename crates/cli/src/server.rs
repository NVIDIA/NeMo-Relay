// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::future::Future;
use std::net::SocketAddr;
use std::path::Path;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::rejection::JsonRejection;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use nemo_relay::plugin::dynamic::{
    DynamicPluginKind, NativePluginActivation, NativePluginLoadSpec, WorkerPluginActivation,
    WorkerPluginLoadSpec, load_native_plugins, load_worker_plugins,
};
use nemo_relay::plugin::{
    PluginComponentSpec, PluginConfig, clear_plugin_configuration, initialize_plugins_exact,
};
use nemo_relay_adaptive::plugin_component::register_adaptive_component;
use nemo_relay_pii_redaction::component::register_pii_redaction_component;
use reqwest::Client;
use serde_json::Value;
use subtle::ConstantTimeEq;
use tokio::net::TcpListener;
use tokio::sync::oneshot;

use crate::adapters::{claude_code, codex, hermes};
use crate::config::{BootstrapChallengeKey, GatewayConfig, ManagedBootstrapIdentity};
use crate::error::CliError;
use crate::gateway;
use crate::plugins::lifecycle::{ActiveDynamicPluginComponent, DynamicPluginActivationSnapshot};
use crate::session::SessionManager;

const HTTP_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const HTTP_REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
const HTTP_READ_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Clone)]
pub(crate) struct AppState {
    pub(crate) config: GatewayConfig,
    pub(crate) bootstrap_fingerprint: Option<String>,
    pub(crate) bootstrap_challenge_key: Option<BootstrapChallengeKey>,
    pub(crate) http: Client,
    pub(crate) sessions: SessionManager,
    pub(crate) last_activity: Arc<Mutex<Instant>>,
    pub(crate) bootstrap_shutdown: Option<BootstrapShutdown>,
    pub(crate) instance_id: String,
}

#[derive(Clone)]
pub(crate) struct BootstrapShutdown {
    token: String,
    sender: Arc<Mutex<Option<oneshot::Sender<()>>>>,
}

/// Binds the configured address and activates enabled dynamic plugins before serving.
pub(crate) async fn serve_with_dynamic(
    config: GatewayConfig,
    dynamic_plugins: Vec<ActiveDynamicPluginComponent>,
    managed_bootstrap: Option<ManagedBootstrapIdentity>,
    ready_file: Option<&Path>,
) -> Result<(), CliError> {
    let listener = bind_listener(config.bind).await?;
    print_startup_status(listener.local_addr()?, &config);
    let bootstrap_fingerprint = managed_bootstrap
        .as_ref()
        .map(|identity| identity.fingerprint().to_owned());
    serve_listener_with_dynamic_inner(
        listener,
        config,
        dynamic_plugins,
        bootstrap_fingerprint,
        managed_bootstrap,
        Some(ShutdownMode::ProcessSignal),
        ready_file,
    )
    .await
}

/// Binds a gateway listener and translates address conflicts into actionable diagnostics.
pub(crate) async fn bind_listener(bind: SocketAddr) -> Result<TcpListener, CliError> {
    TcpListener::bind(bind).await.map_err(|err| {
        // Translate the common bind-failure (port already in use) into an actionable message.
        // Plain `io error: Address already in use (os error 48)` is unhelpful; the friendly
        // version names the likely cause and points at the real fixes.
        if err.kind() == std::io::ErrorKind::AddrInUse {
            CliError::Launch(format!(
                "cannot bind {} — port is already in use. Most likely cause: another \
                 `nemo-relay` daemon is already running. Fix one of:\n  \
                 • stop the running daemon (Unix: `pkill -f nemo-relay`, Windows: \
                 `taskkill /IM nemo-relay.exe`)\n  \
                 • use an ephemeral port: `nemo-relay --bind 127.0.0.1:0`\n  \
                 • pick a free port: `nemo-relay --bind 127.0.0.1:4041`",
                bind
            ))
        } else {
            CliError::Io(err)
        }
    })
}

pub(crate) fn print_startup_status(bind: SocketAddr, config: &GatewayConfig) {
    let use_color = std::io::IsTerminal::is_terminal(&std::io::stderr())
        && std::env::var_os("NO_COLOR").is_none();
    eprint!("{}", render_startup_status(bind, config, use_color));
}

fn render_startup_status(bind: SocketAddr, config: &GatewayConfig, color: bool) -> String {
    let mut lines = vec![
        "NeMo Relay".to_string(),
        format!("  Gateway        http://{bind}"),
    ];
    let destinations = crate::launcher::exporter_destinations(config);
    if destinations.is_empty() {
        lines.push("  Exporters      not configured".into());
    } else {
        for (index, destination) in destinations.iter().enumerate() {
            lines.push(format!(
                "  {}{}",
                if index == 0 {
                    "Exporters      "
                } else {
                    "               "
                },
                destination
            ));
        }
    }

    crate::launcher::render_status_frame(&lines, color)
}

/// Serves the gateway router on a caller-owned listener with optional graceful shutdown.
///
/// A provided shutdown receiver is best-effort: the send side may be dropped after the child agent
/// exits, and either receiving or channel closure is enough to let Axum drain the listener.
#[cfg(test)]
pub(crate) async fn serve_listener(
    listener: TcpListener,
    config: GatewayConfig,
    shutdown: Option<oneshot::Receiver<()>>,
) -> Result<(), CliError> {
    serve_listener_with_dynamic(listener, config, Vec::new(), shutdown).await
}

#[cfg(test)]
pub(crate) async fn serve_listener_with_bootstrap(
    listener: TcpListener,
    config: GatewayConfig,
    bootstrap_fingerprint: String,
    shutdown: Option<oneshot::Receiver<()>>,
) -> Result<(), CliError> {
    serve_listener_with_dynamic_inner(
        listener,
        config,
        Vec::new(),
        Some(bootstrap_fingerprint),
        None,
        shutdown.map(ShutdownMode::Receiver),
        None,
    )
    .await
}

/// Serves the gateway router and activates enabled dynamic plugin components.
pub(crate) async fn serve_listener_with_dynamic(
    listener: TcpListener,
    config: GatewayConfig,
    dynamic_plugins: Vec<ActiveDynamicPluginComponent>,
    shutdown: Option<oneshot::Receiver<()>>,
) -> Result<(), CliError> {
    serve_listener_with_dynamic_inner(
        listener,
        config,
        dynamic_plugins,
        None,
        None,
        shutdown.map(ShutdownMode::Receiver),
        None,
    )
    .await
}

type ShutdownFuture = Pin<Box<dyn Future<Output = ()> + Send>>;

enum ShutdownMode {
    Receiver(oneshot::Receiver<()>),
    ProcessSignal,
}

async fn serve_listener_with_dynamic_inner(
    listener: TcpListener,
    config: GatewayConfig,
    dynamic_plugins: Vec<ActiveDynamicPluginComponent>,
    bootstrap_fingerprint: Option<String>,
    managed_bootstrap: Option<ManagedBootstrapIdentity>,
    shutdown_mode: Option<ShutdownMode>,
    ready_file: Option<&Path>,
) -> Result<(), CliError> {
    let bootstrap_challenge_key = bootstrap_fingerprint
        .as_ref()
        .map(|_| BootstrapChallengeKey::load())
        .transpose()?;
    let plugin_activation =
        PluginActivation::initialize(config.plugin_config.clone(), dynamic_plugins).await?;
    let (bootstrap_shutdown, bootstrap_shutdown_rx) = bootstrap_shutdown_channel();
    let state = AppState::new_with_bootstrap(
        config,
        bootstrap_fingerprint,
        bootstrap_challenge_key,
        bootstrap_shutdown,
    );
    let instance_id = state.instance_id.clone();
    let sessions = state.sessions.clone();
    let last_activity = state.last_activity.clone();
    let app = router_with_state(state);
    let local_address = listener.local_addr()?;
    if let Some(identity) = managed_bootstrap.as_ref() {
        identity.verify_current()?;
    }
    crate::sidecar::publish_sidecar_owner_from_env(local_address).map_err(CliError::Launch)?;
    if let Some(path) = ready_file {
        write_ready_file(path, local_address, &instance_id)?;
    }
    let idle_shutdown = if matches!(&shutdown_mode, None | Some(ShutdownMode::ProcessSignal)) {
        plugin_idle_timeout()?
            .map(|timeout| idle_shutdown_future(last_activity, sessions.clone(), timeout))
    } else {
        None
    };
    let shutdown: Option<ShutdownFuture> = match shutdown_mode {
        Some(ShutdownMode::Receiver(receiver)) => Some(Box::pin(async move {
            let _ = receiver.await;
        })),
        Some(ShutdownMode::ProcessSignal) => Some(Box::pin(async move {
            if let Some(idle) = idle_shutdown {
                tokio::select! {
                    _ = shutdown_signal() => {}
                    _ = idle => {}
                }
            } else {
                shutdown_signal().await;
            }
        })),
        None => idle_shutdown.map(|idle| Box::pin(idle) as ShutdownFuture),
    };
    let shutdown = match (shutdown, bootstrap_shutdown_rx) {
        (Some(shutdown), Some(receiver)) => Some(Box::pin(async move {
            tokio::select! {
                _ = shutdown => {}
                _ = receiver => {}
            }
        }) as ShutdownFuture),
        (None, Some(receiver)) => Some(Box::pin(async move {
            let _ = receiver.await;
        }) as ShutdownFuture),
        (shutdown, None) => shutdown,
    };
    let serve_result = match shutdown {
        Some(shutdown) => {
            axum::serve(listener, app)
                .with_graceful_shutdown(shutdown)
                .await
        }
        None => axum::serve(listener, app).await,
    };
    let close_result = sessions.close_all("gateway_shutdown").await;
    let flush_result = nemo_relay::api::runtime::flush_subscribers().map_err(CliError::from);
    let clear_result = plugin_activation.clear();
    if let Err(serve_error) = serve_result {
        if let Err(close_error) = close_result {
            eprintln!("session teardown failed after server error: {close_error}");
        }
        if let Err(flush_error) = flush_result {
            eprintln!("subscriber flush failed after server error: {flush_error}");
        }
        if let Err(clear_error) = clear_result {
            eprintln!("plugin teardown failed after server error: {clear_error}");
        }
        return Err(serve_error.into());
    }
    close_result?;
    flush_result?;
    clear_result
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("installing SIGTERM handler should succeed");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = terminate.recv() => {}
        }
    }

    #[cfg(windows)]
    {
        let mut ctrl_shutdown = tokio::signal::windows::ctrl_shutdown()
            .expect("installing Windows shutdown handler should succeed");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = ctrl_shutdown.recv() => {}
        }
    }

    #[cfg(not(any(unix, windows)))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

/// Builds the gateway HTTP router and shared state.
///
/// Hook endpoints normalize agent-specific payloads into session events, while gateway endpoints
/// proxy model traffic and emit LLM runtime events against the same `SessionManager`.
#[cfg(test)]
pub(crate) fn router(config: GatewayConfig) -> Router {
    router_with_state(AppState::new(config))
}

impl AppState {
    #[cfg(test)]
    fn new(config: GatewayConfig) -> Self {
        Self::new_with_bootstrap(config, None, None, None)
    }

    fn new_with_bootstrap(
        config: GatewayConfig,
        bootstrap_fingerprint: Option<String>,
        bootstrap_challenge_key: Option<BootstrapChallengeKey>,
        bootstrap_shutdown: Option<BootstrapShutdown>,
    ) -> Self {
        let sessions = SessionManager::new(config.clone());
        sessions.start_idle_sweeper();
        let http = Client::builder()
            .connect_timeout(HTTP_CONNECT_TIMEOUT)
            .timeout(HTTP_REQUEST_TIMEOUT)
            .read_timeout(HTTP_READ_TIMEOUT)
            .build()
            .expect("gateway HTTP client configuration is valid");
        Self {
            config,
            bootstrap_fingerprint,
            bootstrap_challenge_key,
            http,
            sessions,
            last_activity: Arc::new(Mutex::new(Instant::now())),
            bootstrap_shutdown,
            instance_id: uuid::Uuid::now_v7().to_string(),
        }
    }

    pub(crate) fn touch(&self) {
        if let Ok(mut last_activity) = self.last_activity.lock() {
            *last_activity = Instant::now();
        }
    }
}

fn router_with_state(state: AppState) -> Router {
    let max_hook_payload_bytes = state.config.max_hook_payload_bytes;
    Router::new()
        .route("/healthz", get(healthz))
        .route("/bootstrap/shutdown", post(shutdown_bootstrap_sidecar))
        .route("/hooks/codex", post(codex_hook))
        .route("/hooks/claude-code", post(claude_code_hook))
        .route("/hooks/hermes", post(hermes_hook))
        .route("/responses", post(gateway::passthrough))
        .route("/chat/completions", post(gateway::passthrough))
        .route("/models", get(gateway::models))
        .route("/v1/responses", post(gateway::passthrough))
        .route("/v1/chat/completions", post(gateway::passthrough))
        .route("/v1/messages", post(gateway::passthrough))
        .route("/v1/messages/count_tokens", post(gateway::passthrough))
        .route("/v1/models", get(gateway::models))
        .layer(DefaultBodyLimit::max(max_hook_payload_bytes))
        .with_state(state)
}

fn bootstrap_shutdown_channel() -> (Option<BootstrapShutdown>, Option<oneshot::Receiver<()>>) {
    let Some(token) = std::env::var("NEMO_RELAY_BOOTSTRAP_SHUTDOWN_TOKEN")
        .ok()
        .filter(|token| !token.is_empty())
    else {
        return (None, None);
    };
    let (sender, receiver) = oneshot::channel();
    (
        Some(BootstrapShutdown {
            token,
            sender: Arc::new(Mutex::new(Some(sender))),
        }),
        Some(receiver),
    )
}

async fn shutdown_bootstrap_sidecar(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> StatusCode {
    let Some(shutdown) = state.bootstrap_shutdown.as_ref() else {
        return StatusCode::NOT_FOUND;
    };
    if headers
        .get("x-nemo-relay-bootstrap-token")
        .and_then(|value| value.to_str().ok())
        != Some(shutdown.token.as_str())
    {
        return StatusCode::FORBIDDEN;
    }
    let Ok(mut sender) = shutdown.sender.lock() else {
        return StatusCode::INTERNAL_SERVER_ERROR;
    };
    let Some(sender) = sender.take() else {
        return StatusCode::GONE;
    };
    let _ = sender.send(());
    StatusCode::NO_CONTENT
}

async fn healthz(State(state): State<AppState>, headers: HeaderMap) -> Response {
    let presented_fingerprint = headers
        .get("x-nemo-relay-bootstrap-fingerprint")
        .and_then(|value| value.to_str().ok());
    let mut response_headers = HeaderMap::new();
    let compatible = match presented_fingerprint {
        None => true,
        Some(expected) => {
            let fingerprint_matches = state
                .bootstrap_fingerprint
                .as_deref()
                .is_some_and(|actual| bool::from(actual.as_bytes().ct_eq(expected.as_bytes())));
            let nonce = headers
                .get("x-nemo-relay-bootstrap-nonce")
                .and_then(|value| value.to_str().ok())
                .filter(|nonce| {
                    nonce.len() == 64 && nonce.bytes().all(|byte| byte.is_ascii_hexdigit())
                });
            match (
                fingerprint_matches,
                nonce,
                state.bootstrap_challenge_key.as_ref(),
            ) {
                (true, Some(nonce), Some(key)) => {
                    let proof = key.proof(expected, nonce);
                    response_headers.insert(
                        "x-nemo-relay-bootstrap-proof",
                        HeaderValue::from_str(&proof).expect("bootstrap proof is an ASCII value"),
                    );
                    state.touch();
                    true
                }
                _ => false,
            }
        }
    };
    (
        if compatible {
            StatusCode::OK
        } else {
            StatusCode::CONFLICT
        },
        response_headers,
        Json(serde_json::json!({
            "status": if compatible { "ok" } else { "incompatible" },
            "service": "nemo-relay",
            "version": env!("CARGO_PKG_VERSION"),
            "bootstrap_protocol": crate::sidecar::BOOTSTRAP_PROTOCOL_VERSION,
            "instance_id": state.instance_id,
        })),
    )
        .into_response()
}

fn write_ready_file(path: &Path, bind: SocketAddr, instance_id: &str) -> Result<(), CliError> {
    let bytes = serde_json::to_vec(&serde_json::json!({
        "address": bind,
        "service": "nemo-relay",
        "version": env!("CARGO_PKG_VERSION"),
        "bootstrap_protocol": crate::sidecar::BOOTSTRAP_PROTOCOL_VERSION,
        "instance_id": instance_id,
    }))
    .map_err(|error| CliError::Launch(format!("failed to encode readiness file: {error}")))?;
    let temporary = path.with_extension(format!(
        "{}tmp",
        path.extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| format!("{extension}."))
            .unwrap_or_default()
    ));
    std::fs::write(&temporary, bytes).map_err(|error| {
        CliError::Launch(format!(
            "failed to write readiness file {}: {error}",
            temporary.display()
        ))
    })?;
    std::fs::rename(&temporary, path).map_err(|error| {
        let _ = std::fs::remove_file(&temporary);
        CliError::Launch(format!(
            "failed to publish readiness file {}: {error}",
            path.display()
        ))
    })
}

fn plugin_idle_timeout() -> Result<Option<Duration>, CliError> {
    let Some(raw) = std::env::var("NEMO_RELAY_PLUGIN_IDLE_TIMEOUT_SECS").ok() else {
        return Ok(None);
    };
    let seconds = raw.parse::<u64>().map_err(|error| {
        CliError::Config(format!(
            "NEMO_RELAY_PLUGIN_IDLE_TIMEOUT_SECS must be a positive integer: {error}"
        ))
    })?;
    if seconds == 0 {
        return Err(CliError::Config(
            "NEMO_RELAY_PLUGIN_IDLE_TIMEOUT_SECS must be greater than 0".into(),
        ));
    }
    Ok(Some(Duration::from_secs(seconds)))
}

async fn idle_shutdown_future(
    last_activity: Arc<Mutex<Instant>>,
    sessions: SessionManager,
    timeout: Duration,
) {
    let tick = timeout
        .min(Duration::from_secs(5))
        .max(Duration::from_secs(1));
    loop {
        tokio::time::sleep(tick).await;
        if idle_shutdown_ready(&last_activity, timeout, sessions.has_open_sessions()).await {
            break;
        }
    }
}

async fn idle_shutdown_ready<F>(
    last_activity: &Arc<Mutex<Instant>>,
    timeout: Duration,
    has_open_sessions: F,
) -> bool
where
    F: std::future::Future<Output = bool>,
{
    let observed = match last_activity.lock() {
        Ok(last_activity) if last_activity.elapsed() >= timeout => *last_activity,
        Ok(_) => return false,
        Err(_) => return true,
    };
    if has_open_sessions.await {
        return false;
    }
    last_activity.lock().map_or(true, |last_activity| {
        *last_activity == observed && last_activity.elapsed() >= timeout
    })
}

struct PluginActivation {
    active: bool,
    native: Option<NativePluginActivation>,
    worker: Option<WorkerPluginActivation>,
    _snapshots: Vec<Arc<DynamicPluginActivationSnapshot>>,
}

impl PluginActivation {
    async fn initialize(
        config: Option<Value>,
        dynamic_plugins: Vec<ActiveDynamicPluginComponent>,
    ) -> Result<Self, CliError> {
        if config.is_none() && dynamic_plugins.is_empty() {
            return Ok(Self {
                active: false,
                native: None,
                worker: None,
                _snapshots: Vec::new(),
            });
        };
        register_adaptive_component().map_err(|error| {
            CliError::Config(format!("adaptive plugin registration failed: {error}"))
        })?;
        register_pii_redaction_component().map_err(|error| {
            CliError::Config(format!("PII redaction plugin registration failed: {error}"))
        })?;
        for plugin in &dynamic_plugins {
            if let Some(snapshot) = plugin.activation_snapshot.as_ref() {
                snapshot.verify_current()?;
            }
        }
        let native_specs = dynamic_plugins
            .iter()
            .filter(|plugin| plugin.kind == DynamicPluginKind::RustDynamic)
            .map(|plugin| {
                let manifest_ref = plugin
                    .activation_snapshot
                    .as_ref()
                    .map(|snapshot| snapshot.activation_manifest_ref())
                    .or_else(|| plugin.manifest_ref.clone())
                    .ok_or_else(|| {
                        CliError::Config(format!(
                            "native dynamic plugin '{}' has no manifest_ref in lifecycle state",
                            plugin.plugin_id
                        ))
                    })?;
                Ok(NativePluginLoadSpec {
                    plugin_id: plugin.plugin_id.clone(),
                    manifest_ref,
                })
            })
            .collect::<Result<Vec<_>, CliError>>()?;
        let worker_specs = dynamic_plugins
            .iter()
            .filter(|plugin| plugin.kind == DynamicPluginKind::Worker)
            .map(|plugin| {
                let manifest_ref = plugin
                    .activation_snapshot
                    .as_ref()
                    .map(|snapshot| snapshot.activation_manifest_ref())
                    .or_else(|| plugin.manifest_ref.clone())
                    .ok_or_else(|| {
                        CliError::Config(format!(
                            "worker dynamic plugin '{}' has no manifest_ref in lifecycle state",
                            plugin.plugin_id
                        ))
                    })?;
                Ok(WorkerPluginLoadSpec {
                    plugin_id: plugin.plugin_id.clone(),
                    manifest_ref,
                    environment_ref: plugin
                        .activation_snapshot
                        .as_ref()
                        .and_then(|snapshot| snapshot.activation_environment_ref())
                        .map(ToOwned::to_owned)
                        .or_else(|| plugin.environment_ref.clone()),
                    config: plugin.config.clone(),
                })
            })
            .collect::<Result<Vec<_>, CliError>>()?;
        let snapshots = dynamic_plugins
            .iter()
            .filter_map(|plugin| plugin.activation_snapshot.clone())
            .collect();
        let native =
            if native_specs.is_empty() {
                None
            } else {
                Some(load_native_plugins(native_specs).map_err(|error| {
                    CliError::Config(format!("native plugin load failed: {error}"))
                })?)
            };
        for plugin in &dynamic_plugins {
            if let Some(snapshot) = plugin.activation_snapshot.as_ref() {
                snapshot.verify_current()?;
            }
        }
        let worker =
            if worker_specs.is_empty() {
                None
            } else {
                Some(load_worker_plugins(worker_specs).map_err(|error| {
                    CliError::Config(format!("worker plugin load failed: {error}"))
                })?)
            };
        // Gateway already resolved its config; activate exactly (no re-discovery).
        let mut plugin_config: PluginConfig = match config {
            Some(config) => serde_json::from_value(config)
                .map_err(|error| CliError::Config(format!("invalid plugin config: {error}")))?,
            None => PluginConfig::default(),
        };
        plugin_config
            .components
            .extend(
                dynamic_plugins
                    .into_iter()
                    .map(|plugin| PluginComponentSpec {
                        kind: plugin.plugin_id,
                        enabled: true,
                        config: plugin.config,
                    }),
            );
        initialize_plugins_exact(plugin_config)
            .await
            .map_err(|error| CliError::Config(format!("plugin activation failed: {error}")))?;
        Ok(Self {
            active: true,
            native,
            worker,
            _snapshots: snapshots,
        })
    }

    fn clear(mut self) -> Result<(), CliError> {
        let result = if self.active {
            self.active = false;
            clear_plugin_configuration()
                .map_err(|error| CliError::Config(format!("plugin teardown failed: {error}")))?;
            Ok(())
        } else {
            Ok(())
        };
        self.native.take();
        self.worker.take();
        result
    }
}

impl Drop for PluginActivation {
    fn drop(&mut self) {
        if self.active {
            let _ = clear_plugin_configuration();
            self.active = false;
        }
    }
}

// Normalizes a Codex hook payload, applies all resulting events before responding, and returns the
// adapter's pass-through response body so hook delivery stays causally ordered with observability.
async fn codex_hook(
    State(state): State<AppState>,
    headers: HeaderMap,
    payload: Result<Json<Value>, JsonRejection>,
) -> Result<Json<Value>, CliError> {
    state.touch();
    let Json(payload) = payload.map_err(hook_payload_rejection)?;
    let outcome = codex::adapt(payload, &headers);
    state
        .sessions
        .apply_events(&headers, outcome.events)
        .await?;
    Ok(Json(outcome.response))
}

// Handles Claude Code hooks with the adapter's explicit continuation/permission response. Events
// are committed before the response so Claude lifecycle hooks can close scopes deterministically.
async fn claude_code_hook(
    State(state): State<AppState>,
    headers: HeaderMap,
    payload: Result<Json<Value>, JsonRejection>,
) -> Result<Json<Value>, CliError> {
    state.touch();
    let Json(payload) = payload.map_err(hook_payload_rejection)?;
    let outcome = claude_code::adapt(payload, &headers);
    state
        .sessions
        .apply_events(&headers, outcome.events)
        .await?;
    Ok(Json(outcome.response))
}

// Handles Hermes hook payloads from persistent shell integration. The adapter returns a minimal
// body because hook-forward owns the fail-open/fail-closed behavior for Hermes command execution.
async fn hermes_hook(
    State(state): State<AppState>,
    headers: HeaderMap,
    payload: Result<Json<Value>, JsonRejection>,
) -> Result<Json<Value>, CliError> {
    state.touch();
    let Json(payload) = payload.map_err(hook_payload_rejection)?;
    let outcome = hermes::adapt(payload, &headers);
    state
        .sessions
        .apply_events(&headers, outcome.events)
        .await?;
    Ok(Json(outcome.response))
}

fn hook_payload_rejection(rejection: JsonRejection) -> CliError {
    if rejection.status() == axum::http::StatusCode::PAYLOAD_TOO_LARGE {
        CliError::PayloadTooLarge(rejection.to_string())
    } else {
        CliError::InvalidPayload(rejection.to_string())
    }
}

#[cfg(test)]
#[path = "../tests/coverage/server_tests.rs"]
mod tests;
