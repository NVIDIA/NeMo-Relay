// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod types;

pub(crate) use types::GatewayOverrides;

use std::future::Future;
use std::net::SocketAddr;
use std::path::Path;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::rejection::JsonRejection;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, HeaderValue, Request, StatusCode};
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
#[cfg(feature = "switchyard")]
use nemo_relay_switchyard::{
    register_switchyard_component, validate_switchyard_atof_configuration,
};
use reqwest::Client;
use serde_json::Value;
use subtle::ConstantTimeEq;
use tokio::net::TcpListener;
use tokio::sync::oneshot;

use crate::agents::shared::adapters::{claude_code, codex, hermes};
use crate::configuration::{
    BOOTSTRAP_CLIENT_TOKEN_HEADER, BootstrapChallengeKey, GatewayConfig, ManagedBootstrapIdentity,
};
use crate::error::CliError;
use crate::gateway;
use crate::plugins::lifecycle::{ActiveDynamicPluginComponent, DynamicPluginActivationSnapshot};
use crate::sessions::SessionManager;

const HTTP_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const HTTP_REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
const HTTP_READ_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Clone)]
pub(crate) struct AppState {
    pub(crate) config: GatewayConfig,
    pub(crate) bootstrap_fingerprint: Option<String>,
    pub(crate) bootstrap_challenge_key: Option<BootstrapChallengeKey>,
    pub(crate) require_provider_client_token: bool,
    pub(crate) transparent_proxy_credential:
        Option<crate::provider_auth::TransparentProxyCredential>,
    pub(crate) http: Client,
    pub(crate) sessions: SessionManager,
    pub(crate) last_activity: Arc<Mutex<Instant>>,
    pub(crate) bootstrap_shutdown: Option<BootstrapShutdown>,
    pub(crate) instance_id: String,
    pub(crate) bootstrap_tls: Option<Arc<rustls::ServerConfig>>,
    pub(crate) local_address: Option<SocketAddr>,
}

#[derive(Clone)]
pub(crate) struct BootstrapShutdown {
    token: String,
    sender: Arc<Mutex<Option<oneshot::Sender<()>>>>,
}

#[derive(Default)]
struct BootstrapServeOptions<'a> {
    fingerprint: Option<String>,
    identity: Option<ManagedBootstrapIdentity>,
    ready_file: Option<&'a Path>,
    shutdown_token: Option<String>,
    transparent_proxy_credential: Option<crate::provider_auth::TransparentProxyCredential>,
}

/// Binds the configured address and activates enabled dynamic plugins before serving.
pub(crate) async fn serve_with_dynamic(
    config: GatewayConfig,
    dynamic_plugins: Vec<ActiveDynamicPluginComponent>,
    managed_bootstrap: Option<ManagedBootstrapIdentity>,
    ready_file: Option<&Path>,
    bootstrap_shutdown_token: Option<String>,
) -> Result<(), CliError> {
    let bind = config.bind.to_string();
    log::info!(
        target: "nemo_relay.server",
        event = "server_starting",
        bind = bind.as_str();
        "Gateway server is starting"
    );
    let listener = bind_listener(config.bind).await?;
    print_startup_status(listener.local_addr()?, &config);
    let bootstrap_fingerprint = managed_bootstrap
        .as_ref()
        .map(|identity| identity.fingerprint().to_owned());
    serve_listener_with_dynamic_inner(
        listener,
        config,
        dynamic_plugins,
        Some(ShutdownMode::ProcessSignal),
        BootstrapServeOptions {
            fingerprint: bootstrap_fingerprint,
            identity: managed_bootstrap,
            ready_file,
            shutdown_token: bootstrap_shutdown_token,
            ..BootstrapServeOptions::default()
        },
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
                 • use the managed shutdown command, or identify the owning daemon PID and \
                 terminate only that process\n  \
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
    let destinations = crate::process::launcher::exporter_destinations(config);
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

    crate::process::launcher::render_status_frame(&lines, color)
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
        shutdown.map(ShutdownMode::Receiver),
        BootstrapServeOptions {
            fingerprint: Some(bootstrap_fingerprint),
            ..BootstrapServeOptions::default()
        },
    )
    .await
}

/// Serves the gateway router and activates enabled dynamic plugin components.
#[cfg(test)]
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
        shutdown.map(ShutdownMode::Receiver),
        BootstrapServeOptions::default(),
    )
    .await
}

/// Serves a wrapper-owned dynamic gateway with authenticated health while keeping foreground
/// provider-auth semantics. Plugin-owned MCP clients use the proof to borrow only this instance.
pub(crate) async fn serve_transparent_listener_with_dynamic(
    listener: TcpListener,
    config: GatewayConfig,
    dynamic_plugins: Vec<ActiveDynamicPluginComponent>,
    bootstrap_fingerprint: String,
    transparent_proxy_credential: crate::provider_auth::TransparentProxyCredential,
    shutdown: Option<oneshot::Receiver<()>>,
) -> Result<(), CliError> {
    serve_listener_with_dynamic_inner(
        listener,
        config,
        dynamic_plugins,
        shutdown.map(ShutdownMode::Receiver),
        BootstrapServeOptions {
            fingerprint: Some(bootstrap_fingerprint),
            transparent_proxy_credential: Some(transparent_proxy_credential),
            ..BootstrapServeOptions::default()
        },
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
    shutdown_mode: Option<ShutdownMode>,
    bootstrap: BootstrapServeOptions<'_>,
) -> Result<(), CliError> {
    let BootstrapServeOptions {
        fingerprint: bootstrap_fingerprint,
        identity: managed_bootstrap,
        ready_file,
        shutdown_token: bootstrap_shutdown_token,
        transparent_proxy_credential,
    } = bootstrap;
    let bootstrap_challenge_key = bootstrap_fingerprint
        .as_ref()
        .map(|_| BootstrapChallengeKey::load())
        .transpose()?;
    let bootstrap_tls = bootstrap_fingerprint
        .as_ref()
        .map(|_| crate::gateway::tls::RelayTlsIdentity::load_or_create())
        .transpose()
        .map_err(CliError::Launch)?
        .map(|identity| identity.server_config())
        .transpose()
        .map_err(CliError::Launch)?;
    let require_provider_client_token = managed_bootstrap.is_some();
    let plugin_activation =
        initialize_plugin_host(config.plugin_config.clone(), dynamic_plugins).await?;
    let (bootstrap_shutdown, bootstrap_shutdown_rx) =
        bootstrap_shutdown_channel(bootstrap_shutdown_token.clone());
    let mut state = AppState::new_with_bootstrap(
        config,
        bootstrap_fingerprint,
        bootstrap_challenge_key,
        require_provider_client_token,
        bootstrap_shutdown,
        transparent_proxy_credential,
    );
    state.bootstrap_tls = bootstrap_tls;
    state.local_address = Some(listener.local_addr()?);
    let instance_id = state.instance_id.clone();
    let sessions = state.sessions.clone();
    let last_activity = state.last_activity.clone();
    let app = router_with_state(state);
    let local_address = listener.local_addr()?;
    if let Some(identity) = managed_bootstrap.as_ref() {
        identity.verify_current()?;
    }
    let _owner = crate::bootstrap::state::publish_owner_from_env(
        local_address,
        bootstrap_shutdown_token.as_deref(),
    )
    .map_err(CliError::Launch)?;
    if let Some(path) = ready_file {
        write_ready_file(path, local_address, &instance_id)?;
    }
    let address = local_address.to_string();
    log::info!(
        target: "nemo_relay.server",
        event = "server_listening",
        address = address.as_str(),
        instance_id = instance_id.as_str();
        "Gateway server is listening"
    );
    let idle_shutdown: Option<ShutdownFuture> =
        if matches!(&shutdown_mode, None | Some(ShutdownMode::ProcessSignal)) {
            plugin_idle_timeout()?.map(|timeout| {
                Box::pin(idle_shutdown_future(
                    last_activity,
                    sessions.clone(),
                    timeout,
                )) as ShutdownFuture
            })
        } else {
            None
        };
    let shutdown = server_shutdown_future(shutdown_mode, idle_shutdown);
    let shutdown = combine_shutdown_futures(shutdown, bootstrap_shutdown_rx);
    log::info!(
        target: "nemo_relay.server",
        event = "server_shutdown_started",
        instance_id = instance_id.as_str();
        "Gateway server shutdown started"
    );
    let serve_result = match shutdown {
        Some(shutdown) => {
            axum::serve(listener, app)
                .with_graceful_shutdown(shutdown)
                .await
        }
        None => axum::serve(listener, app).await,
    };
    finish_server_shutdown(serve_result, &sessions, plugin_activation, &instance_id).await
}

fn server_shutdown_future(
    shutdown_mode: Option<ShutdownMode>,
    idle_shutdown: Option<ShutdownFuture>,
) -> Option<ShutdownFuture> {
    match shutdown_mode {
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
        None => idle_shutdown,
    }
}

fn combine_shutdown_futures(
    shutdown: Option<ShutdownFuture>,
    bootstrap_shutdown_rx: Option<oneshot::Receiver<()>>,
) -> Option<ShutdownFuture> {
    match (shutdown, bootstrap_shutdown_rx) {
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
    }
}

async fn finish_server_shutdown(
    serve_result: std::io::Result<()>,
    sessions: &SessionManager,
    plugin_activation: Option<ServerPluginActivation>,
    instance_id: &str,
) -> Result<(), CliError> {
    let close_result = sessions.close_all("gateway_shutdown").await;
    let flush_result = nemo_relay::api::runtime::flush_subscribers().map_err(CliError::from);
    let clear_result = plugin_activation
        .map(ServerPluginActivation::clear)
        .unwrap_or(Ok(()));
    if let Err(serve_error) = serve_result {
        log::error!(
            target: "nemo_relay.server",
            event = "server_failed",
            instance_id,
            error_kind = "io";
            "Gateway server failed"
        );
        if let Err(close_error) = close_result {
            log::error!(
                target: "nemo_relay.server",
                event = "server_teardown_failed",
                instance_id,
                component = "sessions",
                error_kind = close_error.log_kind();
                "Gateway server teardown failed"
            );
        }
        if let Err(flush_error) = flush_result {
            log::error!(
                target: "nemo_relay.server",
                event = "server_teardown_failed",
                instance_id,
                component = "subscribers",
                error_kind = flush_error.log_kind();
                "Gateway server teardown failed"
            );
        }
        if let Err(clear_error) = clear_result {
            log::error!(
                target: "nemo_relay.server",
                event = "server_teardown_failed",
                instance_id,
                component = "plugins",
                error_kind = clear_error.log_kind();
                "Gateway server teardown failed"
            );
        }
        return Err(serve_error.into());
    }
    if let Err(error) = &close_result {
        log::error!(
            target: "nemo_relay.server",
            event = "server_teardown_failed",
            instance_id,
            component = "sessions",
            error_kind = error.log_kind();
            "Gateway server teardown failed"
        );
    }
    if let Err(error) = &flush_result {
        log::error!(
            target: "nemo_relay.server",
            event = "server_teardown_failed",
            instance_id,
            component = "subscribers",
            error_kind = error.log_kind();
            "Gateway server teardown failed"
        );
    }
    if let Err(error) = &clear_result {
        log::error!(
            target: "nemo_relay.server",
            event = "server_teardown_failed",
            instance_id,
            component = "plugins",
            error_kind = error.log_kind();
            "Gateway server teardown failed"
        );
    }
    close_result?;
    flush_result?;
    clear_result?;
    log::info!(
        target: "nemo_relay.server",
        event = "server_stopped",
        instance_id;
        "Gateway server stopped"
    );
    Ok(())
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
    pub(crate) fn new(config: GatewayConfig) -> Self {
        Self::new_with_bootstrap(config, None, None, false, None, None)
    }

    fn new_with_bootstrap(
        config: GatewayConfig,
        bootstrap_fingerprint: Option<String>,
        bootstrap_challenge_key: Option<BootstrapChallengeKey>,
        require_provider_client_token: bool,
        bootstrap_shutdown: Option<BootstrapShutdown>,
        transparent_proxy_credential: Option<crate::provider_auth::TransparentProxyCredential>,
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
            require_provider_client_token,
            transparent_proxy_credential,
            http,
            sessions,
            last_activity: Arc::new(Mutex::new(Instant::now())),
            bootstrap_shutdown,
            instance_id: uuid::Uuid::now_v7().to_string(),
            bootstrap_tls: None,
            local_address: None,
        }
    }

    pub(crate) fn touch(&self) {
        if let Ok(mut last_activity) = self.last_activity.lock() {
            *last_activity = Instant::now();
        }
    }

    /// Authenticate an invocation-owned transparent client before interceptors can rewrite its
    /// route. Foreground gateways retain their existing provider-credential behavior; managed
    /// sidecars still require their stable client proof before ambient provider credentials may be
    /// used.
    pub(crate) fn authorize_provider_request(
        &self,
        headers: &mut HeaderMap,
    ) -> Result<crate::provider_auth::ProviderRequestAuthorization, CliError> {
        if let Some(proxy) = &self.transparent_proxy_credential {
            let source_credential = proxy.consume(headers).inspect_err(|error| {
                log::warn!(
                    target: "nemo_relay.gateway",
                    event = "request_rejected",
                    reason = "transparent_proxy_authentication",
                    error_kind = error.log_kind();
                    "Gateway request was rejected during transparent proxy authentication"
                );
            })?;
            return Ok(crate::provider_auth::ProviderRequestAuthorization {
                source_credential,
                allow_environment_provider_auth: true,
            });
        }
        let allow_environment_provider_auth = if !self.require_provider_client_token {
            true
        } else {
            self.bootstrap_challenge_key
                .as_ref()
                .and_then(|key| {
                    headers
                        .get(BOOTSTRAP_CLIENT_TOKEN_HEADER)
                        .and_then(|value| value.to_str().ok())
                        .map(|token| key.verify_client_token(token))
                })
                .unwrap_or(false)
        };
        Ok(crate::provider_auth::ProviderRequestAuthorization {
            source_credential:
                crate::provider_auth::SourceCredentialDisposition::from_provider_headers(headers),
            allow_environment_provider_auth,
        })
    }
}

fn router_with_state(state: AppState) -> Router {
    let max_hook_payload_bytes = state.config.max_hook_payload_bytes;
    Router::new()
        .route("/healthz", get(healthz))
        .route("/bootstrap/tunnel", get(bootstrap_tls_tunnel))
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

async fn bootstrap_tls_tunnel(
    State(state): State<AppState>,
    mut request: Request<Body>,
) -> Response {
    let headers = request.headers();
    let Some(fingerprint) = headers
        .get("x-nemo-relay-bootstrap-fingerprint")
        .and_then(|value| value.to_str().ok())
    else {
        return StatusCode::FORBIDDEN.into_response();
    };
    let Some(nonce) = headers
        .get("x-nemo-relay-bootstrap-nonce")
        .and_then(|value| value.to_str().ok())
        .filter(|nonce| nonce.len() == 64 && nonce.bytes().all(|byte| byte.is_ascii_hexdigit()))
    else {
        return StatusCode::FORBIDDEN.into_response();
    };
    let fingerprint_matches = state
        .bootstrap_fingerprint
        .as_deref()
        .is_some_and(|actual| bool::from(actual.as_bytes().ct_eq(fingerprint.as_bytes())));
    let (Some(key), Some(tls), Some(local_address)) = (
        state.bootstrap_challenge_key.as_ref(),
        state.bootstrap_tls.clone(),
        state.local_address,
    ) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    if !fingerprint_matches
        || headers
            .get(http::header::UPGRADE)
            .and_then(|value| value.to_str().ok())
            != Some("nemo-relay-tls")
    {
        return StatusCode::FORBIDDEN.into_response();
    }
    let proof = key.proof(fingerprint, nonce);
    let upgrade = hyper::upgrade::on(&mut request);
    tokio::spawn(async move {
        let Ok(upgraded) = upgrade.await else {
            return;
        };
        let acceptor = tokio_rustls::TlsAcceptor::from(tls);
        let Ok(mut encrypted) = acceptor
            .accept(hyper_util::rt::TokioIo::new(upgraded))
            .await
        else {
            return;
        };
        let Ok(mut local) = tokio::net::TcpStream::connect(local_address).await else {
            return;
        };
        let _ = tokio::io::copy_bidirectional(&mut encrypted, &mut local).await;
    });
    Response::builder()
        .status(StatusCode::SWITCHING_PROTOCOLS)
        .header(http::header::CONNECTION, "upgrade")
        .header(http::header::UPGRADE, "nemo-relay-tls")
        .header("x-nemo-relay-bootstrap-proof", proof)
        .header(http::header::CONTENT_LENGTH, "0")
        .body(Body::empty())
        .expect("bootstrap TLS upgrade response is valid")
}

fn bootstrap_shutdown_channel(
    token: Option<String>,
) -> (Option<BootstrapShutdown>, Option<oneshot::Receiver<()>>) {
    let Some(token) = token else {
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
            "bootstrap_protocol": crate::bootstrap::BOOTSTRAP_PROTOCOL_VERSION,
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
        "bootstrap_protocol": crate::bootstrap::BOOTSTRAP_PROTOCOL_VERSION,
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

enum ServerPluginActivation {
    Static,
    Dynamic(PluginActivation),
}

impl ServerPluginActivation {
    fn clear(self) -> Result<(), CliError> {
        match self {
            Self::Static => clear_plugin_configuration()
                .map_err(|error| CliError::Config(format!("plugin teardown failed: {error}"))),
            Self::Dynamic(activation) => activation.clear(),
        }
    }
}

#[derive(Debug)]
pub(crate) enum PluginComponentSetupError {
    Adaptive(String),
    PiiRedaction(String),
    #[cfg(feature = "switchyard")]
    Switchyard(String),
    #[cfg(feature = "switchyard")]
    SwitchyardAtof(String),
}

impl PluginComponentSetupError {
    pub(crate) const fn check_name(&self) -> &'static str {
        match self {
            Self::Adaptive(_) => "Adaptive plugin",
            Self::PiiRedaction(_) => "PII redaction plugin",
            #[cfg(feature = "switchyard")]
            Self::Switchyard(_) => "Switchyard plugin",
            #[cfg(feature = "switchyard")]
            Self::SwitchyardAtof(_) => "Switchyard ATOF",
        }
    }

    pub(crate) fn diagnostic_details(&self) -> String {
        match self {
            Self::Adaptive(error) | Self::PiiRedaction(error) => {
                format!("registration failed: {error}")
            }
            #[cfg(feature = "switchyard")]
            Self::Switchyard(error) => format!("registration failed: {error}"),
            #[cfg(feature = "switchyard")]
            Self::SwitchyardAtof(error) => error.clone(),
        }
    }
}

impl std::fmt::Display for PluginComponentSetupError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Adaptive(error) => {
                write!(formatter, "adaptive plugin registration failed: {error}")
            }
            Self::PiiRedaction(error) => {
                write!(
                    formatter,
                    "PII redaction plugin registration failed: {error}"
                )
            }
            #[cfg(feature = "switchyard")]
            Self::Switchyard(error) => {
                write!(formatter, "Switchyard plugin registration failed: {error}")
            }
            #[cfg(feature = "switchyard")]
            Self::SwitchyardAtof(error) => {
                write!(formatter, "Switchyard ATOF validation failed: {error}")
            }
        }
    }
}

pub(crate) fn register_and_validate_plugin_components(
    _plugin_config: &PluginConfig,
) -> Vec<PluginComponentSetupError> {
    let mut errors = Vec::new();
    if let Err(error) = register_adaptive_component() {
        errors.push(PluginComponentSetupError::Adaptive(error.to_string()));
    }
    if let Err(error) = register_pii_redaction_component() {
        errors.push(PluginComponentSetupError::PiiRedaction(error.to_string()));
    }
    #[cfg(feature = "switchyard")]
    if let Err(error) = register_switchyard_component() {
        errors.push(PluginComponentSetupError::Switchyard(error.to_string()));
    }
    #[cfg(feature = "switchyard")]
    if let Err(error) = validate_switchyard_atof_configuration(_plugin_config) {
        errors.push(PluginComponentSetupError::SwitchyardAtof(error));
    }
    errors
}

async fn initialize_plugin_host(
    config: Option<Value>,
    dynamic_plugins: Vec<ActiveDynamicPluginComponent>,
) -> Result<Option<ServerPluginActivation>, CliError> {
    if config.is_none() && dynamic_plugins.is_empty() {
        return Ok(None);
    }
    if dynamic_plugins.is_empty() {
        let plugin_config: PluginConfig = config
            .map(serde_json::from_value)
            .transpose()
            .map_err(|error| CliError::Config(format!("invalid plugin config: {error}")))?
            .unwrap_or_default();
        if let Some(error) = register_and_validate_plugin_components(&plugin_config)
            .into_iter()
            .next()
        {
            return Err(CliError::Config(error.to_string()));
        }
        initialize_plugins_exact(plugin_config)
            .await
            .map_err(|error| CliError::Config(format!("plugin activation failed: {error}")))?;
        return Ok(Some(ServerPluginActivation::Static));
    }
    PluginActivation::initialize(config, dynamic_plugins)
        .await
        .map(ServerPluginActivation::Dynamic)
        .map(Some)
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
        // Gateway already resolved its config; activate exactly (no re-discovery).
        let mut plugin_config: PluginConfig = match config {
            Some(config) => serde_json::from_value(config)
                .map_err(|error| CliError::Config(format!("invalid plugin config: {error}")))?,
            None => PluginConfig::default(),
        };
        plugin_config
            .components
            .extend(dynamic_plugins.iter().map(|plugin| PluginComponentSpec {
                kind: plugin.plugin_id.clone(),
                enabled: true,
                config: plugin.config.clone(),
            }));
        if let Some(error) = register_and_validate_plugin_components(&plugin_config)
            .into_iter()
            .next()
        {
            return Err(CliError::Config(error.to_string()));
        }
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
#[path = "../../tests/coverage/shared/server_tests.rs"]
mod tests;
