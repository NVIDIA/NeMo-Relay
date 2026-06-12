// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::rejection::JsonRejection;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::HeaderMap;
use axum::routing::{get, post};
use axum::{Json, Router};
use nemo_relay::plugin::{PluginConfig, clear_plugin_configuration, initialize_plugins_exact};
use nemo_relay_adaptive::plugin_component::register_adaptive_component;
use nemo_relay_pii_redaction::component::register_pii_redaction_component;
use reqwest::Client;
use serde_json::Value;
use tokio::net::TcpListener;
use tokio::sync::oneshot;

use crate::adapters::{claude_code, codex, cursor, hermes};
use crate::config::GatewayConfig;
use crate::error::CliError;
use crate::gateway;
use crate::session::SessionManager;

const HTTP_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const HTTP_REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
const HTTP_READ_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Clone)]
pub(crate) struct AppState {
    pub(crate) config: GatewayConfig,
    pub(crate) http: Client,
    pub(crate) sessions: SessionManager,
    pub(crate) last_activity: Arc<Mutex<Instant>>,
}

/// Binds the configured address and serves until the process is stopped.
///
/// Tests and transparent run mode use `serve_listener` directly so they can supply an already
/// bound ephemeral listener and optional shutdown channel.
pub(crate) async fn serve(config: GatewayConfig) -> Result<(), CliError> {
    let listener = TcpListener::bind(config.bind).await.map_err(|err| {
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
                config.bind
            ))
        } else {
            CliError::Io(err)
        }
    })?;
    serve_listener(listener, config, None).await
}

/// Serves the gateway router on a caller-owned listener with optional graceful shutdown.
///
/// A provided shutdown receiver is best-effort: the send side may be dropped after the child agent
/// exits, and either receiving or channel closure is enough to let Axum drain the listener.
pub(crate) async fn serve_listener(
    listener: TcpListener,
    config: GatewayConfig,
    shutdown: Option<oneshot::Receiver<()>>,
) -> Result<(), CliError> {
    let plugin_activation = PluginActivation::initialize(config.plugin_config.clone()).await?;
    let state = AppState::new(config);
    let sessions = state.sessions.clone();
    let last_activity = state.last_activity.clone();
    let app = router_with_state(state);
    let idle_shutdown = (shutdown.is_none())
        .then(plugin_idle_timeout)
        .flatten()
        .map(|timeout| idle_shutdown_future(last_activity, sessions.clone(), timeout));
    let serve_result = match (shutdown, idle_shutdown) {
        (Some(receiver), _) => {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = receiver.await;
                })
                .await
        }
        (None, Some(idle)) => {
            axum::serve(listener, app)
                .with_graceful_shutdown(idle)
                .await
        }
        (None, None) => axum::serve(listener, app).await,
    };
    let close_result = sessions.close_all("gateway_shutdown").await;
    let clear_result = plugin_activation.clear();
    if let Err(serve_error) = serve_result {
        if let Err(close_error) = close_result {
            eprintln!("session teardown failed after server error: {close_error}");
        }
        if let Err(clear_error) = clear_result {
            eprintln!("plugin teardown failed after server error: {clear_error}");
        }
        return Err(serve_error.into());
    }
    close_result?;
    clear_result
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
    fn new(config: GatewayConfig) -> Self {
        crate::tls::install_rustls_crypto_provider();
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
            http,
            sessions,
            last_activity: Arc::new(Mutex::new(Instant::now())),
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
        .route("/hooks/codex", post(codex_hook))
        .route("/hooks/claude-code", post(claude_code_hook))
        .route("/hooks/cursor", post(cursor_hook))
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

async fn healthz(State(state): State<AppState>) -> Json<Value> {
    state.touch();
    Json(serde_json::json!({ "status": "ok" }))
}

fn plugin_idle_timeout() -> Option<Duration> {
    let raw = std::env::var("NEMO_RELAY_PLUGIN_IDLE_TIMEOUT_SECS").ok()?;
    let seconds = raw.parse::<u64>().ok()?;
    (seconds > 0).then(|| Duration::from_secs(seconds))
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
        let elapsed = last_activity
            .lock()
            .map(|last_activity| last_activity.elapsed())
            .unwrap_or(timeout);
        if elapsed >= timeout && !sessions.has_open_sessions().await {
            break;
        }
    }
}

struct PluginActivation {
    active: bool,
}

impl PluginActivation {
    async fn initialize(config: Option<Value>) -> Result<Self, CliError> {
        let Some(config) = config else {
            return Ok(Self { active: false });
        };
        register_adaptive_component().map_err(|error| {
            CliError::Config(format!("adaptive plugin registration failed: {error}"))
        })?;
        register_pii_redaction_component().map_err(|error| {
            CliError::Config(format!("PII redaction plugin registration failed: {error}"))
        })?;
        // Gateway already resolved its config; activate exactly (no re-discovery).
        let plugin_config: PluginConfig = serde_json::from_value(config)
            .map_err(|error| CliError::Config(format!("invalid plugin config: {error}")))?;
        initialize_plugins_exact(plugin_config)
            .await
            .map_err(|error| CliError::Config(format!("plugin activation failed: {error}")))?;
        Ok(Self { active: true })
    }

    fn clear(mut self) -> Result<(), CliError> {
        if self.active {
            self.active = false;
            clear_plugin_configuration()
                .map_err(|error| CliError::Config(format!("plugin teardown failed: {error}")))?;
        }
        Ok(())
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

// Handles Cursor hook payloads and preserves Cursor's fail-open response shape. Shell and MCP hook
// names are already normalized by the adapter before session state is updated.
async fn cursor_hook(
    State(state): State<AppState>,
    headers: HeaderMap,
    payload: Result<Json<Value>, JsonRejection>,
) -> Result<Json<Value>, CliError> {
    state.touch();
    let Json(payload) = payload.map_err(hook_payload_rejection)?;
    let outcome = cursor::adapt(payload, &headers);
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
