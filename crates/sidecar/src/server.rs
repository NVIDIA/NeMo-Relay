// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::extract::State;
use axum::http::HeaderMap;
use axum::routing::{get, post};
use axum::{Json, Router};
use reqwest::Client;
use serde_json::Value;
use tokio::net::TcpListener;
use tokio::sync::oneshot;

use crate::adapters::{claude_code, codex, cursor, hermes};
use crate::config::SidecarConfig;
use crate::error::SidecarError;
use crate::gateway;
use crate::session::SessionManager;

#[derive(Clone)]
pub(crate) struct AppState {
    pub(crate) config: SidecarConfig,
    pub(crate) http: Client,
    pub(crate) sessions: SessionManager,
}

/// Binds the configured address and serves until the process is stopped.
///
/// Tests and transparent run mode use `serve_listener` directly so they can supply an already
/// bound ephemeral listener and optional shutdown channel.
pub(crate) async fn serve(config: SidecarConfig) -> Result<(), SidecarError> {
    let listener = TcpListener::bind(config.bind).await?;
    serve_listener(listener, config, None).await
}

/// Serves the sidecar router on a caller-owned listener with optional graceful shutdown.
///
/// A provided shutdown receiver is best-effort: the send side may be dropped after the child agent
/// exits, and either receiving or channel closure is enough to let Axum drain the listener.
pub(crate) async fn serve_listener(
    listener: TcpListener,
    config: SidecarConfig,
    shutdown: Option<oneshot::Receiver<()>>,
) -> Result<(), SidecarError> {
    let app = router(config);
    match shutdown {
        Some(receiver) => {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = receiver.await;
                })
                .await?;
        }
        None => {
            axum::serve(listener, app).await?;
        }
    }
    Ok(())
}

/// Builds the sidecar HTTP router and shared state.
///
/// Hook endpoints normalize agent-specific payloads into session events, while gateway endpoints
/// proxy model traffic and emit LLM runtime events against the same `SessionManager`.
pub(crate) fn router(config: SidecarConfig) -> Router {
    let sessions = SessionManager::new(config.clone());
    let state = AppState {
        config,
        http: Client::new(),
        sessions,
    };
    Router::new()
        .route("/healthz", get(healthz))
        .route("/hooks/codex", post(codex_hook))
        .route("/hooks/claude-code", post(claude_code_hook))
        .route("/hooks/cursor", post(cursor_hook))
        .route("/hooks/hermes", post(hermes_hook))
        .route("/v1/responses", post(gateway::passthrough))
        .route("/v1/chat/completions", post(gateway::passthrough))
        .route("/v1/messages", post(gateway::passthrough))
        .route("/v1/messages/count_tokens", post(gateway::passthrough))
        .route("/v1/models", get(gateway::models))
        .with_state(state)
}

async fn healthz() -> Json<Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

// Normalizes a Codex hook payload, applies all resulting events before responding, and returns the
// adapter's pass-through response body so hook delivery stays causally ordered with observability.
async fn codex_hook(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<Value>,
) -> Result<Json<Value>, SidecarError> {
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
    Json(payload): Json<Value>,
) -> Result<Json<Value>, SidecarError> {
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
    Json(payload): Json<Value>,
) -> Result<Json<Value>, SidecarError> {
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
    Json(payload): Json<Value>,
) -> Result<Json<Value>, SidecarError> {
    let outcome = hermes::adapt(payload, &headers);
    state
        .sessions
        .apply_events(&headers, outcome.events)
        .await?;
    Ok(Json(outcome.response))
}

#[cfg(test)]
#[path = "../tests/coverage/server_tests.rs"]
mod tests;
