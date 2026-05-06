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

pub(crate) async fn serve(config: SidecarConfig) -> Result<(), SidecarError> {
    let listener = TcpListener::bind(config.bind).await?;
    serve_listener(listener, config, None).await
}

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
