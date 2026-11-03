// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use percent_encoding::percent_decode_str;
use serde_json::json;

use super::*;

const CLIENT_ID: &str = "relay-test-client";

/// A scripted OpenID Connect issuer on loopback.
#[derive(Default)]
struct Issuer {
    base: String,
    advertised_issuer: Option<String>,
    token_endpoint: Option<String>,
    /// Replies to device-code polls, in order; the last one repeats.
    device_replies: VecDeque<&'static str>,
    refresh_status: Option<StatusCode>,
    refresh_error: Option<&'static str>,
    grants: Vec<HashMap<String, String>>,
}

type Shared = Arc<Mutex<Issuer>>;

fn parse_form(body: &str) -> HashMap<String, String> {
    body.split('&')
        .filter_map(|pair| pair.split_once('='))
        .map(|(key, value)| {
            let decode = |text: &str| {
                percent_decode_str(&text.replace('+', " "))
                    .decode_utf8_lossy()
                    .into_owned()
            };
            (decode(key), decode(value))
        })
        .collect()
}

async fn discovery(State(issuer): State<Shared>) -> Json<serde_json::Value> {
    let state = issuer.lock().unwrap();
    let base = state.base.clone();
    Json(json!({
        "issuer": state.advertised_issuer.clone().unwrap_or_else(|| base.clone()),
        "token_endpoint": state.token_endpoint.clone().unwrap_or_else(|| format!("{base}/token")),
        "device_authorization_endpoint": format!("{base}/device"),
    }))
}

async fn device(body: String) -> Json<serde_json::Value> {
    let form = parse_form(&body);
    Json(json!({
        "device_code": "device-1",
        "user_code": "WDJB-MJHT",
        "verification_uri": "https://issuer.example/device",
        "expires_in": 60,
        "interval": 0,
        "echo_client": form.get("client_id"),
    }))
}

fn issued(access: &str, refresh: &str) -> Response {
    Json(json!({
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "Bearer",
        "expires_in": 3600,
    }))
    .into_response()
}

fn refused(error: &str) -> Response {
    (StatusCode::BAD_REQUEST, Json(json!({ "error": error }))).into_response()
}

async fn token(State(issuer): State<Shared>, body: String) -> Response {
    let form = parse_form(&body);
    let mut state = issuer.lock().unwrap();
    state.grants.push(form.clone());
    match form["grant_type"].as_str() {
        DEVICE_CODE_GRANT => {
            let reply = if state.device_replies.len() > 1 {
                state.device_replies.pop_front()
            } else {
                state.device_replies.front().copied()
            };
            match reply.unwrap_or("issued") {
                "issued" => issued("access-1", "refresh-1"),
                "unavailable" => {
                    (StatusCode::BAD_GATEWAY, "<html>bad gateway</html>").into_response()
                }
                error => refused(error),
            }
        }
        "refresh_token" => match state.refresh_status {
            Some(status) => (status, "upstream down").into_response(),
            None if state.refresh_error.is_some() => refused(state.refresh_error.unwrap()),
            None if form["refresh_token"] == "refresh-1" => issued("access-2", "refresh-2"),
            None => refused("invalid_grant"),
        },
        _ => refused("unsupported_grant_type"),
    }
}

async fn start_issuer() -> (Provider, Shared) {
    let shared: Shared = Arc::default();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    shared.lock().unwrap().base = base.clone();
    let app = Router::new()
        .route("/.well-known/openid-configuration", get(discovery))
        .route("/device", post(device))
        .route("/token", post(token))
        .with_state(shared.clone());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (
        Provider {
            issuer: base,
            client_id: CLIENT_ID.into(),
            scope: "openid offline_access".into(),
        },
        shared,
    )
}

fn store(dir: &Path, provider: &Provider, refresh: Option<&str>, expires_at: u64) {
    ensure_sessions_dir(dir).unwrap();
    write_session(
        &session_path(dir, provider),
        &Session {
            issuer: provider.issuer.clone(),
            client_id: provider.client_id.clone(),
            access_token: "access-1".into(),
            refresh_token: refresh.map(str::to_owned),
            expires_at,
        },
    )
    .unwrap();
}

fn saved(dir: &Path, provider: &Provider) -> Session {
    read_session(&session_path(dir, provider), provider)
        .unwrap()
        .unwrap()
}

fn device_start() -> DeviceStart {
    DeviceStart {
        device_code: "device-1".into(),
        user_code: "WDJB-MJHT".into(),
        verification_uri: "https://issuer.example/device".into(),
        verification_uri_complete: None,
        expires_in: 60,
        interval: Some(1),
    }
}

async fn poll(
    issuer: &Shared,
    provider: &Provider,
    replies: &[&'static str],
) -> Result<Session, CliError> {
    issuer.lock().unwrap().device_replies = replies.iter().copied().collect();
    let client = OidcClient::discover(provider).await.unwrap();
    client
        .poll_device(&device_start(), Duration::from_millis(1))
        .await
}

#[tokio::test]
async fn device_login_shows_the_code_and_returns_the_issued_tokens() {
    let (provider, issuer) = start_issuer().await;
    issuer.lock().unwrap().device_replies = ["authorization_pending", "issued"].into();
    let mut prompts = Vec::new();

    let client = OidcClient::discover(&provider).await.unwrap();
    let session = client
        .device_login("openid offline_access", |prompt| prompts.push(prompt))
        .await
        .unwrap();

    assert!(prompts.concat().contains("WDJB-MJHT"));
    assert_eq!(session.access_token, "access-1");
    assert_eq!(session.refresh_token.as_deref(), Some("refresh-1"));
    let grants = issuer.lock().unwrap().grants.clone();
    assert_eq!(grants.len(), 2);
    assert!(grants.iter().all(|grant| grant["client_id"] == CLIENT_ID));
}

#[tokio::test]
async fn device_polling_survives_slow_down_and_issuer_outages() {
    let (provider, issuer) = start_issuer().await;

    let session = poll(&issuer, &provider, &["slow_down", "unavailable", "issued"])
        .await
        .unwrap();

    assert_eq!(session.access_token, "access-1");
    assert_eq!(issuer.lock().unwrap().grants.len(), 3);
}

#[tokio::test]
async fn device_polling_stops_when_the_user_declines() {
    let (provider, issuer) = start_issuer().await;

    let error = poll(&issuer, &provider, &["access_denied"])
        .await
        .err()
        .unwrap()
        .to_string();

    assert!(error.contains("declined"), "{error}");
}

#[tokio::test]
async fn token_reuses_a_fresh_access_token_without_calling_the_issuer() {
    let (provider, issuer) = start_issuer().await;
    let dir = tempfile::tempdir().unwrap();
    store(dir.path(), &provider, Some("refresh-1"), 10_000);

    let token = current_token(dir.path(), &provider, 1_000).await.unwrap();

    assert_eq!(token, "access-1");
    assert!(issuer.lock().unwrap().grants.is_empty());
}

#[tokio::test]
async fn token_refreshes_near_expiry_and_keeps_the_rotated_refresh_token() {
    let (provider, issuer) = start_issuer().await;
    let dir = tempfile::tempdir().unwrap();
    store(
        dir.path(),
        &provider,
        Some("refresh-1"),
        1_000 + REFRESH_MARGIN_SECS - 1,
    );

    let token = current_token(dir.path(), &provider, 1_000).await.unwrap();

    assert_eq!(token, "access-2");
    assert_eq!(issuer.lock().unwrap().grants.len(), 1);
    let session = saved(dir.path(), &provider);
    assert_eq!(session.refresh_token.as_deref(), Some("refresh-2"));
    assert_eq!(session.expires_at, 1_000 + 3600);
}

#[tokio::test]
async fn token_asks_for_login_when_the_issuer_rejects_the_refresh_token() {
    let (provider, _issuer) = start_issuer().await;
    let dir = tempfile::tempdir().unwrap();
    store(dir.path(), &provider, Some("revoked"), 0);

    let error = current_token(dir.path(), &provider, 1_000)
        .await
        .unwrap_err()
        .to_string();

    assert!(error.contains("nemo-relay auth login"), "{error}");
    assert!(!error.contains("revoked"), "{error}");
}

#[tokio::test]
async fn token_keeps_the_sign_in_when_the_issuer_is_unavailable() {
    let (provider, issuer) = start_issuer().await;
    issuer.lock().unwrap().refresh_status = Some(StatusCode::SERVICE_UNAVAILABLE);
    let dir = tempfile::tempdir().unwrap();
    store(dir.path(), &provider, Some("refresh-1"), 0);

    let error = current_token(dir.path(), &provider, 1_000)
        .await
        .unwrap_err()
        .to_string();

    assert!(!error.contains("auth login"), "{error}");
    assert!(error.contains("try again"), "{error}");
    assert_eq!(
        saved(dir.path(), &provider).refresh_token.as_deref(),
        Some("refresh-1")
    );
}

#[tokio::test]
async fn token_uses_an_unexpired_token_that_has_no_refresh_token() {
    let (provider, issuer) = start_issuer().await;
    let dir = tempfile::tempdir().unwrap();
    store(dir.path(), &provider, None, 1_000 + 60);

    let token = current_token(dir.path(), &provider, 1_000).await.unwrap();

    assert_eq!(token, "access-1");
    assert!(issuer.lock().unwrap().grants.is_empty());
}

#[tokio::test]
async fn token_for_one_client_never_returns_another_clients_sign_in() {
    let (provider, _issuer) = start_issuer().await;
    let dir = tempfile::tempdir().unwrap();
    store(dir.path(), &provider, Some("refresh-1"), 10_000);
    let other = Provider {
        client_id: "another-client".into(),
        ..provider.clone()
    };

    let error = current_token(dir.path(), &other, 1_000)
        .await
        .unwrap_err()
        .to_string();

    assert!(error.contains("another-client"), "{error}");
}

#[tokio::test]
async fn logout_removes_the_sign_in() {
    let (provider, _issuer) = start_issuer().await;
    let dir = tempfile::tempdir().unwrap();
    store(dir.path(), &provider, Some("refresh-1"), 10_000);

    assert!(remove_session(dir.path(), &provider).await.unwrap());
    let error = current_token(dir.path(), &provider, 1_000)
        .await
        .unwrap_err()
        .to_string();

    assert!(error.contains("nemo-relay auth login"), "{error}");
    assert!(!remove_session(dir.path(), &provider).await.unwrap());
}

#[tokio::test]
async fn discovery_rejects_an_issuer_that_names_another_issuer() {
    let (provider, issuer) = start_issuer().await;
    issuer.lock().unwrap().advertised_issuer = Some("https://elsewhere.example".into());

    let error = OidcClient::discover(&provider)
        .await
        .err()
        .unwrap()
        .to_string();

    assert!(error.contains("different issuer"), "{error}");
}

#[tokio::test]
async fn discovery_rejects_a_token_endpoint_on_another_origin() {
    let (provider, issuer) = start_issuer().await;
    issuer.lock().unwrap().token_endpoint = Some("https://elsewhere.example/token".into());

    let error = OidcClient::discover(&provider)
        .await
        .err()
        .unwrap()
        .to_string();

    assert!(error.contains("another origin"), "{error}");
}

#[test]
fn issuer_urls_must_use_https_unless_they_are_loopback() {
    assert!(secure_origin("https://clerk.openrouter.ai").is_ok());
    assert!(secure_origin("http://127.0.0.1:8080").is_ok());
    assert!(secure_origin("http://localhost:8080").is_ok());
    assert!(secure_origin("http://[::1]:8080").is_ok());
    assert!(secure_origin("http://issuer.example").is_err());
}

#[test]
fn login_reads_the_provider_from_the_configured_token_helper() {
    let args = [
        "auth",
        "token",
        "--issuer",
        "https://clerk.openrouter.ai",
        "--client-id=relay",
        "--scope",
        "openid offline_access inference",
    ]
    .map(String::from);

    let provider = provider_from_helper_args(&args);

    assert_eq!(
        provider.issuer.as_deref(),
        Some("https://clerk.openrouter.ai")
    );
    assert_eq!(provider.client_id.as_deref(), Some("relay"));
    assert_eq!(
        provider.scope.as_deref(),
        Some("openid offline_access inference")
    );
}

#[test]
fn a_helper_that_does_not_run_auth_token_supplies_no_provider() {
    let args = ["--issuer", "https://elsewhere.example", "--client-id", "x"].map(String::from);

    let provider = provider_from_helper_args(&args);

    assert!(provider.issuer.is_none() && provider.client_id.is_none());
}

#[test]
fn explicit_flags_win_and_the_issuer_is_normalized() {
    let provider = ProviderArgs {
        issuer: Some("https://issuer.example/".into()),
        client_id: Some("relay".into()),
        scope: None,
    }
    .resolve()
    .unwrap();

    assert_eq!(provider.issuer, "https://issuer.example");
    assert_eq!(provider.scope, DEFAULT_SCOPE);
}

#[tokio::test]
async fn token_keeps_the_sign_in_when_the_issuer_refuses_for_another_reason() {
    let (provider, issuer) = start_issuer().await;
    issuer.lock().unwrap().refresh_error = Some("temporarily_unavailable");
    let dir = tempfile::tempdir().unwrap();
    store(dir.path(), &provider, Some("refresh-1"), 0);

    let error = current_token(dir.path(), &provider, 1_000)
        .await
        .unwrap_err()
        .to_string();

    assert!(error.contains("temporarily_unavailable"), "{error}");
    assert!(!error.contains("auth login"), "{error}");
}
