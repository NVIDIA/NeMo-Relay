// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use percent_encoding::percent_decode_str;
use serde_json::json;

use super::*;

const CLIENT_ID: &str = "relay-test-client";

#[derive(Default)]
struct Issuer {
    base: String,
    device_polls: usize,
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
    let base = issuer.lock().unwrap().base.clone();
    Json(json!({
        "issuer": base,
        "token_endpoint": format!("{base}/token"),
        "device_authorization_endpoint": format!("{base}/device"),
    }))
}

async fn device(body: String) -> Json<serde_json::Value> {
    let form = parse_form(&body);
    assert_eq!(form["client_id"], CLIENT_ID);
    assert_eq!(form["scope"], "openid offline_access inference");
    Json(json!({
        "device_code": "device-1",
        "user_code": "WDJB-MJHT",
        "verification_uri": "https://issuer.example/device",
        "expires_in": 60,
        "interval": 0,
    }))
}

async fn token(State(issuer): State<Shared>, body: String) -> impl IntoResponse {
    let form = parse_form(&body);
    let mut state = issuer.lock().unwrap();
    state.grants.push(form.clone());
    let issued = |access: &str, refresh: &str| {
        (
            StatusCode::OK,
            Json(json!({
                "access_token": access,
                "refresh_token": refresh,
                "token_type": "Bearer",
                "expires_in": 3600,
            })),
        )
    };
    let refused = |error: &str| (StatusCode::BAD_REQUEST, Json(json!({ "error": error })));
    match form["grant_type"].as_str() {
        DEVICE_CODE_GRANT => {
            state.device_polls += 1;
            if state.device_polls == 1 {
                refused("authorization_pending")
            } else {
                issued("access-1", "refresh-1")
            }
        }
        "refresh_token" if form["refresh_token"] == "refresh-1" => issued("access-2", "refresh-2"),
        _ => refused("invalid_grant"),
    }
}

async fn start_issuer() -> (ProviderArgs, Shared) {
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
        ProviderArgs {
            issuer: base,
            client_id: CLIENT_ID.into(),
        },
        shared,
    )
}

fn stored(dir: &Path, provider: &ProviderArgs, access: &str, expires_at: u64) {
    save_session(
        dir,
        provider,
        &Session {
            issuer: provider.issuer.clone(),
            client_id: provider.client_id.clone(),
            access_token: access.into(),
            refresh_token: Some("refresh-1".into()),
            expires_at,
        },
    )
    .unwrap();
}

#[tokio::test]
async fn device_login_waits_for_approval_and_stores_a_private_session() {
    let (provider, issuer) = start_issuer().await;
    let dir = tempfile::tempdir().unwrap();
    let mut prompts = Vec::new();

    let client = OidcClient::discover(&provider).await.unwrap();
    let session = client
        .device_login("openid offline_access inference", |prompt| {
            prompts.push(prompt)
        })
        .await
        .unwrap();
    save_session(dir.path(), &provider, &session).unwrap();

    assert_eq!(issuer.lock().unwrap().device_polls, 2);
    assert!(prompts.concat().contains("WDJB-MJHT"));
    let path = session_path(dir.path(), &provider);
    let saved: Session = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(saved.access_token, "access-1");
    assert_eq!(saved.refresh_token.as_deref(), Some("refresh-1"));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }
}

#[tokio::test]
async fn token_reuses_a_fresh_access_token_without_calling_the_issuer() {
    let (provider, issuer) = start_issuer().await;
    let dir = tempfile::tempdir().unwrap();
    stored(dir.path(), &provider, "access-1", 10_000);

    let token = current_token(dir.path(), &provider, 1_000).await.unwrap();

    assert_eq!(token, "access-1");
    assert!(issuer.lock().unwrap().grants.is_empty());
}

#[tokio::test]
async fn token_refreshes_near_expiry_and_keeps_the_rotated_refresh_token() {
    let (provider, issuer) = start_issuer().await;
    let dir = tempfile::tempdir().unwrap();
    stored(
        dir.path(),
        &provider,
        "access-1",
        1_000 + REFRESH_MARGIN_SECS - 1,
    );

    let token = current_token(dir.path(), &provider, 1_000).await.unwrap();

    assert_eq!(token, "access-2");
    let grants = issuer.lock().unwrap().grants.clone();
    assert_eq!(grants.len(), 1);
    assert_eq!(grants[0]["client_id"], CLIENT_ID);
    let saved: Session =
        serde_json::from_slice(&std::fs::read(session_path(dir.path(), &provider)).unwrap())
            .unwrap();
    assert_eq!(saved.refresh_token.as_deref(), Some("refresh-2"));
    assert_eq!(saved.expires_at, 1_000 + 3600);
}

#[tokio::test]
async fn token_asks_for_login_when_the_issuer_rejects_the_refresh_token() {
    let (provider, _issuer) = start_issuer().await;
    let dir = tempfile::tempdir().unwrap();
    save_session(
        dir.path(),
        &provider,
        &Session {
            issuer: provider.issuer.clone(),
            client_id: provider.client_id.clone(),
            access_token: "access-1".into(),
            refresh_token: Some("revoked".into()),
            expires_at: 0,
        },
    )
    .unwrap();

    let error = current_token(dir.path(), &provider, 1_000)
        .await
        .unwrap_err()
        .to_string();

    assert!(error.contains("nemo-relay auth login"), "{error}");
    assert!(!error.contains("revoked"), "{error}");
}

#[tokio::test]
async fn token_asks_for_login_without_a_stored_session() {
    let dir = tempfile::tempdir().unwrap();
    let provider = ProviderArgs {
        issuer: "https://issuer.example".into(),
        client_id: CLIENT_ID.into(),
    };

    let error = current_token(dir.path(), &provider, 1_000)
        .await
        .unwrap_err()
        .to_string();

    assert!(error.contains("nemo-relay auth login"), "{error}");
}

#[tokio::test]
async fn discovery_rejects_an_issuer_that_names_another_issuer() {
    let (mut provider, _issuer) = start_issuer().await;
    provider.issuer.push_str("/other");

    assert!(OidcClient::discover(&provider).await.is_err());
}

#[test]
fn issuer_urls_must_use_https_unless_they_are_loopback() {
    assert!(secure_url("https://clerk.openrouter.ai").is_ok());
    assert!(secure_url("http://127.0.0.1:8080").is_ok());
    assert!(secure_url("http://localhost:8080").is_ok());
    assert!(secure_url("http://issuer.example").is_err());
}

#[test]
fn sessions_for_different_clients_use_different_files() {
    let dir = Path::new("/relay");
    let one = ProviderArgs {
        issuer: "https://issuer.example".into(),
        client_id: "a".into(),
    };
    let two = ProviderArgs {
        client_id: "b".into(),
        ..one.clone()
    };
    assert_ne!(session_path(dir, &one), session_path(dir, &two));
}
