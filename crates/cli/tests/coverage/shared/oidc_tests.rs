// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use axum::Router;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use percent_encoding::percent_decode_str;
use serde_json::json;

use std::path::Path;
use std::time::Duration;

use super::client::{DEVICE_CODE_GRANT, DeviceStart, OidcClient, secure_origin};
use super::session::{Session, ensure_sessions_dir, read_session, session_path, write_session};
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
    omit_expires_in: bool,
    /// A refresh moves this clock forward by the given seconds, like a slow issuer.
    refresh_takes: Option<(Arc<AtomicU64>, u64)>,
    grants: Vec<HashMap<String, String>>,
    /// The `code_challenge` of the last authorization address the browser opened.
    challenge: Option<String>,
    revocation_status: Option<StatusCode>,
    revoked: Vec<HashMap<String, String>>,
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
        "authorization_endpoint": format!("{base}/authorize"),
        "device_authorization_endpoint": format!("{base}/device"),
        "revocation_endpoint": format!("{base}/revoke"),
    }))
}

async fn revoke(State(issuer): State<Shared>, body: String) -> Response {
    let mut state = issuer.lock().unwrap();
    state.revoked.push(parse_form(&body));
    state
        .revocation_status
        .unwrap_or(StatusCode::OK)
        .into_response()
}

async fn device(body: String) -> Response {
    if parse_form(&body)["client_id"] == "no-device-grant" {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": "unauthorized_client",
                "error_description": "The device grant is not enabled for this client",
            })),
        )
            .into_response();
    }
    Json(json!({
        "device_code": "device-1",
        "user_code": "WDJB-MJHT",
        "verification_uri": "https://issuer.example/device",
        "expires_in": 60,
        "interval": 1,
    }))
    .into_response()
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
        "authorization_code" => {
            let challenge = super::browser::code_challenge(&form["code_verifier"]);
            if form["code"] == "code-1" && state.challenge.as_deref() == Some(challenge.as_str()) {
                issued("access-1", "refresh-1")
            } else {
                refused("invalid_grant")
            }
        }
        "refresh_token" => {
            if let Some((clock, seconds)) = &state.refresh_takes {
                clock.fetch_add(*seconds, Ordering::SeqCst);
            }
            refresh_reply(&state, &form)
        }
        _ => refused("unsupported_grant_type"),
    }
}

fn refresh_reply(state: &Issuer, form: &HashMap<String, String>) -> Response {
    match (state.refresh_status, state.refresh_error) {
        (Some(status), _) => (status, "upstream down").into_response(),
        (None, Some(error)) => refused(error),
        (None, None) if form["refresh_token"] == "refresh-1" && state.omit_expires_in => {
            Json(json!({ "access_token": "access-2", "token_type": "Bearer" })).into_response()
        }
        (None, None) if form["refresh_token"] == "refresh-1" => issued("access-2", "refresh-2"),
        (None, None) => refused("invalid_grant"),
    }
}

async fn start_issuer() -> (OidcSource, Shared) {
    let shared: Shared = Arc::default();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    shared.lock().unwrap().base = base.clone();
    let app = Router::new()
        .route("/.well-known/openid-configuration", get(discovery))
        .route("/device", post(device))
        .route("/token", post(token))
        .route("/revoke", post(revoke))
        .with_state(shared.clone());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (OidcSource::new(&base, CLIENT_ID, None), shared)
}

fn store(dir: &Path, source: &OidcSource, refresh: Option<&str>, expires_at: u64) {
    ensure_sessions_dir(dir).unwrap();
    write_session(
        &session_path(dir, source),
        &Session {
            issuer: source.issuer.clone(),
            client_id: source.client_id.clone(),
            access_token: "access-1".into(),
            refresh_token: refresh.map(str::to_owned),
            expires_at,
        },
    )
    .unwrap();
}

fn saved(dir: &Path, source: &OidcSource) -> Session {
    read_session(&session_path(dir, source), source)
        .unwrap()
        .unwrap()
}

fn device_start(interval: u64, expires_in: u64) -> DeviceStart {
    DeviceStart {
        device_code: "device-1".into(),
        user_code: "WDJB-MJHT".into(),
        verification_uri: "https://issuer.example/device".into(),
        verification_uri_complete: None,
        expires_in,
        interval: Some(interval),
    }
}

/// Polls with one millisecond standing in for one second.
async fn poll(
    issuer: &Shared,
    source: &OidcSource,
    replies: &[&'static str],
    start: DeviceStart,
) -> Result<Session, CliError> {
    issuer.lock().unwrap().device_replies = replies.iter().copied().collect();
    let client = OidcClient::discover(source).await.unwrap();
    client.poll_device(&start, Duration::from_millis(1)).await
}

fn error_text<T>(result: Result<T, CliError>) -> String {
    match result {
        Ok(_) => panic!("expected an error"),
        Err(error) => error.to_string(),
    }
}

#[tokio::test]
async fn device_login_shows_the_code_and_returns_the_issued_tokens() {
    let (source, issuer) = start_issuer().await;
    issuer.lock().unwrap().device_replies = ["authorization_pending", "issued"].into();
    let mut prompts = Vec::new();

    let client = OidcClient::discover(&source).await.unwrap();
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
    let (source, issuer) = start_issuer().await;

    let session = poll(
        &issuer,
        &source,
        &["slow_down", "unavailable", "issued"],
        device_start(1, 60),
    )
    .await
    .unwrap();

    assert_eq!(session.access_token, "access-1");
    assert_eq!(issuer.lock().unwrap().grants.len(), 3);
}

#[tokio::test]
async fn device_polling_retries_an_unreachable_issuer_until_the_code_expires() {
    let (source, _issuer) = start_issuer().await;
    let mut client = OidcClient::discover(&source).await.unwrap();
    let closed = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    client.token_endpoint = format!("http://{}/token", closed.local_addr().unwrap());
    drop(closed);

    let error = error_text(
        client
            .poll_device(&device_start(1, 30), Duration::from_millis(1))
            .await,
    );

    assert!(error.contains("expired"), "{error}");
}

#[tokio::test]
async fn device_polling_keeps_the_issuers_long_interval_and_stops_before_an_expired_poll() {
    let (source, issuer) = start_issuer().await;

    let error = error_text(poll(&issuer, &source, &["issued"], device_start(120, 100)).await);

    assert!(error.contains("expired"), "{error}");
    assert!(issuer.lock().unwrap().grants.is_empty());
}

#[tokio::test]
async fn device_polling_stops_when_the_user_declines() {
    let (source, issuer) = start_issuer().await;

    let error = error_text(poll(&issuer, &source, &["access_denied"], device_start(1, 60)).await);

    assert!(error.contains("declined"), "{error}");
}

#[tokio::test]
async fn gateway_token_reuses_a_fresh_access_token_without_calling_the_issuer() {
    let (source, issuer) = start_issuer().await;
    let dir = tempfile::tempdir().unwrap();
    store(dir.path(), &source, Some("refresh-1"), 10_000);

    let token = upstream_token_in(dir.path(), &source, &|| 1_000)
        .await
        .unwrap();

    assert_eq!(token, "access-1");
    assert!(issuer.lock().unwrap().grants.is_empty());
}

#[tokio::test]
async fn gateway_token_follows_the_token_expiry_not_a_fixed_cache_interval() {
    let (source, issuer) = start_issuer().await;
    let dir = tempfile::tempdir().unwrap();
    store(
        dir.path(),
        &source,
        Some("refresh-1"),
        1_000 + REFRESH_MARGIN_SECS - 1,
    );

    let token = upstream_token_in(dir.path(), &source, &|| 1_000)
        .await
        .unwrap();

    assert_eq!(token, "access-2");
    assert_eq!(issuer.lock().unwrap().grants.len(), 1);
    let session = saved(dir.path(), &source);
    assert_eq!(session.refresh_token.as_deref(), Some("refresh-2"));
    assert_eq!(session.expires_at, 1_000 + 3600);
}

#[tokio::test]
async fn logout_applies_to_the_next_gateway_request() {
    let (source, _issuer) = start_issuer().await;
    let dir = tempfile::tempdir().unwrap();
    store(dir.path(), &source, Some("refresh-1"), 10_000);
    assert_eq!(
        upstream_token_in(dir.path(), &source, &|| 1_000)
            .await
            .unwrap(),
        "access-1"
    );

    assert_eq!(
        logout(dir.path(), &source).await.unwrap(),
        Logout::Removed(Revocation::Confirmed)
    );
    let error = error_text(upstream_token_in(dir.path(), &source, &|| 1_000).await);

    assert!(error.contains("run `nemo-relay auth login`"), "{error}");
    assert_eq!(
        logout(dir.path(), &source).await.unwrap(),
        Logout::NotSignedIn
    );
}

#[tokio::test]
async fn gateway_error_without_a_sign_in_names_the_login_command() {
    let dir = tempfile::tempdir().unwrap();
    let source = OidcSource::new("https://issuer.example", CLIENT_ID, None);

    let error = error_text(upstream_token_in(dir.path(), &source, &|| 1_000).await);

    assert!(error.contains("run `nemo-relay auth login`"), "{error}");
}

#[tokio::test]
async fn concurrent_token_calls_refresh_once() {
    let (source, issuer) = start_issuer().await;
    let dir = tempfile::tempdir().unwrap();
    store(dir.path(), &source, Some("refresh-1"), 0);

    let (a, b) = tokio::join!(
        current_token(dir.path(), &source, &|| 1_000),
        current_token(dir.path(), &source, &|| 1_000),
    );

    assert_eq!(a.unwrap(), "access-2");
    assert_eq!(b.unwrap(), "access-2");
    assert_eq!(issuer.lock().unwrap().grants.len(), 1);
}

#[tokio::test]
async fn token_asks_for_login_when_the_issuer_rejects_the_refresh_token() {
    let (source, _issuer) = start_issuer().await;
    let dir = tempfile::tempdir().unwrap();
    store(dir.path(), &source, Some("revoked"), 0);

    let error = error_text(current_token(dir.path(), &source, &|| 1_000).await);

    assert!(error.contains("nemo-relay auth login"), "{error}");
    assert!(!error.contains("revoked"), "{error}");
}

#[tokio::test]
async fn token_keeps_the_sign_in_when_the_issuer_is_unavailable() {
    let (source, issuer) = start_issuer().await;
    issuer.lock().unwrap().refresh_status = Some(StatusCode::SERVICE_UNAVAILABLE);
    let dir = tempfile::tempdir().unwrap();
    store(dir.path(), &source, Some("refresh-1"), 0);

    let error = error_text(current_token(dir.path(), &source, &|| 1_000).await);

    assert!(!error.contains("auth login"), "{error}");
    assert!(error.contains("try again"), "{error}");
    assert_eq!(
        saved(dir.path(), &source).refresh_token.as_deref(),
        Some("refresh-1")
    );
}

#[tokio::test]
async fn token_keeps_the_sign_in_when_the_issuer_refuses_for_another_reason() {
    let (source, issuer) = start_issuer().await;
    issuer.lock().unwrap().refresh_error = Some("temporarily_unavailable");
    let dir = tempfile::tempdir().unwrap();
    store(dir.path(), &source, Some("refresh-1"), 0);

    let error = error_text(current_token(dir.path(), &source, &|| 1_000).await);

    assert!(error.contains("temporarily_unavailable"), "{error}");
    assert!(!error.contains("auth login"), "{error}");
}

#[tokio::test]
async fn token_uses_an_unexpired_token_that_has_no_refresh_token() {
    let (source, issuer) = start_issuer().await;
    let dir = tempfile::tempdir().unwrap();
    store(dir.path(), &source, None, 1_000 + 60);

    let token = current_token(dir.path(), &source, &|| 1_000).await.unwrap();

    assert_eq!(token, "access-1");
    assert!(issuer.lock().unwrap().grants.is_empty());
}

#[tokio::test]
async fn a_sign_in_for_one_client_is_never_used_for_another() {
    let (source, _issuer) = start_issuer().await;
    let dir = tempfile::tempdir().unwrap();
    store(dir.path(), &source, Some("refresh-1"), 10_000);
    let other = OidcSource::new(&source.issuer, "another-client", None);

    let error = error_text(current_token(dir.path(), &other, &|| 1_000).await);

    assert!(error.contains("nemo-relay auth login"), "{error}");
}

#[test]
fn sign_in_state_reports_what_doctor_shows() {
    let dir = tempfile::tempdir().unwrap();
    let source = OidcSource::new("https://issuer.example", CLIENT_ID, None);
    let state = |now| sign_in_state(dir.path(), &source, now).unwrap();
    assert_eq!(state(1_000), SignInState::NotSignedIn);

    store(dir.path(), &source, Some("refresh-1"), 2_000);
    assert_eq!(state(1_000), SignInState::Valid(2_000));
    assert_eq!(state(3_000), SignInState::ExpiredCanRefresh);

    store(dir.path(), &source, None, 2_000);
    assert_eq!(state(3_000), SignInState::ExpiredSignInAgain);
}

#[tokio::test]
async fn discovery_rejects_an_issuer_that_names_another_issuer() {
    let (source, issuer) = start_issuer().await;
    issuer.lock().unwrap().advertised_issuer = Some("https://elsewhere.example".into());

    let error = error_text(OidcClient::discover(&source).await);

    assert!(error.contains("different issuer"), "{error}");
}

#[tokio::test]
async fn discovery_rejects_a_token_endpoint_on_another_origin() {
    let (source, issuer) = start_issuer().await;
    issuer.lock().unwrap().token_endpoint = Some("https://elsewhere.example/token".into());

    let error = error_text(OidcClient::discover(&source).await);

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
fn a_sign_in_replaces_every_other_credential_source_for_its_provider() {
    let mut config = crate::configuration::GatewayConfig {
        openai_oidc: Some(OidcSource::new("https://issuer.example", CLIENT_ID, None)),
        ..Default::default()
    };
    assert!(crate::upstream_auth::validate_config(&config).is_ok());
    assert!(crate::upstream_auth::mints_credential(&config, false));
    assert!(!crate::upstream_auth::mints_credential(&config, true));

    config.openai_api_key = Some("sk-static".into());
    let error = error_text(crate::upstream_auth::validate_config(&config));
    assert!(error.contains("mutually exclusive"), "{error}");

    config.openai_api_key = None;
    config.openai_oidc = Some(OidcSource::new("http://issuer.example", CLIENT_ID, None));
    let error = error_text(crate::upstream_auth::validate_config(&config));
    assert!(error.contains("upstream.openai_oidc.issuer"), "{error}");
}

#[tokio::test]
async fn a_refused_device_sign_in_shows_the_issuers_error() {
    let (source, _issuer) = start_issuer().await;
    let source = OidcSource::new(&source.issuer, "no-device-grant", None);
    let client = OidcClient::discover(&source).await.unwrap();

    let error = error_text(client.device_login("openid", |_| {}).await);

    assert!(error.contains("unauthorized_client"), "{error}");
    assert!(error.contains("device grant is not enabled"), "{error}");
}

#[tokio::test]
async fn a_missing_sign_in_is_reported_as_sign_in_required() {
    let dir = tempfile::tempdir().unwrap();
    let source = OidcSource::new("https://issuer.example", CLIENT_ID, None);

    let error = upstream_token_in(dir.path(), &source, &|| 1_000)
        .await
        .err()
        .unwrap();

    assert!(matches!(error, CliError::SignInRequired(_)), "{error}");
}

#[tokio::test]
async fn a_token_reply_without_expires_in_is_refused_and_keeps_the_old_sign_in() {
    let (source, issuer) = start_issuer().await;
    issuer.lock().unwrap().omit_expires_in = true;
    let dir = tempfile::tempdir().unwrap();
    store(dir.path(), &source, Some("refresh-1"), 0);

    let error = error_text(current_token(dir.path(), &source, &|| 1_000).await);

    assert!(error.contains("expires_in"), "{error}");
    assert_eq!(saved(dir.path(), &source).access_token, "access-1");
}

#[tokio::test]
async fn a_temporary_refresh_failure_keeps_using_an_unexpired_token() {
    let (source, issuer) = start_issuer().await;
    issuer.lock().unwrap().refresh_status = Some(StatusCode::SERVICE_UNAVAILABLE);
    let dir = tempfile::tempdir().unwrap();
    store(dir.path(), &source, Some("refresh-1"), 1_000 + 60);

    let token = current_token(dir.path(), &source, &|| 1_000).await.unwrap();

    assert_eq!(token, "access-1");
    assert_eq!(issuer.lock().unwrap().grants.len(), 1);
}

fn slow_refresh(issuer: &Shared, seconds: u64) -> Arc<AtomicU64> {
    let clock = Arc::new(AtomicU64::new(1_000));
    issuer.lock().unwrap().refresh_takes = Some((clock.clone(), seconds));
    clock
}

#[tokio::test]
async fn a_token_that_expires_during_a_slow_failed_refresh_is_not_returned() {
    let (source, issuer) = start_issuer().await;
    issuer.lock().unwrap().refresh_status = Some(StatusCode::SERVICE_UNAVAILABLE);
    let clock = slow_refresh(&issuer, 2);
    let dir = tempfile::tempdir().unwrap();
    store(dir.path(), &source, Some("refresh-1"), 1_001);

    let result = current_token(dir.path(), &source, &|| clock.load(Ordering::SeqCst)).await;

    let error = error_text(result);
    assert!(error.contains("try again"), "{error}");
}

#[tokio::test]
async fn a_slow_refresh_reply_does_not_shorten_the_stored_lifetime() {
    let (source, issuer) = start_issuer().await;
    let clock = slow_refresh(&issuer, 2);
    let dir = tempfile::tempdir().unwrap();
    store(dir.path(), &source, Some("refresh-1"), 0);

    let token = current_token(dir.path(), &source, &|| clock.load(Ordering::SeqCst))
        .await
        .unwrap();

    assert_eq!(token, "access-2");
    assert_eq!(saved(dir.path(), &source).expires_at, 1_002 + 3600);
}

#[tokio::test(start_paused = true)]
async fn a_lock_wait_timeout_keeps_using_an_unexpired_token() {
    let (source, issuer) = start_issuer().await;
    let dir = tempfile::tempdir().unwrap();
    store(dir.path(), &source, Some("refresh-1"), 1_000 + 60);
    let _held = super::session::SessionLock::acquire(&session_path(dir.path(), &source))
        .await
        .unwrap();

    let token = current_token(dir.path(), &source, &|| 1_000).await.unwrap();

    assert_eq!(token, "access-1");
    assert!(issuer.lock().unwrap().grants.is_empty());
}

#[tokio::test(start_paused = true)]
async fn a_lock_wait_timeout_does_not_return_an_expired_token() {
    let (source, _issuer) = start_issuer().await;
    let dir = tempfile::tempdir().unwrap();
    store(dir.path(), &source, Some("refresh-1"), 1_000);
    let _held = super::session::SessionLock::acquire(&session_path(dir.path(), &source))
        .await
        .unwrap();

    let error = error_text(current_token(dir.path(), &source, &|| 1_000).await);

    assert!(error.contains("did not finish in time"), "{error}");
}

#[tokio::test]
async fn a_sign_in_for_one_scope_is_never_used_for_another() {
    let (source, _issuer) = start_issuer().await;
    let dir = tempfile::tempdir().unwrap();
    store(dir.path(), &source, Some("refresh-1"), 10_000);
    let other = OidcSource::new(&source.issuer, CLIENT_ID, Some("openid other-scope"));

    let error = error_text(current_token(dir.path(), &other, &|| 1_000).await);

    assert!(error.contains("nemo-relay auth login"), "{error}");
    assert_eq!(saved(dir.path(), &source).access_token, "access-1");
}

/// The query of the authorization address, by name.
fn query(url: &str) -> HashMap<String, String> {
    reqwest::Url::parse(url)
        .unwrap()
        .query_pairs()
        .map(|(key, value)| (key.into_owned(), value.into_owned()))
        .collect()
}

/// A browser that approves: it records the challenge, then follows the redirect with `code`
/// and the given state (`None` keeps the attempt's own).
fn approving_browser(
    issuer: Shared,
    opened: Arc<Mutex<Vec<String>>>,
    states: Vec<Option<&'static str>>,
) -> impl Fn(&str) -> bool + Send + Sync {
    move |url: &str| {
        let params = query(url);
        issuer.lock().unwrap().challenge = Some(params["code_challenge"].clone());
        opened.lock().unwrap().push(url.to_owned());
        let states = states.clone();
        tokio::spawn(async move {
            for state in states {
                let state = state.map_or_else(|| params["state"].clone(), str::to_owned);
                let mut redirect = reqwest::Url::parse(&params["redirect_uri"]).unwrap();
                redirect
                    .query_pairs_mut()
                    .append_pair("code", "code-1")
                    .append_pair("state", &state);
                reqwest::get(redirect).await.unwrap();
            }
        });
        true
    }
}

fn approval<'a>(
    open: &'a super::browser::Opener,
    pasted: tokio::sync::mpsc::Receiver<String>,
) -> BrowserApproval<'a> {
    BrowserApproval {
        port: 0,
        open,
        pasted,
        timeout: Duration::from_secs(10),
    }
}

fn no_paste() -> tokio::sync::mpsc::Receiver<String> {
    tokio::sync::mpsc::channel(1).1
}

#[tokio::test]
async fn browser_login_exchanges_the_loopback_code_with_the_pkce_verifier() {
    let (source, issuer) = start_issuer().await;
    let dir = tempfile::tempdir().unwrap();
    let opened = Arc::default();
    let browser = approving_browser(issuer.clone(), Arc::clone(&opened), vec![None]);

    login(
        dir.path(),
        &source,
        LoginMethod::Browser(approval(&browser, no_paste())),
        |_| {},
    )
    .await
    .unwrap();

    assert_eq!(saved(dir.path(), &source).access_token, "access-1");
    let url = opened.lock().unwrap()[0].clone();
    let params = query(&url);
    assert!(
        url.starts_with(&format!("{}/authorize?", source.issuer)),
        "{url}"
    );
    assert_eq!(params["response_type"], "code");
    assert_eq!(params["client_id"], CLIENT_ID);
    assert_eq!(params["scope"], "openid offline_access");
    assert_eq!(params["code_challenge_method"], "S256");
    assert!(params["redirect_uri"].starts_with("http://127.0.0.1:"));
    assert!(params["redirect_uri"].ends_with("/callback"));
    assert_eq!(params["state"].len(), 43);
    let grant = issuer.lock().unwrap().grants[0].clone();
    assert_eq!(grant["grant_type"], "authorization_code");
    assert_eq!(grant["redirect_uri"], params["redirect_uri"]);
}

#[tokio::test]
async fn a_redirect_for_another_attempt_does_not_end_the_sign_in() {
    let (source, issuer) = start_issuer().await;
    let opened = Arc::default();
    let browser = approving_browser(issuer.clone(), opened, vec![Some("forged"), None]);
    let client = OidcClient::discover(&source).await.unwrap();

    let session = super::browser::sign_in(
        &client,
        "openid",
        "attempt-state",
        approval(&browser, no_paste()),
        |_| {},
    )
    .await
    .unwrap();

    assert_eq!(session.access_token, "access-1");
    assert_eq!(issuer.lock().unwrap().grants.len(), 1);
}

#[tokio::test]
async fn a_pasted_redirect_completes_a_sign_in_without_a_local_browser() {
    let (source, issuer) = start_issuer().await;
    let client = OidcClient::discover(&source).await.unwrap();
    let captured = issuer.clone();
    let printed = move |url: &str| {
        captured.lock().unwrap().challenge = Some(query(url)["code_challenge"].clone());
        false
    };
    let (paste, pasted) = tokio::sync::mpsc::channel(8);
    for line in [
        "not a url\n".to_owned(),
        "http://127.0.0.1:1/callback?code=code-1&state=other-attempt\n".to_owned(),
        "<http://127.0.0.1:1/callback?code=code-1&state=attempt-state>\n".to_owned(),
    ] {
        paste.send(line).await.unwrap();
    }
    let mut shown = Vec::new();

    let session = super::browser::sign_in(
        &client,
        "openid",
        "attempt-state",
        approval(&printed, pasted),
        |line| shown.push(line),
    )
    .await
    .unwrap();

    assert_eq!(session.refresh_token.as_deref(), Some("refresh-1"));
    let shown = shown.join("\n");
    assert!(shown.contains("open this address in a browser"), "{shown}");
    assert!(shown.contains("not a sign-in redirect"), "{shown}");
    assert!(shown.contains("another sign-in attempt"), "{shown}");
}

#[tokio::test]
async fn a_declined_browser_sign_in_reports_it() {
    let (source, _issuer) = start_issuer().await;
    let client = OidcClient::discover(&source).await.unwrap();
    let printed = |_: &str| false;
    let (paste, pasted) = tokio::sync::mpsc::channel(1);
    paste
        .send("http://127.0.0.1:1/callback?error=access_denied&state=attempt-state".into())
        .await
        .unwrap();

    let error = error_text(
        super::browser::sign_in(
            &client,
            "openid",
            "attempt-state",
            approval(&printed, pasted),
            |_| {},
        )
        .await,
    );

    assert!(error.contains("declined"), "{error}");
}

#[tokio::test]
async fn an_unanswered_browser_sign_in_times_out() {
    let (source, _issuer) = start_issuer().await;
    let client = OidcClient::discover(&source).await.unwrap();
    let printed = |_: &str| false;
    let mut waiting = approval(&printed, no_paste());
    waiting.timeout = Duration::from_millis(50);

    let error = error_text(super::browser::sign_in(&client, "openid", "s", waiting, |_| {}).await);

    assert!(error.contains("timed out"), "{error}");
}

#[tokio::test]
async fn the_browser_sign_in_carries_the_configured_consent_restriction_in_its_state() {
    let (mut source, issuer) = start_issuer().await;
    source.organization_id = Some("org_required".into());
    source.credit_limit = Some(serde_json::Number::from(25));
    let dir = tempfile::tempdir().unwrap();
    let opened = Arc::default();
    let browser = approving_browser(issuer.clone(), Arc::clone(&opened), vec![None]);

    login(
        dir.path(),
        &source,
        LoginMethod::Browser(approval(&browser, no_paste())),
        |_| {},
    )
    .await
    .unwrap();

    let state = query(&opened.lock().unwrap()[0])["state"].clone();
    let (random, segment) = state.split_once('.').unwrap();
    assert_eq!(random.len(), 43);
    assert_eq!(Some(segment.to_owned()), source.restriction_segment());
}

#[tokio::test]
async fn the_device_sign_in_refuses_consent_restrictions_it_cannot_carry() {
    let (mut source, issuer) = start_issuer().await;
    source.workspace_id = Some("workspace".into());
    let dir = tempfile::tempdir().unwrap();

    let error = error_text(login(dir.path(), &source, LoginMethod::Device, |_| {}).await);

    assert!(error.contains("without --device"), "{error}");
    assert!(issuer.lock().unwrap().grants.is_empty());
}

#[tokio::test]
async fn logout_revokes_the_refresh_token_at_the_issuer() {
    let (source, issuer) = start_issuer().await;
    let dir = tempfile::tempdir().unwrap();
    store(dir.path(), &source, Some("refresh-1"), 10_000);

    let outcome = logout(dir.path(), &source).await.unwrap();

    assert_eq!(outcome, Logout::Removed(Revocation::Confirmed));
    let revoked = issuer.lock().unwrap().revoked.clone();
    assert_eq!(revoked.len(), 1);
    assert_eq!(revoked[0]["token"], "refresh-1");
    assert_eq!(revoked[0]["token_type_hint"], "refresh_token");
    assert_eq!(revoked[0]["client_id"], CLIENT_ID);
}

#[tokio::test]
async fn logout_signs_out_locally_when_the_issuer_refuses_the_revocation() {
    let (source, issuer) = start_issuer().await;
    issuer.lock().unwrap().revocation_status = Some(StatusCode::SERVICE_UNAVAILABLE);
    let dir = tempfile::tempdir().unwrap();
    store(dir.path(), &source, Some("refresh-1"), 10_000);

    let outcome = logout(dir.path(), &source).await.unwrap();

    assert!(
        matches!(outcome, Logout::Removed(Revocation::Failed(_))),
        "{outcome:?}"
    );
    let error = error_text(upstream_token_in(dir.path(), &source, &|| 1_000).await);
    assert!(error.contains("nemo-relay auth login"), "{error}");
}

#[tokio::test]
async fn a_new_sign_in_revokes_the_one_it_replaces() {
    let (source, issuer) = start_issuer().await;
    let dir = tempfile::tempdir().unwrap();
    store(dir.path(), &source, Some("refresh-old"), 10_000);
    let browser = approving_browser(issuer.clone(), Arc::default(), vec![None]);

    login(
        dir.path(),
        &source,
        LoginMethod::Browser(approval(&browser, no_paste())),
        |_| {},
    )
    .await
    .unwrap();

    let revoked = issuer.lock().unwrap().revoked.clone();
    assert_eq!(revoked.len(), 1);
    assert_eq!(revoked[0]["token"], "refresh-old");
}
