// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `nemo-relay auth`: OpenID Connect sign-in for upstream providers.
//!
//! `login` runs the OAuth 2.0 device authorization grant (RFC 8628) and keeps the result in the
//! per-user NeMo Relay config directory with owner-only access. `token` prints a current access
//! token and refreshes it near expiry, so an `[upstream.<provider>_api_key_helper]` can run it.
//! Tokens are written only to that file and to the standard output of `token`; they are never
//! logged or included in an error.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use clap::{Args, Subcommand};
use http::header::{ACCEPT, CONTENT_TYPE};
use percent_encoding::{NON_ALPHANUMERIC, utf8_percent_encode};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::CliError;

/// Refresh an access token this many seconds before it expires.
const REFRESH_MARGIN_SECS: u64 = 300;
const DEVICE_CODE_GRANT: &str = "urn:ietf:params:oauth:grant-type:device_code";
const SESSIONS_DIR: &str = "oidc";
const HTTP_TIMEOUT: Duration = Duration::from_secs(30);
/// Longer than one refresh, so a caller waits for a concurrent refresh instead of failing.
const LOCK_WAIT: Duration = Duration::from_secs(45);
/// RFC 8628 section 3.5: the default polling interval, and the increase after `slow_down`.
const DEFAULT_POLL_SECS: u64 = 5;
const SLOW_DOWN_SECS: u64 = 5;
const MAX_POLL_SECS: u64 = 60;
const DEFAULT_SCOPE: &str = "openid offline_access";

#[derive(Debug, Clone, Args)]
pub(crate) struct AuthCommand {
    #[command(subcommand)]
    pub(crate) command: AuthSubcommand,
}

#[derive(Debug, Clone, Subcommand)]
pub(crate) enum AuthSubcommand {
    /// Sign in to an OpenID Connect provider with a code shown in the terminal.
    Login(ProviderArgs),
    /// Print a current access token for a stored sign-in, refreshing it near expiry.
    Token(ProviderArgs),
    /// Remove the stored sign-in for a provider.
    Logout(ProviderArgs),
}

/// Without `--issuer` and `--client-id`, the values come from the configured
/// `[upstream.openai_api_key_helper]` that runs `nemo-relay auth token`.
#[derive(Debug, Clone, Default, Args)]
pub(crate) struct ProviderArgs {
    /// OpenID Connect issuer URL, for example `https://clerk.openrouter.ai`.
    #[arg(long)]
    pub(crate) issuer: Option<String>,
    /// OAuth client identifier registered with the issuer.
    #[arg(long)]
    pub(crate) client_id: Option<String>,
    /// Space-separated scopes for `login`. Include `offline_access` so the sign-in can refresh.
    #[arg(long)]
    pub(crate) scope: Option<String>,
}

/// One OpenID Connect provider, after reading the arguments and the configured helper.
#[derive(Debug, Clone)]
struct Provider {
    issuer: String,
    client_id: String,
    scope: String,
}

impl ProviderArgs {
    fn resolve(self) -> Result<Provider, CliError> {
        let configured = if self.issuer.is_some() && self.client_id.is_some() {
            ProviderArgs::default()
        } else {
            let gateway = crate::configuration::resolve_daemon_config(&Default::default())?;
            gateway
                .openai_api_key_helper
                .map(|helper| provider_from_helper_args(&helper.args))
                .unwrap_or_default()
        };
        let issuer = self.issuer.or(configured.issuer);
        let client_id = self.client_id.or(configured.client_id);
        let (Some(issuer), Some(client_id)) = (issuer, client_id) else {
            return Err(auth_error(
                "pass --issuer and --client-id, or configure [upstream.openai_api_key_helper] to run `nemo-relay auth token` with them".into(),
            ));
        };
        Ok(Provider {
            issuer: issuer.trim_end_matches('/').to_owned(),
            client_id,
            scope: self
                .scope
                .or(configured.scope)
                .unwrap_or_else(|| DEFAULT_SCOPE.to_owned()),
        })
    }
}

/// Reads `--issuer`, `--client-id` and `--scope` from helper arguments that run `auth token`.
fn provider_from_helper_args(args: &[String]) -> ProviderArgs {
    let runs_auth_token = args
        .windows(2)
        .any(|pair| pair[0] == "auth" && pair[1] == "token");
    let mut provider = ProviderArgs::default();
    if !runs_auth_token {
        return provider;
    }
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        let (name, inline) = match arg.split_once('=') {
            Some((name, value)) => (name, Some(value.to_owned())),
            None => (arg.as_str(), None),
        };
        let slot = match name {
            "--issuer" => &mut provider.issuer,
            "--client-id" => &mut provider.client_id,
            "--scope" => &mut provider.scope,
            _ => continue,
        };
        *slot = inline.or_else(|| args.next().cloned());
    }
    provider
}

pub(super) async fn execute(command: AuthCommand) -> Result<ExitCode, CliError> {
    let dir = sessions_dir()?;
    match command.command {
        AuthSubcommand::Login(args) => {
            let provider = args.resolve()?;
            let client = OidcClient::discover(&provider).await?;
            let session = client
                .device_login(&provider.scope, |prompt| eprintln!("{prompt}"))
                .await?;
            ensure_sessions_dir(&dir)?;
            let path = session_path(&dir, &provider);
            let _lock = SessionLock::acquire(&path).await?;
            write_session(&path, &session)?;
            eprintln!("Signed in to {}.", provider.issuer);
        }
        AuthSubcommand::Token(args) => {
            let provider = args.resolve()?;
            println!("{}", current_token(&dir, &provider, now_secs()).await?);
        }
        AuthSubcommand::Logout(args) => {
            let provider = args.resolve()?;
            if remove_session(&dir, &provider).await? {
                eprintln!("Removed the sign-in for {}.", provider.issuer);
            } else {
                eprintln!("No sign-in is stored for {}.", provider.issuer);
            }
        }
    }
    Ok(ExitCode::SUCCESS)
}

/// One stored sign-in. Do not add `Debug`: it holds tokens.
#[derive(Serialize, Deserialize)]
struct Session {
    issuer: String,
    client_id: String,
    access_token: String,
    refresh_token: Option<String>,
    /// Seconds since the Unix epoch.
    expires_at: u64,
}

#[derive(Deserialize)]
struct Discovery {
    issuer: String,
    token_endpoint: String,
    device_authorization_endpoint: Option<String>,
}

#[derive(Deserialize)]
struct DeviceStart {
    device_code: String,
    user_code: String,
    verification_uri: String,
    verification_uri_complete: Option<String>,
    expires_in: u64,
    interval: Option<u64>,
}

#[derive(Deserialize)]
struct TokenReply {
    access_token: Option<String>,
    refresh_token: Option<String>,
    expires_in: Option<u64>,
    error: Option<String>,
}

enum Grant {
    Issued(TokenReply),
    /// An OAuth error code from the issuer.
    Refused(String),
    /// A server error, rate limit or unreadable reply; the request can be repeated.
    Unavailable(String),
}

struct OidcClient {
    http: reqwest::Client,
    issuer: String,
    client_id: String,
    token_endpoint: String,
    device_endpoint: Option<String>,
}

impl OidcClient {
    async fn discover(provider: &Provider) -> Result<Self, CliError> {
        let issuer = provider.issuer.as_str();
        let origin = secure_origin(issuer)?;
        let http = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(HTTP_TIMEOUT)
            .build()?;
        let discovery: Discovery = http
            .get(format!("{issuer}/.well-known/openid-configuration"))
            .header(ACCEPT, "application/json")
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        if discovery.issuer.trim_end_matches('/') != issuer {
            return Err(auth_error(format!(
                "{issuer} published discovery for a different issuer"
            )));
        }
        // Endpoints on another origin could send the refresh token to an unintended host.
        let endpoints = std::iter::once(&discovery.token_endpoint)
            .chain(discovery.device_authorization_endpoint.as_ref());
        for endpoint in endpoints {
            if secure_origin(endpoint)? != origin {
                return Err(auth_error(format!(
                    "{issuer} published an endpoint on another origin: {endpoint}"
                )));
            }
        }
        Ok(Self {
            http,
            issuer: issuer.to_owned(),
            client_id: provider.client_id.clone(),
            token_endpoint: discovery.token_endpoint,
            device_endpoint: discovery.device_authorization_endpoint,
        })
    }

    async fn device_login(
        &self,
        scope: &str,
        mut show: impl FnMut(String),
    ) -> Result<Session, CliError> {
        let endpoint = self.device_endpoint.as_deref().ok_or_else(|| {
            auth_error(format!(
                "{} does not offer device sign-in (no device_authorization_endpoint)",
                self.issuer
            ))
        })?;
        let start: DeviceStart = self
            .post(
                endpoint,
                &[("client_id", &self.client_id), ("scope", scope)],
            )
            .await?
            .error_for_status()?
            .json()
            .await?;
        show(format!(
            "To sign in, open {} and enter the code {}.",
            start
                .verification_uri_complete
                .as_deref()
                .unwrap_or(&start.verification_uri),
            start.user_code
        ));
        self.poll_device(&start, Duration::from_secs(1)).await
    }

    /// `unit` is one second in production; tests shorten it.
    async fn poll_device(&self, start: &DeviceStart, unit: Duration) -> Result<Session, CliError> {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(start.expires_in);
        let mut interval = start
            .interval
            .unwrap_or(DEFAULT_POLL_SECS)
            .clamp(1, MAX_POLL_SECS);
        loop {
            tokio::time::sleep(unit * u32::try_from(interval).unwrap_or(1)).await;
            let grant = self
                .token_request(&[
                    ("grant_type", DEVICE_CODE_GRANT),
                    ("device_code", &start.device_code),
                    ("client_id", &self.client_id),
                ])
                .await?;
            match grant {
                Grant::Issued(reply) => return self.session(reply, None, now_secs()),
                Grant::Refused(error) if error == "slow_down" => {
                    interval = (interval + SLOW_DOWN_SECS).min(MAX_POLL_SECS);
                }
                Grant::Refused(error) if error != "authorization_pending" => {
                    return Err(device_failure(&error));
                }
                Grant::Refused(_) | Grant::Unavailable(_) => {}
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(device_failure("expired_token"));
            }
        }
    }

    async fn refresh(&self, refresh_token: &str, now: u64) -> Result<Session, CliError> {
        let grant = self
            .token_request(&[
                ("grant_type", "refresh_token"),
                ("refresh_token", refresh_token),
                ("client_id", &self.client_id),
            ])
            .await?;
        match grant {
            Grant::Issued(reply) => self.session(reply, Some(refresh_token), now),
            Grant::Refused(error)
                if matches!(error.as_str(), "invalid_grant" | "invalid_client") =>
            {
                Err(login_required(&self.issuer, &self.client_id))
            }
            Grant::Refused(error) | Grant::Unavailable(error) => Err(auth_error(format!(
                "{} could not refresh the sign-in ({error}); try again later",
                self.issuer
            ))),
        }
    }

    async fn token_request(&self, form: &[(&str, &str)]) -> Result<Grant, CliError> {
        let response = self.post(&self.token_endpoint, form).await?;
        let status = response.status();
        let Ok(reply) = response.json::<TokenReply>().await else {
            return Ok(Grant::Unavailable(format!("HTTP {status}")));
        };
        Ok(match (&reply.access_token, reply.error.clone()) {
            (Some(_), None) if status.is_success() => Grant::Issued(reply),
            (_, Some(error)) if !status.is_server_error() => Grant::Refused(error),
            _ => Grant::Unavailable(format!("HTTP {status}")),
        })
    }

    async fn post(&self, url: &str, form: &[(&str, &str)]) -> Result<reqwest::Response, CliError> {
        Ok(self
            .http
            .post(url)
            .header(CONTENT_TYPE, "application/x-www-form-urlencoded")
            .header(ACCEPT, "application/json")
            .body(encode_form(form))
            .send()
            .await?)
    }

    /// A refresh reply may omit `refresh_token`; the earlier one then stays valid.
    fn session(
        &self,
        reply: TokenReply,
        previous_refresh: Option<&str>,
        now: u64,
    ) -> Result<Session, CliError> {
        let access_token = reply
            .access_token
            .ok_or_else(|| auth_error("the issuer returned no access token".into()))?;
        Ok(Session {
            issuer: self.issuer.clone(),
            client_id: self.client_id.clone(),
            access_token,
            refresh_token: reply
                .refresh_token
                .or_else(|| previous_refresh.map(str::to_owned)),
            expires_at: now.saturating_add(reply.expires_in.unwrap_or(0)),
        })
    }
}

/// Returns a usable access token, refreshing and saving the session when it is near expiry.
async fn current_token(dir: &Path, provider: &Provider, now: u64) -> Result<String, CliError> {
    let path = session_path(dir, provider);
    if !dir.exists() {
        return Err(login_required(&provider.issuer, &provider.client_id));
    }
    let _lock = SessionLock::acquire(&path).await?;
    let session = read_session(&path, provider)?
        .ok_or_else(|| login_required(&provider.issuer, &provider.client_id))?;
    if session.expires_at > now.saturating_add(REFRESH_MARGIN_SECS) {
        return Ok(session.access_token);
    }
    let Some(refresh_token) = session.refresh_token else {
        // Without a refresh token, use the access token until it actually expires.
        return if session.expires_at > now {
            Ok(session.access_token)
        } else {
            Err(login_required(&provider.issuer, &provider.client_id))
        };
    };
    let client = OidcClient::discover(provider).await?;
    let renewed = client.refresh(&refresh_token, now).await?;
    write_session(&path, &renewed)?;
    Ok(renewed.access_token)
}

/// Returns `None` when no sign-in is stored for this issuer and client.
fn read_session(path: &Path, provider: &Provider) -> Result<Option<Session>, CliError> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(CliError::Io(error)),
    };
    let session: Session = serde_json::from_slice(&bytes).map_err(|_| {
        auth_error(format!(
            "the sign-in file {} is unreadable; run `nemo-relay auth logout` and sign in again",
            path.display()
        ))
    })?;
    let same_provider =
        session.issuer == provider.issuer && session.client_id == provider.client_id;
    Ok(same_provider.then_some(session))
}

fn write_session(path: &Path, session: &Session) -> Result<(), CliError> {
    let bytes = serde_json::to_vec(session)
        .map_err(|error| auth_error(format!("failed to encode the sign-in: {error}")))?;
    crate::filesystem::atomic_write_private(path, &bytes)
        .map_err(|error| auth_error(format!("failed to save the sign-in: {error}")))
}

async fn remove_session(dir: &Path, provider: &Provider) -> Result<bool, CliError> {
    if !dir.exists() {
        return Ok(false);
    }
    let path = session_path(dir, provider);
    let _lock = SessionLock::acquire(&path).await?;
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(CliError::Io(error)),
    }
}

fn ensure_sessions_dir(dir: &Path) -> Result<(), CliError> {
    if dir.exists() {
        return Ok(());
    }
    if let Some(parent) = dir.parent() {
        crate::filesystem::create_private_dir_all(parent)?;
    }
    match crate::filesystem::create_private_dir(dir) {
        Err(error) if error.kind() != std::io::ErrorKind::AlreadyExists => Err(CliError::Io(error)),
        _ => Ok(()),
    }
}

/// One file per issuer and client, so sign-ins to several providers do not collide.
fn session_path(dir: &Path, provider: &Provider) -> PathBuf {
    let digest = Sha256::digest(format!("{}\n{}", provider.issuer, provider.client_id));
    let name: String = digest[..16]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    dir.join(format!("{name}.json"))
}

fn sessions_dir() -> Result<PathBuf, CliError> {
    crate::configuration::user_config_dir()
        .map(|dir| dir.join(SESSIONS_DIR))
        .ok_or_else(|| {
            CliError::Config(
                "cannot determine the per-user NeMo Relay config directory; set HOME or XDG_CONFIG_HOME"
                    .into(),
            )
        })
}

/// Serializes every write of one sign-in across processes, so a rotated refresh token is never
/// used twice and a refresh cannot restore a sign-in that `logout` removed.
struct SessionLock(std::fs::File);

impl SessionLock {
    async fn acquire(session: &Path) -> Result<Self, CliError> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(session.with_extension("lock"))?;
        let deadline = tokio::time::Instant::now() + LOCK_WAIT;
        while crate::filesystem::try_lock_exclusive(&file)?
            == crate::filesystem::LockAttempt::Contended
        {
            if tokio::time::Instant::now() >= deadline {
                return Err(auth_error(
                    "another sign-in update did not finish in time".into(),
                ));
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        Ok(Self(file))
    }
}

impl Drop for SessionLock {
    fn drop(&mut self) {
        let _ = crate::filesystem::unlock_file(&self.0);
    }
}

/// Accepts `https`, or `http` on a loopback host, and returns the URL's origin.
fn secure_origin(value: &str) -> Result<String, CliError> {
    let url = reqwest::Url::parse(value)
        .map_err(|_| auth_error(format!("{value} is not a valid URL")))?;
    let loopback = matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "[::1]"));
    if url.scheme() == "https" || (url.scheme() == "http" && loopback) {
        Ok(url.origin().ascii_serialization())
    } else {
        Err(auth_error(format!(
            "{value} must use https, or http on a loopback address"
        )))
    }
}

fn encode_form(pairs: &[(&str, &str)]) -> String {
    pairs
        .iter()
        .map(|(key, value)| {
            format!(
                "{}={}",
                utf8_percent_encode(key, NON_ALPHANUMERIC),
                utf8_percent_encode(value, NON_ALPHANUMERIC)
            )
        })
        .collect::<Vec<_>>()
        .join("&")
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

fn device_failure(error: &str) -> CliError {
    auth_error(match error {
        "expired_token" => "the sign-in code expired; run the login again".into(),
        "access_denied" => "the sign-in was declined".into(),
        other => format!("the sign-in was not completed: {other}"),
    })
}

fn login_required(issuer: &str, client_id: &str) -> CliError {
    auth_error(format!(
        "no usable sign-in for {issuer} (client {client_id}); run `nemo-relay auth login`"
    ))
}

fn auth_error(message: String) -> CliError {
    CliError::Config(message)
}

#[cfg(test)]
#[path = "../../tests/coverage/commands/auth_tests.rs"]
mod tests;
