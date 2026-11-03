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
const LOCK_WAIT: Duration = Duration::from_secs(10);
const HTTP_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, Args)]
pub(crate) struct AuthCommand {
    #[command(subcommand)]
    pub(crate) command: AuthSubcommand,
}

#[derive(Debug, Clone, Subcommand)]
pub(crate) enum AuthSubcommand {
    /// Sign in to an OpenID Connect provider with a code shown in the terminal.
    Login(LoginArgs),
    /// Print a current access token for a stored sign-in, refreshing it near expiry.
    Token(ProviderArgs),
    /// Remove the stored sign-in for a provider.
    Logout(ProviderArgs),
}

#[derive(Debug, Clone, Args)]
pub(crate) struct ProviderArgs {
    /// OpenID Connect issuer URL, for example `https://clerk.openrouter.ai`.
    #[arg(long)]
    pub(crate) issuer: String,
    /// OAuth client identifier registered with the issuer.
    #[arg(long)]
    pub(crate) client_id: String,
}

#[derive(Debug, Clone, Args)]
pub(crate) struct LoginArgs {
    #[command(flatten)]
    pub(crate) provider: ProviderArgs,
    /// Space-separated scopes to request. Include `offline_access` so the sign-in can refresh.
    #[arg(long, default_value = "openid offline_access")]
    pub(crate) scope: String,
}

pub(super) async fn execute(command: AuthCommand) -> Result<ExitCode, CliError> {
    let dir = sessions_dir()?;
    match command.command {
        AuthSubcommand::Login(args) => {
            let client = OidcClient::discover(&args.provider).await?;
            let session = client
                .device_login(&args.scope, |prompt| eprintln!("{prompt}"))
                .await?;
            save_session(&dir, &args.provider, &session)?;
            eprintln!("Signed in to {}.", args.provider.issuer);
        }
        AuthSubcommand::Token(provider) => {
            println!("{}", current_token(&dir, &provider, now_secs()).await?);
        }
        AuthSubcommand::Logout(provider) => {
            match std::fs::remove_file(session_path(&dir, &provider)) {
                Ok(()) => eprintln!("Removed the sign-in for {}.", provider.issuer),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    eprintln!("No sign-in is stored for {}.", provider.issuer);
                }
                Err(error) => return Err(CliError::Io(error)),
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
    Refused(String),
}

struct OidcClient {
    http: reqwest::Client,
    issuer: String,
    client_id: String,
    token_endpoint: String,
    device_endpoint: Option<String>,
}

impl OidcClient {
    async fn discover(provider: &ProviderArgs) -> Result<Self, CliError> {
        let issuer = provider.issuer.trim_end_matches('/');
        secure_url(issuer)?;
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
        secure_url(&discovery.token_endpoint)?;
        if let Some(endpoint) = &discovery.device_authorization_endpoint {
            secure_url(endpoint)?;
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
        self.poll_device(&start).await
    }

    async fn poll_device(&self, start: &DeviceStart) -> Result<Session, CliError> {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(start.expires_in);
        let mut interval = start.interval.unwrap_or(5);
        loop {
            tokio::time::sleep(Duration::from_secs(interval)).await;
            if tokio::time::Instant::now() >= deadline {
                return Err(auth_error(
                    "the sign-in code expired; run the login again".into(),
                ));
            }
            let grant = self
                .token_request(&[
                    ("grant_type", DEVICE_CODE_GRANT),
                    ("device_code", &start.device_code),
                    ("client_id", &self.client_id),
                ])
                .await?;
            match grant {
                Grant::Issued(reply) => return self.session(reply, None, now_secs()),
                Grant::Refused(error) if error == "authorization_pending" => {}
                Grant::Refused(error) if error == "slow_down" => interval += 5,
                Grant::Refused(error) => {
                    return Err(auth_error(format!(
                        "the sign-in was not completed: {error}"
                    )));
                }
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
            Grant::Refused(_) => Err(login_required(&self.issuer, &self.client_id)),
        }
    }

    async fn token_request(&self, form: &[(&str, &str)]) -> Result<Grant, CliError> {
        let reply: TokenReply = self.post(&self.token_endpoint, form).await?.json().await?;
        Ok(match (&reply.access_token, reply.error.clone()) {
            (Some(_), None) => Grant::Issued(reply),
            (_, error) => Grant::Refused(error.unwrap_or_else(|| "invalid_response".into())),
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
async fn current_token(dir: &Path, provider: &ProviderArgs, now: u64) -> Result<String, CliError> {
    let path = session_path(dir, provider);
    let _lock = SessionLock::acquire(&path).await?;
    let session =
        load_session(&path).ok_or_else(|| login_required(&provider.issuer, &provider.client_id))?;
    if session.expires_at > now.saturating_add(REFRESH_MARGIN_SECS) {
        return Ok(session.access_token);
    }
    let refresh_token = session
        .refresh_token
        .ok_or_else(|| login_required(&provider.issuer, &provider.client_id))?;
    let client = OidcClient::discover(provider).await?;
    let renewed = client.refresh(&refresh_token, now).await?;
    save_session(dir, provider, &renewed)?;
    Ok(renewed.access_token)
}

fn load_session(path: &Path) -> Option<Session> {
    serde_json::from_slice(&std::fs::read(path).ok()?).ok()
}

fn save_session(dir: &Path, provider: &ProviderArgs, session: &Session) -> Result<(), CliError> {
    if !dir.exists() {
        if let Some(parent) = dir.parent() {
            crate::filesystem::create_private_dir_all(parent)?;
        }
        crate::filesystem::create_private_dir(dir)?;
    }
    let bytes = serde_json::to_vec(session)
        .map_err(|error| auth_error(format!("failed to encode the sign-in: {error}")))?;
    crate::filesystem::atomic_write_private(&session_path(dir, provider), &bytes)
        .map_err(|error| auth_error(format!("failed to save the sign-in: {error}")))
}

/// One file per issuer and client, so sign-ins to several providers do not collide.
fn session_path(dir: &Path, provider: &ProviderArgs) -> PathBuf {
    let digest = Sha256::digest(format!(
        "{}\n{}",
        provider.issuer.trim_end_matches('/'),
        provider.client_id
    ));
    let name: String = digest[..8]
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

/// Serializes refreshes across processes, so a rotated refresh token is never used twice.
struct SessionLock(std::fs::File);

impl SessionLock {
    async fn acquire(session: &Path) -> Result<Self, CliError> {
        let Some(dir) = session.parent() else {
            return Err(auth_error("the sign-in path has no directory".into()));
        };
        if !dir.exists() {
            return Err(login_required_for_path());
        }
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
                    "another token refresh did not finish in time".into(),
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

fn secure_url(value: &str) -> Result<(), CliError> {
    let url = reqwest::Url::parse(value)
        .map_err(|_| auth_error(format!("{value} is not a valid URL")))?;
    let loopback = matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "[::1]"));
    if url.scheme() == "https" || (url.scheme() == "http" && loopback) {
        Ok(())
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

fn login_required(issuer: &str, client_id: &str) -> CliError {
    auth_error(format!(
        "no usable sign-in for {issuer}; run `nemo-relay auth login --issuer {issuer} --client-id {client_id}`"
    ))
}

fn login_required_for_path() -> CliError {
    auth_error("no sign-in is stored; run `nemo-relay auth login` first".into())
}

fn auth_error(message: String) -> CliError {
    CliError::Config(message)
}

#[cfg(test)]
#[path = "../../tests/coverage/commands/auth_tests.rs"]
mod tests;
