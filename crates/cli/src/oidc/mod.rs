// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenID Connect sign-in for upstream providers.
//!
//! `[upstream.openai_oidc]` and `[upstream.anthropic_oidc]` name an issuer and a client. A user
//! signs in once with the OAuth 2.0 device authorization grant (RFC 8628); the sign-in is kept in
//! the per-user NeMo Relay config directory with owner-only access. The gateway reads it for each
//! upstream request and refreshes the access token near expiry, so the token's real expiry and a
//! `logout` apply to the next request. Tokens are written only to that file and to the standard
//! output of `nemo-relay auth token`; they are never logged or included in an error.

mod client;
mod session;

use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use self::client::{OidcClient, secure_origin};
use self::session::{
    SessionLock, ensure_sessions_dir, lock_timeout, read_session, session_path, write_session,
};
use crate::error::CliError;

pub(crate) use self::session::sessions_dir;

/// Refresh an access token this many seconds before it expires.
pub(crate) const REFRESH_MARGIN_SECS: u64 = 300;
/// The current Unix time in seconds. Production passes `now_secs`; tests pass a clock they move.
pub(crate) type Clock<'a> = dyn Fn() -> u64 + Send + Sync + 'a;

/// Bound for one issuer request.
pub(crate) const HTTP_TIMEOUT: Duration = Duration::from_secs(15);

fn default_scope() -> String {
    "openid offline_access".into()
}

/// Built-in settings for a known provider, so an administrator does not copy them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub(crate) enum OidcPreset {
    /// OpenRouter: its issuer, its public client for command-line tools, and the inference scope.
    Openrouter,
}

impl OidcPreset {
    pub(crate) fn source(self) -> OidcSource {
        match self {
            Self::Openrouter => OidcSource::new(
                "https://clerk.openrouter.ai",
                "q3AE4KTq8gq6HlU3",
                Some("openid offline_access email inference"),
            ),
        }
    }
}

/// The fields of `[upstream.<provider>_oidc]` as written; a preset stands for the other three.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OidcSourceFields {
    preset: Option<OidcPreset>,
    issuer: Option<String>,
    client_id: Option<String>,
    scope: Option<String>,
}

impl TryFrom<OidcSourceFields> for OidcSource {
    type Error = String;

    fn try_from(fields: OidcSourceFields) -> Result<Self, Self::Error> {
        if let Some(preset) = fields.preset {
            let written = [
                ("issuer", fields.issuer.is_some()),
                ("client_id", fields.client_id.is_some()),
                ("scope", fields.scope.is_some()),
            ];
            if let Some((name, _)) = written.into_iter().find(|(_, set)| *set) {
                return Err(format!(
                    "preset cannot be combined with {name}; write either the preset or the fields"
                ));
            }
            return Ok(preset.source());
        }
        let issuer = fields
            .issuer
            .ok_or("issuer is required unless preset is set")?;
        let client_id = fields
            .client_id
            .ok_or("client_id is required unless preset is set")?;
        Ok(Self::new(&issuer, &client_id, fields.scope.as_deref()))
    }
}

/// `[upstream.<provider>_oidc]`: the issuer and client that Relay signs in to.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(try_from = "OidcSourceFields")]
pub(crate) struct OidcSource {
    /// OpenID Connect issuer URL.
    pub(crate) issuer: String,
    /// OAuth client identifier registered with the issuer.
    pub(crate) client_id: String,
    /// Space-separated scopes requested at sign-in.
    pub(crate) scope: String,
}

impl OidcSource {
    pub(crate) fn new(issuer: &str, client_id: &str, scope: Option<&str>) -> Self {
        Self {
            issuer: issuer.trim_end_matches('/').to_owned(),
            client_id: client_id.to_owned(),
            scope: scope.map_or_else(default_scope, str::to_owned),
        }
    }

    /// The issuer without a trailing slash, as discovery reports it.
    fn issuer(&self) -> &str {
        self.issuer.trim_end_matches('/')
    }

    /// Checks the issuer when the configuration loads.
    pub(crate) fn validate(&self, name: &str) -> Result<(), CliError> {
        if self.client_id.trim().is_empty() {
            return Err(CliError::Config(format!(
                "{name}.client_id must not be empty"
            )));
        }
        secure_origin(self.issuer()).map(|_| ()).map_err(|_| {
            CliError::Config(format!(
                "{name}.issuer must be a valid https URL, or http on a loopback address"
            ))
        })
    }
}

/// What `nemo-relay doctor` reports about a configured sign-in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SignInState {
    NotSignedIn,
    /// The access token is valid until the given Unix time.
    Valid(u64),
    /// The access token has expired; the stored refresh token can renew it.
    ExpiredCanRefresh,
    /// The access token has expired and no refresh token is stored.
    ExpiredSignInAgain,
}

/// The access token for one upstream request. The common case reads the stored sign-in without
/// a lock; only a refresh takes the lock.
pub(crate) async fn upstream_token(source: &OidcSource) -> Result<String, CliError> {
    upstream_token_in(&sessions_dir()?, source, &now_secs).await
}

pub(crate) async fn upstream_token_in(
    dir: &Path,
    source: &OidcSource,
    now: &Clock<'_>,
) -> Result<String, CliError> {
    if let Some(session) = read_session(&session_path(dir, source), source)?
        && session.expires_at > now().saturating_add(REFRESH_MARGIN_SECS)
    {
        return Ok(session.access_token);
    }
    current_token(dir, source, now).await
}

/// Returns a usable access token, refreshing and saving the session when it is near expiry.
pub(crate) async fn current_token(
    dir: &Path,
    source: &OidcSource,
    now: &Clock<'_>,
) -> Result<String, CliError> {
    let path = session_path(dir, source);
    if !dir.exists() {
        return Err(login_required(source.issuer()));
    }
    let Some(_lock) = SessionLock::try_acquire(&path).await? else {
        // A slow update by another request must not drop a token that is still valid.
        return match read_session(&path, source)? {
            Some(session) if session.expires_at > now() => Ok(session.access_token),
            _ => Err(lock_timeout()),
        };
    };
    let session = read_session(&path, source)?.ok_or_else(|| login_required(source.issuer()))?;
    if session.expires_at > now().saturating_add(REFRESH_MARGIN_SECS) {
        return Ok(session.access_token);
    }
    let Some(refresh_token) = session.refresh_token else {
        // Without a refresh token, use the access token until it actually expires.
        return if session.expires_at > now() {
            Ok(session.access_token)
        } else {
            Err(login_required(source.issuer()))
        };
    };
    let attempt = async {
        OidcClient::discover(source)
            .await?
            .refresh(&refresh_token, now)
            .await
    };
    match attempt.await {
        Ok(renewed) => {
            write_session(&path, &renewed)?;
            Ok(renewed.access_token)
        }
        // A temporary issuer failure must not drop a token that is still valid. Read the clock
        // again: the lock wait, discovery and the refresh request all take time.
        Err(error)
            if !matches!(error, CliError::SignInRequired(_)) && session.expires_at > now() =>
        {
            Ok(session.access_token)
        }
        Err(error) => Err(error),
    }
}

/// Signs in and stores the session.
pub(crate) async fn login(
    dir: &Path,
    source: &OidcSource,
    show: impl FnMut(String),
) -> Result<(), CliError> {
    let client = OidcClient::discover(source).await?;
    let session = client.device_login(&source.scope, show).await?;
    ensure_sessions_dir(dir)?;
    let path = session_path(dir, source);
    let _lock = SessionLock::acquire(&path).await?;
    write_session(&path, &session)
}

/// Removes a stored sign-in; returns whether one existed.
pub(crate) async fn logout(dir: &Path, source: &OidcSource) -> Result<bool, CliError> {
    if !dir.exists() {
        return Ok(false);
    }
    let path = session_path(dir, source);
    let _lock = SessionLock::acquire(&path).await?;
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(CliError::Io(error)),
    }
}

/// Reads the stored sign-in without contacting the issuer.
pub(crate) fn sign_in_state(
    dir: &Path,
    source: &OidcSource,
    now: u64,
) -> Result<SignInState, CliError> {
    let Some(session) = read_session(&session_path(dir, source), source)? else {
        return Ok(SignInState::NotSignedIn);
    };
    Ok(if session.expires_at > now {
        SignInState::Valid(session.expires_at)
    } else if session.refresh_token.is_some() {
        SignInState::ExpiredCanRefresh
    } else {
        SignInState::ExpiredSignInAgain
    })
}

pub(crate) fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

/// The gateway returns this text as the request error, so it names the exact fix.
pub(crate) fn login_required(issuer: &str) -> CliError {
    CliError::SignInRequired(format!(
        "no usable OpenID Connect sign-in for {issuer}; run `nemo-relay auth login`"
    ))
}

fn auth_error(message: String) -> CliError {
    CliError::Config(message)
}

#[cfg(test)]
#[path = "../../tests/coverage/shared/oidc_tests.rs"]
mod tests;
