// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OpenID Connect sign-in for upstream providers.
//!
//! `[upstream.openai_oidc]` and `[upstream.anthropic_oidc]` name an issuer and a client. A user
//! signs in once in the browser with an authorization code and PKCE (RFC 7636, RFC 8252), or with
//! the device authorization grant (RFC 8628) on request; the sign-in is kept in
//! the per-user NeMo Relay config directory with owner-only access. The gateway reads it for each
//! upstream request and refreshes the access token near expiry, so the token's real expiry and a
//! `logout` apply to the next request. Tokens are written only to that file and to the standard
//! output of `nemo-relay auth token`; they are never logged or included in an error.

mod browser;
mod client;
mod consent;
mod error;
mod session;

use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use self::client::{ISSUER_UNREACHABLE, OidcClient, secure_origin};
use self::session::{
    Session, SessionLock, ensure_sessions_dir, lock_timeout, read_session, session_path,
    write_session,
};
use crate::error::CliError;

pub(crate) use self::browser::BrowserApproval;
pub(crate) use self::client::Revocation;
pub(crate) use self::consent::{ConsentField, CreditLimitReset};
pub(crate) use self::error::SignInError;
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

/// `[upstream.<provider>_oidc]`: the issuer and client that Relay signs in to.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OidcSource {
    /// OpenID Connect issuer URL.
    pub(crate) issuer: String,
    /// OAuth client identifier registered with the issuer.
    pub(crate) client_id: String,
    /// Space-separated scopes requested at sign-in.
    #[serde(default = "default_scope")]
    pub(crate) scope: String,
    /// Consent restriction: the only organization the consent page offers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) organization_id: Option<String>,
    /// Consent restriction: the only workspace the consent page offers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) workspace_id: Option<String>,
    /// Consent restriction: the credit limit, in USD, of the key the sign-in yields.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) credit_limit: Option<serde_json::Number>,
    /// Consent restriction: how often that credit limit resets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) credit_limit_reset: Option<CreditLimitReset>,
    /// Consent restriction: consent page controls to hide.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) hidden_fields: Vec<ConsentField>,
}

impl OidcSource {
    pub(crate) fn new(issuer: &str, client_id: &str, scope: Option<&str>) -> Self {
        Self {
            issuer: issuer.trim_end_matches('/').to_owned(),
            client_id: client_id.to_owned(),
            scope: scope.map_or_else(default_scope, str::to_owned),
            organization_id: None,
            workspace_id: None,
            credit_limit: None,
            credit_limit_reset: None,
            hidden_fields: Vec::new(),
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
        secure_origin(self.issuer()).map_err(|_| {
            CliError::Config(format!(
                "{name}.issuer must be a valid https URL, or http on a loopback address"
            ))
        })?;
        self.validate_consent_restrictions(name)
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

/// How `login` asks the user to approve the sign-in.
pub(crate) enum LoginMethod<'a> {
    /// Authorization code with PKCE, returned to a loopback redirect or pasted back.
    Browser(BrowserApproval<'a>),
    /// The device authorization grant: a code entered on the issuer's own page.
    Device,
}

/// The result of `logout`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Logout {
    NotSignedIn,
    /// The stored sign-in was removed, and its refresh token revoked or not.
    Removed(Revocation),
}

/// Signs in and stores the session. A sign-in it replaces is revoked at the issuer.
pub(crate) async fn login(
    dir: &Path,
    source: &OidcSource,
    method: LoginMethod<'_>,
    show: impl FnMut(String),
) -> Result<(), CliError> {
    let client = OidcClient::discover(source).await?;
    let session = sign_in(&client, source, method, show).await?;
    let replaced = save_session(dir, source, &session).await?;
    if let Some(old_token) = replaced.and_then(|old| old.refresh_token)
        && session.refresh_token.as_deref() != Some(old_token.as_str())
    {
        client.revoke(&old_token).await;
    }
    Ok(())
}

async fn sign_in(
    client: &OidcClient,
    source: &OidcSource,
    method: LoginMethod<'_>,
    show: impl FnMut(String),
) -> Result<Session, CliError> {
    match method {
        LoginMethod::Browser(approval) => {
            let state = source.authorization_state(&browser::random_value()?);
            browser::sign_in(client, &source.scope, &state, approval, show).await
        }
        LoginMethod::Device if source.has_consent_restrictions() => {
            Err(SignInError::DeviceCannotCarryRestrictions.into())
        }
        LoginMethod::Device => client.device_login(&source.scope, show).await,
    }
}

/// Writes the session and returns the one it replaced.
async fn save_session(
    dir: &Path,
    source: &OidcSource,
    session: &Session,
) -> Result<Option<Session>, CliError> {
    ensure_sessions_dir(dir)?;
    let path = session_path(dir, source);
    let _lock = SessionLock::acquire(&path).await?;
    let replaced = read_session(&path, source).ok().flatten();
    write_session(&path, session)?;
    Ok(replaced)
}

/// Removes a stored sign-in, then revokes its refresh token at the issuer. The local sign-in is
/// removed even when the issuer cannot be reached.
pub(crate) async fn logout(dir: &Path, source: &OidcSource) -> Result<Logout, CliError> {
    let outcome = match remove_session(dir, source).await? {
        RemovedSession::Absent => Logout::NotSignedIn,
        RemovedSession::Removed(Session {
            refresh_token: Some(refresh_token),
            ..
        }) => Logout::Removed(revoke_at_issuer(source, &refresh_token).await),
        RemovedSession::Removed(_) | RemovedSession::Unreadable => {
            Logout::Removed(Revocation::NothingToRevoke)
        }
    };
    Ok(outcome)
}

async fn revoke_at_issuer(source: &OidcSource, refresh_token: &str) -> Revocation {
    match OidcClient::discover(source).await {
        Ok(client) => client.revoke(refresh_token).await,
        Err(_) => Revocation::Failed(ISSUER_UNREACHABLE.into()),
    }
}

/// What deleting the stored sign-in file found.
enum RemovedSession {
    Absent,
    /// The file existed but did not hold a sign-in for this source.
    Unreadable,
    Removed(Session),
}

async fn remove_session(dir: &Path, source: &OidcSource) -> Result<RemovedSession, CliError> {
    if !dir.exists() {
        return Ok(RemovedSession::Absent);
    }
    let path = session_path(dir, source);
    let _lock = SessionLock::acquire(&path).await?;
    let stored = read_session(&path, source).ok().flatten();
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(stored.map_or(RemovedSession::Unreadable, RemovedSession::Removed)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(RemovedSession::Absent),
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
