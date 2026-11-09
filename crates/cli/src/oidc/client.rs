// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Requests to the OpenID Connect issuer: discovery, the device sign-in, the authorization code
//! exchange and token refresh.

use std::time::Duration;

use http::header::{ACCEPT, CONTENT_TYPE};
use percent_encoding::{NON_ALPHANUMERIC, utf8_percent_encode};
use serde::Deserialize;

use super::error::SignInError;
use super::session::Session;
use super::{Clock, HTTP_TIMEOUT, OidcSource, auth_error, login_required, now_secs};
use crate::error::CliError;

pub(crate) const DEVICE_CODE_GRANT: &str = "urn:ietf:params:oauth:grant-type:device_code";
/// RFC 8628 section 3.5: the default polling interval, and the increase after `slow_down`.
const DEFAULT_POLL_SECS: u64 = 5;
const SLOW_DOWN_SECS: u64 = 5;

#[derive(Deserialize)]
struct Discovery {
    issuer: String,
    token_endpoint: String,
    authorization_endpoint: Option<String>,
    device_authorization_endpoint: Option<String>,
}

const AUTHORIZATION_CODE_GRANT: &str = "authorization_code";
const PKCE_METHOD: &str = "S256";

#[derive(Deserialize)]
pub(crate) struct DeviceStart {
    pub(crate) device_code: String,
    pub(crate) user_code: String,
    pub(crate) verification_uri: String,
    pub(crate) verification_uri_complete: Option<String>,
    pub(crate) expires_in: u64,
    pub(crate) interval: Option<u64>,
}

#[derive(Deserialize)]
struct TokenReply {
    access_token: Option<String>,
    refresh_token: Option<String>,
    expires_in: Option<u64>,
    error: Option<String>,
    error_description: Option<String>,
}

/// An OAuth error reply (RFC 6749 section 5.2).
#[derive(Deserialize)]
struct OAuthError {
    error: String,
    error_description: Option<String>,
}

enum Grant {
    Issued(TokenReply),
    /// An OAuth error from the issuer: its code, and the code with the issuer's description.
    Refused {
        code: String,
        detail: String,
    },
    /// A network failure, server error, rate limit or unreadable reply; it can be repeated.
    Unavailable(String),
}

pub(crate) struct OidcClient {
    http: reqwest::Client,
    issuer: String,
    metadata_issuer: String,
    client_id: String,
    pub(super) token_endpoint: String,
    authorization_endpoint: Option<String>,
    device_endpoint: Option<String>,
}

impl OidcClient {
    pub(crate) async fn discover(source: &OidcSource) -> Result<Self, CliError> {
        let issuer = source.issuer();
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
            .chain(discovery.device_authorization_endpoint.as_ref())
            .chain(discovery.authorization_endpoint.as_ref());
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
            metadata_issuer: discovery.issuer,
            client_id: source.client_id.clone(),
            token_endpoint: discovery.token_endpoint,
            authorization_endpoint: discovery.authorization_endpoint,
            device_endpoint: discovery.device_authorization_endpoint,
        })
    }

    /// The issuer exactly as discovery published it, which RFC 9207 compares `iss` with.
    pub(crate) fn metadata_issuer(&self) -> &str {
        &self.metadata_issuer
    }

    /// The address the user opens to approve the sign-in.
    pub(crate) fn authorization_url(
        &self,
        scope: &str,
        redirect_uri: &str,
        state: &str,
        challenge: &str,
    ) -> Result<String, SignInError> {
        let endpoint = self.authorization_endpoint.as_deref().ok_or_else(|| {
            SignInError::BrowserSignInNotOffered {
                issuer: self.issuer.clone(),
            }
        })?;
        let mut url = reqwest::Url::parse(endpoint).map_err(|_| SignInError::InvalidEndpoint {
            endpoint: endpoint.to_owned(),
        })?;
        url.query_pairs_mut()
            .append_pair("response_type", "code")
            .append_pair("client_id", &self.client_id)
            .append_pair("redirect_uri", redirect_uri)
            .append_pair("scope", scope)
            .append_pair("state", state)
            .append_pair("code_challenge", challenge)
            .append_pair("code_challenge_method", PKCE_METHOD);
        Ok(url.into())
    }

    /// Exchanges an approved authorization code and its PKCE verifier for tokens.
    pub(crate) async fn exchange_code(
        &self,
        code: &str,
        redirect_uri: &str,
        verifier: &str,
    ) -> Result<Session, CliError> {
        let grant = self
            .token_request(&[
                ("grant_type", AUTHORIZATION_CODE_GRANT),
                ("code", code),
                ("redirect_uri", redirect_uri),
                ("client_id", &self.client_id),
                ("code_verifier", verifier),
            ])
            .await;
        match grant {
            Grant::Issued(reply) => self.session(reply, None, now_secs()),
            Grant::Refused { detail, .. } | Grant::Unavailable(detail) => {
                Err(SignInError::TokenNotIssued {
                    issuer: self.issuer.clone(),
                    detail,
                }
                .into())
            }
        }
    }

    pub(crate) async fn device_login(
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
        let response = self
            .post(
                endpoint,
                &[("client_id", &self.client_id), ("scope", scope)],
            )
            .await?;
        let status = response.status();
        if !status.is_success() {
            let detail = response.json::<OAuthError>().await.map_or_else(
                |_| format!("HTTP {status}"),
                |error| describe(&error.error, error.error_description.as_deref()),
            );
            return Err(auth_error(format!(
                "{} refused the device sign-in: {detail}",
                self.issuer
            )));
        }
        let start: DeviceStart = response.json().await?;
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

    /// Polls at the issuer's interval until the code is approved, refused or expired. `unit`
    /// is one second in production; tests shorten it.
    pub(crate) async fn poll_device(
        &self,
        start: &DeviceStart,
        unit: Duration,
    ) -> Result<Session, CliError> {
        let deadline =
            tokio::time::Instant::now() + unit.saturating_mul(saturating_u32(start.expires_in));
        let mut interval = start.interval.unwrap_or(DEFAULT_POLL_SECS).max(1);
        loop {
            let next_poll =
                tokio::time::Instant::now() + unit.saturating_mul(saturating_u32(interval));
            if next_poll >= deadline {
                return Err(device_failure("expired_token", "expired_token"));
            }
            tokio::time::sleep_until(next_poll).await;
            let grant = self
                .token_request(&[
                    ("grant_type", DEVICE_CODE_GRANT),
                    ("device_code", &start.device_code),
                    ("client_id", &self.client_id),
                ])
                .await;
            match grant {
                Grant::Issued(reply) => return self.session(reply, None, now_secs()),
                Grant::Refused { code, .. } if code == "slow_down" => {
                    interval = interval.saturating_add(SLOW_DOWN_SECS);
                }
                Grant::Refused { code, detail } if code != "authorization_pending" => {
                    return Err(device_failure(&code, &detail));
                }
                Grant::Refused { .. } | Grant::Unavailable(_) => {}
            }
        }
    }

    pub(super) async fn refresh(
        &self,
        refresh_token: &str,
        now: &Clock<'_>,
    ) -> Result<Session, CliError> {
        let grant = self
            .token_request(&[
                ("grant_type", "refresh_token"),
                ("refresh_token", refresh_token),
                ("client_id", &self.client_id),
            ])
            .await;
        match grant {
            // The lifetime starts when the reply arrives, not before the request.
            Grant::Issued(reply) => self.session(reply, Some(refresh_token), now()),
            Grant::Refused { code, .. }
                if matches!(code.as_str(), "invalid_grant" | "invalid_client") =>
            {
                Err(login_required(&self.issuer))
            }
            Grant::Refused { detail: error, .. } | Grant::Unavailable(error) => {
                Err(auth_error(format!(
                    "{} could not refresh the sign-in ({error}); try again later",
                    self.issuer
                )))
            }
        }
    }

    /// Never fails: a request that does not reach the issuer is `Unavailable`.
    async fn token_request(&self, form: &[(&str, &str)]) -> Grant {
        let response = match self.post(&self.token_endpoint, form).await {
            Ok(response) => response,
            Err(_) => return Grant::Unavailable("the issuer could not be reached".into()),
        };
        let status = response.status();
        let Ok(reply) = response.json::<TokenReply>().await else {
            return Grant::Unavailable(format!("HTTP {status}"));
        };
        match (&reply.access_token, reply.error.clone()) {
            (Some(_), None) if status.is_success() => Grant::Issued(reply),
            (_, Some(code)) if !status.is_server_error() => Grant::Refused {
                detail: describe(&code, reply.error_description.as_deref()),
                code,
            },
            _ => Grant::Unavailable(format!("HTTP {status}")),
        }
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
        let expires_in = reply.expires_in.ok_or_else(|| {
            auth_error("the issuer returned an access token without expires_in".into())
        })?;
        Ok(Session {
            issuer: self.issuer.clone(),
            client_id: self.client_id.clone(),
            access_token,
            refresh_token: reply
                .refresh_token
                .or_else(|| previous_refresh.map(str::to_owned)),
            expires_at: now.saturating_add(expires_in),
        })
    }
}

/// Accepts `https`, or `http` on a loopback host, and returns the URL's origin.
pub(crate) fn secure_origin(value: &str) -> Result<String, CliError> {
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

fn saturating_u32(value: u64) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

fn device_failure(code: &str, detail: &str) -> CliError {
    auth_error(match code {
        "expired_token" => "the sign-in code expired; run the login again".into(),
        "access_denied" => "the sign-in was declined".into(),
        _ => format!("the sign-in was not completed: {detail}"),
    })
}

/// The OAuth error code, with the issuer's description when it sent one.
fn describe(code: &str, description: Option<&str>) -> String {
    match description.filter(|text| !text.trim().is_empty()) {
        Some(text) => format!("{code} ({text})"),
        None => code.to_owned(),
    }
}
