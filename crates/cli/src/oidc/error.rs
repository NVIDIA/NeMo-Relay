// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Why a browser sign-in did not complete. No variant holds a token.

use super::browser::APPROVAL_TIMEOUT;

#[derive(Debug, thiserror::Error)]
pub(crate) enum SignInError {
    #[error("the system random number generator failed")]
    NoRandomness,
    #[error("cannot listen on 127.0.0.1:{port} for the sign-in redirect: {source}")]
    CallbackUnavailable { port: u16, source: std::io::Error },
    #[error("{issuer} does not offer browser sign-in (no authorization_endpoint); use --device")]
    BrowserSignInNotOffered { issuer: String },
    #[error("{endpoint} is not a valid URL")]
    InvalidEndpoint { endpoint: String },
    #[error(
        "timed out after {} minutes waiting for the sign-in; run the login again",
        APPROVAL_TIMEOUT.as_secs() / 60
    )]
    TimedOut,
    #[error("the sign-in redirect names another issuer")]
    ForeignIssuer,
    #[error("the sign-in redirect carries no code")]
    MissingCode,
    #[error("the sign-in was declined")]
    Declined,
    #[error("the sign-in was not completed: {detail}")]
    NotCompleted { detail: String },
    #[error("{issuer} did not issue a token for the approved sign-in: {detail}")]
    TokenNotIssued { issuer: String, detail: String },
}

impl SignInError {
    /// An OAuth error code from the redirect (RFC 6749 section 4.1.2.1).
    pub(crate) fn from_redirect_error(code: &str, description: Option<&str>) -> Self {
        match (code, description.filter(|text| !text.trim().is_empty())) {
            ("access_denied", _) => Self::Declined,
            (code, Some(text)) => Self::NotCompleted {
                detail: format!("{code} ({text})"),
            },
            (code, None) => Self::NotCompleted {
                detail: code.to_owned(),
            },
        }
    }
}
