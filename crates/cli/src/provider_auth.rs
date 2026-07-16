// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-credential provenance for CLI gateway requests.
//!
//! Transparent wrappers authenticate to their invocation-owned loopback gateway with a random
//! credential. The gateway consumes that credential before request intercepts run, preserving a
//! clear boundary between wrapper authentication and credentials intended for an upstream model.

use std::sync::Arc;

use axum::http::{HeaderMap, HeaderValue, header};
use ring::rand::{SecureRandom, SystemRandom};
use subtle::ConstantTimeEq;

use crate::error::CliError;

pub(crate) const TRANSPARENT_PROXY_CREDENTIAL_ENV: &str = "NEMO_RELAY_PROXY_CREDENTIAL";
pub(crate) const TRANSPARENT_PROXY_CREDENTIAL_HEADER: &str = "x-nemo-relay-proxy-token";

const TOKEN_BYTES: usize = 32;
const PROVIDER_API_KEY_HEADERS: [&str; 3] = ["x-api-key", "api-key", "anthropic-api-key"];

#[derive(Clone)]
pub(crate) struct TransparentProxyCredential(Arc<str>);

impl TransparentProxyCredential {
    pub(crate) fn generate() -> Result<Self, CliError> {
        let mut bytes = [0_u8; TOKEN_BYTES];
        SystemRandom::new().fill(&mut bytes).map_err(|_| {
            CliError::Launch("failed to generate transparent proxy credential".into())
        })?;
        let encoded = bytes
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        Ok(Self(format!("nrp_{encoded}").into()))
    }

    pub(crate) fn expose(&self) -> &str {
        &self.0
    }

    /// Verify and consume this invocation's proxy credential without disturbing an independent
    /// provider credential carried by a dedicated header.
    pub(crate) fn consume(
        &self,
        headers: &mut HeaderMap,
    ) -> Result<SourceCredentialDisposition, CliError> {
        let mut authenticated = false;
        if let Some(value) = headers.get(TRANSPARENT_PROXY_CREDENTIAL_HEADER) {
            if !self.matches_raw(value) {
                return Err(CliError::Unauthorized(
                    "transparent proxy token did not match this Relay invocation".into(),
                ));
            }
            authenticated = true;
            headers.remove(TRANSPARENT_PROXY_CREDENTIAL_HEADER);
        }

        if headers
            .get(header::AUTHORIZATION)
            .is_some_and(|value| self.matches_bearer(value))
        {
            authenticated = true;
            headers.remove(header::AUTHORIZATION);
        }
        for name in PROVIDER_API_KEY_HEADERS {
            if headers
                .get(name)
                .is_some_and(|value| self.matches_raw(value))
            {
                authenticated = true;
                headers.remove(name);
            }
        }

        if !authenticated {
            return Err(CliError::Unauthorized(
                "request did not present this transparent Relay invocation's proxy token".into(),
            ));
        }
        Ok(SourceCredentialDisposition::RelayProxyCredential {
            provider_credential_present: has_provider_credential(headers),
        })
    }

    fn matches_raw(&self, value: &HeaderValue) -> bool {
        value
            .to_str()
            .ok()
            .is_some_and(|value| constant_time_eq(value.as_bytes(), self.0.as_bytes()))
    }

    fn matches_bearer(&self, value: &HeaderValue) -> bool {
        value
            .to_str()
            .ok()
            .and_then(|value| value.strip_prefix("Bearer "))
            .is_some_and(|value| constant_time_eq(value.as_bytes(), self.0.as_bytes()))
    }

    #[cfg(test)]
    pub(crate) fn from_static(value: &'static str) -> Self {
        Self(value.into())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SourceCredentialDisposition {
    RelayProxyCredential { provider_credential_present: bool },
    ProviderCredential,
    Absent,
}

impl SourceCredentialDisposition {
    pub(crate) fn from_provider_headers(headers: &HeaderMap) -> Self {
        if has_provider_credential(headers) {
            Self::ProviderCredential
        } else {
            Self::Absent
        }
    }

    pub(crate) const fn provider_credential_present(self) -> bool {
        match self {
            Self::RelayProxyCredential {
                provider_credential_present,
            } => provider_credential_present,
            Self::ProviderCredential => true,
            Self::Absent => false,
        }
    }

    pub(crate) fn after_source_normalization(self, headers: &HeaderMap) -> Self {
        let provider_credential_present = has_provider_credential(headers);
        match self {
            Self::RelayProxyCredential { .. } => Self::RelayProxyCredential {
                provider_credential_present,
            },
            Self::ProviderCredential | Self::Absent => Self::from_provider_headers(headers),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ProviderRequestAuthorization {
    pub(crate) source_credential: SourceCredentialDisposition,
    pub(crate) allow_environment_provider_auth: bool,
}

pub(crate) fn has_provider_credential(headers: &HeaderMap) -> bool {
    headers.contains_key(header::AUTHORIZATION)
        || PROVIDER_API_KEY_HEADERS
            .iter()
            .any(|name| headers.contains_key(*name))
}

pub(crate) fn remove_provider_credentials(headers: &mut HeaderMap) {
    headers.remove(header::AUTHORIZATION);
    for name in PROVIDER_API_KEY_HEADERS {
        headers.remove(name);
    }
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    left.ct_eq(right).into()
}
