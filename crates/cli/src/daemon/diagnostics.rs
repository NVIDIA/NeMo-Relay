// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Daemon-owned observations and negotiated whole-source read authority.
use super::common::{control::now_unix_ms, protocol::SensitiveString};
use crate::error::CliError;
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use nemo_relay::diagnostics::*;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};
pub(crate) const CAPABILITY: &str = "daemon_diagnostics_v1";
use std::num::NonZeroU64;

/// Independent collection admission ceiling, including bounded session metadata.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct CollectorConfig {
    pub collect_workers: bool,
    pub max_owned_bytes: usize,
}
impl Default for CollectorConfig {
    fn default() -> Self {
        Self {
            collect_workers: false,
            max_owned_bytes: 16 * 1024 * 1024,
        }
    }
}
impl CollectorConfig {
    pub(crate) fn validate(&self) -> Result<(), CliError> {
        if self.collect_workers && self.max_owned_bytes < super::broker::collection::owned_bytes() {
            return Err(CliError::Config(
                "daemon diagnostics collector exceeds max_owned_bytes".into(),
            ));
        }
        Ok(())
    }
}
pub(crate) struct DaemonDiagnostics {
    recorders: [Option<ComponentRecorder>; 3],
    keys: [&'static str; 3],
}
impl DaemonDiagnostics {
    pub(crate) fn new(context: &Diagnostics, role: &str) -> Option<Self> {
        let keys = match role {
            "daemon_broker" => [
                "daemon.broker.control",
                "daemon.broker.forwarding",
                "daemon.broker.collection",
            ],
            "daemon_worker" => [
                "daemon.worker.runtime",
                "daemon.worker.forwarding",
                "daemon.worker.collection",
            ],
            _ => return None,
        };
        let recorders = std::array::from_fn(|index| {
            if role == "daemon_worker" && index == 2 {
                return None;
            }
            let codes = BuiltinDiagnosticCode::all()
                .iter()
                .copied()
                .filter(|c| match c {
                    BuiltinDiagnosticCode::Failure(FailureCode::Daemon(code)) => {
                        Self::index(*code) == index
                            && (if role == "daemon_worker" {
                                matches!(
                                    code,
                                    DaemonFailureCode::Forward
                                        | DaemonFailureCode::ControlLost
                                        | DaemonFailureCode::WorkerInitialize
                                        | DaemonFailureCode::WorkerListener
                                        | DaemonFailureCode::WorkerTeardown
                                )
                            } else {
                                !matches!(
                                    code,
                                    DaemonFailureCode::WorkerInitialize
                                        | DaemonFailureCode::WorkerListener
                                        | DaemonFailureCode::WorkerTeardown
                                )
                            })
                    }
                    _ => false,
                })
                .collect::<Vec<_>>();
            context
                .register_component(ComponentRegistration::new(keys[index], &codes))
                .ok()
        });
        Some(Self { recorders, keys })
    }
    fn index(code: DaemonFailureCode) -> usize {
        match code {
            DaemonFailureCode::Forward => 1,
            DaemonFailureCode::Collection => 2,
            _ => 0,
        }
    }
    pub(crate) fn coverage(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.recorders
            .iter()
            .enumerate()
            .filter_map(|(i, r)| r.as_ref().and_then(|r| r.identity()).map(|_| self.keys[i]))
    }

    pub(crate) fn record(&self, code: DaemonFailureCode, cause: FailureClass, detail: &str) {
        if let Some(recorder) = &self.recorders[Self::index(code)] {
            let _ = recorder.record_without_projection_lazy(|| {
                Observation::new(
                    Classification::failure(FailureCode::Daemon(code), cause)
                        .expect("daemon cause"),
                    detail,
                    NonZeroU64::MIN,
                )
            });
        }
    }
}
pub(crate) fn record(code: DaemonFailureCode, cause: FailureClass, detail: &str) {
    if let Some(owner) = crate::runtime_diagnostics::runtime().and_then(|r| r.daemon.as_ref()) {
        owner.record(code, cause, detail);
    }
}
/// Causes come only from structured variants; flattened errors stay Unknown.
pub(crate) struct DaemonErrorClassifier(pub DaemonFailureCode);
impl DaemonErrorClassifier {
    pub(crate) fn cause(error: &CliError) -> FailureClass {
        match error {
            CliError::Io(_) => FailureClass::Io,
            CliError::Http(_) => FailureClass::InvalidInput,
            CliError::ProviderFailure(failure) => match failure.class {
                nemo_relay::error::UpstreamFailureClass::Timeout => FailureClass::Timeout,
                nemo_relay::error::UpstreamFailureClass::Connection => FailureClass::Unavailable,
                _ => FailureClass::Unknown,
            },
            CliError::InvalidPayload(_) => FailureClass::InvalidInput,
            CliError::Unauthorized(_) | CliError::RouteCredentialRejected(_) => {
                FailureClass::Authentication
            }
            CliError::Upstream(e) if e.is_timeout() => FailureClass::Timeout,
            CliError::Upstream(e) if e.is_connect() => FailureClass::Unavailable,
            _ => FailureClass::Unknown,
        }
    }
}
impl DiagnosticClassifier<CliError> for DaemonErrorClassifier {
    fn classify(&self, error: &CliError) -> Classification {
        Classification::failure(FailureCode::Daemon(self.0), Self::cause(error))
            .expect("daemon cause")
    }
}
pub(crate) fn result<T>(code: DaemonFailureCode, result: &Result<T, CliError>) {
    if let Err(error) = result
        && let Some(recorder) = crate::runtime_diagnostics::runtime()
            .and_then(|r| r.daemon.as_ref())
            .and_then(|r| r.recorders[DaemonDiagnostics::index(code)].as_ref())
    {
        let _ = recorder.record_without_projection_lazy(|| {
            Observation::new(
                DaemonErrorClassifier(code).classify(error),
                DiagnosticDetailRef::Display(error),
                NonZeroU64::MIN,
            )
        });
    }
}

#[cfg(test)]
#[path = "../../tests/coverage/daemon/diagnostics_tests.rs"]
mod tests;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct DiagnosticOffer {
    pub source: uuid::Uuid,
    pub schema_version: u32,
    pub enabled: bool,
    pub query_available: bool,
    pub whole_context_read: bool,
    pub max_page_bytes: usize,
}
pub(crate) fn offer() -> Option<DiagnosticOffer> {
    let runtime = crate::runtime_diagnostics::runtime()?;
    let reader = runtime.context.reader();
    Some(DiagnosticOffer {
        source: runtime.context.source().instance_id,
        schema_version: DIAGNOSTIC_SCHEMA_VERSION,
        enabled: reader.availability() == DiagnosticAvailability::Available,
        query_available: reader.limits().max_queries != 0,
        whole_context_read: runtime.http_read,
        max_page_bytes: reader
            .limits()
            .max_snapshot_bytes
            .min(super::broker::collection::PAGE_BYTES),
    })
}
/// Independent session read key. Expiring tokens bind source, generation and policy.
/// Sent only over authenticated control; recovery replaces the authority.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct DiagnosticReadGrant {
    pub source: uuid::Uuid,
    pub generation: String,
    pub policy_revision: u32,
    pub max_page_bytes: usize,
    key: SensitiveString,
}
impl DiagnosticReadGrant {
    pub(crate) fn issue(
        source: uuid::Uuid,
        generation: &str,
        max_page_bytes: usize,
    ) -> Option<Self> {
        if generation.is_empty()
            || generation.len() > super::broker::collection::MAX_GENERATION_BYTES
        {
            return None;
        }
        Some(Self {
            source,
            generation: generation.to_owned(),
            policy_revision: 1,
            max_page_bytes,
            key: SensitiveString::new(
                super::common::control::random_secret(
                    super::broker::collection::DIAGNOSTIC_KEY_BYTES,
                )
                .ok()?,
            )
            .ok()?,
        })
    }
    pub(crate) fn fence(&self) -> super::common::identity::TokenDigest {
        super::common::identity::TokenDigest::from_token(self.key.expose().as_bytes())
    }
    pub(crate) fn matches(&self, offer: &DiagnosticOffer, generation: &str) -> bool {
        self.source == offer.source
            && self.generation == generation
            && self.policy_revision == 1
            && self.key.expose().len() == super::broker::collection::ENCODED_DIAGNOSTIC_KEY_BYTES
            && self.max_page_bytes > 0
            && self.max_page_bytes <= offer.max_page_bytes
            && offer.whole_context_read
            && offer.enabled
            && offer.query_available
    }
    fn message(&self, expires: u64) -> Vec<u8> {
        let mut message = Vec::with_capacity(96 + self.generation.len());
        message.extend_from_slice(b"nemo-relay/daemon-diagnostics/whole-context/v1");
        message.extend_from_slice(self.source.as_bytes());
        message.extend_from_slice(&(self.generation.len() as u32).to_be_bytes());
        message.extend_from_slice(self.generation.as_bytes());
        message.extend_from_slice(&self.policy_revision.to_be_bytes());
        message.extend_from_slice(&(self.max_page_bytes as u64).to_be_bytes());
        message.extend_from_slice(&expires.to_be_bytes());
        message
    }
    pub(crate) fn token(&self, now: u64) -> String {
        let expires = now.saturating_add(60_000);
        let key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, self.key.expose().as_bytes());
        format!(
            "{expires}.{}",
            URL_SAFE_NO_PAD.encode(ring::hmac::sign(&key, &self.message(expires)))
        )
    }
    fn verify(&self, token: &str, now: u64) -> bool {
        let Some((expires, tag)) = token.split_once('.') else {
            return false;
        };
        let Ok(expires) = expires.parse::<u64>() else {
            return false;
        };
        if self.policy_revision != 1 || expires <= now || expires > now.saturating_add(120_000) {
            return false;
        }
        let mut decoded = [0u8; 32];
        if URL_SAFE_NO_PAD.decode_slice(tag, &mut decoded) != Ok(32) {
            return false;
        }
        let key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, self.key.expose().as_bytes());
        ring::hmac::verify(&key, &self.message(expires), &decoded).is_ok()
    }
}
/// Swappable authority, independent of request admission. Control loss revokes it.
#[derive(Clone, Default)]
pub(crate) struct WorkerReadAuthority(Arc<RwLock<Option<DiagnosticReadGrant>>>);
impl WorkerReadAuthority {
    pub(crate) fn replace(&self, grant: Option<DiagnosticReadGrant>) {
        *self.0.write().unwrap_or_else(|e| e.into_inner()) = grant;
    }
    pub(crate) fn verify(&self, token: &str) -> bool {
        self.0
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(|grant| grant.verify(token, now_unix_ms()))
    }
}
