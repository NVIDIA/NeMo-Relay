// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded diagnostic delivery, independent of execution and operational logs.
//! Counters, history, and recording permission remain owned by the Relay host.

use nemo_relay_types::diagnostics::{BuiltinDiagnosticCode, DiagnosticAvailability};
use nemo_relay_worker_proto::v1 as wire;
use prost::Message;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tonic::{Code, transport::Channel};
#[path = "diagnostic_delivery.rs"]
mod delivery;

pub use wire::{
    DiagnosticCodeDefinition, DiagnosticEvidenceRetention, DiagnosticExtensionDefinition,
    DiagnosticObservation, DiagnosticObservationOutcome, DiagnosticObservationSeverity,
    DiagnosticReceipt, DiagnosticReceiptStatus,
};
const MAX_REQUEST_BYTES: usize = 16 * 1024;
const MAX_RESPONSE_BYTES: usize = 128 * 1024;
const MIN_RESPONSE_BYTES: usize = 256;
const MAX_PENDING_SUBMISSIONS: usize = 32;
const MAX_RETRY_ATTEMPTS: u32 = 8;
const MAX_RETRY_TIMEOUT: Duration = Duration::from_secs(10);
const BOOTSTRAP_TIMEOUT: Duration = Duration::from_secs(2);
/// Space reserved in the submission frame for credentials, session identity,
/// sequence ordering, and protobuf field framing. The hydrated request is also
/// checked against the exact host-provided frame limit before it is sent.
const SUBMISSION_ENVELOPE_ALLOWANCE_BYTES: usize = 256;
const MIN_SUBMISSION_BYTES: usize = 2;

#[derive(Debug, Clone, Copy)]
struct DeliveryPolicy {
    max_request_bytes: usize,
    max_response_bytes: usize,
    max_pending_submissions: usize,
    retry_attempts: u32,
    retry_timeout: Duration,
}
impl DeliveryPolicy {
    fn from_capabilities(capabilities: &wire::DiagnosticCapabilities) -> Option<Self> {
        let max_request_bytes = usize::try_from(capabilities.max_request_bytes).ok()?;
        let max_response_bytes = usize::try_from(capabilities.max_response_bytes).ok()?;
        let max_pending_submissions = usize::try_from(capabilities.max_pending_submissions).ok()?;
        let retry_timeout = Duration::from_millis(u64::from(capabilities.retry_timeout_millis));
        if max_request_bytes < SUBMISSION_ENVELOPE_ALLOWANCE_BYTES + MIN_SUBMISSION_BYTES
            || max_request_bytes > MAX_REQUEST_BYTES
            || max_response_bytes < MIN_RESPONSE_BYTES
            || max_response_bytes > MAX_RESPONSE_BYTES
            || max_pending_submissions == 0
            || max_pending_submissions > MAX_PENDING_SUBMISSIONS
            || capabilities.retry_attempts == 0
            || capabilities.retry_attempts > MAX_RETRY_ATTEMPTS
            || retry_timeout.is_zero()
            || retry_timeout > MAX_RETRY_TIMEOUT
        {
            return None;
        }
        Some(Self {
            max_request_bytes,
            max_response_bytes,
            max_pending_submissions,
            retry_attempts: capabilities.retry_attempts,
            retry_timeout,
        })
    }
}

/// Delivery outcome, independent of plugin execution or health.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DiagnosticDeliveryError {
    /// The queue cannot accept another submission.
    #[error("diagnostic queue full")]
    Full,
    /// The payload is invalid or exceeds its bound.
    #[error("invalid diagnostic submission")]
    Invalid,
    /// Explicit host rejection that did not consume a sequence; counted as rejected.
    #[error("diagnostic submission not consumed")]
    NotConsumed,
    /// Terminal host rejection; the sequence was consumed.
    #[error("diagnostic submission rejected")]
    Rejected,
    /// Acceptance could not be recovered; this session cannot replay further.
    #[error("diagnostic acceptance uncertain; session fenced")]
    Uncertain,
    /// The facade has stopped accepting submissions.
    #[error("diagnostic delivery closed")]
    Closed,
}
/// SDK-local saturating delivery totals; never replayed as host observations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DiagnosticDeliveryStats {
    /// Queue or input admission losses.
    pub dropped: u64,
    /// Host-rejected submissions, whether or not their sequence was consumed.
    pub rejected: u64,
    /// Submissions with uncertain acceptance.
    pub uncertain: u64,
}
#[derive(Default)]
struct DeliveryState {
    stopped: AtomicBool,
    dropped: AtomicU64,
    rejected: AtomicU64,
    uncertain: AtomicU64,
}
fn increment(value: &AtomicU64) {
    let _ = value.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
        Some(v.saturating_add(1))
    });
}
enum Pending {
    Submission {
        frozen: Vec<u8>,
        receipt: Option<oneshot::Sender<Result<DiagnosticReceipt, DiagnosticDeliveryError>>>,
    },
    Flush(oneshot::Sender<Result<(), DiagnosticDeliveryError>>),
}

struct DeliveryTask(tokio::task::AbortHandle);
impl Drop for DeliveryTask {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// One negotiated facade per activation. Disabled/unsupported facades are inert.
#[derive(Clone)]
pub struct WorkerDiagnostics {
    availability: DiagnosticAvailability,
    sender: Option<mpsc::Sender<Pending>>,
    state: Arc<DeliveryState>,
    task: Option<Arc<DeliveryTask>>,
    policy: Option<DeliveryPolicy>,
}
impl WorkerDiagnostics {
    pub(crate) async fn connect(
        channel: Result<Channel, crate::WorkerSdkError>,
        activation_id: String,
        auth_token: String,
    ) -> Self {
        let mut facade = Self {
            availability: DiagnosticAvailability::Unavailable,
            sender: None,
            state: Arc::new(DeliveryState::default()),
            task: None,
            policy: None,
        };
        let Ok(channel) = channel else {
            return facade;
        };
        let mut client =
            wire::relay_diagnostics_client::RelayDiagnosticsClient::new(channel.clone())
                .max_encoding_message_size(MAX_REQUEST_BYTES)
                .max_decoding_message_size(MAX_RESPONSE_BYTES);
        let credentials = wire::DiagnosticCredentials {
            activation_id,
            auth_token,
        };
        let capabilities = match tokio::time::timeout(
            BOOTSTRAP_TIMEOUT,
            client.get_diagnostic_capabilities(credentials.clone()),
        )
        .await
        {
            Ok(Ok(response)) => response.into_inner(),
            Ok(Err(error)) if error.code() == Code::Unimplemented => {
                facade.availability = DiagnosticAvailability::Unsupported;
                return facade;
            }
            _ => return facade,
        };
        if !capabilities.enabled {
            facade.availability = DiagnosticAvailability::Disabled;
            return facade;
        }
        if capabilities.protocol_version != 1
            || capabilities.provenance_version != 1
            || capabilities.snapshot_version != 1
        {
            facade.availability = DiagnosticAvailability::Unsupported;
            return facade;
        }
        if DeliveryPolicy::from_capabilities(&capabilities).is_none() {
            facade.availability = DiagnosticAvailability::Unsupported;
            return facade;
        }
        let opened = match tokio::time::timeout(
            BOOTSTRAP_TIMEOUT,
            client.open_diagnostic_session(wire::DiagnosticOpenRequest {
                credentials: Some(credentials.clone()),
                protocol_version: 1,
                snapshot_version: 1,
                provenance_version: 1,
            }),
        )
        .await
        {
            Ok(Ok(response)) => response.into_inner(),
            _ => return facade,
        };
        let Some(session_capabilities) = opened.capabilities.as_ref() else {
            facade.availability = DiagnosticAvailability::Unsupported;
            return facade;
        };
        if !session_capabilities.enabled {
            facade.availability = DiagnosticAvailability::Disabled;
            return facade;
        }
        if session_capabilities != &capabilities {
            facade.availability = DiagnosticAvailability::Unsupported;
            return facade;
        }
        let Some(policy) = DeliveryPolicy::from_capabilities(session_capabilities) else {
            facade.availability = DiagnosticAvailability::Unsupported;
            return facade;
        };
        if opened.session_id.len() != 16 {
            return facade;
        }
        let client = wire::relay_diagnostics_client::RelayDiagnosticsClient::new(channel)
            .max_encoding_message_size(policy.max_request_bytes)
            .max_decoding_message_size(policy.max_response_bytes);
        let (sender, receiver) = mpsc::channel(policy.max_pending_submissions);
        facade.availability = DiagnosticAvailability::Available;
        facade.sender = Some(sender);
        facade.policy = Some(policy);
        let task = tokio::spawn(delivery::run(
            client,
            credentials,
            opened.session_id,
            receiver,
            facade.state.clone(),
            policy,
        ));
        facade.task = Some(Arc::new(DeliveryTask(task.abort_handle())));
        facade
    }
    /// Negotiated availability, independent of deployment health.
    pub fn availability(&self) -> DiagnosticAvailability {
        self.availability
    }
    /// Read local delivery totals without resetting them.
    pub fn delivery_stats(&self) -> DiagnosticDeliveryStats {
        DiagnosticDeliveryStats {
            dropped: self.state.dropped.load(Ordering::Relaxed),
            rejected: self.state.rejected.load(Ordering::Relaxed),
            uncertain: self.state.uncertain.load(Ordering::Relaxed),
        }
    }
    /// Register a bounded vocabulary; disabled mode returns an inert component.
    pub async fn component(
        &self,
        name: &str,
        codes: Vec<DiagnosticCodeDefinition>,
    ) -> Result<DiagnosticComponent, DiagnosticDeliveryError> {
        if self.sender.is_none() {
            return Ok(DiagnosticComponent {
                diagnostics: self.clone(),
                handle: 0,
            });
        }
        let receipt = self
            .submit(wire::diagnostic_submission::Operation::RegisterComponent(
                wire::DiagnosticComponentRegistration {
                    local_name: name.into(),
                    codes,
                },
            ))
            .await?;
        if receipt.status != DiagnosticReceiptStatus::Registered as i32 {
            return Err(DiagnosticDeliveryError::Rejected);
        }
        Ok(DiagnosticComponent {
            diagnostics: self.clone(),
            handle: receipt.component,
        })
    }
    fn freeze(
        operation: wire::diagnostic_submission::Operation,
        max_request_bytes: usize,
    ) -> Result<Vec<u8>, DiagnosticDeliveryError> {
        let request = wire::DiagnosticSubmission {
            operation: Some(operation),
            ..Default::default()
        };
        // Queue entries own encoded bytes, never caller-owned spare string/vector
        // capacity. Reserve the named session-envelope allowance for delivery.
        if request.encoded_len() > max_request_bytes - SUBMISSION_ENVELOPE_ALLOWANCE_BYTES {
            return Err(DiagnosticDeliveryError::Invalid);
        }
        Ok(request.encode_to_vec())
    }
    async fn submit(
        &self,
        operation: wire::diagnostic_submission::Operation,
    ) -> Result<DiagnosticReceipt, DiagnosticDeliveryError> {
        if self.state.stopped.load(Ordering::Acquire) {
            return Err(DiagnosticDeliveryError::Closed);
        }
        let sender = self
            .sender
            .as_ref()
            .ok_or(DiagnosticDeliveryError::Closed)?;
        let permit = sender
            .try_reserve()
            .map_err(|_| DiagnosticDeliveryError::Full)?;
        let policy = self.policy.ok_or(DiagnosticDeliveryError::Closed)?;
        let frozen = Self::freeze(operation, policy.max_request_bytes)?;
        let (receipt, receiver) = oneshot::channel();
        permit.send(Pending::Submission {
            frozen,
            receipt: Some(receipt),
        });
        receiver
            .await
            .map_err(|_| DiagnosticDeliveryError::Uncertain)?
    }
    /// Wait for previously admitted work within a deadline; no disk/exporter acknowledgment.
    pub async fn flush(&self, deadline: Duration) -> Result<(), DiagnosticDeliveryError> {
        let Some(sender) = &self.sender else {
            return Ok(());
        };
        let (done, wait) = oneshot::channel();
        tokio::time::timeout(deadline, async {
            sender
                .send(Pending::Flush(done))
                .await
                .map_err(|_| DiagnosticDeliveryError::Closed)?;
            wait.await.map_err(|_| DiagnosticDeliveryError::Uncertain)?
        })
        .await
        .map_err(|_| DiagnosticDeliveryError::Uncertain)?
    }
    /// Drain and close within the existing plugin shutdown deadline.
    pub async fn close(&self, deadline: Duration) -> Result<(), DiagnosticDeliveryError> {
        if self.sender.is_none() {
            return Ok(());
        }
        let result = tokio::time::timeout(deadline, async {
            self.flush(deadline).await?;
            self.submit(wire::diagnostic_submission::Operation::CloseSession(
                wire::DiagnosticSessionClose {},
            ))
            .await
            .map(|_| ())
        })
        .await
        .map_err(|_| DiagnosticDeliveryError::Uncertain)
        .and_then(|v| v);
        self.state.stopped.store(true, Ordering::Release);
        if let Some(task) = &self.task {
            task.0.abort();
        }
        result
    }
}
/// Component-scoped authoring handle; code IDs are one-based registration positions.
#[derive(Clone)]
pub struct DiagnosticComponent {
    diagnostics: WorkerDiagnostics,
    handle: u32,
}
impl DiagnosticComponent {
    /// Enqueue a distinct internal sub-operation. Callback completion remains host-owned.
    /// Disabled or full facades skip the factory; encoding freezes caller-owned data.
    pub fn try_record_lazy(
        &self,
        make: impl FnOnce() -> DiagnosticObservation,
    ) -> Result<(), DiagnosticDeliveryError> {
        if self.handle == 0 {
            return Ok(());
        }
        if self.diagnostics.state.stopped.load(Ordering::Acquire) {
            return Err(DiagnosticDeliveryError::Closed);
        }
        let sender = self
            .diagnostics
            .sender
            .as_ref()
            .ok_or(DiagnosticDeliveryError::Closed)?;
        let permit = sender.try_reserve().map_err(|_| {
            increment(&self.diagnostics.state.dropped);
            DiagnosticDeliveryError::Full
        })?;
        let policy = self
            .diagnostics
            .policy
            .ok_or(DiagnosticDeliveryError::Closed)?;
        let mut observation = make();
        if observation.count == 0 {
            return Err(DiagnosticDeliveryError::Invalid);
        }
        observation.component = self.handle;
        let frozen = WorkerDiagnostics::freeze(
            wire::diagnostic_submission::Operation::Record(observation),
            policy.max_request_bytes,
        )
        .inspect_err(|_| increment(&self.diagnostics.state.dropped))?;
        permit.send(Pending::Submission {
            frozen,
            receipt: None,
        });
        Ok(())
    }
    /// Close registration through the same ordered receipt boundary.
    pub async fn close(&self) -> Result<(), DiagnosticDeliveryError> {
        if self.handle == 0 {
            return Ok(());
        }
        self.diagnostics
            .submit(wire::diagnostic_submission::Operation::CloseComponent(
                wire::DiagnosticComponentClose {
                    component: self.handle,
                },
            ))
            .await
            .map(|_| ())
    }
}
/// Select a typed catalogue code without copying its spelling into plugin code.
pub fn builtin_code(code: BuiltinDiagnosticCode) -> DiagnosticCodeDefinition {
    DiagnosticCodeDefinition {
        definition: Some(wire::diagnostic_code_definition::Definition::BuiltinName(
            code.canonical_name().into(),
        )),
    }
}

/// Define a bounded internal operation in the host-assigned session namespace.
pub fn extension_code(definition: DiagnosticExtensionDefinition) -> DiagnosticCodeDefinition {
    DiagnosticCodeDefinition {
        definition: Some(wire::diagnostic_code_definition::Definition::Extension(
            definition,
        )),
    }
}

#[cfg(test)]
#[path = "../tests/unit/diagnostics_tests.rs"]
mod tests;
