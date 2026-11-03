// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Authenticated plugin submissions. One receipt per session makes acceptance
//! cancellation-safe without retaining a second event history.

use std::num::NonZeroU64;
use std::sync::{Arc, Mutex, OnceLock};

use chrono::{DateTime, Utc};
use nemo_relay_worker_proto::nvidia_worker_v1 as wire;
use prost::Message;
use sha2::{Digest, Sha256};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tonic::{Request, Response, Status};
use uuid::Uuid;

use nemo_relay_diagnostics::{
    BuiltinDiagnosticCode, Classification, ComponentOwner, ComponentRecorder,
    ComponentRegistration, DiagnosticCorrelation, DiagnosticDetailRef, DiagnosticOutcome,
    DiagnosticSeverity, Diagnostics, EvidenceDisposition, ExtensionHandle, ExtensionRegistration,
    FailureClass, Observation, PluginExecution, PluginObservationContext, PluginPhase,
    RecordOutcome, RemoteDiagnosticOrigin,
};

pub(super) const REQUEST_BYTES: usize = 16 * 1024;
pub(super) const RESPONSE_BYTES: usize = 128 * 1024;
const COMPONENTS: usize = 8;
const EXTENSIONS: usize = 16;
const CODES: usize = 32;
const METADATA_BYTES: usize = 16 * 1024;
// SDK queue policy and host-side session admission are independent budgets.
const MAX_PENDING_SUBMISSIONS: usize = 32;
const MAX_CONCURRENT_SESSIONS: usize = 32;
const SUBMISSION_ATTEMPTS: u32 = 3;
const SUBMISSION_TIMEOUT_MILLIS: u32 = 2000;
static SESSIONS: OnceLock<Arc<Semaphore>> = OnceLock::new();

pub(super) struct WorkerDiagnostics {
    pub(super) context: Diagnostics,
    activation: Uuid,
    capabilities: wire::DiagnosticCapabilities,
    state: Mutex<SessionState>,
}

struct SessionState {
    revoked: bool,
    session: Option<Session>,
}

struct Session {
    id: Uuid,
    _permit: OwnedSemaphorePermit,
    closed: bool,
    components: Vec<Option<RegisteredComponent>>,
    extensions: usize,
    metadata: usize,
    last: Option<Receipt>,
}

struct Receipt {
    digest: [u8; 32],
    value: wire::DiagnosticReceipt,
}

struct RegisteredComponent {
    owner: ComponentOwner,
    recorder: ComponentRecorder,
    codes: Vec<Code>,
}

#[derive(Clone, Copy)]
enum Code {
    Builtin(BuiltinDiagnosticCode),
    Extension(ExtensionHandle, DiagnosticOutcome),
}

impl WorkerDiagnostics {
    pub(super) fn new(context: Diagnostics, activation: Uuid) -> Self {
        let capabilities = Self::capabilities_for(&context);
        Self {
            context,
            activation,
            capabilities,
            state: Mutex::new(SessionState {
                revoked: false,
                session: None,
            }),
        }
    }

    pub(super) fn revoke(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.revoked = true;
        if let Some(session) = state.session.take() {
            for component in session.components.into_iter().flatten() {
                let _ = component.owner.close();
            }
        }
    }

    fn capabilities_for(context: &Diagnostics) -> wire::DiagnosticCapabilities {
        wire::DiagnosticCapabilities {
            protocol: Some(wire::ProtocolIdentity {
                name: wire::WORKER_PROTOCOL.into(),
                version: wire::CONTRACT_VERSION.into(),
            }),
            snapshot_contract_version: "1.0.0".into(),
            catalogue_version: nemo_relay_diagnostics::DIAGNOSTIC_CATALOGUE_VERSION,
            provenance_contract_version: "1.0.0".into(),
            enabled: context.reader().availability()
                == nemo_relay_diagnostics::DiagnosticAvailability::Available,
            can_read_context: false,
            can_read_discovery: false,
            can_inspect_memory: false,
            max_components: COMPONENTS as u32,
            max_extensions: EXTENSIONS as u32,
            max_metadata_bytes: METADATA_BYTES as u32,
            max_request_bytes: REQUEST_BYTES as u32,
            max_detail_bytes: context.limits().max_detail_bytes as u32,
            max_correlation_bytes: context.limits().max_correlation_bytes as u32,
            max_response_bytes: RESPONSE_BYTES as u32,
            max_pending_submissions: MAX_PENDING_SUBMISSIONS as u32,
            retry_attempts: SUBMISSION_ATTEMPTS,
            retry_timeout_duration: Some(prost_types::Duration {
                seconds: (SUBMISSION_TIMEOUT_MILLIS / 1000) as i64,
                nanos: ((SUBMISSION_TIMEOUT_MILLIS % 1000) * 1_000_000) as i32,
            }),
        }
    }

    fn capabilities(&self) -> wire::DiagnosticCapabilities {
        self.capabilities.clone()
    }

    fn open(
        &self,
        request: &wire::OpenDiagnosticSessionRequest,
    ) -> Result<wire::OpenDiagnosticSessionResponse, Status> {
        if request
            .protocol
            .as_ref()
            .is_none_or(|p| p.name != wire::WORKER_PROTOCOL || p.version != wire::CONTRACT_VERSION)
            || request.provenance_contract_version != "1.0.0"
            || request.snapshot_contract_version != "1.0.0"
        {
            return Err(Status::failed_precondition(
                "unsupported diagnostic contract",
            ));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| Status::internal("diagnostic session fenced"))?;
        if state.revoked {
            return Err(Status::failed_precondition("diagnostic activation expired"));
        }
        let capabilities = self.capabilities();
        if !capabilities.enabled {
            return Err(Status::failed_precondition("diagnostic delivery disabled"));
        }
        if state.session.is_none() {
            let permit = SESSIONS
                .get_or_init(|| Arc::new(Semaphore::new(MAX_CONCURRENT_SESSIONS)))
                .clone()
                .try_acquire_owned()
                .map_err(|_| Status::resource_exhausted("diagnostic session capacity"))?;
            state.session = Some(Session {
                id: Uuid::now_v7(),
                _permit: permit,
                closed: false,
                components: Vec::with_capacity(COMPONENTS),
                extensions: 0,
                metadata: 0,
                last: None,
            });
        }
        let session = state.session.as_ref().expect("admitted session");
        if session.closed {
            return Err(Status::failed_precondition("diagnostic session closed"));
        }
        Ok(wire::OpenDiagnosticSessionResponse {
            session_id: session.id.as_bytes().to_vec(),
            capabilities: Some(capabilities),
        })
    }

    fn submit(
        &self,
        request: wire::SubmitDiagnosticRequest,
    ) -> Result<wire::DiagnosticReceipt, Status> {
        // Frame admission happens before prost decoding. Validate structure before
        // touching ordering; these errors explicitly consume no sequence.
        if request.sequence == 0 || request.operation.is_none() || request.session_id.len() != 16 {
            return Err(Status::invalid_argument(
                "invalid diagnostic submission; not consumed",
            ));
        }
        if request.encoded_len() > self.capabilities.max_request_bytes as usize {
            return Err(Status::invalid_argument(
                "diagnostic frame too large; not consumed",
            ));
        }
        if let Some(wire::submit_diagnostic_request::Operation::Record(observation)) =
            &request.operation
            && (observation.count == 0
                || observation.observe_time.as_ref().is_some_and(|time| {
                    nemo_relay_worker_proto::conversions::decode::<DateTime<Utc>, _>(*time).is_err()
                }))
        {
            return Err(Status::invalid_argument(
                "invalid diagnostic observation; not consumed",
            ));
        }
        let digest: [u8; 32] = Sha256::digest(request.encode_to_vec()).into();
        let mut state = self
            .state
            .lock()
            .map_err(|_| Status::internal("diagnostic session fenced"))?;
        if state.revoked {
            return Err(Status::failed_precondition("diagnostic activation expired"));
        }
        let session = state
            .session
            .as_mut()
            .filter(|s| s.id.as_bytes() == request.session_id.as_slice())
            .ok_or_else(|| Status::failed_precondition("diagnostic session expired"))?;
        let previous = session
            .last
            .as_ref()
            .map_or(0, |r| r.value.submission_sequence);
        if request.sequence == previous {
            let receipt = session.last.as_ref().expect("positive sequence");
            return if receipt.digest == digest {
                Ok(receipt.value.clone())
            } else {
                Err(Status::already_exists(
                    "conflicting diagnostic retry; not consumed",
                ))
            };
        }
        if request.sequence < previous {
            return Err(Status::out_of_range("diagnostic replay window expired"));
        }
        if previous.checked_add(1) != Some(request.sequence) {
            return Err(Status::out_of_range(
                "diagnostic sequence gap; not consumed",
            ));
        }
        if session.closed {
            return Err(Status::failed_precondition("diagnostic session closed"));
        }
        let mut receipt = wire::DiagnosticReceipt {
            submission_sequence: request.sequence,
            ..Default::default()
        };
        // No await, projection, network, or fallible serialization can intervene
        // between core acceptance and publishing the bounded replay receipt.
        let result = self.apply(
            session,
            request.sequence,
            request.operation.expect("validated operation"),
            &mut receipt,
        );
        if let Err(reason) = result {
            receipt.status = wire::DiagnosticReceiptStatus::Rejected as i32;
            receipt.reason = reason.into();
        }
        session.last = Some(Receipt {
            digest,
            value: receipt.clone(),
        });
        Ok(receipt)
    }

    fn apply(
        &self,
        session: &mut Session,
        sequence: u64,
        operation: wire::submit_diagnostic_request::Operation,
        receipt: &mut wire::DiagnosticReceipt,
    ) -> Result<(), &'static str> {
        use wire::submit_diagnostic_request::Operation;
        if self.context.reader().availability()
            == nemo_relay_diagnostics::DiagnosticAvailability::Disabled
        {
            receipt.status = wire::DiagnosticReceiptStatus::Disabled as i32;
            return Ok(());
        }
        match operation {
            Operation::RegisterComponent(registration) => {
                self.register(session, registration, receipt)
            }
            Operation::Record(observation) => self.record(session, sequence, observation, receipt),
            Operation::CloseComponent(close) => {
                let slot = close
                    .component
                    .checked_sub(1)
                    .and_then(|n| session.components.get_mut(n as usize))
                    .ok_or("unknown diagnostic component")?;
                if let Some(component) = slot.take() {
                    let _ = component.owner.close();
                }
                receipt.status = wire::DiagnosticReceiptStatus::Closed as i32;
                Ok(())
            }
            Operation::CloseSession(_) => {
                for component in &mut session.components {
                    if let Some(component) = component.take() {
                        let _ = component.owner.close();
                    }
                }
                session.closed = true;
                receipt.status = wire::DiagnosticReceiptStatus::Closed as i32;
                Ok(())
            }
        }
    }

    fn register(
        &self,
        session: &mut Session,
        registration: wire::DiagnosticComponentRegistration,
        receipt: &mut wire::DiagnosticReceipt,
    ) -> Result<(), &'static str> {
        if session.components.len() >= COMPONENTS
            || registration.codes.is_empty()
            || registration.codes.len() > CODES
        {
            return Err("diagnostic component or code capacity");
        }
        let key = registration.local_name;
        if key.is_empty() || key.len() > self.context.limits().max_identity_bytes {
            return Err("invalid diagnostic component name");
        }
        let mut builtins = Vec::with_capacity(CODES);
        let mut extensions = Vec::with_capacity(EXTENSIONS);
        let mut metadata = key.len();
        for code in &registration.codes {
            match code.definition.as_ref().ok_or("missing diagnostic code")? {
                wire::diagnostic_code_definition::Definition::BuiltinName(name) => {
                    let code = BuiltinDiagnosticCode::from_canonical_name(name)
                        .ok_or("unknown diagnostic code")?;
                    // Lifecycle/callback outcomes belong to the host. Authors
                    // use common failures or extensions for internal operations.
                    if !matches!(
                        code,
                        BuiltinDiagnosticCode::Failure(
                            nemo_relay_diagnostics::FailureCode::Plugin(
                                nemo_relay_diagnostics::PluginFailureCode::Common(_)
                            )
                        )
                    ) || builtins.contains(&code)
                    {
                        return Err("diagnostic code outside producer vocabulary");
                    }
                    builtins.push(code);
                }
                wire::diagnostic_code_definition::Definition::Extension(extension) => {
                    if extension.name.is_empty()
                        || extension.name.len() > 32
                        || extension.description.len() > 256
                        || extension.allowed_failure_classes.len() > 16
                        || extensions.iter().any(
                            |(e, _, _): &(&wire::DiagnosticExtensionDefinition, _, _)| {
                                e.name == extension.name
                            },
                        )
                    {
                        return Err("invalid diagnostic extension");
                    }
                    let outcome = outcome(extension.outcome)?;
                    let classes = extension
                        .allowed_failure_classes
                        .iter()
                        .map(|name| {
                            FailureClass::from_canonical_name(name)
                                .ok_or("unknown diagnostic cause")
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    if (outcome == DiagnosticOutcome::Failure) == classes.is_empty() {
                        return Err("invalid diagnostic extension causes");
                    }
                    metadata +=
                        extension.name.len() + extension.description.len() + classes.len() * 32;
                    extensions.push((extension, outcome, classes));
                }
            }
        }
        if session.extensions + extensions.len() > EXTENSIONS
            || session.metadata + metadata > METADATA_BYTES
        {
            return Err("diagnostic metadata capacity");
        }
        let mut input = ComponentRegistration::new(&key, &builtins);
        let extension_count = extensions.len();
        drop(extensions);
        input.ownership.plugin_instance_id = Some(self.activation);
        let (owner, recorder) = self
            .context
            .register_owned_component(input)
            .map_err(|_| "diagnostic component admission rejected")?;
        let mut codes = Vec::with_capacity(registration.codes.len());
        let namespace = format!("worker_{}", session.id.simple());
        for code in registration.codes {
            let admitted = match code.definition.expect("validated definition") {
                wire::diagnostic_code_definition::Definition::BuiltinName(name) => {
                    Ok(Code::Builtin(
                        BuiltinDiagnosticCode::from_canonical_name(&name)
                            .expect("validated builtin"),
                    ))
                }
                wire::diagnostic_code_definition::Definition::Extension(extension) => {
                    let classes = extension
                        .allowed_failure_classes
                        .iter()
                        .filter_map(|n| FailureClass::from_canonical_name(n))
                        .collect::<Vec<_>>();
                    let outcome = outcome(extension.outcome).expect("validated outcome");
                    recorder
                        .register_extension(ExtensionRegistration {
                            namespace: &namespace,
                            name: &extension.name,
                            description: &extension.description,
                            allowed_classes: &classes,
                            outcome,
                        })
                        .map_err(|_| "diagnostic extension admission rejected")
                        .and_then(|handle| {
                            handle
                                .map(|h| Code::Extension(h, outcome))
                                .ok_or("diagnostics disabled")
                        })
                }
            };
            match admitted {
                Ok(code) => codes.push(code),
                Err(reason) => {
                    owner.discard_unpublished();
                    return Err(reason);
                }
            }
        }
        session.extensions += extension_count;
        session.metadata += metadata;
        receipt.component = session.components.len() as u32 + 1;
        receipt.code_count = codes.len() as u32;
        session.components.push(Some(RegisteredComponent {
            owner,
            recorder,
            codes,
        }));
        receipt.status = wire::DiagnosticReceiptStatus::Registered as i32;
        Ok(())
    }

    fn record(
        &self,
        session: &Session,
        sequence: u64,
        value: wire::DiagnosticObservation,
        receipt: &mut wire::DiagnosticReceipt,
    ) -> Result<(), &'static str> {
        let component = value
            .component
            .checked_sub(1)
            .and_then(|n| session.components.get(n as usize))
            .and_then(Option::as_ref)
            .ok_or("unknown or closed diagnostic component")?;
        let code = value
            .code
            .checked_sub(1)
            .and_then(|n| component.codes.get(n as usize))
            .ok_or("unknown diagnostic code handle")?;
        let class = value
            .failure_class
            .as_deref()
            .map(|name| FailureClass::from_canonical_name(name).ok_or("unknown diagnostic cause"))
            .transpose()?;
        let count = NonZeroU64::new(value.count).ok_or("zero diagnostic count")?;
        let outcome = outcome(value.outcome)?;
        let detail = value
            .detail
            .as_deref()
            .map_or(DiagnosticDetailRef::None, DiagnosticDetailRef::Text);
        let mut observation = match *code {
            Code::Builtin(code) => {
                let expected = match code {
                    BuiltinDiagnosticCode::Failure(_) => DiagnosticOutcome::Failure,
                    BuiltinDiagnosticCode::Success(_) => DiagnosticOutcome::Success,
                    _ => DiagnosticOutcome::Unknown,
                };
                if outcome != expected {
                    return Err("diagnostic outcome does not match code");
                }
                Observation::new(
                    Classification::new(code, class)
                        .map_err(|_| "invalid diagnostic classification")?,
                    detail,
                    count,
                )
            }
            Code::Extension(handle, expected) => {
                if outcome != expected {
                    return Err("diagnostic outcome does not match extension");
                }
                Observation::extension(handle, class, outcome, detail, count)
            }
        };
        observation.severity = severity(value.severity)?;
        observation.context = PluginObservationContext {
            execution: Some(PluginExecution::WorkerProtocol.into()),
            phase: Some(PluginPhase::Runtime.into()),
        };
        observation.correlation = DiagnosticCorrelation {
            session_id: value.session_id.as_deref().map(Into::into),
            request_id: value.request_id.as_deref().map(Into::into),
        };
        observation = observation.with_remote_origin(RemoteDiagnosticOrigin {
            source_instance_id: session.id,
            activation_instance_id: self.activation,
            submission_sequence: sequence.into(),
            observed_at: value
                .observe_time
                .map(nemo_relay_worker_proto::conversions::decode)
                .transpose()
                .map_err(|_| "invalid observation time")?,
            received_at: Utc::now(),
            accepted_elapsed_millis: 0.into(),
        });
        match component
            .recorder
            .record_without_projection(observation)
            .map_err(|_| "diagnostic recording rejected")?
        {
            RecordOutcome::Disabled => {
                receipt.status = wire::DiagnosticReceiptStatus::Disabled as i32
            }
            RecordOutcome::Recorded(recorded) => {
                receipt.status = wire::DiagnosticReceiptStatus::Recorded as i32;
                receipt.host_sequence = recorded.sequence;
                receipt.evidence = match recorded.evidence {
                    EvidenceDisposition::Retained => wire::DiagnosticEvidenceRetention::Retained,
                    EvidenceDisposition::Truncated => wire::DiagnosticEvidenceRetention::Truncated,
                    EvidenceDisposition::Dropped => wire::DiagnosticEvidenceRetention::Dropped,
                } as i32;
            }
        }
        Ok(())
    }
}

fn outcome(value: i32) -> Result<DiagnosticOutcome, &'static str> {
    match wire::DiagnosticObservationOutcome::try_from(value) {
        Ok(wire::DiagnosticObservationOutcome::Unspecified) => Ok(DiagnosticOutcome::Unknown),
        Ok(wire::DiagnosticObservationOutcome::Success) => Ok(DiagnosticOutcome::Success),
        Ok(wire::DiagnosticObservationOutcome::Failure) => Ok(DiagnosticOutcome::Failure),
        _ => Err("unknown diagnostic outcome"),
    }
}
fn severity(value: i32) -> Result<DiagnosticSeverity, &'static str> {
    match wire::DiagnosticObservationSeverity::try_from(value) {
        Ok(wire::DiagnosticObservationSeverity::Trace) => Ok(DiagnosticSeverity::Trace),
        Ok(wire::DiagnosticObservationSeverity::Debug) => Ok(DiagnosticSeverity::Debug),
        Ok(wire::DiagnosticObservationSeverity::Info) => Ok(DiagnosticSeverity::Info),
        Ok(wire::DiagnosticObservationSeverity::Warning) => Ok(DiagnosticSeverity::Warning),
        Ok(wire::DiagnosticObservationSeverity::Error) => Ok(DiagnosticSeverity::Error),
        _ => Err("unknown diagnostic severity"),
    }
}

pub(super) struct Service {
    pub(super) state: Arc<super::WorkerHostRuntimeState>,
}
impl Service {
    fn authorize(
        &self,
        credentials: Option<&wire::Credentials>,
    ) -> Result<&WorkerDiagnostics, Status> {
        let credentials = credentials
            .ok_or_else(|| Status::unauthenticated("diagnostic credentials required"))?;
        self.state
            .authorize(&credentials.activation_id, &credentials.auth_token)?;
        self.state
            .diagnostics
            .as_deref()
            .ok_or_else(|| Status::unimplemented("diagnostics unavailable"))
    }
}

#[tonic::async_trait]
impl wire::relay_diagnostics_service_server::RelayDiagnosticsService for Service {
    async fn get_diagnostic_capabilities(
        &self,
        request: Request<wire::GetDiagnosticCapabilitiesRequest>,
    ) -> Result<Response<wire::GetDiagnosticCapabilitiesResponse>, Status> {
        let service = self.authorize(request.get_ref().credentials.as_ref())?;
        if service
            .state
            .lock()
            .map_err(|_| Status::internal("diagnostic session fenced"))?
            .revoked
        {
            return Err(Status::failed_precondition("diagnostic activation expired"));
        }
        Ok(Response::new(wire::GetDiagnosticCapabilitiesResponse {
            capabilities: Some(service.capabilities()),
        }))
    }
    async fn open_diagnostic_session(
        &self,
        request: Request<wire::OpenDiagnosticSessionRequest>,
    ) -> Result<Response<wire::OpenDiagnosticSessionResponse>, Status> {
        let request = request.into_inner();
        self.authorize(request.credentials.as_ref())?
            .open(&request)
            .map(Response::new)
    }
    async fn submit_diagnostic(
        &self,
        request: Request<wire::SubmitDiagnosticRequest>,
    ) -> Result<Response<wire::SubmitDiagnosticResponse>, Status> {
        let request = request.into_inner();
        let service = self.authorize(request.credentials.as_ref())?;
        nemo_relay_worker_proto::validation::validate(&request)
            .map_err(Status::invalid_argument)?;
        service.submit(request).map(|receipt| {
            Response::new(wire::SubmitDiagnosticResponse {
                receipt: Some(receipt),
            })
        })
    }
    async fn read_diagnostic_page(
        &self,
        request: Request<wire::ReadDiagnosticPageRequest>,
    ) -> Result<Response<wire::ReadDiagnosticPageResponse>, Status> {
        self.authorize(request.get_ref().credentials.as_ref())?;
        Err(Status::permission_denied(
            "worker snapshot inspection is not granted",
        ))
    }
    async fn get_diagnostic_discovery(
        &self,
        request: Request<wire::GetDiagnosticDiscoveryRequest>,
    ) -> Result<Response<wire::GetDiagnosticDiscoveryResponse>, Status> {
        self.authorize(request.get_ref().credentials.as_ref())?;
        Err(Status::permission_denied(
            "worker discovery inspection is not granted",
        ))
    }
    async fn inspect_diagnostic_memory(
        &self,
        request: Request<wire::InspectDiagnosticMemoryRequest>,
    ) -> Result<Response<wire::InspectDiagnosticMemoryResponse>, Status> {
        self.authorize(request.get_ref().credentials.as_ref())?;
        Err(Status::permission_denied(
            "worker memory inspection is not granted",
        ))
    }
}

#[cfg(test)]
#[path = "../../tests/unit/worker_diagnostics_tests.rs"]
mod tests;
