// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use tonic::{Request, Response, Status};
use wire::relay_diagnostics_server::{RelayDiagnostics, RelayDiagnosticsServer};

#[derive(Clone, Default)]
struct Host {
    requests: Arc<Mutex<Vec<wire::DiagnosticSubmission>>>,
    failures: Arc<AtomicUsize>,
    invalid_arguments: Arc<AtomicUsize>,
    slow: Arc<AtomicBool>,
    oversized_response: Arc<AtomicBool>,
    disabled: bool,
    policy: Option<wire::DiagnosticCapabilities>,
    open_policy: Option<wire::DiagnosticCapabilities>,
    unsupported: bool,
}
impl Host {
    fn capabilities(&self) -> wire::DiagnosticCapabilities {
        let mut capabilities = self.policy.clone().unwrap_or(wire::DiagnosticCapabilities {
            protocol_version: 1,
            snapshot_version: 1,
            provenance_version: 1,
            max_request_bytes: MAX_REQUEST_BYTES as u32,
            max_response_bytes: MAX_RESPONSE_BYTES as u32,
            max_pending_submissions: MAX_PENDING_SUBMISSIONS as u32,
            retry_attempts: 3,
            retry_timeout_millis: 2000,
            ..Default::default()
        });
        capabilities.enabled = !self.disabled;
        capabilities
    }
}
#[tonic::async_trait]
impl RelayDiagnostics for Host {
    async fn get_diagnostic_capabilities(
        &self,
        _: Request<wire::DiagnosticCredentials>,
    ) -> Result<Response<wire::DiagnosticCapabilities>, Status> {
        if self.unsupported {
            return Err(Status::unimplemented("diagnostics unavailable"));
        }
        Ok(Response::new(self.capabilities()))
    }
    async fn open_diagnostic_session(
        &self,
        _: Request<wire::DiagnosticOpenRequest>,
    ) -> Result<Response<wire::DiagnosticSession>, Status> {
        Ok(Response::new(wire::DiagnosticSession {
            session_id: vec![1; 16],
            capabilities: Some(self.open_policy.clone().unwrap_or(self.capabilities())),
        }))
    }
    async fn submit_diagnostic(
        &self,
        request: Request<wire::DiagnosticSubmission>,
    ) -> Result<Response<wire::DiagnosticReceipt>, Status> {
        let request = request.into_inner();
        self.requests.lock().unwrap().push(request.clone());
        if self.slow.load(Ordering::Relaxed) {
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        if self
            .failures
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_sub(1))
            .is_ok()
        {
            return Err(Status::unavailable("lost reply"));
        }
        if matches!(
            &request.operation,
            Some(wire::diagnostic_submission::Operation::Record(_))
        ) && self
            .invalid_arguments
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_sub(1))
            .is_ok()
        {
            return Err(Status::invalid_argument("invalid observation timestamp"));
        }
        let is_record = matches!(
            &request.operation,
            Some(wire::diagnostic_submission::Operation::Record(_))
        );
        let status = match request.operation.unwrap() {
            wire::diagnostic_submission::Operation::RegisterComponent(_) => {
                DiagnosticReceiptStatus::Registered
            }
            wire::diagnostic_submission::Operation::Record(_) => DiagnosticReceiptStatus::Recorded,
            _ => DiagnosticReceiptStatus::Closed,
        };
        let mut response = wire::DiagnosticReceipt {
            submission_sequence: request.sequence,
            status: status as i32,
            component: 1,
            ..Default::default()
        };
        if self.oversized_response.load(Ordering::Relaxed) && is_record {
            response.reason = "x".repeat(5000);
        }
        Ok(Response::new(response))
    }
    async fn read_diagnostic_page(
        &self,
        _: Request<wire::DiagnosticReadRequest>,
    ) -> Result<Response<wire::DiagnosticReadResponse>, Status> {
        Err(Status::permission_denied("no read grant"))
    }
}
struct Server(tokio::task::JoinHandle<()>);
impl Drop for Server {
    fn drop(&mut self) {
        self.0.abort();
    }
}
async fn connect(host: Host) -> (WorkerDiagnostics, Server) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(RelayDiagnosticsServer::new(host))
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
            .await
            .unwrap();
    });
    let channel = Channel::from_shared(endpoint)
        .unwrap()
        .connect()
        .await
        .unwrap();
    (
        WorkerDiagnostics::connect(Ok(channel), "activation".into(), "token".into()).await,
        Server(server),
    )
}
fn observation() -> DiagnosticObservation {
    DiagnosticObservation {
        code: 1,
        count: 2,
        outcome: DiagnosticObservationOutcome::Failure as i32,
        severity: DiagnosticObservationSeverity::Error as i32,
        failure_class: Some("unknown".into()),
        detail: Some("restricted".into()),
        ..Default::default()
    }
}

#[tokio::test]
async fn disabled_facade_skips_authoring_and_registration() {
    let host = Host {
        disabled: true,
        ..Default::default()
    };
    let (facade, _server) = connect(host.clone()).await;
    assert_eq!(facade.availability(), DiagnosticAvailability::Disabled);
    facade
        .component("internal", vec![])
        .await
        .unwrap()
        .try_record_lazy(|| panic!("disabled factory ran"))
        .unwrap();
    facade.flush(Duration::from_secs(2)).await.unwrap();
    facade.close(Duration::from_secs(2)).await.unwrap();
    assert!(host.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn retry_is_identical_and_fences_after_unknown_acceptance() {
    let host = Host::default();
    let (facade, _server) = connect(host.clone()).await;
    let component = facade.component("internal", vec![]).await.unwrap();
    host.failures.store(1, Ordering::Relaxed);
    component.try_record_lazy(observation).unwrap();
    facade.flush(Duration::from_secs(2)).await.unwrap();
    {
        let requests = host.requests.lock().unwrap();
        assert_eq!(requests[1], requests[2]);
        assert_eq!(requests[1].sequence, 2);
    }
    host.failures.store(3, Ordering::Relaxed);
    component.try_record_lazy(observation).unwrap();
    assert_eq!(
        facade.flush(Duration::from_secs(2)).await,
        Err(DiagnosticDeliveryError::Uncertain)
    );
    assert_eq!(facade.delivery_stats().uncertain, 1);
    assert_eq!(
        component.try_record_lazy(|| panic!("fenced factory ran")),
        Err(DiagnosticDeliveryError::Closed)
    );
    assert_eq!(
        facade.close(Duration::from_secs(2)).await,
        Err(DiagnosticDeliveryError::Uncertain)
    );
    let requests = host.requests.lock().unwrap();
    assert_eq!(requests.len(), 6);
    assert_eq!(requests[3], requests[4]);
    assert_eq!(requests[4], requests[5]);
}

#[tokio::test]
async fn invalid_argument_counts_rejection_without_consuming_sequence() {
    let host = Host::default();
    host.invalid_arguments.store(1, Ordering::Relaxed);
    let (facade, _server) = connect(host.clone()).await;
    let component = facade.component("internal", vec![]).await.unwrap();

    component.try_record_lazy(observation).unwrap();
    facade.flush(Duration::from_secs(2)).await.unwrap();
    assert_eq!(facade.delivery_stats().rejected, 1);
    assert_eq!(facade.delivery_stats().uncertain, 0);

    let mut next = observation();
    next.count = 3;
    component.try_record_lazy(|| next).unwrap();
    facade.flush(Duration::from_secs(2)).await.unwrap();
    let requests = host.requests.lock().unwrap();
    let record_sequences = requests
        .iter()
        .filter(|request| {
            matches!(
                request.operation,
                Some(wire::diagnostic_submission::Operation::Record(_))
            )
        })
        .map(|request| request.sequence)
        .collect::<Vec<_>>();
    assert_eq!(record_sequences, vec![2, 2]);
    assert_eq!(facade.delivery_stats().rejected, 1);
}

#[tokio::test]
async fn applies_host_delivery_policy_and_rejects_invalid_policy_inertly() {
    let mut capabilities = Host::default().capabilities();
    capabilities.max_request_bytes = 1024;
    capabilities.max_response_bytes = 4096;
    capabilities.max_pending_submissions = 2;
    capabilities.retry_attempts = 2;
    capabilities.retry_timeout_millis = 25;
    let host = Host {
        policy: Some(capabilities),
        ..Default::default()
    };
    let (facade, _server) = connect(host.clone()).await;
    let policy = facade.policy.unwrap();
    assert_eq!(policy.max_request_bytes, 1024);
    assert_eq!(policy.max_response_bytes, 4096);
    assert_eq!(policy.max_pending_submissions, 2);
    assert_eq!(policy.retry_attempts, 2);
    assert_eq!(policy.retry_timeout, Duration::from_millis(25));
    assert_eq!(facade.sender.as_ref().unwrap().max_capacity(), 2);

    let component = facade.component("internal", vec![]).await.unwrap();
    let mut over_limit = observation();
    over_limit.detail = Some("x".repeat(800));
    assert_eq!(
        component.try_record_lazy(|| over_limit),
        Err(DiagnosticDeliveryError::Invalid)
    );
    let mut valid = observation();
    valid.detail = Some("x".repeat(600));
    component.try_record_lazy(|| valid).unwrap();
    facade.flush(Duration::from_secs(2)).await.unwrap();

    host.slow.store(true, Ordering::Relaxed);
    component.try_record_lazy(observation).unwrap();
    assert_eq!(
        facade.flush(Duration::from_secs(2)).await,
        Err(DiagnosticDeliveryError::Uncertain)
    );
    let requests = host.requests.lock().unwrap();
    assert!(requests[1].encoded_len() <= policy.max_request_bytes);
    assert_eq!(requests[2], requests[3]);
    assert_eq!(requests.len(), 4); // registration, record, and two bounded attempts
    drop(requests);

    let mut response_caps = Host::default().capabilities();
    response_caps.max_response_bytes = 4096;
    response_caps.retry_attempts = 1;
    let response_host = Host {
        policy: Some(response_caps),
        ..Default::default()
    };
    response_host
        .oversized_response
        .store(true, Ordering::Relaxed);
    let (response_limited, _server) = connect(response_host).await;
    let response_component = response_limited
        .component("internal", vec![])
        .await
        .unwrap();
    response_component.try_record_lazy(observation).unwrap();
    assert_eq!(
        response_limited.flush(Duration::from_secs(2)).await,
        Err(DiagnosticDeliveryError::Uncertain)
    );
    assert_eq!(response_limited.delivery_stats().uncertain, 1);

    let mut changed = Host::default().capabilities();
    changed.retry_attempts = 2;
    let changed_host = Host {
        open_policy: Some(changed),
        ..Default::default()
    };
    let (changed_session, _server) = connect(changed_host).await;
    assert_eq!(
        changed_session.availability(),
        DiagnosticAvailability::Unsupported
    );

    let mut invalid = Host::default().capabilities();
    invalid.retry_attempts = 0;
    let invalid_host = Host {
        policy: Some(invalid),
        ..Default::default()
    };
    let (inert, _server) = connect(invalid_host.clone()).await;
    assert_eq!(inert.availability(), DiagnosticAvailability::Unsupported);
    inert.component("internal", vec![]).await.unwrap();
    assert!(invalid_host.requests.lock().unwrap().is_empty());

    let old_host = Host {
        unsupported: true,
        ..Default::default()
    };
    let (unsupported, _server) = connect(old_host).await;
    assert_eq!(
        unsupported.availability(),
        DiagnosticAvailability::Unsupported
    );
}

#[tokio::test]
async fn queue_admission_skips_factory_and_freezes_bounded_bytes() {
    let policy = DeliveryPolicy {
        max_request_bytes: MAX_REQUEST_BYTES,
        max_response_bytes: MAX_RESPONSE_BYTES,
        max_pending_submissions: MAX_PENDING_SUBMISSIONS,
        retry_attempts: 3,
        retry_timeout: Duration::from_secs(2),
    };
    let (sender, mut receiver) = mpsc::channel(policy.max_pending_submissions);
    let facade = WorkerDiagnostics {
        availability: DiagnosticAvailability::Available,
        sender: Some(sender),
        state: Arc::new(DeliveryState::default()),
        task: None,
        policy: Some(policy),
    };
    let component = DiagnosticComponent {
        diagnostics: facade.clone(),
        handle: 7,
    };
    let mut large = observation();
    large.detail = Some("x".repeat(MAX_REQUEST_BYTES));
    assert_eq!(
        component.try_record_lazy(|| large),
        Err(DiagnosticDeliveryError::Invalid)
    );
    for _ in 0..MAX_PENDING_SUBMISSIONS {
        component.try_record_lazy(observation).unwrap();
    }
    assert_eq!(
        component.try_record_lazy(|| panic!("full factory ran")),
        Err(DiagnosticDeliveryError::Full)
    );
    assert_eq!(facade.delivery_stats().dropped, 2);
    let Pending::Submission { frozen, .. } = receiver.recv().await.unwrap() else {
        panic!("expected submission")
    };
    let request = wire::DiagnosticSubmission::decode(frozen.as_slice()).unwrap();
    let Some(wire::diagnostic_submission::Operation::Record(observation)) = request.operation
    else {
        panic!("expected record")
    };
    assert_eq!(observation.component, 7);
    assert!(frozen.len() < MAX_REQUEST_BYTES);
}

#[tokio::test]
async fn closing_aborts_a_blocked_queue_within_deadline() {
    let policy = DeliveryPolicy {
        max_request_bytes: MAX_REQUEST_BYTES,
        max_response_bytes: MAX_RESPONSE_BYTES,
        max_pending_submissions: MAX_PENDING_SUBMISSIONS,
        retry_attempts: 3,
        retry_timeout: Duration::from_secs(2),
    };
    let (sender, mut receiver) = mpsc::channel(policy.max_pending_submissions);
    let state = Arc::new(DeliveryState::default());
    let handle = tokio::spawn(async move {
        let _ = &mut receiver;
        std::future::pending::<()>().await;
    });
    let facade = WorkerDiagnostics {
        availability: DiagnosticAvailability::Available,
        sender: Some(sender),
        state,
        task: Some(Arc::new(DeliveryTask(handle.abort_handle()))),
        policy: Some(policy),
    };
    assert_eq!(
        facade.close(Duration::from_millis(10)).await,
        Err(DiagnosticDeliveryError::Uncertain)
    );
    assert!(handle.await.unwrap_err().is_cancelled());
    assert!(facade.state.stopped.load(Ordering::Acquire));
}
