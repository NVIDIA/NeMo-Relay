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
    record_errors: Arc<Mutex<std::collections::VecDeque<Status>>>,
    slow: Arc<AtomicBool>,
    oversized_response: Arc<AtomicBool>,
    disabled: bool,
    policy: Option<wire::DiagnosticCapabilities>,
    open_policy: Option<wire::DiagnosticCapabilities>,
    unsupported: bool,
    bootstrap_error: Option<tonic::Code>,
    open_error: Option<tonic::Code>,
    session: Option<wire::DiagnosticSession>,
    record_gate: Option<Arc<tokio::sync::Notify>>,
    record_entered: Arc<tokio::sync::Notify>,
    receipt_status: Option<i32>,
    wrong_sequence: bool,
    open_calls: Arc<AtomicUsize>,
}
impl Host {
    fn capabilities(&self) -> wire::DiagnosticCapabilities {
        let mut capabilities = self.policy.unwrap_or(wire::DiagnosticCapabilities {
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
        if let Some(code) = self.bootstrap_error {
            return Err(Status::new(code, "bootstrap unavailable"));
        }
        if self.unsupported {
            return Err(Status::unimplemented("diagnostics unavailable"));
        }
        Ok(Response::new(self.capabilities()))
    }
    async fn open_diagnostic_session(
        &self,
        _: Request<wire::DiagnosticOpenRequest>,
    ) -> Result<Response<wire::DiagnosticSession>, Status> {
        self.open_calls.fetch_add(1, Ordering::Relaxed);
        if let Some(code) = self.open_error {
            return Err(Status::new(code, "open unavailable"));
        }
        if let Some(session) = &self.session {
            return Ok(Response::new(session.clone()));
        }
        Ok(Response::new(wire::DiagnosticSession {
            session_id: vec![1; 16],
            capabilities: Some(self.open_policy.unwrap_or(self.capabilities())),
        }))
    }
    async fn submit_diagnostic(
        &self,
        request: Request<wire::DiagnosticSubmission>,
    ) -> Result<Response<wire::DiagnosticReceipt>, Status> {
        let request = request.into_inner();
        self.requests.lock().unwrap().push(request.clone());
        if matches!(
            &request.operation,
            Some(wire::diagnostic_submission::Operation::Record(_))
        ) && let Some(error) = self.record_errors.lock().unwrap().pop_front()
        {
            return Err(error);
        }
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
        if is_record {
            self.record_entered.notify_one();
            if let Some(gate) = &self.record_gate {
                gate.notified().await;
            }
        }
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
        if is_record {
            if let Some(status) = self.receipt_status {
                response.status = status;
            }
            if self.wrong_sequence {
                response.submission_sequence += 1;
            }
        }
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
async fn start_host(host: Host) -> (Channel, Server) {
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
    (channel, Server(server))
}
async fn connect(host: Host) -> (WorkerDiagnostics, Server) {
    let (channel, server) = start_host(host).await;
    (
        WorkerDiagnostics::connect(Ok(channel), "activation".into(), "token".into()).await,
        server,
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
    {
        let requests = host.requests.lock().unwrap();
        assert!(requests[1].encoded_len() <= policy.max_request_bytes);
        assert_eq!(requests[2], requests[3]);
        assert_eq!(requests.len(), 4); // registration, record, and two bounded attempts
    }

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
async fn closing_aborts_real_delivery_and_completes_queued_waiters() {
    let host = Host {
        record_gate: Some(Arc::new(tokio::sync::Notify::new())),
        ..Default::default()
    };
    let (facade, _server) = connect(host.clone()).await;
    let component = facade.component("internal", vec![]).await.unwrap();
    component.try_record_lazy(observation).unwrap();
    tokio::time::timeout(Duration::from_secs(2), host.record_entered.notified())
        .await
        .unwrap();
    component.try_record_lazy(observation).unwrap();
    // Poll these public calls once so their requests are queued behind the
    // blocked record, without relying on scheduler sleeps.
    let registration = facade.component("queued", vec![]);
    tokio::pin!(registration);
    assert!(futures_util::poll!(&mut registration).is_pending());
    let flush = facade.flush(Duration::from_secs(2));
    tokio::pin!(flush);
    assert!(futures_util::poll!(&mut flush).is_pending());
    assert_eq!(
        facade.close(Duration::from_millis(10)).await,
        Err(DiagnosticDeliveryError::Uncertain)
    );
    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(2), registration)
            .await
            .unwrap(),
        Err(DiagnosticDeliveryError::Closed)
    ));
    assert_eq!(flush.await, Err(DiagnosticDeliveryError::Closed));
    assert_eq!(
        facade.delivery_stats(),
        DiagnosticDeliveryStats {
            uncertain: 1,
            dropped: 2,
            rejected: 0,
        }
    );
    assert_eq!(host.requests.lock().unwrap().len(), 2);
    assert_eq!(
        component.try_record_lazy(|| panic!("closed factory ran")),
        Err(DiagnosticDeliveryError::Closed)
    );
    host.record_gate.as_ref().unwrap().notify_waiters();
}

#[tokio::test]
async fn component_and_session_close_follow_record_sequence() {
    let host = Host::default();
    let (facade, _server) = connect(host.clone()).await;
    let component = facade.component("internal", vec![]).await.unwrap();
    component.try_record_lazy(observation).unwrap();
    component.close().await.unwrap();
    facade.close(Duration::from_secs(2)).await.unwrap();
    let requests = host.requests.lock().unwrap();
    assert_eq!(
        requests.iter().map(|r| r.sequence).collect::<Vec<_>>(),
        [1, 2, 3, 4]
    );
    assert!(matches!(
        requests[2].operation,
        Some(wire::diagnostic_submission::Operation::CloseComponent(
            wire::DiagnosticComponentClose { component: 1 }
        ))
    ));
    assert!(matches!(
        requests[3].operation,
        Some(wire::diagnostic_submission::Operation::CloseSession(_))
    ));
    assert_eq!(facade.delivery_stats(), DiagnosticDeliveryStats::default());
}

fn admission_rejection() -> Status {
    let mut status = Status::resource_exhausted("at capacity");
    status.metadata_mut().insert(
        nemo_relay_worker_proto::DIAGNOSTIC_NOT_CONSUMED_METADATA,
        "1".parse().unwrap(),
    );
    status
}

#[tokio::test]
async fn capacity_rejection_counts_loss_and_keeps_delivery_open() {
    let host = Host::default();
    host.record_errors
        .lock()
        .unwrap()
        .push_back(admission_rejection());
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
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[1].sequence, requests[2].sequence);
    assert_ne!(requests[1], requests[2]);
}

#[tokio::test]
async fn rejection_cannot_erase_an_earlier_unknown_acceptance() {
    for rejection in [
        admission_rejection(),
        Status::invalid_argument("invalid observation"),
    ] {
        for recover in [true, false] {
            let host = Host::default();
            let mut errors = vec![Status::unavailable("lost receipt"), rejection.clone()];
            if !recover {
                errors.push(rejection.clone());
            }
            host.record_errors.lock().unwrap().extend(errors);
            let (facade, _server) = connect(host.clone()).await;
            let component = facade.component("internal", vec![]).await.unwrap();
            component.try_record_lazy(observation).unwrap();
            let flushed = facade.flush(Duration::from_secs(2)).await;
            assert_eq!(facade.delivery_stats().rejected, 0);
            if recover {
                flushed.unwrap();
                assert_eq!(facade.delivery_stats().uncertain, 0);
                component.try_record_lazy(observation).unwrap();
                facade.flush(Duration::from_secs(2)).await.unwrap();
            } else {
                assert_eq!(flushed, Err(DiagnosticDeliveryError::Uncertain));
                assert_eq!(facade.delivery_stats().uncertain, 1);
                assert_eq!(
                    component.try_record_lazy(observation),
                    Err(DiagnosticDeliveryError::Closed)
                );
            }
            let requests = host.requests.lock().unwrap();
            assert_eq!(requests[1], requests[2]);
            assert_eq!(requests[2], requests[3]);
            if recover {
                assert_eq!(requests[4].sequence, requests[3].sequence + 1);
            }
        }
    }
}

#[tokio::test]
async fn unmarked_resource_errors_do_not_claim_nonacceptance() {
    let host = Host::default();
    host.record_errors
        .lock()
        .unwrap()
        .extend((0..3).map(|_| Status::resource_exhausted("local receive limit")));
    let (facade, _server) = connect(host).await;
    let component = facade.component("internal", vec![]).await.unwrap();
    component.try_record_lazy(observation).unwrap();
    assert_eq!(
        facade.flush(Duration::from_secs(2)).await,
        Err(DiagnosticDeliveryError::Uncertain)
    );
    assert_eq!(facade.delivery_stats().rejected, 0);
    assert_eq!(facade.delivery_stats().uncertain, 1);
}

#[tokio::test]
async fn incompatible_or_unavailable_sessions_are_inert() {
    let baseline = Host::default().capabilities();
    let mut cases = Vec::new();
    for field in 0..3 {
        let mut policy = baseline;
        match field {
            0 => policy.protocol_version = 2,
            1 => policy.snapshot_version = 2,
            _ => policy.provenance_version = 2,
        }
        cases.push((
            Host {
                policy: Some(policy),
                ..Default::default()
            },
            DiagnosticAvailability::Unsupported,
        ));
    }
    for (session, availability) in [
        (
            wire::DiagnosticSession {
                session_id: vec![1; 16],
                capabilities: None,
            },
            DiagnosticAvailability::Unsupported,
        ),
        (
            wire::DiagnosticSession {
                session_id: vec![1; 16],
                capabilities: Some(wire::DiagnosticCapabilities {
                    enabled: false,
                    ..baseline
                }),
            },
            DiagnosticAvailability::Disabled,
        ),
        (
            wire::DiagnosticSession {
                session_id: vec![1; 15],
                capabilities: Some(baseline),
            },
            DiagnosticAvailability::Unavailable,
        ),
    ] {
        cases.push((
            Host {
                session: Some(session),
                ..Default::default()
            },
            availability,
        ));
    }
    cases.push((
        Host {
            bootstrap_error: Some(tonic::Code::Unavailable),
            ..Default::default()
        },
        DiagnosticAvailability::Unavailable,
    ));
    cases.push((
        Host {
            open_error: Some(tonic::Code::Unavailable),
            ..Default::default()
        },
        DiagnosticAvailability::Unavailable,
    ));
    for (host, availability) in cases {
        let (facade, _server) = connect(host.clone()).await;
        assert_eq!(facade.availability(), availability);
        let component = facade.component("internal", vec![]).await.unwrap();
        component
            .try_record_lazy(|| panic!("inert factory ran"))
            .unwrap();
        component.close().await.unwrap();
        facade.flush(Duration::from_secs(1)).await.unwrap();
        facade.close(Duration::from_secs(1)).await.unwrap();
        assert!(host.requests.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn delivery_policy_safety_boundaries_are_enforced_at_initialization() {
    for (field, minimum, maximum) in [
        ("request", 258, 16384),
        ("response", 256, 131072),
        ("pending", 1, 32),
        ("attempts", 1, 8),
        ("timeout", 1, 10000),
    ] {
        for value in [minimum - 1, minimum, maximum, maximum + 1] {
            let mut policy = Host::default().capabilities();
            match field {
                "request" => policy.max_request_bytes = value,
                "response" => policy.max_response_bytes = value,
                "pending" => policy.max_pending_submissions = value,
                "attempts" => policy.retry_attempts = value,
                _ => policy.retry_timeout_millis = value,
            }
            let (facade, _server) = connect(Host {
                policy: Some(policy),
                ..Default::default()
            })
            .await;
            let accepted = (minimum..=maximum).contains(&value);
            assert_eq!(
                facade.availability(),
                if accepted {
                    DiagnosticAvailability::Available
                } else {
                    DiagnosticAvailability::Unsupported
                },
                "{field}={value}"
            );
            if !accepted {
                facade
                    .component("internal", vec![])
                    .await
                    .unwrap()
                    .try_record_lazy(|| panic!("invalid policy ran producer"))
                    .unwrap();
            }
        }
    }
}

#[tokio::test]
async fn malformed_receipts_fence_delivery_without_claiming_rejection() {
    for (receipt_status, wrong_sequence) in [(Some(0), false), (Some(999), false), (None, true)] {
        let host = Host {
            receipt_status,
            wrong_sequence,
            ..Default::default()
        };
        let (facade, _server) = connect(host).await;
        let component = facade.component("internal", vec![]).await.unwrap();
        component.try_record_lazy(observation).unwrap();
        assert_eq!(
            facade.flush(Duration::from_secs(2)).await,
            Err(DiagnosticDeliveryError::Uncertain)
        );
        assert_eq!(
            facade.delivery_stats(),
            DiagnosticDeliveryStats {
                uncertain: 1,
                ..Default::default()
            }
        );
        assert_eq!(
            component.try_record_lazy(|| panic!("fenced producer ran")),
            Err(DiagnosticDeliveryError::Closed)
        );
    }
}

#[tokio::test]
async fn terminal_rejection_consumes_sequence_and_delivery_continues() {
    let host = Host {
        receipt_status: Some(DiagnosticReceiptStatus::Rejected as i32),
        ..Default::default()
    };
    let (facade, _server) = connect(host.clone()).await;
    let component = facade.component("internal", vec![]).await.unwrap();
    component.try_record_lazy(observation).unwrap();
    facade.flush(Duration::from_secs(2)).await.unwrap();
    component.close().await.unwrap();
    facade.close(Duration::from_secs(2)).await.unwrap();
    assert_eq!(
        facade.delivery_stats(),
        DiagnosticDeliveryStats {
            rejected: 1,
            ..Default::default()
        }
    );
    assert_eq!(
        host.requests
            .lock()
            .unwrap()
            .iter()
            .map(|r| r.sequence)
            .collect::<Vec<_>>(),
        [1, 2, 3, 4]
    );
}

#[tokio::test]
async fn runtime_initializes_one_shared_diagnostic_session_under_concurrency() {
    let host = Host::default();
    let (channel, _server) = start_host(host.clone()).await;
    let runtime = crate::PluginRuntime {
        diagnostics: Arc::new(tokio::sync::OnceCell::new()),
        activation_id: "activation".into(),
        auth_token: "token".into(),
        host_endpoint: "unused".into(),
        host_channel: Arc::new(tokio::sync::OnceCell::new_with(Some(channel))),
        conditional_middleware_callbacks: Default::default(),
    };
    let cloned = runtime.clone();
    let (first, second) = tokio::join!(runtime.diagnostics(), cloned.diagnostics());
    assert!(std::ptr::eq(first, second));
    assert_eq!(first.availability(), DiagnosticAvailability::Available);
    assert_eq!(host.open_calls.load(Ordering::Relaxed), 1);
    first.close(Duration::from_secs(2)).await.unwrap();
}

#[tokio::test]
async fn hydrated_frame_limit_drops_without_transmission_or_sequence_consumption() {
    let mut policy = Host::default().capabilities();
    policy.max_request_bytes = 1024;
    let host = Host {
        policy: Some(policy),
        ..Default::default()
    };
    let (channel, _server) = start_host(host.clone()).await;
    let facade =
        WorkerDiagnostics::connect(Ok(channel), "activation".into(), "t".repeat(900)).await;
    assert_eq!(facade.availability(), DiagnosticAvailability::Available);
    let component = facade.component("internal", vec![]).await.unwrap();

    // This payload passes local admission, but its credentials and session
    // envelope bring the complete submission above the negotiated frame cap.
    let mut oversized_frame = observation();
    oversized_frame.detail = Some("x".repeat(200));
    component.try_record_lazy(|| oversized_frame).unwrap();
    facade.flush(Duration::from_secs(2)).await.unwrap();
    assert_eq!(host.requests.lock().unwrap().len(), 1);
    assert_eq!(
        facade.delivery_stats(),
        DiagnosticDeliveryStats {
            dropped: 1,
            ..Default::default()
        }
    );

    let mut smaller = observation();
    smaller.count = 3;
    component.try_record_lazy(|| smaller).unwrap();
    facade.flush(Duration::from_secs(2)).await.unwrap();
    {
        let requests = host.requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[1].sequence, 2);
        assert!(requests[1].encoded_len() <= policy.max_request_bytes as usize);
        assert!(matches!(
            &requests[1].operation,
            Some(wire::diagnostic_submission::Operation::Record(record)) if record.count == 3
        ));
    }
    assert_eq!(
        facade.delivery_stats(),
        DiagnosticDeliveryStats {
            dropped: 1,
            ..Default::default()
        }
    );
    facade.close(Duration::from_secs(2)).await.unwrap();
}

#[tokio::test]
async fn terminal_authorization_and_session_errors_fence_without_retrying() {
    for code in [
        Code::PermissionDenied,
        Code::Unauthenticated,
        Code::FailedPrecondition,
        Code::OutOfRange,
        Code::AlreadyExists,
    ] {
        let host = Host::default();
        host.record_errors
            .lock()
            .unwrap()
            .push_back(Status::new(code, "terminal session failure"));
        let (facade, _server) = connect(host.clone()).await;
        let component = facade.component("internal", vec![]).await.unwrap();
        component.try_record_lazy(observation).unwrap();
        assert_eq!(
            facade.flush(Duration::from_secs(2)).await,
            Err(DiagnosticDeliveryError::Uncertain),
            "{code:?}"
        );
        assert_eq!(
            facade.delivery_stats(),
            DiagnosticDeliveryStats {
                uncertain: 1,
                ..Default::default()
            },
            "{code:?}"
        );
        assert_eq!(
            component.try_record_lazy(|| panic!("fenced producer ran")),
            Err(DiagnosticDeliveryError::Closed)
        );
        // One registration and one failed record: a terminal error is never retried.
        assert_eq!(host.requests.lock().unwrap().len(), 2, "{code:?}");
        assert_eq!(
            facade.close(Duration::from_secs(2)).await,
            Err(DiagnosticDeliveryError::Uncertain)
        );
    }
}
