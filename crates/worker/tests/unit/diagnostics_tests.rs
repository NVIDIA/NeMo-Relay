// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use tonic::{Request, Response, Status};
use wire::relay_diagnostics_server::{RelayDiagnostics, RelayDiagnosticsServer};

#[derive(Clone, Default)]
struct Host {
    requests: Arc<Mutex<Vec<wire::DiagnosticSubmission>>>,
    failures: Arc<AtomicUsize>,
    disabled: bool,
}
#[tonic::async_trait]
impl RelayDiagnostics for Host {
    async fn get_diagnostic_capabilities(
        &self,
        _: Request<wire::DiagnosticCredentials>,
    ) -> Result<Response<wire::DiagnosticCapabilities>, Status> {
        Ok(Response::new(wire::DiagnosticCapabilities {
            enabled: !self.disabled,
            protocol_version: 1,
            snapshot_version: 1,
            provenance_version: 1,
            max_request_bytes: FRAME_BYTES as u32,
            max_pending_submissions: QUEUE_SLOTS as u32,
            ..Default::default()
        }))
    }
    async fn open_diagnostic_session(
        &self,
        _: Request<wire::DiagnosticOpenRequest>,
    ) -> Result<Response<wire::DiagnosticSession>, Status> {
        Ok(Response::new(wire::DiagnosticSession {
            session_id: vec![1; 16],
            ..Default::default()
        }))
    }
    async fn submit_diagnostic(
        &self,
        request: Request<wire::DiagnosticSubmission>,
    ) -> Result<Response<wire::DiagnosticReceipt>, Status> {
        let request = request.into_inner();
        self.requests.lock().unwrap().push(request.clone());
        if self
            .failures
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_sub(1))
            .is_ok()
        {
            return Err(Status::unavailable("lost reply"));
        }
        let status = match request.operation.unwrap() {
            wire::diagnostic_submission::Operation::RegisterComponent(_) => {
                DiagnosticReceiptStatus::Registered
            }
            wire::diagnostic_submission::Operation::Record(_) => DiagnosticReceiptStatus::Recorded,
            _ => DiagnosticReceiptStatus::Closed,
        };
        Ok(Response::new(wire::DiagnosticReceipt {
            submission_sequence: request.sequence,
            status: status as i32,
            component: 1,
            ..Default::default()
        }))
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
    facade.flush(TIMEOUT).await.unwrap();
    facade.close(TIMEOUT).await.unwrap();
    assert!(host.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn retry_is_identical_and_fences_after_unknown_acceptance() {
    let host = Host::default();
    let (facade, _server) = connect(host.clone()).await;
    let component = facade.component("internal", vec![]).await.unwrap();
    host.failures.store(1, Ordering::Relaxed);
    component.try_record_lazy(observation).unwrap();
    facade.flush(TIMEOUT).await.unwrap();
    {
        let requests = host.requests.lock().unwrap();
        assert_eq!(requests[1], requests[2]);
        assert_eq!(requests[1].sequence, 2);
    }
    host.failures.store(3, Ordering::Relaxed);
    component.try_record_lazy(observation).unwrap();
    assert_eq!(
        facade.flush(TIMEOUT).await,
        Err(DiagnosticDeliveryError::Uncertain)
    );
    assert_eq!(facade.delivery_stats().uncertain, 1);
    assert_eq!(
        component.try_record_lazy(|| panic!("fenced factory ran")),
        Err(DiagnosticDeliveryError::Closed)
    );
    assert_eq!(
        facade.close(TIMEOUT).await,
        Err(DiagnosticDeliveryError::Uncertain)
    );
    let requests = host.requests.lock().unwrap();
    assert_eq!(requests.len(), 6);
    assert_eq!(requests[3], requests[4]);
    assert_eq!(requests[4], requests[5]);
}

#[tokio::test]
async fn queue_admission_skips_factory_and_freezes_bounded_bytes() {
    let (sender, mut receiver) = mpsc::channel(QUEUE_SLOTS);
    let facade = WorkerDiagnostics {
        availability: DiagnosticAvailability::Available,
        sender: Some(sender),
        state: Arc::new(DeliveryState::default()),
        task: None,
    };
    let component = DiagnosticComponent {
        diagnostics: facade.clone(),
        handle: 7,
    };
    let mut large = observation();
    large.detail = Some("x".repeat(FRAME_BYTES));
    assert_eq!(
        component.try_record_lazy(|| large),
        Err(DiagnosticDeliveryError::Invalid)
    );
    for _ in 0..QUEUE_SLOTS {
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
    assert!(frozen.len() < FRAME_BYTES);
}

#[tokio::test]
async fn closing_aborts_a_blocked_queue_within_deadline() {
    let (sender, mut receiver) = mpsc::channel(QUEUE_SLOTS);
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
    };
    assert_eq!(
        facade.close(Duration::from_millis(10)).await,
        Err(DiagnosticDeliveryError::Uncertain)
    );
    assert!(handle.await.unwrap_err().is_cancelled());
    assert!(facade.state.stopped.load(Ordering::Acquire));
}
