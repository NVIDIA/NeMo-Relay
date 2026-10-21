// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::diagnostics::{DiagnosticLimits, DiagnosticQuery, DiagnosticsSettings, SourceIdentity};
use wire::diagnostic_submission::Operation;
use wire::relay_diagnostics_server::RelayDiagnostics;

fn context(limits: DiagnosticLimits) -> Diagnostics {
    Diagnostics::from_settings(
        SourceIdentity {
            instance_id: Uuid::now_v7(),
            process_id: None,
            process_role: "test".into(),
            relay_version: "test".into(),
        },
        DiagnosticsSettings {
            enabled: true,
            limits,
        },
    )
    .unwrap()
}
fn host() -> WorkerDiagnostics {
    WorkerDiagnostics::new(context(DiagnosticLimits::default()), Uuid::now_v7())
}
fn open(host: &WorkerDiagnostics) -> Vec<u8> {
    host.open(&wire::DiagnosticOpenRequest {
        protocol_version: 1,
        snapshot_version: 1,
        provenance_version: 1,
        ..Default::default()
    })
    .unwrap()
    .session_id
}
fn registration() -> wire::DiagnosticComponentRegistration {
    wire::DiagnosticComponentRegistration {
        local_name: "plugin.internal".into(),
        codes: vec![wire::DiagnosticCodeDefinition {
            definition: Some(wire::diagnostic_code_definition::Definition::BuiltinName(
                "plugin.internal".into(),
            )),
        }],
    }
}
fn submission(session: &[u8], sequence: u64, operation: Operation) -> wire::DiagnosticSubmission {
    wire::DiagnosticSubmission {
        session_id: session.into(),
        sequence,
        operation: Some(operation),
        ..Default::default()
    }
}
fn observation(count: u64) -> wire::DiagnosticObservation {
    wire::DiagnosticObservation {
        component: 1,
        code: 1,
        count,
        outcome: wire::DiagnosticObservationOutcome::Failure as i32,
        severity: wire::DiagnosticObservationSeverity::Error as i32,
        failure_class: Some("internal".into()),
        detail: Some("restricted-detail-sentinel".into()),
        request_id: Some("restricted-correlation".into()),
        observed_unix_micros: Some(0),
        ..Default::default()
    }
}
fn snapshot(host: &WorkerDiagnostics) -> crate::diagnostics::DiagnosticSnapshot {
    host.context
        .snapshot(&DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap()
}
fn register(host: &WorkerDiagnostics, session: &[u8]) {
    let result = host
        .submit(submission(
            session,
            1,
            Operation::RegisterComponent(registration()),
        ))
        .unwrap();
    assert_eq!(
        result.status,
        wire::DiagnosticReceiptStatus::Registered as i32
    );
    assert_eq!(result.component, 1);
}

#[test]
fn replay_preserves_counts_provenance_and_restricted_projection_boundary() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let host = host();
    let projected = Arc::new(AtomicUsize::new(0));
    let calls = projected.clone();
    let _projection = host
        .context
        .subscribe_projection(move |_| {
            calls.fetch_add(1, Ordering::Relaxed);
        })
        .unwrap();
    let session = open(&host);
    assert_eq!(open(&host), session);
    let registration = submission(&session, 1, Operation::RegisterComponent(registration()));
    let first = host.submit(registration.clone()).unwrap();
    assert_eq!(host.submit(registration).unwrap(), first);
    for (seq, count) in [(2, 2), (3, 3)] {
        let request = submission(&session, seq, Operation::Record(observation(count)));
        let receipt = host.submit(request.clone()).unwrap();
        assert_eq!(
            receipt.status,
            wire::DiagnosticReceiptStatus::Recorded as i32
        );
        assert_eq!(host.submit(request).unwrap(), receipt);
    }
    let snapshot = snapshot(&host);
    assert_eq!(projected.load(Ordering::Relaxed), 0);
    assert_eq!(snapshot.components.len(), 1);
    assert_eq!(
        snapshot.lifetime.totals.recorded_events_total.value.get(),
        2
    );
    assert_eq!(snapshot.lifetime.totals.failures_total.value.get(), 5);
    assert_eq!(snapshot.events.len(), 2);
    assert_remote_provenance(&host, &session, &snapshot);
}

fn assert_remote_provenance(
    host: &WorkerDiagnostics,
    session: &[u8],
    snapshot: &crate::diagnostics::DiagnosticSnapshot,
) {
    for event in &snapshot.events {
        let remote = event.remote.unwrap();
        assert_eq!(remote.source_instance_id.as_bytes(), session);
        assert_eq!(remote.activation_instance_id, host.activation);
        assert_eq!(remote.observed_at.unwrap().timestamp(), 0);
        assert!(remote.received_at.timestamp() > 0);
        assert!(remote.submission_sequence.get() >= 2);
        assert!(remote.accepted_elapsed_millis <= snapshot.lifetime.elapsed_millis);
        assert_eq!(
            event.context.execution.as_ref().unwrap().as_str(),
            "worker_protocol"
        );
    }
}

#[test]
fn rejected_structure_does_not_consume_but_terminal_rejections_do() {
    let host = host();
    let session = open(&host);
    register(&host, &session);
    assert_eq!(
        host.submit(submission(&session, 2, Operation::Record(observation(0))))
            .unwrap_err()
            .code(),
        tonic::Code::InvalidArgument
    );
    let mut bad = observation(1);
    bad.component = 99;
    let bad = submission(&session, 2, Operation::Record(bad));
    let rejected = host.submit(bad.clone()).unwrap();
    assert_eq!(
        rejected.status,
        wire::DiagnosticReceiptStatus::Rejected as i32
    );
    assert_eq!(host.submit(bad).unwrap(), rejected);
    assert_eq!(
        host.submit(submission(&session, 2, Operation::Record(observation(1))))
            .unwrap_err()
            .code(),
        tonic::Code::AlreadyExists
    );
    assert_eq!(
        host.submit(submission(&session, 4, Operation::Record(observation(1))))
            .unwrap_err()
            .code(),
        tonic::Code::OutOfRange
    );
    host.submit(submission(&session, 3, Operation::Record(observation(1))))
        .unwrap();
    assert_eq!(
        host.submit(submission(
            &session,
            1,
            Operation::RegisterComponent(registration())
        ))
        .unwrap_err()
        .code(),
        tonic::Code::OutOfRange
    );
    assert_eq!(
        snapshot(&host)
            .lifetime
            .totals
            .occurrences_total
            .value
            .get(),
        1
    );
}

#[test]
fn close_and_revoke_fence_handles_without_closing_the_runtime_context() {
    let host = host();
    let session = open(&host);
    register(&host, &session);
    let close = submission(
        &session,
        2,
        Operation::CloseComponent(wire::DiagnosticComponentClose { component: 1 }),
    );
    host.submit(close).unwrap();
    assert_eq!(
        host.submit(submission(&session, 3, Operation::Record(observation(1))))
            .unwrap()
            .status,
        wire::DiagnosticReceiptStatus::Rejected as i32
    );
    let close = submission(
        &session,
        4,
        Operation::CloseSession(wire::DiagnosticSessionClose {}),
    );
    let receipt = host.submit(close.clone()).unwrap();
    assert_eq!(host.submit(close).unwrap(), receipt);
    assert!(
        host.submit(submission(&session, 5, Operation::Record(observation(1))))
            .is_err()
    );
    host.revoke();
    assert!(
        host.submit(submission(
            &session,
            4,
            Operation::CloseSession(wire::DiagnosticSessionClose {})
        ))
        .is_err()
    );
    assert!(
        host.context
            .register_component(ComponentRegistration::new("unrelated", &[]))
            .is_ok()
    );
}

#[test]
fn component_quota_is_not_replenished_by_close_and_frames_are_bounded() {
    let host = host();
    let session = open(&host);
    for sequence in 1..=8 {
        let receipt = host
            .submit(submission(
                &session,
                sequence,
                Operation::RegisterComponent(registration()),
            ))
            .unwrap();
        assert_eq!(
            receipt.status,
            wire::DiagnosticReceiptStatus::Registered as i32
        );
    }
    host.submit(submission(
        &session,
        9,
        Operation::CloseComponent(wire::DiagnosticComponentClose { component: 1 }),
    ))
    .unwrap();
    assert_eq!(
        host.submit(submission(
            &session,
            10,
            Operation::RegisterComponent(registration())
        ))
        .unwrap()
        .status,
        wire::DiagnosticReceiptStatus::Rejected as i32
    );
    let mut large = observation(1);
    large.detail = Some("x".repeat(REQUEST_BYTES));
    assert_eq!(
        host.submit(submission(&session, 11, Operation::Record(large)))
            .unwrap_err()
            .code(),
        tonic::Code::InvalidArgument
    );
    assert_eq!(
        host.state
            .lock()
            .unwrap()
            .session
            .as_ref()
            .unwrap()
            .components
            .len(),
        8
    );
}

#[test]
fn failed_extension_admission_rolls_back_unpublished_component() {
    let host = WorkerDiagnostics::new(
        context(DiagnosticLimits {
            max_components: 1,
            max_extensions: 1,
            ..DiagnosticLimits::default()
        }),
        Uuid::now_v7(),
    );
    let session = open(&host);
    let extension = |name: &str| wire::DiagnosticCodeDefinition {
        definition: Some(wire::diagnostic_code_definition::Definition::Extension(
            wire::DiagnosticExtensionDefinition {
                name: name.into(),
                description: "internal success".into(),
                outcome: wire::DiagnosticObservationOutcome::Success as i32,
                allowed_failure_classes: vec![],
            },
        )),
    };
    let bad = wire::DiagnosticComponentRegistration {
        local_name: "internal".into(),
        codes: vec![extension("one"), extension("two")],
    };
    assert_eq!(
        host.submit(submission(&session, 1, Operation::RegisterComponent(bad)))
            .unwrap()
            .status,
        wire::DiagnosticReceiptStatus::Rejected as i32
    );
    let good = wire::DiagnosticComponentRegistration {
        local_name: "internal".into(),
        codes: vec![extension("one")],
    };
    assert_eq!(
        host.submit(submission(&session, 2, Operation::RegisterComponent(good)))
            .unwrap()
            .status,
        wire::DiagnosticReceiptStatus::Registered as i32
    );
    assert_eq!(snapshot(&host).components.len(), 1);
}

#[tokio::test]
async fn authentication_and_read_authority_are_separate() {
    let mut state = super::super::WorkerHostRuntimeState::new("activation".into(), "secret".into());
    state.diagnostics = Some(Arc::new(host()));
    let service = Service {
        state: Arc::new(state),
    };
    let credentials = wire::DiagnosticCredentials {
        activation_id: "activation".into(),
        auth_token: "secret".into(),
    };
    assert!(
        service
            .get_diagnostic_capabilities(Request::new(wire::DiagnosticCredentials::default()))
            .await
            .is_err()
    );
    let capability = service
        .get_diagnostic_capabilities(Request::new(credentials.clone()))
        .await
        .unwrap()
        .into_inner();
    assert!(capability.enabled);
    assert!(!capability.can_read_context);
    assert_eq!(
        service
            .read_diagnostic_page(Request::new(wire::DiagnosticReadRequest {
                credentials: Some(credentials),
                ..Default::default()
            }))
            .await
            .unwrap_err()
            .code(),
        tonic::Code::PermissionDenied
    );
}

#[test]
fn disabled_host_admits_no_session_or_metadata() {
    let host = WorkerDiagnostics::new(Diagnostics::default(), Uuid::now_v7());
    assert!(open(&host).is_empty());
    assert!(host.state.lock().unwrap().session.is_none());
    assert!(!host.capabilities().enabled);
}

#[tokio::test]
async fn diagnostic_decode_limit_rejects_oversized_frames_before_acceptance() {
    let mut state = super::super::WorkerHostRuntimeState::new("activation".into(), "secret".into());
    let context = Arc::new(host());
    state.diagnostics = Some(context.clone());
    let service = super::super::diagnostic_limits::BoundedService(
        wire::relay_diagnostics_server::RelayDiagnosticsServer::new(Service {
            state: Arc::new(state),
        })
        .max_decoding_message_size(REQUEST_BYTES)
        .max_encoding_message_size(RESPONSE_BYTES),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(service)
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
            .await
            .unwrap();
    });
    let channel = tonic::transport::Channel::from_shared(endpoint)
        .unwrap()
        .connect()
        .await
        .unwrap();
    let mut client = wire::relay_diagnostics_client::RelayDiagnosticsClient::new(channel);
    let credentials = wire::DiagnosticCredentials {
        activation_id: "activation".into(),
        auth_token: "secret".into(),
    };
    let session = open(&context);
    let mut value = observation(1);
    value.detail = Some("x".repeat(REQUEST_BYTES));
    let mut request = submission(&session, 1, Operation::Record(value));
    request.credentials = Some(credentials.clone());
    assert!(client.submit_diagnostic(request).await.is_err());
    let mut request = submission(&session, 1, Operation::RegisterComponent(registration()));
    request.credentials = Some(credentials);
    assert_eq!(
        client
            .submit_diagnostic(request)
            .await
            .unwrap()
            .into_inner()
            .status,
        wire::DiagnosticReceiptStatus::Registered as i32
    );
    assert_eq!(
        snapshot(&context)
            .lifetime
            .totals
            .recorded_events_total
            .value
            .get(),
        0
    );
    server.abort();
}
