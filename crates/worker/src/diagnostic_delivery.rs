// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Single-flight delivery with one immutable request and bounded retry attempts.

use super::*;

struct Queue {
    receiver: mpsc::Receiver<Pending>,
    state: Arc<DeliveryState>,
    in_flight: bool,
}
impl Drop for Queue {
    fn drop(&mut self) {
        self.state.stopped.store(true, Ordering::Release);
        if self.in_flight {
            increment(&self.state.uncertain);
        }
        self.receiver.close();
        while let Ok(pending) = self.receiver.try_recv() {
            match pending {
                Pending::Submission { receipt, .. } => {
                    increment(&self.state.dropped);
                    if let Some(receipt) = receipt {
                        let _ = receipt.send(Err(DiagnosticDeliveryError::Closed));
                    }
                }
                Pending::Flush(done) => {
                    let _ = done.send(Err(DiagnosticDeliveryError::Closed));
                }
            }
        }
    }
}

pub(super) async fn run(
    mut client: wire::relay_diagnostics_client::RelayDiagnosticsClient<Channel>,
    credentials: wire::DiagnosticCredentials,
    session_id: Vec<u8>,
    receiver: mpsc::Receiver<Pending>,
    state: Arc<DeliveryState>,
) {
    let mut queue = Queue {
        receiver,
        state: state.clone(),
        in_flight: false,
    };
    let mut sequence = 1u64;
    while let Some(pending) = queue.receiver.recv().await {
        match pending {
            Pending::Flush(done) => {
                let result = if state.uncertain.load(Ordering::Relaxed) > 0 {
                    Err(DiagnosticDeliveryError::Uncertain)
                } else if state.stopped.load(Ordering::Acquire) {
                    Err(DiagnosticDeliveryError::Closed)
                } else {
                    Ok(())
                };
                let _ = done.send(result);
            }
            Pending::Submission { frozen, receipt } => {
                if state.stopped.load(Ordering::Acquire) {
                    increment(&state.dropped);
                    if let Some(receipt) = receipt {
                        let _ = receipt.send(Err(DiagnosticDeliveryError::Closed));
                    }
                    continue;
                }
                let mut request = wire::DiagnosticSubmission::decode(frozen.as_slice())
                    .expect("SDK-frozen request");
                request.credentials = Some(credentials.clone());
                request.session_id = session_id.clone();
                request.sequence = sequence;
                queue.in_flight = true;
                let result = deliver(&mut client, &request, &state, &mut sequence).await;
                if result == Err(DiagnosticDeliveryError::Uncertain) {
                    increment(&state.uncertain);
                    state.stopped.store(true, Ordering::Release);
                }
                if let Some(receipt) = receipt {
                    let _ = receipt.send(result);
                }
                queue.in_flight = false;
            }
        }
    }
}

async fn deliver(
    client: &mut wire::relay_diagnostics_client::RelayDiagnosticsClient<Channel>,
    request: &wire::DiagnosticSubmission,
    state: &DeliveryState,
    sequence: &mut u64,
) -> Result<DiagnosticReceipt, DiagnosticDeliveryError> {
    let mut result = Err(DiagnosticDeliveryError::Uncertain);
    for _ in 0..3 {
        if state.stopped.load(Ordering::Acquire) {
            break;
        }
        let mut rpc = tonic::Request::new(request.clone());
        rpc.set_timeout(TIMEOUT);
        match tokio::time::timeout(TIMEOUT, client.submit_diagnostic(rpc)).await {
            Ok(Ok(response)) if response.get_ref().submission_sequence == *sequence => {
                let response = response.into_inner();
                let valid = DiagnosticReceiptStatus::try_from(response.status)
                    .is_ok_and(|s| s != DiagnosticReceiptStatus::Unspecified);
                if !valid {
                    break;
                }
                if let Some(next) = sequence.checked_add(1) {
                    *sequence = next;
                } else {
                    state.stopped.store(true, Ordering::Release);
                }
                if response.status == DiagnosticReceiptStatus::Rejected as i32 {
                    increment(&state.rejected);
                    result = Err(DiagnosticDeliveryError::Rejected);
                } else {
                    result = Ok(response);
                }
                break;
            }
            Ok(Err(error)) if error.code() == Code::InvalidArgument => {
                result = Err(DiagnosticDeliveryError::NotConsumed);
                break;
            }
            Ok(Err(error))
                if matches!(
                    error.code(),
                    Code::PermissionDenied
                        | Code::Unauthenticated
                        | Code::FailedPrecondition
                        | Code::OutOfRange
                        | Code::AlreadyExists
                ) =>
            {
                break;
            }
            _ => {}
        }
    }
    result
}
