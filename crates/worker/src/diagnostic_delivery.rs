// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Single-flight delivery with one immutable request and bounded retry attempts.

use super::*;
use nemo_relay_worker_proto::DIAGNOSTIC_NOT_CONSUMED_METADATA;

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
    mut client: wire::relay_diagnostics_service_client::RelayDiagnosticsServiceClient<Channel>,
    credentials: wire::Credentials,
    session_id: Vec<u8>,
    receiver: mpsc::Receiver<Pending>,
    state: Arc<DeliveryState>,
    policy: DeliveryPolicy,
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
                let mut request = wire::SubmitDiagnosticRequest::decode(frozen.as_slice())
                    .expect("SDK-frozen request");
                request.credentials = Some(credentials.clone());
                request.session_id = session_id.clone();
                request.sequence = sequence;
                queue.in_flight = true;
                let result = deliver(&mut client, &request, &state, &mut sequence, policy).await;
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
    client: &mut wire::relay_diagnostics_service_client::RelayDiagnosticsServiceClient<Channel>,
    request: &wire::SubmitDiagnosticRequest,
    state: &DeliveryState,
    sequence: &mut u64,
    policy: DeliveryPolicy,
) -> Result<DiagnosticReceipt, DiagnosticDeliveryError> {
    if request.encoded_len() > policy.max_request_bytes {
        increment(&state.dropped);
        return Err(DiagnosticDeliveryError::Invalid);
    }
    let mut result = Err(DiagnosticDeliveryError::Uncertain);
    let mut acceptance_uncertain = false;
    for _ in 0..policy.retry_attempts {
        if state.stopped.load(Ordering::Acquire) {
            break;
        }
        let mut rpc = tonic::Request::new(request.clone());
        rpc.set_timeout(policy.retry_timeout);
        match tokio::time::timeout(policy.retry_timeout, client.submit_diagnostic(rpc)).await {
            Ok(Ok(response))
                if response
                    .get_ref()
                    .receipt
                    .as_ref()
                    .is_some_and(|v| v.submission_sequence == *sequence) =>
            {
                let response = response.into_inner();
                if response.encoded_len() > policy.max_response_bytes
                    || nemo_relay_worker_proto::validation::validate(&response).is_err()
                {
                    break;
                }
                let Some(response) = response.receipt else {
                    break;
                };
                if !receipt_matches_operation(&response, request.operation.as_ref()) {
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
            Ok(Err(error))
                if error.code() == Code::InvalidArgument
                    || (error.code() == Code::ResourceExhausted
                        && error
                            .metadata()
                            .get(DIAGNOSTIC_NOT_CONSUMED_METADATA)
                            .is_some_and(|value| value == "1")) =>
            {
                // Admission rejected this attempt. A previous attempt may still
                // have been accepted: keep retrying the identical request then.
                if !acceptance_uncertain {
                    increment(&state.rejected);
                    result = Err(DiagnosticDeliveryError::NotConsumed);
                    break;
                }
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
            _ => acceptance_uncertain = true,
        }
    }
    result
}

/// A receipt acknowledges this operation, not merely a consumed sequence.
fn receipt_matches_operation(
    receipt: &DiagnosticReceipt,
    operation: Option<&wire::submit_diagnostic_request::Operation>,
) -> bool {
    use wire::submit_diagnostic_request::Operation;
    match DiagnosticReceiptStatus::try_from(receipt.status) {
        Ok(DiagnosticReceiptStatus::Disabled | DiagnosticReceiptStatus::Rejected) => true,
        Ok(DiagnosticReceiptStatus::Registered) => match operation {
            Some(Operation::RegisterComponent(registration)) => {
                receipt.code_count as usize == registration.codes.len()
            }
            _ => false,
        },
        Ok(DiagnosticReceiptStatus::Recorded) => matches!(operation, Some(Operation::Record(_))),
        Ok(DiagnosticReceiptStatus::Closed) => matches!(
            operation,
            Some(Operation::CloseComponent(_) | Operation::CloseSession(_))
        ),
        _ => false,
    }
}
