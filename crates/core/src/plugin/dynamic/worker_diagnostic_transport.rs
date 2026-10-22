// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic-only admission before protobuf decoding. Permits stay charged
//! through response-body delivery; execution/log RPCs do not use these limits.

use http_body::{Body, Frame};
use nemo_relay_worker_proto::DIAGNOSTIC_NOT_CONSUMED_METADATA;
use std::convert::Infallible;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tonic::Status;
use tonic::body::Body as TonicBody;
use tonic::codegen::Bytes;
use tonic::server::NamedService;
use tower::Service;

static REQUESTS: OnceLock<Arc<Semaphore>> = OnceLock::new();
static READS: OnceLock<Arc<Semaphore>> = OnceLock::new();
const DEADLINE: Duration = Duration::from_secs(2);

fn capacity_rejection(message: &'static str) -> Status {
    let mut status = Status::resource_exhausted(message);
    status
        .metadata_mut()
        .insert(DIAGNOSTIC_NOT_CONSUMED_METADATA, "1".parse().unwrap());
    status
}

#[derive(Clone)]
pub(super) struct BoundedService<S> {
    inner: S,
    requests: Arc<Semaphore>,
    reads: Arc<Semaphore>,
}
impl<S> BoundedService<S> {
    pub(super) fn new(inner: S) -> Self {
        Self {
            inner,
            requests: REQUESTS
                .get_or_init(|| Arc::new(Semaphore::new(16)))
                .clone(),
            reads: READS.get_or_init(|| Arc::new(Semaphore::new(4))).clone(),
        }
    }

    #[cfg(test)]
    pub(super) fn with_test_limits(inner: S, request_limit: usize, read_limit: usize) -> Self {
        Self {
            inner,
            requests: Arc::new(Semaphore::new(request_limit)),
            reads: Arc::new(Semaphore::new(read_limit)),
        }
    }
}
impl<S: NamedService> NamedService for BoundedService<S> {
    const NAME: &'static str = S::NAME;
}
impl<S> Service<http::Request<TonicBody>> for BoundedService<S>
where
    S: Service<http::Request<TonicBody>, Response = http::Response<TonicBody>, Error = Infallible>
        + Clone
        + Send
        + 'static,
    S::Future: Send + 'static,
{
    type Response = http::Response<TonicBody>;
    type Error = Infallible;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;
    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }
    fn call(&mut self, request: http::Request<TonicBody>) -> Self::Future {
        let permit = self.requests.clone().try_acquire_owned();
        let read = request.uri().path().ends_with("/ReadDiagnosticPage");
        let read_permit = read.then(|| self.reads.clone().try_acquire_owned());
        let service = self.inner.clone();
        Box::pin(async move {
            let (Ok(permit), read_permit) = (permit, read_permit) else {
                return Ok(
                    capacity_rejection("diagnostic transport capacity; not consumed").into_http(),
                );
            };
            let read_permit = match read_permit.transpose() {
                Ok(value) => value,
                Err(_) => {
                    return Ok(
                        capacity_rejection("diagnostic response capacity; not consumed")
                            .into_http(),
                    );
                }
            };
            let response =
                tokio::time::timeout(DEADLINE, tower::ServiceExt::oneshot(service, request)).await;
            let response = match response {
                Ok(Ok(response)) => response,
                Ok(Err(never)) => match never {},
                Err(_) => {
                    return Ok(Status::deadline_exceeded(
                        "diagnostic delivery deadline; acceptance may be unknown",
                    )
                    .into_http());
                }
            };
            Ok(response.map(|body| {
                TonicBody::new(PermittedBody {
                    body,
                    _permit: permit,
                    _read_permit: read_permit,
                    deadline: Box::pin(tokio::time::sleep(DEADLINE)),
                })
            }))
        })
    }
}

struct PermittedBody {
    body: TonicBody,
    _permit: OwnedSemaphorePermit,
    _read_permit: Option<OwnedSemaphorePermit>,
    deadline: Pin<Box<tokio::time::Sleep>>,
}
impl Body for PermittedBody {
    type Data = Bytes;
    type Error = Status;
    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Status>>> {
        if self.deadline.as_mut().poll(cx).is_ready() {
            return Poll::Ready(Some(Err(Status::deadline_exceeded(
                "diagnostic response delivery deadline",
            ))));
        }
        Pin::new(&mut self.body).poll_frame(cx)
    }
    fn is_end_stream(&self) -> bool {
        self.body.is_end_stream()
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/worker_diagnostic_transport_tests.rs"]
mod tests;
