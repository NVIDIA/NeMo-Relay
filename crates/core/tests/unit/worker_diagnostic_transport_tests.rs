// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Default)]
struct Decoder(Arc<AtomicUsize>);
struct PendingBody;
impl Body for PendingBody {
    type Data = Bytes;
    type Error = Status;
    fn poll_frame(
        self: Pin<&mut Self>,
        _: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Status>>> {
        Poll::Pending
    }
}
impl NamedService for Decoder {
    const NAME: &'static str = "test.Decoder";
}
impl Service<http::Request<TonicBody>> for Decoder {
    type Response = http::Response<TonicBody>;
    type Error = Infallible;
    type Future = std::future::Ready<Result<Self::Response, Self::Error>>;
    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }
    fn call(&mut self, _: http::Request<TonicBody>) -> Self::Future {
        self.0.fetch_add(1, Ordering::Relaxed);
        std::future::ready(Ok(http::Response::new(TonicBody::new(PendingBody))))
    }
}

#[tokio::test]
async fn request_permits_are_reserved_before_decoder_and_retained_by_response() {
    let decoder = Decoder::default();
    let mut service = BoundedService::with_test_limits(decoder.clone(), 16, 4);
    let requests: Vec<_> = (0..16)
        .map(|_| service.call(http::Request::new(TonicBody::empty())))
        .collect();
    assert_eq!(decoder.0.load(Ordering::Relaxed), 0);
    let rejected = service
        .call(http::Request::new(TonicBody::empty()))
        .await
        .unwrap();
    assert_eq!(rejected.headers()["grpc-status"], "8");
    assert_eq!(rejected.headers()[DIAGNOSTIC_NOT_CONSUMED_METADATA], "1");
    assert_eq!(decoder.0.load(Ordering::Relaxed), 0);
    drop(requests);
    let response = service
        .call(http::Request::new(TonicBody::empty()))
        .await
        .unwrap();
    assert_eq!(decoder.0.load(Ordering::Relaxed), 1);
    assert_eq!(service.requests.available_permits(), 15);
    drop(response);
    assert_eq!(service.requests.available_permits(), 16);
}

fn read_request() -> http::Request<TonicBody> {
    http::Request::builder()
        .uri("/nemo.relay.worker.v1.RelayDiagnostics/ReadDiagnosticPage")
        .body(TonicBody::empty())
        .unwrap()
}

#[tokio::test]
async fn read_quota_is_independent_and_released_on_cancellation_and_body_drop() {
    let decoder = Decoder::default();
    let mut service = BoundedService::with_test_limits(decoder.clone(), 16, 4);
    let reads: Vec<_> = (0..4).map(|_| service.call(read_request())).collect();
    let rejected = service.call(read_request()).await.unwrap();
    assert_eq!(rejected.headers()["grpc-status"], "8");
    assert_eq!(rejected.headers()[DIAGNOSTIC_NOT_CONSUMED_METADATA], "1");
    assert_eq!(decoder.0.load(Ordering::Relaxed), 0);
    // A full read budget must not prevent a submission.
    let submission = service
        .call(http::Request::new(TonicBody::empty()))
        .await
        .unwrap();
    assert_eq!(decoder.0.load(Ordering::Relaxed), 1);
    drop(reads);
    assert_eq!(service.reads.available_permits(), 4);
    assert_eq!(service.requests.available_permits(), 15);
    let response = service.call(read_request()).await.unwrap();
    assert_eq!(service.reads.available_permits(), 3);
    drop(response);
    drop(submission);
    assert_eq!(service.reads.available_permits(), 4);
    assert_eq!(service.requests.available_permits(), 16);
}

#[derive(Clone)]
struct BlockedHandler;
impl Service<http::Request<TonicBody>> for BlockedHandler {
    type Response = http::Response<TonicBody>;
    type Error = Infallible;
    type Future = std::future::Pending<Result<Self::Response, Self::Error>>;
    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }
    fn call(&mut self, _: http::Request<TonicBody>) -> Self::Future {
        std::future::pending()
    }
}

#[tokio::test(start_paused = true)]
async fn handler_deadline_releases_permits_without_claiming_nonacceptance() {
    let mut service = BoundedService::with_test_limits(BlockedHandler, 1, 1);
    let response = service.call(read_request()).await.unwrap();
    assert_eq!(response.headers()["grpc-status"], "4");
    assert!(
        !response
            .headers()
            .contains_key(DIAGNOSTIC_NOT_CONSUMED_METADATA)
    );
    assert_eq!(service.requests.available_permits(), 1);
    assert_eq!(service.reads.available_permits(), 1);
}

#[tokio::test(start_paused = true)]
async fn response_body_deadline_releases_permits_when_delivery_is_dropped() {
    let mut service = BoundedService::with_test_limits(Decoder::default(), 1, 1);
    let mut response = service.call(read_request()).await.unwrap();
    let result = std::future::poll_fn(|cx| Pin::new(response.body_mut()).poll_frame(cx)).await;
    let error = result.unwrap().unwrap_err();
    assert_eq!(error.code(), tonic::Code::DeadlineExceeded);
    assert!(
        !error
            .metadata()
            .contains_key(DIAGNOSTIC_NOT_CONSUMED_METADATA)
    );
    assert_eq!(service.requests.available_permits(), 0);
    drop(response);
    assert_eq!(service.requests.available_permits(), 1);
    assert_eq!(service.reads.available_permits(), 1);
}
