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
