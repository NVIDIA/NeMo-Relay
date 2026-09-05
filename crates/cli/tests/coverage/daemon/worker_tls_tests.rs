// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::convert::Infallible;
use std::sync::Arc;

use bytes::Bytes;
use http::{Request, Response, StatusCode};
use http_body_util::{BodyExt as _, Empty, Full};
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder;
use tokio::net::TcpListener;
use tokio_rustls::TlsAcceptor;

use super::{WorkerClientPool, WorkerTlsIdentity, pooled_worker_tls_client};
use crate::daemon::common::transport::box_body;

#[test]
fn generates_a_daemon_pinnable_worker_identity() {
    let identity = WorkerTlsIdentity::generate("127.0.0.1").expect("generate worker identity");
    assert!(!identity.root_certificate().is_empty());
    pooled_worker_tls_client(identity.root_certificate()).expect("build pinned worker client");
    WorkerTlsIdentity::generate("worker.example.com").expect("hostname SAN identity");
}

#[test]
fn rejects_invalid_or_non_concrete_worker_roots_and_hosts() {
    assert!(WorkerTlsIdentity::generate("").is_err());
    assert!(WorkerTlsIdentity::generate("0.0.0.0").is_err());
    assert!(pooled_worker_tls_client("").is_err());
    assert!(pooled_worker_tls_client("not_base64!").is_err());
}

#[test]
fn process_wide_worker_pool_reuses_cleartext_and_matching_tls_trust() {
    let pool = WorkerClientPool::with_tls_capacity(2).expect("worker client pool");
    let first_cleartext = pool.client(None).expect("cleartext client");
    let second_cleartext = pool.client(None).expect("reused cleartext client");
    assert!(Arc::ptr_eq(&first_cleartext, &second_cleartext));

    let first_identity = WorkerTlsIdentity::generate("127.0.0.1").expect("first identity");
    let second_identity = WorkerTlsIdentity::generate("127.0.0.1").expect("second identity");
    let first_tls = pool
        .client(Some(first_identity.root_certificate()))
        .expect("first TLS client");
    let first_tls_reused = pool
        .client(Some(first_identity.root_certificate()))
        .expect("reused first TLS client");
    let second_tls = pool
        .client(Some(second_identity.root_certificate()))
        .expect("isolated second TLS client");

    assert!(Arc::ptr_eq(&first_tls, &first_tls_reused));
    assert!(!Arc::ptr_eq(&first_tls, &second_tls));
}

#[test]
fn worker_tls_pool_cache_is_bounded_and_evicts_least_recently_used_root() {
    let pool = WorkerClientPool::with_tls_capacity(2).expect("worker client pool");
    let first_identity = WorkerTlsIdentity::generate("127.0.0.1").expect("first identity");
    let second_identity = WorkerTlsIdentity::generate("127.0.0.1").expect("second identity");
    let third_identity = WorkerTlsIdentity::generate("127.0.0.1").expect("third identity");

    let first = pool
        .client(Some(first_identity.root_certificate()))
        .expect("first TLS client");
    let second = pool
        .client(Some(second_identity.root_certificate()))
        .expect("second TLS client");
    let first_reused = pool
        .client(Some(first_identity.root_certificate()))
        .expect("refresh first TLS client recency");
    assert!(Arc::ptr_eq(&first, &first_reused));

    pool.client(Some(third_identity.root_certificate()))
        .expect("third TLS client");
    assert_eq!(super::lock(&pool.tls).entries.len(), 2);

    let second_after_eviction = pool
        .client(Some(second_identity.root_certificate()))
        .expect("recreated second TLS client");
    assert!(!Arc::ptr_eq(&second, &second_after_eviction));
    assert_eq!(super::lock(&pool.tls).entries.len(), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_matching_tls_lookups_share_one_pool() {
    const LOOKUPS: usize = 32;

    let pool = Arc::new(WorkerClientPool::with_tls_capacity(2).expect("worker client pool"));
    let identity = WorkerTlsIdentity::generate("127.0.0.1").expect("worker identity");
    let root = Arc::new(identity.root_certificate().to_owned());
    let barrier = Arc::new(tokio::sync::Barrier::new(LOOKUPS));
    let mut lookups = Vec::with_capacity(LOOKUPS);

    for _ in 0..LOOKUPS {
        let pool = Arc::clone(&pool);
        let root = Arc::clone(&root);
        let barrier = Arc::clone(&barrier);
        lookups.push(tokio::spawn(async move {
            barrier.wait().await;
            pool.client(Some(root.as_str())).expect("TLS client")
        }));
    }

    let first = lookups.remove(0).await.expect("first lookup task");
    for lookup in lookups {
        let client = lookup.await.expect("lookup task");
        assert!(Arc::ptr_eq(&first, &client));
    }
    assert_eq!(super::lock(&pool.tls).entries.len(), 1);
}

#[tokio::test]
async fn daemon_pinned_client_completes_a_real_tls_worker_round_trip() {
    let identity = WorkerTlsIdentity::generate("127.0.0.1").expect("generate worker identity");
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind TLS worker");
    let address = listener.local_addr().expect("worker address");
    let acceptor = TlsAcceptor::from(identity.server_config());
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("accept daemon");
        let stream = acceptor.accept(stream).await.expect("authenticate TLS");
        Builder::new(TokioExecutor::new())
            .serve_connection(
                TokioIo::new(stream),
                service_fn(|_request: Request<Incoming>| async {
                    let mut response = Response::new(Full::new(Bytes::from_static(b"ready")));
                    *response.status_mut() = StatusCode::CREATED;
                    Ok::<_, Infallible>(response)
                }),
            )
            .await
            .expect("serve pinned daemon request");
    });

    let pool = WorkerClientPool::with_tls_capacity(2).expect("worker client pool");
    let client = pool
        .client(Some(identity.root_certificate()))
        .expect("pinned client");
    let request = Request::get(format!("https://127.0.0.1:{}/probe", address.port()))
        .body(box_body(Empty::<Bytes>::new()))
        .expect("request");
    let response = client.request(request).await.expect("TLS worker response");
    assert_eq!(response.status(), StatusCode::CREATED);
    assert_eq!(
        response
            .into_body()
            .collect()
            .await
            .expect("response body")
            .to_bytes(),
        "ready"
    );
    server.abort();
    assert!(
        server
            .await
            .expect_err("server is stopped after the round trip")
            .is_cancelled()
    );
}

#[tokio::test]
async fn tls_pool_never_reuses_trust_across_different_roots() {
    let serving_identity =
        WorkerTlsIdentity::generate("127.0.0.1").expect("serving worker identity");
    let unrelated_identity =
        WorkerTlsIdentity::generate("127.0.0.1").expect("unrelated worker identity");
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind TLS worker");
    let address = listener.local_addr().expect("worker address");
    let acceptor = TlsAcceptor::from(serving_identity.server_config());
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.expect("accept daemon");
        // The client rejects this server before an HTTP request can be delivered.
        let _ = acceptor.accept(stream).await;
    });

    let pool = WorkerClientPool::with_tls_capacity(2).expect("worker client pool");
    let unrelated_client = pool
        .client(Some(unrelated_identity.root_certificate()))
        .expect("unrelated pinned client");
    let request = Request::get(format!("https://127.0.0.1:{}/probe", address.port()))
        .body(box_body(Empty::<Bytes>::new()))
        .expect("request");
    assert!(unrelated_client.request(request).await.is_err());

    server.abort();
    let _ = server.await;
}
