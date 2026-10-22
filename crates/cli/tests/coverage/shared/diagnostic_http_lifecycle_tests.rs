// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use axum::http::Request;
use tower::ServiceExt;
#[tokio::test]
async fn plain_http_cannot_return_detail_and_shutdown_stops_new_reads() {
    let context = Diagnostics::default();
    let key = BootstrapChallengeKey::from_bytes(b"lifecycle-key");
    let token = key.diagnostics_token();
    let adapter = DiagnosticsHttp::new(context.reader(), key, vec![]).unwrap();
    let request = || {
        Request::builder()
            .uri(PREFIX)
            .header(AUTH_HEADER, &token)
            .body(Body::empty())
            .unwrap()
    };
    assert_eq!(
        adapter
            .clone()
            .unsecured_router()
            .oneshot(request())
            .await
            .unwrap()
            .status(),
        StatusCode::UPGRADE_REQUIRED
    );
    adapter.close();
    assert_eq!(
        adapter.router().oneshot(request()).await.unwrap().status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
}

#[tokio::test]
async fn inspection_connections_are_bounded_and_shutdown_notifies_active_readers() {
    let context = Diagnostics::from_settings(
        Diagnostics::default().source().clone(),
        DiagnosticsSettings {
            enabled: true,
            limits: DiagnosticLimits {
                max_queries: 0,
                ..Default::default()
            },
        },
    )
    .unwrap();
    let adapter = DiagnosticsHttp::new(
        context.reader(),
        BootstrapChallengeKey::from_bytes(b"lifecycle"),
        vec![],
    )
    .unwrap();
    let permit = adapter.try_connection().unwrap();
    assert!(adapter.try_connection().is_err());
    drop(permit);
    let _permit = adapter.try_connection().unwrap();
    let waiting = adapter.clone();
    let task = tokio::spawn(async move { waiting.wait_closed().await });
    tokio::task::yield_now().await;
    adapter.close();
    tokio::time::timeout(std::time::Duration::from_secs(1), task)
        .await
        .unwrap()
        .unwrap();
    adapter.wait_closed().await;
    assert!(adapter.try_connection().is_err());
}
