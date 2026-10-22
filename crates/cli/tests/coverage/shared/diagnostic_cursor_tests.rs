// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use axum::http::Request;
use http_body_util::BodyExt;
use tower::ServiceExt;
#[tokio::test]
async fn continuations_reauthenticate_and_reject_changed_scope_or_revision() {
    let context = Diagnostics::from_settings(
        Diagnostics::default().source().clone(),
        DiagnosticsSettings {
            enabled: true,
            ..Default::default()
        },
    )
    .unwrap();
    let code = BuiltinDiagnosticCode::Success(SuccessCode::CliCommandCompleted);
    let recorder = context
        .register_component(ComponentRegistration::new("cli.command", &[code]))
        .unwrap();
    for _ in 0..3 {
        recorder
            .record_lazy(|| {
                Observation::new(
                    Classification::success(SuccessCode::CliCommandCompleted),
                    "observed",
                    std::num::NonZeroU64::MIN,
                )
            })
            .unwrap();
    }
    let key = BootstrapChallengeKey::from_bytes(b"cursor-key");
    let token = key.diagnostics_token();
    let router = DiagnosticsHttp::new(context.reader(), key, vec![])
        .unwrap()
        .router();
    let make = |query: &serde_json::Value, auth: bool| {
        let path = format!(
            "{PREFIX}?query={}",
            percent_encoding::utf8_percent_encode(
                &query.to_string(),
                percent_encoding::NON_ALPHANUMERIC
            )
        );
        let mut request = Request::builder().uri(path);
        if auth {
            request = request.header(AUTH_HEADER, &token);
        }
        request.body(Body::empty()).unwrap()
    };
    let response = router
        .clone()
        .oneshot(make(&serde_json::json!({"max_events":1}), true))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let page: DiagnosticSnapshot =
        serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap();
    let cursor = page.continuation.expect("page continuation");
    let query = serde_json::json!({"max_events":1, "continuation":cursor});
    assert_eq!(
        router
            .clone()
            .oneshot(make(&query, false))
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );
    let changed =
        serde_json::json!({"max_events":1, "component_key":"cli.command", "continuation":cursor});
    assert_eq!(
        router
            .clone()
            .oneshot(make(&changed, true))
            .await
            .unwrap()
            .status(),
        StatusCode::CONFLICT
    );
    assert_eq!(
        router
            .clone()
            .oneshot(make(&query, true))
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );
    recorder
        .record_lazy(|| {
            Observation::new(
                Classification::success(SuccessCode::CliCommandCompleted),
                "later",
                std::num::NonZeroU64::MIN,
            )
        })
        .unwrap();
    assert_eq!(
        router.oneshot(make(&query, true)).await.unwrap().status(),
        StatusCode::CONFLICT
    );
}
