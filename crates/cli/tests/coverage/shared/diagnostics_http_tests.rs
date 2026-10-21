// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use axum::http::Request;
use http_body_util::BodyExt;
use std::num::NonZeroU64;
use tower::ServiceExt;

fn context(enabled: bool, queries: usize) -> Diagnostics {
    Diagnostics::from_settings(
        Diagnostics::default().source().clone(),
        DiagnosticsSettings {
            enabled,
            limits: DiagnosticLimits {
                max_queries: queries,
                ..Default::default()
            },
        },
    )
    .unwrap()
}
fn adapter(context: &Diagnostics) -> DiagnosticsHttp {
    DiagnosticsHttp::new(
        context.reader(),
        BootstrapChallengeKey::from_bytes(b"diagnostic-test-key"),
        vec![],
    )
    .unwrap()
}
fn request(path: &str, token: Option<&str>) -> Request<Body> {
    let mut request = Request::builder().uri(path);
    if let Some(token) = token {
        request = request.header(AUTH_HEADER, token);
    }
    request.body(Body::empty()).unwrap()
}
async fn json(response: Response) -> serde_json::Value {
    serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap()
}
#[tokio::test]
async fn authorization_precedes_parsing_and_private_reads() {
    let context = context(true, 1);
    let adapter = adapter(&context);
    let token = adapter.key.diagnostics_token();
    let provider = adapter.key.client_token();
    let router = adapter.router();
    assert_eq!(
        router
            .clone()
            .oneshot(request(&format!("{PREFIX}?invalid"), None))
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        router
            .clone()
            .oneshot(request(PREFIX, Some(&provider)))
            .await
            .unwrap()
            .status(),
        StatusCode::FORBIDDEN
    );
    let mut browser = request(PREFIX, Some(&token));
    browser
        .headers_mut()
        .insert(header::ORIGIN, "https://example.com".parse().unwrap());
    assert_eq!(
        router.clone().oneshot(browser).await.unwrap().status(),
        StatusCode::FORBIDDEN
    );
    let response = router.oneshot(request(PREFIX, Some(&token))).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
}
#[tokio::test]
async fn discovery_distinguishes_disabled_collection_and_disabled_queries() {
    for enabled in [true, false] {
        let context = context(enabled, 0);
        let adapter = adapter(&context);
        let token = adapter.key.diagnostics_token();
        let router = adapter.router();
        let discovery = json(
            router
                .clone()
                .oneshot(request(&format!("{PREFIX}/capabilities"), Some(&token)))
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(discovery["collection_enabled"], enabled);
        assert_eq!(discovery["queries_enabled"], false);
        let response = router.oneshot(request(PREFIX, Some(&token))).await.unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            json(response).await["error"]["reason"],
            if enabled {
                "query_disabled"
            } else {
                "disabled"
            }
        );
    }
}
#[tokio::test]
async fn body_owns_response_budget_but_releases_core_lease_before_delivery() {
    let context = context(true, 1);
    let adapter = adapter(&context);
    let token = adapter.key.diagnostics_token();
    let pool = adapter.pool.clone();
    let router = adapter.router();
    let response = router
        .clone()
        .oneshot(request(PREFIX, Some(&token)))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(pool.lock().unwrap().is_empty());
    assert!(context.snapshot(&DiagnosticQuery::default()).is_ok());
    assert_eq!(
        router
            .clone()
            .oneshot(request(PREFIX, Some(&token)))
            .await
            .unwrap()
            .status(),
        StatusCode::TOO_MANY_REQUESTS
    );
    // Cancellation returns the same backing allocation to the transport pool.
    drop(response);
    assert_eq!(pool.lock().unwrap().len(), 1);
    assert_eq!(
        router
            .oneshot(request(PREFIX, Some(&token)))
            .await
            .unwrap()
            .status(),
        StatusCode::OK
    );
}
#[tokio::test]
async fn authorized_snapshot_preserves_restricted_detail_without_projection_disclosure() {
    let context = context(true, 1);
    let code = BuiltinDiagnosticCode::Failure(FailureCode::Cli(CliFailureCode::Command));
    let component = context
        .register_component(ComponentRegistration::new("cli.command", &[code]))
        .unwrap();
    let projected = Arc::new(Mutex::new(Vec::new()));
    let output = projected.clone();
    let _projection = context
        .subscribe_projection(move |summary| {
            output
                .lock()
                .unwrap()
                .push(serde_json::to_string(summary).unwrap())
        })
        .unwrap();
    component
        .record_lazy(|| {
            Observation::new(
                Classification::failure(
                    FailureCode::Cli(CliFailureCode::Command),
                    FailureClass::Unknown,
                )
                .unwrap(),
                "RESTRICTED_MARKER",
                NonZeroU64::MIN,
            )
            .with_correlation(DiagnosticCorrelation {
                session_id: Some("RESTRICTED_SESSION".into()),
                request_id: None,
            })
        })
        .unwrap();
    let adapter = adapter(&context);
    let token = adapter.key.diagnostics_token();
    let response = adapter
        .router()
        .oneshot(request(PREFIX, Some(&token)))
        .await
        .unwrap();
    let page = json(response).await;
    assert_eq!(page["events"][0]["detail"]["message"], "RESTRICTED_MARKER");
    let output = projected.lock().unwrap();
    assert_eq!(output.len(), 1);
    assert!(!output[0].contains("RESTRICTED"));
    for field in ["detail", "correlation", "origin", "context", "component"] {
        assert!(
            serde_json::from_str::<serde_json::Value>(&output[0])
                .unwrap()
                .get(field)
                .is_none()
        );
    }
}
#[tokio::test]
async fn bounded_query_and_version_errors_are_safe() {
    let context = context(true, 1);
    let adapter = adapter(&context);
    let token = adapter.key.diagnostics_token();
    let router = adapter.router();
    for (query, expected) in [
        (r#"{"schema_version":99}"#, StatusCode::NOT_ACCEPTABLE),
        (r#"{"unknown":"PRIVATE"}"#, StatusCode::BAD_REQUEST),
        (r#"{"max_events":99999}"#, StatusCode::BAD_REQUEST),
    ] {
        let path = format!(
            "{PREFIX}?query={}",
            percent_encoding::utf8_percent_encode(query, percent_encoding::NON_ALPHANUMERIC)
        );
        let response = router
            .clone()
            .oneshot(request(&path, Some(&token)))
            .await
            .unwrap();
        assert_eq!(response.status(), expected);
        assert!(!json(response).await.to_string().contains("PRIVATE"));
    }
}
