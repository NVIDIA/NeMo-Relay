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
    let token = bootstrap(&adapter).diagnostics_token();
    let provider = bootstrap(&adapter).client_token();
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
        let runtime = crate::runtime_diagnostics::CliDiagnostics::new(context.clone(), true);
        let adapter = DiagnosticsHttp::new(
            context.reader(),
            BootstrapChallengeKey::from_bytes(b"diagnostic-test-key"),
            runtime.coverage(),
        )
        .unwrap();
        let token = bootstrap(&adapter).diagnostics_token();
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
        // Supported families remain discoverable before plugin registration and
        // without allocating a snapshot workspace, even with collection disabled.
        let producers = discovery["producers"].as_array().unwrap();
        for family in ["cli.command", "plugin.native_abi", "plugin.worker_protocol"] {
            assert!(producers.iter().any(|value| value == family));
        }
        assert_eq!(
            discovery["producer_support"],
            serde_json::to_value(context.reader().discovery().producer_support).unwrap()
        );
        assert!(discovery.get("deferred").is_none());
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
    let token = bootstrap(&adapter).diagnostics_token();
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
    let token = bootstrap(&adapter).diagnostics_token();
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
    let token = bootstrap(&adapter).diagnostics_token();
    let router = adapter.router();
    for (query, expected, reason) in [
        (
            r#"{"schema_version":99}"#,
            StatusCode::NOT_ACCEPTABLE,
            "unsupported_schema",
        ),
        (
            r#"{"unknown":"PRIVATE"}"#,
            StatusCode::BAD_REQUEST,
            "invalid_query",
        ),
        (
            r#"{"max_events":99999}"#,
            StatusCode::BAD_REQUEST,
            "invalid_query",
        ),
        (
            r#"{"ownership":{"configuration_typo":"PRIVATE"}}"#,
            StatusCode::BAD_REQUEST,
            "invalid_query",
        ),
        (
            r#"{"ownership":{"configuration_id":"selected","configuration_typo":"PRIVATE"}}"#,
            StatusCode::BAD_REQUEST,
            "invalid_query",
        ),
        (
            r#"{"continuation":{"source_instance_id":"00000000-0000-0000-0000-000000000001","sequence":"0","component_offset":"0","event_offset":"0","scope":"0","scope_typo":"PRIVATE"}}"#,
            StatusCode::BAD_REQUEST,
            "invalid_query",
        ),
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
        let error = json(response).await;
        assert_eq!(error["error"]["reason"], reason);
        assert!(!error.to_string().contains("PRIVATE"));
    }
}

#[tokio::test]
async fn ownership_configuration_filter_preserves_matching_and_default_selection() {
    let context = context(true, 1);
    let codes = [BuiltinDiagnosticCode::Failure(FailureCode::Cli(
        CliFailureCode::Command,
    ))];
    let mut registration = ComponentRegistration::new("cli.command", &codes);
    registration.ownership.configuration_id = Some("selected".into());
    let _component = context.register_component(registration).unwrap();
    let adapter = adapter(&context);
    let token = bootstrap(&adapter).diagnostics_token();
    let router = adapter.router();
    for (query, expected) in [
        (r#"{"ownership":{"configuration_id":"selected"}}"#, 1),
        (r#"{"ownership":{"configuration_id":"other"}}"#, 0),
        ("{}", 1),
    ] {
        let path = format!(
            "{PREFIX}/components?query={}",
            percent_encoding::utf8_percent_encode(query, percent_encoding::NON_ALPHANUMERIC)
        );
        let response = router
            .clone()
            .oneshot(request(&path, Some(&token)))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{query}");
        assert_eq!(
            json(response).await["components"].as_array().unwrap().len(),
            expected,
            "{query}"
        );
    }
}

fn bootstrap(adapter: &DiagnosticsHttp) -> &BootstrapChallengeKey {
    match &adapter.authority {
        ReadAuthority::Bootstrap(key) => key,
        _ => panic!("expected bootstrap authority"),
    }
}

#[tokio::test]
async fn shared_query_fixture_and_core_read_errors_have_canonical_http_reasons() {
    let cases: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../types/tests/fixtures/diagnostics/read-queries.json"
    ))
    .unwrap();
    let context = context(true, 1);
    let adapter = adapter(&context);
    let token = bootstrap(&adapter).diagnostics_token();
    let router = adapter.router();
    for case in cases.as_array().unwrap() {
        let input = case["query"].to_string();
        let path = format!(
            "{PREFIX}?query={}",
            percent_encoding::utf8_percent_encode(&input, percent_encoding::NON_ALPHANUMERIC)
        );
        let response = router
            .clone()
            .oneshot(request(&path, Some(&token)))
            .await
            .unwrap();
        if let Some(reason) = case["reason"].as_str() {
            let body = json(response).await;
            assert_eq!(body["error"]["reason"], reason);
            assert!(!body.to_string().contains("restricted-input"));
            if reason == "unsupported_schema" {
                assert_eq!(
                    body["error"]["supported_schema_version"],
                    DIAGNOSTIC_SCHEMA_VERSION
                );
            }
        } else {
            assert_eq!(response.status(), StatusCode::OK);
        }
    }
    for (core, status, reason) in [
        (
            DiagnosticError::Disabled,
            StatusCode::SERVICE_UNAVAILABLE,
            "disabled",
        ),
        (
            DiagnosticError::QueryDisabled,
            StatusCode::SERVICE_UNAVAILABLE,
            "query_disabled",
        ),
        (
            DiagnosticError::QueryCapacity,
            StatusCode::TOO_MANY_REQUESTS,
            "busy",
        ),
        (
            DiagnosticError::SnapshotCapacity,
            StatusCode::UNPROCESSABLE_ENTITY,
            "snapshot_capacity",
        ),
        (
            DiagnosticError::InvalidCursor,
            StatusCode::CONFLICT,
            "invalid_cursor",
        ),
        (
            DiagnosticError::InvalidQuery,
            StatusCode::BAD_REQUEST,
            "invalid_query",
        ),
    ] {
        let response = reader_error(core);
        assert_eq!(response.status(), status);
        if reason == "busy" {
            assert_eq!(response.headers()[header::RETRY_AFTER], "1");
        }
        assert_eq!(json(response).await["error"]["reason"], reason);
    }
}

#[tokio::test]
async fn http_read_is_the_same_source_evidence_without_recounting() {
    let context = context(true, 1);
    let failure = BuiltinDiagnosticCode::Failure(FailureCode::Cli(CliFailureCode::Command));
    let component = context
        .register_component(ComponentRegistration::new("example", &[failure]))
        .unwrap();
    for count in [2, 3] {
        component
            .record_without_projection_lazy(|| {
                Observation::new(
                    Classification::failure(
                        FailureCode::Cli(CliFailureCode::Command),
                        FailureClass::Unknown,
                    )
                    .unwrap(),
                    "restricted-fact",
                    NonZeroU64::new(count).unwrap(),
                )
            })
            .unwrap();
    }
    let direct = context
        .snapshot(&DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap();
    let adapter = adapter(&context);
    let token = bootstrap(&adapter).diagnostics_token();
    let page: DiagnosticSnapshot = serde_json::from_value(
        json(
            adapter
                .router()
                .oneshot(request(PREFIX, Some(&token)))
                .await
                .unwrap(),
        )
        .await,
    )
    .unwrap();
    assert_eq!(page.source, direct.source);
    assert_eq!(page.sequence, direct.sequence);
    assert_eq!(page.events, direct.events);
    assert_eq!(page.lifetime.totals, direct.lifetime.totals);
    assert_eq!(page.lifetime.totals.failures_total.value.get(), 5);
    assert_eq!(page.lifetime.totals.recorded_events_total.value.get(), 2);
}
