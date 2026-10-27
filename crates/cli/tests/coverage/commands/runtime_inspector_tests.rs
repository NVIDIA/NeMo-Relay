// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use clap::Parser;

#[derive(Parser)]
struct Inspector {
    #[command(flatten)]
    command: InspectCommand,
}

#[test]
fn inspector_distinguishes_missing_workers_from_unrecognized_endpoints() {
    use crate::gateway::client::VerifiedHttpResponse;
    for (worker, body, expected) in [
        (
            true,
            br#"{"error":{"reason":"missing"}}"#.as_slice(),
            "worker not found",
        ),
        (
            false,
            br#"{"error":{"reason":"missing"}}"#,
            "listener diagnostics endpoint unavailable (HTTP 404)",
        ),
        (
            true,
            b"",
            "listener diagnostics endpoint unavailable (HTTP 404)",
        ),
        (
            true,
            b"unknown endpoint",
            "listener diagnostics endpoint unavailable (HTTP 404)",
        ),
        (
            true,
            br#"{"error":{"reason":"future_reason"}}"#,
            "listener diagnostics endpoint unavailable (HTTP 404)",
        ),
        (
            true,
            br#"{"error":{"reason":"disabled"}}"#,
            "listener diagnostics endpoint unavailable (HTTP 404)",
        ),
    ] {
        let mut output = Vec::new();
        let error = write_inspection(
            VerifiedHttpResponse {
                status: 404,
                body: body.to_vec(),
            },
            worker,
            &mut output,
        )
        .unwrap_err();
        assert_eq!(error.to_string(), format!("launcher error: {expected}"));
        assert!(output.is_empty());
    }
    for (status, body, expected) in [
        (
            200,
            br#"{"snapshot":null,"collection":{"freshness":"missing"}}"#.as_slice(),
            ExitCode::SUCCESS,
        ),
        (
            503,
            br#"{"error":{"reason":"shutting_down"}}"#,
            ExitCode::FAILURE,
        ),
    ] {
        let mut output = Vec::new();
        assert_eq!(
            write_inspection(
                VerifiedHttpResponse {
                    status,
                    body: body.to_vec()
                },
                true,
                &mut output
            )
            .unwrap(),
            expected
        );
        assert_eq!(output, [body, b"\n"].concat());
    }
}

#[test]
fn inspector_defaults_and_response_bounds_are_enforced_by_parser() {
    let command = Inspector::try_parse_from(["inspect"]).unwrap().command;
    assert_eq!(command.gateway_url, crate::bootstrap::DEFAULT_URL);
    assert!(command.max_response_bytes.is_none());
    for limit in ["4096", "33554432"] {
        assert!(Inspector::try_parse_from(["inspect", "--max-response-bytes", limit]).is_ok());
    }
    for limit in ["4095", "33554433", "-1"] {
        assert!(Inspector::try_parse_from(["inspect", "--max-response-bytes", limit]).is_err());
    }
}

#[tokio::test]
async fn inspector_rejects_large_queries_before_attempting_transport() {
    let query = "x".repeat(crate::diagnostics_http::MAX_QUERY_BYTES + 1);
    let command = Inspector::try_parse_from(["inspect", "--query", &query])
        .unwrap()
        .command;
    assert!(matches!(inspect(command).await, Err(CliError::Config(_))));
}

#[tokio::test]
async fn inspector_reports_unavailable_for_gateway_and_daemon_worker_transport() {
    let worker = uuid::Uuid::now_v7().to_string();
    for args in [
        vec!["inspect", "--gateway-url", "invalid-url"],
        vec![
            "inspect",
            "--daemon-url",
            "invalid-url",
            "--worker",
            &worker,
        ],
    ] {
        let command = Inspector::try_parse_from(args).unwrap().command;
        let error = inspect(command).await.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("diagnostic inspection unavailable")
        );
    }
}

#[test]
fn worker_inspection_routes_require_daemon_and_uuid_without_header_injection() {
    for (path, daemon) in [
        ("/_nemo_relay/v1/workers/not-a-uuid/diagnostics", true),
        (
            "/_nemo_relay/v1/workers/00000000-0000-0000-0000-000000000000/diagnostics",
            false,
        ),
        ("/_nemo_relay/v1/diagnostics\r\nInjected: true", true),
    ] {
        let error =
            crate::gateway::client::get_diagnostics("invalid-url", path, 8192, daemon).unwrap_err();
        assert!(error.to_string().contains("invalid diagnostic route"));
    }
}

#[test]
fn inspector_enforces_encoded_http_budget_including_parameter_name() {
    let limit = crate::diagnostics_http::MAX_QUERY_BYTES;
    assert_eq!(encode_query(&"x".repeat(limit - 6)).unwrap().len(), limit);
    assert!(encode_query(&"x".repeat(limit - 5)).is_err());
    assert!(encode_query(&" ".repeat(limit / 3)).is_err());
    assert!(encode_query(&"é".repeat(limit / 6)).is_err());
}

#[tokio::test]
async fn inspector_continues_http_pages_beyond_the_old_query_limit() {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use http_body_util::BodyExt;
    use nemo_relay::diagnostics::*;
    use std::num::NonZeroU64;
    use tower::ServiceExt;
    let d = Diagnostics::from_settings(
        Diagnostics::default().source().clone(),
        DiagnosticsSettings {
            enabled: true,
            ..Default::default()
        },
    )
    .unwrap();
    let key = "k".repeat(128);
    let ownership = ComponentOwnership {
        subscriber_instance_id: Some(uuid::Uuid::nil()),
        plugin_instance_id: Some(uuid::Uuid::nil()),
        configuration_id: Some("c".repeat(128)),
        responsibility: Some("r".repeat(128)),
    };
    let recorder = d
        .register_component(ComponentRegistration {
            ownership: ownership.clone(),
            ..ComponentRegistration::new(&key, &[])
        })
        .unwrap();
    let name = "e".repeat(123);
    let code = format!("acme.{name}");
    let extension = recorder
        .register_extension(ExtensionRegistration {
            namespace: "acme",
            name: &name,
            description: "probe",
            allowed_classes: &[],
            outcome: DiagnosticOutcome::Unknown,
        })
        .unwrap()
        .expect("extension admitted");
    for _ in 0..2 {
        recorder
            .record_lazy(|| {
                Observation::extension(
                    extension,
                    None,
                    DiagnosticOutcome::Unknown,
                    "probe",
                    NonZeroU64::MIN,
                )
            })
            .unwrap();
    }
    let observed_generation = d
        .snapshot(&DiagnosticQuery::default())
        .unwrap()
        .to_snapshot()
        .unwrap()
        .events[0]
        .origin
        .refresh_generation;
    let mut q = DiagnosticQuery {
        schema_version: Some(1),
        component_key: Some(key),
        component_instance_id: Some(recorder.identity().unwrap().instance_id),
        origin_generation: Some(observed_generation),
        code: Some(code),
        ownership,
        after_sequence: Some(0.into()),
        max_components: Some(1),
        max_events: Some(1),
        max_response_bytes: Some(131072),
        ..Default::default()
    };
    fn compact(q: &DiagnosticQuery) -> String {
        let mut value = serde_json::to_value(q).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .retain(|_, value| !value.is_null());
        serde_json::to_string(&value).unwrap()
    }

    let key = crate::configuration::BootstrapChallengeKey::from_bytes(b"inspector-pagination");
    let token = key.diagnostics_token();
    let router = crate::diagnostics_http::DiagnosticsHttp::new(d.reader(), key, vec![])
        .unwrap()
        .router();
    let mut pages = Vec::new();
    for page_index in 0..2 {
        let wire = compact(&q);
        if page_index == 0 {
            assert!(wire.len() <= 1024);
        } else {
            assert!(wire.len() > 1024);
        }
        let path = format!(
            "{}?{}",
            crate::diagnostics_http::PREFIX,
            encode_query(&wire).unwrap()
        );
        let response = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri(path)
                    .header(crate::diagnostics_http::AUTH_HEADER, &token)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let page: DiagnosticSnapshot =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(page.events.len(), 1);
        if page_index == 0 {
            q.continuation = Some(page.continuation.clone().expect("next page"));
        }
        pages.push(page);
    }
    assert_ne!(pages[0].events[0].sequence, pages[1].events[0].sequence);
}
