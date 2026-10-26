// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Golden wire contracts for diagnostic HTTP producers and consumers.
use nemo_relay_types::diagnostics::*;
use serde_json::json;

#[test]
fn diagnostic_http_errors_preserve_stable_reasons_statuses_and_schema_metadata() {
    use DiagnosticHttpReason::*;
    for (reason, wire, status) in [
        (Disabled, "disabled", 503),
        (QueryDisabled, "query_disabled", 503),
        (UnsupportedSchema, "unsupported_schema", 406),
        (InvalidQuery, "invalid_query", 400),
        (InvalidCursor, "invalid_cursor", 409),
        (Busy, "busy", 429),
        (SnapshotCapacity, "snapshot_capacity", 422),
        (InvalidLimits, "invalid_limits", 500),
        (AllocationFailed, "allocation_failed", 500),
        (Internal, "internal", 500),
        (
            AuthenticatedTransportRequired,
            "authenticated_transport_required",
            426,
        ),
        (Unauthorized, "unauthorized", 401),
        (Forbidden, "forbidden", 403),
        (ShuttingDown, "shutting_down", 503),
        (HttpReadDisabled, "http_read_disabled", 503),
        (CollectionDisabled, "collection_disabled", 503),
        (Missing, "missing", 404),
        (Unsupported, "unsupported", 501),
        (ReadAuthorityUnavailable, "read_authority_unavailable", 503),
    ] {
        let error = DiagnosticHttpError::new(reason);
        let mut expected = json!({"error": {"reason": wire}});
        if reason == UnsupportedSchema {
            expected["error"]["supported_schema_version"] = json!(DIAGNOSTIC_SCHEMA_VERSION);
        }
        assert_eq!(reason.http_status(), status, "{wire}");
        assert_eq!(serde_json::to_value(&error).unwrap(), expected);
        assert_eq!(
            DiagnosticHttpError::from_response(status, &serde_json::to_vec(&expected).unwrap()),
            Some(error)
        );
    }
    for reason in [
        DiagnosticReadReason::Disabled,
        DiagnosticReadReason::QueryDisabled,
        DiagnosticReadReason::UnsupportedSchema,
        DiagnosticReadReason::InvalidQuery,
        DiagnosticReadReason::InvalidCursor,
        DiagnosticReadReason::Busy,
        DiagnosticReadReason::SnapshotCapacity,
        DiagnosticReadReason::InvalidLimits,
        DiagnosticReadReason::AllocationFailed,
        DiagnosticReadReason::Internal,
    ] {
        assert_eq!(
            serde_json::to_value(DiagnosticHttpReason::from(reason)).unwrap(),
            json!(reason.as_str())
        );
    }
}

#[test]
fn diagnostic_http_consumers_reject_ambiguous_responses_and_accept_additive_fields() {
    for (status, body) in [
        (404, b"".as_slice()),
        (503, b"not JSON"),
        (503, br#"{"error":{"reason":"future_reason"}}"#),
        (503, br#"{"error":{"reason":{"disabled":null}}}"#),
        (503, br#"{"error":["disabled"]}"#),
        (503, br#"[["disabled"]]"#),
        (503, br#"[{"reason":"disabled"}]"#),
        (503, br#"{"error":{"reason":null}}"#),
        (503, br#"{"error":{"reason":503}}"#),
        (503, br#"{"error":{"reason":"disabled"}} {}"#),
        (
            503,
            br#"{"error":{"reason":"disabled"},"error":{"reason":"shutting_down"}}"#,
        ),
        (
            503,
            br#"{"error":{"reason":"disabled","reason":"shutting_down"}}"#,
        ),
        (404, br#"{"error":{"reason":"disabled"}}"#),
        (200, br#"{"error":{"reason":"missing"}}"#),
    ] {
        assert!(
            DiagnosticHttpError::from_response(status, body).is_none(),
            "accepted {}",
            String::from_utf8_lossy(body)
        );
    }
    let mut body = br#"{"error":{"reason":"missing","future_field":true},"version":2}"#.to_vec();
    assert_eq!(
        DiagnosticHttpError::from_response(404, &body)
            .unwrap()
            .error
            .reason,
        DiagnosticHttpReason::Missing
    );
    body.resize(MAX_DIAGNOSTIC_HTTP_ERROR_BYTES + 1, b' ');
    assert!(DiagnosticHttpError::from_response(404, &body).is_none());
}

#[test]
fn diagnostic_http_consumers_accept_escaped_strings_and_bounded_whitespace() {
    let mut body =
        br#" {"error":{"reason":"dis\u0061bled","future":{"nested":[1,true]}}} "#.to_vec();
    body.resize(MAX_DIAGNOSTIC_HTTP_ERROR_BYTES, b' ');
    assert_eq!(
        DiagnosticHttpError::from_response(503, &body),
        Some(DiagnosticHttpError::new(DiagnosticHttpReason::Disabled))
    );
}
