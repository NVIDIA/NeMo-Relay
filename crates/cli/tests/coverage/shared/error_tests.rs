// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use nemo_relay::error::{FlowError, UpstreamFailure, UpstreamFailureClass};
use nemo_relay::observability::openinference::OpenInferenceError;

use super::*;

#[test]
fn log_kinds_cover_every_operational_error_class() {
    let upstream = reqwest::Client::new()
        .get("not a valid URL")
        .build()
        .expect_err("invalid URL should produce a reqwest error");
    let http = http::Request::builder()
        .uri("not a valid URI")
        .body(())
        .expect_err("invalid URI should produce an HTTP error");
    let errors = [
        (
            CliError::GuardrailRejected("blocked".into()),
            "guardrail_rejected",
        ),
        (
            CliError::InvalidPayload("invalid".into()),
            "invalid_payload",
        ),
        (
            CliError::PayloadTooLarge("large".into()),
            "payload_too_large",
        ),
        (
            CliError::Unauthorized("missing token".into()),
            "unauthorized",
        ),
        (CliError::Upstream(upstream), "upstream"),
        (
            CliError::ProviderFailure(UpstreamFailure {
                status: Some(503),
                body: "unavailable".into(),
                headers: BTreeMap::new(),
                class: UpstreamFailureClass::ModelUnavailable,
            }),
            "provider_failure",
        ),
        (CliError::Http(http), "http"),
        (CliError::Io(std::io::Error::other("io")), "io"),
        (CliError::Install("install".into()), "install"),
        (CliError::Config("config".into()), "configuration"),
        (CliError::Launch("launch".into()), "launch"),
        (
            CliError::HookDelivery {
                source: Box::new(CliError::Install("hook transport".into())),
            },
            "install",
        ),
        (
            CliError::PluginLifecycle {
                command: "add",
                target: None,
                kind: PluginLifecycleFailureKind::Failed,
                code: None,
                message: "plugin".into(),
            },
            "plugin_lifecycle",
        ),
        (
            CliError::Flow(FlowError::Internal("runtime".into())),
            "runtime",
        ),
        (
            CliError::OpenInference(OpenInferenceError::Provider("export".into())),
            "openinference",
        ),
    ];

    for (error, expected) in errors {
        assert_eq!(error.log_kind(), expected);
    }
}
