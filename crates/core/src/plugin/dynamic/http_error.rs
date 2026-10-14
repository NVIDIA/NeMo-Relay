// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::error::{FlowError, UpstreamFailure, UpstreamFailureClass};
use nemo_relay_types::plugin::{http_error_message, parse_http_error_message};

pub(super) fn from_message(message: String) -> FlowError {
    upstream_from_message(&message).unwrap_or(FlowError::Internal(message))
}

pub(super) fn upstream_from_message(message: &str) -> Option<FlowError> {
    let (status, body) = parse_http_error_message(message)?;
    Some(FlowError::Upstream(UpstreamFailure {
        status: Some(status),
        body: body.to_owned(),
        headers: Default::default(),
        class: UpstreamFailureClass::Other,
    }))
}

pub(super) fn to_message(error: &FlowError) -> String {
    if let FlowError::Upstream(failure) = error
        && let Some(status) = failure.status
        && let Ok(message) = http_error_message(status, &failure.body)
    {
        return message;
    }
    error.to_string()
}

/// Adds the host-verified reporting identity without claiming the plugin caused the failure.
pub(super) fn reported_by(
    mut error: FlowError,
    plugin_id: &str,
    boundary: &'static str,
) -> FlowError {
    if let FlowError::Upstream(failure) = &mut error {
        let context = format!("Reported by plugin '{plugin_id}': ");
        // An unchanged error returned through next may cross this same boundary again.
        if !failure.body.starts_with(&context) {
            failure.body = format!("{context}{}", failure.body);
        }
        log::error!(
            target: "nemo_relay.plugin",
            event = "plugin_http_error_reported",
            plugin_id,
            boundary,
            status_code = failure.status;
            "Plugin reported an HTTP error"
        );
    }
    error
}

#[cfg(test)]
#[path = "../../../tests/unit/plugin_http_reporting_tests.rs"]
mod tests;
