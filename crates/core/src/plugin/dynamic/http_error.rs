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
