// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Privacy-preserving operational context and records for CLI request boundaries.

use std::time::Instant;

use axum::http::{HeaderMap, HeaderValue};
use uuid::Uuid;

/// Internal-only header that joins a short-lived hook-forward process to the gateway that handles
/// its request. It is removed at ingress and is never forwarded to a provider or exposed through
/// event metadata.
pub(crate) const OPERATION_ID_HEADER: &str = "x-nemo-relay-operation-id";

pub(crate) const UPSTREAM_RESPONSE_THRESHOLD_MILLIS: u64 = 10_000;
pub(crate) const UPSTREAM_STREAM_STALL_THRESHOLD_MILLIS: u64 = 60_000;

#[derive(Clone, Debug)]
pub(crate) struct OperationalContext {
    operation_id: String,
    session_id: Option<String>,
    started_at: Instant,
}

impl OperationalContext {
    pub(crate) fn new() -> Self {
        Self {
            operation_id: Uuid::now_v7().to_string(),
            session_id: None,
            started_at: Instant::now(),
        }
    }

    /// Creates a new gateway operation and drops any caller-supplied correlation value. Gateway
    /// callers are never allowed to select an operational log ID.
    pub(crate) fn new_gateway(headers: &mut HeaderMap) -> Self {
        headers.remove(OPERATION_ID_HEADER);
        Self::new()
    }

    /// Reads a Relay-generated operation ID when present. Invalid external input is ignored so it
    /// cannot create a misleading correlation link or be reflected into logs.
    pub(crate) fn take_from_headers(headers: &mut HeaderMap) -> Self {
        let operation_id = headers
            .remove(OPERATION_ID_HEADER)
            .and_then(|value| value.to_str().ok().map(ToOwned::to_owned))
            .and_then(|value| Uuid::parse_str(&value).ok().map(|_| value))
            .unwrap_or_else(|| Uuid::now_v7().to_string());
        Self {
            operation_id,
            session_id: None,
            started_at: Instant::now(),
        }
    }

    pub(crate) fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    pub(crate) fn attach_header(&self, headers: &mut HeaderMap) {
        // UUID text is always a valid HTTP header value.
        headers.insert(
            OPERATION_ID_HEADER,
            HeaderValue::from_str(&self.operation_id).expect("operation ID is valid header text"),
        );
    }

    pub(crate) fn elapsed_millis(&self) -> u128 {
        self.started_at.elapsed().as_millis()
    }

    pub(crate) fn operation_id(&self) -> &str {
        &self.operation_id
    }

    fn fields(&self) -> (&str, Option<&str>) {
        (&self.operation_id, self.session_id.as_deref())
    }
}

/// Emits one fixed-schema record while omitting `session_id` until Relay has selected one.
/// Event names are literals at every call site so the record contract cannot drift at runtime.
macro_rules! operational_log {
    ($level:expr, $event:literal, $context:expr; $($fields:tt)*) => {{
        let (operation_id, session_id) = $context.fields();
        match session_id {
            Some(session_id) => log::log!(
                target: "nemo_relay.operational",
                $level,
                event = $event,
                operation_id,
                session_id,
                $($fields)*;
                "CLI operational record"
            ),
            None => log::log!(
                target: "nemo_relay.operational",
                $level,
                event = $event,
                operation_id,
                $($fields)*;
                "CLI operational record"
            ),
        }
    }};
}

pub(crate) fn hook_started(context: &OperationalContext, boundary: &'static str) {
    operational_log!(
        log::Level::Debug,
        "hook_started",
        context;
        boundary,
        outcome = "started",
        elapsed_millis = 0
    );
}

pub(crate) fn hook_completed(
    context: &OperationalContext,
    boundary: &'static str,
    outcome: &'static str,
) {
    operational_log!(
        log::Level::Debug,
        "hook_completed",
        context;
        boundary,
        outcome,
        elapsed_millis = context.elapsed_millis()
    );
}

pub(crate) fn hook_failed(
    context: &OperationalContext,
    boundary: &'static str,
    error_kind: &'static str,
    fail_closed: bool,
) {
    hook_failed_with_status(context, boundary, error_kind, fail_closed, None);
}

pub(crate) fn hook_failed_with_status(
    context: &OperationalContext,
    boundary: &'static str,
    error_kind: &'static str,
    fail_closed: bool,
    status_code: Option<u16>,
) {
    let level = if fail_closed {
        log::Level::Error
    } else {
        log::Level::Warn
    };
    if let Some(status_code) = status_code {
        operational_log!(
            level,
            "hook_failed",
            context;
            boundary,
            error_kind,
            status_code,
            elapsed_millis = context.elapsed_millis()
        );
    } else {
        operational_log!(
            level,
            "hook_failed",
            context;
            boundary,
            error_kind,
            elapsed_millis = context.elapsed_millis()
        );
    }
}

pub(crate) fn limit_exceeded(
    context: &OperationalContext,
    boundary: &'static str,
    limit_name: &'static str,
    limit_bytes: usize,
) {
    operational_log!(
        log::Level::Error,
        "limit_exceeded",
        context;
        boundary,
        limit_name,
        limit_bytes,
        elapsed_millis = context.elapsed_millis()
    );
}

pub(crate) fn upstream_started(context: &OperationalContext, streaming: bool) {
    operational_log!(
        log::Level::Debug,
        "upstream_started",
        context;
        boundary = "upstream",
        outcome = if streaming { "streaming" } else { "buffered" },
        elapsed_millis = context.elapsed_millis()
    );
}

pub(crate) fn upstream_headers_received(context: &OperationalContext, streaming: bool) {
    operational_log!(
        log::Level::Debug,
        "upstream_headers_received",
        context;
        boundary = "upstream",
        outcome = if streaming { "streaming" } else { "buffered" },
        elapsed_millis = context.elapsed_millis()
    );
}

pub(crate) fn upstream_completed(context: &OperationalContext, streaming: bool) {
    operational_log!(
        log::Level::Debug,
        "upstream_completed",
        context;
        boundary = "upstream",
        outcome = if streaming { "streaming" } else { "buffered" },
        elapsed_millis = context.elapsed_millis()
    );
}

pub(crate) fn upstream_first_event(context: &OperationalContext) {
    operational_log!(
        log::Level::Debug,
        "upstream_first_event",
        context;
        boundary = "upstream",
        outcome = "streaming",
        elapsed_millis = context.elapsed_millis()
    );
}

pub(crate) fn upstream_cancelled(context: &OperationalContext) {
    operational_log!(
        log::Level::Debug,
        "upstream_cancelled",
        context;
        boundary = "upstream",
        outcome = "streaming",
        elapsed_millis = context.elapsed_millis()
    );
}

pub(crate) fn upstream_delayed(
    context: &OperationalContext,
    event: &'static str,
    threshold_millis: u64,
) {
    match event {
        "upstream_headers_delayed" => operational_log!(
            log::Level::Warn,
            "upstream_headers_delayed",
            context;
            boundary = "upstream",
            phase = "headers",
            threshold_millis,
            elapsed_millis = context.elapsed_millis()
        ),
        "upstream_first_event_delayed" => operational_log!(
            log::Level::Warn,
            "upstream_first_event_delayed",
            context;
            boundary = "upstream",
            phase = "first_event",
            threshold_millis,
            elapsed_millis = context.elapsed_millis()
        ),
        "upstream_stream_stalled" => operational_log!(
            log::Level::Warn,
            "upstream_stream_stalled",
            context;
            boundary = "upstream",
            phase = "stream",
            threshold_millis,
            elapsed_millis = context.elapsed_millis()
        ),
        _ => unreachable!("operational delay records use a fixed phase"),
    }
}

pub(crate) fn upstream_status(context: &OperationalContext, status_code: u16) {
    operational_log!(
        log::Level::Warn,
        "upstream_non_success",
        context;
        boundary = "upstream",
        status_code,
        elapsed_millis = context.elapsed_millis()
    );
}

pub(crate) fn upstream_failed(context: &OperationalContext, error_kind: &'static str) {
    operational_log!(
        log::Level::Error,
        "upstream_failed",
        context;
        boundary = "upstream",
        error_kind,
        elapsed_millis = context.elapsed_millis()
    );
}

pub(crate) fn upstream_stream_read_failed(context: &OperationalContext) {
    operational_log!(
        log::Level::Warn,
        "upstream_stream_read_failed",
        context;
        boundary = "upstream",
        error_kind = "stream_read",
        elapsed_millis = context.elapsed_millis()
    );
}
