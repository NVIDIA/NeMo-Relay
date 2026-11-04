// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic Relay trace and span identity.

#[doc(hidden)]
pub fn relay_trace_id(uuid: uuid::Uuid) -> opentelemetry::trace::TraceId {
    opentelemetry::trace::TraceId::from_bytes(*uuid.as_bytes())
}

#[doc(hidden)]
pub fn relay_span_id(uuid: uuid::Uuid) -> opentelemetry::trace::SpanId {
    let mut bytes = [0; 8];
    bytes.copy_from_slice(&uuid.as_bytes()[8..]);
    opentelemetry::trace::SpanId::from_bytes(bytes)
}

/// Format a W3C traceparent from Relay's deterministic trace and span IDs.
#[doc(hidden)]
pub fn format_traceparent(trace_uuid: uuid::Uuid, span_uuid: uuid::Uuid) -> String {
    let trace_id = trace_uuid
        .as_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let span_id = &span_uuid.as_bytes()[8..];
    let span_id = span_id
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("00-{trace_id}-{span_id}-01")
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;
    #[test]
    fn relay_id_conversions_preserve_zero_bytes() {
        let uuid = Uuid::nil();

        assert_eq!(relay_trace_id(uuid).to_bytes(), [0; 16]);
        assert_eq!(relay_span_id(uuid).to_bytes(), [0; 8]);
    }
}
