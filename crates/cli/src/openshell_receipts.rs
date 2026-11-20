// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use http::{HeaderMap, StatusCode};
use nemo_relay_runtime::api::runtime::MiddlewareContinuationContext;
use nemo_relay_runtime::api::runtime::scope_stack::active_event_uuid;
use nemo_relay_runtime::api::scope::{self, EmitMarkEventParams};
use serde_json::json;
use uuid::Uuid;

const REQUEST_ID: &str = "x-openshell-request-id";

fn request_id(headers: &HeaderMap) -> Option<Uuid> {
    let mut values = headers.get_all(REQUEST_ID).iter();
    let value = values.next()?.to_str().ok()?;
    if values.next().is_some() || value.len() != 36 {
        return None;
    }
    let id = Uuid::parse_str(value).ok()?;
    id.hyphenated()
        .to_string()
        .eq_ignore_ascii_case(value)
        .then_some(id)
}

pub(crate) fn observe(headers: &HeaderMap, status: StatusCode) {
    let Some(id) = request_id(headers) else {
        return;
    };
    if active_event_uuid().is_none() || status.as_u16() >= 600 {
        return;
    }
    // Keep the actual managed call as the owner across task/thread boundaries.
    let context = MiddlewareContinuationContext::capture();
    if context
        .run_sync(|| {
            scope::event(
                EmitMarkEventParams::builder()
                    .name("openshell.request.receipt")
                    .data(json!({"request_id": id, "http_status": status.as_u16()}))
                    .build(),
            )
        })
        .is_err()
    {
        log::warn!(
            target: "nemo_relay.gateway",
            event = "openshell_receipt_not_recorded";
            "Could not record the OpenShell response receipt"
        );
    }
}

#[cfg(test)]
#[path = "../tests/coverage/shared/openshell_receipts_tests.rs"]
mod tests;
