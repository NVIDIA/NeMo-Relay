// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Both SDKs accept the same transport-independent plugin callback input.

use nemo_relay_types::api::tool::ToolExecutionContext;
use serde_json::json;

fn shared_tool_callback(context: ToolExecutionContext) -> nemo_relay_types::Json {
    assert_eq!(context.tool_name(), "lookup");
    assert_eq!(context.tool_call_id(), Some("provider-call"));
    assert_eq!(context.args()["id"], u64::MAX);
    context.into_args()
}

#[test]
fn native_and_worker_contexts_are_the_same_owned_domain_type() {
    let native: nemo_relay_plugin::ToolExecutionContext = nemo_relay_worker::ToolExecutionContext {
        tool_name: "lookup".into(),
        args: json!({"id": u64::MAX, "value": null}),
        tool_call_id: Some("provider-call".into()),
    };
    let worker: nemo_relay_worker::ToolExecutionContext = native.clone();
    assert_eq!(shared_tool_callback(native), shared_tool_callback(worker));
}
