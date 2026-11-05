// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::checked::lifecycle_tests::{Next, raw_invocation};
use a::Callback as _;

#[tokio::test]
async fn conditional_gates_reject_async_dispatch_and_contain_author_panics() {
    let gate = Gate(|kinds: &BTreeSet<RuntimeRegistrationKind>, name: &str| {
        assert!(kinds.contains(&RuntimeRegistrationKind::ToolExecutionIntercept));
        assert_eq!(name, "raw-tool");
        Some("blocked".to_owned())
    });
    let invocation = || raw_invocation(json!(["tool_execution_intercept"]));
    assert_eq!(
        from_abi(gate.invoke_sync(invocation()).into_result().unwrap()).unwrap(),
        json!("blocked")
    );
    for value in [Json::Null, json!(["unknown_kind"]), json!({})] {
        assert!(gate.invoke_sync(raw_invocation(value)).is_err());
    }
    assert!(gate.invoke(invocation(), ROption::RNone).await.is_err());
    assert!(
        gate.invoke_tool(
            invocation(),
            a::Continuation_TO::from_value(Next, TD_Opaque)
        )
        .await
        .is_err()
    );
    assert!(
        gate.open_stream(
            invocation(),
            a::Continuation_TO::from_value(Next, TD_Opaque)
        )
        .await
        .is_err()
    );
    let gate = Gate(
        |_: &BTreeSet<RuntimeRegistrationKind>, _: &str| -> Option<String> {
            panic!("author callback panic")
        },
    );
    let error = gate.invoke_sync(invocation()).into_result().unwrap_err();
    assert!(error.message.contains("native gate callback panicked"));
}
