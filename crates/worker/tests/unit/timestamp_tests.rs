// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use super::*;

#[tokio::test]
async fn historical_scope_rejects_timestamp_outside_wire_range_before_rpc() {
    let Some(outside_wire_range) =
        UNIX_EPOCH.checked_add(Duration::from_micros(i64::MAX as u64 + 1))
    else {
        // Windows FILETIME cannot represent a SystemTime this far after the
        // epoch, so the public API cannot receive this overflow case there.
        return;
    };
    let runtime = PluginRuntime {
        activation_id: "activation".into(),
        auth_token: "token".into(),
        host_endpoint: "unsupported://host".into(),
        host_channel: Arc::new(OnceCell::new()),
        conditional_middleware_callbacks: Arc::new(Mutex::new(HashMap::new())),
    };

    let push_error = runtime
        .push_scope_at(
            None,
            "historical",
            ScopeType::Custom,
            None,
            None,
            None,
            outside_wire_range,
        )
        .await
        .expect_err("out-of-range scope start should fail before the host call");
    let pop_error = runtime
        .pop_scope_at("scope-handle", None, None, outside_wire_range)
        .await
        .expect_err("out-of-range scope end should fail before the host call");

    for error in [push_error, pop_error] {
        assert!(matches!(error, WorkerSdkError::InvalidInput(_)));
        assert_eq!(
            error.to_string(),
            "invalid input: scope timestamp exceeds the supported range"
        );
    }
}
