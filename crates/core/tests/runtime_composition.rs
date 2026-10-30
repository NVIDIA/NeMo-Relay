// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The public SDK exposes flat APIs over the shared runtime and plugin host.

#[test]
fn sdk_and_runtime_share_state_and_builtin_registrations() {
    use nemo_relay::scope::{self, EmitMarkEventParams, PopScopeParams, PushScopeParams};

    nemo_relay::initialize_runtime();
    let handle = scope::push_scope(
        PushScopeParams::builder()
            .name("sdk-contract")
            .scope_type(scope::ScopeType::Agent)
            .build(),
    )
    .expect("SDK scope should start");
    assert_eq!(
        handle.uuid,
        nemo_relay_runtime::api::scope::get_handle().unwrap().uuid
    );
    let data: nemo_relay::Json = serde_json::json!({"source": "sdk"});
    scope::event(
        EmitMarkEventParams::builder()
            .name("sdk-mark")
            .data(data)
            .build(),
    )
    .expect("SDK should emit through the shared runtime");
    scope::pop_scope(PopScopeParams::builder().handle_uuid(&handle.uuid).build())
        .expect("SDK scope should close");
    let sdk_plugins = nemo_relay::plugin::list_plugin_kinds();
    assert_eq!(sdk_plugins, nemo_relay_plugin_host::list_plugin_kinds());
    for kind in ["observability", "pricing", "resource_metrics"] {
        assert!(sdk_plugins.iter().any(|registered| registered == kind));
    }
}
