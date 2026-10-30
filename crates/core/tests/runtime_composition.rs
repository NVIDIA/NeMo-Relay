// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The public SDK composes built-ins over one runtime instance.

#[test]
fn sdk_and_runtime_share_state_and_builtin_registrations() {
    nemo_relay::initialize_runtime();
    let sdk_plugins = nemo_relay::plugin::list_plugin_kinds();
    assert_eq!(sdk_plugins, nemo_relay_runtime::plugin::list_plugin_kinds());
    for kind in ["observability", "pricing", "resource_metrics"] {
        assert!(sdk_plugins.iter().any(|registered| registered == kind));
    }
}
