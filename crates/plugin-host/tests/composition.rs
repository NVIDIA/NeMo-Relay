// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Each composition selection is process-wide; run these through nextest.

#[test]
fn custom_runtime_services_still_receive_default_builtin_plugins() {
    use nemo_relay_runtime::composition::{RuntimeServices, install_runtime_services, services};

    install_runtime_services(|| RuntimeServices {
        infer_model_provider: |_, _| Some("embedding-provider".into()),
        ..Default::default()
    });
    nemo_relay_plugin_host::initialize_runtime();
    assert_eq!(
        (services().infer_model_provider)("model", None).as_deref(),
        Some("embedding-provider")
    );
    let kinds = nemo_relay_plugin_host::list_plugin_kinds();
    for kind in ["observability", "pricing", "resource_metrics"] {
        assert!(kinds.iter().any(|registered| registered == kind));
    }
}

#[test]
fn custom_host_services_keep_control_over_builtin_composition() {
    use nemo_relay_plugin_host::composition::{HostServices, install_host_services};
    use std::sync::atomic::{AtomicUsize, Ordering};
    static REGISTRATIONS: AtomicUsize = AtomicUsize::new(0);

    install_host_services(|| HostServices {
        register_builtin_plugins: || {
            REGISTRATIONS.fetch_add(1, Ordering::SeqCst);
            Ok(())
        },
        ..Default::default()
    });
    nemo_relay_plugin_host::initialize_runtime();
    nemo_relay_plugin_host::ensure_builtin_plugins_registered().unwrap();
    assert!(REGISTRATIONS.load(Ordering::SeqCst) > 0);
    assert!(nemo_relay_plugin_host::list_plugin_kinds().is_empty());
}
