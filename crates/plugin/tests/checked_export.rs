// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Verify checked SDK export hygiene and bootstrap purity without extra author dependencies.
use nemo_relay_plugin::{
    Json, Plugin, PluginContext, PluginFuture,
    abi::{NATIVE_API_VERSION, bootstrap},
};
use serde_json::Map;
use std::sync::atomic::{AtomicUsize, Ordering};
static CONSTRUCTED: AtomicUsize = AtomicUsize::new(0);
struct Author;
impl Plugin for Author {
    fn plugin_kind(&self) -> &str {
        "checked_macro_fixture"
    }
    fn allows_multiple_components(&self) -> bool {
        false
    }
    fn register<'a>(
        &'a self,
        _: &'a Map<String, Json>,
        _: &'a mut PluginContext<'_>,
    ) -> PluginFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }
}
nemo_relay_plugin::nemo_relay_plugin!(kind = "checked_macro_fixture", || {
    CONSTRUCTED.fetch_add(1, Ordering::SeqCst);
    Author
});
#[test]
fn checked_bootstrap_does_not_construct_author_state() {
    let raw = __nemo_relay_checked_export::nemo_relay_native_bootstrap();
    let (descriptor, manifest) = unsafe { bootstrap::check(raw) }.unwrap();
    assert_eq!(descriptor.native_api, NATIVE_API_VERSION);
    assert_eq!(descriptor.plugin_kind, "checked_macro_fixture");
    let factory = unsafe { bootstrap::factory(manifest) }.unwrap();
    assert_eq!(CONSTRUCTED.load(Ordering::SeqCst), 0);
    let plugin = factory.create().unwrap();
    assert!(!plugin.allows_multiple_components());
    assert_eq!(CONSTRUCTED.load(Ordering::SeqCst), 1);
}
