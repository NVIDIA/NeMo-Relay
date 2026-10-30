// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The execution engine is usable without SDK defaults or concrete plugin dependencies.

use nemo_relay_runtime::{
    api::{
        scope::{self, EmitMarkEventParams, PopScopeParams, PushScopeParams, ScopeType},
        subscriber,
    },
    plugin,
};
use std::sync::{Arc, Mutex};

#[test]
fn runtime_executes_without_sdk_composition() {
    assert!(plugin::list_plugin_kinds().is_empty());
    let names = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&names);
    subscriber::register_subscriber(
        "runtime-only",
        Arc::new(move |event| {
            captured.lock().unwrap().push(event.name().to_owned());
        }),
    )
    .unwrap();
    let handle = scope::push_scope(
        PushScopeParams::builder()
            .name("standalone")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .unwrap();
    scope::event(
        EmitMarkEventParams::builder()
            .name("mark")
            .parent(&handle)
            .build(),
    )
    .unwrap();
    scope::pop_scope(PopScopeParams::builder().handle_uuid(&handle.uuid).build()).unwrap();
    subscriber::flush_subscribers().unwrap();
    assert_eq!(*names.lock().unwrap(), ["standalone", "mark", "standalone"]);
    assert!(plugin::list_plugin_kinds().is_empty());
    subscriber::deregister_subscriber("runtime-only").unwrap();
}
