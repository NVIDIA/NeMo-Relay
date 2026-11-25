// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay::scope::{
    self, EmitMarkEventParams, PopScopeParams, PushScopeParams, ScopeAttributes, ScopeType,
};
use nemo_relay::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use serde_json::json;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    register_subscriber(
        "scope-check",
        Arc::new(|event| {
            println!("{} {}", event.kind(), event.name());
        }),
    )?;

    let handle = scope::push_scope(
        PushScopeParams::builder()
            .name("agent-run")
            .scope_type(ScopeType::Agent)
            .attributes(ScopeAttributes::empty())
            .input(json!({"request_id": "req-123"}))
            .build(),
    )?;

    scope::event(
        EmitMarkEventParams::builder()
            .name("planning-started")
            .parent(&handle)
            .data(json!({"step": 1}))
            .build(),
    )?;
    scope::event(
        EmitMarkEventParams::builder()
            .name("planning-finished")
            .parent(&handle)
            .data(json!({"step": 2}))
            .build(),
    )?;

    scope::pop_scope(PopScopeParams::builder().handle_uuid(&handle.uuid).build())?;
    flush_subscribers()?;
    let _ = deregister_subscriber("scope-check")?;
    Ok(())
}
