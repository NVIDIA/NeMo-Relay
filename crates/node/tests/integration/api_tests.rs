// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_flow::context::global::global_context;
use nemo_flow::context::state::NemoFlowContextState;
use serde_json::json;

use crate::api;
use crate::types::ScopeType;

fn reset_global() {
    let context = global_context();
    *context.write().unwrap() = NemoFlowContextState::new();
}

#[test]
fn scope_stack_functions_round_trip_through_public_api() {
    reset_global();
    api::clear_last_callback_error();

    let stack = api::create_scope_stack();
    api::set_thread_scope_stack(&stack);
    assert!(api::scope_stack_active());

    let handle = api::push_scope(
        "node.integration".into(),
        ScopeType::Agent,
        None,
        None,
        Some(json!({"source": "integration"})),
        None,
    )
    .unwrap();

    assert_eq!(handle.name(), "node.integration");
    assert!(matches!(handle.scope_type(), ScopeType::Agent));
    assert_eq!(handle.data(), Some(json!({"source": "integration"})));

    api::pop_scope(&handle).unwrap();
}

#[test]
fn stream_public_api_is_safe_for_unregistered_stream_ids() {
    api::clear_last_callback_error();
    assert_eq!(api::get_last_callback_error(), None);

    assert!(!api::push_stream_chunk(404.0, json!({"chunk": true})));
    api::end_stream(404.0);

    api::clear_last_callback_error();
    assert_eq!(api::get_last_callback_error(), None);
}
