// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hook delivery, command encoding, generated definitions, and configuration merging.

// Match the native agent hook budget across gateway and daemon/worker delivery.
pub(crate) const HOOK_FORWARD_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

mod config;
mod delivery;
mod destination;
mod encoding;
#[cfg(test)]
mod merging;
mod response;
mod types;

pub(crate) use config::HookCommandConfig;
#[cfg(test)]
pub(crate) use delivery::send_verified_hook_forward_request;
#[cfg(test)]
pub(crate) use delivery::{gateway_headers, insert_header, read_hook_payload_from};
pub(crate) use delivery::{hook_forward, hook_forward_pi};
#[cfg(test)]
pub(crate) use destination::{
    HookGatewayLifecycle, resolve_hook_destination, transparent_gateway_spec,
};
#[cfg(test)]
pub(crate) use encoding::transparent_hook_forward_commands;
pub(crate) use encoding::{
    GeneratedHookCommands, event_requires_fail_closed, generated_policy_hooks,
    persistent_hook_config_path, persistent_hook_forward_commands,
    transparent_hook_forward_commands_with_config,
};
#[cfg(test)]
pub(crate) use encoding::{
    event_matches_tools, generated_hooks, persistent_hook_forward_commands_for_platform,
    transparent_hook_forward_commands_for_platform,
};
#[cfg(test)]
pub(crate) use merging::merge_hooks;
#[cfg(test)]
pub(crate) use response::{handle_hook_forward_status, handle_verified_hook_forward_response};
pub(crate) use types::{GatewayMode, HookFailurePolicy, HookForwardRequest};

#[cfg(test)]
use serde_json::json;

/// Unknown payloads can hide a permission event. Block delivery failures by default.
pub(crate) fn effective_fail_closed(policy: HookFailurePolicy, payload: Option<&[u8]>) -> bool {
    match policy {
        HookFailurePolicy::FailOpen => false,
        HookFailurePolicy::FailClosed => true,
        HookFailurePolicy::Default => policy.fail_closed() || payload_requires_fail_closed(payload),
    }
}

pub(crate) fn payload_requires_fail_closed(payload: Option<&[u8]>) -> bool {
    payload
        .and_then(|payload| serde_json::from_slice::<serde_json::Value>(payload).ok())
        .and_then(|payload| {
            ["hook_event_name", "event_name", "event", "type"]
                .into_iter()
                .find_map(|name| payload.get(name).and_then(serde_json::Value::as_str))
                .map(event_requires_fail_closed)
        })
        .unwrap_or(true)
}

#[cfg(test)]
#[path = "../../tests/coverage/shared/installer_tests.rs"]
mod tests;
