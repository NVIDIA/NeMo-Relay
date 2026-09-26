// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hook delivery, command encoding, generated definitions, and configuration merging.

mod config;
mod delivery;
mod destination;
mod encoding;
mod merging;
mod response;
mod types;

pub(crate) use config::HookCommandConfig;
pub(crate) use config::NATIVE_INVOCATION_CONFIG;
pub(crate) use delivery::hook_forward;
#[cfg(test)]
pub(crate) use delivery::send_verified_hook_forward_request;
#[cfg(test)]
pub(crate) use delivery::{gateway_headers, insert_header, read_hook_payload_from};
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
pub(crate) use merging::merge_hooks;
#[cfg(test)]
pub(crate) use response::{handle_hook_forward_status, handle_verified_hook_forward_response};
pub(crate) use types::{GatewayMode, HookFailurePolicy, HookForwardRequest};

#[cfg(test)]
use serde_json::json;

#[cfg(test)]
#[path = "../../tests/coverage/shared/installer_tests.rs"]
mod tests;
