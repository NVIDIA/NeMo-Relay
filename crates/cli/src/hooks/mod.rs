// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hook delivery, command encoding, generated definitions, and configuration merging.

mod delivery;
mod destination;
mod encoding;
mod merging;
mod response;
mod types;

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
pub(crate) use encoding::{
    decode_windows_hook_command, encoded_windows_hook_command, event_matches_tools,
    persistent_hook_forward_command_for_platform, transparent_hook_forward_command_for_platform,
};
pub(crate) use encoding::{
    generated_hooks, persistent_hook_forward_command, transparent_hook_forward_command,
};
pub(crate) use merging::merge_hooks;
#[cfg(test)]
pub(crate) use response::{handle_hook_forward_status, handle_verified_hook_forward_response};
pub(crate) use types::{GatewayMode, HookForwardRequest};

#[cfg(test)]
use serde_json::json;

#[cfg(test)]
#[path = "../../tests/coverage/shared/installer_tests.rs"]
mod tests;
