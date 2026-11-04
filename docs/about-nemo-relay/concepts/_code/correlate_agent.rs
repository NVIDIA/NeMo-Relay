// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay::context::current_scope_stack_snapshot;
use nemo_relay::scope::ScopeType;

fn agent_scope_uuid() -> nemo_relay::Result<Option<String>> {
    let snapshot = current_scope_stack_snapshot()?;
    Ok(snapshot.as_ref().and_then(|snapshot| {
        snapshot
            .scopes
            .iter()
            .rev()
            .find(|entry| entry.scope_type == ScopeType::Agent)
            .map(|entry| entry.scope_id.to_string())
    }))
}
// Correlate callbacks that share this Agent using (agent_scope_uuid, tool_call_id).
