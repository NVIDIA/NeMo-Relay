// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Context helpers for reading scope metadata on the intercept hot path.
//!
//! These functions read from the NeMo Relay scope stack (via [`current_scope_stack`])
//! to extract information needed by the LLM request intercept:
//!
//! - [`extract_scope_path`]: collects function names from the scope stack for trie lookup
//! - [`read_manual_latency_sensitivity`]: walks all scopes for manual `latency_sensitive` annotations
//! - [`resolve_agent_id`]: returns the first Agent scope name from the scope stack
//!
//! All functions are safe to call from sync contexts (intercepts are sync closures).
//! They acquire a read lock on the scope stack, which is always fast.
//!
//! # Metadata Convention
//!
//! Manual latency sensitivity is stored in scope metadata under the JSON path
//! `/nemo_relay_adaptive/latency_sensitivity` as a positive integer.

use nemo_relay::api::runtime::current_scope_stack;
use nemo_relay::api::scope::ScopeType;
use uuid::Uuid;

/// Metadata key path for manual latency sensitivity annotation.
pub const LATENCY_SENSITIVITY_POINTER: &str = "/nemo_relay_adaptive/latency_sensitivity";

/// Session-local scope identity used to coordinate warm-first cohorts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedParentScopeIdentity {
    /// UUID of the root scope for the current execution tree.
    pub root_uuid: Uuid,
    /// UUID of the parent scope shared by sibling fan-out work.
    pub shared_parent_uuid: Uuid,
}

/// Extracts the current function call path from the NeMo Relay scope stack.
///
/// Walks all scopes from root to top, skipping the root scope (index 0),
/// and collects names of Agent and Function scopes. This path is used
/// for prediction trie lookup.
///
/// # Returns
/// A vector of scope names from the current Agent and Function scope path.
/// Returns an empty vector when the scope stack cannot be read safely.
///
/// # Notes
/// The implicit root scope is always skipped.
pub fn extract_scope_path() -> Vec<String> {
    let stack_handle = current_scope_stack();
    let stack = match stack_handle.read() {
        Ok(s) => s,
        Err(_) => return vec![],
    };
    stack
        .scopes()
        .iter()
        .skip(1) // skip root
        .filter(|s| matches!(s.scope_type, ScopeType::Agent | ScopeType::Function))
        .map(|s| s.name.clone())
        .collect()
}

/// Reads the maximum manual latency sensitivity from all scopes in the current scope stack.
///
/// Walks all scopes and checks metadata for `/nemo_relay_adaptive/latency_sensitivity`.
/// Uses max-merge semantics: if multiple scopes have annotations, the highest wins.
///
/// # Returns
/// The highest manual latency sensitivity annotation visible on the current
/// scope stack, or `None` when no annotation exists.
///
/// # Notes
/// Returns `None` when the scope stack cannot be read safely.
pub fn read_manual_latency_sensitivity() -> Option<u32> {
    let stack_handle = current_scope_stack();
    let stack = match stack_handle.read() {
        Ok(s) => s,
        Err(_) => return None,
    };
    let mut max_val: Option<u32> = None;
    for scope in stack.scopes() {
        if let Some(ref meta) = scope.metadata
            && let Some(val) = meta
                .pointer(LATENCY_SENSITIVITY_POINTER)
                .and_then(|v| v.as_u64())
        {
            let val = val as u32;
            max_val = Some(max_val.map_or(val, |prev: u32| prev.max(val)));
        }
    }
    max_val
}

/// Sets latency sensitivity on the current (top) scope using max-merge semantics.
///
/// If the current scope already has a latency_sensitivity value, the new value
/// is only applied if it is greater than the existing one.
///
/// # Parameters
/// - `value`: New non-negative latency sensitivity hint (`>= 0`) for the
///   current top scope.
///
/// # Returns
/// `Ok(())` when the current scope metadata has been updated or left unchanged.
///
/// # Errors
/// Returns an error string when the scope stack lock is poisoned.
///
/// # Notes
/// Existing non-negative latency sensitivity values are updated using
/// max-merge semantics.
pub fn set_latency_sensitivity(value: u32) -> std::result::Result<(), String> {
    let stack_handle = current_scope_stack();
    let mut stack = stack_handle
        .write()
        .map_err(|e| format!("scope stack lock poisoned: {e}"))?;
    let scope = stack.top_mut();

    let existing = scope
        .metadata
        .as_ref()
        .and_then(|m| m.pointer(LATENCY_SENSITIVITY_POINTER))
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);

    let effective = match existing {
        Some(prev) if prev >= value => return Ok(()),
        _ => value,
    };

    let meta = scope.metadata.get_or_insert_with(|| serde_json::json!({}));
    if let Some(obj) = meta.as_object_mut() {
        let nemo_relay_adaptive = obj
            .entry("nemo_relay_adaptive")
            .or_insert_with(|| serde_json::json!({}));
        if let Some(np_obj) = nemo_relay_adaptive.as_object_mut() {
            np_obj.insert(
                "latency_sensitivity".to_string(),
                serde_json::json!(effective),
            );
        }
    }
    Ok(())
}

/// Resolves the agent ID from the current scope stack.
///
/// Walks all scopes from root to top, skipping the implicit root scope
/// (index 0, name="root"), and returns the name of the first Agent-typed scope.
///
/// # Returns
/// The first Agent scope name found on the current stack, or `None` when no
/// Agent scope is active.
///
/// # Notes
/// Returns `None` when the scope stack cannot be read safely.
pub fn resolve_agent_id() -> Option<String> {
    let stack_handle = current_scope_stack();
    let stack = match stack_handle.read() {
        Ok(s) => s,
        Err(_) => return None,
    };
    stack
        .scopes()
        .iter()
        .skip(1) // skip implicit root
        .find(|s| matches!(s.scope_type, ScopeType::Agent))
        .map(|s| s.name.clone())
}

/// Resolves the session-local identity used by warm-first cohort coordination.
///
/// The shared parent must come from the parent scope, not the current scope's
/// own UUID, so siblings under the same fan-out coordinate with one another.
/// Returns `None` if the scope stack cannot be read.
pub fn resolve_shared_parent_scope_identity() -> Option<SharedParentScopeIdentity> {
    let stack_handle = current_scope_stack();
    let stack = match stack_handle.read() {
        Ok(s) => s,
        Err(_) => return None,
    };

    let root_uuid = stack.root_uuid();
    let shared_parent_uuid = stack.top().parent_uuid.unwrap_or(root_uuid);

    Some(SharedParentScopeIdentity {
        root_uuid,
        shared_parent_uuid,
    })
}

#[cfg(test)]
#[path = "../tests/unit/context_helpers_tests.rs"]
mod tests;
