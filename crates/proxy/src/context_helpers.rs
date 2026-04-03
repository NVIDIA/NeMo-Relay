// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Context helpers for reading scope metadata on the intercept hot path.
//!
//! These functions read from the Nexus scope stack (via [`current_scope_stack`])
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
//! `/nexus_proxy/latency_sensitivity` as a positive integer.

use nvidia_nat_nexus_core::{current_scope_stack, ScopeType};

/// Metadata key path for manual latency sensitivity annotation.
pub const LATENCY_SENSITIVITY_POINTER: &str = "/nexus_proxy/latency_sensitivity";

/// Extracts the current function call path from the Nexus scope stack.
///
/// Walks all scopes from root to top, skipping the root scope (index 0),
/// and collects names of Agent and Function scopes. This path is used
/// for prediction trie lookup.
///
/// Returns an empty vec if the scope stack is unavailable or poisoned.
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
/// Walks all scopes and checks metadata for `/nexus_proxy/latency_sensitivity`.
/// Uses max-merge semantics: if multiple scopes have annotations, the highest wins.
///
/// Returns `None` if no manual annotation exists or the scope stack is unavailable.
pub fn read_manual_latency_sensitivity() -> Option<u32> {
    let stack_handle = current_scope_stack();
    let stack = match stack_handle.read() {
        Ok(s) => s,
        Err(_) => return None,
    };
    let mut max_val: Option<u32> = None;
    for scope in stack.scopes() {
        if let Some(ref meta) = scope.metadata {
            if let Some(val) = meta
                .pointer(LATENCY_SENSITIVITY_POINTER)
                .and_then(|v| v.as_u64())
            {
                let val = val as u32;
                max_val = Some(max_val.map_or(val, |prev: u32| prev.max(val)));
            }
        }
    }
    max_val
}

/// Sets latency sensitivity on the current (top) scope using max-merge semantics.
///
/// If the current scope already has a latency_sensitivity value, the new value
/// is only applied if it is greater than the existing one.
///
/// Returns `Ok(())` on success, `Err` if the scope stack lock is poisoned.
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
        let nexus_proxy = obj
            .entry("nexus_proxy")
            .or_insert_with(|| serde_json::json!({}));
        if let Some(np_obj) = nexus_proxy.as_object_mut() {
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
/// Returns `None` if no Agent scope has been pushed or the scope stack is
/// unavailable.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_sensitivity_pointer_is_valid_json_pointer() {
        // JSON pointer must start with /
        assert!(LATENCY_SENSITIVITY_POINTER.starts_with('/'));
    }

    #[test]
    fn test_set_latency_sensitivity_basic() {
        // Sets value on the thread-local scope stack's root scope
        set_latency_sensitivity(3).unwrap();
        assert_eq!(read_manual_latency_sensitivity(), Some(3));

        // Clean up: reset root scope metadata
        let stack_handle = current_scope_stack();
        let mut stack = stack_handle.write().unwrap();
        stack.top_mut().metadata = None;
    }

    #[test]
    fn test_set_latency_sensitivity_max_merge_higher_wins() {
        set_latency_sensitivity(3).unwrap();
        set_latency_sensitivity(5).unwrap();
        assert_eq!(read_manual_latency_sensitivity(), Some(5));

        // Clean up
        let stack_handle = current_scope_stack();
        let mut stack = stack_handle.write().unwrap();
        stack.top_mut().metadata = None;
    }

    #[test]
    fn test_set_latency_sensitivity_max_merge_lower_noop() {
        set_latency_sensitivity(5).unwrap();
        set_latency_sensitivity(3).unwrap();
        // Lower value should not override
        assert_eq!(read_manual_latency_sensitivity(), Some(5));

        // Clean up
        let stack_handle = current_scope_stack();
        let mut stack = stack_handle.write().unwrap();
        stack.top_mut().metadata = None;
    }

    #[test]
    fn test_set_latency_sensitivity_read_roundtrip() {
        // Ensure read_manual_latency_sensitivity reads what set_latency_sensitivity writes
        set_latency_sensitivity(7).unwrap();
        assert_eq!(read_manual_latency_sensitivity(), Some(7));

        // Clean up
        let stack_handle = current_scope_stack();
        let mut stack = stack_handle.write().unwrap();
        stack.top_mut().metadata = None;
    }
}
