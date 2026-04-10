// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-wide runtime ownership guard.
//!
//! NeMo Flow does not support multiple bindings claiming the runtime in the
//! same OS process. This module provides a minimal process-wide owner token so
//! the first binding (or direct Rust caller) claims ownership and later
//! incompatible bindings fail fast instead of silently creating a second
//! independent runtime.

#[cfg(not(target_arch = "wasm32"))]
use std::fmt;
#[cfg(any(test, not(target_arch = "wasm32")))]
use std::sync::Mutex;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::OnceLock;

use crate::error::{FlowError, Result};

#[cfg(not(target_arch = "wasm32"))]
const BINDING_KIND_ENV: &str = "NEMO_FLOW_BINDING_KIND";
#[cfg(not(target_arch = "wasm32"))]
const OWNER_TOKEN_ENV: &str = "NEMO_FLOW_RUNTIME_OWNER";

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, PartialEq, Eq)]
struct RuntimeOwner {
    pid: u32,
    binding_kind: String,
    major_version: String,
}

#[cfg(not(target_arch = "wasm32"))]
impl RuntimeOwner {
    fn current(binding_kind: String) -> Result<Self> {
        Ok(Self {
            pid: std::process::id(),
            binding_kind,
            major_version: current_compatibility_version()?.to_string(),
        })
    }

    fn parse(token: &str) -> Result<Self> {
        let mut pid = None;
        let mut binding_kind = None;
        let mut version = None;

        for field in token.split(';') {
            if let Some(value) = field.strip_prefix("pid=") {
                pid = Some(value.parse::<u32>().map_err(|e| {
                    FlowError::Internal(
                        format!("invalid NeMo Flow owner token pid {value:?}: {e}",),
                    )
                })?);
            } else if let Some(value) = field.strip_prefix("binding=") {
                if value.is_empty() {
                    return Err(FlowError::Internal(
                        "invalid NeMo Flow owner token: binding kind is empty".into(),
                    ));
                }
                binding_kind = Some(value.to_string());
            } else if let Some(value) = field.strip_prefix("version=") {
                version = Some(compatibility_major_version(value)?.to_string());
            }
        }

        Ok(Self {
            pid: pid.ok_or_else(|| {
                FlowError::Internal("invalid NeMo Flow owner token: missing pid".into())
            })?,
            binding_kind: binding_kind.ok_or_else(|| {
                FlowError::Internal("invalid NeMo Flow owner token: missing binding".into())
            })?,
            major_version: version.ok_or_else(|| {
                FlowError::Internal("invalid NeMo Flow owner token: missing version".into())
            })?,
        })
    }

    fn token(&self) -> String {
        format!(
            "pid={};binding={};version={}",
            self.pid, self.binding_kind, self.major_version
        )
    }

    fn same_owner(&self, other: &Self) -> bool {
        self.pid == other.pid
            && self.binding_kind == other.binding_kind
            && self.major_version == other.major_version
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl fmt::Display for RuntimeOwner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}@{} pid={}",
            self.binding_kind, self.major_version, self.pid
        )
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
struct RuntimeOwnerController {
    binding_kind: Option<String>,
}

#[cfg(not(target_arch = "wasm32"))]
static RUNTIME_OWNER_CONTROLLER: OnceLock<Mutex<RuntimeOwnerController>> = OnceLock::new();

#[cfg(not(target_arch = "wasm32"))]
fn runtime_owner_controller() -> &'static Mutex<RuntimeOwnerController> {
    RUNTIME_OWNER_CONTROLLER.get_or_init(|| Mutex::new(RuntimeOwnerController::default()))
}

fn compatibility_major_version(version: &str) -> Result<&str> {
    version
        .split('.')
        .next()
        .filter(|value| !value.is_empty() && value.chars().all(|c| c.is_ascii_digit()))
        .ok_or_else(|| {
            FlowError::Internal(format!(
                "invalid NeMo Flow version {version:?}: expected a semver-compatible major",
            ))
        })
}

fn current_compatibility_version() -> Result<&'static str> {
    compatibility_major_version(env!("CARGO_PKG_VERSION"))
}

#[cfg(not(target_arch = "wasm32"))]
fn resolve_binding_kind(binding_kind: Option<String>) -> String {
    binding_kind
        .or_else(|| std::env::var(BINDING_KIND_ENV).ok())
        .unwrap_or_else(|| "rust".to_string())
}

#[cfg(not(target_arch = "wasm32"))]
fn read_process_runtime_owner() -> Result<Option<RuntimeOwner>> {
    let Some(token) = std::env::var(OWNER_TOKEN_ENV)
        .ok()
        .filter(|value| !value.is_empty())
    else {
        return Ok(None);
    };

    match RuntimeOwner::parse(&token) {
        Ok(owner) => Ok(Some(owner)),
        Err(_) => {
            clear_process_runtime_owner();
            Ok(None)
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn publish_process_runtime_owner(owner: &RuntimeOwner) {
    // Runtime ownership is intentionally process-global.
    unsafe { std::env::set_var(OWNER_TOKEN_ENV, owner.token()) };
}

#[cfg(not(target_arch = "wasm32"))]
fn clear_process_runtime_owner() {
    // Runtime ownership is intentionally process-global.
    unsafe { std::env::remove_var(OWNER_TOKEN_ENV) };
}

#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub fn initialize_shared_runtime_binding(binding_kind: &str) -> Result<()> {
    let previous_binding_kind = {
        let controller = runtime_owner_controller();
        let mut guard = controller.lock().map_err(|e| {
            FlowError::Internal(format!("runtime owner controller lock poisoned: {e}"))
        })?;
        if let Some(existing) = guard.binding_kind.as_deref()
            && existing != binding_kind
        {
            return Err(FlowError::InvalidArgument(format!(
                "NeMo Flow binding identity is already initialized as {existing}; attempted={binding_kind}",
            )));
        }
        let previous = guard.binding_kind.clone();
        guard
            .binding_kind
            .get_or_insert_with(|| binding_kind.to_string());
        previous
    };

    if let Err(error) = ensure_process_runtime_owner() {
        if previous_binding_kind.is_none() {
            let controller = runtime_owner_controller();
            let mut guard = controller.lock().map_err(|e| {
                FlowError::Internal(format!("runtime owner controller lock poisoned: {e}"))
            })?;
            if guard.binding_kind.as_deref() == Some(binding_kind) {
                guard.binding_kind = None;
            }
        }
        return Err(error);
    }

    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[doc(hidden)]
pub fn initialize_shared_runtime_binding(_binding_kind: &str) -> Result<()> {
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn ensure_process_runtime_owner() -> Result<()> {
    let binding_kind = {
        let controller = runtime_owner_controller();
        let guard = controller.lock().map_err(|e| {
            FlowError::Internal(format!("runtime owner controller lock poisoned: {e}"))
        })?;
        resolve_binding_kind(guard.binding_kind.clone())
    };
    let current = RuntimeOwner::current(binding_kind)?;

    match read_process_runtime_owner()? {
        Some(existing) if existing.same_owner(&current) => Ok(()),
        Some(existing) if existing.pid != current.pid => {
            publish_process_runtime_owner(&current);
            Ok(())
        }
        Some(existing) => Err(FlowError::InvalidArgument(format!(
            "NeMo Flow does not support multiple bindings in one process; existing owner={} attempted={}",
            existing, current
        ))),
        None => {
            publish_process_runtime_owner(&current);
            Ok(())
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn ensure_process_runtime_owner() -> Result<()> {
    Ok(())
}

#[cfg(test)]
static TEST_MUTEX: Mutex<()> = Mutex::new(());

#[cfg(test)]
pub(crate) fn runtime_owner_test_mutex() -> &'static Mutex<()> {
    &TEST_MUTEX
}

#[cfg(test)]
pub(crate) fn reset_runtime_owner_for_tests() {
    #[cfg(target_arch = "wasm32")]
    {
        return;
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        clear_process_runtime_owner();
        let controller = runtime_owner_controller();
        let mut guard = controller.lock().unwrap();
        guard.binding_kind = None;
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    use crate::api;

    fn acquire_test_lock() -> std::sync::MutexGuard<'static, ()> {
        runtime_owner_test_mutex()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn owner_token_for(pid: u32, binding_kind: &str, version: &str) -> String {
        format!("pid={pid};binding={binding_kind};version={version}")
    }

    #[test]
    fn test_initialize_binding_claims_process_owner() {
        let _lock = acquire_test_lock();
        reset_runtime_owner_for_tests();

        initialize_shared_runtime_binding("python").unwrap();

        let owner = read_process_runtime_owner().unwrap().unwrap();
        assert_eq!(owner.pid, std::process::id());
        assert_eq!(owner.binding_kind, "python");
        assert_eq!(
            owner.major_version,
            current_compatibility_version().unwrap().to_string()
        );
    }

    #[test]
    fn test_same_binding_same_major_reuses_process_owner() {
        let _lock = acquire_test_lock();
        reset_runtime_owner_for_tests();

        let version = format!("{}.999.999", current_compatibility_version().unwrap());
        unsafe {
            std::env::set_var(
                OWNER_TOKEN_ENV,
                owner_token_for(std::process::id(), "python", &version),
            )
        };

        initialize_shared_runtime_binding("python").unwrap();

        let owner = read_process_runtime_owner().unwrap().unwrap();
        assert_eq!(owner.binding_kind, "python");
        assert_eq!(
            owner.major_version,
            current_compatibility_version().unwrap().to_string()
        );
    }

    #[test]
    fn test_conflicting_binding_is_rejected() {
        let _lock = acquire_test_lock();
        reset_runtime_owner_for_tests();

        unsafe {
            std::env::set_var(
                OWNER_TOKEN_ENV,
                owner_token_for(
                    std::process::id(),
                    "python",
                    current_compatibility_version().unwrap(),
                ),
            )
        };

        let error = initialize_shared_runtime_binding("node").unwrap_err();
        let message = error.to_string();
        assert!(message.contains("NeMo Flow does not support multiple bindings in one process"));
        assert!(message.contains("existing owner=python@"));
        assert!(message.contains("attempted=node@"));
    }

    #[test]
    fn test_stale_pid_owner_is_replaced() {
        let _lock = acquire_test_lock();
        reset_runtime_owner_for_tests();

        unsafe {
            std::env::set_var(
                OWNER_TOKEN_ENV,
                owner_token_for(
                    std::process::id() + 1,
                    "python",
                    current_compatibility_version().unwrap(),
                ),
            )
        };

        initialize_shared_runtime_binding("node").unwrap();

        let owner = read_process_runtime_owner().unwrap().unwrap();
        assert_eq!(owner.pid, std::process::id());
        assert_eq!(owner.binding_kind, "node");
    }

    #[test]
    fn test_api_use_claims_default_rust_owner() {
        let _lock = acquire_test_lock();
        reset_runtime_owner_for_tests();

        let handle = api::nemo_flow_get_handle().unwrap();
        assert_eq!(handle.name, "root");

        let owner = read_process_runtime_owner().unwrap().unwrap();
        assert_eq!(owner.binding_kind, "rust");
    }

    #[test]
    fn test_api_use_rejects_conflicting_owner() {
        let _lock = acquire_test_lock();
        reset_runtime_owner_for_tests();

        unsafe {
            std::env::set_var(
                OWNER_TOKEN_ENV,
                owner_token_for(
                    std::process::id(),
                    "python",
                    current_compatibility_version().unwrap(),
                ),
            )
        };

        let error = api::nemo_flow_get_handle().unwrap_err();
        let message = error.to_string();
        assert!(message.contains("NeMo Flow does not support multiple bindings in one process"));
        assert!(message.contains("existing owner=python@"));
        assert!(message.contains("attempted=rust@"));
    }
}
