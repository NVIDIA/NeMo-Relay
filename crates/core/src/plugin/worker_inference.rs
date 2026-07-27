// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host-owned worker inference used by plugin components.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use serde_json::Value as Json;

use super::{PluginDeregistrationOutcome, PluginError, Result};

/// Versioned JSON request-response callback implemented by a worker.
#[doc(hidden)]
pub type WorkerInferenceFn = Arc<dyn Fn(Json, Duration) -> Result<Json> + Send + Sync + 'static>;

/// Stable identity and request-response contract for one worker inference callback.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerInferenceDescriptor {
    name: String,
    contract: String,
}

impl WorkerInferenceDescriptor {
    /// Creates a descriptor after validating its stable identifiers.
    pub fn new(name: impl Into<String>, contract: impl Into<String>) -> Result<Self> {
        let name = normalized_identifier(name.into(), "worker inference name")?;
        let contract = normalized_identifier(contract.into(), "worker inference contract")?;
        Ok(Self { name, contract })
    }

    /// Returns the host-qualified worker inference name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the versioned request-response contract identifier.
    pub fn contract(&self) -> &str {
        &self.contract
    }
}

/// Resolved worker inference callback whose contract has already been checked.
#[doc(hidden)]
#[derive(Clone)]
pub struct WorkerInference {
    descriptor: WorkerInferenceDescriptor,
    callback: WorkerInferenceFn,
}

impl WorkerInference {
    /// Returns the worker inference descriptor.
    pub fn descriptor(&self) -> &WorkerInferenceDescriptor {
        &self.descriptor
    }

    /// Invokes the worker with the component-owned request and deadline.
    pub fn invoke(&self, request: Json, timeout: Duration) -> Result<Json> {
        (self.callback)(request, timeout)
    }
}

struct RegisteredWorkerInference {
    registration_id: u64,
    descriptor: WorkerInferenceDescriptor,
    callback: WorkerInferenceFn,
}

struct WorkerInferenceRegistryInner {
    entries: RwLock<HashMap<String, RegisteredWorkerInference>>,
    next_registration_id: AtomicU64,
}

/// Host-scoped registry for versioned worker inference callbacks.
#[doc(hidden)]
#[derive(Clone)]
pub struct WorkerInferenceRegistry {
    inner: Arc<WorkerInferenceRegistryInner>,
}

impl Default for WorkerInferenceRegistry {
    fn default() -> Self {
        Self {
            inner: Arc::new(WorkerInferenceRegistryInner {
                entries: RwLock::new(HashMap::new()),
                next_registration_id: AtomicU64::new(1),
            }),
        }
    }
}

impl WorkerInferenceRegistry {
    /// Registers worker inference and returns an ownership handle.
    pub fn register(
        &self,
        descriptor: WorkerInferenceDescriptor,
        callback: WorkerInferenceFn,
    ) -> Result<WorkerInferenceRegistration> {
        let mut entries = self.inner.entries.write().map_err(|error| {
            PluginError::Internal(format!("worker inference registry lock poisoned: {error}"))
        })?;
        if entries.contains_key(descriptor.name()) {
            return Err(PluginError::RegistrationFailed(format!(
                "worker inference '{}' is already registered",
                descriptor.name()
            )));
        }
        let registration_id = self
            .inner
            .next_registration_id
            .fetch_add(1, Ordering::Relaxed);
        let name = descriptor.name().to_string();
        entries.insert(
            name.clone(),
            RegisteredWorkerInference {
                registration_id,
                descriptor,
                callback,
            },
        );
        Ok(WorkerInferenceRegistration {
            registry: self.clone(),
            name,
            registration_id: Some(registration_id),
        })
    }

    /// Resolves worker inference only when its declared contract exactly matches.
    pub fn resolve(&self, name: &str, expected_contract: &str) -> Result<WorkerInference> {
        let name = normalized_identifier(name.to_string(), "worker inference name")?;
        let expected_contract =
            normalized_identifier(expected_contract.to_string(), "worker inference contract")?;
        let entries = self.inner.entries.read().map_err(|error| {
            PluginError::Internal(format!("worker inference registry lock poisoned: {error}"))
        })?;
        let entry = entries.get(&name).ok_or_else(|| {
            PluginError::NotFound(format!("worker inference '{name}' is not registered"))
        })?;
        if entry.descriptor.contract() != expected_contract {
            return Err(PluginError::RegistrationFailed(format!(
                "worker inference '{name}' implements contract '{}' but '{}' is required",
                entry.descriptor.contract(),
                expected_contract
            )));
        }
        Ok(WorkerInference {
            descriptor: entry.descriptor.clone(),
            callback: Arc::clone(&entry.callback),
        })
    }

    fn deregister(&self, name: &str, registration_id: u64) -> Result<PluginDeregistrationOutcome> {
        let mut entries = self.inner.entries.write().map_err(|error| {
            PluginError::Internal(format!("worker inference registry lock poisoned: {error}"))
        })?;
        match entries.get(name) {
            Some(entry) if entry.registration_id == registration_id => {
                entries.remove(name);
                Ok(PluginDeregistrationOutcome::Removed)
            }
            Some(_) => Ok(PluginDeregistrationOutcome::Replaced),
            None => Ok(PluginDeregistrationOutcome::Missing),
        }
    }
}

/// Owned registration for one worker inference callback in a host registry.
#[doc(hidden)]
pub struct WorkerInferenceRegistration {
    registry: WorkerInferenceRegistry,
    name: String,
    registration_id: Option<u64>,
}

impl WorkerInferenceRegistration {
    /// Returns the registered worker inference name.
    pub fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn deregister_checked(&mut self) -> Result<PluginDeregistrationOutcome> {
        let Some(registration_id) = self.registration_id.take() else {
            return Ok(PluginDeregistrationOutcome::Missing);
        };
        self.registry.deregister(&self.name, registration_id)
    }
}

impl Drop for WorkerInferenceRegistration {
    fn drop(&mut self) {
        if let Err(error) = self.deregister_checked() {
            log::error!(
                target: "nemo_relay.plugin",
                event = "worker_inference_cleanup_failed",
                worker_inference = self.name.as_str();
                "Worker inference cleanup failed during drop: {error}"
            );
        }
    }
}

fn normalized_identifier(value: String, field: &str) -> Result<String> {
    let normalized = value.trim();
    if normalized.is_empty() {
        return Err(PluginError::RegistrationFailed(format!(
            "{field} must not be empty"
        )));
    }
    Ok(normalized.to_string())
}
