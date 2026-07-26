// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host-owned inference-provider services used by plugin components.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use serde_json::Value as Json;

use super::{PluginDeregistrationOutcome, PluginError, Result};

/// Versioned JSON request-response callback implemented by an inference provider.
#[doc(hidden)]
pub type InferenceProviderFn = Arc<dyn Fn(Json, Duration) -> Result<Json> + Send + Sync + 'static>;

/// Stable identity and request-response contract for one inference provider.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceProviderDescriptor {
    name: String,
    contract: String,
}

impl InferenceProviderDescriptor {
    /// Creates a provider descriptor after validating its stable identifiers.
    pub fn new(name: impl Into<String>, contract: impl Into<String>) -> Result<Self> {
        let name = normalized_identifier(name.into(), "inference provider name")?;
        let contract = normalized_identifier(contract.into(), "inference provider contract")?;
        Ok(Self { name, contract })
    }

    /// Returns the host-qualified provider name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the versioned request-response contract identifier.
    pub fn contract(&self) -> &str {
        &self.contract
    }
}

/// Resolved inference provider whose contract has already been checked.
#[doc(hidden)]
#[derive(Clone)]
pub struct InferenceProvider {
    descriptor: InferenceProviderDescriptor,
    callback: InferenceProviderFn,
}

impl InferenceProvider {
    /// Returns the provider descriptor.
    pub fn descriptor(&self) -> &InferenceProviderDescriptor {
        &self.descriptor
    }

    /// Invokes the provider with the component-owned request and deadline.
    pub fn invoke(&self, request: Json, timeout: Duration) -> Result<Json> {
        (self.callback)(request, timeout)
    }
}

struct RegisteredInferenceProvider {
    registration_id: u64,
    descriptor: InferenceProviderDescriptor,
    callback: InferenceProviderFn,
}

struct InferenceProviderRegistryInner {
    providers: RwLock<HashMap<String, RegisteredInferenceProvider>>,
    next_registration_id: AtomicU64,
}

/// Host-scoped registry for versioned inference providers.
#[doc(hidden)]
#[derive(Clone)]
pub struct InferenceProviderRegistry {
    inner: Arc<InferenceProviderRegistryInner>,
}

impl Default for InferenceProviderRegistry {
    fn default() -> Self {
        Self {
            inner: Arc::new(InferenceProviderRegistryInner {
                providers: RwLock::new(HashMap::new()),
                next_registration_id: AtomicU64::new(1),
            }),
        }
    }
}

impl InferenceProviderRegistry {
    /// Registers a provider and returns an ownership handle.
    pub fn register(
        &self,
        descriptor: InferenceProviderDescriptor,
        callback: InferenceProviderFn,
    ) -> Result<InferenceProviderRegistration> {
        let mut providers = self.inner.providers.write().map_err(|error| {
            PluginError::Internal(format!(
                "inference provider registry lock poisoned: {error}"
            ))
        })?;
        if providers.contains_key(descriptor.name()) {
            return Err(PluginError::RegistrationFailed(format!(
                "inference provider '{}' is already registered",
                descriptor.name()
            )));
        }
        let registration_id = self
            .inner
            .next_registration_id
            .fetch_add(1, Ordering::Relaxed);
        let name = descriptor.name().to_string();
        providers.insert(
            name.clone(),
            RegisteredInferenceProvider {
                registration_id,
                descriptor,
                callback,
            },
        );
        Ok(InferenceProviderRegistration {
            registry: self.clone(),
            name,
            registration_id: Some(registration_id),
        })
    }

    /// Resolves a provider only when its declared contract exactly matches.
    pub fn resolve(&self, name: &str, expected_contract: &str) -> Result<InferenceProvider> {
        let name = normalized_identifier(name.to_string(), "inference provider name")?;
        let expected_contract =
            normalized_identifier(expected_contract.to_string(), "inference provider contract")?;
        let providers = self.inner.providers.read().map_err(|error| {
            PluginError::Internal(format!(
                "inference provider registry lock poisoned: {error}"
            ))
        })?;
        let provider = providers.get(&name).ok_or_else(|| {
            PluginError::NotFound(format!("inference provider '{name}' is not registered"))
        })?;
        if provider.descriptor.contract() != expected_contract {
            return Err(PluginError::RegistrationFailed(format!(
                "inference provider '{name}' implements contract '{}' but '{}' is required",
                provider.descriptor.contract(),
                expected_contract
            )));
        }
        Ok(InferenceProvider {
            descriptor: provider.descriptor.clone(),
            callback: Arc::clone(&provider.callback),
        })
    }

    fn deregister(&self, name: &str, registration_id: u64) -> Result<PluginDeregistrationOutcome> {
        let mut providers = self.inner.providers.write().map_err(|error| {
            PluginError::Internal(format!(
                "inference provider registry lock poisoned: {error}"
            ))
        })?;
        match providers.get(name) {
            Some(provider) if provider.registration_id == registration_id => {
                providers.remove(name);
                Ok(PluginDeregistrationOutcome::Removed)
            }
            Some(_) => Ok(PluginDeregistrationOutcome::Replaced),
            None => Ok(PluginDeregistrationOutcome::Missing),
        }
    }
}

/// Owned registration for one provider in a host registry.
#[doc(hidden)]
pub struct InferenceProviderRegistration {
    registry: InferenceProviderRegistry,
    name: String,
    registration_id: Option<u64>,
}

impl InferenceProviderRegistration {
    /// Returns the registered provider name.
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

impl Drop for InferenceProviderRegistration {
    fn drop(&mut self) {
        if let Err(error) = self.deregister_checked() {
            log::error!(
                target: "nemo_relay.plugin",
                event = "inference_provider_cleanup_failed",
                provider = self.name.as_str();
                "Inference provider cleanup failed during drop: {error}"
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
