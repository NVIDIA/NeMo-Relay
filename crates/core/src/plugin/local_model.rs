// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-local model-provider registry used by first-party plugin components.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, RwLock};
use std::time::Duration;

use serde_json::Value as Json;

use super::{PluginDeregistrationOutcome, PluginError, Result};

/// JSON request-response provider backed by a local runtime or worker process.
#[doc(hidden)]
pub type LocalModelProviderFn = Arc<dyn Fn(Json, Duration) -> Result<Json> + Send + Sync + 'static>;

struct RegisteredLocalModelProvider {
    registration_id: u64,
    callback: LocalModelProviderFn,
}

static LOCAL_MODEL_PROVIDERS: LazyLock<RwLock<HashMap<String, RegisteredLocalModelProvider>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));
static NEXT_LOCAL_MODEL_PROVIDER_ID: AtomicU64 = AtomicU64::new(1);

/// Registers a named local-model provider and returns its ownership token.
#[doc(hidden)]
pub fn register_local_model_provider_tracked(
    name: &str,
    callback: LocalModelProviderFn,
) -> Result<u64> {
    let name = name.trim();
    if name.is_empty() {
        return Err(PluginError::RegistrationFailed(
            "local-model provider name must not be empty".into(),
        ));
    }
    let mut providers = LOCAL_MODEL_PROVIDERS.write().map_err(|error| {
        PluginError::Internal(format!(
            "local-model provider registry lock poisoned: {error}"
        ))
    })?;
    if providers.contains_key(name) {
        return Err(PluginError::RegistrationFailed(format!(
            "local-model provider '{name}' is already registered"
        )));
    }
    let registration_id = NEXT_LOCAL_MODEL_PROVIDER_ID.fetch_add(1, Ordering::Relaxed);
    providers.insert(
        name.to_string(),
        RegisteredLocalModelProvider {
            registration_id,
            callback,
        },
    );
    Ok(registration_id)
}

/// Resolves a named local-model provider.
#[doc(hidden)]
pub fn local_model_provider(name: &str) -> Result<LocalModelProviderFn> {
    let name = name.trim();
    LOCAL_MODEL_PROVIDERS
        .read()
        .map_err(|error| {
            PluginError::Internal(format!(
                "local-model provider registry lock poisoned: {error}"
            ))
        })?
        .get(name)
        .map(|provider| Arc::clone(&provider.callback))
        .ok_or_else(|| {
            PluginError::NotFound(format!("local-model provider '{name}' is not registered"))
        })
}

/// Deregisters a local-model provider when the ownership token still matches.
#[doc(hidden)]
pub fn deregister_local_model_provider(name: &str, registration_id: u64) -> Result<bool> {
    deregister_local_model_provider_checked(name, registration_id).map(|outcome| {
        matches!(
            outcome,
            PluginDeregistrationOutcome::Removed | PluginDeregistrationOutcome::Missing
        )
    })
}

pub(crate) fn deregister_local_model_provider_checked(
    name: &str,
    registration_id: u64,
) -> Result<PluginDeregistrationOutcome> {
    let name = name.trim();
    let mut providers = LOCAL_MODEL_PROVIDERS.write().map_err(|error| {
        PluginError::Internal(format!(
            "local-model provider registry lock poisoned: {error}"
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
