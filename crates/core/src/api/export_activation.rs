// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Activation-time policy hooks for Relay-managed remote exporters.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, LazyLock, RwLock};

use crate::error::{FlowError, Result};
pub use nemo_relay_types::plugin::{
    ExportActivationDecision, ExportActivationRequest, ExportActivationTargetKind,
};

/// Asynchronous callback registered by one export-activation policy provider.
pub type ExportActivationPolicyFn = Arc<
    dyn Fn(
            ExportActivationRequest,
        ) -> Pin<Box<dyn Future<Output = Result<ExportActivationDecision>> + Send>>
        + Send
        + Sync,
>;

static EXPORT_ACTIVATION_POLICIES: LazyLock<RwLock<HashMap<String, ExportActivationPolicyFn>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

pub(crate) fn register_export_activation_policy(
    provider: &str,
    callback: ExportActivationPolicyFn,
) -> Result<()> {
    let mut policies = EXPORT_ACTIVATION_POLICIES.write().map_err(|error| {
        FlowError::Internal(format!(
            "export activation policy registry lock poisoned: {error}"
        ))
    })?;
    if policies.contains_key(provider) {
        return Err(FlowError::AlreadyExists(provider.to_string()));
    }
    policies.insert(provider.to_string(), callback);
    Ok(())
}

pub(crate) fn deregister_export_activation_policy(provider: &str) -> Result<bool> {
    EXPORT_ACTIVATION_POLICIES
        .write()
        .map(|mut policies| policies.remove(provider).is_some())
        .map_err(|error| {
            FlowError::Internal(format!(
                "export activation policy registry lock poisoned: {error}"
            ))
        })
}

pub(crate) async fn evaluate_export_activation_policy(
    provider: &str,
    request: ExportActivationRequest,
) -> Result<ExportActivationDecision> {
    let callback = EXPORT_ACTIVATION_POLICIES
        .read()
        .map_err(|error| {
            FlowError::Internal(format!(
                "export activation policy registry lock poisoned: {error}"
            ))
        })?
        .get(provider)
        .cloned()
        .ok_or_else(|| FlowError::NotFound(provider.to_string()))?;
    callback(request).await
}
