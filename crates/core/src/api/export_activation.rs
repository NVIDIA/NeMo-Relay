// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Activation-time policy hooks for Relay-managed remote exporters.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};

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

/// Export-activation policy callbacks owned by one plugin activation.
#[derive(Default)]
pub(crate) struct ExportActivationPolicyRegistry {
    policies: RwLock<HashMap<String, ExportActivationPolicyFn>>,
}

impl ExportActivationPolicyRegistry {
    pub(crate) fn register(
        &self,
        provider: &str,
        callback: ExportActivationPolicyFn,
    ) -> Result<()> {
        let mut policies = self.policies.write().map_err(|error| {
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

    pub(crate) fn deregister(&self, provider: &str) -> Result<bool> {
        self.policies
            .write()
            .map(|mut policies| policies.remove(provider).is_some())
            .map_err(|error| {
                FlowError::Internal(format!(
                    "export activation policy registry lock poisoned: {error}"
                ))
            })
    }

    pub(crate) async fn evaluate(
        &self,
        provider: &str,
        request: ExportActivationRequest,
    ) -> Result<ExportActivationDecision> {
        let callback = self
            .policies
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
}
