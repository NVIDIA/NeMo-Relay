// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Activation-time hooks for plugin-managed export targets.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use crate::error::{FlowError, Result};
pub use nemo_relay_types::plugin::{
    ExportActivationDecision, ExportActivationPolicyConfig, ExportActivationRequest,
    ExportActivationTargetKind, ExportTargetRegistration, MAX_EXPORT_ACTIVATION_TIMEOUT_MILLIS,
    MIN_EXPORT_ACTIVATION_TIMEOUT_MILLIS,
};

/// Asynchronous callback registered by one export-activation policy provider.
pub type ExportActivationPolicyFn = Arc<
    dyn Fn(
            ExportActivationRequest,
        ) -> Pin<Box<dyn Future<Output = Result<ExportActivationDecision>> + Send>>
        + Send
        + Sync,
>;

/// Deferred constructor for one local or remote export target.
pub type ExportTargetActivationFn =
    Arc<dyn Fn() -> Pin<Box<dyn Future<Output = Result<()>> + Send>> + Send + Sync>;

struct PendingExportTarget {
    qualified_id: String,
    registration: ExportTargetRegistration,
    activate: ExportTargetActivationFn,
}

/// Policy callbacks and pending export targets owned by one plugin activation.
#[derive(Default)]
#[doc(hidden)]
pub struct ExportActivationRegistry {
    policies: RwLock<HashMap<String, ExportActivationPolicyFn>>,
    targets: Mutex<Vec<PendingExportTarget>>,
    active_targets: Mutex<Vec<PendingExportTarget>>,
}

#[doc(hidden)]
pub type ExportActivationPolicyRegistry = ExportActivationRegistry;

impl ExportActivationRegistry {
    /// Registers one provider callback for this activation.
    pub fn register(&self, provider: &str, callback: ExportActivationPolicyFn) -> Result<()> {
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

    /// Removes one provider callback, returning whether it existed.
    pub fn deregister(&self, provider: &str) -> Result<bool> {
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

    /// Adds one deferred exporter constructor under an activation-unique ID.
    pub fn register_target(
        &self,
        qualified_id: String,
        registration: ExportTargetRegistration,
        activate: ExportTargetActivationFn,
    ) -> Result<()> {
        validate_target_registration(&registration)?;
        let mut targets = self.targets.lock().map_err(|error| {
            FlowError::Internal(format!("export target registry lock poisoned: {error}"))
        })?;
        if targets
            .iter()
            .any(|target| target.qualified_id == qualified_id)
        {
            return Err(FlowError::AlreadyExists(qualified_id));
        }
        targets.push(PendingExportTarget {
            qualified_id,
            registration,
            activate,
        });
        Ok(())
    }

    /// Removes one pending exporter constructor, returning whether it existed.
    pub fn deregister_target(&self, qualified_id: &str) -> Result<bool> {
        let remove = |targets: &Mutex<Vec<PendingExportTarget>>| -> Result<bool> {
            let mut targets = targets.lock().map_err(|error| {
                FlowError::Internal(format!("export target registry lock poisoned: {error}"))
            })?;
            let original_len = targets.len();
            targets.retain(|target| target.qualified_id != qualified_id);
            Ok(targets.len() != original_len)
        };
        let pending = remove(&self.targets)?;
        let active = remove(&self.active_targets)?;
        Ok(pending || active)
    }

    pub(crate) async fn activate_targets(&self) -> Result<()> {
        let targets = {
            let mut targets = self.targets.lock().map_err(|error| {
                FlowError::Internal(format!("export target registry lock poisoned: {error}"))
            })?;
            std::mem::take(&mut *targets)
        };
        for target in targets {
            if self
                .target_allowed(&target.qualified_id, &target.registration)
                .await
            {
                (target.activate)().await?;
                self.active_targets
                    .lock()
                    .map_err(|error| {
                        FlowError::Internal(format!(
                            "active export target registry lock poisoned: {error}"
                        ))
                    })?
                    .push(target);
            }
        }
        Ok(())
    }

    async fn target_allowed(
        &self,
        qualified_id: &str,
        registration: &ExportTargetRegistration,
    ) -> bool {
        let Some(policy) = &registration.activation_policy else {
            return true;
        };
        let request = ExportActivationRequest {
            target_kind: registration.target_kind.clone(),
            config: policy.config.clone(),
        };
        let outcome = tokio::time::timeout(
            export_activation_timeout(policy),
            self.evaluate(&policy.provider, request),
        )
        .await;
        let (allowed, reason) = match outcome {
            Ok(Ok(ExportActivationDecision::Allow)) => (true, "allowed"),
            Ok(Ok(ExportActivationDecision::Deny)) => (false, "denied"),
            Ok(Err(FlowError::NotFound(_))) => (false, "provider_unavailable"),
            Ok(Err(_)) => (false, "provider_error"),
            Err(_) => (false, "timeout"),
        };
        if !allowed {
            log::warn!(
                target: "nemo_relay.plugin",
                event = "export_activation_policy_denied",
                provider = policy.provider.as_str(),
                target_kind = registration.target_kind.as_str(),
                target_id = qualified_id,
                reason;
                "Export target suppressed by activation policy"
            );
        }
        allowed
    }
}

fn validate_target_registration(registration: &ExportTargetRegistration) -> Result<()> {
    if registration.id.trim().is_empty() || registration.id.trim() != registration.id {
        return Err(FlowError::InvalidArgument(
            "export target id must be nonblank and have no surrounding whitespace".into(),
        ));
    }
    if let Some(policy) = &registration.activation_policy {
        if policy.provider.trim().is_empty() || policy.provider.trim() != policy.provider {
            return Err(FlowError::InvalidArgument(
                "export activation provider must be nonblank and have no surrounding whitespace"
                    .into(),
            ));
        }
        if !(MIN_EXPORT_ACTIVATION_TIMEOUT_MILLIS..=MAX_EXPORT_ACTIVATION_TIMEOUT_MILLIS)
            .contains(&policy.timeout_millis)
        {
            return Err(FlowError::InvalidArgument(format!(
                "export activation timeout_millis must be between {MIN_EXPORT_ACTIVATION_TIMEOUT_MILLIS} and {MAX_EXPORT_ACTIVATION_TIMEOUT_MILLIS}"
            )));
        }
    }
    Ok(())
}

fn export_activation_timeout(policy: &ExportActivationPolicyConfig) -> Duration {
    Duration::from_millis(policy.timeout_millis.clamp(
        MIN_EXPORT_ACTIVATION_TIMEOUT_MILLIS,
        MAX_EXPORT_ACTIVATION_TIMEOUT_MILLIS,
    ))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use serde_json::json;

    use super::*;

    fn registration(id: &str, provider: Option<&str>, allow: bool) -> ExportTargetRegistration {
        ExportTargetRegistration {
            id: id.into(),
            target_kind: ExportActivationTargetKind::new("example.telemetry.exporter").unwrap(),
            activation_policy: provider.map(|provider| ExportActivationPolicyConfig {
                provider: provider.into(),
                timeout_millis: 30_000,
                config: json!({"allow": allow}),
            }),
        }
    }

    #[tokio::test]
    async fn self_provider_gates_deferred_targets_once_and_missing_provider_denies() {
        let registry = ExportActivationRegistry::default();
        let policy_calls = Arc::new(AtomicUsize::new(0));
        let policy_calls_callback = Arc::clone(&policy_calls);
        registry
            .register(
                "example.plugin",
                Arc::new(move |request| {
                    let policy_calls = Arc::clone(&policy_calls_callback);
                    Box::pin(async move {
                        policy_calls.fetch_add(1, Ordering::SeqCst);
                        Ok(if request.config["allow"] == true {
                            ExportActivationDecision::Allow
                        } else {
                            ExportActivationDecision::Deny
                        })
                    })
                }),
            )
            .unwrap();
        registry
            .register(
                "error.plugin",
                Arc::new(|_| Box::pin(async { Err(FlowError::Internal("policy failed".into())) })),
            )
            .unwrap();

        let activations = Arc::new(AtomicUsize::new(0));
        for (id, provider, allow) in [
            ("ungated", None, true),
            ("allowed", Some("example.plugin"), true),
            ("denied", Some("example.plugin"), false),
            ("missing", Some("missing.plugin"), true),
            ("error", Some("error.plugin"), true),
        ] {
            let activations = Arc::clone(&activations);
            registry
                .register_target(
                    format!("example:{id}"),
                    registration(id, provider, allow),
                    Arc::new(move || {
                        let activations = Arc::clone(&activations);
                        Box::pin(async move {
                            activations.fetch_add(1, Ordering::SeqCst);
                            Ok(())
                        })
                    }),
                )
                .unwrap();
        }

        registry.activate_targets().await.unwrap();
        registry.activate_targets().await.unwrap();

        assert_eq!(policy_calls.load(Ordering::SeqCst), 2);
        assert_eq!(activations.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn target_kind_deserialization_rejects_non_namespaced_values() {
        assert!(serde_json::from_value::<ExportActivationTargetKind>(json!("otlp_trace")).is_err());
        assert_eq!(
            serde_json::from_value::<ExportActivationTargetKind>(json!("example.otlp.trace"))
                .unwrap()
                .as_str(),
            "example.otlp.trace"
        );

        let registry = ExportActivationRegistry::default();
        registry
            .register_target(
                "example:duplicate".into(),
                registration("duplicate", None, true),
                Arc::new(|| Box::pin(async { Ok(()) })),
            )
            .unwrap();
        assert!(
            registry
                .register_target(
                    "example:duplicate".into(),
                    registration("duplicate", None, true),
                    Arc::new(|| Box::pin(async { Ok(()) })),
                )
                .is_err()
        );
    }
}
