// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

use nemo_relay::plugin::{PluginError, PluginRegistrationContext, Result as PluginResult};

use super::component::PiiRedactionConfig;

#[doc(hidden)]
pub type LocalBackendProvider = Arc<
    dyn Fn(PiiRedactionConfig, &mut PluginRegistrationContext) -> PluginResult<()> + Send + Sync,
>;

static LOCAL_BACKEND_PROVIDER: LazyLock<Mutex<Option<LocalBackendProvider>>> =
    LazyLock::new(|| Mutex::new(None));

fn local_backend_provider_guard() -> PluginResult<MutexGuard<'static, Option<LocalBackendProvider>>>
{
    LOCAL_BACKEND_PROVIDER.lock().map_err(|e| {
        PluginError::Internal(format!(
            "PII redaction local backend provider lock poisoned: {e}"
        ))
    })
}

#[doc(hidden)]
pub fn register_local_backend_provider(provider: LocalBackendProvider) -> PluginResult<()> {
    let mut guard = local_backend_provider_guard()?;
    *guard = Some(provider);
    Ok(())
}

#[doc(hidden)]
pub fn clear_local_backend_provider() -> PluginResult<()> {
    let mut guard = local_backend_provider_guard()?;
    *guard = None;
    Ok(())
}

pub(super) fn register_local_backend(
    config: PiiRedactionConfig,
    ctx: &mut PluginRegistrationContext,
) -> PluginResult<()> {
    let provider = local_backend_provider_guard()?.clone();

    let Some(provider) = provider else {
        log::warn!(
            target: "nemo_relay.plugin",
            event = "plugin_resource_access_failed",
            plugin_kind = "pii_redaction",
            resource_kind = "local_model_backend",
            permission = "execute",
            reason = "provider_unavailable";
            "Plugin resource access validation failed"
        );
        return Err(PluginError::RegistrationFailed(
            "PII redaction local-model backend is unavailable in this runtime".to_string(),
        ));
    };
    log::info!(
        target: "nemo_relay.plugin",
        event = "plugin_resource_access_pending",
        plugin_kind = "pii_redaction",
        resource_kind = "local_model_backend",
        permission = "execute";
        "Plugin resource access validation started"
    );
    match provider(config, ctx) {
        Ok(()) => {
            log::info!(
                target: "nemo_relay.plugin",
                event = "plugin_resource_access_validated",
                plugin_kind = "pii_redaction",
                resource_kind = "local_model_backend",
                permission = "execute";
                "Plugin resource access validated"
            );
            Ok(())
        }
        Err(error) => {
            log::warn!(
                target: "nemo_relay.plugin",
                event = "plugin_resource_access_failed",
                plugin_kind = "pii_redaction",
                resource_kind = "local_model_backend",
                permission = "execute",
                reason = "initialization_failed";
                "Plugin resource access validation failed"
            );
            Err(error)
        }
    }
}
