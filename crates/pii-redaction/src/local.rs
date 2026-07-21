// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

use nemo_relay::plugin::{
    PluginError, PluginRegistrationContext, Result as PluginResult, rollback_registrations,
};

use super::component::PiiRedactionConfig;
use super::component::profile_registration_prefix;

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
    profile_name: Option<&str>,
) -> PluginResult<()> {
    let provider = local_backend_provider_guard()?.clone();

    let Some(provider) = provider else {
        log::warn!(
            target: "nemo_relay.plugin",
            event = "plugin_resource_access_failed",
            plugin_kind = "pii_redaction",
            profile = profile_name.unwrap_or("legacy"),
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
        profile = profile_name.unwrap_or("legacy"),
        resource_kind = "local_model_backend",
        permission = "execute";
        "Plugin resource access validation started"
    );
    let mut scoped_context = profile_name.map(|profile_name| {
        PluginRegistrationContext::with_namespace(
            ctx.qualify_name(&format!("{}/", profile_registration_prefix(profile_name))),
        )
    });
    let provider_context = scoped_context.as_mut().unwrap_or(ctx);
    match provider(config, provider_context) {
        Ok(()) => {
            if let Some(scoped_context) = scoped_context {
                ctx.extend_registrations(scoped_context.into_registrations());
            }
            log::info!(
                target: "nemo_relay.plugin",
                event = "plugin_resource_access_validated",
                plugin_kind = "pii_redaction",
                profile = profile_name.unwrap_or("legacy"),
                resource_kind = "local_model_backend",
                permission = "execute";
                "Plugin resource access validated"
            );
            Ok(())
        }
        Err(error) => {
            if let Some(scoped_context) = scoped_context {
                let mut registrations = scoped_context.into_registrations();
                rollback_registrations(&mut registrations);
            }
            log::warn!(
                target: "nemo_relay.plugin",
                event = "plugin_resource_access_failed",
                plugin_kind = "pii_redaction",
                profile = profile_name.unwrap_or("legacy"),
                resource_kind = "local_model_backend",
                permission = "execute",
                reason = "initialization_failed";
                "Plugin resource access validation failed"
            );
            Err(error)
        }
    }
}
