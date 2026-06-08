// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

use nemo_relay::plugin::{PluginError, PluginRegistrationContext, Result as PluginResult};

use super::component::PiiRedactionConfig;

type LocalBackendProvider = Arc<
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

    match provider {
        Some(provider) => provider(config, ctx),
        None => Err(PluginError::RegistrationFailed(
            "PII redaction local-model backend is unavailable in this runtime".to_string(),
        )),
    }
}
