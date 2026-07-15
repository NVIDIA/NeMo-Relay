// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::{Duration, Instant};

use crate::configuration::{GATEWAY_URL_ENV, TRANSPARENT_RUN_ENV};
use crate::error::CliError;

use super::HookForwardRequest;

const HOOK_GATEWAY_RETRY_TIMEOUT: Duration = Duration::from_secs(20);

pub(crate) struct HookDestination {
    pub(crate) gateway_url: String,
    pub(crate) lifecycle: HookGatewayLifecycle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HookGatewayLifecycle {
    /// A transparent run owns the dynamic gateway and passes its URL through the environment.
    Transparent,
    /// Persistent hooks use the authenticated gateway started and maintained by MCP.
    Existing,
}

// Installed hooks use the shared fixed gateway that MCP owns. Transparent runs set the dynamic
// environment URL and already own that gateway's lifecycle.
pub(super) fn hook_destination(command: &HookForwardRequest) -> HookDestination {
    resolve_hook_destination(
        command.gateway_url.clone(),
        std::env::var(GATEWAY_URL_ENV).ok(),
        command.forward_only,
        command.transparent_run,
    )
}

pub(super) fn transparent_run_active() -> bool {
    std::env::var(TRANSPARENT_RUN_ENV).ok().as_deref() == Some("1")
}

pub(crate) fn resolve_hook_destination(
    command_url: Option<String>,
    environment_url: Option<String>,
    forward_only: bool,
    transparent_run: bool,
) -> HookDestination {
    if transparent_run {
        return HookDestination {
            gateway_url: command_url
                .or(environment_url)
                .unwrap_or_else(|| crate::bootstrap::DEFAULT_URL.into()),
            lifecycle: HookGatewayLifecycle::Transparent,
        };
    }
    if forward_only {
        return HookDestination {
            gateway_url: command_url.unwrap_or_else(|| crate::bootstrap::DEFAULT_URL.into()),
            lifecycle: HookGatewayLifecycle::Existing,
        };
    }
    if let Some(gateway_url) = command_url {
        return HookDestination {
            gateway_url,
            lifecycle: HookGatewayLifecycle::Existing,
        };
    }
    if let Some(gateway_url) = environment_url {
        return HookDestination {
            gateway_url,
            lifecycle: HookGatewayLifecycle::Transparent,
        };
    }
    HookDestination {
        gateway_url: crate::bootstrap::DEFAULT_URL.into(),
        lifecycle: HookGatewayLifecycle::Existing,
    }
}

pub(super) async fn wait_for_existing_gateway(
    gateway: crate::bootstrap::GatewaySpec,
    gateway_url: String,
) -> Result<(), CliError> {
    tokio::task::spawn_blocking(move || {
        let deadline = Instant::now() + HOOK_GATEWAY_RETRY_TIMEOUT;
        loop {
            match gateway.existing_healthy_instance(&gateway_url) {
                Ok(Some(_instance_id)) => return Ok(()),
                Ok(None) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                Ok(None) => {
                    return Err(format!(
                        "no compatible Relay gateway became ready at {gateway_url}; ensure the host started `nemo-relay mcp`"
                    ));
                }
                Err(error) => return Err(error),
            }
        }
    })
    .await
    .map_err(|error| CliError::Launch(format!("hook gateway verification task failed: {error}")))?
    .map_err(CliError::Launch)
}

pub(super) fn recovery_plan(
    gateway_url: &str,
) -> Result<crate::bootstrap::PluginGatewaySpec, CliError> {
    let bind = crate::gateway::client::loopback_bind(gateway_url).map_err(CliError::Install)?;
    crate::bootstrap::resolve_plugin_gateway(&Default::default(), bind)
}

pub(crate) fn transparent_gateway_spec(
    gateway_url: &str,
) -> Result<crate::bootstrap::GatewaySpec, CliError> {
    let bind = crate::gateway::client::loopback_bind(gateway_url).map_err(CliError::Install)?;
    Ok(crate::bootstrap::GatewaySpec::new(bind).with_fingerprint(
        crate::configuration::transparent_gateway_fingerprint(gateway_url),
    ))
}
