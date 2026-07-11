// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Acquisition and liveness lease for a shared coding-agent gateway.

#[cfg(test)]
use std::ffi::OsString;
use std::net::SocketAddr;
use std::time::Duration;

use crate::config::CodingAgent;
use crate::config::ServerArgs;
use crate::error::CliError;
use crate::install_generation::InstallGeneration;
use crate::sidecar::{GatewayEndpoint, GatewaySpec};

const UNHEALTHY_CONFIRMATIONS: u8 = 3;
const UNHEALTHY_CONFIRMATION_INTERVAL: Duration = Duration::from_millis(50);

pub(super) struct GatewayPlan {
    spec: GatewaySpec,
    heartbeat_interval: Duration,
    generation: Option<InstallGeneration>,
}

impl GatewayPlan {
    pub(super) async fn resolve(
        agent: CodingAgent,
        server_args: &ServerArgs,
    ) -> Result<Self, CliError> {
        let generation = tokio::task::spawn_blocking(InstallGeneration::capture_from_env)
            .await
            .map_err(|error| {
                CliError::Launch(format!("MCP generation capture task failed: {error}"))
            })?
            .map_err(CliError::Launch)?;
        let bind = server_args.bind.unwrap_or_else(super::default_mcp_bind);
        let launch = crate::sidecar::resolve_plugin_gateway(agent, server_args, bind)?;
        let heartbeat_interval =
            crate::sidecar::plugin_heartbeat_interval().map_err(CliError::Launch)?;
        Ok(Self {
            spec: launch.gateway,
            heartbeat_interval,
            generation,
        })
    }

    pub(super) async fn acquire(self) -> Result<GatewayLease, CliError> {
        let endpoint = ensure_gateway(self.spec.clone(), self.generation.clone()).await?;
        let monitor = tokio::spawn(async move { self.monitor(endpoint).await });
        Ok(GatewayLease { monitor })
    }

    async fn monitor(self, endpoint: crate::sidecar::GatewayEndpoint) -> Result<(), CliError> {
        let health_spec = self.spec.clone();
        let restart_spec = self.spec.clone();
        let restart_generation = self.generation.clone();
        let verify_generation = self.generation;
        maintain_gateway_instances_with_generation(
            self.spec.bind(),
            endpoint,
            self.heartbeat_interval,
            move |url, _expected_instance| {
                let spec = health_spec.clone();
                async move {
                    tokio::task::spawn_blocking(move || spec.healthy_instance(&url))
                        .await
                        .map_err(|error| {
                            CliError::Launch(format!("gateway heartbeat task failed: {error}"))
                        })
                }
            },
            move |_bind| ensure_gateway(restart_spec.clone(), restart_generation.clone()),
            move || {
                let generation = verify_generation.clone();
                async move { verify_generation_async(generation).await }
            },
        )
        .await
    }

    #[cfg(test)]
    pub(super) fn test_lease(
        bind: SocketAddr,
        gateway_url: String,
        sidecar_args: Vec<OsString>,
        bootstrap_fingerprint: String,
        heartbeat_interval: Duration,
    ) -> GatewayLease {
        let plan = Self {
            spec: GatewaySpec::new(CodingAgent::Codex, bind)
                .with_launch_args(sidecar_args)
                .with_fingerprint(bootstrap_fingerprint),
            heartbeat_interval,
            generation: None,
        };
        let instance_id = plan
            .spec
            .healthy_instance(&gateway_url)
            .unwrap_or_else(|| "test-initial-instance".into());
        let endpoint = crate::sidecar::GatewayEndpoint {
            address: bind,
            url: gateway_url,
            instance_id,
        };
        let monitor = tokio::spawn(async move { plan.monitor(endpoint).await });
        GatewayLease { monitor }
    }
}

/// An active liveness lease. Dropping it stops heartbeats immediately.
pub(super) struct GatewayLease {
    monitor: tokio::task::JoinHandle<Result<(), CliError>>,
}

impl GatewayLease {
    pub(super) async fn wait(&mut self) -> Result<(), CliError> {
        (&mut self.monitor).await.map_err(|error| {
            CliError::Launch(format!("gateway maintenance task failed: {error}"))
        })?
    }
}

impl Drop for GatewayLease {
    fn drop(&mut self) {
        self.monitor.abort();
    }
}

async fn ensure_gateway(
    spec: GatewaySpec,
    generation: Option<InstallGeneration>,
) -> Result<GatewayEndpoint, CliError> {
    tokio::task::spawn_blocking(move || {
        if let Some(generation) = generation.as_ref() {
            generation.verify_current()?;
        }
        let endpoint = spec.ensure()?;
        if let Some(generation) = generation.as_ref() {
            verify_bootstrap_generation(generation)?;
        }
        Ok(endpoint)
    })
    .await
    .map_err(|error| CliError::Launch(format!("gateway bootstrap task failed: {error}")))?
    .map_err(CliError::Launch)
}

pub(super) fn verify_bootstrap_generation(generation: &InstallGeneration) -> Result<(), String> {
    // A replacement MCP may already be reusing a compatible gateway started between the two
    // generation checks. Leave a ready gateway for reuse or normal idle cleanup. Failures before
    // readiness remain armed and are terminated by the sidecar launcher.
    generation.verify_current()
}

async fn verify_generation_async(generation: Option<InstallGeneration>) -> Result<(), CliError> {
    tokio::task::spawn_blocking(move || {
        generation
            .as_ref()
            .map_or(Ok(()), InstallGeneration::verify_current)
    })
    .await
    .map_err(|error| CliError::Launch(format!("MCP generation verification task failed: {error}")))?
    .map_err(CliError::Launch)
}

#[cfg(test)]
pub(super) async fn maintain_gateway_with<H, HFuture, R, RFuture>(
    bind: SocketAddr,
    gateway_url: String,
    heartbeat_interval: Duration,
    healthy: H,
    restart: R,
) -> Result<(), CliError>
where
    H: FnMut(String) -> HFuture,
    HFuture: std::future::Future<Output = Result<bool, CliError>>,
    R: FnMut(SocketAddr) -> RFuture,
    RFuture: std::future::Future<Output = Result<GatewayEndpoint, CliError>>,
{
    maintain_gateway_with_generation(
        bind,
        gateway_url,
        heartbeat_interval,
        healthy,
        restart,
        || async { Ok(()) },
    )
    .await
}

#[cfg(test)]
pub(super) async fn maintain_gateway_with_generation<H, HFuture, R, RFuture, G, GFuture>(
    bind: SocketAddr,
    gateway_url: String,
    heartbeat_interval: Duration,
    mut healthy: H,
    restart: R,
    verify_generation: G,
) -> Result<(), CliError>
where
    H: FnMut(String) -> HFuture,
    HFuture: std::future::Future<Output = Result<bool, CliError>>,
    R: FnMut(SocketAddr) -> RFuture,
    RFuture: std::future::Future<Output = Result<GatewayEndpoint, CliError>>,
    G: FnMut() -> GFuture,
    GFuture: std::future::Future<Output = Result<(), CliError>>,
{
    maintain_gateway_instances_with_generation(
        bind,
        crate::sidecar::GatewayEndpoint {
            address: bind,
            url: gateway_url,
            instance_id: "test-initial-instance".into(),
        },
        heartbeat_interval,
        move |url, expected_instance| {
            let probe = healthy(url);
            async move {
                probe
                    .await
                    .map(|is_healthy| is_healthy.then_some(expected_instance))
            }
        },
        restart,
        verify_generation,
    )
    .await
}

async fn maintain_gateway_instances_with_generation<H, HFuture, R, RFuture, G, GFuture>(
    bind: SocketAddr,
    mut endpoint: crate::sidecar::GatewayEndpoint,
    heartbeat_interval: Duration,
    mut healthy: H,
    mut restart: R,
    mut verify_generation: G,
) -> Result<(), CliError>
where
    H: FnMut(String, String) -> HFuture,
    HFuture: std::future::Future<Output = Result<Option<String>, CliError>>,
    R: FnMut(SocketAddr) -> RFuture,
    RFuture: std::future::Future<Output = Result<GatewayEndpoint, CliError>>,
    G: FnMut() -> GFuture,
    GFuture: std::future::Future<Output = Result<(), CliError>>,
{
    let mut heartbeat = tokio::time::interval(heartbeat_interval);
    let mut recovery = RecoveryState::new(endpoint.instance_id.clone());
    heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    heartbeat.tick().await;
    loop {
        heartbeat.tick().await;
        verify_generation().await?;
        let mut observed_instance = None;
        for confirmation in 0..UNHEALTHY_CONFIRMATIONS {
            if confirmation > 0 {
                tokio::time::sleep(UNHEALTHY_CONFIRMATION_INTERVAL).await;
                verify_generation().await?;
            }
            observed_instance =
                healthy(endpoint.url.clone(), recovery.instance_id().into()).await?;
            if observed_instance.is_some() {
                break;
            }
        }
        if let Some(instance_id) = observed_instance {
            recovery.observe(instance_id)?;
            continue;
        }
        recovery.require_restart()?;
        verify_generation().await?;
        let recovered = restart(bind).await?;
        recovery.observe(recovered.instance_id.clone())?;
        endpoint = recovered;
    }
}

struct RecoveryState {
    instance_id: String,
    recovered: bool,
}

impl RecoveryState {
    fn new(instance_id: String) -> Self {
        Self {
            instance_id,
            recovered: false,
        }
    }

    fn instance_id(&self) -> &str {
        &self.instance_id
    }

    fn observe(&mut self, instance_id: String) -> Result<(), CliError> {
        if instance_id == self.instance_id {
            return Ok(());
        }
        if self.recovered {
            return Err(CliError::Launch(
                "shared Relay gateway was replaced again after its coordinated restart".into(),
            ));
        }
        self.instance_id = instance_id;
        self.recovered = true;
        Ok(())
    }

    fn require_restart(&self) -> Result<(), CliError> {
        if self.recovered {
            Err(CliError::Launch(
                "shared Relay gateway became unhealthy after its coordinated restart".into(),
            ))
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
#[path = "../../tests/coverage/mcp_gateway_tests.rs"]
mod tests;
