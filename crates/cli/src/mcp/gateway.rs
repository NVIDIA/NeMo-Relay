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
use crate::sidecar::{GatewayBootstrap, GatewaySpec};

const UNHEALTHY_CHECKS_BEFORE_RESTART: u8 = 3;

#[derive(Clone)]
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

    pub(super) async fn acquire(&self) -> Result<GatewayLease, CliError> {
        let bootstrap = ensure_gateway(self.spec.clone(), self.generation.clone()).await?;
        let plan = self.clone();
        let monitor = tokio::spawn(async move { plan.monitor(bootstrap.endpoint.url).await });
        Ok(GatewayLease { monitor })
    }

    async fn monitor(self, gateway_url: String) -> Result<(), CliError> {
        let health_spec = self.spec.clone();
        let restart_spec = self.spec.clone();
        let restart_generation = self.generation.clone();
        let verify_generation = self.generation;
        maintain_gateway_with_generation(
            self.spec.bind(),
            gateway_url,
            self.heartbeat_interval,
            move |url| {
                let spec = health_spec.clone();
                async move {
                    tokio::task::spawn_blocking(move || spec.is_healthy(&url))
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
        let monitor = tokio::spawn(async move { plan.monitor(gateway_url).await });
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
) -> Result<GatewayBootstrap, CliError> {
    tokio::task::spawn_blocking(move || {
        if let Some(generation) = generation.as_ref() {
            generation.verify_current()?;
        }
        let bootstrap = spec.ensure()?;
        if let Some(generation) = generation.as_ref() {
            verify_bootstrap_generation(generation)?;
        }
        Ok(bootstrap)
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
    RFuture: std::future::Future<Output = Result<GatewayBootstrap, CliError>>,
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

pub(super) async fn maintain_gateway_with_generation<H, HFuture, R, RFuture, G, GFuture>(
    bind: SocketAddr,
    mut gateway_url: String,
    heartbeat_interval: Duration,
    mut healthy: H,
    mut restart: R,
    mut verify_generation: G,
) -> Result<(), CliError>
where
    H: FnMut(String) -> HFuture,
    HFuture: std::future::Future<Output = Result<bool, CliError>>,
    R: FnMut(SocketAddr) -> RFuture,
    RFuture: std::future::Future<Output = Result<GatewayBootstrap, CliError>>,
    G: FnMut() -> GFuture,
    GFuture: std::future::Future<Output = Result<(), CliError>>,
{
    let mut heartbeat = tokio::time::interval(heartbeat_interval);
    let mut recovery = RecoveryState::default();
    heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    heartbeat.tick().await;
    loop {
        heartbeat.tick().await;
        verify_generation().await?;
        if healthy(gateway_url.clone()).await? {
            recovery.record_healthy();
            continue;
        }
        if !recovery.record_failure()? {
            continue;
        }
        verify_generation().await?;
        let bootstrap = restart(bind).await?;
        gateway_url = bootstrap.endpoint.url;
        recovery.record_recovery(bootstrap.started);
    }
}

#[derive(Default)]
struct RecoveryState {
    consecutive_failures: u8,
    restarted: bool,
}

impl RecoveryState {
    fn record_healthy(&mut self) {
        self.consecutive_failures = 0;
    }

    /// Returns true when the caller should coordinate recovery.
    fn record_failure(&mut self) -> Result<bool, CliError> {
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
        if self.consecutive_failures < UNHEALTHY_CHECKS_BEFORE_RESTART {
            return Ok(false);
        }
        if self.restarted {
            return Err(CliError::Launch(
                "shared Relay gateway became unhealthy after its coordinated restart".into(),
            ));
        }
        Ok(true)
    }

    fn record_recovery(&mut self, started: bool) {
        self.consecutive_failures = 0;
        self.restarted |= started;
    }
}

#[cfg(test)]
#[path = "../../tests/coverage/mcp_gateway_tests.rs"]
mod tests;
