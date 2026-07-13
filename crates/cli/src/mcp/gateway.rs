// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Acquisition and liveness lease for a shared coding-agent gateway.

use std::net::SocketAddr;
use std::time::Duration;

use crate::bootstrap::{GatewayEndpoint, GatewaySpec};
use crate::error::CliError;
use crate::installation::generation::{ActiveGenerationGuard, InstallGeneration};
use crate::server::GatewayOverrides;

const UNHEALTHY_CONFIRMATIONS: u8 = 3;
const UNHEALTHY_CONFIRMATION_INTERVAL: Duration = Duration::from_millis(50);
const BORROWED_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(30);

pub(super) struct GatewayPlan {
    spec: GatewaySpec,
    heartbeat_interval: Duration,
    generation: Option<InstallGeneration>,
    generation_guard: Option<ActiveGenerationGuard>,
}

impl GatewayPlan {
    pub(super) async fn resolve(server_args: &GatewayOverrides) -> Result<Self, CliError> {
        let captured = tokio::task::spawn_blocking(InstallGeneration::capture_guarded_from_env)
            .await
            .map_err(|error| {
                CliError::Launch(format!("MCP generation capture task failed: {error}"))
            })?
            .map_err(CliError::Launch)?;
        let (generation, generation_guard) = captured
            .map(|(generation, guard)| (Some(generation), Some(guard)))
            .unwrap_or((None, None));
        let bind = server_args.bind.unwrap_or_else(super::default_mcp_bind);
        let launch = crate::bootstrap::resolve_plugin_gateway(server_args, bind)?;
        let heartbeat_interval =
            crate::bootstrap::plugin_heartbeat_interval().map_err(CliError::Launch)?;
        Ok(Self {
            spec: launch.gateway,
            heartbeat_interval,
            generation,
            generation_guard,
        })
    }

    pub(super) async fn acquire(mut self) -> Result<GatewayLease, CliError> {
        let endpoint = acquire_gateway(self.spec.clone(), self.generation_guard.take()).await?;
        let monitor = tokio::spawn(async move { self.monitor(endpoint).await });
        Ok(GatewayLease { monitor })
    }

    async fn monitor(self, endpoint: crate::bootstrap::GatewayEndpoint) -> Result<(), CliError> {
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
            move |_bind, expected_instance| {
                recover_gateway(
                    restart_spec.clone(),
                    restart_generation.clone(),
                    expected_instance,
                )
            },
            move || {
                let generation = verify_generation.clone();
                async move { verify_lifecycle_async(generation).await }
            },
        )
        .await
    }
}

/// An active liveness lease. Dropping it stops heartbeats immediately.
pub(super) struct GatewayLease {
    monitor: tokio::task::JoinHandle<Result<(), CliError>>,
}

impl GatewayLease {
    #[cfg(test)]
    pub(super) fn test_pending() -> Self {
        let monitor = tokio::spawn(std::future::pending::<Result<(), CliError>>());
        Self { monitor }
    }

    pub(super) async fn borrow(
        gateway_url: String,
        bootstrap_fingerprint: String,
    ) -> Result<Self, CliError> {
        Self::borrow_with_interval(
            gateway_url,
            bootstrap_fingerprint,
            BORROWED_HEARTBEAT_INTERVAL,
        )
        .await
    }

    pub(super) async fn borrow_with_interval(
        gateway_url: String,
        bootstrap_fingerprint: String,
        heartbeat_interval: Duration,
    ) -> Result<Self, CliError> {
        let expected_instance = authenticated_instance_id(
            gateway_url.clone(),
            bootstrap_fingerprint.clone(),
        )
        .await?
        .ok_or_else(|| {
            CliError::Launch(format!(
                "{} does not identify the authenticated NeMo Relay gateway owned by this transparent run",
                crate::configuration::GATEWAY_URL_ENV
            ))
        })?;
        let monitor = tokio::spawn(async move {
            loop {
                tokio::time::sleep(heartbeat_interval).await;
                let current =
                    authenticated_instance_id(gateway_url.clone(), bootstrap_fingerprint.clone())
                        .await?;
                match current {
                    Some(instance) if instance == expected_instance => {}
                    Some(instance) => {
                        return Err(CliError::Launch(format!(
                            "transparent Relay gateway instance changed from {expected_instance} to {instance}"
                        )));
                    }
                    None => {
                        return Err(CliError::Launch(format!(
                            "transparent Relay gateway at {gateway_url} is no longer available"
                        )));
                    }
                }
            }
        });
        Ok(Self { monitor })
    }

    pub(super) async fn wait(&mut self) -> Result<(), CliError> {
        (&mut self.monitor).await.map_err(|error| {
            CliError::Launch(format!("gateway maintenance task failed: {error}"))
        })?
    }
}

async fn authenticated_instance_id(
    gateway_url: String,
    bootstrap_fingerprint: String,
) -> Result<Option<String>, CliError> {
    tokio::task::spawn_blocking(move || {
        crate::gateway::client::authenticated_instance_id(&gateway_url, &bootstrap_fingerprint)
    })
    .await
    .map_err(|error| {
        CliError::Launch(format!(
            "transparent gateway verification task failed: {error}"
        ))
    })
}

impl Drop for GatewayLease {
    fn drop(&mut self) {
        self.monitor.abort();
    }
}

async fn acquire_gateway(
    spec: GatewaySpec,
    generation_guard: Option<ActiveGenerationGuard>,
) -> Result<GatewayEndpoint, CliError> {
    tokio::task::spawn_blocking(move || {
        let _generation_guard = generation_guard;
        spec.acquire()
    })
    .await
    .map_err(|error| CliError::Launch(format!("gateway bootstrap task failed: {error}")))?
    .map_err(CliError::Launch)
}

async fn recover_gateway(
    spec: GatewaySpec,
    generation: Option<InstallGeneration>,
    expected_instance: String,
) -> Result<GatewayEndpoint, CliError> {
    tokio::task::spawn_blocking(move || {
        let _generation_guard = generation
            .as_ref()
            .map(InstallGeneration::guard_current)
            .transpose()?;
        spec.recover(&expected_instance)
    })
    .await
    .map_err(|error| CliError::Launch(format!("gateway recovery task failed: {error}")))?
    .map_err(CliError::Launch)
}

async fn verify_lifecycle_async(generation: Option<InstallGeneration>) -> Result<(), CliError> {
    loop {
        let generation = generation.clone();
        let current = tokio::task::spawn_blocking(move || {
            if let Some(generation) = generation.as_ref()
                && !generation.try_verify_current()?
            {
                return Ok(false);
            }
            Ok(true)
        })
        .await
        .map_err(|error| {
            CliError::Launch(format!("MCP lifecycle verification task failed: {error}"))
        })?
        .map_err(CliError::Launch)?;
        if current {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
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
    R: FnMut(SocketAddr, String) -> RFuture,
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
    R: FnMut(SocketAddr, String) -> RFuture,
    RFuture: std::future::Future<Output = Result<GatewayEndpoint, CliError>>,
    G: FnMut() -> GFuture,
    GFuture: std::future::Future<Output = Result<(), CliError>>,
{
    maintain_gateway_instances_with_generation(
        bind,
        crate::bootstrap::GatewayEndpoint {
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
    mut endpoint: crate::bootstrap::GatewayEndpoint,
    heartbeat_interval: Duration,
    mut healthy: H,
    mut restart: R,
    mut verify_generation: G,
) -> Result<(), CliError>
where
    H: FnMut(String, String) -> HFuture,
    HFuture: std::future::Future<Output = Result<Option<String>, CliError>>,
    R: FnMut(SocketAddr, String) -> RFuture,
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
            // The health probe can queue or block while an integration replacement rotates the
            // endpoint cohort. Revalidate before accepting either its instance or its failure so
            // an old client cannot adopt the replacement across that asynchronous gap.
            verify_generation().await?;
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
        let recovered = restart(bind, recovery.instance_id().into()).await?;
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
#[path = "../../tests/coverage/shared/mcp_gateway_tests.rs"]
mod tests;
