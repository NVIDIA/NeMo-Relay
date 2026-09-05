// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Broker-attached MCP stdio process. It advertises no MCP tools.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::process::Stdio;
use std::time::Duration;

use reqwest::Client;
use tokio::io::AsyncWriteExt;
use tokio::net::UdpSocket;
use tokio::process::{Child, Command};

use super::common::address::daemon_url;
use super::common::client::{
    ControlRetryPolicy, begin_handshake, control_client, post_empty_idempotent, post_json,
    post_json_idempotent,
};
use super::common::control::{
    ACTIVATION_LIFETIME_MS, ActivationFailedPayload, EmptyPayload, MCP_ACTIVATION_FAILED_PATH,
    MCP_HEARTBEAT_INTERVAL_MS, MCP_HEARTBEAT_PATH, MCP_LEASE_MS, MCP_REGISTER_PATH,
    McpHeartbeatResponse, McpRegisterRequest, McpRegisterResponse, SessionRequest, WorkerBootstrap,
    WorkerNetworkHint, WorkerNetworkHintProof,
};
use super::common::identity::MachineIdentity;
use super::common::protocol::{BrokerDirective, ComponentRole, SensitiveString};
use super::common::state::{ROUTE_TOKEN_ENV, RouteCredential, load_or_create_machine_identity};
use crate::error::CliError;

// Includes the full two-minute legal drain plus reconciliation margin before a replacement
// activation is issued.
const ACTIVATION_POLL_MAX: Duration = Duration::from_secs(150);
const REGISTRATION_RETRY_MAX: Duration = Duration::from_secs(30);
const REGISTRATION_RETRY_DELAY: Duration = Duration::from_millis(100);
const MIN_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(1);
const MAX_HEARTBEAT_INTERVAL: Duration = Duration::from_millis(MCP_LEASE_MS / 3);
const HEARTBEAT_RETRY_WINDOW_MS: u64 = MCP_LEASE_MS - MCP_HEARTBEAT_INTERVAL_MS - 5_000;
const HEARTBEAT_RETRY_POLICY: ControlRetryPolicy = ControlRetryPolicy::new(
    Duration::from_secs(2),
    Duration::from_millis(HEARTBEAT_RETRY_WINDOW_MS),
    Duration::from_millis(250),
);
const RELEASE_RETRY_POLICY: ControlRetryPolicy = ControlRetryPolicy::new(
    Duration::from_millis(500),
    Duration::from_secs(2),
    Duration::from_millis(100),
);
const WORKER_ADVERTISE_ENV: &str = "NEMO_RELAY_WORKER_ADVERTISE_ADDRESS";
const WORKER_PORT_ENV: &str = "NEMO_RELAY_WORKER_PORT";

#[derive(Debug, Clone)]
pub(crate) struct Options {
    pub(crate) daemon_address: String,
}

struct McpLease {
    client: Client,
    daemon_origin: String,
    route_credential: RouteCredential,
    identity: MachineIdentity,
    session_id: String,
    session_token: SensitiveString,
    heartbeat_interval: Duration,
    sequence: u64,
    pending_heartbeat: Option<SessionRequest<EmptyPayload>>,
}

struct Registration {
    directive: BrokerDirective,
    session_token: SensitiveString,
    heartbeat_interval: Duration,
}

pub(crate) async fn run(options: Options) -> Result<(), CliError> {
    let daemon = daemon_url(&options.daemon_address)?;
    let daemon_origin = daemon.as_str().trim_end_matches('/').to_owned();
    let client = control_client()?;
    let route_credential = RouteCredential::from_environment()?;
    let identity = load_or_create_machine_identity()?;
    let session_id = uuid::Uuid::now_v7().to_string();
    let registration = register(
        &client,
        &daemon_origin,
        &route_credential,
        &identity,
        &session_id,
    )
    .await?;
    let mut lease = McpLease {
        client,
        daemon_origin,
        route_credential,
        identity,
        session_id,
        session_token: registration.session_token,
        heartbeat_interval: registration.heartbeat_interval,
        sequence: 0,
        pending_heartbeat: None,
    };
    make_route_ready(&mut lease, registration.directive).await?;

    log::info!(
        target: "nemo_relay.daemon.mcp",
        event = "daemon_mcp_ready";
        "Broker reference acquired; MCP protocol is ready"
    );
    let result = {
        let protocol = crate::mcp::serve_daemon_stdio();
        let control = maintain_lease(&mut lease);
        tokio::pin!(protocol);
        tokio::pin!(control);
        tokio::select! {
            result = &mut protocol => result,
            result = &mut control => result,
        }
    };
    release(&mut lease).await;
    result
}

async fn register(
    client: &Client,
    daemon_origin: &str,
    credential: &RouteCredential,
    identity: &MachineIdentity,
    session_id: &str,
) -> Result<Registration, CliError> {
    let deadline = tokio::time::Instant::now() + REGISTRATION_RETRY_MAX;
    loop {
        match register_once(client, daemon_origin, credential, identity, session_id).await {
            Ok(registration) => return Ok(registration),
            Err(error @ CliError::Upstream(_)) => {
                let now = tokio::time::Instant::now();
                if now >= deadline {
                    return Err(error);
                }
                tokio::time::sleep_until(deadline.min(now + REGISTRATION_RETRY_DELAY)).await;
            }
            Err(error) => return Err(error),
        }
    }
}

async fn register_once(
    client: &Client,
    daemon_origin: &str,
    credential: &RouteCredential,
    identity: &MachineIdentity,
    session_id: &str,
) -> Result<Registration, CliError> {
    let handshake = begin_handshake(
        client,
        daemon_origin,
        ComponentRole::Mcp,
        identity,
        session_id,
        Some(credential.digest()),
    )
    .await?;
    let worker_network = worker_network_hint(daemon_origin).await?;
    let worker_network = WorkerNetworkHintProof::sign(
        worker_network,
        &handshake.proof.transcript.daemon_target,
        session_id,
        &handshake.proof.transcript.challenge_id,
        &identity.fingerprint(),
        identity,
    )?;
    let response: McpRegisterResponse = post_json(
        client,
        &format!("{daemon_origin}{MCP_REGISTER_PATH}"),
        &McpRegisterRequest {
            proof: handshake.proof.clone(),
            worker_network,
        },
        Some(credential.expose()),
    )
    .await?;
    handshake.authenticate_daemon(&response.daemon_proof)?;
    Ok(Registration {
        directive: response.directive,
        session_token: response.session_token,
        heartbeat_interval: validate_heartbeat_interval(response.heartbeat_interval_ms)?,
    })
}

async fn worker_network_hint(daemon_origin: &str) -> Result<WorkerNetworkHint, CliError> {
    let advertised_override = optional_environment(WORKER_ADVERTISE_ENV)?;
    let port_override = optional_environment(WORKER_PORT_ENV)?;
    let (advertised_override, port) =
        parse_worker_network_overrides(advertised_override.as_deref(), port_override.as_deref())?;
    let daemon = daemon_url(daemon_origin)?;
    let daemon_addresses = tokio::net::lookup_host((
        daemon
            .host_str()
            .ok_or_else(|| CliError::Config("daemon address is missing a host".into()))?,
        daemon
            .port()
            .ok_or_else(|| CliError::Config("daemon address is missing a port".into()))?,
    ))
    .await
    .map_err(|error| CliError::Launch(format!("failed to resolve daemon IPv4 route: {error}")))?
    .filter_map(|address| match address {
        SocketAddr::V4(address) => Some(address),
        SocketAddr::V6(_) => None,
    })
    .collect::<Vec<_>>();
    let daemon_address = daemon_addresses
        .iter()
        .copied()
        .find(|address| !address.ip().is_loopback())
        .or_else(|| daemon_addresses.first().copied())
        .ok_or_else(|| {
            CliError::Config(
                "daemon target has no IPv4 route; daemon workers support IPv4 networking only"
                    .into(),
            )
        })?;
    let advertised_host = match advertised_override {
        Some(address) => address,
        None if daemon_address.ip().is_loopback() => Ipv4Addr::LOCALHOST.to_string(),
        None => {
            let socket = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0)).await?;
            socket.connect(daemon_address).await?;
            match socket.local_addr()?.ip() {
                IpAddr::V4(address) if !address.is_unspecified() => address.to_string(),
                _ => {
                    return Err(CliError::Launch(
                        "failed to determine a concrete local IPv4 route to the daemon".into(),
                    ));
                }
            }
        }
    };
    let advertised_is_loopback = advertised_host.eq_ignore_ascii_case("localhost")
        || advertised_host
            .parse::<Ipv4Addr>()
            .is_ok_and(|address| address.is_loopback());
    if !daemon_address.ip().is_loopback() && advertised_is_loopback {
        return Err(CliError::Config(format!(
            "{WORKER_ADVERTISE_ENV} cannot be loopback for a remote daemon"
        )));
    }
    WorkerNetworkHint::new(advertised_host, port)
}

fn parse_worker_network_overrides(
    advertised: Option<&str>,
    port: Option<&str>,
) -> Result<(Option<String>, Option<u16>), CliError> {
    let advertised = advertised
        .map(str::trim)
        .map(|value| {
            WorkerNetworkHint::new(value, None)
                .map(|hint| hint.advertised_host)
                .map_err(|_| {
                    CliError::Config(format!(
                        "{WORKER_ADVERTISE_ENV} must be a concrete hostname or IPv4 address"
                    ))
                })
        })
        .transpose()?;
    let port = port
        .map(str::trim)
        .map(|value| {
            value
                .parse::<u16>()
                .ok()
                .filter(|port| *port != 0)
                .ok_or_else(|| {
                    CliError::Config(format!(
                        "{WORKER_PORT_ENV} must be an integer between 1 and 65535"
                    ))
                })
        })
        .transpose()?;
    Ok((advertised, port))
}

fn optional_environment(name: &str) -> Result<Option<String>, CliError> {
    std::env::var_os(name)
        .map(|value| {
            value
                .into_string()
                .map_err(|_| CliError::Config(format!("{name} must contain valid Unicode text")))
        })
        .transpose()
}

fn validate_heartbeat_interval(milliseconds: u64) -> Result<Duration, CliError> {
    let interval = Duration::from_millis(milliseconds);
    if !(MIN_HEARTBEAT_INTERVAL..=MAX_HEARTBEAT_INTERVAL).contains(&interval) {
        return Err(CliError::Unauthorized(
            "daemon returned an invalid MCP heartbeat interval".into(),
        ));
    }
    Ok(interval)
}

async fn make_route_ready(
    lease: &mut McpLease,
    mut directive: BrokerDirective,
) -> Result<(), CliError> {
    let started = tokio::time::Instant::now();
    let mut launched: Option<(String, Child, tokio::time::Instant)> = None;
    loop {
        match directive {
            BrokerDirective::ReuseWorker { .. } | BrokerDirective::UsePassThrough => return Ok(()),
            BrokerDirective::LaunchWorker { .. } => {
                let bootstrap = WorkerBootstrap::from_directive(directive.clone())
                    .expect("launch directive was matched");
                let already_launched = launched
                    .as_ref()
                    .is_some_and(|(id, _, _)| id == &bootstrap.activation_id);
                if !already_launched {
                    match launch_worker(&lease.daemon_origin, &bootstrap).await {
                        Ok(child) => {
                            launched = Some((
                                bootstrap.activation_id.clone(),
                                child,
                                tokio::time::Instant::now(),
                            ));
                        }
                        Err(error) => {
                            report_activation_failed(lease, &bootstrap.activation_id, &error)
                                .await?;
                            directive = refresh_registration(lease).await?.directive;
                            continue;
                        }
                    }
                }
                if let Some((_, child, _)) = launched.as_mut()
                    && let Some(status) = child.try_wait().map_err(CliError::Io)?
                {
                    let error = CliError::Launch(format!(
                        "activated worker exited before readiness with {status}"
                    ));
                    report_activation_failed(lease, &bootstrap.activation_id, &error).await?;
                    directive = refresh_registration(lease).await?.directive;
                    continue;
                }
                if launched
                    .as_ref()
                    .is_some_and(|(activation_id, _, started)| {
                        activation_timed_out(
                            &bootstrap.activation_id,
                            activation_id,
                            *started,
                            tokio::time::Instant::now(),
                        )
                    })
                {
                    let error = CliError::Launch(
                        "activated worker did not register within 15 seconds".into(),
                    );
                    report_activation_failed(lease, &bootstrap.activation_id, &error).await?;
                    directive = refresh_registration(lease).await?.directive;
                    continue;
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            BrokerDirective::WaitForWorker { retry_after_ms } => {
                tokio::time::sleep(Duration::from_millis(retry_after_ms.clamp(10, 1_000))).await;
            }
        }
        if started.elapsed() > ACTIVATION_POLL_MAX {
            return Err(CliError::Launch(
                "timed out waiting for the broker route to become ready".into(),
            ));
        }
        directive = refresh_registration(lease).await?.directive;
    }
}

fn activation_timed_out(
    current_activation_id: &str,
    launched_activation_id: &str,
    launched_at: tokio::time::Instant,
    now: tokio::time::Instant,
) -> bool {
    current_activation_id == launched_activation_id
        && now.saturating_duration_since(launched_at)
            >= Duration::from_millis(ACTIVATION_LIFETIME_MS)
}

async fn refresh_registration(lease: &mut McpLease) -> Result<Registration, CliError> {
    let registration = register(
        &lease.client,
        &lease.daemon_origin,
        &lease.route_credential,
        &lease.identity,
        &lease.session_id,
    )
    .await?;
    apply_registration(lease, &registration);
    Ok(registration)
}

fn apply_registration(lease: &mut McpLease, registration: &Registration) {
    let session_rotated = lease.session_token != registration.session_token;
    lease.session_token = registration.session_token.clone();
    lease.heartbeat_interval = registration.heartbeat_interval;
    if session_rotated {
        lease.sequence = 0;
        lease.pending_heartbeat = None;
    }
}

async fn launch_worker(
    daemon_origin: &str,
    bootstrap: &WorkerBootstrap,
) -> Result<Child, CliError> {
    let executable = std::env::current_exe().map_err(|error| {
        CliError::Launch(format!(
            "failed to resolve the nemo-relay executable: {error}"
        ))
    })?;
    let mut command = worker_command(&executable, daemon_origin, bootstrap);
    let mut child = command
        .spawn()
        .map_err(|error| CliError::Launch(format!("failed to launch daemon worker: {error}")))?;
    let mut stdin = child.stdin.take().ok_or_else(|| {
        CliError::Launch("failed to create the protected worker activation pipe".into())
    })?;
    let payload = serde_json::to_vec(bootstrap).map_err(|error| {
        CliError::Launch(format!("failed to encode worker activation grant: {error}"))
    })?;
    stdin.write_all(&payload).await.map_err(|error| {
        CliError::Launch(format!(
            "failed to transfer worker activation grant: {error}"
        ))
    })?;
    stdin.shutdown().await.map_err(|error| {
        CliError::Launch(format!("failed to close worker activation pipe: {error}"))
    })?;
    Ok(child)
}

fn worker_command(
    executable: &std::path::Path,
    daemon_origin: &str,
    bootstrap: &WorkerBootstrap,
) -> Command {
    let mut command = Command::new(executable);
    command
        .arg("daemon")
        .arg("worker")
        .arg("--daemon-address")
        .arg(daemon_origin)
        .arg("--bind")
        .arg(bootstrap.bind_ip.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .env_remove(ROUTE_TOKEN_ENV)
        .kill_on_drop(false);
    if bootstrap.port != 0 {
        command.arg("--port").arg(bootstrap.port.to_string());
    }
    if let Some(advertise_address) = bootstrap.advertise_address.as_deref() {
        command.arg("--advertise-address").arg(advertise_address);
    }
    command
}

async fn report_activation_failed(
    lease: &mut McpLease,
    activation_id: &str,
    error: &CliError,
) -> Result<(), CliError> {
    log::error!(
        target: "nemo_relay.daemon.mcp",
        event = "worker_launch_failed",
        error_kind = error.log_kind();
        "MCP could not activate the broker-selected worker"
    );
    lease.sequence = lease.sequence.saturating_add(1);
    let request = SessionRequest::new(
        lease.session_id.clone(),
        lease.session_token.clone(),
        lease.sequence,
        ActivationFailedPayload {
            activation_id: activation_id.to_owned(),
            reason: error.to_string(),
        },
    )?;
    post_empty_idempotent(
        &lease.client,
        &format!("{}{}", lease.daemon_origin, MCP_ACTIVATION_FAILED_PATH),
        &request,
        RELEASE_RETRY_POLICY,
    )
    .await
}

async fn maintain_lease(lease: &mut McpLease) -> Result<(), CliError> {
    let mut interval = heartbeat_interval(lease.heartbeat_interval);
    interval.tick().await;
    loop {
        interval.tick().await;
        let response = match renew_lease_with(lease, HEARTBEAT_RETRY_POLICY).await {
            Ok(response) => response,
            Err(CliError::Unauthorized(_)) => {
                // A daemon restart invalidates its in-memory session token. Re-authenticate using
                // the pinned daemon identity and the same user-machine identity instead of
                // tearing down an otherwise healthy MCP stdio session.
                let registration = refresh_registration(lease).await?;
                make_route_ready(lease, registration.directive).await?;
                interval = heartbeat_interval(lease.heartbeat_interval);
                continue;
            }
            Err(error) => return Err(error),
        };
        if let Some(directive) = response.directive {
            make_route_ready(lease, directive).await?;
            interval = heartbeat_interval(lease.heartbeat_interval);
        }
    }
}

async fn renew_lease_with(
    lease: &mut McpLease,
    retry_policy: ControlRetryPolicy,
) -> Result<McpHeartbeatResponse, CliError> {
    if lease.pending_heartbeat.is_none() {
        lease.sequence = lease
            .sequence
            .checked_add(1)
            .ok_or_else(|| CliError::Launch("daemon MCP control sequence was exhausted".into()))?;
        lease.pending_heartbeat = Some(SessionRequest::new(
            lease.session_id.clone(),
            lease.session_token.clone(),
            lease.sequence,
            EmptyPayload::default(),
        )?);
    }
    let request = lease
        .pending_heartbeat
        .as_ref()
        .expect("pending MCP heartbeat was initialized");
    let response = post_json_idempotent(
        &lease.client,
        &format!("{}{}", lease.daemon_origin, MCP_HEARTBEAT_PATH),
        request,
        None,
        retry_policy,
    )
    .await?;
    lease.pending_heartbeat = None;
    Ok(response)
}

fn heartbeat_interval(duration: Duration) -> tokio::time::Interval {
    let mut interval = tokio::time::interval(duration);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    interval
}

async fn release(lease: &mut McpLease) {
    if let Some(request) = lease.pending_heartbeat.as_ref()
        && let Err(error) = post_json_idempotent::<_, McpHeartbeatResponse>(
            &lease.client,
            &format!("{}{}", lease.daemon_origin, MCP_HEARTBEAT_PATH),
            request,
            None,
            RELEASE_RETRY_POLICY,
        )
        .await
    {
        log::warn!(
            target: "nemo_relay.daemon.mcp",
            event = "mcp_release_failed",
            error_kind = error.log_kind();
            "Failed to settle the pending MCP heartbeat before release"
        );
        return;
    }
    lease.pending_heartbeat = None;
    lease.sequence = match lease.sequence.checked_add(1) {
        Some(sequence) => sequence,
        None => {
            log::warn!(
                target: "nemo_relay.daemon.mcp",
                event = "mcp_release_failed";
                "Daemon MCP control sequence was exhausted before release"
            );
            return;
        }
    };
    let request = match SessionRequest::new(
        lease.session_id.clone(),
        lease.session_token.clone(),
        lease.sequence,
        EmptyPayload::default(),
    ) {
        Ok(request) => request,
        Err(error) => {
            log::warn!(
                target: "nemo_relay.daemon.mcp",
                event = "mcp_release_failed",
                error_kind = error.log_kind();
                "Failed to construct the MCP release message"
            );
            return;
        }
    };
    if let Err(error) = post_empty_idempotent(
        &lease.client,
        &format!(
            "{}{}",
            lease.daemon_origin,
            super::common::control::MCP_RELEASE_PATH
        ),
        &request,
        RELEASE_RETRY_POLICY,
    )
    .await
    {
        log::warn!(
            target: "nemo_relay.daemon.mcp",
            event = "mcp_release_failed",
            error_kind = error.log_kind();
            "Failed to release the daemon MCP reference"
        );
    }
}

#[cfg(test)]
#[path = "../../../tests/coverage/daemon/mcp_tests.rs"]
mod tests;
