// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Broker-attached MCP stdio process. It advertises no MCP tools.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::process::Stdio;
use std::time::Duration;

use super::common::socket::{Client, Command as ControlCommand, Event};
use tokio::io::AsyncWriteExt;
use tokio::net::UdpSocket;
use tokio::process::{Child, Command};

use super::common::address::{daemon_url, explicit_daemon_origin};
use super::common::client::{begin_handshake, control_client};
use super::common::control::{
    ACTIVATION_LIFETIME_MS, ActivationFailedPayload, EmptyPayload, McpRegisterRequest,
    McpRegisterResponse, SessionRequest, WorkerActivationFailureReason, WorkerBootstrap,
    WorkerNetworkHint, WorkerNetworkHintProof,
};
use super::common::identity::MachineIdentity;
use super::common::protocol::{BrokerDirective, ComponentRole, SensitiveString};
use super::common::state::{ROUTE_TOKEN_ENV, RouteCredential, load_or_create_machine_identity};
use crate::error::CliError;

// Includes the full two-minute legal drain plus reconciliation margin before a replacement
// activation is issued.
const ACTIVATION_WAIT_MAX: Duration = Duration::from_secs(150);
const WORKER_ADVERTISE_ENV: &str = "NEMO_RELAY_WORKER_ADVERTISE_ADDRESS";
const WORKER_PORT_ENV: &str = "NEMO_RELAY_WORKER_PORT";

#[derive(Debug, Clone)]
pub(crate) struct Options {
    pub(crate) daemon_address: String,
}

struct McpSession {
    client: Client,
    daemon_origin: String,
    route_credential: RouteCredential,
    identity: MachineIdentity,
    session_id: String,
    session_token: SensitiveString,
    sequence: u64,
}

struct Registration {
    directive: BrokerDirective,
    session_token: SensitiveString,
}

pub(crate) async fn run(options: Options) -> Result<(), CliError> {
    let daemon_origin = explicit_daemon_origin(&options.daemon_address)?;
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
    let mut lease = McpSession {
        client,
        daemon_origin,
        route_credential,
        identity,
        session_id,
        session_token: registration.session_token,
        sequence: 0,
    };
    make_route_ready(&mut lease, registration.directive).await?;

    log::info!(
        target: "nemo_relay.daemon.mcp",
        event = "daemon_mcp_ready";
        "Broker reference acquired; MCP protocol is ready"
    );
    let result = {
        let protocol = crate::mcp::serve_daemon_stdio();
        let control = maintain_session(&mut lease);
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
    super::common::socket::retry(|| {
        register_once(client, daemon_origin, credential, identity, session_id)
    })
    .await
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
    let response: McpRegisterResponse = client
        .request(ControlCommand::RegisterMcp {
            request: McpRegisterRequest {
                proof: handshake.proof.clone(),
                worker_network,
            },
            credential: SensitiveString::new(credential.expose())
                .map_err(|error| CliError::Launch(error.to_string()))?,
        })
        .await?;
    handshake.authenticate_daemon(&response.daemon_proof)?;
    Ok(Registration {
        directive: response.directive,
        session_token: response.session_token,
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
            .port_or_known_default()
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

/// A pending worker must not survive failed activation or cancellation. Readiness transfers
/// ownership to the broker; only that success path disarms this guard.
struct ActivationChild {
    child: Child,
    published: bool,
}

struct WorkerActivationError {
    failure_reason: WorkerActivationFailureReason,
    source: CliError,
}

impl WorkerActivationError {
    fn new(failure_reason: WorkerActivationFailureReason, source: CliError) -> Self {
        Self {
            failure_reason,
            source,
        }
    }
}

impl Drop for ActivationChild {
    fn drop(&mut self) {
        if !self.published {
            let _ = self.child.start_kill();
        }
    }
}

type PendingLaunch = Option<(String, ActivationChild, tokio::time::Instant)>;

fn route_activation_timed_out(started: tokio::time::Instant, directive: &BrokerDirective) -> bool {
    started.elapsed() > ACTIVATION_WAIT_MAX
        && !matches!(
            directive,
            BrokerDirective::ReuseWorker { .. } | BrokerDirective::UsePassThrough
        )
}

async fn stop_pending_launch(launched: &mut PendingLaunch) -> Result<(), CliError> {
    if let Some((_, mut child, _)) = launched.take() {
        child.child.kill().await.map_err(CliError::Io)?;
    }
    Ok(())
}

async fn make_route_ready(
    lease: &mut McpSession,
    mut directive: BrokerDirective,
) -> Result<(), CliError> {
    let started = tokio::time::Instant::now();
    let mut launched: PendingLaunch = None;
    let result = async {
        loop {
            if route_activation_timed_out(started, &directive) {
                return Err(CliError::Launch(
                    "timed out waiting for the broker route to become ready".into(),
                ));
            }
            match directive {
                BrokerDirective::ReuseWorker { .. } => {
                    if let Some((_, child, _)) = launched.as_mut() {
                        child.published = true;
                    }
                    launched.take();
                    return Ok(());
                }
                BrokerDirective::UsePassThrough => return Ok(()),
                BrokerDirective::LaunchWorker { .. } => {
                    let bootstrap = WorkerBootstrap::from_directive(directive.clone())
                        .expect("launch directive was matched");
                    let already_launched = launched
                        .as_ref()
                        .is_some_and(|(id, _, _)| id == &bootstrap.activation_id);
                    if !already_launched {
                        stop_pending_launch(&mut launched).await?;
                        match launch_worker(&lease.daemon_origin, &bootstrap).await {
                            Ok(child) => {
                                launched = Some((
                                    bootstrap.activation_id.clone(),
                                    child,
                                    tokio::time::Instant::now(),
                                ));
                            }
                            Err(error) => {
                                report_activation_failed(
                                    lease,
                                    &bootstrap.activation_id,
                                    error.failure_reason,
                                    &error.source,
                                )
                                .await?;
                                directive = refresh_registration(lease).await?.directive;
                                continue;
                            }
                        }
                    }
                    if let Some((activation_id, child, _)) = launched.as_mut()
                        && activation_id == &bootstrap.activation_id
                        && let Some(status) = child.child.try_wait().map_err(CliError::Io)?
                    {
                        let error = CliError::Launch(format!(
                            "activated worker exited before readiness with {status}"
                        ));
                        report_activation_failed(
                            lease,
                            &bootstrap.activation_id,
                            WorkerActivationFailureReason::WorkerExitedBeforeReady,
                            &error,
                        )
                        .await?;
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
                        stop_pending_launch(&mut launched).await?;
                        report_activation_failed(
                            lease,
                            &bootstrap.activation_id,
                            WorkerActivationFailureReason::WorkerReadinessTimeout,
                            &error,
                        )
                        .await?;
                        directive = refresh_registration(lease).await?.directive;
                        continue;
                    }
                }
                BrokerDirective::WaitForWorker { .. } => {}
            }
            // The local timer supervises the child; it sends no network traffic.
            let event = tokio::select! {
                event = lease.client.next() => Some(event),
                _ = tokio::time::sleep(Duration::from_millis(100)) => None,
            };
            if let Some(event) = event {
                directive = receive_directive(lease, event).await?;
            }
        }
    }
    .await;
    let cleanup = stop_pending_launch(&mut launched).await;
    result.and(cleanup)
}

async fn next_directive(lease: &mut McpSession) -> Result<BrokerDirective, CliError> {
    let event = lease.client.next().await;
    receive_directive(lease, event).await
}
async fn receive_directive(
    lease: &mut McpSession,
    event: Result<Event, CliError>,
) -> Result<BrokerDirective, CliError> {
    match event {
        Ok(Event::Directive {
            request_id,
            directive,
        }) => {
            if lease.client.acknowledge(request_id).await.is_err() {
                return Ok(refresh_registration(lease).await?.directive);
            }
            Ok(directive)
        }
        Ok(_) => Err(CliError::Launch("unexpected MCP control event".into())),
        Err(_) => Ok(refresh_registration(lease).await?.directive),
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

async fn refresh_registration(lease: &mut McpSession) -> Result<Registration, CliError> {
    let deadline = lease.client.recovery_deadline().await;
    let registration = super::common::socket::retry_until(deadline, || {
        register_once(
            &lease.client,
            &lease.daemon_origin,
            &lease.route_credential,
            &lease.identity,
            &lease.session_id,
        )
    })
    .await?;
    apply_registration(lease, &registration);
    Ok(registration)
}

fn apply_registration(lease: &mut McpSession, registration: &Registration) {
    lease.session_token = registration.session_token.clone();
    lease.sequence = 0;
}

async fn launch_worker(
    daemon_origin: &str,
    bootstrap: &WorkerBootstrap,
) -> Result<ActivationChild, WorkerActivationError> {
    let executable = std::env::current_exe()
        .map_err(|error| {
            CliError::Launch(format!(
                "failed to resolve the nemo-relay executable: {error}"
            ))
        })
        .map_err(|error| {
            WorkerActivationError::new(
                WorkerActivationFailureReason::WorkerExecutableResolutionFailed,
                error,
            )
        })?;
    let mut command = worker_command(&executable, daemon_origin, bootstrap);
    let child = command
        .spawn()
        .map_err(|error| CliError::Launch(format!("failed to launch daemon worker: {error}")))
        .map_err(|error| {
            WorkerActivationError::new(
                WorkerActivationFailureReason::WorkerProcessSpawnFailed,
                error,
            )
        })?;
    let mut child = ActivationChild {
        child,
        published: false,
    };
    let transfer = async {
        let mut stdin = child
            .child
            .stdin
            .take()
            .ok_or_else(|| {
                CliError::Launch("failed to create the protected worker activation pipe".into())
            })
            .map_err(|error| {
                WorkerActivationError::new(
                    WorkerActivationFailureReason::WorkerActivationPipeUnavailable,
                    error,
                )
            })?;
        let payload = serde_json::to_vec(bootstrap)
            .map_err(|error| {
                CliError::Launch(format!("failed to encode worker activation grant: {error}"))
            })
            .map_err(|error| {
                WorkerActivationError::new(
                    WorkerActivationFailureReason::WorkerActivationGrantSerializationFailed,
                    error,
                )
            })?;
        stdin
            .write_all(&payload)
            .await
            .map_err(|error| {
                CliError::Launch(format!(
                    "failed to transfer worker activation grant: {error}"
                ))
            })
            .map_err(|error| {
                WorkerActivationError::new(
                    WorkerActivationFailureReason::WorkerActivationGrantWriteFailed,
                    error,
                )
            })?;
        stdin
            .shutdown()
            .await
            .map_err(|error| {
                CliError::Launch(format!("failed to close worker activation pipe: {error}"))
            })
            .map_err(|error| {
                WorkerActivationError::new(
                    WorkerActivationFailureReason::WorkerActivationPipeCloseFailed,
                    error,
                )
            })?;
        Ok::<(), WorkerActivationError>(())
    }
    .await;
    if let Err(error) = transfer {
        child
            .child
            .kill()
            .await
            .map_err(CliError::Io)
            .map_err(|error| {
                WorkerActivationError::new(
                    WorkerActivationFailureReason::WorkerActivationCleanupFailed,
                    error,
                )
            })?;
        return Err(error);
    }
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
    lease: &mut McpSession,
    activation_id: &str,
    failure_reason: WorkerActivationFailureReason,
    error: &CliError,
) -> Result<(), CliError> {
    log::error!(
        target: "nemo_relay.daemon.mcp",
        event = "worker_launch_failed",
        error_kind = error.log_kind(),
        failure_reason = failure_reason.as_str();
        "MCP could not activate the broker-selected worker"
    );
    lease.sequence = lease.sequence.saturating_add(1);
    let request = SessionRequest::new(
        lease.session_id.clone(),
        lease.session_token.clone(),
        lease.sequence,
        ActivationFailedPayload {
            activation_id: activation_id.to_owned(),
            failure_reason,
        },
    )?;
    lease
        .client
        .request::<()>(ControlCommand::ActivationFailed(request))
        .await
}

async fn maintain_session(lease: &mut McpSession) -> Result<(), CliError> {
    loop {
        let directive = next_directive(lease).await?;
        make_route_ready(lease, directive).await?;
    }
}

async fn release(lease: &mut McpSession) {
    lease.sequence = lease.sequence.saturating_add(1);
    if let Ok(request) = SessionRequest::new(
        lease.session_id.clone(),
        lease.session_token.clone(),
        lease.sequence,
        EmptyPayload::default(),
    ) {
        let _ = lease
            .client
            .request::<()>(ControlCommand::Release(request))
            .await;
    }
}

#[cfg(test)]
#[path = "../../../tests/coverage/daemon/mcp_tests.rs"]
mod tests;
