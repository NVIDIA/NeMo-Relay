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
use super::common::client_token::resolve_route_credential;
use super::common::control::{
    ActivationFailedPayload, EmptyPayload, McpRegisterRequest, McpRegisterResponse, SessionRequest,
    WorkerActivationFailureReason, WorkerBootstrap, WorkerNetworkHint, WorkerNetworkHintProof,
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
    publication_cleanup: bool,
}

struct Registration {
    directive: BrokerDirective,
    session_token: SensitiveString,
    publication_cleanup: bool,
}

pub(crate) async fn run(options: Options) -> Result<(), CliError> {
    let daemon_origin = explicit_daemon_origin(&options.daemon_address)?;
    let Some(resolved) = resolve_route_credential() else {
        return serve_without_route(PassThroughReason::MissingCredential).await;
    };
    log::debug!(
        target: "nemo_relay.daemon.mcp",
        event = "route_credential_resolved",
        source = resolved.source.as_str();
        "Resolved the managed route credential"
    );
    let route_credential = resolved.credential;
    let client = control_client()?;
    let identity = load_or_create_machine_identity()?;
    let session_id = uuid::Uuid::now_v7().to_string();
    let registration = match register(
        &client,
        &daemon_origin,
        &route_credential,
        &identity,
        &session_id,
    )
    .await
    {
        Ok(registration) => registration,
        Err(CliError::RouteCredentialRejected(_)) => {
            drop(client);
            return serve_without_route(PassThroughReason::CredentialRejected).await;
        }
        Err(error) => return Err(error),
    };
    let mut lease = McpSession {
        client,
        daemon_origin,
        route_credential,
        identity,
        session_id,
        session_token: registration.session_token,
        sequence: 0,
        publication_cleanup: registration.publication_cleanup,
    };
    let initial = registration.directive;
    if !lease.publication_cleanup {
        // Older daemons cannot arbitrate shutdown racing readiness publication.
        make_route_ready(&mut lease, initial.clone()).await?;
    }

    log::info!(
        target: "nemo_relay.daemon.mcp",
        event = "daemon_mcp_ready";
        "Broker reference acquired; MCP protocol is ready"
    );
    let (shutdown, stop) = tokio::sync::watch::channel(false);
    let mut control = tokio::spawn(async move {
        let result = maintain_session(&mut lease, initial, stop).await;
        release(&mut lease).await;
        result
    });
    let protocol = crate::mcp::serve_daemon_stdio();
    tokio::pin!(protocol);
    tokio::select! {
        result = &mut protocol => {
            let _ = shutdown.send(true);
            // Cooperative shutdown lets the broker arbitrate publication before child cleanup.
            control.await.map_err(|error| CliError::Launch(format!("MCP control task failed: {error}")))??;
            result
        }
        result = &mut control => {
            let result = result.map_err(|error| CliError::Launch(format!("MCP control task failed: {error}")))?;
            match control_end(result) {
                ControlEnd::ServeWithoutRoute => {
                    // A rejected re-registration (for example after a daemon restart) must not take
                    // down an MCP server the host requires; keep serving it without a route.
                    log_pass_through(PassThroughReason::CredentialRejectedOnReregistration);
                    protocol.await
                }
                ControlEnd::Finish(result) => result,
            }
        }
    }
}

/// What `run` does when the control task ends before the MCP protocol does.
#[derive(Debug)]
enum ControlEnd {
    /// The daemon definitively rejected the credential; keep serving MCP without a route.
    ServeWithoutRoute,
    /// Return the control task's result, as before.
    Finish(Result<(), CliError>),
}

fn control_end(result: Result<(), CliError>) -> ControlEnd {
    match result {
        Err(CliError::RouteCredentialRejected(_)) => ControlEnd::ServeWithoutRoute,
        other => ControlEnd::Finish(other),
    }
}

/// Why `daemon mcp` serves without a daemon route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PassThroughReason {
    /// No credential resolved from the environment or the token file.
    MissingCredential,
    /// The daemon definitively rejected the credential at registration.
    CredentialRejected,
    /// The daemon definitively rejected the credential when a running MCP registered again.
    CredentialRejectedOnReregistration,
}

/// Serves the MCP protocol without registering a route when no usable credential is available.
///
/// Hosts can mark this server as required, so it must stay up. With no registered route the
/// daemon treats this user's requests as pass-through, matching `BrokerDirective::UsePassThrough`
/// where no worker is ever launched.
async fn serve_without_route(reason: PassThroughReason) -> Result<(), CliError> {
    log_pass_through(reason);
    crate::mcp::serve_daemon_stdio().await
}

fn log_pass_through(reason: PassThroughReason) {
    match reason {
        PassThroughReason::MissingCredential => log::warn!(
            target: "nemo_relay.daemon.mcp",
            event = "daemon_mcp_pass_through",
            route_mode = "pass_through",
            reason = "missing_credential";
            "No NeMo Relay client credential is available; serving MCP without a daemon route"
        ),
        PassThroughReason::CredentialRejected => log::warn!(
            target: "nemo_relay.daemon.mcp",
            event = "daemon_mcp_pass_through",
            route_mode = "pass_through",
            reason = "credential_rejected";
            "The daemon rejected the NeMo Relay client credential; serving MCP without a daemon route"
        ),
        PassThroughReason::CredentialRejectedOnReregistration => log::warn!(
            target: "nemo_relay.daemon.mcp",
            event = "daemon_mcp_pass_through",
            route_mode = "pass_through",
            reason = "credential_rejected",
            phase = "reregistration";
            "The daemon rejected the NeMo Relay client credential when MCP registered again; continuing to serve MCP without a daemon route"
        ),
    }
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
        publication_cleanup: handshake
            .proof
            .transcript
            .responder
            .capabilities
            .contains(super::common::control::ACTIVATION_CANCEL_CAPABILITY),
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

enum WorkerChild {
    Local(Child),
    #[cfg(windows)]
    Detached {
        child: crate::process::detached::DetachedChild,
        stdin: Option<tokio::fs::File>,
    },
}
impl From<Child> for WorkerChild {
    fn from(child: Child) -> Self {
        Self::Local(child)
    }
}
impl WorkerChild {
    fn try_wait(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
        match self {
            Self::Local(child) => child.try_wait(),
            #[cfg(windows)]
            Self::Detached { child, .. } => child.try_wait(),
        }
    }
    fn start_kill(&mut self) -> std::io::Result<()> {
        match self {
            Self::Local(child) => child.start_kill(),
            #[cfg(windows)]
            Self::Detached { child, .. } => child.start_kill(),
        }
    }
    async fn kill(&mut self) -> std::io::Result<()> {
        match self {
            Self::Local(child) => child.kill().await,
            #[cfg(windows)]
            Self::Detached { child, .. } => child.kill().await,
        }
    }
    fn take_stdin(&mut self) -> Option<std::pin::Pin<Box<dyn tokio::io::AsyncWrite + Send>>> {
        match self {
            Self::Local(child) => child.stdin.take().map(|stdin| Box::pin(stdin) as _),
            #[cfg(windows)]
            Self::Detached { stdin, .. } => stdin.take().map(|stdin| Box::pin(stdin) as _),
        }
    }
}

/// A pending worker must not survive failed activation or cancellation. Readiness transfers
/// ownership to the broker; only that success path disarms this guard.
struct ActivationChild {
    child: WorkerChild,
    published: bool,
    publication_uncertain: bool,
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
        if !self.published && !self.publication_uncertain {
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

#[cfg(test)]
async fn stop_pending_launch(launched: &mut PendingLaunch) -> Result<(), CliError> {
    if let Some((_, mut child, _)) = launched.take() {
        child.child.kill().await.map_err(CliError::Io)?;
    }
    Ok(())
}

async fn cleanup_pending_launch(
    lease: &mut McpSession,
    launched: &mut PendingLaunch,
) -> Result<(), CliError> {
    let Some((activation_id, mut child, _)) = launched.take() else {
        return Ok(());
    };
    if lease.publication_cleanup {
        lease.sequence = lease.sequence.saturating_add(1);
        let request = SessionRequest::new(
            lease.session_id.clone(),
            lease.session_token.clone(),
            lease.sequence,
            super::common::control::CancelActivationPayload { activation_id },
        )?;
        let outcome = tokio::time::timeout(
            super::common::socket::ATTEMPT_TIMEOUT,
            lease
                .client
                .request::<super::common::control::ActivationCancellation>(
                    ControlCommand::CancelActivation(request),
                ),
        )
        .await;
        match outcome {
            Ok(Ok(super::common::control::ActivationCancellation::Published)) => {
                child.published = true
            }
            Ok(Ok(_)) => child.child.kill().await.map_err(CliError::Io)?,
            _ => {
                // The broker may already own this child. Its grant/control deadline owns cleanup.
                child.publication_uncertain = true;
            }
        }
    } else if !child.published {
        child.child.kill().await.map_err(CliError::Io)?;
    }
    Ok(())
}

async fn make_route_ready(
    lease: &mut McpSession,
    directive: BrokerDirective,
) -> Result<(), CliError> {
    supervise_route(lease, directive, None).await
}

async fn supervise_route(
    lease: &mut McpSession,
    mut directive: BrokerDirective,
    mut shutdown: Option<&mut tokio::sync::watch::Receiver<bool>>,
) -> Result<(), CliError> {
    let started = tokio::time::Instant::now();
    let mut launched: PendingLaunch = None;
    let result = async {
        loop {
            if !lease.publication_cleanup && route_activation_timed_out(started, &directive) {
                return Err(CliError::Launch(
                    "timed out waiting for the broker route to become ready".into(),
                ));
            }
            match directive {
                BrokerDirective::ReuseWorker { .. } => {
                    if !lease.publication_cleanup {
                        if let Some((_, child, _)) = launched.as_mut() { child.published = true; }
                        launched.take();
                    } else { cleanup_pending_launch(lease, &mut launched).await?; }
                    return Ok(());
                }
                BrokerDirective::UsePassThrough => return Ok(()),
                BrokerDirective::LaunchWorker { .. } => {
                    let bootstrap = WorkerBootstrap::from_directive(directive.clone())
                        .expect("launch directive was matched");
                    if let Some(next_directive) =
                        supervise_worker_launch(lease, &bootstrap, &mut launched).await?
                    {
                        directive = next_directive;
                        continue;
                    }
                }
                BrokerDirective::WaitForWorker { .. } => {}
            }
            if shutdown.as_ref().is_some_and(|stop| *stop.borrow()) { return Ok(()); }
            // The local timer supervises the child; it sends no network traffic.
            let event = tokio::select! {
                _ = async { match shutdown.as_mut() { Some(stop) => { let _ = stop.changed().await; }, None => std::future::pending().await } } => return Ok(()),
                event = lease.client.next() => Some(event),
                _ = tokio::time::sleep(Duration::from_millis(100)) => None,
            };
            if let Some(event) = event {
                directive = receive_directive(lease, event).await?;
            }
        }
    }
    .await;
    let cleanup = cleanup_pending_launch(lease, &mut launched).await;
    result.and(cleanup)
}

async fn supervise_worker_launch(
    lease: &mut McpSession,
    bootstrap: &WorkerBootstrap,
    launched: &mut PendingLaunch,
) -> Result<Option<BrokerDirective>, CliError> {
    let already_launched = launched
        .as_ref()
        .is_some_and(|(id, _, _)| id == &bootstrap.activation_id);
    if !already_launched {
        cleanup_pending_launch(lease, launched).await?;
        match launch_worker(&lease.daemon_origin, bootstrap).await {
            Ok(child) => {
                *launched = Some((
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
                return Ok(Some(refresh_registration(lease).await?.directive));
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
        return Ok(Some(refresh_registration(lease).await?.directive));
    }
    if launched
        .as_ref()
        .is_some_and(|(id, _, _)| id == &bootstrap.activation_id)
        && super::common::control::now_unix_ms() >= bootstrap.deadline_unix_ms
    {
        let error = CliError::Launch("activated worker exceeded its startup deadline".into());
        cleanup_pending_launch(lease, launched).await?;
        report_activation_failed(
            lease,
            &bootstrap.activation_id,
            WorkerActivationFailureReason::WorkerReadinessTimeout,
            &error,
        )
        .await?;
        return Ok(Some(refresh_registration(lease).await?.directive));
    }
    Ok(None)
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

#[cfg(test)]
fn activation_timed_out(
    current_activation_id: &str,
    launched_activation_id: &str,
    launched_at: tokio::time::Instant,
    now: tokio::time::Instant,
) -> bool {
    current_activation_id == launched_activation_id
        && now.saturating_duration_since(launched_at)
            >= Duration::from_millis(super::common::control::ACTIVATION_LIFETIME_MS)
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
    lease.publication_cleanup = registration.publication_cleanup;
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
    let command = worker_command(&executable, daemon_origin, bootstrap);
    #[cfg(not(windows))]
    let spawn = {
        let mut command = command;
        command.spawn().map(WorkerChild::from)
    };
    #[cfg(windows)]
    let spawn = crate::process::detached::inherited_stderr()
        .and_then(|stderr| {
            crate::process::detached::spawn_worker_detached(command.as_std(), &stderr)
        })
        .map(|(child, stdin)| WorkerChild::Detached {
            child,
            stdin: Some(tokio::fs::File::from_std(stdin)),
        });
    let child = spawn
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
        publication_uncertain: false,
    };
    let transfer = async {
        let mut stdin = child
            .child
            .take_stdin()
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
    child.publication_uncertain = true;
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
    #[cfg(unix)]
    crate::process::detached::configure_detached(command.as_std_mut());
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
        "MCP could not activate the broker-selected worker: {error}"
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

async fn maintain_session(
    lease: &mut McpSession,
    mut directive: BrokerDirective,
    mut shutdown: tokio::sync::watch::Receiver<bool>,
) -> Result<(), CliError> {
    if !lease.publication_cleanup {
        // Legacy startup completed before stdio began. Only a new directive can launch again.
        directive = tokio::select! {
            _ = shutdown.changed() => return Ok(()),
            directive = next_directive(lease) => directive?,
        };
    }
    loop {
        supervise_route(lease, directive, Some(&mut shutdown)).await?;
        if *shutdown.borrow() {
            return Ok(());
        }
        directive = tokio::select! {
            _ = shutdown.changed() => return Ok(()),
            directive = next_directive(lease) => directive?,
        };
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
