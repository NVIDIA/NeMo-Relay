// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use tokio::net::TcpListener;

#[test]
fn pending_worker_child_fixture() {
    if std::env::var_os("NEMO_RELAY_TEST_PENDING_WORKER_FIXTURE").is_none() {
        return;
    }
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    println!("PENDING_WORKER_READY {}", listener.local_addr().unwrap());
    use std::io::Write as _;
    std::io::stdout().flush().unwrap();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
}

async fn pending_worker_fixture() -> (ActivationChild, SocketAddr) {
    use tokio::io::AsyncBufReadExt as _;
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "daemon::mcp::tests::pending_worker_child_fixture",
            "--nocapture",
        ])
        .env("NEMO_RELAY_TEST_PENDING_WORKER_FIXTURE", "1")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let stdout = child.stdout.take().unwrap();
    let child = ActivationChild {
        child,
        published: false,
    };
    let address = tokio::time::timeout(Duration::from_secs(10), async {
        let mut lines = tokio::io::BufReader::new(stdout).lines();
        while let Some(line) = lines.next_line().await.unwrap() {
            if let Some(address) = line.strip_prefix("PENDING_WORKER_READY ") {
                return address.parse::<SocketAddr>().unwrap();
            }
        }
        panic!("worker fixture exited before readiness");
    })
    .await
    .unwrap();
    (child, address)
}

#[tokio::test]
async fn failed_activation_cleanup_reaps_child_and_frees_its_listener() {
    let (child, address) = pending_worker_fixture().await;
    let mut pending = Some((
        "failed-activation".into(),
        child,
        tokio::time::Instant::now(),
    ));
    stop_pending_launch(&mut pending).await.unwrap();
    assert!(pending.is_none());
    let _replacement = TcpListener::bind(address)
        .await
        .expect("old worker listener released");
    stop_pending_launch(&mut pending).await.unwrap();
}

#[tokio::test]
async fn published_worker_survives_guard_drop_but_pending_worker_is_killed() {
    for published in [false, true] {
        let (mut child, address) = pending_worker_fixture().await;
        child.published = published;
        let mut stdin = child.child.stdin.take().unwrap();
        drop(child);
        if published {
            let _connection = tokio::net::TcpStream::connect(address)
                .await
                .expect("published worker must remain alive");
            stdin.write_all(b"exit\n").await.unwrap();
        }
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if let Ok(listener) = TcpListener::bind(address).await {
                    break listener;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("worker must terminate");
        drop(stdin);
    }
}

#[test]
fn launch_directive_is_the_only_directive_with_a_worker_bootstrap() {
    assert!(WorkerBootstrap::from_directive(BrokerDirective::UsePassThrough).is_none());
    assert!(
        WorkerBootstrap::from_directive(BrokerDirective::WaitForWorker { retry_after_ms: 10 })
            .is_none()
    );
}

#[test]
fn activation_timeout_uses_the_mcp_monotonic_clock() {
    let started = tokio::time::Instant::now();
    assert!(!activation_timed_out(
        "activation",
        "activation",
        started,
        started + Duration::from_millis(ACTIVATION_LIFETIME_MS - 1),
    ));
    assert!(activation_timed_out(
        "activation",
        "activation",
        started,
        started + Duration::from_millis(ACTIVATION_LIFETIME_MS),
    ));
    assert!(!activation_timed_out(
        "replacement",
        "activation",
        started,
        started + Duration::from_millis(ACTIVATION_LIFETIME_MS),
    ));
}

#[test]
fn prescribed_worker_network_accepts_host_or_ipv4_and_rejects_unsafe_values() {
    assert_eq!(
        parse_worker_network_overrides(Some("Worker.Example.com"), Some("9443"))
            .expect("hostname override"),
        (Some("worker.example.com".into()), Some(9443))
    );
    assert_eq!(
        parse_worker_network_overrides(Some("192.0.2.10"), None).expect("IPv4 override"),
        (Some("192.0.2.10".into()), None)
    );
    assert!(parse_worker_network_overrides(Some("0.0.0.0"), None).is_err());
    assert!(parse_worker_network_overrides(Some("[::1]"), None).is_err());
    assert!(parse_worker_network_overrides(Some("https://worker.example.com"), None).is_err());
    assert!(parse_worker_network_overrides(None, Some("0")).is_err());
}

#[tokio::test]
async fn loopback_daemon_uses_loopback_worker_network_and_environment_overrides() {
    let _environment = crate::test_support::EnvScope::set(&[
        (WORKER_ADVERTISE_ENV, None),
        (WORKER_PORT_ENV, None),
    ]);
    let hint = worker_network_hint("http://127.0.0.1:47632")
        .await
        .expect("loopback daemon network hint");
    assert_eq!(hint.advertised_host, "127.0.0.1");
    assert_eq!(hint.port, None);

    for origin in ["http://127.0.0.1:80", "https://127.0.0.1:443"] {
        let hint = worker_network_hint(origin)
            .await
            .expect("default-port network hint");
        assert_eq!(hint.advertised_host, "127.0.0.1");
        assert_eq!(hint.port, None);
    }

    drop(_environment);
    let _environment = crate::test_support::EnvScope::set(&[
        (
            WORKER_ADVERTISE_ENV,
            Some(std::ffi::OsStr::new("worker.example")),
        ),
        (WORKER_PORT_ENV, Some(std::ffi::OsStr::new("9443"))),
    ]);
    let hint = worker_network_hint("http://127.0.0.1:47632")
        .await
        .expect("explicit network hint");
    assert_eq!(hint.advertised_host, "worker.example");
    assert_eq!(hint.port, Some(9443));
}

#[cfg(unix)]
#[test]
fn worker_network_environment_requires_unicode() {
    use std::os::unix::ffi::OsStrExt;

    let _environment = crate::test_support::EnvScope::set(&[(
        WORKER_ADVERTISE_ENV,
        Some(std::ffi::OsStr::from_bytes(b"worker-\xff")),
    )]);
    assert!(optional_environment(WORKER_ADVERTISE_ENV).is_err());
}

#[tokio::test]
async fn ipv6_only_daemon_has_no_supported_worker_route() {
    let _environment = crate::test_support::EnvScope::set(&[
        (WORKER_ADVERTISE_ENV, None),
        (WORKER_PORT_ENV, None),
    ]);
    assert!(worker_network_hint("http://[::1]:47632").await.is_err());
}

#[tokio::test]
async fn remote_daemon_rejects_an_explicit_loopback_worker_advertisement() {
    let _environment = crate::test_support::EnvScope::set(&[
        (
            WORKER_ADVERTISE_ENV,
            Some(std::ffi::OsStr::new("127.0.0.1")),
        ),
        (WORKER_PORT_ENV, None),
    ]);
    let error = worker_network_hint("https://192.0.2.1:8443")
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("cannot be loopback for a remote daemon"),
        "unexpected network validation error: {error}"
    );
}

#[tokio::test]
async fn remote_daemon_derives_a_concrete_non_loopback_worker_advertisement() {
    let _environment = crate::test_support::EnvScope::set(&[
        (WORKER_ADVERTISE_ENV, None),
        (WORKER_PORT_ENV, Some(std::ffi::OsStr::new("9443"))),
    ]);
    let hint = match worker_network_hint("https://192.0.2.1:8443").await {
        Ok(hint) => hint,
        Err(CliError::Io(error))
            if matches!(
                error.kind(),
                std::io::ErrorKind::NetworkUnreachable | std::io::ErrorKind::HostUnreachable
            ) =>
        {
            return;
        }
        Err(error) => panic!("unexpected route derivation error: {error}"),
    };
    assert!(
        !hint
            .advertised_host
            .parse::<Ipv4Addr>()
            .unwrap()
            .is_loopback()
    );
    assert_eq!(hint.port, Some(9443));
}

#[test]
fn spawned_worker_explicitly_removes_the_public_route_credential() {
    let bootstrap = WorkerBootstrap {
        activation_id: "activation".into(),
        activation_token: SensitiveString::new("secret").expect("secret"),
        deadline_unix_ms: u64::MAX,
        bind_ip: Ipv4Addr::LOCALHOST,
        port: 0,
        advertise_address: None,
    };
    let command = worker_command(
        std::path::Path::new("nemo-relay"),
        "http://127.0.0.1:47632",
        &bootstrap,
    );
    assert!(
        command
            .as_std()
            .get_envs()
            .any(|(name, value)| { name == ROUTE_TOKEN_ENV && value.is_none() })
    );
}

#[test]
fn worker_command_keeps_only_documented_network_arguments() {
    let bootstrap = WorkerBootstrap {
        activation_id: "activation".into(),
        activation_token: SensitiveString::new("secret").expect("secret"),
        deadline_unix_ms: u64::MAX,
        bind_ip: Ipv4Addr::UNSPECIFIED,
        port: 9443,
        advertise_address: Some("worker.example".into()),
    };
    let origin = explicit_daemon_origin("https://daemon.example:443/").unwrap();
    let command = worker_command(std::path::Path::new("nemo-relay"), &origin, &bootstrap);
    let arguments = command
        .as_std()
        .get_args()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    assert_eq!(
        arguments,
        [
            "daemon",
            "worker",
            "--daemon-address",
            "https://daemon.example:443",
            "--bind",
            "0.0.0.0",
            "--port",
            "9443",
            "--advertise-address",
            "worker.example",
        ]
    );
}

#[tokio::test]
async fn ready_directives_complete_without_spawning_or_polling() {
    let mut lease = test_lease("http://127.0.0.1:1".into());
    make_route_ready(&mut lease, BrokerDirective::UsePassThrough)
        .await
        .unwrap();
    make_route_ready(
        &mut lease,
        BrokerDirective::ReuseWorker {
            endpoint: "http://127.0.0.1:2".into(),
        },
    )
    .await
    .unwrap();
}

fn test_lease(daemon_origin: String) -> McpSession {
    McpSession {
        client: control_client().expect("client"),
        daemon_origin,
        route_credential: RouteCredential::parse(
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA".into(),
        )
        .expect("route credential"),
        identity: MachineIdentity::generate().expect("identity").identity,
        session_id: "mcp-test-session".into(),
        session_token: SensitiveString::new("session-secret").expect("session token"),
        sequence: 0,
    }
}

#[tokio::test]
async fn worker_activation_failure_sends_sequence_one_and_propagates_socket_rejection() {
    use crate::daemon::common::socket::Request;
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
        let message = socket.next().await.unwrap().unwrap();
        let request: Request = serde_json::from_str(message.to_text().unwrap()).unwrap();
        let ControlCommand::ActivationFailed(payload) = request.command else {
            panic!("activation failure expected")
        };
        assert_eq!(payload.sequence, 1);
        assert_eq!(payload.session_id, "mcp-test-session");
        assert_eq!(payload.payload.activation_id, "failed-activation");
        assert_eq!(
            payload.payload.failure_reason,
            WorkerActivationFailureReason::WorkerExitedBeforeReady
        );
        assert!(payload.validate_payload_hash());
        let response = Event::Reply {
            request_id: request.request_id,
            status: 500,
            payload: serde_json::json!({"error":{"message":"activation failure rejected"}}),
        };
        socket
            .send(Message::Text(
                serde_json::to_string(&response).unwrap().into(),
            ))
            .await
            .unwrap();
    });
    let mut lease = test_lease(origin);
    lease
        .client
        .connect(&lease.daemon_origin, ComponentRole::Mcp)
        .await
        .unwrap();
    // The unit-test executable rejects the worker CLI arguments and exits before readiness.
    let directive = BrokerDirective::LaunchWorker {
        activation_id: "failed-activation".into(),
        activation_token: SensitiveString::new("secret").unwrap(),
        deadline_unix_ms: u64::MAX,
        bind_ip: Ipv4Addr::LOCALHOST,
        port: 0,
        advertise_address: None,
    };
    let error = tokio::time::timeout(
        Duration::from_secs(5),
        make_route_ready(&mut lease, directive),
    )
    .await
    .unwrap()
    .unwrap_err();
    assert!(
        error.to_string().contains("activation failure rejected"),
        "{error}"
    );
    server.await.unwrap();
}
