// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::daemon::common::control::ACTIVATION_LIFETIME_MS;
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
        child: child.into(),
        published: false,
        publication_uncertain: false,
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
        let mut stdin = child.child.take_stdin().unwrap();
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

#[tokio::test]
async fn normal_shutdown_retains_only_negotiated_bootstrapped_children() {
    for (capable, transferred, reason, retain) in [
        (true, true, Shutdown::Normal, true),
        (false, true, Shutdown::Normal, false),
        (true, false, Shutdown::Normal, false),
        (true, true, Shutdown::Failed, false),
    ] {
        let (mut child, address) = pending_worker_fixture().await;
        let stdin = child.child.take_stdin().unwrap();
        child.publication_uncertain = transferred;
        let mut pending = Some(("retained".into(), child, tokio::time::Instant::now()));
        let mut lease = test_lease("http://127.0.0.1:1".into());
        lease.idle_retention = capable;
        retain_pending_launch(&mut lease, &mut pending, reason);
        assert_eq!(
            lease.retained_activation_id.as_deref(),
            retain.then_some("retained")
        );
        assert_eq!(pending.is_none(), retain);
        if retain {
            assert!(
                TcpListener::bind(address).await.is_err(),
                "retained child still owns its listener"
            );
            drop(stdin);
            tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    if TcpListener::bind(address).await.is_ok() {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("fixture exits after its input closes");
        } else {
            stop_pending_launch(&mut pending).await.unwrap();
            drop(stdin);
            TcpListener::bind(address)
                .await
                .expect("cancelled child released its listener");
        }
    }
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
        publication_cleanup: false,
        idle_retention: false,
        retained_activation_id: None,
        worker_exit_report: false,
        watched_worker: None,
    }
}

#[tokio::test]
async fn legacy_startup_is_not_replayed_after_mcp_initialization() {
    use futures_util::StreamExt;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let (shutdown, stop) = tokio::sync::watch::channel(Shutdown::Running);
    let (finished, finish) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
        let unsolicited = tokio::time::timeout(Duration::from_millis(300), socket.next()).await;
        shutdown.send(Shutdown::Normal).unwrap();
        assert!(
            unsolicited.is_err(),
            "legacy MCP replayed its completed startup instead of waiting for a new directive"
        );
        let _ = finish.await;
    });
    let mut lease = test_lease(origin);
    assert!(!lease.publication_cleanup);
    lease
        .client
        .connect(&lease.daemon_origin, ComponentRole::Mcp)
        .await
        .unwrap();
    // An older daemon's startup grant has already completed before MCP begins serving stdio.
    let initial = BrokerDirective::LaunchWorker {
        activation_id: crate::daemon::common::control::random_secret(16).unwrap(),
        activation_token: SensitiveString::new("completed-startup-grant").unwrap(),
        deadline_unix_ms: u64::MAX,
        bind_ip: Ipv4Addr::LOCALHOST,
        port: 0,
        advertise_address: None,
    };
    let result = tokio::time::timeout(
        Duration::from_secs(2),
        maintain_session(&mut lease, initial, stop),
    )
    .await;
    let _ = finished.send(());
    server.await.unwrap();
    result.unwrap().unwrap();
}

#[tokio::test]
async fn worker_activation_failure_sends_sequence_one_and_propagates_socket_rejection() {
    use crate::daemon::common::socket::Request;
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let activation_id = crate::daemon::common::control::random_secret(16).unwrap();
    let expected_activation_id = activation_id.clone();
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
        assert_eq!(payload.payload.activation_id, expected_activation_id);
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
        activation_id,
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

#[cfg(unix)]
#[test]
fn detached_worker_launcher_fixture() {
    let Some(path) = std::env::var_os("NEMO_RELAY_TEST_DETACHED_WORKER_PID") else {
        return;
    };
    let mut command = std::process::Command::new("sleep");
    command
        .arg("30")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    crate::process::detached::configure_detached(&mut command);
    let mut child = command.spawn().unwrap();
    std::fs::write(path, child.id().to_string()).unwrap();
    child.wait().unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn launcher_process_group_cleanup_preserves_detached_worker() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("worker.pid");
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args([
            "--exact",
            "daemon::mcp::tests::detached_worker_launcher_fixture",
            "--nocapture",
        ])
        .env("NEMO_RELAY_TEST_DETACHED_WORKER_PID", &path)
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let mut launcher = crate::process::SupervisedChild::spawn(&mut command)
        .await
        .unwrap();
    let worker = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if let Ok(pid) = std::fs::read_to_string(&path)
                && let Ok(pid) = pid.parse::<i32>()
            {
                break pid;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(unsafe { libc::getsid(worker) }, worker);
    launcher.terminate().await.unwrap();
    let alive = unsafe { libc::kill(worker, 0) };
    unsafe {
        libc::kill(worker, libc::SIGKILL);
    }
    assert_eq!(
        alive, 0,
        "launcher group cleanup must not terminate its detached worker"
    );
}

#[cfg(windows)]
#[test]
fn detached_windows_worker_process_fixture() {
    if std::env::var_os("NEMO_RELAY_TEST_WINDOWS_DETACHED_WORKER").is_some() {
        std::thread::sleep(Duration::from_secs(30));
    }
}

#[cfg(windows)]
fn windows_worker_fixture_logging(path: &std::path::Path) -> nemo_relay::logging::LoggingRuntime {
    use nemo_relay::logging::{
        FileLogSinkConfig, LogFormat, LogLevel, LogSinkConfig, LoggingConfig, init_logging,
    };
    init_logging(&LoggingConfig {
        level: LogLevel::Info,
        stderr_enabled: false,
        sinks: vec![LogSinkConfig::File(FileLogSinkConfig {
            path: path.to_owned(),
            level: LogLevel::Info,
            format: LogFormat::Jsonl,
            ..FileLogSinkConfig::default()
        })],
        ..LoggingConfig::default()
    })
    .unwrap()
}

#[cfg(windows)]
#[test]
fn detached_windows_worker_launcher_fixture() {
    let Some(path) = std::env::var_os("NEMO_RELAY_TEST_WINDOWS_DETACHED_PID") else {
        return;
    };
    let path = std::path::PathBuf::from(path);
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    command
        .args([
            "--exact",
            "daemon::mcp::tests::detached_windows_worker_process_fixture",
        ])
        .env("NEMO_RELAY_TEST_WINDOWS_DETACHED_WORKER", "1");
    let stderr = std::fs::File::create(path.with_extension("stderr")).unwrap();
    if std::env::var_os("NEMO_RELAY_TEST_WINDOWS_EXPECT_BREAKAWAY_DENIED").is_some() {
        let logging = windows_worker_fixture_logging(&path.with_extension("jsonl"));
        let (mut child, _bootstrap) =
            crate::process::detached::spawn_worker_detached(&command, &stderr).unwrap();
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::JobObjects::IsProcessInJob;
        use windows_sys::Win32::System::Threading::{
            OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
        };
        // Verify confinement without changing or bypassing the enclosing job's policy.
        let process = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, child.id()) };
        assert!(!process.is_null());
        let mut in_job = 0;
        let queried = unsafe { IsProcessInJob(process, std::ptr::null_mut(), &mut in_job) };
        unsafe { CloseHandle(process) };
        child.start_kill().unwrap();
        child.wait().unwrap();
        assert_ne!(queried, 0);
        assert_ne!(in_job, 0, "worker escaped the restrictive parent job");
        logging.shutdown();
        std::fs::write(path, "host_scoped").unwrap();
        return;
    }
    let (child, _bootstrap) =
        crate::process::detached::spawn_worker_detached(&command, &stderr).unwrap();
    std::fs::write(path, child.id().to_string()).unwrap();
    std::thread::sleep(Duration::from_secs(30));
}

#[cfg(windows)]
#[tokio::test]
async fn launcher_job_cleanup_preserves_explicitly_detached_worker() {
    use windows_sys::Win32::Foundation::{CloseHandle, WAIT_TIMEOUT};
    use windows_sys::Win32::System::Threading::{
        OpenProcess, PROCESS_SYNCHRONIZE, PROCESS_TERMINATE, TerminateProcess, WaitForSingleObject,
    };
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("worker.pid");
    let mut command = Command::new(std::env::current_exe().unwrap());
    let launcher_stderr = path.with_extension("launcher.stderr");
    command
        .args([
            "--exact",
            "daemon::mcp::tests::detached_windows_worker_launcher_fixture",
            "--nocapture",
        ])
        .env("NEMO_RELAY_TEST_WINDOWS_DETACHED_PID", &path)
        .stdout(Stdio::null())
        .stderr(std::fs::File::create(&launcher_stderr).unwrap());
    let mut launcher = crate::process::SupervisedChild::spawn(&mut command)
        .await
        .unwrap();
    let worker = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if let Ok(pid) = std::fs::read_to_string(&path)
                && let Ok(pid) = pid.parse::<u32>()
            {
                break pid;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap_or_else(|error| {
        panic!(
            "worker launcher did not report its child: {error}: {}",
            std::fs::read_to_string(&launcher_stderr).unwrap_or_default()
        )
    });
    // SAFETY: OpenProcess returns a separately owned handle, closed after cleanup below.
    let handle = unsafe { OpenProcess(PROCESS_TERMINATE | PROCESS_SYNCHRONIZE, 0, worker) };
    assert!(!handle.is_null());
    launcher.terminate().await.unwrap();
    // SAFETY: The live handle has synchronization and termination rights.
    let alive = unsafe { WaitForSingleObject(handle, 0) };
    unsafe {
        TerminateProcess(handle, 1);
        WaitForSingleObject(handle, 5000);
        CloseHandle(handle);
    }
    assert_eq!(
        alive, WAIT_TIMEOUT,
        "Relay Job cleanup killed a worker that requested breakaway"
    );
}

#[cfg(windows)]
#[tokio::test]
async fn restrictive_external_job_fixture() {
    use std::os::windows::io::{FromRawHandle, OwnedHandle};
    use windows_sys::Win32::System::JobObjects::{AssignProcessToJobObject, CreateJobObjectW};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let Some(path) = std::env::var_os("NEMO_RELAY_TEST_WINDOWS_EXTERNAL_JOB") else {
        return;
    };
    // This fixture runs in a separate process; its default job disallows all breakaway.
    // SAFETY: Null arguments create a private job. The owned handle closes it once.
    let raw_job = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
    assert!(!raw_job.is_null());
    let _job = unsafe { OwnedHandle::from_raw_handle(raw_job) };
    assert_ne!(
        unsafe { AssignProcessToJobObject(raw_job, GetCurrentProcess()) },
        0
    );
    let path = std::path::PathBuf::from(path);
    let logging = windows_worker_fixture_logging(&path.with_extension("failed.jsonl"));
    // Exercise direct breakaway denial before SupervisedChild adds a nested Relay job.
    let missing_worker = std::process::Command::new(path.with_extension("missing.exe"));
    let stderr = std::fs::File::create(path.with_extension("failed.stderr")).unwrap();
    assert!(crate::process::detached::spawn_worker_detached(&missing_worker, &stderr).is_err());
    logging.shutdown();
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args([
            "--exact",
            "daemon::mcp::tests::detached_windows_worker_launcher_fixture",
        ])
        .env("NEMO_RELAY_TEST_WINDOWS_DETACHED_PID", &path)
        .env("NEMO_RELAY_TEST_WINDOWS_EXPECT_BREAKAWAY_DENIED", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let mut launcher = crate::process::SupervisedChild::spawn(&mut command)
        .await
        .unwrap();
    assert!(launcher.wait().await.unwrap().success());
}

#[cfg(windows)]
#[tokio::test]
async fn worker_starts_without_escaping_restrictive_external_parent_job() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("result");
    let output = tokio::time::timeout(
        Duration::from_secs(5),
        Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "daemon::mcp::tests::restrictive_external_job_fixture",
            ])
            .env("NEMO_RELAY_TEST_WINDOWS_EXTERNAL_JOB", &path)
            .output(),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "host_scoped");
    let records: Vec<serde_json::Value> = std::fs::read_to_string(path.with_extension("jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .filter(|record: &serde_json::Value| record["event"] == "worker_lifetime_limited")
        .collect();
    assert_eq!(records.len(), 1, "only the successful launch should warn");
    assert!(matches!(
        records[0]["fields"]["reason"].as_str(),
        Some("windows_job_breakaway_denied" | "windows_parent_job_retained")
    ));
    let failed_records = std::fs::read_to_string(path.with_extension("failed.jsonl")).unwrap();
    assert!(
        failed_records.lines().all(|line| {
            let record: serde_json::Value = serde_json::from_str(line).unwrap();
            record["event"] != "worker_lifetime_limited"
        }),
        "failed launches must not emit a worker lifetime warning"
    );
}

#[test]
fn a_credential_rejection_after_reconnect_keeps_serving_without_a_route() {
    assert!(matches!(
        control_end(Err(CliError::RouteCredentialRejected(
            "route token limit reached".into()
        ))),
        ControlEnd::ServeWithoutRoute
    ));
    // Every other ending keeps its previous behavior.
    assert!(matches!(control_end(Ok(())), ControlEnd::Finish(Ok(()))));
    assert!(matches!(
        control_end(Err(CliError::Launch(
            "control reconnect grace period expired".into()
        ))),
        ControlEnd::Finish(Err(CliError::Launch(_)))
    ));
}

/// The launching MCP keeps the published worker's handle so a crash is seen at once, not after
/// the broker's reconnect grace.
#[tokio::test]
async fn launcher_sees_its_published_worker_exit() {
    use tokio::io::AsyncWriteExt as _;
    let (mut child, _address) = pending_worker_fixture().await;
    child.published = true;
    let mut stdin = child.child.take_stdin().unwrap();
    let mut lease = test_lease("http://127.0.0.1:9".into());
    lease.worker_exit_report = true;
    assert_eq!(
        take_exited_worker(&mut lease),
        None,
        "nothing is watched yet"
    );
    lease.watched_worker = Some(("activation-1".to_owned(), child));
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(
        take_exited_worker(&mut lease),
        None,
        "a running worker is not reported"
    );
    assert!(lease.watched_worker.is_some());
    stdin.write_all(b"\n").await.unwrap();
    drop(stdin);
    let activation_id = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if let Some(activation_id) = take_exited_worker(&mut lease) {
                return activation_id;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("the worker exit is seen");
    assert_eq!(activation_id, "activation-1");
    assert!(lease.watched_worker.is_none(), "an exit is reported once");
}

/// A child that has already exited, standing in for a crashed worker.
async fn exited_worker_child() -> ActivationChild {
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "daemon::mcp::tests::no_such_test"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    tokio::time::timeout(Duration::from_secs(10), async {
        while child.try_wait().unwrap().is_none() {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("the stand-in worker exits");
    ActivationChild {
        child: child.into(),
        published: true,
        publication_uncertain: false,
    }
}

async fn reply(
    socket: &mut tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>,
    request_id: String,
    status: u16,
    payload: serde_json::Value,
) {
    use futures_util::SinkExt;
    let response = Event::Reply {
        request_id,
        status,
        payload,
    };
    socket
        .send(tokio_tungstenite::tungstenite::Message::Text(
            serde_json::to_string(&response).unwrap().into(),
        ))
        .await
        .unwrap();
}

/// The MCP spends a crashed worker's reconnect grace inside `supervise_route`, so the exit must be
/// reported from there; a real SIGKILL run waited the full 30 s when only the idle loop checked.
#[tokio::test]
async fn supervise_route_reports_a_dead_watched_worker() {
    use crate::daemon::common::socket::Request;
    use futures_util::StreamExt;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let (shutdown, mut stop) = tokio::sync::watch::channel(Shutdown::Running);
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
        let message = socket.next().await.unwrap().unwrap();
        let request: Request = serde_json::from_str(message.to_text().unwrap()).unwrap();
        let ControlCommand::WorkerExited(report) = request.command else {
            panic!("worker exit report expected")
        };
        assert_eq!(report.sequence, 1);
        assert_eq!(report.payload.activation_id, "activation-1");
        assert!(report.validate_payload_hash());
        reply(
            &mut socket,
            request.request_id,
            204,
            serde_json::Value::Null,
        )
        .await;
        shutdown.send(Shutdown::Normal).unwrap();
        let _ = socket.next().await;
    });
    let mut lease = test_lease(origin);
    lease
        .client
        .connect(&lease.daemon_origin, ComponentRole::Mcp)
        .await
        .unwrap();
    lease.worker_exit_report = true;
    lease.watched_worker = Some(("activation-1".into(), exited_worker_child().await));
    tokio::time::timeout(
        Duration::from_secs(5),
        supervise_route(
            &mut lease,
            BrokerDirective::WaitForWorker {
                retry_after_ms: 1_000,
            },
            Some(&mut stop),
        ),
    )
    .await
    .expect("the report is sent while waiting for the broker")
    .unwrap();
    assert!(lease.watched_worker.is_none(), "an exit is reported once");
    server.abort();
}

/// The worker's control socket can close a moment after the process exits, so a refusal for a
/// still-connected worker is retried with the next sequence number.
#[tokio::test]
async fn worker_exit_report_retries_while_the_worker_control_is_still_connected() {
    use crate::daemon::common::socket::Request;
    use futures_util::StreamExt;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
        let mut sequences = Vec::new();
        for attempt in 0..2 {
            let message = socket.next().await.unwrap().unwrap();
            let request: Request = serde_json::from_str(message.to_text().unwrap()).unwrap();
            let ControlCommand::WorkerExited(report) = request.command else {
                panic!("worker exit report expected")
            };
            sequences.push(report.sequence);
            if attempt == 0 {
                let refusal =
                    serde_json::json!({"error": {"message": "worker control is still connected"}});
                reply(&mut socket, request.request_id, 409, refusal).await;
            } else {
                reply(
                    &mut socket,
                    request.request_id,
                    204,
                    serde_json::Value::Null,
                )
                .await;
            }
        }
        sequences
    });
    let mut lease = test_lease(origin);
    lease
        .client
        .connect(&lease.daemon_origin, ComponentRole::Mcp)
        .await
        .unwrap();
    tokio::time::timeout(
        Duration::from_secs(5),
        report_worker_exited(&mut lease, "activation-1"),
    )
    .await
    .expect("the retry finishes");
    assert_eq!(server.await.unwrap(), vec![1, 2]);
    assert_eq!(lease.sequence, 2);
}

/// When the broker answers a launch cleanup with `Published`, the launcher keeps the child so it
/// can report the worker's exit later instead of dropping the handle.
#[tokio::test]
async fn published_launch_cleanup_hands_the_child_back() {
    use crate::daemon::common::socket::Request;
    use futures_util::StreamExt;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
        let message = socket.next().await.unwrap().unwrap();
        let request: Request = serde_json::from_str(message.to_text().unwrap()).unwrap();
        let ControlCommand::CancelActivation(cancel) = request.command else {
            panic!("activation cancellation expected")
        };
        assert_eq!(cancel.payload.activation_id, "activation-1");
        reply(
            &mut socket,
            request.request_id,
            200,
            serde_json::json!("published"),
        )
        .await;
        let _ = socket.next().await;
    });
    let mut lease = test_lease(origin);
    lease.publication_cleanup = true;
    lease
        .client
        .connect(&lease.daemon_origin, ComponentRole::Mcp)
        .await
        .unwrap();
    let mut child = exited_worker_child().await;
    child.published = false;
    let mut launched: PendingLaunch =
        Some(("activation-1".into(), child, tokio::time::Instant::now()));
    let kept = tokio::time::timeout(
        Duration::from_secs(5),
        cleanup_pending_launch(&mut lease, &mut launched),
    )
    .await
    .unwrap()
    .unwrap();
    let (activation_id, child) = kept.expect("a published child is handed back");
    assert_eq!(activation_id, "activation-1");
    assert!(child.published);
    server.abort();
}

/// A daemon without `worker_exit_report_v1` would close the control connection on the command,
/// so a watched exit is dropped instead of reported after re-registering with such a daemon.
#[tokio::test]
async fn worker_exit_is_not_reported_without_the_daemon_capability() {
    let mut lease = test_lease("http://127.0.0.1:9".into());
    assert!(!lease.worker_exit_report);
    lease.watched_worker = Some(("activation-1".into(), exited_worker_child().await));
    assert_eq!(take_exited_worker(&mut lease), None);
    assert!(lease.watched_worker.is_none());
}

/// After an announced clean exit the broker has usually replaced the worker already; the MCP stops
/// at the first "not known" answer instead of retrying.
#[tokio::test]
async fn worker_exit_report_stops_when_the_worker_is_already_gone() {
    use crate::daemon::common::socket::Request;
    use futures_util::StreamExt;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
        let message = socket.next().await.unwrap().unwrap();
        let request: Request = serde_json::from_str(message.to_text().unwrap()).unwrap();
        assert!(matches!(request.command, ControlCommand::WorkerExited(_)));
        let gone = serde_json::json!({"error": {"message": "worker control is not known"}});
        reply(&mut socket, request.request_id, 404, gone).await;
        tokio::time::timeout(Duration::from_millis(500), socket.next())
            .await
            .is_err()
    });
    let mut lease = test_lease(origin);
    lease
        .client
        .connect(&lease.daemon_origin, ComponentRole::Mcp)
        .await
        .unwrap();
    tokio::time::timeout(
        Duration::from_secs(5),
        report_worker_exited(&mut lease, "activation-1"),
    )
    .await
    .expect("the report gives up at once");
    assert!(server.await.unwrap(), "no retry after the worker is gone");
    assert_eq!(lease.sequence, 1);
}

/// The report blocks directive handling, so a broker that never answers costs at most the report
/// window, not the control client's longer request timeout.
#[tokio::test]
async fn worker_exit_report_gives_up_within_its_window() {
    use futures_util::StreamExt;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let (received, mut frames) = tokio::sync::mpsc::unbounded_channel();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
        // Read requests and never answer them.
        while socket.next().await.is_some() {
            let _ = received.send(());
        }
    });
    let mut lease = test_lease(origin);
    lease
        .client
        .connect(&lease.daemon_origin, ComponentRole::Mcp)
        .await
        .unwrap();
    let started = tokio::time::Instant::now();
    report_worker_exited(&mut lease, "activation-1").await;
    let waited = started.elapsed();
    assert!(
        waited < WORKER_EXIT_REPORT_WINDOW + Duration::from_millis(500),
        "the report waited {waited:?}"
    );
    assert!(
        waited >= WORKER_EXIT_REPORT_WINDOW - Duration::from_millis(100),
        "the report gave up early after {waited:?}"
    );
    assert!(frames.try_recv().is_ok(), "the report reached the broker");
    server.abort();
}

/// A report that times out has already spent its sequence number, so it must reach the broker
/// before the MCP's next request; the broker only accepts sequence numbers in order.
#[tokio::test]
async fn a_timed_out_worker_exit_report_keeps_its_place_before_the_next_request() {
    use crate::daemon::common::socket::Request;
    use futures_util::StreamExt;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let (received, mut sequences) = tokio::sync::mpsc::unbounded_channel();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
        // Never answer, so the report runs out its window, then record what arrives in order.
        while let Some(Ok(message)) = socket.next().await {
            let Ok(text) = message.to_text() else {
                continue;
            };
            let request: Request = serde_json::from_str(text).unwrap();
            let sequence = match request.command {
                ControlCommand::WorkerExited(report) => ("worker_exited", report.sequence),
                ControlCommand::Release(release) => ("release", release.sequence),
                _ => continue,
            };
            let _ = received.send(sequence);
        }
    });
    let mut lease = test_lease(origin);
    lease
        .client
        .connect(&lease.daemon_origin, ComponentRole::Mcp)
        .await
        .unwrap();
    report_worker_exited(&mut lease, "activation-1").await;
    release(&mut lease).await;
    let mut arrived = Vec::new();
    while arrived.len() < 2 {
        let next = tokio::time::timeout(Duration::from_secs(10), sequences.recv())
            .await
            .expect("both requests reach the broker")
            .unwrap();
        arrived.push(next);
    }
    assert_eq!(arrived, vec![("worker_exited", 1), ("release", 2)]);
    server.abort();
}
