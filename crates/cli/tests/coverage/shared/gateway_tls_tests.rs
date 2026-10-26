// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

struct Environment {
    _guard: std::sync::MutexGuard<'static, ()>,
    previous: Vec<(&'static str, Option<std::ffi::OsString>)>,
}

impl Environment {
    fn set(values: &[(&'static str, &std::ffi::OsStr)]) -> Self {
        let guard = crate::test_support::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let previous = values
            .iter()
            .map(|(name, _)| (*name, std::env::var_os(name)))
            .collect();
        for (name, value) in values {
            unsafe { std::env::set_var(name, value) };
        }
        Self {
            _guard: guard,
            previous,
        }
    }
}

impl Drop for Environment {
    fn drop(&mut self) {
        for (name, value) in self.previous.drain(..) {
            unsafe {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }
}

#[test]
fn per_user_tls_identity_round_trips_as_server_and_pinned_client_configs() {
    let temp = tempfile::tempdir().unwrap();
    let _environment = Environment::set(&[
        ("XDG_CONFIG_HOME", temp.path().as_os_str()),
        ("HOME", temp.path().as_os_str()),
    ]);

    let identity = RelayTlsIdentity::load_or_create().unwrap();
    identity.server_config().unwrap();
    identity.client_config().unwrap();
    let reloaded = RelayTlsIdentity::load().unwrap();
    reloaded.server_config().unwrap();
    reloaded.client_config().unwrap();
}

#[test]
fn concurrent_first_gateways_keep_the_same_pinned_tls_identity() {
    let temp = tempfile::tempdir().unwrap();
    let _environment = Environment::set(&[
        ("XDG_CONFIG_HOME", temp.path().as_os_str()),
        ("HOME", temp.path().as_os_str()),
    ]);
    let start = std::sync::Barrier::new(16);
    let certificates = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..16)
            .map(|_| {
                scope.spawn(|| {
                    start.wait();
                    RelayTlsIdentity::load_or_create()
                        .unwrap()
                        .record
                        .certificate_der
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>()
    });
    let pinned = RelayTlsIdentity::load().unwrap().record.certificate_der;
    assert!(
        certificates
            .iter()
            .all(|certificate| certificate == &pinned)
    );
}

/// Read a response head without consuming bytes from the TLS or HTTP body.
async fn read_response_head(stream: &mut (impl tokio::io::AsyncRead + Unpin)) -> String {
    use tokio::io::AsyncReadExt;
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        let mut head = Vec::new();
        while !head.ends_with(b"\r\n\r\n") {
            head.push(stream.read_u8().await.unwrap());
            assert!(head.len() < 16 * 1024);
        }
        String::from_utf8(head).unwrap()
    })
    .await
    .expect("server returns a response head")
}

/// Open a real HTTP connection so Hyper owns the upgrade and its lifetime.
async fn open_tunnel(
    address: std::net::SocketAddr,
    protocol: &str,
) -> (tokio::net::TcpStream, String) {
    use tokio::io::AsyncWriteExt;
    let mut stream = tokio::net::TcpStream::connect(address).await.unwrap();
    stream.write_all(format!(
        "GET /bootstrap/tunnel HTTP/1.1\r\nHost: {address}\r\nConnection: upgrade\r\nUpgrade: {protocol}\r\n\r\n"
    ).as_bytes()).await.unwrap();
    let response = read_response_head(&mut stream).await;
    (stream, response)
}

/// Wait for the server task to release the slot after its handshake completes.
async fn wait_for_handshake_slot(permits: &Semaphore) {
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while permits.available_permits() == 0 {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("handshake slot is released");
}

#[tokio::test]
async fn tunnel_handshake_limit_rejects_excess_clients_and_recovers() {
    use axum::body::Body;
    use axum::http::Request;
    use axum::routing::get;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let certified = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
    let identity = RelayTlsIdentity {
        record: IdentityRecord {
            certificate_der: certified.cert.der().to_vec(),
            private_key_der: certified.key_pair.serialize_der(),
        },
    };
    let tls = identity.server_config().unwrap();
    let permits = Arc::new(Semaphore::new(1));
    let route_permits = Arc::clone(&permits);
    let app = axum::Router::new().route(
        "/bootstrap/tunnel",
        get(move |request: Request<Body>| {
            let tls = Arc::clone(&tls);
            let permits = Arc::clone(&route_permits);
            async move {
                let inner = axum::Router::new().route("/healthz", get(|| async { "ok" }));
                upgrade(request, tls, inner, permits)
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let (_invalid, response) = open_tunnel(address, "invalid").await;
    assert!(response.starts_with("HTTP/1.1 403"), "{response}");
    assert_eq!(permits.available_permits(), 1);

    let (pending, response) = open_tunnel(address, "nemo-relay-tls").await;
    assert!(response.starts_with("HTTP/1.1 101"), "{response}");
    let (_excess, response) = open_tunnel(address, "nemo-relay-tls").await;
    assert!(response.starts_with("HTTP/1.1 503"), "{response}");

    // An abandoned handshake must not permanently exhaust the listener.
    drop(pending);
    wait_for_handshake_slot(&permits).await;

    let (stream, response) = open_tunnel(address, "nemo-relay-tls").await;
    assert!(response.starts_with("HTTP/1.1 101"), "{response}");
    let connector = tokio_rustls::TlsConnector::from(identity.client_config().unwrap());
    let mut encrypted = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        connector.connect(
            rustls::pki_types::ServerName::try_from("localhost").unwrap(),
            stream,
        ),
    )
    .await
    .unwrap()
    .unwrap();
    wait_for_handshake_slot(&permits).await;
    encrypted
        .write_all(b"GET /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .await
        .unwrap();
    assert!(
        read_response_head(&mut encrypted)
            .await
            .starts_with("HTTP/1.1 200")
    );
    let mut body = [0; 2];
    encrypted.read_exact(&mut body).await.unwrap();
    assert_eq!(&body, b"ok");

    // A live HTTP connection must not retain a handshake slot. Invalid TLS also releases it.
    let (mut malformed, response) = open_tunnel(address, "nemo-relay-tls").await;
    assert!(response.starts_with("HTTP/1.1 101"), "{response}");
    malformed.write_all(b"invalid TLS header").await.unwrap();
    wait_for_handshake_slot(&permits).await;
    server.abort();
    let _ = server.await;
}

/// Points the system config directory at a test directory for this thread.
struct SystemConfigDir(Option<std::path::PathBuf>);

impl SystemConfigDir {
    fn set(path: &std::path::Path) -> Self {
        Self(crate::configuration::set_test_system_config_dir(Some(
            path.to_path_buf(),
        )))
    }
}

impl Drop for SystemConfigDir {
    fn drop(&mut self) {
        crate::configuration::set_test_system_config_dir(self.0.take());
    }
}

fn generated_identity() -> RelayTlsIdentity {
    let certified = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
    RelayTlsIdentity {
        record: IdentityRecord {
            certificate_der: certified.cert.der().to_vec(),
            private_key_der: certified.key_pair.serialize_der(),
        },
    }
}

/// Runs one TLS handshake between a hook client and a listener presenting `server`.
async fn handshake(
    client: Arc<rustls::ClientConfig>,
    server: &RelayTlsIdentity,
) -> Result<(), String> {
    let (client_stream, server_stream) = tokio::io::duplex(64 * 1024);
    let acceptor = tokio_rustls::TlsAcceptor::from(server.server_config().unwrap());
    let server_task = tokio::spawn(async move { acceptor.accept(server_stream).await.map(drop) });
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        tokio_rustls::TlsConnector::from(client).connect(
            rustls::pki_types::ServerName::try_from("localhost").unwrap(),
            client_stream,
        ),
    )
    .await
    .expect("handshake finishes")
    .map(drop)
    .map_err(|error| error.to_string());
    server_task.abort();
    result
}

#[test]
fn certificate_pem_exports_only_the_certificate() {
    let identity = generated_identity();
    let pem = identity.certificate_pem();

    assert!(pem.starts_with("-----BEGIN CERTIFICATE-----\n"));
    assert!(pem.ends_with("-----END CERTIFICATE-----\n"));
    assert!(!pem.contains("PRIVATE KEY"));
    assert!(pem.lines().all(|line| line.len() <= 64));
    assert_eq!(
        decode_pem_blocks(pem.as_bytes(), "CERTIFICATE").unwrap(),
        vec![identity.record.certificate_der.clone()]
    );
}

#[test]
fn daemon_trust_requires_an_identity_or_a_published_certificate() {
    let temp = tempfile::tempdir().unwrap();
    let user = temp.path().join("user");
    let system = temp.path().join("system");
    std::fs::create_dir_all(&system).unwrap();
    let _environment = Environment::set(&[
        ("XDG_CONFIG_HOME", user.as_os_str()),
        ("HOME", user.as_os_str()),
    ]);
    let _system = SystemConfigDir::set(&system);

    let error = daemon_client_config().unwrap_err();

    assert!(
        error.contains("no pinned certificate for the daemon"),
        "{error}"
    );
    assert!(
        error.contains(&system.join(DAEMON_CERTIFICATE_FILE).display().to_string()),
        "{error}"
    );
    assert!(
        error.contains("nemo-relay daemon tls-certificate"),
        "{error}"
    );
    // Checking for trust must not create per-user state for hook-only users.
    assert!(!user.join("nemo-relay").exists());
}

#[tokio::test]
async fn daemon_trust_pins_the_certificate_published_for_another_account() {
    let temp = tempfile::tempdir().unwrap();
    let user = temp.path().join("user");
    let system = temp.path().join("system");
    std::fs::create_dir_all(&system).unwrap();
    let daemon = generated_identity();
    std::fs::write(
        system.join(DAEMON_CERTIFICATE_FILE),
        daemon.certificate_pem(),
    )
    .unwrap();
    let client = {
        let _environment = Environment::set(&[
            ("XDG_CONFIG_HOME", user.as_os_str()),
            ("HOME", user.as_os_str()),
        ]);
        let _system = SystemConfigDir::set(&system);
        daemon_client_config().unwrap()
    };

    handshake(Arc::clone(&client), &daemon).await.unwrap();
    // A listener that squats the port cannot present the published certificate.
    let error = handshake(client, &generated_identity()).await.unwrap_err();
    assert!(error.contains("certificate"), "{error}");
    assert!(!user.join("nemo-relay").exists());
}

#[tokio::test]
async fn daemon_trust_keeps_the_same_account_identity_next_to_a_published_certificate() {
    let temp = tempfile::tempdir().unwrap();
    let user = temp.path().join("user");
    let system = temp.path().join("system");
    std::fs::create_dir_all(&system).unwrap();
    let daemon = generated_identity();
    std::fs::write(
        system.join(DAEMON_CERTIFICATE_FILE),
        daemon.certificate_pem(),
    )
    .unwrap();
    let (own, client) = {
        let _environment = Environment::set(&[
            ("XDG_CONFIG_HOME", user.as_os_str()),
            ("HOME", user.as_os_str()),
        ]);
        let _system = SystemConfigDir::set(&system);
        let own = RelayTlsIdentity::load_or_create().unwrap();
        (own, daemon_client_config().unwrap())
    };

    handshake(Arc::clone(&client), &own).await.unwrap();
    handshake(Arc::clone(&client), &daemon).await.unwrap();
    assert!(handshake(client, &generated_identity()).await.is_err());
}

#[tokio::test]
async fn daemon_trust_without_a_published_certificate_matches_the_same_account_daemon() {
    let temp = tempfile::tempdir().unwrap();
    let user = temp.path().join("user");
    let system = temp.path().join("system");
    std::fs::create_dir_all(&system).unwrap();
    let (own, client) = {
        let _environment = Environment::set(&[
            ("XDG_CONFIG_HOME", user.as_os_str()),
            ("HOME", user.as_os_str()),
        ]);
        let _system = SystemConfigDir::set(&system);
        let own = RelayTlsIdentity::load_or_create().unwrap();
        (own, daemon_client_config().unwrap())
    };

    handshake(Arc::clone(&client), &own).await.unwrap();
    assert!(handshake(client, &generated_identity()).await.is_err());
}

#[cfg(unix)]
#[test]
fn published_daemon_certificate_must_be_one_administrator_controlled_block() {
    use std::os::unix::fs::PermissionsExt;

    let temp = tempfile::tempdir().unwrap();
    let user = temp.path().join("user");
    let system = temp.path().join("system");
    std::fs::create_dir_all(&system).unwrap();
    std::fs::set_permissions(&system, std::fs::Permissions::from_mode(0o755)).unwrap();
    let path = system.join(DAEMON_CERTIFICATE_FILE);
    let daemon = generated_identity();
    let _environment = Environment::set(&[
        ("XDG_CONFIG_HOME", user.as_os_str()),
        ("HOME", user.as_os_str()),
    ]);
    let _system = SystemConfigDir::set(&system);
    let publish = |contents: &str, mode: u32| {
        let _ = std::fs::remove_file(&path);
        std::fs::write(&path, contents).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
    };

    publish(&daemon.certificate_pem(), 0o644);
    daemon_client_config().unwrap();

    publish(&daemon.certificate_pem(), 0o664);
    let error = daemon_client_config().unwrap_err();
    assert!(
        error.contains("must not be writable by group or other users"),
        "{error}"
    );

    publish(&daemon.certificate_pem(), 0o644);
    std::fs::set_permissions(&system, std::fs::Permissions::from_mode(0o777)).unwrap();
    let error = daemon_client_config().unwrap_err();
    assert!(
        error.contains("directory of the published daemon TLS certificate"),
        "{error}"
    );
    std::fs::set_permissions(&system, std::fs::Permissions::from_mode(0o755)).unwrap();

    let twice = format!(
        "{}{}",
        daemon.certificate_pem(),
        generated_identity().certificate_pem()
    );
    publish(&twice, 0o644);
    let error = daemon_client_config().unwrap_err();
    assert!(error.contains("exactly one CERTIFICATE block"), "{error}");

    publish("not a certificate", 0o644);
    let error = daemon_client_config().unwrap_err();
    assert!(error.contains("exactly one CERTIFICATE block"), "{error}");

    publish(
        "-----BEGIN CERTIFICATE-----\n!!!\n-----END CERTIFICATE-----\n",
        0o644,
    );
    let error = daemon_client_config().unwrap_err();
    assert!(error.contains("invalid CERTIFICATE base64"), "{error}");

    // A symlink could point at a file that a user controls.
    let elsewhere = temp.path().join("elsewhere.pem");
    std::fs::write(&elsewhere, daemon.certificate_pem()).unwrap();
    let _ = std::fs::remove_file(&path);
    std::os::unix::fs::symlink(&elsewhere, &path).unwrap();
    let error = daemon_client_config().unwrap_err();
    assert!(error.contains("must be a regular file"), "{error}");
}
