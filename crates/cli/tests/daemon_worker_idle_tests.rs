// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-level coverage of MCP exit during managed worker initialization.

#![cfg(all(feature = "__skip-implicit-config", any(unix, windows)))]

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

use base64::Engine;
use nemo_relay_plugin_host::dynamic::DynamicPluginManifest;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sysinfo::{Pid, ProcessRefreshKind, ProcessStatus, ProcessesToUpdate, System};

const WAIT: Duration = Duration::from_secs(15);
// Allow the existing request-drain interval plus test observation slack.
const EXIT_WAIT: Duration = Duration::from_secs(150);

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<CapturedOutput>>);

#[derive(Default)]
struct CapturedOutput {
    bytes: Vec<u8>,
    error: Option<String>,
}

impl Capture {
    fn read(reader: impl Read + Send + 'static) -> Self {
        let capture = Self::default();
        let destination = capture.clone();
        std::thread::spawn(move || {
            let mut reader = reader;
            let mut buffer = [0; 4096];
            loop {
                match reader.read(&mut buffer) {
                    Ok(0) => break,
                    Ok(length) => destination
                        .0
                        .lock()
                        .unwrap()
                        .bytes
                        .extend(&buffer[..length]),
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(error) => {
                        destination.0.lock().unwrap().error = Some(error.to_string());
                        break;
                    }
                }
            }
        });
        capture
    }

    fn text(&self) -> String {
        let output = self.0.lock().unwrap();
        assert!(
            output.error.is_none(),
            "captured child output: {:?}",
            output.error
        );
        String::from_utf8_lossy(&output.bytes).into_owned()
    }

    fn events(&self, name: &str) -> Vec<Value> {
        self.text()
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .filter(|row| row["event"] == name)
            .collect()
    }
}

struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

struct Mcp {
    child: ChildGuard,
    stdout: mpsc::Receiver<std::io::Result<String>>,
    stderr: Capture,
}

impl Mcp {
    async fn finish(mut self) {
        // Closing stdin is the MCP host's normal shutdown request, not cancellation.
        drop(self.child.0.stdin.take());
        let status = self.wait_for_exit().await;
        assert!(status.success(), "MCP stderr: {}", self.stderr.text());
    }

    #[cfg(unix)]
    async fn terminate(mut self) {
        use std::os::unix::process::ExitStatusExt;

        // Codex signals its MCP process before closing stdin. Preserve that ordering.
        assert!(self.child.0.stdin.is_some());
        // SAFETY: This is the live child owned by this MCP fixture, not a process group.
        assert_eq!(
            unsafe { libc::kill(self.child.0.id().try_into().unwrap(), libc::SIGTERM) },
            0
        );
        let status = self.wait_for_exit().await;
        assert_eq!(status.signal(), Some(libc::SIGTERM));
    }

    async fn wait_for_exit(&mut self) -> ExitStatus {
        let status = wait_until("MCP process exit", || self.child.0.try_wait().unwrap()).await;
        wait_until("MCP stdout EOF", || match self.stdout.try_recv() {
            Err(mpsc::TryRecvError::Disconnected) => Some(()),
            Ok(Err(error)) => panic!("MCP stdout: {error}"),
            _ => None,
        })
        .await;
        status
    }
}

#[derive(Clone, Copy)]
struct ProcessIdentity {
    pid: Pid,
    started: u64,
}

fn processes(pids: &[Pid]) -> System {
    let mut system = System::new();
    system.refresh_processes_specifics(
        ProcessesToUpdate::Some(pids),
        true,
        ProcessRefreshKind::nothing().without_tasks(),
    );
    system
}

impl ProcessIdentity {
    fn present(self) -> bool {
        processes(&[self.pid])
            .process(self.pid)
            .is_some_and(|process| {
                process.start_time() == self.started && process.status() != ProcessStatus::Zombie
            })
    }

    fn cleanup(self) {
        let system = processes(&[self.pid]);
        if let Some(process) = system.process(self.pid)
            && process.start_time() == self.started
        {
            let _ = process.kill();
        }
    }
}

struct IdleDaemon {
    root: tempfile::TempDir,
    daemon: ChildGuard,
    logs: Capture,
    address: String,
    token: String,
    client: reqwest::Client,
    provider: tokio::task::JoinHandle<()>,
    requests: Arc<Mutex<Vec<Value>>>,
    descendants: Vec<ProcessIdentity>,
    mcp_logs: Mutex<Vec<Capture>>,
}

impl IdleDaemon {
    async fn start(idle_seconds: u64) -> Self {
        use axum::{Json, Router, routing::post};

        let root = tempfile::tempdir().unwrap();
        for directory in [
            "daemon/.config/nemo-relay",
            "user/.config/nemo-relay",
            "system",
            "gate",
            "traces",
            "plugin",
        ] {
            std::fs::create_dir_all(root.path().join(directory)).unwrap();
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(
                root.path().join("system"),
                std::fs::Permissions::from_mode(0o755),
            )
            .unwrap();
        }
        let fixture = std::env::var_os("NEMO_RELAY_TEST_WORKER_PLUGIN")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../target/test-plugin-fixtures/debug")
                    .join(format!(
                        "nemo-relay-worker-plugin-fixture{}",
                        std::env::consts::EXE_SUFFIX
                    ))
            });
        assert!(
            fixture.is_file(),
            "build the worker fixture with just build-test-plugin-fixtures: {}",
            fixture.display()
        );
        let packaged_fixture = root
            .path()
            .join("plugin")
            .join(fixture.file_name().unwrap());
        std::fs::copy(&fixture, &packaged_fixture).unwrap();
        let fixture = packaged_fixture;
        let manifest = root.path().join("plugin/relay-plugin.toml");
        write_toml(
            &manifest,
            json!({
                "manifest_version": 1,
                "plugin": {"id": "fixture_worker", "kind": "worker"},
                "compat": {"relay": format!("={}", env!("CARGO_PKG_VERSION")), "worker_protocol": "nvidia.nemo.relay.worker.v1"},
                "defaults": {"enabled": false},
                "capabilities": {"items": ["plugin_worker"]},
                "source": {"artifact": fixture},
                "integrity": {"sha256": fixture_digest(&fixture)},
                "load": {"runtime": "rust", "entrypoint": fixture}
            }),
        );
        let (plugin_manifest, manifest_ref) =
            DynamicPluginManifest::load_from_path(&manifest).unwrap();
        let mut record = plugin_manifest.into_record(Some(manifest_ref)).unwrap();
        record.spec.enabled = true;
        let records = vec![record];
        let dynamic = vec![json!({"manifest": manifest, "config": {
            "event_metadata_injector_only": true,
            "register_gate_directory": root.path().join("gate")
        }})];
        let policies = json!({"fixture_worker": {"attestation": "integrity_only"}});
        std::fs::write(
            root.path()
                .join("user/.config/nemo-relay/.dynamic-plugins.json"),
            serde_json::to_vec_pretty(&json!({"schema_version": 1, "records": records})).unwrap(),
        )
        .unwrap();
        write_toml(
            &root.path().join("user/.config/nemo-relay/plugins.toml"),
            json!({
                "version": 1,
                "plugins": {
                    "policy": {"overrides": policies},
                    "dynamic": dynamic
                },
                "components": [{"kind": "observability", "enabled": true, "config": {
                    "version": 4, "enable_full_payloads": true,
                    "atof": {"enabled": true, "sinks": [{"type": "file", "output_directory": root.path().join("traces"), "filename": "events.jsonl", "mode": "append"}]}
                }}]
            }),
        );
        let requests = Arc::new(Mutex::new(Vec::new()));
        let captured = requests.clone();
        let provider_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let provider_address = provider_listener.local_addr().unwrap();
        let router = Router::new().route("/v1/chat/completions", post(move |Json(body): Json<Value>| {
            let captured = captured.clone();
            async move {
                let marker = body["messages"][0]["content"].as_str().unwrap().to_owned();
                captured.lock().unwrap().push(body);
                Json(json!({"id":"idle-completion", "object":"chat.completion", "model":"idle-model",
                    "choices":[{"index":0,"message":{"role":"assistant","content":marker},"finish_reason":"stop"}],
                    "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}))
            }
        }));
        let provider =
            tokio::spawn(async move { axum::serve(provider_listener, router).await.unwrap() });
        write_toml(
            &root.path().join("system/config.toml"),
            json!({
                "daemon": {"worker_idle_timeout_secs": idle_seconds,
                    "logging": {"level":"info", "stderr_enabled":true, "stderr_format":"jsonl"}},
                "logging": {"level":"info", "stderr_enabled":true, "stderr_format":"jsonl"},
                "upstream": {"openai_base_url": format!("http://{provider_address}/v1")}
            }),
        );
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let child = account_command(root.path(), "daemon")
            .args(["daemon", "--port", &port.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut child = ChildGuard(child);
        let logs = Capture::read(child.0.stderr.take().unwrap());
        let setup = Self {
            root,
            daemon: child,
            logs,
            address: format!("http://127.0.0.1:{port}"),
            token: base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([0x5a; 32]),
            client: reqwest::Client::builder()
                .no_proxy()
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap(),
            provider,
            requests,
            descendants: Vec::new(),
            mcp_logs: Mutex::new(Vec::new()),
        };
        let deadline = Instant::now() + WAIT;
        loop {
            if setup
                .client
                .get(format!("{}/healthz", setup.address))
                .send()
                .await
                .is_ok()
            {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "daemon did not start: {}",
                setup.logs.text()
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        setup
    }

    async fn mcp(&self) -> Mcp {
        let child = account_command(self.root.path(), "user")
            .args(["daemon", "mcp", "--daemon-address", &self.address])
            .env("NEMO_RELAY_CLIENT_TOKEN", &self.token)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut child = ChildGuard(child);
        let stderr = Capture::read(child.0.stderr.take().unwrap());
        self.mcp_logs.lock().unwrap().push(stderr.clone());
        let (sender, stdout) = mpsc::channel();
        let output = child.0.stdout.take().unwrap();
        std::thread::spawn(move || {
            for line in BufReader::new(output).lines() {
                if sender.send(line).is_err() {
                    break;
                }
            }
        });
        writeln!(child.0.stdin.as_mut().unwrap(), "{}", json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{
            "protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"idle-retention-test","version":"1"}}})).unwrap();
        let mut session = Mcp {
            child,
            stdout,
            stderr,
        };
        wait_until("MCP initialization", || {
            session.stdout.try_recv().ok().map(|line| {
                let response: Value = serde_json::from_str(&line.unwrap()).unwrap();
                assert_eq!(response["id"], 1, "{response}");
                assert!(
                    response.get("result").is_some(),
                    "{response}; {}",
                    session.stderr.text()
                );
            })
        })
        .await;
        writeln!(
            session.child.0.stdin.as_mut().unwrap(),
            "{}",
            json!({"jsonrpc":"2.0", "method":"notifications/initialized"})
        )
        .unwrap();
        session
    }

    async fn entered(&mut self) {
        let path = self.root.path().join("gate/entered.json");
        let marker = wait_until("plugin registration gate", || {
            std::fs::read(&path)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        })
        .await;
        let plugin_pid = Pid::from_u32(marker["pid"].as_u64().unwrap().try_into().unwrap());
        let system = processes(&[plugin_pid]);
        let plugin = system
            .process(plugin_pid)
            .expect("entered plugin must still exist");
        let worker_pid = plugin.parent().expect("worker plugin parent");
        let worker_system = processes(&[worker_pid]);
        let worker = worker_system
            .process(worker_pid)
            .expect("managed worker must still exist");
        self.descendants = vec![
            ProcessIdentity {
                pid: plugin_pid,
                started: plugin.start_time(),
            },
            ProcessIdentity {
                pid: worker_pid,
                started: worker.start_time(),
            },
        ];
    }

    fn release_gate(&self) {
        std::fs::write(self.root.path().join("gate/release"), []).unwrap();
    }

    async fn statuses(&self) -> Vec<Value> {
        self.client
            .get(format!("{}/_nemo_relay/v1/workers/status", self.address))
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap()
            .json::<Value>()
            .await
            .unwrap()["workers"]
            .as_array()
            .unwrap()
            .clone()
    }

    async fn ready(&self) -> Value {
        let deadline = Instant::now() + WAIT;
        loop {
            if let Some(worker) = self
                .statuses()
                .await
                .into_iter()
                .find(|worker| worker["state"] == "ready")
            {
                return worker;
            }
            assert!(
                Instant::now() < deadline,
                "worker did not become ready: {}",
                self.logs.text()
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    async fn descendants_exited(&self) {
        wait_until_deadline(
            "task-owned worker and plugin exit",
            Instant::now() + EXIT_WAIT,
            || {
                self.descendants
                    .iter()
                    .all(|process| !process.present())
                    .then_some(())
            },
        )
        .await;
    }
}

impl Drop for IdleDaemon {
    fn drop(&mut self) {
        if std::thread::panicking() {
            for process in &self.descendants {
                eprintln!(
                    "task-owned process {} remains active: {}",
                    process.pid,
                    process.present()
                );
            }
            for capture in
                std::iter::once(&self.logs).chain(self.mcp_logs.get_mut().unwrap().iter())
            {
                if let Ok(output) = capture.0.lock() {
                    eprintln!(
                        "task-owned process stderr (read error: {:?}):\n{}",
                        output.error,
                        String::from_utf8_lossy(&output.bytes)
                    );
                }
            }
        }
        // Unblock synchronous fixture code before direct-child cleanup, even on panic.
        let _ = std::fs::write(self.root.path().join("gate/release"), []);
        for process in &self.descendants {
            process.cleanup();
        }
        let _ = self.daemon.0.kill();
        let _ = self.daemon.0.wait();
        self.provider.abort();
    }
}

fn write_toml(path: &Path, value: Value) {
    std::fs::write(
        path,
        toml::to_string(&toml::Value::try_from(value).unwrap()).unwrap(),
    )
    .unwrap();
}

fn fixture_digest(path: &Path) -> String {
    format!(
        "sha256:{}",
        Sha256::digest(std::fs::read(path).unwrap())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    )
}

fn account_command(root: &Path, account: &str) -> Command {
    let home = root.join(account);
    let mut command = Command::new(env!("CARGO_BIN_EXE_nemo-relay"));
    // Keep only executable/OS plumbing. Provider credentials, proxies, and the
    // developer's Relay settings must not reach these isolated accounts.
    command.env_clear();
    for name in [
        "PATH",
        "PATHEXT",
        "SystemRoot",
        "WINDIR",
        "TEMP",
        "TMP",
        "TMPDIR",
    ] {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    command
        .current_dir(root)
        .env("HOME", &home)
        .env("USERPROFILE", &home)
        .env("XDG_CONFIG_HOME", home.join(".config"))
        .env("APPDATA", &home)
        .env("LOCALAPPDATA", &home)
        .env("NEMO_RELAY_TEST_SYSTEM_CONFIG_DIR", root.join("system"))
        .env("NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG", "0");
    command
}

async fn wait_until<T>(description: &str, check: impl FnMut() -> Option<T>) -> T {
    wait_until_deadline(description, Instant::now() + WAIT, check).await
}

async fn wait_until_deadline<T>(
    description: &str,
    deadline: Instant,
    mut check: impl FnMut() -> Option<T>,
) -> T {
    loop {
        if let Some(value) = check() {
            return value;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {description}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn short_sessions_reuse_initializing_worker_and_capture_later_call() {
    let mut daemon = IdleDaemon::start(10).await;
    let first = daemon.mcp().await;
    daemon.entered().await;
    #[cfg(unix)]
    first.terminate().await;
    #[cfg(windows)]
    first.finish().await;
    wait_until("initiating MCP disconnect transition", || {
        daemon
            .logs
            .events("mcp_control_disconnected")
            .first()
            .cloned()
    })
    .await;
    daemon.mcp().await.finish().await;
    assert!(daemon.logs.events("worker_added").is_empty());
    assert_eq!(daemon.logs.events("worker_staged").len(), 1);
    assert!(daemon.descendants.iter().all(|process| process.present()));
    daemon.release_gate();
    let ready = daemon.ready().await;
    assert_eq!(ready["reference_count"], 0);
    let worker_id = ready["worker_id"].clone();
    let later = daemon.mcp().await;
    let marker = format!("idle-reuse-{}", uuid::Uuid::now_v7());
    let result: Value = daemon.client.post(format!("{}/v1/chat/completions", daemon.address))
        .header("x-nemo-relay-client-token", &daemon.token)
        .header("x-nemo-relay-session-id", "idle-reuse-session")
        .json(&json!({"model":"idle-model", "messages":[{"role":"user","content":marker}], "stream":false}))
        .send().await.unwrap().error_for_status().unwrap().json().await.unwrap();
    assert_eq!(result["choices"][0]["message"]["content"], marker);
    assert_eq!(daemon.requests.lock().unwrap().len(), 1);
    assert_eq!(daemon.ready().await["worker_id"], worker_id);
    let trace = daemon.root.path().join("traces/events.jsonl");
    wait_until("fresh paired ATOF model scope", || {
        let events: Vec<Value> = std::fs::read_to_string(&trace)
            .unwrap_or_default()
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect();
        events
            .iter()
            .find(|event| {
                event["category"] == "llm"
                    && event["scope_category"] == "end"
                    && event["uuid"].as_str().is_some_and(|uuid| !uuid.is_empty())
                    && event.to_string().contains(&marker)
            })
            .and_then(|end| {
                events.iter().find(|start| {
                    start["category"] == "llm"
                        && start["scope_category"] == "start"
                        && start["uuid"] == end["uuid"]
                })
            })
            .map(|_| ())
    })
    .await;
    later.finish().await;
    assert_eq!(daemon.ready().await["reference_count"], 0);
    daemon.descendants_exited().await;
    assert!(
        daemon
            .statuses()
            .await
            .iter()
            .all(|worker| worker["worker_id"] != worker_id)
    );
    assert_eq!(daemon.logs.events("worker_added").len(), 1);
    std::fs::remove_file(daemon.root.path().join("gate/entered.json")).unwrap();
    let replacement = daemon.mcp().await;
    daemon.entered().await;
    assert_ne!(daemon.ready().await["worker_id"], worker_id);
    replacement.finish().await;
    assert_eq!(daemon.ready().await["reference_count"], 0);
    daemon.descendants_exited().await;
}
