// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal synchronous client for the stable Codex app-server hook APIs.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(super) struct CodexHookMetadata {
    pub(super) key: String,
    pub(super) event_name: String,
    pub(super) handler_type: String,
    pub(super) command: Option<String>,
    pub(super) source_path: String,
    pub(super) source: String,
    #[serde(default)]
    pub(super) plugin_id: Option<String>,
    pub(super) enabled: bool,
    pub(super) current_hash: String,
    pub(super) trust_status: String,
}

pub(super) trait CodexHooksClient {
    fn list_hooks(&mut self, cwd: &Path) -> Result<Vec<CodexHookMetadata>, String>;
    fn trust_hooks(&mut self, hooks: &[CodexHookMetadata]) -> Result<(), String>;
    fn clear_hook_trust(&mut self, keys: &[String]) -> Result<(), String>;
    fn restore_hook_trust(&mut self, state: &[(String, Option<Value>)]) -> Result<(), String>;
}

pub(super) struct CodexAppServerClient {
    child: Child,
    stdin: ChildStdin,
    messages: Receiver<Result<Value, String>>,
    next_id: u64,
}

impl CodexAppServerClient {
    pub(super) fn start() -> Result<Self, String> {
        let mut command = codex_app_server_command();
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| format!("failed to start `codex app-server`: {error}"))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "failed to open Codex app-server stdin".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "failed to open Codex app-server stdout".to_string())?;
        let (sender, messages) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let parsed = line
                    .map_err(|error| format!("failed to read Codex app-server response: {error}"))
                    .and_then(|line| {
                        serde_json::from_str(&line)
                            .map_err(|error| format!("invalid JSON from Codex app-server: {error}"))
                    });
                if sender.send(parsed).is_err() {
                    break;
                }
            }
        });
        let mut client = Self {
            child,
            stdin,
            messages,
            next_id: 1,
        };
        client.request(
            "initialize",
            json!({
                "clientInfo": {
                    "name": "nemo_relay",
                    "title": "NeMo Relay",
                    "version": env!("CARGO_PKG_VERSION")
                }
            }),
        )?;
        client.notify("initialized", None)?;
        Ok(client)
    }

    fn request(&mut self, method: &str, params: Value) -> Result<Value, String> {
        let id = self.next_id;
        self.next_id += 1;
        self.write_message(&json!({"method": method, "id": id, "params": params}))?;
        let deadline = Instant::now() + REQUEST_TIMEOUT;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(format!(
                    "timed out waiting for Codex app-server `{method}` response"
                ));
            }
            let message = self.messages.recv_timeout(remaining).map_err(|error| {
                format!("timed out waiting for Codex app-server `{method}` response: {error}")
            })??;
            if message.get("id").and_then(Value::as_u64) != Some(id) {
                continue;
            }
            if let Some(error) = message.get("error") {
                return Err(format!("Codex app-server `{method}` failed: {error}"));
            }
            return message
                .get("result")
                .cloned()
                .ok_or_else(|| format!("Codex app-server `{method}` response had no result"));
        }
    }

    fn notify(&mut self, method: &str, params: Option<Value>) -> Result<(), String> {
        let mut message = json!({"method": method});
        if let Some(params) = params {
            message["params"] = params;
        }
        self.write_message(&message)
    }

    fn write_message(&mut self, message: &Value) -> Result<(), String> {
        serde_json::to_writer(&mut self.stdin, message)
            .map_err(|error| format!("failed to encode Codex app-server request: {error}"))?;
        self.stdin
            .write_all(b"\n")
            .and_then(|_| self.stdin.flush())
            .map_err(|error| format!("failed to write Codex app-server request: {error}"))
    }

    fn batch_write(&mut self, edits: Vec<Value>) -> Result<(), String> {
        self.request(
            "config/batchWrite",
            json!({"edits": edits, "reloadUserConfig": true}),
        )?;
        Ok(())
    }
}

impl CodexHooksClient for CodexAppServerClient {
    fn list_hooks(&mut self, cwd: &Path) -> Result<Vec<CodexHookMetadata>, String> {
        let response = self.request("hooks/list", json!({"cwds": [cwd]}))?;
        let entry = response
            .get("data")
            .and_then(Value::as_array)
            .and_then(|data| data.first())
            .ok_or_else(|| "Codex app-server returned no hook-list entry".to_string())?;
        if let Some(errors) = entry.get("errors").and_then(Value::as_array)
            && !errors.is_empty()
        {
            return Err(format!("Codex app-server could not load hooks: {errors:?}"));
        }
        serde_json::from_value(entry.get("hooks").cloned().unwrap_or_else(|| json!([])))
            .map_err(|error| format!("invalid hooks/list response from Codex app-server: {error}"))
    }

    fn trust_hooks(&mut self, hooks: &[CodexHookMetadata]) -> Result<(), String> {
        let state = hooks
            .iter()
            .fold(serde_json::Map::new(), |mut state, hook| {
                state.insert(
                    hook.key.clone(),
                    json!({"trusted_hash": hook.current_hash, "enabled": true}),
                );
                state
            });
        self.batch_write(vec![json!({
            "keyPath": "hooks.state",
            "value": state,
            "mergeStrategy": "upsert"
        })])
    }

    fn clear_hook_trust(&mut self, keys: &[String]) -> Result<(), String> {
        let edits = keys
            .iter()
            .map(|key| {
                json!({
                    "keyPath": hook_state_key_path(key),
                    "value": null,
                    "mergeStrategy": "upsert"
                })
            })
            .collect();
        self.batch_write(edits)
    }

    fn restore_hook_trust(&mut self, state: &[(String, Option<Value>)]) -> Result<(), String> {
        let edits = state
            .iter()
            .map(|(key, value)| {
                json!({
                    "keyPath": hook_state_key_path(key),
                    "value": value,
                    "mergeStrategy": "upsert"
                })
            })
            .collect();
        self.batch_write(edits)
    }
}

impl Drop for CodexAppServerClient {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub(super) fn hook_state_key_path(key: &str) -> String {
    let quoted = serde_json::to_string(key).expect("serializing a string cannot fail");
    format!("hooks.state.{quoted}")
}

fn codex_app_server_command() -> Command {
    crate::agent_process::std_command(&["codex".into(), "app-server".into()])
}
