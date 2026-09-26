// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Private-home preparation for callers that retain native process ownership.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::AsyncWriteExt;

use super::*;

const MAX_REQUEST_BYTES: u64 = 1024 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    version: u32,
    agent: String,
    argv: Vec<String>,
    home: PathBuf,
    upstream_url: String,
}

/// Owns a gateway until the private caller connection closes; never spawns an agent.
pub(crate) async fn prepare(inherited: &GatewayOverrides) -> Result<ExitCode, CliError> {
    if inherited.config.is_none() || inherited.plugin_config_path.is_none() {
        return Err(invalid(
            "preparation requires explicit runtime and plugin configuration",
        ));
    }
    let mut input = crate::mcp::spawn_stdin_reader()?;
    let line = tokio::time::timeout(Duration::from_secs(30), input.recv())
        .await
        .map_err(|_| invalid("preparation request timed out"))?
        .ok_or_else(|| invalid("missing preparation request"))??;
    let request: Request =
        serde_json::from_str(&line).map_err(|_| invalid("invalid preparation request"))?;
    let agent = request.validate()?;
    match std::fs::symlink_metadata(request.home.join(crate::hooks::NATIVE_INVOCATION_CONFIG)) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Ok(_) => return Err(invalid("native invocation marker already exists")),
        Err(error) => return Err(error.into()),
    }
    let path = request.home.join(match agent {
        CodingAgent::Codex => "config.toml",
        _ => "settings.json",
    });
    validate_file(&path)?;
    let snapshot = crate::filesystem::snapshot_optional_file(&path).map_err(CliError::Launch)?;
    let metadata = std::fs::symlink_metadata(&request.home)?;
    let mut owned = NativeHome {
        home: request.home.clone(),
        home_metadata: metadata,
        snapshot: Some(snapshot),
        prepared: None,
    };
    let overrides = RunOverrides {
        agent: Some(agent),
        config: inherited.config.clone(),
        openai_base_url: Some(request.upstream_url.clone()),
        anthropic_base_url: Some(request.upstream_url),
        session_metadata: None,
        plugin_config_path: inherited.plugin_config_path.clone(),
        dry_run: false,
        print: false,
        command: request.argv,
    };
    let mut resolved = resolve_run_config(&overrides, Some(inherited))?;
    let plugins_path = crate::configuration::explicit_plugin_config_path(
        inherited.config.as_ref(),
        inherited.plugin_config_path.as_ref(),
    );
    let dynamic_plugins = crate::plugins::lifecycle::active_dynamic_plugin_components(
        plugins_path.as_ref(),
        &resolved,
    )?;
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let gateway_url = format!("http://{address}");
    resolved.gateway.bind = address;
    // Reuse the integration's own generated wiring. This command never calls spawn.
    let prepared = PreparedAgentLaunch::build(
        agent,
        vec![agent.executable().into()],
        0,
        &gateway_url,
        &resolved,
        false,
    )?;
    owned.prepared = Some(prepared);
    let prepared = owned.prepared.as_ref().expect("prepared above");
    let state_dir = crate::bootstrap::state::state_dir().map_err(CliError::Launch)?;
    materialize(agent, prepared, &path, &gateway_url, &state_dir)?;
    let mut environment: BTreeMap<_, _> = prepared.env.iter().cloned().collect();
    if let Ok(path) = std::env::var("NEMO_RELAY_INVOCATION_STATE_DIR") {
        environment.insert("NEMO_RELAY_INVOCATION_STATE_DIR".into(), path);
    }
    let fingerprint = crate::configuration::transparent_gateway_fingerprint(&gateway_url);
    let mut gateway = RunningGateway::start(
        listener,
        resolved.gateway,
        dynamic_plugins,
        fingerprint.clone(),
        prepared.proxy_credential.clone(),
    );
    let mut gateway_observed = false;
    let lifetime = async {
        wait_for_health(&gateway_url, &fingerprint).await?;
        write_message(&json!({"version":1,"environment":environment})).await?;
        tokio::select! {
            read = input.recv() => {
                if read.is_some() { return Err(invalid("owner channel accepts no further messages")); }
                Ok(())
            }
            result = gateway.wait() => {
                gateway_observed = true;
                result?;
                Err(invalid("prepared capture connection stopped"))
            }
        }
    }
    .await;
    // Once wait() consumes the task, stop() must not poll its JoinHandle again.
    let stopped = if gateway_observed {
        Ok(())
    } else {
        gateway.stop().await
    };
    let restored = owned.restore();
    lifetime?;
    stopped?;
    restored?;
    write_message(&json!({"version":1,"cleanup":"complete"})).await?;
    Ok(ExitCode::SUCCESS)
}

impl Request {
    fn validate(&self) -> Result<CodingAgent, CliError> {
        if self.version != 1 {
            return Err(invalid("unsupported preparation version"));
        }
        let agent = match self.agent.as_str() {
            "codex" => CodingAgent::Codex,
            "claude" => CodingAgent::ClaudeCode,
            _ => return Err(invalid("unsupported prepared agent")),
        };
        if !self.home.is_absolute() || self.home.canonicalize()? != self.home {
            return Err(invalid("native home must be an absolute resolved path"));
        }
        let metadata = std::fs::symlink_metadata(&self.home)?;
        if !metadata.is_dir() {
            return Err(invalid("native home must be a directory"));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            // SAFETY: geteuid has no preconditions and does not mutate process state.
            if metadata.mode() & 0o077 != 0 || metadata.uid() != unsafe { libc::geteuid() } {
                return Err(invalid("native home must be owner-private"));
            }
        }
        let variable = match agent {
            CodingAgent::Codex => "CODEX_HOME",
            _ => "CLAUDE_CONFIG_DIR",
        };
        if std::env::var_os(variable).map(PathBuf::from).as_ref() != Some(&self.home) {
            return Err(invalid("native home does not match child environment"));
        }
        let url =
            reqwest::Url::parse(&self.upstream_url).map_err(|_| invalid("invalid upstream URL"))?;
        if !matches!(url.scheme(), "http" | "https")
            || url.host().is_none()
            || !url.username().is_empty()
            || url.password().is_some()
            || url.fragment().is_some()
        {
            return Err(invalid("unsupported upstream URL"));
        }
        validate_argv(agent, &self.argv)?;
        if agent == CodingAgent::ClaudeCode
            && ["CLAUDE_CODE_SAFE_MODE", "CLAUDE_CODE_SIMPLE"]
                .iter()
                .any(|name| {
                    std::env::var(name).is_ok_and(|value| !value.is_empty() && value != "0")
                })
        {
            return Err(invalid("Claude environment disables required hooks"));
        }
        Ok(agent)
    }
}

fn validate_argv(agent: CodingAgent, argv: &[String]) -> Result<(), CliError> {
    if argv.first().and_then(|arg| CodingAgent::infer(arg)) != Some(agent) {
        return Err(invalid("native executable does not match prepared agent"));
    }
    let mut arguments = argv.iter().skip(1);
    while let Some(arg) = arguments.next() {
        if arg == "--" {
            break;
        }
        let (flag, takes_value) = match agent {
            CodingAgent::Codex => (
                matches!(
                    arg.as_str(),
                    "exec"
                        | "--json"
                        | "--ephemeral"
                        | "--skip-git-repo-check"
                        | "--dangerously-bypass-approvals-and-sandbox"
                ),
                matches!(arg.as_str(), "--cd" | "--model" | "-m" | "--sandbox"),
            ),
            _ => (
                matches!(
                    arg.as_str(),
                    "-p" | "--verbose" | "--dangerously-skip-permissions"
                ),
                matches!(
                    arg.as_str(),
                    "--output-format"
                        | "--model"
                        | "--mcp-config"
                        | "--append-system-prompt"
                        | "--allowedTools"
                        | "--disallowedTools"
                        | "--thinking"
                        | "--max-thinking-tokens"
                        | "--max-turns"
                ),
            ),
        };
        if takes_value {
            if arguments.next().is_none() {
                return Err(invalid("native option is missing a value"));
            }
        } else if !flag {
            return Err(invalid("native option can defeat prepared capture"));
        }
    }
    Ok(())
}

fn validate_file(path: &Path) -> Result<(), CliError> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if !metadata.is_file() || metadata.len() > MAX_REQUEST_BYTES => Err(invalid(
            "native configuration must be a bounded regular file",
        )),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn materialize(
    agent: CodingAgent,
    prepared: &PreparedAgentLaunch,
    path: &Path,
    gateway_url: &str,
    state_dir: &Path,
) -> Result<(), CliError> {
    let token = prepared.proxy_credential.expose();
    if agent == CodingAgent::Codex {
        let current = std::fs::read_to_string(path).or_else(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                Ok(String::new())
            } else {
                Err(error)
            }
        })?;
        let mut document: toml_edit::DocumentMut = current
            .parse()
            .map_err(|_| invalid("invalid native Codex configuration"))?;
        if document
            .get("hooks")
            .and_then(toml_edit::Item::as_table_like)
            .is_some_and(|hooks| hooks.iter().any(|(key, _)| key != "state"))
        {
            return Err(invalid(
                "pre-existing native Codex hooks are not supported by private preparation",
            ));
        }
        for pair in prepared.argv[1..].chunks_exact(2) {
            if pair[0] != "--config" {
                return Err(invalid("unexpected generated Codex wiring"));
            }
            let quoted_path = serde_json::to_string(&path.to_string_lossy())
                .map_err(|_| invalid("invalid native path"))?;
            let setting = pair[1].replace(
                "/<session-flags>/config.toml",
                &quoted_path[1..quoted_path.len() - 1],
            );
            let overlay: toml_edit::DocumentMut = setting
                .parse()
                .map_err(|_| invalid("invalid generated Codex wiring"))?;
            merge_toml(document.as_table_mut(), overlay.as_table());
        }
        let provider = document
            .get_mut("model_providers")
            .and_then(toml_edit::Item::as_table_like_mut)
            .and_then(|models| models.get_mut("nemo-relay-openai"))
            .and_then(toml_edit::Item::as_table_like_mut)
            .ok_or_else(|| invalid("missing prepared Codex model provider"))?;
        provider.remove("env_http_headers");
        if provider.get("http_headers").is_none() {
            provider.insert(
                "http_headers",
                toml_edit::Item::Table(toml_edit::Table::new()),
            );
        }
        provider
            .get_mut("http_headers")
            .and_then(toml_edit::Item::as_table_like_mut)
            .ok_or_else(|| invalid("invalid prepared Codex model headers"))?
            .insert(
                crate::provider_auth::TRANSPARENT_PROXY_CREDENTIAL_HEADER,
                toml_edit::value(token),
            );
        crate::filesystem::atomic_write_private(path, document.to_string().as_bytes())
            .map_err(CliError::Launch)?;
    } else {
        let mut settings: Value = match std::fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map_err(|_| invalid("invalid native Claude settings"))?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => json!({}),
            Err(error) => return Err(error.into()),
        };
        let root = prepared
            .temp_dirs
            .first()
            .ok_or_else(|| invalid("missing generated Claude wiring"))?;
        let hooks: Value = serde_json::from_slice(&std::fs::read(root.join("hooks/hooks.json"))?)
            .map_err(|_| invalid("invalid generated Claude hooks"))?;
        settings = crate::hooks::merge_hooks(settings, hooks)?;
        let object = settings
            .as_object_mut()
            .ok_or_else(|| invalid("Claude settings must be an object"))?;
        if object.get("disableAllHooks") == Some(&json!(true)) {
            return Err(invalid("native settings disable hooks"));
        }
        let environment = object
            .entry("env")
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .ok_or_else(|| invalid("Claude settings env must be an object"))?;
        for (key, value) in &prepared.env {
            if key != "ANTHROPIC_BASE_URL" && environment.contains_key(key) {
                return Err(invalid("Claude settings override the prepared environment"));
            }
            if key == "ANTHROPIC_BASE_URL" || key == "ANTHROPIC_CUSTOM_HEADERS" {
                environment.insert(key.clone(), json!(value));
            }
        }
        crate::filesystem::atomic_write_private(
            path,
            &serde_json::to_vec(&settings).map_err(|_| invalid("cannot encode Claude settings"))?,
        )
        .map_err(CliError::Launch)?;
    }
    let root = prepared
        .temp_dirs
        .first()
        .ok_or_else(|| invalid("missing prepared native hook configuration"))?;
    let hook_path = root.join(".nemo-relay-hook-config.json");
    crate::hooks::HookCommandConfig::load(&hook_path)
        .map_err(CliError::Launch)?
        .with_proxy_credential(token)
        .with_invocation_state_dir(state_dir)
        .write(&hook_path)
        .map_err(CliError::Launch)?;
    crate::hooks::HookCommandConfig::transparent(agent, gateway_url)
        .with_proxy_credential(token)
        .with_invocation_state_dir(state_dir)
        .write(&path.with_file_name(crate::hooks::NATIVE_INVOCATION_CONFIG))
        .map_err(CliError::Launch)
}

fn merge_toml(destination: &mut dyn toml_edit::TableLike, source: &dyn toml_edit::TableLike) {
    for (key, value) in source.iter() {
        if let (Some(existing), Some(overlay)) = (
            destination
                .get_mut(key)
                .and_then(toml_edit::Item::as_table_like_mut),
            value.as_table_like(),
        ) {
            merge_toml(existing, overlay);
        } else {
            destination.insert(key, value.clone());
        }
    }
}

struct NativeHome {
    home: PathBuf,
    home_metadata: std::fs::Metadata,
    snapshot: Option<crate::filesystem::FileSnapshot>,
    prepared: Option<PreparedAgentLaunch>,
}

impl NativeHome {
    fn restore(&mut self) -> Result<(), CliError> {
        let configuration = match std::fs::symlink_metadata(&self.home) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                self.snapshot.take();
                Ok(())
            }
            Ok(current) if same_private_home(&self.home_metadata, &current) => {
                let revoked = crate::filesystem::atomic_write_private(
                    &self.home.join(crate::hooks::NATIVE_INVOCATION_CONFIG),
                    br#"{"version":1,"revoked":true}"#,
                )
                .map_err(CliError::Launch);
                let restored = self
                    .snapshot
                    .take()
                    .map(|snapshot| {
                        crate::filesystem::restore_file_snapshot(&snapshot)
                            .map_err(CliError::Launch)
                    })
                    .unwrap_or(Ok(()));
                revoked.and(restored)
            }
            Ok(_) => Err(invalid("native home changed before preparation cleanup")),
            Err(error) => Err(error.into()),
        };
        let temporary = self
            .prepared
            .take()
            .map(|prepared| prepared.restore())
            .unwrap_or(Ok(()));
        configuration.and(temporary)
    }
}

fn same_private_home(original: &std::fs::Metadata, current: &std::fs::Metadata) -> bool {
    if !current.is_dir() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        current.uid() == unsafe { libc::geteuid() }
            && current.mode() & 0o077 == 0
            && original.dev() == current.dev()
            && original.ino() == current.ino()
    }
    #[cfg(not(unix))]
    {
        let _ = original;
        true
    }
}

impl Drop for NativeHome {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

async fn write_message(value: &Value) -> Result<(), CliError> {
    let mut bytes =
        serde_json::to_vec(value).map_err(|_| invalid("cannot encode lifecycle result"))?;
    bytes.push(b'\n');
    let mut output = tokio::io::stdout();
    output.write_all(&bytes).await?;
    output.flush().await?;
    Ok(())
}

fn invalid(message: &str) -> CliError {
    CliError::Launch(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attachment_materializes_private_codex_model_and_hook_credentials() {
        let temporary = tempfile::tempdir().unwrap();
        let home = temporary.path().join("codex");
        std::fs::create_dir(&home).unwrap();
        let config = home.join("config.toml");
        std::fs::write(&config, "model_provider=\"gym\"\n").unwrap();
        let hooks = temporary.path().join("hooks");
        std::fs::create_dir(&hooks).unwrap();
        crate::hooks::HookCommandConfig::transparent(CodingAgent::Codex, "http://127.0.0.1:4567")
            .write(&hooks.join(".nemo-relay-hook-config.json"))
            .unwrap();
        let prepared = PreparedAgentLaunch {
            argv: vec![
                "codex".into(),
                "--config".into(),
                "model_provider=\"nemo-relay-openai\"".into(),
                "--config".into(),
                "model_providers.nemo-relay-openai={name=\"relay\",base_url=\"http://127.0.0.1:4567\",wire_api=\"responses\",env_http_headers={\"x-nemo-relay-proxy-token\"=\"NEMO_RELAY_PROXY_CREDENTIAL\"}}".into(),
            ],
            host_index: 0,
            env: vec![],
            temp_dirs: vec![hooks.clone()],
            notes: vec![],
            non_tty_warnings: vec![],
            proxy_credential: crate::provider_auth::TransparentProxyCredential::generate().unwrap(),
            secret_env_names: vec![],
        };
        materialize(
            CodingAgent::Codex,
            &prepared,
            &config,
            "http://127.0.0.1:4567",
            &temporary.path().join("state"),
        )
        .unwrap();
        let rendered: toml_edit::DocumentMut =
            std::fs::read_to_string(&config).unwrap().parse().unwrap();
        let provider = &rendered["model_providers"]["nemo-relay-openai"];
        assert!(provider.get("env_http_headers").is_none());
        assert_eq!(
            provider["http_headers"][crate::provider_auth::TRANSPARENT_PROXY_CREDENTIAL_HEADER]
                .as_str(),
            Some(prepared.proxy_credential.expose())
        );
        let hook =
            crate::hooks::HookCommandConfig::load(&hooks.join(".nemo-relay-hook-config.json"))
                .unwrap();
        assert_eq!(
            hook.prepared_gateway(CodingAgent::Codex).unwrap(),
            "http://127.0.0.1:4567"
        );
        let marker = crate::hooks::HookCommandConfig::load(
            &home.join(crate::hooks::NATIVE_INVOCATION_CONFIG),
        )
        .unwrap();
        assert_eq!(
            marker.prepared_gateway(CodingAgent::Codex).unwrap(),
            "http://127.0.0.1:4567"
        );
    }

    #[test]
    fn attachment_cleanup_does_not_recreate_a_deleted_native_home() {
        let temporary = tempfile::tempdir().unwrap();
        let home = temporary.path().join("codex");
        std::fs::create_dir(&home).unwrap();
        let config = home.join("config.toml");
        std::fs::write(&config, "model_provider=\"gym\"\n").unwrap();
        let mut owned = NativeHome {
            home: home.clone(),
            home_metadata: std::fs::symlink_metadata(&home).unwrap(),
            snapshot: Some(crate::filesystem::snapshot_optional_file(&config).unwrap()),
            prepared: None,
        };
        std::fs::remove_dir_all(&home).unwrap();
        owned.restore().unwrap();
        assert!(!home.exists());
    }

    #[test]
    fn attachment_merges_native_plugin_hook_trust() {
        let mut original: toml_edit::DocumentMut =
            "[hooks.state]\nforeign={enabled=true}\n".parse().unwrap();
        let overlay: toml_edit::DocumentMut = "hooks.state={local={enabled=true}}".parse().unwrap();
        merge_toml(original.as_table_mut(), overlay.as_table());
        assert!(original["hooks"]["state"].get("foreign").is_some());
        assert!(original["hooks"]["state"].get("local").is_some());
    }

    #[test]
    fn attachment_rejects_capture_overrides_before_preparation() {
        for (agent, args) in [
            (
                CodingAgent::Codex,
                vec!["codex", "exec", "-c", "model_provider=other"],
            ),
            (CodingAgent::ClaudeCode, vec!["claude", "-p", "--bare"]),
            (
                CodingAgent::ClaudeCode,
                vec!["claude", "-p", "--settings", "other.json"],
            ),
        ] {
            assert!(
                validate_argv(
                    agent,
                    &args.into_iter().map(String::from).collect::<Vec<_>>()
                )
                .is_err()
            );
        }
        assert!(
            validate_argv(
                CodingAgent::Codex,
                &["codex", "exec", "--json", "--", "--config is prompt text"].map(String::from)
            )
            .is_ok()
        );
        assert!(
            validate_argv(
                CodingAgent::ClaudeCode,
                &["claude", "-p", "--model", "test", "--", "prompt"].map(String::from)
            )
            .is_ok()
        );
    }
}
