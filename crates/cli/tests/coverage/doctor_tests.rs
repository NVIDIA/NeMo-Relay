// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::ffi::OsString;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::config::ResolvedDynamicPluginConfig;

fn start_doctor_http_capture_server() -> (String, Arc<Mutex<String>>, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let body = Arc::new(Mutex::new(String::new()));
    let thread_body = Arc::clone(&body);
    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut data = Vec::new();
        let mut buf = [0_u8; 1];
        while !data.ends_with(b"\r\n\r\n") {
            stream.read_exact(&mut buf).unwrap();
            data.push(buf[0]);
        }
        let headers = String::from_utf8_lossy(&data).to_string();
        let length = headers
            .lines()
            .find_map(|line| {
                line.split_once(':').and_then(|(name, value)| {
                    name.eq_ignore_ascii_case("content-length")
                        .then_some(value.trim())
                })
            })
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap();
        let mut request_body = vec![0_u8; length];
        stream.read_exact(&mut request_body).unwrap();
        *thread_body.lock().unwrap() = String::from_utf8(request_body).unwrap();
        stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
            .unwrap();
    });
    (url, body, handle)
}

struct EnvScope {
    _guard: std::sync::MutexGuard<'static, ()>,
    values: Vec<(&'static str, Option<OsString>)>,
}

impl EnvScope {
    fn set(values: &[(&'static str, Option<&std::ffi::OsStr>)]) -> Self {
        let guard = crate::test_support::ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let previous = values
            .iter()
            .map(|(key, _)| (*key, std::env::var_os(key)))
            .collect::<Vec<_>>();
        for (key, value) in values {
            unsafe {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
        Self {
            _guard: guard,
            values: previous,
        }
    }
}

impl Drop for EnvScope {
    fn drop(&mut self) {
        for (key, value) in self.values.drain(..) {
            unsafe {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
    }
}

fn empty_report() -> DoctorReport {
    DoctorReport {
        schema_version: 1,
        binary_version: "0.0.0-test",
        target_agent: None,
        environment: EnvironmentInfo {
            os: "macos 25.3.0".into(),
            arch: "aarch64",
            shell: Some("zsh".into()),
        },
        configuration: ConfigurationInfo {
            workspace: ConfigLayer {
                path: PathBuf::from("/x/.nemo-relay/config.toml"),
                status: Status::Info,
                active: false,
                details: "not present".into(),
            },
            global: ConfigLayer {
                path: PathBuf::from("/x/.config/nemo-relay/config.toml"),
                status: Status::Info,
                active: false,
                details: "not present".into(),
            },
            system: ConfigLayer {
                path: PathBuf::from("/etc/nemo-relay/config.toml"),
                status: Status::Info,
                active: false,
                details: "not present".into(),
            },
            plugin_configs: vec![],
            plugin_resolution: Check {
                name: "Plugin resolution",
                status: Status::Info,
                details: "plugins.toml not configured".into(),
            },
            resolution: Check {
                name: "Resolution",
                status: Status::Pass,
                details: "valid".into(),
            },
            default_agent: None,
            configured_agents: vec![],
            dynamic_plugins: vec![],
        },
        agents: vec![],
        host_plugins: vec![],
        observability: vec![],
        completions: vec![],
    }
}

#[test]
fn exit_code_passes_when_no_failures() {
    let report = empty_report();
    assert_eq!(exit_code(&report), 0);
}

#[test]
fn exit_code_fails_when_observability_check_fails() {
    let mut report = empty_report();
    report.observability.push(Check {
        name: "ATIF dir",
        status: Status::Fail,
        details: "not writable".into(),
    });
    assert_eq!(exit_code(&report), 1);
}

#[test]
fn exit_code_passes_with_warn_only() {
    let mut report = empty_report();
    report.observability.push(Check {
        name: "OpenInference endpoint",
        status: Status::Warn,
        details: "HTTP 500".into(),
    });
    assert_eq!(exit_code(&report), 0);
}

#[test]
fn exit_code_fails_when_workspace_config_is_invalid() {
    let mut report = empty_report();
    report.configuration.workspace.status = Status::Fail;
    report.configuration.workspace.details = "invalid TOML".into();
    assert_eq!(exit_code(&report), 1);
}

#[test]
fn exit_code_fails_when_config_resolution_fails() {
    let mut report = empty_report();
    report.configuration.resolution.status = Status::Fail;
    report.configuration.resolution.details = "invalid gateway configuration shape".into();
    assert_eq!(exit_code(&report), 1);
}

#[test]
fn exit_code_fails_when_agent_readiness_fails() {
    let mut report = empty_report();
    report.agents.push(AgentInfo {
        name: "codex",
        status: Status::Fail,
        configured: true,
        command: "codex".into(),
        path: None,
        version: None,
        annotation: "configured command not found on $PATH".into(),
    });
    assert_eq!(exit_code(&report), 1);
}

#[test]
fn exit_code_fails_when_an_installed_host_plugin_is_unready() {
    let mut report = empty_report();
    report
        .host_plugins
        .push(crate::plugin_install::HostPluginReadiness {
            host: "codex".into(),
            remediation: "nemo-relay install codex --force".into(),
            state_path: PathBuf::from("/tmp/codex.json"),
            marketplace: Some(PathBuf::from("/tmp/codex-marketplace")),
            plugin: Some(PathBuf::from(
                "/tmp/codex-marketplace/plugins/nemo-relay-plugin",
            )),
            checks: vec![crate::plugin_install::HostPluginReadinessCheck {
                name: "Host CLI".into(),
                ok: false,
                details: "required `codex` CLI was not found on PATH".into(),
            }],
            relay: None,
            host_plugin_registered: None,
            host_marketplace_registered: None,
            plugin_setup: None,
        });

    assert_eq!(exit_code(&report), 1);
    let rendered = format_human(&report);
    assert!(rendered.contains("Host plugins"));
    assert!(rendered.contains("repair: nemo-relay install codex --force"));
    let json: serde_json::Value = serde_json::from_str(&format_json(&report).unwrap()).unwrap();
    assert_eq!(json["schema_version"], 1);
    assert_eq!(json["host_plugins"][0]["checks"][0]["ok"], false);
    assert_eq!(
        json["host_plugins"][0]["remediation"],
        "nemo-relay install codex --force"
    );
}

#[test]
fn format_human_emits_fixed_section_order() {
    let report = empty_report();
    let rendered = format_human(&report);

    // Locking in the section order so users can diff `doctor` output across machines.
    let env_idx = rendered.find("Environment").expect("Environment header");
    let cfg_idx = rendered
        .find("Configuration")
        .expect("Configuration header");
    let plugins_idx = rendered
        .find("Plugin configuration")
        .expect("Plugin configuration header");
    let agents_idx = rendered.find("Agents detected").expect("Agents header");
    let obs_idx = rendered
        .find("Observability")
        .expect("Observability header");
    let comp_idx = rendered.find("Completions").expect("Completions header");

    assert!(env_idx < cfg_idx);
    assert!(cfg_idx < plugins_idx);
    assert!(plugins_idx < agents_idx);
    assert!(agents_idx < obs_idx);
    assert!(obs_idx < comp_idx);
}

#[test]
fn format_human_distinguishes_plugin_files_from_plugin_resolution() {
    let mut report = empty_report();
    report.configuration.plugin_configs.push(ConfigLayer {
        path: PathBuf::from("/tmp/plugins.toml"),
        status: Status::Pass,
        active: true,
        details: "discovered and contributes to plugin resolution".into(),
    });

    let rendered = format_human(&report);

    assert!(rendered.contains("Plugin files /tmp/plugins.toml"));
    assert!(rendered.contains("Plugins    · plugins.toml not configured"));
}

#[test]
fn format_human_reports_all_checks_passed_on_clean_report() {
    let report = empty_report();
    let rendered = format_human(&report);
    assert!(rendered.contains("All checks passed."));
    assert!(!rendered.contains("warnings"));
}

#[test]
fn format_human_uses_symbols_for_agent_statuses() {
    let mut report = empty_report();
    report.agents = vec![
        AgentInfo {
            name: "claude",
            status: Status::Pass,
            configured: true,
            command: "claude".into(),
            path: Some(PathBuf::from("/bin/claude")),
            version: Some("1.0.0".into()),
            annotation: "hooks: injected during run".into(),
        },
        AgentInfo {
            name: "codex",
            status: Status::Info,
            configured: false,
            command: "codex".into(),
            path: None,
            version: None,
            annotation: "not configured".into(),
        },
    ];

    let rendered = format_human(&report);

    assert!(rendered.contains("    ✓  claude"));
    assert!(rendered.contains("    ·  codex"));
    assert!(!rendered.contains("    pass "));
    assert!(!rendered.contains("    info "));
}

#[test]
fn format_human_reports_failure_summary_when_anything_failed() {
    let mut report = empty_report();
    report.observability.push(Check {
        name: "ATIF dir",
        status: Status::Fail,
        details: "not writable".into(),
    });
    let rendered = format_human(&report);
    assert!(rendered.contains("Some checks FAILED"));
}

#[test]
fn format_human_reports_config_resolution_failure() {
    let mut report = empty_report();
    report.configuration.resolution.status = Status::Fail;
    report.configuration.resolution.details =
        "could not resolve merged config: invalid plugin TOML".into();

    let rendered = format_human(&report);

    assert!(rendered.contains("Resolution ✗ could not resolve merged config"));
    assert!(rendered.contains("Some checks FAILED"));
}

#[test]
fn format_human_distinguishes_pass_with_warnings_from_clean_pass() {
    let mut report = empty_report();
    report.observability.push(Check {
        name: "ATIF dir",
        status: Status::Warn,
        details: "directory missing — will be created on first write".into(),
    });
    let rendered = format_human(&report);
    // Exit code stays 0 (warns don't fail), but the footer must call out that warnings exist
    // so users aren't lulled by an "All checks passed." string.
    assert!(rendered.contains("All checks passed"));
    assert!(
        rendered.contains("warnings"),
        "warn-only report should surface the word `warnings` in the footer, got:\n{rendered}"
    );
}

#[test]
fn format_json_is_stable_and_versioned() {
    let report = empty_report();
    let json = format_json(&report).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    // schema_version pins the wire format. Bump only on breaking renames/removals.
    assert_eq!(parsed["schema_version"], 1);
    assert!(parsed["target_agent"].is_null());
    assert!(parsed["environment"]["os"].is_string());
    assert!(parsed["agents"].is_array());
    assert!(parsed["configuration"]["dynamic_plugins"].is_array());
}

#[test]
fn format_json_reports_discovered_dynamic_plugin_fields() {
    let mut report = empty_report();
    report.configuration.dynamic_plugins = vec![DynamicPluginReferenceInfo {
        plugin_id: "acme.worker".into(),
        manifest_ref: "/tmp/plugins/acme/relay-plugin.toml".into(),
        source: PathBuf::from("/tmp/plugins.toml"),
        host_config_status: DynamicPluginHostConfigStatus::Present,
    }];

    let json = format_json(&report).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    let plugin = &parsed["configuration"]["dynamic_plugins"][0];

    assert_eq!(plugin["plugin_id"], "acme.worker");
    assert_eq!(
        plugin["manifest_ref"],
        "/tmp/plugins/acme/relay-plugin.toml"
    );
    assert_eq!(plugin["source"], "/tmp/plugins.toml");
    assert_eq!(plugin["host_config_status"], "present");
}

#[test]
fn check_dir_writable_does_not_create_missing_dir() {
    let temp = tempfile::tempdir().unwrap();
    let missing = temp.path().join("missing-atif");

    assert!(check_dir_writable(&missing).is_err());
    assert!(
        !missing.exists(),
        "doctor should not create missing ATIF directories while probing"
    );
}

#[test]
fn layer_status_reports_missing_valid_invalid_and_non_directory_paths() {
    let temp = tempfile::tempdir().unwrap();
    let missing = temp.path().join("missing.toml");
    assert_eq!(layer_status(&missing).status, Status::Info);

    let valid = temp.path().join("config.toml");
    std::fs::write(&valid, "[upstream]\nopenai_base_url = \"http://local\"\n").unwrap();
    let valid_layer = layer_status(&valid);
    assert_eq!(valid_layer.status, Status::Pass);
    assert!(valid_layer.active);

    let invalid = temp.path().join("invalid.toml");
    std::fs::write(&invalid, "[upstream\n").unwrap();
    let invalid_layer = layer_status(&invalid);
    assert_eq!(invalid_layer.status, Status::Fail);
    assert!(invalid_layer.details.contains("invalid TOML"));

    let dir = temp.path().join("config-dir");
    std::fs::create_dir(&dir).unwrap();
    let dir_layer = layer_status(&dir);
    assert_eq!(dir_layer.status, Status::Fail);
    assert!(
        dir_layer.details.contains("unreadable") || dir_layer.details.contains("Is a directory")
    );
}

#[test]
fn plugin_layer_status_marks_a_discovered_plugins_toml_as_contributing() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("plugins.toml");
    std::fs::write(&path, "version = 1\ncomponents = []\n").unwrap();

    let layer = plugin_layer_status(&path, std::slice::from_ref(&path), None);

    assert_eq!(layer.status, Status::Pass);
    assert!(layer.active);
    assert!(layer.details.contains("contributes to plugin resolution"));
}

#[test]
fn plugin_layer_status_marks_non_contributing_files_as_inactive() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("plugins.toml");
    std::fs::write(&path, "version = 1\ncomponents = []\n").unwrap();

    let layer = plugin_layer_status(&path, &[], None);

    assert_eq!(layer.status, Status::Pass);
    assert!(!layer.active);
    assert!(layer.details.contains("does not contribute"));
}

#[test]
fn plugin_layer_status_surfaces_source_specific_semantic_errors() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("plugins.toml");
    std::fs::write(&path, "[plugins]\ndynamic = 1\n").unwrap();
    let error = format!("invalid dynamic plugin config in {}", path.display());

    let layer = plugin_layer_status(&path, &[], Some(&error));

    assert_eq!(layer.status, Status::Fail);
    assert!(!layer.active);
    assert!(layer.details.contains("invalid plugin configuration"));
}

#[test]
fn plugin_resolution_check_covers_all_resolution_outcomes() {
    let valid = Check {
        name: "Resolution",
        status: Status::Pass,
        details: "valid".into(),
    };
    let failed = Check {
        name: "Resolution",
        status: Status::Fail,
        details: "merged configuration failed".into(),
    };
    let mut with_runtime_config = ResolvedConfig::default();
    with_runtime_config.gateway.plugin_config = Some(serde_json::json!({"version": 1}));
    let mut with_dynamic_plugin = ResolvedConfig::default();
    with_dynamic_plugin
        .dynamic_plugins
        .push(ResolvedDynamicPluginConfig {
            plugin_id: "acme.worker".into(),
            manifest_ref: "/tmp/relay-plugin.toml".into(),
            config: serde_json::Map::new(),
            has_explicit_config: false,
            source: PathBuf::from("/tmp/plugins.toml"),
        });

    let cases = [
        (
            ResolvedConfig::default(),
            &valid,
            Some("invalid plugin TOML"),
            Status::Fail,
            "could not resolve plugins.toml",
        ),
        (
            ResolvedConfig::default(),
            &failed,
            None,
            Status::Fail,
            "merged configuration failed",
        ),
        (
            with_runtime_config,
            &valid,
            None,
            Status::Info,
            "effective plugin configuration loaded",
        ),
        (
            with_dynamic_plugin,
            &valid,
            None,
            Status::Info,
            "dynamic plugin configuration loaded",
        ),
        (
            ResolvedConfig::default(),
            &valid,
            None,
            Status::Info,
            "plugins.toml not configured",
        ),
    ];

    for (resolved, resolution, plugin_error, status, detail) in cases {
        let check = plugin_resolution_check(&resolved, resolution, plugin_error);
        assert_eq!(check.name, "Plugin resolution");
        assert_eq!(check.status, status);
        assert!(check.details.contains(detail));
    }
}

#[test]
fn collect_configuration_uses_xdg_global_path_and_renders_resolution_branches() {
    let temp = tempfile::tempdir().unwrap();
    let workspace = temp.path().join("workspace");
    let workspace_config = workspace.join(".nemo-relay/config.toml");
    std::fs::create_dir_all(workspace_config.parent().unwrap()).unwrap();
    std::fs::write(
        &workspace_config,
        "[upstream]\nopenai_base_url = \"http://local\"\n",
    )
    .unwrap();

    let xdg = temp.path().join("xdg");
    let global_config = xdg.join("nemo-relay/config.toml");
    std::fs::create_dir_all(global_config.parent().unwrap()).unwrap();
    std::fs::write(&global_config, "[upstream\n").unwrap();
    let home = temp.path().join("home");
    std::fs::create_dir(&home).unwrap();
    let _env = EnvScope::set(&[
        ("XDG_CONFIG_HOME", Some(xdg.as_os_str())),
        ("HOME", Some(home.as_os_str())),
        ("USERPROFILE", None),
    ]);

    let configuration = collect_configuration(
        Some(&workspace),
        Some(&home),
        Check {
            name: "Resolution",
            status: Status::Warn,
            details: "using fallback layer".into(),
        },
        vec!["codex".into(), "hermes".into()],
        &[],
        &PluginConfigurationDiagnostics {
            sources: vec![],
            error: None,
            resolution: Check {
                name: "Plugin resolution",
                status: Status::Pass,
                details: "valid".into(),
            },
        },
    );

    assert_eq!(configuration.workspace.status, Status::Pass);
    assert!(configuration.workspace.active);
    assert_eq!(configuration.global.path, global_config);
    assert_eq!(configuration.global.status, Status::Fail);
    assert!(configuration.global.details.contains("invalid TOML"));

    let mut report = empty_report();
    report.configuration = configuration;
    let rendered = format_human(&report);
    assert!(rendered.contains("Global"));
    assert!(rendered.contains("Resolution ! using fallback layer"));
    assert!(rendered.contains("Agents     codex, hermes"));
}

#[test]
fn agent_helper_statuses_cover_configured_target_and_hook_paths() {
    assert_eq!(command_executable("codex --full-auto"), "codex");
    assert_eq!(command_executable(""), "");
    assert_eq!(
        agent_command(CodingAgent::ClaudeCode, &AgentConfigs::default(), "claude"),
        "claude"
    );
    assert_eq!(
        agent_command_status(Some(std::path::Path::new("/bin/codex")), false, true),
        Status::Warn
    );
    assert_eq!(agent_command_status(None, true, false), Status::Fail);
    assert_eq!(
        combine_status(Status::Pass, Status::Warn, true),
        Status::Warn
    );
    assert_eq!(
        combine_status(Status::Pass, Status::Warn, false),
        Status::Pass
    );

    let mut agents = AgentConfigs::default();
    agents.hermes.hooks_path = Some(PathBuf::from("/tmp/hermes.yaml"));
    assert!(agent_configured(CodingAgent::Hermes, &agents));
    assert_eq!(configured_agent_names(&agents), vec!["hermes".to_string()]);
    assert_eq!(
        hook_status(CodingAgent::ClaudeCode, &agents, true),
        (Status::Pass, "hooks: injected during run".into())
    );
    assert_eq!(
        hook_status(CodingAgent::Codex, &agents, true),
        (Status::Pass, "hooks: injected during run".into())
    );
    let temp = tempfile::tempdir().unwrap();
    let hook = temp.path().join("hooks.yaml");
    std::fs::write(&hook, "cmd: nemo-relay hook-forward hermes\n").unwrap();
    let (status, details) = hook_file_status(Ok(hook.clone()), CodingAgent::Hermes, true, "hooks");
    assert_eq!(status, Status::Pass);
    assert!(details.contains(hook.to_str().unwrap()));

    std::fs::write(&hook, "cmd: custom\n").unwrap();
    let (status, details) = hook_file_status(Ok(hook.clone()), CodingAgent::Hermes, true, "hooks");
    assert_eq!(status, Status::Fail);
    assert!(details.contains("missing NeMo Relay hook"));
    let (status, _) = hook_file_status(Ok(hook), CodingAgent::Hermes, false, "hooks");
    assert_eq!(status, Status::Info);

    let agents = AgentConfigs::default();
    let (status, details) = hook_status(CodingAgent::Hermes, &agents, true);
    assert_eq!(status, Status::Fail);
    assert!(details.contains("not installed"));
    let (status, details) = hook_status(CodingAgent::Hermes, &agents, false);
    assert_eq!(status, Status::Info);
    assert!(details.contains("not configured"));
}

#[test]
fn collect_completions_reports_shell_specific_paths() {
    let temp = tempfile::tempdir().unwrap();
    let zsh_completion = temp.path().join(".zfunc/_nemo-relay");
    std::fs::create_dir_all(zsh_completion.parent().unwrap()).unwrap();
    std::fs::write(&zsh_completion, "#compdef nemo-relay\n").unwrap();

    let _env = EnvScope::set(&[("SHELL", Some(std::ffi::OsStr::new("/bin/zsh")))]);
    let checks = collect_completions(Some(temp.path()));
    assert_eq!(checks[0].status, Status::Pass);
    assert!(checks[0].details.contains("_nemo-relay"));

    drop(_env);
    let _env = EnvScope::set(&[("SHELL", Some(std::ffi::OsStr::new("/bin/fish")))]);
    let checks = collect_completions(Some(temp.path()));
    assert_eq!(checks[0].status, Status::Info);
    assert!(checks[0].details.contains("nemo-relay.fish"));

    drop(_env);
    let _env = EnvScope::set(&[("SHELL", None)]);
    let checks = collect_completions(Some(temp.path()));
    assert_eq!(checks[0].status, Status::Info);
    assert!(checks[0].details.contains("no $SHELL"));
}

#[test]
fn collect_environment_and_completions_cover_missing_home_and_unknown_shell() {
    let _env = EnvScope::set(&[("SHELL", Some(std::ffi::OsStr::new("/opt/bin/elvish")))]);
    let environment = collect_environment();
    assert_eq!(environment.shell.as_deref(), Some("elvish"));

    let checks = collect_completions(None);
    assert_eq!(checks[0].status, Status::Info);
    assert!(checks[0].details.contains("could not resolve home dir"));

    drop(_env);
    let _env = EnvScope::set(&[("SHELL", Some(std::ffi::OsStr::new("/bin/nu")))]);
    let home = tempfile::tempdir().unwrap();
    let checks = collect_completions(Some(home.path()));
    assert_eq!(checks[0].status, Status::Info);
    assert!(checks[0].details.contains("no known completion path"));
}

#[cfg(unix)]
#[tokio::test]
#[allow(clippy::await_holding_lock)]
async fn collect_agents_filters_target_and_records_version() {
    let temp = tempfile::tempdir().unwrap();
    let codex = temp.path().join("codex");
    std::fs::write(&codex, "#!/bin/sh\nprintf 'codex 1.2.3\\n'\n").unwrap();
    make_executable(&codex);

    let mut resolved = ResolvedConfig::default();
    resolved.agents.codex.command = Some(codex.to_string_lossy().into_owned());
    let agents = collect_agents(Some(CodingAgent::Codex), &resolved).await;

    assert_eq!(agents.len(), 1);
    assert_eq!(agents[0].name, "codex");
    assert_eq!(agents[0].status, Status::Pass);
    assert_eq!(agents[0].path.as_deref(), Some(codex.as_path()));
    assert_eq!(agents[0].version.as_deref(), Some("codex 1.2.3"));
}

#[cfg(unix)]
#[tokio::test]
async fn probe_version_returns_none_for_empty_output_and_spawn_failures() {
    let temp = tempfile::tempdir().unwrap();
    let quiet = temp.path().join("quiet-agent");
    std::fs::write(&quiet, "#!/bin/sh\nexit 0\n").unwrap();
    make_executable(&quiet);

    assert_eq!(probe_version(&quiet).await, None);
    assert_eq!(
        probe_version(&temp.path().join("missing-agent")).await,
        None
    );
}

#[test]
fn configuration_and_path_helpers_cover_direct_paths_and_fallbacks() {
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let workspace = temp.path().join("workspace");
    std::fs::create_dir_all(workspace.join(".nemo-relay")).unwrap();
    std::fs::write(
        workspace.join(".nemo-relay").join("config.toml"),
        "[agents.codex]\ncommand = \"codex\"\n",
    )
    .unwrap();
    let _env = EnvScope::set(&[
        ("HOME", Some(home.as_os_str())),
        ("XDG_CONFIG_HOME", None),
        ("PATH", None),
    ]);

    let info = collect_configuration(
        Some(&workspace),
        Some(&home),
        Check {
            name: "Resolution",
            status: Status::Pass,
            details: "valid".into(),
        },
        vec!["codex".into()],
        &[],
        &PluginConfigurationDiagnostics {
            sources: vec![],
            error: None,
            resolution: Check {
                name: "Plugin resolution",
                status: Status::Pass,
                details: "valid".into(),
            },
        },
    );
    assert_eq!(info.workspace.status, Status::Pass);
    assert!(info.global.path.starts_with(&home));
    assert_eq!(info.configured_agents, vec!["codex".to_string()]);

    assert_eq!(which_on_path("definitely-missing"), None);
    assert_eq!(which_command("/definitely/missing"), None);
    let binary = temp.path().join("agent-bin");
    std::fs::write(&binary, "").unwrap();
    assert_eq!(
        which_command(binary.to_str().unwrap()).as_deref(),
        Some(binary.as_path())
    );
}

#[test]
fn format_human_reports_discovered_dynamic_plugins_in_configuration() {
    let mut report = empty_report();
    report.configuration.dynamic_plugins = vec![
        DynamicPluginReferenceInfo {
            plugin_id: "acme.worker".into(),
            manifest_ref: "/tmp/plugins/acme/relay-plugin.toml".into(),
            source: PathBuf::from("/tmp/plugins.toml"),
            host_config_status: DynamicPluginHostConfigStatus::Present,
        },
        DynamicPluginReferenceInfo {
            plugin_id: "acme.native".into(),
            manifest_ref: "/tmp/plugins/native/relay-plugin.toml".into(),
            source: PathBuf::from("/tmp/plugins.toml"),
            host_config_status: DynamicPluginHostConfigStatus::Absent,
        },
    ];

    let rendered = format_human(&report);

    assert!(rendered.contains("Dynamic"));
    assert!(rendered.contains("acme.worker (/tmp/plugins/acme/relay-plugin.toml); host config"));
    assert!(rendered.contains("acme.native (/tmp/plugins/native/relay-plugin.toml)"));
    assert!(rendered.contains("acme.worker resolved from /tmp/plugins/acme/relay-plugin.toml"));
    assert!(
        rendered
            .contains("acme.native discovered via host config only; not enabled by config alone")
    );
    assert!(rendered.contains("not enabled by config alone"));
}

#[test]
fn observability_component_helpers_cover_disabled_and_default_paths() {
    let plugin = serde_json::json!({
        "version": 1,
        "components": [{
            "kind": OBSERVABILITY_PLUGIN_KIND,
            "enabled": true,
            "config": {
                "version": 1,
                "atof": { "enabled": true },
                "openinference": {
                    "enabled": true,
                    "endpoint": "http://127.0.0.1:1"
                }
            }
        }]
    });
    let config = observability_component_config(&plugin).unwrap();
    assert!(section_enabled(config, "atof"));
    assert_eq!(section_output_directory(config, "atof"), None);
    assert_eq!(
        section_endpoint(config, "openinference").as_deref(),
        Some("http://127.0.0.1:1")
    );
    assert!(
        observability_component_config(&serde_json::json!({
            "components": [{ "kind": "other", "config": {} }]
        }))
        .is_none()
    );
    assert!(observability_file_exporter_check(config, "missing").is_none());
    let default_dir = observability_file_exporter_check(config, "atof").unwrap();
    assert_eq!(default_dir.status, Status::Info);
    assert!(default_dir.details.contains("runtime default"));
}

#[test]
fn check_directory_reports_pass_warn_and_fail() {
    let temp = tempfile::tempdir().unwrap();
    let pass = check_directory("ATOF dir", temp.path());
    assert_eq!(pass.status, Status::Pass);

    let missing = check_directory("ATOF dir", &temp.path().join("missing"));
    assert_eq!(missing.status, Status::Warn);

    let file = temp.path().join("file");
    std::fs::write(&file, "").unwrap();
    let fail = check_directory("ATOF dir", &file);
    assert_eq!(fail.status, Status::Fail);
}

#[test]
fn hook_file_status_covers_resolution_and_missing_paths() {
    let resolution_error = hook_file_status(
        Err(CliError::Config("bad path".into())),
        CodingAgent::Hermes,
        true,
        "hooks",
    );
    assert_eq!(resolution_error.0, Status::Fail);
    assert!(resolution_error.1.contains("could not resolve path"));

    let missing = tempfile::tempdir().unwrap().path().join("missing.yaml");
    let (status, details) =
        hook_file_status(Ok(missing.clone()), CodingAgent::Hermes, true, "hooks");
    assert_eq!(status, Status::Fail);
    assert!(details.contains("missing"));

    let (status, details) = hook_file_status(Ok(missing), CodingAgent::Hermes, false, "hooks");
    assert_eq!(status, Status::Info);
    assert!(details.contains("missing"));
}

#[test]
fn hook_file_status_covers_plain_files_and_read_errors() {
    let temp = tempfile::tempdir().unwrap();
    let hooks_path = temp.path().join("config.yaml");
    std::fs::write(&hooks_path, "hooks:\n  PreToolUse: []\n").unwrap();

    let (status, details) =
        hook_file_status(Ok(hooks_path.clone()), CodingAgent::Hermes, true, "hooks");
    assert_eq!(status, Status::Fail);
    assert!(details.contains("missing NeMo Relay hook"));

    let (status, details) =
        hook_file_status(Ok(hooks_path.clone()), CodingAgent::Hermes, false, "hooks");
    assert_eq!(status, Status::Info);
    assert!(details.contains("no NeMo Relay hook"));

    let unreadable_path = temp.path().join("hooks-dir");
    std::fs::create_dir(&unreadable_path).unwrap();
    let (status, details) =
        hook_file_status(Ok(unreadable_path), CodingAgent::Hermes, false, "hooks");
    assert_eq!(status, Status::Fail);
    assert!(details.contains("could not read"));
}

#[tokio::test]
async fn collect_observability_warns_for_missing_atif_dir_without_creating_it() {
    let temp = tempfile::tempdir().unwrap();
    let missing = temp.path().join("missing-atif");
    let gateway = GatewayConfig {
        plugin_config: Some(serde_json::json!({
            "version": 1,
            "components": [{
                "kind": "observability",
                "enabled": true,
                "config": {
                    "version": 1,
                    "atif": {
                        "enabled": true,
                        "output_directory": missing
                    }
                }
            }]
        })),
        ..GatewayConfig::default()
    };

    let checks = collect_observability(&gateway).await;

    let atif_check = checks
        .iter()
        .find(|check| check.name == "ATIF dir")
        .expect("ATIF directory check");
    assert_eq!(atif_check.status, Status::Warn);
    assert!(!missing.exists());
}

#[tokio::test]
async fn collect_observability_registers_adaptive_before_validation() {
    let gateway = GatewayConfig {
        plugin_config: Some(serde_json::json!({
            "version": 1,
            "components": [
                {
                    "kind": "observability",
                    "enabled": true,
                    "config": { "version": 1 }
                },
                {
                    "kind": "adaptive",
                    "enabled": false,
                    "config": {
                        "policy": {
                            "unknown_component": "warn",
                            "unknown_field": "warn",
                            "unsupported_value": "error"
                        }
                    }
                }
            ]
        })),
        ..GatewayConfig::default()
    };

    let checks = collect_observability(&gateway).await;

    assert!(
        !checks.iter().any(|check| check
            .details
            .contains("plugin component kind 'adaptive' is unsupported")),
        "doctor should register adaptive before plugin validation: {checks:?}"
    );
}

#[tokio::test]
async fn collect_observability_registers_pii_redaction_before_validation() {
    let gateway = GatewayConfig {
        plugin_config: Some(serde_json::json!({
            "version": 1,
            "components": [
                {
                    "kind": "observability",
                    "enabled": true,
                    "config": { "version": 1 }
                },
                {
                    "kind": "pii_redaction",
                    "enabled": false,
                    "config": {
                        "version": 1,
                        "mode": "builtin",
                        "policy": {
                            "unknown_component": "warn",
                            "unknown_field": "warn",
                            "unsupported_value": "error"
                        },
                        "builtin": {
                            "action": "remove"
                        }
                    }
                }
            ]
        })),
        ..GatewayConfig::default()
    };

    let checks = collect_observability(&gateway).await;

    assert!(
        !checks.iter().any(|check| check
            .details
            .contains("plugin component kind 'pii_redaction' is unsupported")),
        "doctor should register pii_redaction before plugin validation: {checks:?}"
    );
}

#[tokio::test]
async fn collect_observability_reports_invalid_pii_redaction_config() {
    let gateway = GatewayConfig {
        plugin_config: Some(serde_json::json!({
            "version": 1,
            "components": [
                {
                    "kind": "pii_redaction",
                    "enabled": true,
                    "config": {
                        "version": 2,
                        "mode": "builtin",
                        "builtin": {
                            "action": "remove"
                        }
                    }
                }
            ]
        })),
        ..GatewayConfig::default()
    };

    let checks = collect_observability(&gateway).await;

    let diagnostic = checks
        .iter()
        .find(|check| check.name == "Plugin diagnostic")
        .expect("plugin diagnostic check");
    assert_eq!(diagnostic.status, Status::Fail);
    assert!(diagnostic.details.contains("unsupported_config_version"));
}

#[tokio::test]
async fn collect_observability_probes_atof_streaming_endpoint() {
    let (url, body, server_thread) = start_doctor_http_capture_server();
    let gateway = GatewayConfig {
        plugin_config: Some(serde_json::json!({
            "version": 1,
            "components": [{
                "kind": "observability",
                "enabled": true,
                "config": {
                    "version": 1,
                    "atof": {
                        "enabled": true,
                        "endpoints": [{
                            "url": url,
                            "transport": "http_post",
                            "headers": {"X-Test": "doctor"}
                        }]
                    }
                }
            }]
        })),
        ..GatewayConfig::default()
    };

    let checks = collect_observability(&gateway).await;
    let body = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            let captured = body.lock().unwrap().clone();
            if captured.contains("\"kind\":\"mark\"")
                && captured.contains("\"name\":\"nemo_relay.doctor.atof_probe\"")
            {
                break captured;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("doctor probe body should be captured");

    let endpoint = checks
        .iter()
        .find(|check| check.name == "ATOF endpoint")
        .expect("ATOF endpoint check");
    assert_eq!(endpoint.status, Status::Pass);
    assert!(body.contains("\"kind\":\"mark\""));
    assert!(body.contains("\"name\":\"nemo_relay.doctor.atof_probe\""));
    server_thread.join().unwrap();
}

#[tokio::test]
async fn collect_observability_covers_absent_invalid_and_componentless_configs() {
    let absent = collect_observability(&GatewayConfig::default()).await;
    assert_eq!(absent[0].status, Status::Info);
    assert!(absent[0].details.contains("not configured"));

    let invalid = collect_observability(&GatewayConfig {
        plugin_config: Some(serde_json::json!({"version": "bad"})),
        ..GatewayConfig::default()
    })
    .await;
    assert_eq!(invalid[0].status, Status::Fail);
    assert!(invalid[0].details.contains("invalid plugin config"));

    let no_observability = collect_observability(&GatewayConfig {
        plugin_config: Some(serde_json::json!({
            "version": 1,
            "components": []
        })),
        ..GatewayConfig::default()
    })
    .await;
    assert!(
        no_observability
            .iter()
            .any(|check| check.name == "Observability plugin"
                && check.details.contains("component not configured"))
    );
    assert!(
        no_observability
            .iter()
            .any(|check| check.name == "Model pricing" && check.details.contains("not configured"))
    );
}

#[tokio::test]
async fn collect_observability_rejects_websocket_endpoint_http_scheme() {
    let gateway = GatewayConfig {
        plugin_config: Some(serde_json::json!({
            "version": 1,
            "components": [{
                "kind": "observability",
                "enabled": true,
                "config": {
                    "version": 1,
                    "atof": {
                        "enabled": true,
                        "endpoints": [{
                            "url": "http://localhost:9/events",
                            "transport": "websocket"
                        }]
                    }
                }
            }]
        })),
        ..GatewayConfig::default()
    };

    let checks = collect_observability(&gateway).await;

    let endpoint = checks
        .iter()
        .find(|check| check.name == "ATOF endpoint")
        .expect("ATOF endpoint check");
    assert_eq!(endpoint.status, Status::Fail);
    assert!(endpoint.details.contains("invalid scheme"));
    assert!(endpoint.details.contains("must be ws or wss"));
}

#[tokio::test]
async fn atof_endpoint_validation_rejects_missing_url_headers_timeout_and_transport() {
    let missing_url = probe_atof_endpoint(0, &serde_json::json!({})).await;
    assert_eq!(missing_url.status, Status::Fail);
    assert!(missing_url.details.contains("missing url"));

    let timeout_zero = probe_atof_endpoint(
        1,
        &serde_json::json!({
            "url": "http://127.0.0.1:1/events",
            "timeout_millis": 0
        }),
    )
    .await;
    assert_eq!(timeout_zero.status, Status::Fail);
    assert!(timeout_zero.details.contains("timeout_millis"));

    let bad_headers = probe_atof_endpoint(
        2,
        &serde_json::json!({
            "url": "http://127.0.0.1:1/events",
            "headers": []
        }),
    )
    .await;
    assert_eq!(bad_headers.status, Status::Fail);
    assert!(bad_headers.details.contains("headers must be an object"));

    let non_string_header = probe_atof_endpoint(
        3,
        &serde_json::json!({
            "url": "http://127.0.0.1:1/events",
            "headers": {"x-test": 1}
        }),
    )
    .await;
    assert_eq!(non_string_header.status, Status::Fail);
    assert!(
        non_string_header
            .details
            .contains("headers.x-test must be a string")
    );

    let unsupported = probe_atof_endpoint(
        4,
        &serde_json::json!({
            "url": "http://127.0.0.1:1/events",
            "transport": "grpc"
        }),
    )
    .await;
    assert_eq!(unsupported.status, Status::Fail);
    assert!(unsupported.details.contains("unsupported transport"));
}

#[tokio::test]
async fn atof_http_and_websocket_probes_report_failure_branches() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = [0_u8; 1024];
        let _ = stream.read(&mut buf).unwrap();
        stream
            .write_all(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n")
            .unwrap();
    });

    let failed_http = probe_atof_http_post(
        &url,
        Vec::new(),
        doctor_atof_probe_payload().unwrap(),
        std::time::Duration::from_secs(2),
        5,
    )
    .await;
    assert_eq!(failed_http.status, Status::Fail);
    assert!(failed_http.details.contains("HTTP 500"));
    handle.join().unwrap();

    let bad_header_name = probe_atof_websocket(
        "ws://127.0.0.1:1/events",
        vec![("bad header".to_string(), "value".to_string())],
        "{}".to_string(),
        std::time::Duration::from_millis(10),
        6,
    )
    .await;
    assert_eq!(bad_header_name.status, Status::Fail);

    let bad_header_value = probe_atof_websocket(
        "ws://127.0.0.1:1/events",
        vec![("x-test".to_string(), "bad\r\nvalue".to_string())],
        "{}".to_string(),
        std::time::Duration::from_millis(10),
        7,
    )
    .await;
    assert_eq!(bad_header_value.status, Status::Fail);

    let bad_websocket_url = probe_atof_websocket(
        "ws://[",
        Vec::new(),
        "{}".to_string(),
        std::time::Duration::from_millis(10),
        8,
    )
    .await;
    assert_eq!(bad_websocket_url.status, Status::Fail);
}

#[tokio::test]
async fn atof_http_and_websocket_timeout_errors_are_reported() {
    let http_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let http_url = format!("http://{}", http_listener.local_addr().unwrap());
    let http_handle = std::thread::spawn(move || {
        let (_stream, _) = http_listener.accept().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(75));
    });

    let http_timeout = probe_atof_http_post(
        &http_url,
        Vec::new(),
        doctor_atof_probe_payload().unwrap(),
        std::time::Duration::from_millis(10),
        9,
    )
    .await;
    assert_eq!(http_timeout.status, Status::Fail);
    assert!(http_timeout.details.contains("http_post"));
    assert!(
        http_timeout
            .details
            .to_ascii_lowercase()
            .contains("timeout")
            || http_timeout
                .details
                .to_ascii_lowercase()
                .contains("timed out")
            || http_timeout.details.contains("error sending request"),
        "timeout detail was: {}",
        http_timeout.details
    );
    http_handle.join().unwrap();

    let ws_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let ws_url = format!("ws://{}", ws_listener.local_addr().unwrap());
    let ws_handle = std::thread::spawn(move || {
        let (_stream, _) = ws_listener.accept().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(75));
    });
    let websocket_timeout = probe_atof_websocket(
        &ws_url,
        Vec::new(),
        "{}".to_string(),
        std::time::Duration::from_millis(10),
        10,
    )
    .await;
    assert_eq!(websocket_timeout.status, Status::Fail);
    assert!(websocket_timeout.details.contains("timed out"));
    ws_handle.join().unwrap();
}

#[tokio::test]
async fn probe_http_named_warns_on_http_errors() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = [0_u8; 1024];
        let _ = stream.read(&mut buf).unwrap();
        stream
            .write_all(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n")
            .unwrap();
    });

    let check = probe_http_named("OpenTelemetry endpoint", &url).await;
    assert_eq!(check.status, Status::Warn);
    assert!(check.details.contains("HTTP 500"));
    handle.join().unwrap();
}

#[tokio::test]
async fn http_probe_passes_success_and_ndjson_upload_success() {
    let success_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let success_url = format!("http://{}", success_listener.local_addr().unwrap());
    let success_handle = std::thread::spawn(move || {
        let (mut stream, _) = success_listener.accept().unwrap();
        let mut buf = [0_u8; 1024];
        let _ = stream.read(&mut buf).unwrap();
        stream
            .write_all(b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n")
            .unwrap();
    });
    let check = probe_http_named("OpenTelemetry endpoint", &success_url).await;
    assert_eq!(check.status, Status::Pass);
    success_handle.join().unwrap();

    let (url, body, server_thread) = start_doctor_http_capture_server();
    let check = probe_atof_ndjson(
        &url,
        Vec::new(),
        doctor_atof_probe_payload().unwrap(),
        std::time::Duration::from_secs(2),
        11,
    )
    .await;
    assert_eq!(check.status, Status::Pass);
    assert!(
        body.lock()
            .unwrap()
            .contains("nemo_relay.doctor.atof_probe")
    );
    server_thread.join().unwrap();
}

#[tokio::test]
async fn collect_observability_validates_pricing_file_source() {
    let temp = tempfile::tempdir().unwrap();
    let catalog = temp.path().join("pricing.json");
    std::fs::write(
        &catalog,
        serde_json::json!({
            "version": 1,
            "entries": [{
                "provider": "openai",
                "model_id": "gpt-test",
                "currency": "USD",
                "unit": "per_token",
                "rates": {
                    "input_per_million": 1.0,
                    "output_per_million": 2.0
                },
                "prompt_cache": {
                    "read_accounting": "separate"
                },
                "pricing_as_of": "2026-06-06",
                "pricing_source": "test"
            }]
        })
        .to_string(),
    )
    .unwrap();
    let gateway = GatewayConfig {
        plugin_config: Some(serde_json::json!({
            "version": 1,
            "components": [{
                "kind": "pricing",
                "config": {
                    "sources": [{
                        "type": "file",
                        "path": catalog
                    }]
                }
            }]
        })),
        ..GatewayConfig::default()
    };

    let checks = collect_observability(&gateway).await;

    let pricing = checks
        .iter()
        .find(|check| check.name == "Model pricing source")
        .expect("model pricing source check");
    assert_eq!(pricing.status, Status::Pass);
    assert!(pricing.details.contains("valid (1 entries)"));
}

#[tokio::test]
async fn collect_observability_fails_for_missing_pricing_file_source() {
    let missing = tempfile::tempdir()
        .unwrap()
        .path()
        .join("missing-pricing.json");
    let gateway = GatewayConfig {
        plugin_config: Some(serde_json::json!({
            "version": 1,
            "components": [{
                "kind": "pricing",
                "config": {
                    "sources": [{
                        "type": "file",
                        "path": missing
                    }]
                }
            }]
        })),
        ..GatewayConfig::default()
    };

    let checks = collect_observability(&gateway).await;

    let pricing = checks
        .iter()
        .find(|check| check.name == "Model pricing source")
        .expect("model pricing source check");
    assert_eq!(pricing.status, Status::Fail);
    assert!(pricing.details.contains("unreadable"));
}

#[tokio::test]
async fn collect_observability_reports_pricing_disabled_empty_inline_and_invalid_catalogs() {
    let disabled_config: PluginConfig = serde_json::from_value(serde_json::json!({
        "version": 1,
        "components": [{
            "kind": "pricing",
            "enabled": false,
            "config": {}
        }]
    }))
    .unwrap();
    let mut checks = Vec::new();
    collect_pricing_component_checks(&mut checks, &disabled_config);
    assert_eq!(checks[0].status, Status::Info);
    assert!(checks[0].details.contains("disabled"));

    let empty_config: PluginConfig = serde_json::from_value(serde_json::json!({
        "version": 1,
        "components": [{
            "kind": "pricing",
            "config": {"sources": []}
        }]
    }))
    .unwrap();
    let mut checks = Vec::new();
    collect_pricing_component_checks(&mut checks, &empty_config);
    assert_eq!(checks[0].status, Status::Info);
    assert!(checks[0].details.contains("no sources"));

    let inline_config: PluginConfig = serde_json::from_value(serde_json::json!({
        "version": 1,
        "components": [{
            "kind": "pricing",
            "config": {
                "sources": [{
                    "type": "inline",
                    "catalog": {
                        "version": 1,
                        "entries": [{
                            "provider": "test",
                            "model_id": "model-a",
                            "rates": {"input_per_million": 1.0, "output_per_million": 2.0},
                            "prompt_cache": {"read_accounting": "separate"},
                            "pricing_as_of": "2026-06-06",
                            "pricing_source": "unit"
                        }]
                    }
                }]
            }
        }]
    }))
    .unwrap();
    let mut checks = Vec::new();
    collect_pricing_component_checks(&mut checks, &inline_config);
    assert_eq!(checks[0].status, Status::Pass);
    assert!(checks[0].details.contains("inline:0 valid (1 entries)"));

    let temp = tempfile::tempdir().unwrap();
    let invalid = temp.path().join("pricing.json");
    std::fs::write(&invalid, r#"{"version":1,"entries":[{"bad":true}]}"#).unwrap();
    let invalid_check = pricing_source_check(9, &PricingSourceConfig::File { path: invalid });
    assert_eq!(invalid_check.status, Status::Fail);
    assert!(invalid_check.details.contains("invalid catalog"));
}

#[cfg(unix)]
fn make_executable(path: &std::path::Path) {
    use std::os::unix::fs::PermissionsExt;
    let mut permissions = std::fs::metadata(path).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(path, permissions).unwrap();
}

#[test]
fn format_agents_human_lists_supported_and_separates_detected() {
    let agents = vec![
        AgentInfo {
            name: "claude",
            status: Status::Pass,
            configured: true,
            command: "claude".into(),
            path: Some(PathBuf::from("/opt/homebrew/bin/claude")),
            version: Some("2.1.4".into()),
            annotation: "hooks: injected during run".into(),
        },
        AgentInfo {
            name: "codex",
            status: Status::Info,
            configured: false,
            command: "codex".into(),
            path: None,
            version: None,
            annotation: "not configured".into(),
        },
    ];
    let rendered = format_agents_human(&agents);
    assert!(rendered.contains("Supported"));
    assert!(rendered.contains("Detected on this machine"));
    // Supported lists everything; detected only the one with a path.
    assert!(rendered.contains("claude\n"));
    assert!(rendered.contains("codex\n"));
    assert!(rendered.contains("/opt/homebrew/bin/claude"));
    // codex must NOT show up under the detected block because path is None.
    let detected_block = rendered.split("Detected on this machine").nth(1).unwrap();
    assert!(!detected_block.contains("codex"));
}

#[test]
fn format_agents_json_matches_doctor_agents_shape() {
    let agents = vec![AgentInfo {
        name: "claude",
        status: Status::Pass,
        configured: true,
        command: "claude".into(),
        path: Some(PathBuf::from("/opt/homebrew/bin/claude")),
        version: Some("2.1.4".into()),
        annotation: "hooks: injected during run".into(),
    }];
    let json = format_agents_json(&agents).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(parsed.is_array());
    assert_eq!(parsed[0]["name"], "claude");
    assert_eq!(parsed[0]["status"], "pass");
    assert_eq!(parsed[0]["configured"], true);
    assert_eq!(parsed[0]["command"], "claude");
    assert_eq!(parsed[0]["version"], "2.1.4");
    assert_eq!(parsed[0]["path"], "/opt/homebrew/bin/claude");
}
