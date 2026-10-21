// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn document(contents: &str) -> ConfigDocument {
    ConfigDocument {
        path: PathBuf::from("config.toml"),
        document: contents.parse().unwrap(),
        logging_section: "logging",
    }
}

#[test]
fn document_preserves_toml_and_redacts_standard_inline_and_dotted_auth_headers() {
    let mut standard = document(
        "# keep this comment\n[agents.codex]\ncommand = \"codex\"\n\n[upstream]\nopenai_auth_header = \"Bearer secret\"\nanthropic_auth_header = \"Basic secret\"\n",
    );
    standard
        .set_positive_integer("gateway", "max_hook_payload_bytes", 42)
        .unwrap();

    let preview = standard.preview();
    assert!(preview.contains("# keep this comment"));
    assert!(preview.contains("[agents.codex]"));
    assert!(preview.contains("<redacted>"));
    assert!(!preview.contains("Bearer secret"));
    assert!(!preview.contains("Basic secret"));
    assert!(standard.document.to_string().contains("Bearer secret"));

    let mut inline = document(
        "upstream = { openai_auth_header = \"Bearer inline\", anthropic_auth_header = \"Basic inline\" }\n",
    );
    assert_eq!(inline.secret_summary("openai_auth_header"), "configured");
    inline
        .set_auth_header("openai_auth_header", "Bearer replacement".into())
        .unwrap();
    inline
        .clear_key("upstream", "anthropic_auth_header")
        .unwrap();
    let preview = inline.preview();
    assert!(preview.contains("<redacted>"));
    assert!(!preview.contains("Bearer inline"));
    assert!(!preview.contains("Bearer replacement"));
    assert!(!preview.contains("Basic inline"));

    let dotted = document("upstream.openai_auth_header = \"Bearer dotted\"\n");
    assert_eq!(dotted.secret_summary("openai_auth_header"), "configured");
    assert!(!dotted.preview().contains("Bearer dotted"));
}

#[test]
fn edits_and_clears_supported_scalars() {
    let mut document = document(
        "[gateway]\nmax_hook_payload_bytes = \"not-a-number\"\n\n[upstream]\nopenai_base_url = \"https://example.test/v1\"\n\n[logging]\nlevel = \"info\"\nstderr_format = \"human\"\n",
    );
    assert_eq!(
        document.integer_summary("gateway", "max_hook_payload_bytes"),
        "invalid"
    );
    document
        .set_positive_integer("gateway", "max_hook_payload_bytes", 2048)
        .unwrap();
    document
        .set_enum("logging", "level", "debug", LOG_LEVELS)
        .unwrap();
    document
        .set_enum("logging", "stderr_format", "jsonl", LOG_FORMATS)
        .unwrap();
    document
        .set_integer("logging", "flush_interval_millis", 0)
        .unwrap();
    document.clear_key("upstream", "openai_base_url").unwrap();

    let rendered = document.document.to_string();
    assert!(rendered.contains("max_hook_payload_bytes = 2048"));
    assert!(rendered.contains("level = \"debug\""));
    assert!(rendered.contains("stderr_format = \"jsonl\""));
    assert!(rendered.contains("flush_interval_millis = 0"));
    assert!(!rendered.contains("example.test"));
}

#[test]
fn validates_gateway_auth_and_sink_values() {
    let mut document = document("");
    assert!(
        document
            .set_positive_integer("gateway", "max_hook_payload_bytes", 0)
            .is_err()
    );
    assert!(
        document
            .set_auth_header("openai_auth_header", "Bearer\nsecret".into())
            .is_err()
    );
    assert!(
        document
            .set_integer("logging", "flush_interval_millis", u64::MAX)
            .is_err()
    );

    document.add_sink("relay.log".into()).unwrap();
    assert!(document.set_sink_queue_capacity(0, 0).is_err());
    assert!(
        document
            .set_sink_queue_capacity(0, MAX_FILE_SINK_QUEUE_ENTRIES as u64 + 1)
            .is_err()
    );
    assert!(document.set_sink_rotation(0, 1024, 10).is_err());
    assert!(document.set_sink_rotation(0, u64::MAX, 1).is_err());
    assert!(document.set_sink_rotation(0, 1024, u64::MAX).is_err());
    assert!(
        document
            .set_sink_enum(0, "level", "invalid", LOG_LEVELS)
            .is_err()
    );
}

#[test]
fn manages_sink_lifecycle_and_summaries() {
    let mut document = document("");
    document.add_sink("relay.log".into()).unwrap();
    document.set_sink_queue_capacity(0, 128).unwrap();
    document.set_sink_rotation(0, 1024 * 1024, 2).unwrap();
    assert_eq!(document.gateway_summary(), "defaults");
    assert_eq!(document.logging_summary(), "configured");
    assert_eq!(document.sink_labels(), ["sink 1 (relay.log)"]);
    assert_eq!(document.sink_integer_summary(0, "queue_capacity"), "128");
    assert_eq!(
        document.sink_rotation_summary(0),
        "1048576 bytes, 2 backups"
    );
    document.clear_sink_rotation(0).unwrap();
    document.remove_sink(0).unwrap();
    assert_eq!(document.sink_count(), 0);
    assert!(!document.document.to_string().contains("[logging]"));
}

#[test]
fn malformed_sections_and_missing_sinks_report_errors() {
    let mut malformed = document("gateway = \"invalid\"\nlogging = \"invalid\"\n");
    assert!(
        malformed
            .set_positive_integer("gateway", "max_hook_payload_bytes", 1)
            .is_err()
    );
    assert!(malformed.add_sink("relay.log".into()).is_err());

    for section in ["logging", "daemon.logging"] {
        let mut malformed = document(&format!("[{section}]\nsinks = \"invalid\"\n"));
        malformed.logging_section = section;
        let error = malformed.add_sink("relay.log".into()).unwrap_err();
        assert_eq!(
            error.to_string(),
            format!(
                "configuration error: {section}.sinks must be an array of tables before it can be edited"
            )
        );
    }

    let mut document = document("");
    assert!(document.remove_sink(0).is_err());
    assert!(document.sink_has_key(0, "path").is_err());
    assert!(document.clear_sink_key(0, "path").is_err());
}

#[test]
fn target_selection_and_file_loading_behave_as_expected() {
    let user = ConfigEditCommand::default();
    assert_eq!(TargetScope::from(&user), TargetScope::User);

    let root = tempfile::tempdir().unwrap();
    let invalid = root.path().join("invalid.toml");
    std::fs::write(&invalid, "[gateway\n").unwrap();
    let error = match ConfigDocument::read(invalid.clone()) {
        Ok(_) => panic!("invalid TOML should be rejected"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("invalid TOML"));
    assert!(error.contains(&invalid.display().to_string()));
}

#[test]
fn config_editor_treats_explicit_config_as_the_user_target() {
    let inherited = PathBuf::from("/managed/config.toml");
    let (scope, path) =
        resolve_edit_target(&ConfigEditCommand::default(), Some(inherited.clone())).unwrap();
    assert_eq!(scope, TargetScope::User);
    assert_eq!(path, inherited);

    let user = ConfigEditCommand {
        user: true,
        ..ConfigEditCommand::default()
    };
    let (scope, path) = resolve_edit_target(&user, Some(inherited.clone())).unwrap();
    assert_eq!(scope, TargetScope::User);
    assert_eq!(path, inherited);

    let (scope, path) = resolve_edit_target(&user, None).unwrap();
    assert_eq!(scope, TargetScope::User);
    assert_eq!(
        path,
        crate::configuration::user_config_dir()
            .unwrap()
            .join("config.toml")
    );

    let global = ConfigEditCommand {
        global: true,
        ..ConfigEditCommand::default()
    };
    let (scope, path) =
        resolve_edit_target(&global, Some(PathBuf::from("/ignored/config.toml"))).unwrap();
    assert_eq!(scope, TargetScope::Global);
    assert_eq!(
        path,
        crate::configuration::system_config_dir().join("config.toml")
    );
}

#[test]
fn documents_are_written_atomically_with_scope_appropriate_permissions() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("user/nested/config.toml");
    assert!(!path.parent().unwrap().exists());
    let document = ConfigDocument::read(path.clone()).unwrap();
    assert!(!path.exists());
    document.write(TargetScope::User).unwrap();
    assert!(path.exists());
    assert!(path.parent().unwrap().is_dir());

    let original = std::fs::read_to_string(&path).unwrap();
    crate::filesystem::fail_next_atomic_write(&path);
    let error = document.write(TargetScope::User).unwrap_err().to_string();
    assert!(error.contains("injected test failure"));
    assert_eq!(std::fs::read_to_string(&path).unwrap(), original);

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }

    let global_path = directory.path().join("system/nested/config.toml");
    assert!(!global_path.parent().unwrap().exists());
    ConfigDocument::read(global_path.clone())
        .unwrap()
        .write(TargetScope::Global)
        .unwrap();
    assert!(global_path.is_file());
    assert!(global_path.parent().unwrap().is_dir());

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(global_path).unwrap().permissions().mode() & 0o777,
            0o644
        );
    }
}

#[test]
fn global_document_rejects_authorization_headers() {
    let directory = tempfile::tempdir().unwrap();
    for (index, contents) in [
        "[upstream]\nopenai_auth_header = \"Bearer secret\"\n",
        "upstream = { anthropic_auth_header = \"Bearer secret\" }\n",
        "upstream.openai_auth_header = \"Bearer secret\"\n",
        "[upstream]\nopenai_api_key = \"secret\"\n",
        "upstream = { anthropic_api_key = \"secret\" }\n",
    ]
    .into_iter()
    .enumerate()
    {
        let path = directory.path().join(format!("config-{index}.toml"));
        std::fs::write(&path, "original").unwrap();
        let mut document = document(contents);
        document.path = path.clone();

        let error = document.write(TargetScope::Global).unwrap_err().to_string();
        assert!(error.contains("global config cannot include upstream authorization headers"));
        assert_eq!(std::fs::read_to_string(path).unwrap(), "original");
    }
}

#[test]
fn credentials_switch_sources_and_round_trip_without_running_helpers() {
    use crate::upstream_auth::{ApiKeyHelper, AuthScheme};

    let root = tempfile::tempdir().unwrap();
    let mut document = document(
        "# keep\n[agents.codex]\ncommand = 'codex'\n[upstream]\nanthropic_auth_header = \"Bearer other\"\n",
    );
    document.path = root.path().join("config.toml");
    for provider in ["openai", "anthropic"] {
        let key = format!("{provider}_api_key");
        let header = format!("{provider}_auth_header");
        let helper = format!("{provider}_api_key_helper");
        document.set_api_key(&key, "static-secret".into()).unwrap();
        assert!(!document.has_key("upstream", &header));
        assert!(!document.preview().contains("static-secret"));
        document.write(TargetScope::User).unwrap();
        document
            .set_api_key_helper(
                provider,
                ApiKeyHelper {
                    command: "never-execute-this-secret-command".into(),
                    args: vec!["argument with spaces".into(), "secret-argument\"\\".into()],
                    timeout_ms: 2345,
                    auth_scheme: Some(AuthScheme::Bearer),
                },
            )
            .unwrap();
        assert!(!document.has_key("upstream", &key));
        assert!(!document.has_key("upstream", &header));
        let preview = document.preview();
        for secret in [
            "static-secret",
            "never-execute-this-secret-command",
            "secret-argument",
        ] {
            assert!(!preview.contains(secret));
        }
        document.write(TargetScope::User).unwrap();
        let saved: toml::Value =
            toml::from_str(&std::fs::read_to_string(&document.path).unwrap()).unwrap();
        let parsed: ApiKeyHelper = saved["upstream"][&helper].clone().try_into().unwrap();
        assert_eq!(parsed.command, "never-execute-this-secret-command");
        assert_eq!(parsed.args, ["argument with spaces", "secret-argument\"\\"]);
        assert_eq!(parsed.timeout_ms, 2345);
        assert!(matches!(parsed.auth_scheme, Some(AuthScheme::Bearer)));

        document
            .set_auth_header(&header, "Bearer replacement".into())
            .unwrap();
        assert!(!document.has_key("upstream", &helper));
        document
            .set_api_key(&key, "replacement-key".into())
            .unwrap();
        assert!(!document.has_key("upstream", &header));
        document.clear_key("upstream", &key).unwrap();
        assert!(!document.has_key("upstream", &key));
    }
    assert!(document.document.to_string().contains("# keep"));
}

#[test]
fn helper_replacement_supports_nested_inline_and_dotted_tables_and_default_schemes() {
    use crate::upstream_auth::{ApiKeyHelper, AuthScheme};

    for contents in [
        "[upstream.openai_api_key_helper]\ncommand = \"old-secret\"\nargs = [\"old-argument\"]\n",
        "upstream = { openai_api_key_helper = { command = \"old-secret\", args = [\"old-argument\"] } }\n",
        "upstream.openai_api_key_helper.command = \"old-secret\"\nupstream.openai_api_key_helper.args = [\"old-argument\"]\n",
    ] {
        let mut document = document(contents);
        assert!(!document.preview().contains("old-secret"));
        assert!(!document.preview().contains("old-argument"));
        for provider in ["openai", "anthropic"] {
            let key = format!("{provider}_api_key_helper");
            document
                .set_api_key_helper(
                    provider,
                    ApiKeyHelper {
                        command: "new-secret".into(),
                        args: Vec::new(),
                        timeout_ms: 10_000,
                        auth_scheme: None,
                    },
                )
                .unwrap();
            assert!(!document.preview().contains("new-secret"));
            let parsed: toml::Value = toml::from_str(&document.document.to_string()).unwrap();
            assert!(parsed["upstream"][&key].get("auth_scheme").is_none());
            crate::configuration::validate_config_document(parsed).unwrap();
            document.clear_key("upstream", &key).unwrap();
            assert!(!document.has_key("upstream", &key));
        }
    }

    let mut document = document("");
    document
        .set_api_key_helper(
            "anthropic",
            ApiKeyHelper {
                command: "helper".into(),
                args: Vec::new(),
                timeout_ms: 1,
                auth_scheme: Some(AuthScheme::ApiKey),
            },
        )
        .unwrap();
    crate::configuration::validate_config_document(
        toml::from_str(&document.document.to_string()).unwrap(),
    )
    .unwrap();
}

#[test]
fn invalid_credential_replacements_preserve_the_previous_source() {
    use crate::upstream_auth::{ApiKeyHelper, AuthScheme};

    let mut document = document("[upstream]\nopenai_auth_header = \"Bearer keep\"\n");
    let original = document.document.to_string();
    for token in ["", "two tokens", "token\nwith-newline"] {
        assert!(
            document
                .set_api_key("openai_api_key", token.into())
                .is_err()
        );
        assert_eq!(document.document.to_string(), original);
    }
    for (command, timeout_ms, auth_scheme) in [
        (" ", 1000, None),
        ("helper", 0, None),
        ("helper", u64::MAX, None),
        ("helper", 1000, Some(AuthScheme::ApiKey)),
    ] {
        assert!(
            document
                .set_api_key_helper(
                    "openai",
                    ApiKeyHelper {
                        command: command.into(),
                        args: Vec::new(),
                        timeout_ms,
                        auth_scheme,
                    }
                )
                .is_err()
        );
        assert_eq!(document.document.to_string(), original);
    }
}

#[test]
fn inline_sink_arrays_support_edit_add_remove_and_preserve_toml_form() {
    let root = tempfile::tempdir().unwrap();
    for (index, (section, contents)) in [
        ("logging", "# keep\n[logging]\nsinks = [{ path = 'original.log', level = 'info' }] # sinks\n"),
        ("logging", "# keep\nlogging = { sinks = [{ path = 'original.log', level = 'info' }] } # sinks\n"),
        ("logging", "# keep\nlogging.sinks = [{ path = 'original.log', level = 'info' }] # sinks\n"),
        ("daemon.logging", "# keep\n[daemon.logging]\nsinks = [{ path = 'original.log', level = 'info' }] # sinks\n"),
        ("daemon.logging", "# keep\ndaemon = { logging = { sinks = [{ path = 'original.log', level = 'info' }] } } # sinks\n"),
        ("daemon.logging", "# keep\ndaemon.logging.sinks = [{ path = 'original.log', level = 'info' }] # sinks\n"),
    ].into_iter().enumerate() {
        let mut document = document(contents);
        document.path = root.path().join(format!("config-{index}.toml"));
        document.logging_section = section;
        assert_eq!(document.sink_count(), 1);
        assert_eq!(document.sink_labels(), ["sink 1 (original.log)"]);
        document.set_sink_string(0, "path", "changed.log".into()).unwrap();
        document.set_sink_enum(0, "format", "jsonl", LOG_FORMATS).unwrap();
        document.set_sink_queue_capacity(0, 128).unwrap();
        document.set_sink_rotation(0, 1024 * 1024, 2).unwrap();
        document.clear_sink_key(0, "level").unwrap();
        document.add_sink("additional.log".into()).unwrap();
        document.write(TargetScope::User).unwrap();
        let reloaded = ConfigDocument::read(document.path.clone()).unwrap();
        let rendered = reloaded.document.to_string();
        assert!(rendered.contains("# keep"));
        assert!(rendered.contains("# sinks"));
        assert!(!rendered.contains("[["));
        let parsed: toml::Value = toml::from_str(&rendered).unwrap();
        let logging = if section == "logging" { &parsed["logging"] } else { &parsed["daemon"]["logging"] };
        let mut runtime_document = toml::map::Map::new();
        runtime_document.insert("logging".into(), logging.clone());
        let runtime = nemo_relay::logging::LoggingConfig::from_toml_document(&toml::to_string(&runtime_document).unwrap()).unwrap();
        assert_eq!(runtime.sinks.len(), 2);
        let nemo_relay::logging::LogSinkConfig::File(sink) = &runtime.sinks[0];
        assert_eq!(sink.path, PathBuf::from("changed.log"));
        assert_eq!(sink.queue_capacity, 128);
        assert!(sink.rotation.is_some());
        document.clear_sink_rotation(0).unwrap();
        document.remove_sink(0).unwrap();
        assert_eq!(document.sink_labels(), ["sink 1 (additional.log)"]);
        document.remove_sink(0).unwrap();
        assert_eq!(document.sink_count(), 0);
        assert!(document.document.get(section.split('.').next().unwrap()).is_none());
    }
}

#[test]
fn invalid_configuration_cannot_overwrite_or_create_a_file() {
    let root = tempfile::tempdir().unwrap();
    for contents in [
        "[upstream]\nopenai_base_url = 'not-a-url'\n",
        "[upstream]\nanthropic_base_url = 'http://remote.example.test'\n",
        "[upstream]\nchatgpt_codex_base_url = 'https://example.test/?query=1'\n",
        "[upstream]\nopenai_api_key = 'two tokens'\n",
        "[upstream]\nopenai_api_key = 'key'\nopenai_auth_header = 'Bearer header'\n",
        "[upstream]\nanthropic_api_key = 'key'\nanthropic_api_key_helper = { command = 'helper' }\n",
        "[upstream.openai_api_key_helper]\ncommand = ''\n",
        "[upstream.openai_api_key_helper]\ncommand = 'helper'\ntimeout_ms = 0\n",
        "[upstream.openai_api_key_helper]\ncommand = 'helper'\nauth_scheme = 'api_key'\n",
        "[gateway]\nmax_hook_payload_bytes = 0\n",
        "[logging]\nlevel = 'invalid'\n",
        "[logging]\nsinks = [{ path = 'relay.log', queue_capacity = 0 }]\n",
        "[daemon.logging]\nstderr_enabled = 'invalid'\n",
        "[[daemon.logging.sinks]]\npath = 'daemon.log'\nretained_files = 1\n",
    ] {
        let mut document = document(contents);
        document.path = root.path().join("existing.toml");
        std::fs::write(&document.path, "# original\n").unwrap();
        assert!(document.write(TargetScope::User).is_err(), "{contents}");
        assert_eq!(
            std::fs::read_to_string(&document.path).unwrap(),
            "# original\n"
        );
        document.path = root.path().join("missing/config.toml");
        assert!(document.write(TargetScope::User).is_err(), "{contents}");
        assert!(!document.path.parent().unwrap().exists());
    }
}

#[test]
fn save_validation_does_not_display_malformed_credential_values() {
    let root = tempfile::tempdir().unwrap();
    for contents in [
        "[upstream.openai_api_key_helper]\ncommand = 'secret-command'\nargs = 'secret-argument'\n",
        "[upstream.anthropic_api_key_helper]\ncommand = 'secret-command'\nauth_scheme = 'secret-scheme'\n",
        "[upstream]\nopenai_api_key = ['secret-token']\n",
    ] {
        let mut document = document(contents);
        document.path = root.path().join("config.toml");
        let error = document.write(TargetScope::User).unwrap_err().to_string();
        assert!(error.contains("invalid configuration shape"));
        assert!(!error.contains("secret-"));
        assert!(!document.path.exists());
    }
}

#[test]
fn noninteractive_editor_guard_is_deterministic() {
    let error = ensure_tty_with(false).unwrap_err().to_string();
    assert_eq!(
        error,
        "configuration error: interactive configuration editing requires a TTY"
    );
    assert!(ensure_tty_with(true).is_ok());
}

#[test]
fn document_accessors_cover_inline_values_defaults_and_invalid_shapes() {
    let mut document = document(
        "gateway = { max_hook_payload_bytes = 64, max_passthrough_body_bytes = -1 }\nupstream = { openai_base_url = \"https://example.test\", openai_auth_header = 7 }\nlogging = { level = 9 }\n",
    );

    assert_eq!(document.path(), Path::new("config.toml"));
    assert_eq!(document.gateway_summary(), "configured");
    assert_eq!(document.upstream_summary(), "configured");
    assert_eq!(document.logging_summary(), "configured");
    assert_eq!(
        document.integer_summary("gateway", "max_hook_payload_bytes"),
        "64"
    );
    assert_eq!(
        document.integer_summary("gateway", "max_passthrough_body_bytes"),
        "invalid"
    );
    assert_eq!(
        document.string_summary("upstream", "openai_base_url"),
        "https://example.test"
    );
    assert_eq!(document.string_summary("logging", "level"), "invalid");
    assert_eq!(document.string_summary("logging", "missing"), "unset");
    assert_eq!(document.secret_summary("anthropic_auth_header"), "unset");

    document
        .set_string("upstream", "openai_base_url", "https://changed.test".into())
        .unwrap();
    assert_eq!(
        document.string("upstream", "openai_base_url").as_deref(),
        Some("https://changed.test")
    );
    document.clear_key("missing", "value").unwrap();
    assert!(!document.has_key("missing", "value"));
}

#[test]
fn sink_accessors_report_invalid_and_incomplete_entries() {
    let mut document = document(
        "[[logging.sinks]]\npath = 7\nlevel = \"debug\"\nqueue_capacity = -1\nmax_file_size_bytes = 1024\n",
    );

    assert_eq!(document.sink_labels(), ["sink 1 (invalid path)"]);
    assert_eq!(document.sink_string_summary(0, "path"), "invalid");
    assert_eq!(document.sink_string_summary(0, "level"), "debug");
    assert_eq!(document.sink_string_summary(0, "format"), "unset");
    assert_eq!(
        document.sink_integer_summary(0, "queue_capacity"),
        "invalid"
    );
    assert_eq!(document.sink_integer_summary(0, "retained_files"), "unset");
    assert_eq!(document.sink_rotation_summary(0), "incomplete");

    document
        .set_sink_string(0, "path", "relay.log".into())
        .unwrap();
    document.clear_sink_key(0, "level").unwrap();
    assert!(!document.sink_has_key(0, "level").unwrap());
    assert_eq!(
        document.sink_string(0, "path").as_deref(),
        Some("relay.log")
    );
}

#[test]
fn newly_exposed_settings_round_trip_through_runtime_configuration() {
    let root = tempfile::tempdir().unwrap();
    let mut document = document("# preserved\n[agents.codex]\ncommand = \"old-codex\"\n");
    document.path = root.path().join("config.toml");
    document
        .set_integer("upstream", "response_timeout_secs", 0)
        .unwrap();
    document
        .set_string(
            "upstream",
            "chatgpt_codex_base_url",
            "https://chatgpt.example.test/backend-api".into(),
        )
        .unwrap();
    document
        .set_value("logging", "stderr_enabled", Value::from(false))
        .unwrap();
    for agent in ["claude", "codex", "pi"] {
        document
            .set_string(
                &format!("agents.{agent}"),
                "command",
                format!("/custom/{agent}"),
            )
            .unwrap();
    }
    document.write(TargetScope::User).unwrap();

    let _environment = crate::test_support::ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let previous_system =
        crate::configuration::set_test_system_config_dir(Some(root.path().join("system")));
    let resolved = crate::configuration::resolve_server_config(&crate::server::GatewayOverrides {
        config: Some(document.path.clone()),
        ..Default::default()
    })
    .unwrap();
    crate::configuration::set_test_system_config_dir(previous_system);
    assert_eq!(resolved.gateway.response_timeout_secs, 0);
    assert_eq!(
        resolved.gateway.chatgpt_codex_base_url.as_deref(),
        Some("https://chatgpt.example.test/backend-api")
    );
    assert!(!resolved.logging.stderr_enabled);
    assert_eq!(
        resolved.agents.claude.command.as_deref(),
        Some("/custom/claude")
    );
    assert_eq!(
        resolved.agents.codex.command.as_deref(),
        Some("/custom/codex")
    );
    assert_eq!(resolved.agents.pi.command.as_deref(), Some("/custom/pi"));

    document
        .set_integer("upstream", "response_timeout_secs", 90)
        .unwrap();
    document.write(TargetScope::User).unwrap();
    let reloaded = ConfigDocument::read(document.path.clone()).unwrap();
    assert_eq!(
        reloaded.integer("upstream", "response_timeout_secs"),
        Some(90)
    );
    assert_eq!(
        reloaded.boolean_summary("logging", "stderr_enabled"),
        "false"
    );
    assert!(reloaded.document.to_string().contains("# preserved"));

    for (section, key) in [
        ("upstream", "response_timeout_secs"),
        ("upstream", "chatgpt_codex_base_url"),
        ("logging", "stderr_enabled"),
        ("agents.claude", "command"),
        ("agents.codex", "command"),
        ("agents.pi", "command"),
    ] {
        document.clear_key(section, key).unwrap();
        assert!(!document.has_key(section, key));
    }
    assert!(document.document.get("agents").is_none());
    assert!(document.document.get("logging").is_none());
    assert!(document.document.get("upstream").is_none());
}

#[test]
fn nested_agent_commands_and_boolean_support_all_toml_table_forms() {
    for contents in [
        "[agents.codex]\ncommand = \"old\"\n[logging]\nstderr_enabled = false\n",
        "agents.codex.command = \"old\"\nlogging.stderr_enabled = false\n",
        "agents = { codex = { command = \"old\" }, pi = { command = \"keep\" } }\nlogging = { stderr_enabled = false }\n",
    ] {
        let mut document = document(contents);
        assert_eq!(document.string_summary("agents.codex", "command"), "old");
        assert_eq!(document.boolean("logging", "stderr_enabled"), Some(false));
        document
            .set_string("agents.codex", "command", "new".into())
            .unwrap();
        document
            .set_value("logging", "stderr_enabled", Value::from(true))
            .unwrap();
        let parsed: toml::Value = toml::from_str(&document.document.to_string()).unwrap();
        assert_eq!(parsed["agents"]["codex"]["command"].as_str(), Some("new"));
        assert_eq!(parsed["logging"]["stderr_enabled"].as_bool(), Some(true));
        document.clear_key("agents.codex", "command").unwrap();
        assert!(!document.has_key("agents.codex", "command"));
        if contents.contains("keep") {
            assert_eq!(document.string_summary("agents.pi", "command"), "keep");
        }
    }
    let mut invalid = document("agents = { codex = false }\nlogging = { stderr_enabled = 1 }\n");
    assert!(
        invalid
            .set_string("agents.codex", "command", "codex".into())
            .is_err()
    );
    assert_eq!(
        invalid.boolean_summary("logging", "stderr_enabled"),
        "invalid"
    );
    invalid.clear_key("logging", "stderr_enabled").unwrap();
    assert_eq!(
        invalid.boolean_summary("logging", "stderr_enabled"),
        "unset"
    );
}

#[test]
fn daemon_logging_edits_preserve_application_logging() {
    let root = tempfile::tempdir().unwrap();
    let mut document = document(
        "[logging]\nlevel = \"warn\"\nstderr_enabled = true\n[[logging.sinks]]\npath = \"app.log\"\n",
    );
    document.path = root.path().join("config.toml");
    document.logging_section = "daemon.logging";
    document
        .set_enum("daemon.logging", "level", "debug", LOG_LEVELS)
        .unwrap();
    document
        .set_enum("daemon.logging", "stderr_format", "jsonl", LOG_FORMATS)
        .unwrap();
    document
        .set_value("daemon.logging", "stderr_enabled", Value::from(false))
        .unwrap();
    document
        .set_integer("daemon.logging", "flush_interval_millis", 0)
        .unwrap();
    document.add_sink("daemon.log".into()).unwrap();
    document
        .set_sink_enum(0, "level", "trace", LOG_LEVELS)
        .unwrap();
    document
        .set_sink_enum(0, "format", "human", LOG_FORMATS)
        .unwrap();
    document.set_sink_queue_capacity(0, 128).unwrap();
    document.set_sink_rotation(0, 1024 * 1024, 2).unwrap();
    document.write(TargetScope::User).unwrap();

    let _environment = crate::test_support::ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let previous_system =
        crate::configuration::set_test_system_config_dir(Some(root.path().join("system")));
    let daemon = crate::configuration::resolve_daemon_logging_config(Some(&document.path)).unwrap();
    crate::configuration::set_test_system_config_dir(previous_system);
    assert!(!daemon.stderr_enabled);
    assert_eq!(daemon.level, nemo_relay::logging::LogLevel::Debug);
    assert_eq!(daemon.flush_interval_millis, 0);
    let nemo_relay::logging::LogSinkConfig::File(sink) = &daemon.sinks[0];
    assert_eq!(sink.path, PathBuf::from("daemon.log"));
    assert_eq!(sink.queue_capacity, 128);
    assert!(sink.rotation.is_some());

    document.clear_sink_rotation(0).unwrap();
    document.remove_sink(0).unwrap();
    for key in [
        "level",
        "stderr_format",
        "stderr_enabled",
        "flush_interval_millis",
    ] {
        document.clear_key("daemon.logging", key).unwrap();
    }
    assert!(document.document.get("daemon").is_none());
    document.logging_section = "logging";
    assert_eq!(document.sink_labels(), ["sink 1 (app.log)"]);
    assert_eq!(document.string_summary("logging", "level"), "warn");
    assert_eq!(document.boolean("logging", "stderr_enabled"), Some(true));
}
