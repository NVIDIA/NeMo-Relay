// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn document(contents: &str) -> ConfigDocument {
    ConfigDocument {
        path: PathBuf::from("config.toml"),
        document: contents.parse().unwrap(),
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
