// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
#[test]
fn preparation_preserves_existing_merge_order_and_reuses_parsed_inputs() {
    let directory = tempfile::tempdir().unwrap();
    let user = directory.path().join("user.toml");
    let system = directory.path().join("system.toml");
    std::fs::write(
        &user,
        "[diagnostics]\nenabled = true\n[diagnostics.limits]\nmax_events = 7\n",
    )
    .unwrap();
    std::fs::write(&system, "[diagnostics.limits]\nmax_events = 9\n").unwrap();
    let document = read_shared_documents(vec![(user.clone(), true), (system, true)]).unwrap();
    let diagnostics: RuntimeDiagnosticsConfig = document["diagnostics"].clone().try_into().unwrap();
    assert!(diagnostics.enabled);
    assert_eq!(diagnostics.limits.max_events, 9);
    diagnostics.validate().unwrap();
    let prepared = PreparedConfig {
        document: std::sync::Mutex::new(Some(document)),
        diagnostics,
    };
    std::fs::write(user, "this is now malformed").unwrap();
    let resolved =
        finish_shared_config(prepared.take_document().unwrap(), vec![], |_| Ok(())).unwrap();
    assert_eq!(resolved.diagnostics.limits.max_events, 9);
}
#[test]
fn diagnostics_defaults_and_invalid_limits_are_explicit() {
    let config: RuntimeDiagnosticsConfig = toml::from_str("").unwrap();
    assert!(!config.enabled);
    assert!(!config.http_read);
    assert!(toml::from_str::<RuntimeDiagnosticsConfig>("unknown = true").is_err());
    let config: RuntimeDiagnosticsConfig = toml::from_str("[limits]\nmax_owned_bytes = 1").unwrap();
    assert!(config.validate().is_err());
}
#[test]
fn diagnostic_credentials_do_not_authorize_provider_requests() {
    let key = BootstrapChallengeKey::from_bytes(b"test-key");
    assert!(key.verify_diagnostics_token(&key.diagnostics_token()));
    assert!(!key.verify_client_token(&key.diagnostics_token()));
    assert!(!key.verify_diagnostics_token(&key.client_token()));
    assert!(!key.verify_diagnostics_token(&key.hook_client_token("test")));
    assert!(
        !BootstrapChallengeKey::from_bytes(b"other-key")
            .verify_diagnostics_token(&key.diagnostics_token())
    );
}

#[test]
fn diagnostic_settings_and_environment_do_not_change_execution_fingerprint() {
    use crate::test_support::EnvScope;
    use std::ffi::OsStr;
    let temp = tempfile::tempdir().unwrap();
    let _env = EnvScope::set(&[
        ("XDG_CONFIG_HOME", Some(temp.path().as_os_str())),
        ("NEMO_RELAY_DIAGNOSTICS", None),
        ("NEMO_RELAY_DIAGNOSTICS_HTTP", None),
    ]);
    let mut resolved = ResolvedConfig::default();
    let before = persistent_bootstrap_fingerprint(&resolved, &[]).unwrap();
    resolved.diagnostics.enabled = true;
    resolved.diagnostics.http_read = true;
    resolved.diagnostics.limits.max_events = 1;
    assert_eq!(
        before,
        persistent_bootstrap_fingerprint(&resolved, &[]).unwrap()
    );
    // This test owns the environment lock through EnvScope.
    unsafe {
        std::env::set_var("NEMO_RELAY_DIAGNOSTICS", OsStr::new("true"));
        std::env::set_var("NEMO_RELAY_DIAGNOSTICS_HTTP", OsStr::new("true"));
    }
    assert!(
        crate::mcp_environment::forwarded_names(vec!["NEMO_RELAY_DIAGNOSTICS".into()], None)
            .iter()
            .any(|name| name == "NEMO_RELAY_DIAGNOSTICS")
    );
    assert_eq!(
        before,
        persistent_bootstrap_fingerprint(&resolved, &[]).unwrap()
    );
    resolved.gateway.openai_base_url = "http://127.0.0.1:4321".into();
    assert_ne!(
        before,
        persistent_bootstrap_fingerprint(&resolved, &[]).unwrap()
    );
}

#[test]
fn staged_logging_uses_the_existing_decoder_and_validation() {
    let temp = tempfile::tempdir().unwrap();
    let file = temp.path().join("config.toml");
    for contents in [
        "[logging]\nstderr_format = 'human'",
        "[logging]\nunknown = true",
        "[logging]\nflush_interval_millis = -1",
    ] {
        std::fs::write(&file, contents).unwrap();
        let prepared = PreparedConfig::load(Some(&file)).unwrap();
        let old = resolve_logging_config(Some(&file));
        let staged = prepared.logging();
        match (old, staged) {
            (Ok(expected), Ok(actual)) => assert_eq!(expected, actual),
            (Err(expected), Err(actual)) => assert_eq!(expected.to_string(), actual.to_string()),
            _ => panic!("staging changed logging configuration semantics"),
        }
    }
}
