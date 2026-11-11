// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn document(text: &str) -> toml::Value {
    toml::Value::Table(text.parse().unwrap())
}

#[test]
fn root_preset_expands_both_providers_and_preserves_types() {
    let resolved = resolve(vec![document(
        r#"
preset = "openrouter"
[presets.openrouter.upstream]
openai_base_url = "https://openrouter.ai/api/v1"
anthropic_base_url = "https://openrouter.ai/api/"
[presets.openrouter.upstream.openai_oidc]
issuer = "https://clerk.openrouter.ai"
client_id = "example"
scope = "openid offline_access email inference"
[presets.openrouter.upstream.anthropic_oidc]
issuer = "https://clerk.openrouter.ai"
client_id = "example"
scope = "openid offline_access email inference"
[presets.openrouter.diagnostics]
enabled = true
[presets.openrouter.gateway]
max_hook_payload_bytes = 4096
[presets.openrouter.agents.codex]
args = ["--example", "value"]
"#,
    )])
    .unwrap();
    assert!(resolved.get("preset").is_none());
    assert!(resolved.get("presets").is_none());
    let mut config = super::super::ResolvedConfig::default();
    super::super::apply_file_config(&mut config, resolved.clone()).unwrap();
    assert_eq!(
        config.gateway.openai_base_url,
        "https://openrouter.ai/api/v1"
    );
    assert_eq!(
        config.gateway.anthropic_base_url,
        "https://openrouter.ai/api/"
    );
    assert_eq!(config.gateway.openai_oidc, config.gateway.anthropic_oidc);
    assert_eq!(config.gateway.openai_oidc.unwrap().client_id, "example");
    assert_eq!(resolved["diagnostics"]["enabled"].as_bool(), Some(true));
    assert_eq!(config.gateway.max_hook_payload_bytes, 4096);
    assert_eq!(
        resolved["agents"]["codex"]["args"][1].as_str(),
        Some("value")
    );
}

#[test]
fn root_and_upstream_selectors_use_the_same_root_preset_definition() {
    let definitions = document(
        "[presets.team.upstream]\nopenai_base_url = 'https://openrouter.ai/api/v1'\nanthropic_base_url = 'https://openrouter.ai/api/'\n[presets.team.upstream.openai_oidc]\nissuer = 'https://clerk.openrouter.ai'\nclient_id = 'team'\n[presets.team.upstream.anthropic_oidc]\nissuer = 'https://clerk.openrouter.ai'\nclient_id = 'team'\n[presets.team.gateway]\nmax_hook_payload_bytes = 4096\n",
    );
    let root = resolve(vec![definitions.clone(), document("preset = 'team'")]).unwrap();
    let nested = resolve(vec![definitions, document("[upstream]\npreset = 'team'")]).unwrap();
    assert_eq!(root["upstream"], nested["upstream"]);
    assert!(nested.get("gateway").is_none());
    assert!(super::super::parse_file_config(nested).is_ok());
}

#[test]
fn deeply_nested_selectors_project_the_matching_ancestor_subtree() {
    let resolved = resolve(vec![document(
        "[agents.codex]\npreset = 'team'\n[agents.presets.team.codex]\ncommand = 'team-codex'\n[agents.presets.team.claude]\ncommand = 'team-claude'\n",
    )]).unwrap();
    assert_eq!(
        resolved["agents"]["codex"]["command"].as_str(),
        Some("team-codex")
    );
    assert!(resolved["agents"].get("claude").is_none());
}

#[test]
fn nearest_definition_shadows_outer_definitions_and_requires_a_matching_table() {
    let resolved = resolve(vec![document(
        "[agents.codex]\npreset = 'team'\n[agents.codex.presets.team]\ncommand = 'local-codex'\n[presets.team.agents.codex]\ncommand = 'root-codex'\n",
    )]).unwrap();
    assert_eq!(
        resolved["agents"]["codex"]["command"].as_str(),
        Some("local-codex")
    );
    for definitions in [
        "[presets.team.gateway]\nmax_hook_payload_bytes = 4096",
        "[presets.team]\nupstream = 42",
    ] {
        let error = resolve(vec![
            document(definitions),
            document("[upstream]\npreset = 'team'"),
        ])
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("no matching configuration table at [upstream]"),
            "{error}"
        );
    }
}

#[test]
fn nested_presets_expand_direct_fields_at_arbitrary_depth() {
    let resolved = resolve(vec![document(
        r#"
[upstream]
preset = "provider"
[upstream.presets.provider]
openai_base_url = "https://provider.example/v1"
[upstream.presets.provider.openai_oidc]
issuer = "https://issuer.example"
client_id = "example"
[agents.codex]
preset = "local"
[agents.codex.presets.local]
command = "my-codex"
args = ["--local"]
"#,
    )])
    .unwrap();
    assert_eq!(
        resolved["upstream"]["openai_oidc"]["client_id"].as_str(),
        Some("example")
    );
    assert_eq!(
        resolved["agents"]["codex"]["command"].as_str(),
        Some("my-codex")
    );
    assert!(resolved["agents"]["codex"].get("presets").is_none());
    assert!(resolved["upstream"].get("preset").is_none());
}

#[test]
fn system_presets_are_available_to_root_and_nested_user_selections() {
    let temp = tempfile::tempdir().unwrap();
    let xdg = temp.path().join("xdg");
    let user_dir = xdg.join("nemo-relay");
    let system_dir = temp.path().join("system");
    std::fs::create_dir_all(&user_dir).unwrap();
    std::fs::create_dir_all(&system_dir).unwrap();
    let _environment = crate::test_support::EnvScope::set(&[
        ("XDG_CONFIG_HOME", Some(xdg.as_os_str())),
        ("NEMO_RELAY_TEST_SKIP_IMPLICIT_CONFIG", None),
        ("NEMO_RELAY_OPENAI_BASE_URL", None),
    ]);
    let previous_system = super::super::set_test_system_config_dir(Some(system_dir.clone()));
    std::fs::write(system_dir.join("config.toml"), "[presets.team.upstream]\nresponse_timeout_secs = 50\nopenai_base_url = 'https://team.example/v1'\n[presets.team.upstream.openai_oidc]\nissuer = 'https://issuer.example'\nclient_id = 'team'\n").unwrap();
    for selection in ["preset = 'team'\n[upstream]", "[upstream]\npreset = 'team'"] {
        std::fs::write(
            user_dir.join("config.toml"),
            format!("{selection}\nresponse_timeout_secs = 25\n"),
        )
        .unwrap();
        let resolved =
            super::super::resolve_daemon_config(&crate::server::GatewayOverrides::default())
                .unwrap();
        assert_eq!(resolved.gateway.response_timeout_secs, 25);
        assert_eq!(resolved.gateway.openai_base_url, "https://team.example/v1");
        assert_eq!(resolved.gateway.openai_oidc.unwrap().client_id, "team");
    }
    super::super::set_test_system_config_dir(previous_system);
}

#[test]
fn higher_definitions_replace_whole_presets_and_selection_leaves_no_residue() {
    let resolved = resolve(vec![
        document("preset = 'old'\n[presets.old.gateway]\nmax_hook_payload_bytes = 123\n[presets.team.upstream]\nopenai_base_url = 'https://old.example'\nanthropic_base_url = 'https://old.example'\n"),
        document("preset = 'team'\n[presets.team.upstream]\nopenai_base_url = 'https://new.example'\n"),
    ]).unwrap();
    assert!(resolved.get("gateway").is_none());
    assert!(resolved["upstream"].get("anthropic_base_url").is_none());
    assert_eq!(
        resolved["upstream"]["openai_base_url"].as_str(),
        Some("https://new.example")
    );
}

#[test]
fn inactive_presets_are_inert_and_no_selection_preserves_explicit_values() {
    let resolved = resolve(vec![document("[upstream]\nresponse_timeout_secs = 12\n[presets.unused.upstream]\nopenai_oidc = { invalid = true }\n")]).unwrap();
    assert_eq!(
        resolved,
        document("[upstream]\nresponse_timeout_secs = 12\n")
    );
}

#[test]
fn invalid_preset_metadata_and_unknown_names_report_the_scope() {
    for (text, expected) in [
        ("[upstream]\npreset = 'missing'", "unknown preset"),
        (
            "[upstream]\npreset = 42",
            "upstream.preset must be a non-empty string",
        ),
        (
            "[upstream]\npreset = ''",
            "upstream.preset must be a non-empty string",
        ),
        (
            "[upstream]\npresets = []",
            "upstream.presets must be a table",
        ),
        (
            "[upstream.presets]\nbroken = 'value'",
            "upstream.presets.broken must be a table",
        ),
        (
            "preset = 'loop'\n[presets.loop]\npreset = 'loop'",
            "cannot contain preset selectors",
        ),
    ] {
        let error = resolve(vec![document(text)]).unwrap_err().to_string();
        assert!(error.contains(expected), "{error}");
    }
}

#[test]
fn explicit_endpoint_change_clears_preset_credentials() {
    let resolved = resolve(vec![document(
        "preset = 'team'\n[upstream]\nopenai_base_url = 'https://other.example'\n[presets.team.upstream]\nopenai_base_url = 'https://team.example'\nopenai_oidc = { issuer = 'https://issuer.example', client_id = 'team' }\n",
    )]).unwrap();
    assert!(resolved["upstream"].get("openai_oidc").is_none());
}

#[test]
fn explicit_oidc_sections_never_inherit_preset_fields() {
    let resolved = resolve(vec![document(
        "preset = 'team'\n[upstream.openai_oidc]\nclient_id = 'different'\n[presets.team.upstream.openai_oidc]\nissuer = 'https://issuer.example'\nclient_id = 'team'\n",
    )]).unwrap();
    assert!(resolved["upstream"]["openai_oidc"].get("issuer").is_none());
    assert!(super::super::parse_file_config(resolved).is_err());
}

#[test]
fn preset_credentials_still_obey_mutual_exclusion() {
    let resolved = resolve(vec![document("preset = 'team'\n[presets.team.upstream]\nopenai_api_key = 'key'\nopenai_oidc = { issuer = 'https://issuer.example', client_id = 'team' }\n")]).unwrap();
    assert!(super::super::validate_config_document(resolved).is_err());
}

#[test]
fn selected_presets_cannot_bypass_legacy_section_validation() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("config.toml");
    std::fs::write(
        &path,
        "preset = 'legacy'\n[presets.legacy.plugins]\nvalue = true\n",
    )
    .unwrap();
    assert!(super::super::read_shared_documents(vec![(path, true)]).is_err());
}

#[test]
fn nested_selection_overrides_root_defaults_without_changing_siblings() {
    let resolved = resolve(vec![document(
        "preset = 'root'\n[presets.root.gateway]\nmax_hook_payload_bytes = 10\n[presets.root.agents.codex]\ncommand = 'root-codex'\n[agents.codex]\npreset = 'local'\n[agents.codex.presets.local]\ncommand = 'local-codex'\n",
    )]).unwrap();
    assert_eq!(
        resolved["gateway"]["max_hook_payload_bytes"].as_integer(),
        Some(10)
    );
    assert_eq!(
        resolved["agents"]["codex"]["command"].as_str(),
        Some("local-codex")
    );
}

#[test]
fn selecting_a_preset_inside_an_oidc_section_keeps_its_complete_credentials() {
    let resolved = resolve(vec![document(
        "[upstream.openai_oidc]\npreset = 'team'\n[upstream.openai_oidc.presets.team]\nissuer = 'https://issuer.example'\nclient_id = 'team'\n",
    )]).unwrap();
    let config = super::super::parse_file_config(resolved).unwrap();
    assert_eq!(
        config.upstream.unwrap().openai_oidc.unwrap().client_id,
        "team"
    );
}

#[test]
fn defining_presets_in_a_higher_layer_does_not_replace_active_oidc_credentials() {
    let resolved = resolve(vec![
        document(
            "[upstream.openai_oidc]\nissuer = 'https://issuer.example'\nclient_id = 'active'\n",
        ),
        document("[upstream.openai_oidc.presets.unused]\nclient_id = 'unused'\n"),
    ])
    .unwrap();
    assert_eq!(
        resolved["upstream"]["openai_oidc"]["client_id"].as_str(),
        Some("active")
    );
}

#[test]
fn editor_and_doctor_validate_local_presets_but_allow_external_definitions() {
    assert!(
        super::super::validate_shared_config_shape(document("[upstream]\npreset = 'external'\n"))
            .unwrap()
            .is_empty()
    );
    assert!(
        super::super::validate_config_document(document("[upstream]\npreset = 'external'\n"))
            .is_ok()
    );
    assert!(
        super::super::validate_config_document(document(
            "preset = 'bad'\n[presets.bad.upstream.openai_oidc]\nclient_id = 'missing-issuer'\n"
        ))
        .is_err()
    );
    assert_eq!(
        super::super::validate_shared_config_shape(document(
            "preset = 'typo'\n[presets.typo.upstream]\nmisspelled_key = true\n"
        ))
        .unwrap(),
        ["upstream.misspelled_key"]
    );
}

#[test]
fn startup_logging_and_daemon_resolve_presets_from_the_explicit_file() {
    let _scope = crate::test_support::CwdTestScope::locked();
    let dir = tempfile::tempdir().unwrap();
    let previous_system = super::super::set_test_system_config_dir(Some(dir.path().join("system")));
    let path = dir.path().join("config.toml");
    std::fs::write(&path, "preset = 'team'\n[presets.team.logging]\nlevel = 'warn'\n[presets.team.daemon.logging]\nlevel = 'error'\n[presets.team.upstream]\nresponse_timeout_secs = 71\n").unwrap();
    assert_eq!(
        super::super::resolve_logging_config(Some(&path))
            .unwrap()
            .level,
        nemo_relay_logging::LogLevel::Warn
    );
    assert_eq!(
        super::super::resolve_daemon_logging_config(Some(&path))
            .unwrap()
            .level,
        nemo_relay_logging::LogLevel::Error
    );
    let resolved = super::super::resolve_daemon_config(&crate::server::GatewayOverrides {
        config: Some(path),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(resolved.gateway.response_timeout_secs, 71);
    super::super::set_test_system_config_dir(previous_system);
}
