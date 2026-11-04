// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn source(issuer: &str) -> OidcSource {
    OidcSource::new(issuer, "relay", Some("openid offline_access inference"))
}

fn config(openai: Option<OidcSource>, anthropic: Option<OidcSource>) -> GatewayConfig {
    GatewayConfig {
        openai_oidc: openai,
        anthropic_oidc: anthropic,
        ..Default::default()
    }
}

fn configured_without_base_url_environment(server: &ServerArgs) -> Result<GatewayConfig, CliError> {
    let _env = crate::test_support::EnvScope::set(&[
        ("NEMO_RELAY_OPENAI_BASE_URL", None),
        ("NEMO_RELAY_ANTHROPIC_BASE_URL", None),
    ]);
    configured(server)
}

fn args(provider: Option<ProviderArg>) -> ProviderArgs {
    ProviderArgs {
        provider,
        ..Default::default()
    }
}

#[test]
fn login_uses_the_configured_sign_in_section() {
    let configured = source("https://openai-issuer.example");

    let selected =
        select_source(args(None), || Ok(config(Some(configured.clone()), None))).unwrap();

    assert_eq!(selected, configured);
}

#[test]
fn login_works_with_only_an_anthropic_sign_in() {
    let configured = source("https://anthropic-issuer.example");

    let selected =
        select_source(args(None), || Ok(config(None, Some(configured.clone())))).unwrap();

    assert_eq!(selected, configured);
}

#[test]
fn two_different_sign_ins_need_an_explicit_provider() {
    let load = || {
        Ok(config(
            Some(source("https://openai-issuer.example")),
            Some(source("https://anthropic-issuer.example")),
        ))
    };

    let error = select_source(args(None), load).unwrap_err().to_string();
    assert!(error.contains("--provider"), "{error}");

    let selected = select_source(args(Some(ProviderArg::Anthropic)), load).unwrap();
    assert_eq!(selected.issuer, "https://anthropic-issuer.example");
}

#[test]
fn a_missing_section_names_what_to_configure() {
    let error = select_source(args(Some(ProviderArg::Openai)), || Ok(config(None, None)))
        .unwrap_err()
        .to_string();

    assert!(error.contains("[upstream.openai_oidc]"), "{error}");
}

#[test]
fn explicit_flags_skip_the_configuration() {
    let selected = select_source(
        ProviderArgs {
            issuer: Some("https://issuer.example/".into()),
            client_id: Some("relay".into()),
            ..Default::default()
        },
        || panic!("the configuration must not be read"),
    )
    .unwrap();

    assert_eq!(selected.issuer, "https://issuer.example");
    assert_eq!(selected.scope, "openid offline_access");
}

#[test]
fn login_reads_the_file_named_by_the_global_config_flag() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("team.toml");
    std::fs::write(
        &path,
        "[upstream.anthropic_oidc]\nissuer = \"https://team-issuer.example\"\nclient_id = \"team\"\n",
    )
    .unwrap();
    let server = ServerArgs {
        config: Some(path),
        ..Default::default()
    };

    let selected = select_source(args(None), || {
        configured_without_base_url_environment(&server)
    })
    .unwrap();

    assert_eq!(selected.issuer, "https://team-issuer.example");
    assert_eq!(selected.client_id, "team");
}

#[test]
fn the_openrouter_preset_needs_no_configuration() {
    let selected = select_source(
        ProviderArgs {
            preset: Some(OidcPreset::Openrouter),
            ..Default::default()
        },
        || panic!("the configuration must not be read"),
    )
    .unwrap();

    assert_eq!(selected.issuer, "https://clerk.openrouter.ai");
    assert_eq!(selected.client_id, "q3AE4KTq8gq6HlU3");
    assert!(selected.scope.contains("inference"));
}

#[test]
fn a_config_section_with_only_a_preset_is_complete() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("team.toml");
    std::fs::write(&path, "[upstream.openai_oidc]\npreset = \"openrouter\"\n").unwrap();
    let server = ServerArgs {
        config: Some(path),
        ..Default::default()
    };

    let selected = select_source(args(None), || {
        configured_without_base_url_environment(&server)
    })
    .unwrap();

    assert_eq!(selected, OidcPreset::Openrouter.source());
}

#[test]
fn explicit_fields_override_the_preset_and_missing_ones_fail_without_it() {
    let parse = |text: &str| toml::from_str::<OidcSource>(text);

    let custom = parse("preset = \"openrouter\"\nscope = \"openid\"").unwrap();
    assert_eq!(custom.client_id, "q3AE4KTq8gq6HlU3");
    assert_eq!(custom.scope, "openid");

    let error = parse("client_id = \"relay\"").unwrap_err().to_string();
    assert!(
        error.contains("issuer is required unless preset is set"),
        "{error}"
    );
}

#[test]
fn login_with_only_the_upstream_preset_needs_no_provider_flag() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("team.toml");
    std::fs::write(&path, "[upstream]\npreset = \"openrouter\"\n").unwrap();
    let server = ServerArgs {
        config: Some(path),
        ..Default::default()
    };

    let selected = select_source(args(None), || {
        configured_without_base_url_environment(&server)
    })
    .unwrap();

    assert_eq!(selected, OidcPreset::Openrouter.source());
}
