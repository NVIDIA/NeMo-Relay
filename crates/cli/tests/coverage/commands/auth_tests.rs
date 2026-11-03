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

    let selected = select_source(args(None), || configured(&server)).unwrap();

    assert_eq!(selected.issuer, "https://team-issuer.example");
    assert_eq!(selected.client_id, "team");
}
