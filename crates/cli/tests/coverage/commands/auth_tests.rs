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

#[test]
fn a_sign_in_section_needs_an_issuer_and_a_client_id() {
    let parse = |text: &str| toml::from_str::<OidcSource>(text).map_err(|e| e.to_string());

    let error = parse("client_id = \"relay\"").unwrap_err();
    assert!(error.contains("issuer"), "{error}");
    let error = parse("issuer = \"https://issuer.example\"").unwrap_err();
    assert!(error.contains("client_id"), "{error}");
    let section = parse("issuer = \"https://issuer.example\"\nclient_id = \"relay\"").unwrap();
    assert_eq!(section.scope, "openid offline_access");
}

#[test]
fn logout_names_the_token_that_stays_valid_when_the_issuer_revokes_only_one() {
    use crate::oidc::{Logout, Revocation};
    let issuer = "https://issuer.example";
    let access_only = logout_message(
        issuer,
        &Logout::Removed(Revocation::AccessTokenOnly("invalid_request".into())),
    );
    assert_eq!(
        access_only,
        "Removed the sign-in for https://issuer.example. https://issuer.example revoked the access token but not the refresh token (invalid_request), so the refresh token stays valid until it expires."
    );
    let refresh_only = logout_message(
        issuer,
        &Logout::Removed(Revocation::RefreshTokenOnly(
            "unsupported_token_type".into(),
        )),
    );
    assert!(
        refresh_only.contains("so the access token stays valid until it expires"),
        "{refresh_only}"
    );
}
