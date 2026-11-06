// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;

use super::*;

const NONCE: &str = "n0nce_Abc-123xyz";
const WORKSPACE: &str = "0b6f6c8e-6d59-4d5b-9a4a-2f1d8f1c0a11";

fn source() -> OidcSource {
    OidcSource::new("https://clerk.openrouter.ai", "client", None)
}

fn parse(toml_text: &str) -> OidcSource {
    toml::from_str(&format!(
        "issuer = \"https://clerk.openrouter.ai\"\nclient_id = \"client\"\n{toml_text}"
    ))
    .unwrap()
}

fn payload(state: &str) -> String {
    let (head, segment) = state.split_once('.').expect("a restriction segment");
    assert_eq!(head, NONCE);
    assert!(
        !segment.contains('.') && !segment.contains('='),
        "{segment}"
    );
    String::from_utf8(URL_SAFE_NO_PAD.decode(segment).unwrap()).unwrap()
}

#[test]
fn an_unrestricted_section_sends_the_bare_random_state() {
    assert_eq!(source().authorization_state(NONCE), NONCE);
}

#[test]
fn organization_and_workspace_encode_as_ori_does() {
    let state = parse(&format!(
        "organization_id = \"org_required\"\nworkspace_id = \"{WORKSPACE}\""
    ))
    .authorization_state(NONCE);

    assert_eq!(
        payload(&state),
        format!(r#"{{"organizationId":"org_required","workspaceId":"{WORKSPACE}"}}"#)
    );
}

#[test]
fn an_unconstrained_dimension_is_omitted() {
    let state = parse("organization_id = \"org_required\"").authorization_state(NONCE);
    assert_eq!(payload(&state), r#"{"organizationId":"org_required"}"#);

    let state = parse(&format!("workspace_id = \"{WORKSPACE}\"")).authorization_state(NONCE);
    assert_eq!(
        payload(&state),
        format!(r#"{{"workspaceId":"{WORKSPACE}"}}"#)
    );
}

#[test]
fn hidden_fields_follow_the_organization_and_workspace() {
    let state = parse(
        "organization_id = \"org_required\"\nhidden_fields = [\"keyLabel\", \"existingKey\"]",
    )
    .authorization_state(NONCE);

    assert_eq!(
        payload(&state),
        r#"{"organizationId":"org_required","hiddenFields":["keyLabel","existingKey"]}"#
    );
}

#[test]
fn a_credit_limit_encodes_as_javascript_writes_the_number() {
    for (written, encoded) in [("25", "25"), ("25.0", "25"), ("0", "0"), ("12.5", "12.5")] {
        let state = parse(&format!("credit_limit = {written}")).authorization_state(NONCE);
        assert_eq!(payload(&state), format!(r#"{{"creditLimit":{encoded}}}"#));
    }
}

#[test]
fn every_member_is_written_in_the_consent_service_order() {
    let state = parse(&format!(
        "hidden_fields = [\"creditLimit\"]\ncredit_limit_reset = \"monthly\"\ncredit_limit = 25\nworkspace_id = \"{WORKSPACE}\"\norganization_id = \"org_required\""
    ))
    .authorization_state(NONCE);

    assert_eq!(
        payload(&state),
        format!(
            r#"{{"organizationId":"org_required","workspaceId":"{WORKSPACE}","creditLimit":25,"creditLimitReset":"monthly","hiddenFields":["creditLimit"]}}"#
        )
    );
}

#[test]
fn restrictions_the_consent_service_would_refuse_fail_when_the_configuration_loads() {
    let refused = |toml_text: &str| {
        parse(toml_text)
            .validate_consent_restrictions("upstream.openai_oidc")
            .unwrap_err()
            .to_string()
    };

    assert!(refused("credit_limit_reset = \"monthly\"").contains("needs credit_limit"));
    assert!(refused("credit_limit = -1").contains("zero or more"));
    assert!(refused("organization_id = \"\"").contains("must not be empty"));
    assert!(refused("hidden_fields = [\"keyLabel\"]").contains("needs organization_id"));
    assert!(
        refused("credit_limit = 5\nhidden_fields = [\"keyLabel\", \"keyLabel\"]").contains("twice")
    );
    let long = "o".repeat(800);
    assert!(refused(&format!("organization_id = \"{long}\"")).contains("1024"));
    assert!(
        toml::from_str::<OidcSource>(
            "issuer = \"https://i.example\"\nclient_id = \"c\"\ncredit_limit_reset = \"yearly\""
        )
        .is_err()
    );

    let accepted = parse("credit_limit = 25\ncredit_limit_reset = \"daily\"");
    assert!(
        accepted
            .validate_consent_restrictions("upstream.openai_oidc")
            .is_ok()
    );
    assert!(
        source()
            .validate_consent_restrictions("upstream.openai_oidc")
            .is_ok()
    );
}
