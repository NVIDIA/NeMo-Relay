// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn classifies_only_supported_public_paths() {
    assert_eq!(
        PublicRoute::from_path("/v1/messages"),
        Some(PublicRoute::Provider(ProviderRoute::Anthropic))
    );
    assert_eq!(
        PublicRoute::from_path("/hooks/codex"),
        Some(PublicRoute::Hook(HookRoute::Codex))
    );
    assert_eq!(PublicRoute::from_path("/admin"), None);
    assert_eq!(
        PublicRoute::from_path("/typesafe/v1/models"),
        Some(PublicRoute::Provider(ProviderRoute::TypeSafe))
    );
}

#[test]
fn composes_namespaced_typesafe_paths_for_the_sdk() {
    let config = GatewayConfig::default();
    assert_eq!(
        ProviderRoute::TypeSafe.upstream_url(&config, "/typesafe/v1/systemone?x=1"),
        "https://api.typesafe.ai/v1/systemone?x=1"
    );
    assert_eq!(
        ProviderRoute::TypeSafe.upstream_url(&config, "/typesafe/v1/models"),
        "https://api.typesafe.ai/v1/models"
    );
}

#[test]
fn composes_openai_v1_once() {
    let config = GatewayConfig::default();
    assert_eq!(
        ProviderRoute::OpenAi.upstream_url(&config, "/v1/responses?x=1"),
        "https://api.openai.com/v1/responses?x=1"
    );
    assert_eq!(
        ProviderRoute::OpenAi.upstream_url(&config, "/responses"),
        "https://api.openai.com/v1/responses"
    );
    assert_eq!(
        ProviderRoute::OpenAi.upstream_url(&config, "/backend-api/codex/responses?client=codex"),
        "https://api.openai.com/v1/responses?client=codex"
    );
    assert_eq!(
        PublicRoute::from_path("/backend-api/codex/responses"),
        Some(PublicRoute::Provider(ProviderRoute::OpenAi))
    );
}

#[test]
fn composes_anthropic_paths_without_openai_normalization() {
    let mut config = GatewayConfig {
        anthropic_base_url: "https://api.anthropic.com/custom/".into(),
        ..GatewayConfig::default()
    };
    assert_eq!(
        ProviderRoute::Anthropic.upstream_url(&config, "/v1/messages?beta=true"),
        "https://api.anthropic.com/custom/v1/messages?beta=true"
    );
    config.anthropic_base_url = "https://api.anthropic.com".into();
    assert_eq!(
        ProviderRoute::Anthropic.upstream_url(&config, "/v1/messages"),
        "https://api.anthropic.com/v1/messages"
    );
    config.anthropic_base_url = "https://api.anthropic.com/v1".into();
    assert_eq!(
        ProviderRoute::Anthropic.upstream_url(&config, "/v1/messages"),
        "https://api.anthropic.com/v1/messages"
    );
}
