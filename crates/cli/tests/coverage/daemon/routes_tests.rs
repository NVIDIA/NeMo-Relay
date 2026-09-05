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
