// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider route classification and agent alignment policy.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ProviderRoute {
    OpenAiResponses,
    OpenAiChatCompletions,
    OpenAiModels,
    AnthropicMessages,
    AnthropicCountTokens,
}

#[derive(Clone, Copy)]
pub(super) struct ProviderForwarding {
    pub(super) route: ProviderRoute,
    pub(super) allow_environment_provider_auth: bool,
}

impl ProviderForwarding {
    pub(super) fn new(route: ProviderRoute, allow_environment_provider_auth: bool) -> Self {
        Self {
            route,
            allow_environment_provider_auth,
        }
    }
}

impl ProviderRoute {
    // Maps public gateway paths to known upstream provider routes. Unsupported paths return `None`
    // so the caller can fail as a bad hook/gateway payload instead of constructing arbitrary URLs.
    pub(super) fn from_path(path: &str) -> Option<Self> {
        match path {
            "/responses" => Some(Self::OpenAiResponses),
            "/v1/responses" => Some(Self::OpenAiResponses),
            "/chat/completions" => Some(Self::OpenAiChatCompletions),
            "/v1/chat/completions" => Some(Self::OpenAiChatCompletions),
            "/models" => Some(Self::OpenAiModels),
            "/v1/models" => Some(Self::OpenAiModels),
            "/v1/messages" => Some(Self::AnthropicMessages),
            "/v1/messages/count_tokens" => Some(Self::AnthropicCountTokens),
            _ => None,
        }
    }

    pub(super) fn from_dispatch_override(value: &str) -> Option<Self> {
        match value {
            "openai_chat"
            | "openai_chat_completions"
            | "openai.chat_completions"
            | "/v1/chat/completions" => Some(Self::OpenAiChatCompletions),
            "openai_responses" | "openai.responses" | "/v1/responses" => {
                Some(Self::OpenAiResponses)
            }
            "anthropic_messages" | "anthropic.messages" | "/v1/messages" => {
                Some(Self::AnthropicMessages)
            }
            _ => None,
        }
    }

    pub(super) const fn provider_surface(self) -> Option<ProviderSurface> {
        match self {
            Self::OpenAiResponses => Some(ProviderSurface::OpenAIResponses),
            Self::OpenAiChatCompletions => Some(ProviderSurface::OpenAIChat),
            Self::AnthropicMessages => Some(ProviderSurface::AnthropicMessages),
            Self::AnthropicCountTokens | Self::OpenAiModels => None,
        }
    }

    // Returns the provider route name recorded on managed LLM events. These names split OpenAI API
    // variants because their request/response schemas differ even when they share a base URL, and
    // they double as codec hints for ambiguous provider request shapes.
    pub(super) const fn name(self) -> &'static str {
        self.alignment_route().name()
    }

    // Builds the upstream URL by combining the configured provider base with the original path and
    // query string. Trailing slashes are stripped from the base to avoid double-slash variants in
    // configured enterprise or local proxy endpoints.
    pub(super) fn upstream_url(
        self,
        config: &crate::configuration::GatewayConfig,
        path_and_query: &str,
    ) -> String {
        let base = match self {
            Self::OpenAiResponses | Self::OpenAiChatCompletions | Self::OpenAiModels => {
                config.openai_base_url.as_str()
            }
            Self::AnthropicMessages | Self::AnthropicCountTokens => {
                config.anthropic_base_url.as_str()
            }
        };
        self.upstream_url_with_base(base, path_and_query)
    }

    // Like `upstream_url` but with an explicit base URL. This keeps OpenAI `/v1` normalization in
    // one place for configured public, enterprise, or local proxy bases.
    pub(super) fn upstream_url_with_base(self, base: &str, path_and_query: &str) -> String {
        let base = base.trim_end_matches('/');
        let path_and_query = match self {
            Self::OpenAiResponses | Self::OpenAiChatCompletions | Self::OpenAiModels => {
                normalize_openai_path_for_base(base, path_and_query)
            }
            _ => path_and_query.to_string(),
        };
        format!("{base}{path_and_query}")
    }

    // Narrows gateway routing to the smaller taxonomy used by trace alignment. Keeping this
    // conversion here prevents provider-specific alignment code from depending on gateway URL
    // routing internals.
    pub(super) const fn alignment_route(self) -> GatewayRouteKind {
        match self {
            Self::OpenAiResponses => GatewayRouteKind::OpenAiResponses,
            Self::OpenAiChatCompletions => GatewayRouteKind::OpenAiChatCompletions,
            Self::OpenAiModels => GatewayRouteKind::OpenAiModels,
            Self::AnthropicMessages => GatewayRouteKind::AnthropicMessages,
            Self::AnthropicCountTokens => GatewayRouteKind::AnthropicCountTokens,
        }
    }
}

pub(super) fn normalize_openai_path_for_base(base: &str, path_and_query: &str) -> String {
    match (base.ends_with("/v1"), path_and_query.starts_with("/v1/")) {
        (true, true) => path_and_query
            .strip_prefix("/v1")
            .expect("path was checked to start with /v1")
            .to_string(),
        (false, false) => format!("/v1{path_and_query}"),
        _ => path_and_query.to_string(),
    }
}

// Gives alignment adapters a chance to choose an agent-native upstream before default provider
// routing runs. Today this supports Codex ChatGPT auth; future harness fallbacks should stay in
// alignment rather than adding provider-shaped checks here.
pub(super) fn gateway_upstream_url_override(
    route: ProviderRoute,
    headers: &HeaderMap,
    path_and_query: &str,
    allow_environment_provider_auth: bool,
) -> Option<String> {
    gateway_upstream_url_override_with_openai_key_state(
        route,
        headers,
        path_and_query,
        allow_environment_provider_auth && env_var_is_nonempty("OPENAI_API_KEY"),
    )
}

pub(super) fn gateway_upstream_url_override_with_openai_key_state(
    route: ProviderRoute,
    headers: &HeaderMap,
    path_and_query: &str,
    has_openai_replacement_key: bool,
) -> Option<String> {
    alignment::gateway_upstream_url_override(
        headers,
        route.alignment_route(),
        path_and_query,
        has_openai_replacement_key,
    )
}

// Lets alignment adapters strip agent-native credentials only when the gateway can replace them
// with standard provider API keys. Whitespace-only env vars are treated as missing because
// forwarding an empty bearer value only replaces one authentication failure with another.
pub(super) fn strip_replaceable_agent_auth_headers(
    headers: &HeaderMap,
    route: ProviderRoute,
    allow_environment_provider_auth: bool,
) -> HeaderMap {
    strip_replaceable_agent_auth_headers_with_openai_key_state(
        headers,
        route,
        allow_environment_provider_auth && env_var_is_nonempty("OPENAI_API_KEY"),
    )
}

pub(super) fn strip_replaceable_agent_auth_headers_with_openai_key_state(
    headers: &HeaderMap,
    route: ProviderRoute,
    has_openai_replacement_key: bool,
) -> HeaderMap {
    alignment::gateway_forward_headers(headers, route.alignment_route(), has_openai_replacement_key)
}

pub(super) fn env_var_is_nonempty(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .is_some()
}

// Delegates provider-specific session fallbacks to `alignment` so request construction stays
// generic and each coding-agent quirk has one documented adapter.
pub(super) fn gateway_session_id(
    headers: &HeaderMap,
    body: &Value,
    route: ProviderRoute,
) -> Option<String> {
    alignment::gateway_session_id(headers, body, route.alignment_route())
}

pub(super) fn gateway_subagent_id(
    headers: &HeaderMap,
    body: &Value,
    route: ProviderRoute,
) -> Option<String> {
    alignment::gateway_subagent_id(headers, body, route.alignment_route())
}

// Keeps the gateway-facing helper local for tests while the generic extraction pattern lives in
// `alignment`.
pub(super) fn gateway_identifier(
    headers: &HeaderMap,
    body: &Value,
    header_name: &'static str,
    body_paths: &[&[&str]],
) -> Option<String> {
    alignment::gateway_identifier(headers, body, header_name, body_paths)
}

// Copies only non-sensitive, forwardable request headers into LLM request metadata. This preserves
// correlation headers while excluding credentials and hop-by-hop transport details.
