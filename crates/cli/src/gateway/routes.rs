// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider route classification and agent alignment policy.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ProviderRoute {
    OpenAiResponses,
    OpenAiChatCompletions,
    OpenAiImagesGenerations,
    OpenAiModels,
    AnthropicMessages,
    AnthropicCountTokens,
}

#[derive(Clone)]
pub(super) struct ProviderForwarding {
    pub(super) source_route: ProviderRoute,
    pub(super) authorization: crate::provider_auth::ProviderRequestAuthorization,
    openai_auth_header: Option<String>,
    anthropic_auth_header: Option<String>,
}

impl ProviderForwarding {
    pub(super) fn new(
        source_route: ProviderRoute,
        authorization: crate::provider_auth::ProviderRequestAuthorization,
        config: &crate::configuration::GatewayConfig,
    ) -> Self {
        Self {
            source_route,
            authorization,
            openai_auth_header: config.openai_auth_header.clone(),
            anthropic_auth_header: config.anthropic_auth_header.clone(),
        }
    }

    pub(super) fn configured_auth_header(&self, route: ProviderRoute) -> Option<&str> {
        configured_auth_header(
            route,
            self.openai_auth_header.as_deref(),
            self.anthropic_auth_header.as_deref(),
        )
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
            "/v1/images/generations" => Some(Self::OpenAiImagesGenerations),
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
            "openai_images_generations"
            | "openai.images.generations"
            | "/v1/images/generations" => Some(Self::OpenAiImagesGenerations),
            "openai_models" | "openai.models" | "/models" | "/v1/models" => {
                Some(Self::OpenAiModels)
            }
            "anthropic_messages" | "anthropic.messages" | "/v1/messages" => {
                Some(Self::AnthropicMessages)
            }
            "anthropic_count_tokens" | "anthropic.count_tokens" | "/v1/messages/count_tokens" => {
                Some(Self::AnthropicCountTokens)
            }
            _ => None,
        }
    }

    pub(super) const fn provider_surface(self) -> Option<ProviderSurface> {
        match self {
            Self::OpenAiResponses => Some(ProviderSurface::OpenAIResponses),
            Self::OpenAiChatCompletions => Some(ProviderSurface::OpenAIChat),
            Self::AnthropicMessages => Some(ProviderSurface::AnthropicMessages),
            Self::AnthropicCountTokens | Self::OpenAiImagesGenerations | Self::OpenAiModels => None,
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
            Self::OpenAiResponses
            | Self::OpenAiChatCompletions
            | Self::OpenAiImagesGenerations
            | Self::OpenAiModels => config.openai_base_url.as_str(),
            Self::AnthropicMessages | Self::AnthropicCountTokens => {
                config.anthropic_base_url.as_str()
            }
        };
        self.upstream_url_with_base(base, path_and_query)
    }

    pub(super) fn configured_auth_header(
        self,
        config: &crate::configuration::GatewayConfig,
    ) -> Option<&str> {
        configured_auth_header(
            self,
            config.openai_auth_header.as_deref(),
            config.anthropic_auth_header.as_deref(),
        )
    }

    // Like `upstream_url` but with an explicit base URL. This keeps OpenAI `/v1` normalization in
    // one place for configured public, enterprise, or local proxy bases.
    pub(super) fn upstream_url_with_base(self, base: &str, path_and_query: &str) -> String {
        let base = base.trim_end_matches('/');
        let path_and_query = match self {
            Self::OpenAiResponses
            | Self::OpenAiChatCompletions
            | Self::OpenAiImagesGenerations
            | Self::OpenAiModels => normalize_openai_path_for_base(base, path_and_query),
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
            Self::OpenAiImagesGenerations => GatewayRouteKind::OpenAiImagesGenerations,
            Self::OpenAiModels => GatewayRouteKind::OpenAiModels,
            Self::AnthropicMessages => GatewayRouteKind::AnthropicMessages,
            Self::AnthropicCountTokens => GatewayRouteKind::AnthropicCountTokens,
        }
    }
}

fn configured_auth_header<'a>(
    route: ProviderRoute,
    openai_auth_header: Option<&'a str>,
    anthropic_auth_header: Option<&'a str>,
) -> Option<&'a str> {
    match route {
        ProviderRoute::OpenAiResponses
        | ProviderRoute::OpenAiChatCompletions
        | ProviderRoute::OpenAiImagesGenerations
        | ProviderRoute::OpenAiModels => openai_auth_header,
        ProviderRoute::AnthropicMessages | ProviderRoute::AnthropicCountTokens => {
            anthropic_auth_header
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
    config: &crate::configuration::GatewayConfig,
) -> Option<String> {
    gateway_upstream_url_override_with_openai_key_state(
        route,
        headers,
        path_and_query,
        has_openai_replacement_auth(
            route,
            allow_environment_provider_auth,
            route.configured_auth_header(config),
        ),
    )
}

// The upstream this request names for itself, composed with the request path.
//
// Separate from `gateway_upstream_url_override` because the two answer different questions: that
// one asks alignment which upstream a *harness* implies, from evidence the client did not choose
// (a ChatGPT token's shape); this one is the client stating a destination outright, which is only
// honored when it proved it is this invocation's own agent. Composition stays here so the OpenAI
// `/v1` normalization has one home.
// Composed by plain concatenation, deliberately *not* through `upstream_url_with_base`.
//
// That helper normalizes `/v1` for a *configured* base, where the operator may or may not have
// included it and Relay has to reconcile the two. A named base is not that: it is the endpoint
// the client would otherwise have called itself, and the path is the one its own SDK produced.
// Reconciling them changes the destination. Concretely, Relay registers its root as pi's base,
// so pi's OpenAI SDK sends `/chat/completions`; normalizing that against a base of
// `https://api.example.com` yields `https://api.example.com/v1/chat/completions`, while pi
// unredirected would have called `https://api.example.com/chat/completions`. Bases that already
// end in `/v1` hide the difference, which is why it survived a live run against one.
pub(super) fn client_named_upstream_url(
    route: ProviderRoute,
    headers: &HeaderMap,
    path_and_query: &str,
    invocation_authenticated: bool,
) -> crate::agents::pi::alignment::NamedUpstream {
    let _ = route;
    match crate::agents::pi::alignment::client_named_upstream_base(
        headers,
        invocation_authenticated,
    ) {
        crate::agents::pi::alignment::NamedUpstream::Named(base) => {
            crate::agents::pi::alignment::NamedUpstream::Named(format!(
                "{}{path_and_query}",
                base.trim_end_matches('/')
            ))
        }
        other => other,
    }
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
    configured_auth_header: Option<&str>,
) -> HeaderMap {
    strip_replaceable_agent_auth_headers_with_openai_key_state(
        headers,
        route,
        has_openai_replacement_auth(
            route,
            allow_environment_provider_auth,
            configured_auth_header,
        ),
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

fn has_openai_replacement_auth(
    route: ProviderRoute,
    allow_environment_provider_auth: bool,
    configured_auth_header: Option<&str>,
) -> bool {
    allow_environment_provider_auth
        && matches!(
            route,
            ProviderRoute::OpenAiResponses
                | ProviderRoute::OpenAiChatCompletions
                | ProviderRoute::OpenAiImagesGenerations
                | ProviderRoute::OpenAiModels
        )
        && (configured_auth_header.is_some() || env_var_is_nonempty("OPENAI_API_KEY"))
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
