// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stable public route classification shared by daemon pass-through and workers.

use crate::configuration::GatewayConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PublicRoute {
    Hook(HookRoute),
    Provider(ProviderRoute),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HookRoute {
    Codex,
    Claude,
    Pi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProviderRoute {
    OpenAi,
    Anthropic,
}

impl PublicRoute {
    pub(crate) fn from_path(path: &str) -> Option<Self> {
        match path {
            "/hooks/codex" => Some(Self::Hook(HookRoute::Codex)),
            "/hooks/claude-code" => Some(Self::Hook(HookRoute::Claude)),
            "/hooks/pi" => Some(Self::Hook(HookRoute::Pi)),
            "/responses"
            | "/chat/completions"
            | "/models"
            | "/v1/responses"
            | "/backend-api/codex/responses"
            | "/v1/chat/completions"
            | "/responses/compact"
            | "/v1/responses/compact"
            | "/backend-api/codex/responses/compact"
            | "/realtime/calls"
            | "/v1/realtime/calls"
            | "/backend-api/codex/realtime/calls"
            | "/realtime"
            | "/v1/realtime"
            | "/live"
            | "/v1/live"
            | "/live/sessions"
            | "/v1/live/sessions"
            | "/v1/images/generations"
            | "/images/edits"
            | "/v1/images/edits"
            | "/memories/trace_summarize"
            | "/v1/memories/trace_summarize"
            | "/alpha/search"
            | "/v1/alpha/search"
            | "/v1/models" => Some(Self::Provider(ProviderRoute::OpenAi)),
            "/v1/messages" | "/v1/messages/count_tokens" => {
                Some(Self::Provider(ProviderRoute::Anthropic))
            }
            path if crate::gateway::websocket::live_sideband_path(path) => {
                Some(Self::Provider(ProviderRoute::OpenAi))
            }
            _ => None,
        }
    }
}

impl HookRoute {
    pub(crate) const fn pass_through_body(self) -> &'static [u8] {
        match self {
            Self::Codex | Self::Pi => b"{}",
            Self::Claude => br#"{"continue":true}"#,
        }
    }
}

impl ProviderRoute {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Anthropic => "anthropic",
        }
    }

    pub(crate) fn upstream_url(self, config: &GatewayConfig, path_and_query: &str) -> String {
        let base = match self {
            Self::OpenAi => config.openai_base_url.as_str(),
            Self::Anthropic => config.anthropic_base_url.as_str(),
        }
        .trim_end_matches('/');
        let path = match self {
            Self::OpenAi => canonical_openai_path(path_and_query),
            Self::Anthropic => path_and_query.to_owned(),
        };
        let path = normalize_v1_path(base, &path);
        format!("{base}{path}")
    }
}

fn canonical_openai_path(path_and_query: &str) -> String {
    path_and_query
        .strip_prefix("/backend-api/codex/responses")
        .map_or_else(
            || path_and_query.to_owned(),
            |suffix| format!("/responses{suffix}"),
        )
}

fn normalize_v1_path(base: &str, path_and_query: &str) -> String {
    match (base.ends_with("/v1"), path_and_query.starts_with("/v1/")) {
        (true, true) => path_and_query
            .strip_prefix("/v1")
            .expect("prefix was checked")
            .to_owned(),
        (false, false) => format!("/v1{path_and_query}"),
        _ => path_and_query.to_owned(),
    }
}

#[cfg(test)]
#[path = "../../../tests/coverage/daemon/routes_tests.rs"]
mod tests;
