// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Re-exported normalized LLM request data types.

use crate::json::Json;
pub use nemo_relay_types::codec::request::*;

/// Normalize identities without exposing opaque provider configuration as telemetry.
/// Native identities are derived only from recognized provider tool shapes.
pub(crate) fn tool_definition_identities(tool: &ToolDefinition) -> Vec<(&str, &str)> {
    match tool {
        ToolDefinition::Function { function, .. } => {
            if function.name.trim().is_empty() {
                Vec::new()
            } else {
                vec![("function", &function.name)]
            }
        }
        ToolDefinition::ProviderNative {
            provider,
            kind,
            value,
        } => {
            if let Some(name) = value.get("name") {
                let tool_type = value.get("type").and_then(Json::as_str).unwrap_or(kind);
                return name
                    .as_str()
                    .filter(|name| !name.trim().is_empty() && !tool_type.trim().is_empty())
                    .map(|name| vec![(tool_type, name)])
                    .unwrap_or_default();
            }
            match provider.as_str() {
                "openai_responses" => value
                    .get("type")
                    .and_then(Json::as_str)
                    .filter(|kind| {
                        matches!(
                            *kind,
                            "web_search"
                                | "web_search_preview"
                                | "file_search"
                                | "computer_use_preview"
                                | "code_interpreter"
                                | "image_generation"
                                | "local_shell"
                                | "shell"
                                | "apply_patch"
                        )
                    })
                    .map(|kind| vec![(kind, kind)])
                    .unwrap_or_default(),
                "gemini" => value
                    .as_object()
                    .map(|group| {
                        group
                            .iter()
                            .filter(|(key, value)| {
                                value.is_object()
                                    && matches!(
                                        key.as_str(),
                                        "googleSearch"
                                            | "googleSearchRetrieval"
                                            | "codeExecution"
                                            | "urlContext"
                                            | "retrieval"
                                            | "googleMaps"
                                            | "enterpriseWebSearch"
                                    )
                            })
                            .map(|(key, _)| (key.as_str(), key.as_str()))
                            .collect()
                    })
                    .unwrap_or_default(),
                _ => Vec::new(),
            }
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/codec/request_tests.rs"]
mod tests;
