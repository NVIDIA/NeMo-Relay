// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Credential-free requests for host-owned provider execution.

use serde::{Deserialize, Serialize};

use crate::Json;

/// Provider protocol accepted by a configured caller-credential target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmProviderFormat {
    /// OpenAI Chat Completions.
    OpenaiChat,
    /// OpenAI Responses.
    OpenaiResponses,
    /// Anthropic Messages.
    AnthropicMessages,
}

/// One provider call authorized and executed by the host.
///
/// The target is a host-configured name, never a URL. Credentials, headers,
/// and credential-selection overrides are deliberately absent.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LlmProviderRequest {
    /// Name of the host-authorized destination.
    pub target: String,
    /// Request body in the destination's configured provider format.
    pub content: Json,
}
