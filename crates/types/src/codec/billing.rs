// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider token accounting contracts and checked usage arithmetic.

use crate::Json;
use crate::codec::request::{ContentPart, MessageContent};
use crate::codec::response::{ApiSpecificResponse, Usage};
use std::borrow::Cow;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[doc(hidden)]
#[allow(missing_docs)]
pub struct SeparateUsageCounts {
    pub thoughts_tokens: Option<u64>,
    pub tool_use_prompt_tokens: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
#[allow(missing_docs)]
pub enum ToolUsePromptBilling {
    InputTokens,
    GroundingRequests,
    Undetermined,
}

impl ToolUsePromptBilling {
    #[doc(hidden)]
    #[allow(missing_docs)]
    pub fn from_tool_evidence(
        grounding_metadata: Option<&Json>,
        candidate_extra: &serde_json::Map<String, Json>,
        message: Option<&MessageContent>,
    ) -> Self {
        let mut grounding_requests = false;
        let mut input_tokens = has_url_context(candidate_extra) || has_code_execution(message);
        if let Some(metadata) = grounding_metadata {
            grounding_requests = has_nonempty_string(metadata.get("webSearchQueries"))
                || has_nonempty_string(metadata.get("imageSearchQueries"))
                || metadata
                    .get("googleMapsWidgetContextToken")
                    .and_then(Json::as_str)
                    .is_some_and(|token| !token.is_empty());
            let chunks = metadata.get("groundingChunks").and_then(Json::as_array);
            for chunk in chunks.into_iter().flatten() {
                if chunk.get("maps").is_some() {
                    grounding_requests = true;
                }
                if let Some(context) = chunk.get("retrievedContext") {
                    let file_search = context
                        .get("fileSearchStore")
                        .and_then(Json::as_str)
                        .is_some_and(|store| !store.is_empty());
                    if !file_search {
                        return Self::Undetermined;
                    }
                    input_tokens = true;
                }
            }
        }
        match (grounding_requests, input_tokens) {
            (true, false) => Self::GroundingRequests,
            (false, true) => Self::InputTokens,
            (true, true) | (false, false) => Self::Undetermined,
        }
    }
}

fn has_nonempty_string(values: Option<&Json>) -> bool {
    values.and_then(Json::as_array).is_some_and(|values| {
        values
            .iter()
            .any(|value| value.as_str().is_some_and(|value| !value.is_empty()))
    })
}

fn has_url_context(candidate_extra: &serde_json::Map<String, Json>) -> bool {
    candidate_extra
        .get("urlContextMetadata")
        .and_then(|metadata| metadata.get("urlMetadata"))
        .and_then(Json::as_array)
        .is_some_and(|urls| !urls.is_empty())
}

fn has_code_execution(message: Option<&MessageContent>) -> bool {
    let Some(MessageContent::Parts(parts)) = message else {
        return false;
    };
    parts.iter().any(|part| {
        matches!(
            part,
            ContentPart::ProviderNative { kind, .. }
                if matches!(
                    kind.as_str(),
                    "executableCode"
                        | "codeExecutionResult"
                        | "executable_code"
                        | "code_execution_result"
                )
        )
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
#[allow(missing_docs)]
pub struct GeminiBilling {
    pub separate: SeparateUsageCounts,
    pub tool_use_billing: ToolUsePromptBilling,
}

#[doc(hidden)]
#[allow(missing_docs)]
pub struct BilledUsage<'a> {
    pub usage: Cow<'a, Usage>,
    pub reported: &'a Usage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
#[allow(missing_docs)]
pub enum UnbillableUsage {
    TokenOverflow,
    UndeterminedToolUsePromptTokens,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
#[allow(missing_docs)]
pub enum InputSplit {
    Separate,
    Inclusive,
    Either,
}

impl InputSplit {
    #[doc(hidden)]
    #[allow(missing_docs)]
    pub fn of_api(api: &ApiSpecificResponse) -> Option<Self> {
        match api {
            ApiSpecificResponse::AnthropicMessages { .. } => Some(Self::Separate),
            ApiSpecificResponse::OpenAIChat { .. }
            | ApiSpecificResponse::OpenAIResponses { .. }
            | ApiSpecificResponse::OCIGenAI { .. }
            | ApiSpecificResponse::GeminiGenerateContent { .. } => Some(Self::Inclusive),
            ApiSpecificResponse::Custom { .. } => None,
        }
    }
}

#[doc(hidden)]
#[allow(missing_docs)]
pub fn input_split(usage: &Usage) -> Option<InputSplit> {
    let uncached = usage.uncached_input_tokens?;
    let with_cache = input_tokens_with_cache(usage, uncached)?;
    let Some(prompt) = usage.prompt_tokens else {
        return Some(InputSplit::Either);
    };
    match (prompt == uncached, prompt == with_cache) {
        (true, true) => Some(InputSplit::Either),
        (true, false) => Some(InputSplit::Separate),
        (false, true) => Some(InputSplit::Inclusive),
        (false, false) => None,
    }
}

#[doc(hidden)]
pub fn input_tokens_with_cache(usage: &Usage, uncached: u64) -> Option<u64> {
    [usage.cache_read_tokens, usage.cache_write_tokens]
        .into_iter()
        .flatten()
        .try_fold(uncached, u64::checked_add)
}

#[doc(hidden)]
#[allow(missing_docs)]
pub fn billed_usage<'a>(
    usage: &'a Usage,
    gemini: Option<GeminiBilling>,
) -> Result<BilledUsage<'a>, UnbillableUsage> {
    let Some(GeminiBilling {
        separate,
        tool_use_billing,
    }) = gemini
    else {
        return Ok(BilledUsage {
            usage: Cow::Borrowed(usage),
            reported: usage,
        });
    };
    let input_rate_tool_use_prompt_tokens =
        match (tool_use_billing, separate.tool_use_prompt_tokens) {
            (_, None | Some(0)) | (ToolUsePromptBilling::GroundingRequests, _) => None,
            (ToolUsePromptBilling::InputTokens, count) => count,
            (ToolUsePromptBilling::Undetermined, Some(_)) => {
                return Err(UnbillableUsage::UndeterminedToolUsePromptTokens);
            }
        };
    let (prompt_tokens, uncached_input_tokens) = match usage.uncached_input_tokens {
        Some(uncached) => (
            usage
                .prompt_tokens
                .map(|prompt| add_to_count(prompt, input_rate_tool_use_prompt_tokens))
                .transpose()?,
            Some(add_to_count(uncached, input_rate_tool_use_prompt_tokens)?),
        ),
        None => (
            add_separate_count(usage.prompt_tokens, input_rate_tool_use_prompt_tokens)?,
            None,
        ),
    };
    let completion_tokens = add_separate_count(usage.completion_tokens, separate.thoughts_tokens)?;
    Ok(BilledUsage {
        usage: Cow::Owned(Usage {
            prompt_tokens,
            completion_tokens,
            uncached_input_tokens,
            ..usage.clone()
        }),
        reported: usage,
    })
}

fn add_separate_count(
    reported: Option<u64>,
    separate: Option<u64>,
) -> Result<Option<u64>, UnbillableUsage> {
    match reported {
        Some(reported) => add_to_count(reported, separate).map(Some),
        None => Ok(separate),
    }
}

fn add_to_count(reported: u64, separate: Option<u64>) -> Result<u64, UnbillableUsage> {
    separate
        .map_or(Some(reported), |separate| reported.checked_add(separate))
        .ok_or(UnbillableUsage::TokenOverflow)
}
