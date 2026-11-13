// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Variable content detection and extraction from prompt blocks.

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::acg::prompt_ir::{PromptBlock, SpanId};

/// Category assigned to an extracted variable.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VariableCategory {
    /// Timestamp-like content such as ISO 8601 strings.
    Timestamp,
    /// Request or trace identifier content.
    RequestId,
    /// Session identifier content.
    SessionId,
    /// Locale identifier content.
    Locale,
    /// Caller-defined variable category.
    Custom(String),
}

/// Regex-based pattern used to detect variable content.
pub struct VariablePattern {
    /// Stable placeholder name inserted into the template.
    pub name: String,
    /// Regex used to detect matching content.
    pub regex: Regex,
    /// Semantic category assigned to matches from this pattern.
    pub category: VariableCategory,
}

impl std::fmt::Debug for VariablePattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VariablePattern")
            .field("name", &self.name)
            .field("regex", &self.regex.as_str())
            .field("category", &self.category)
            .finish()
    }
}

/// One extracted variable occurrence within a prompt block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractedVariable {
    /// Name of the pattern that matched this value.
    pub pattern_name: String,
    /// Original matched value before replacement.
    pub original_value: String,
    /// Byte offset of the match in the original content.
    pub byte_offset: usize,
    /// Byte length of the original match.
    pub byte_length: usize,
    /// Semantic category assigned to the variable.
    pub category: VariableCategory,
}

/// Variable extraction result for one prompt block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtractionResult {
    /// Span identifier of the analyzed block.
    pub span_id: SpanId,
    /// Block content with extracted values replaced by placeholders.
    pub template_content: String,
    /// Variables extracted from the block.
    pub variables: Vec<ExtractedVariable>,
}

/// Return the default regex patterns used for variable extraction.
///
/// # Returns
/// A vector of built-in [`VariablePattern`] values covering common timestamps
/// and request identifiers.
pub fn default_variable_patterns() -> Vec<VariablePattern> {
    vec![
        VariablePattern {
            name: "iso8601_timestamp".to_string(),
            regex: Regex::new(
                r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})",
            )
            .expect("iso8601_timestamp regex is valid"),
            category: VariableCategory::Timestamp,
        },
        VariablePattern {
            name: "uuid".to_string(),
            regex: Regex::new(
                r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
            )
            .expect("uuid regex is valid"),
            category: VariableCategory::RequestId,
        },
        VariablePattern {
            name: "request_id".to_string(),
            regex: Regex::new(r"(?:req|trace|span|run|call|session)[-_][a-zA-Z0-9]{8,}")
                .expect("request_id regex is valid"),
            category: VariableCategory::RequestId,
        },
        VariablePattern {
            name: "date_string".to_string(),
            regex: Regex::new(r"\d{4}-\d{2}-\d{2}").expect("date_string regex is valid"),
            category: VariableCategory::Timestamp,
        },
        VariablePattern {
            name: "unix_timestamp".to_string(),
            regex: Regex::new(r"\b1[0-9]{9,12}\b").expect("unix_timestamp regex is valid"),
            category: VariableCategory::Timestamp,
        },
    ]
}

/// Extract variables from one content string.
///
/// Matching patterns are applied greedily by start position, preferring longer
/// matches when multiple patterns overlap.
///
/// # Parameters
/// - `content`: Block content to analyze.
/// - `span_id`: Span identifier associated with the content.
/// - `patterns`: Variable patterns to evaluate.
///
/// # Returns
/// `Some(ExtractionResult)` when at least one variable is found and `None`
/// otherwise.
pub fn extract_variables(
    content: &str,
    span_id: &SpanId,
    patterns: &[VariablePattern],
) -> Option<ExtractionResult> {
    let mut all_matches: Vec<(usize, usize, &VariablePattern)> = Vec::new();
    for pattern in patterns {
        for matched in pattern.regex.find_iter(content) {
            all_matches.push((matched.start(), matched.end(), pattern));
        }
    }

    if all_matches.is_empty() {
        return None;
    }

    all_matches.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| (right.1 - right.0).cmp(&(left.1 - left.0)))
    });

    let mut selected: Vec<(usize, usize, &VariablePattern)> = Vec::new();
    for candidate in all_matches {
        let overlaps = selected
            .iter()
            .any(|selected_match| candidate.0 < selected_match.1 && candidate.1 > selected_match.0);
        if !overlaps {
            selected.push(candidate);
        }
    }

    selected.sort_by_key(|selected_match| selected_match.0);

    let mut template = String::with_capacity(content.len());
    let mut variables: Vec<ExtractedVariable> = Vec::new();
    let mut last_end = 0;

    for (start, end, pattern) in &selected {
        template.push_str(&content[last_end..*start]);
        template.push_str(&format!("{{{{{}}}}}", pattern.name));
        variables.push(ExtractedVariable {
            pattern_name: pattern.name.clone(),
            original_value: content[*start..*end].to_string(),
            byte_offset: *start,
            byte_length: end - start,
            category: pattern.category.clone(),
        });
        last_end = *end;
    }

    template.push_str(&content[last_end..]);

    Some(ExtractionResult {
        span_id: span_id.clone(),
        template_content: template,
        variables,
    })
}

/// Extract variables from a set of prompt blocks.
///
/// # Parameters
/// - `blocks`: Prompt blocks to analyze.
/// - `patterns`: Variable patterns to evaluate.
///
/// # Returns
/// One [`ExtractionResult`] per block that contained at least one variable.
pub fn extract_variables_from_blocks(
    blocks: &[PromptBlock],
    patterns: &[VariablePattern],
) -> Vec<ExtractionResult> {
    blocks
        .iter()
        .filter(|block| {
            !matches!(
                block.content_type,
                crate::acg::prompt_ir::BlockContentType::Opaque
                    | crate::acg::prompt_ir::BlockContentType::Image
            )
        })
        .filter_map(|block| extract_variables(&block.content, &block.span_id, patterns))
        .collect()
}

#[cfg(test)]
#[path = "../../tests/unit/acg/variable_extractor_tests.rs"]
mod tests;
