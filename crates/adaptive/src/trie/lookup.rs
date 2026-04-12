// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prediction trie lookup with three-level fallback chain.

use super::data_models::{LlmCallPrediction, PredictionTrieNode};

/// Looks up predictions in a prediction trie with graceful fallback.
///
/// Fallback chain:
/// 1. Exact path + exact call_index
/// 2. Exact path + predictions_any_index
/// 3. Deepest ancestor + exact call_index
/// 4. Deepest ancestor + predictions_any_index
/// 5. ... continue to root ...
/// 6. None (no predictions available)
pub struct PredictionTrieLookup<'a> {
    root: &'a PredictionTrieNode,
}

impl<'a> PredictionTrieLookup<'a> {
    /// Create a new lookup over the given trie root.
    pub fn new(root: &'a PredictionTrieNode) -> Self {
        Self { root }
    }

    /// Find the best matching prediction for the given path and call index.
    ///
    /// Walks the trie following `path` elements, tracking the deepest node that
    /// has a prediction. At each node, prefers `predictions_by_call_index[call_index]`
    /// over `predictions_any_index`.
    pub fn find(&self, path: &[String], call_index: u32) -> Option<&'a LlmCallPrediction> {
        let mut node = self.root;
        let mut deepest_match: Option<&LlmCallPrediction> = None;

        // Check root node first
        if let Some(pred) = Self::get_prediction(node, call_index) {
            deepest_match = Some(pred);
        }

        // Walk the trie as far as path matches
        for func_name in path {
            match node.children.get(func_name.as_str()) {
                Some(child) => {
                    node = child;
                    if let Some(pred) = Self::get_prediction(node, call_index) {
                        deepest_match = Some(pred);
                    }
                }
                None => break,
            }
        }

        deepest_match
    }

    /// At a given node, prefer exact call_index match over any_index fallback.
    fn get_prediction(node: &PredictionTrieNode, call_index: u32) -> Option<&LlmCallPrediction> {
        node.predictions_by_call_index
            .get(&call_index)
            .or(node.predictions_any_index.as_ref())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/trie/lookup_tests.rs"]
mod tests;
