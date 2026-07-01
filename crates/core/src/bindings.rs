// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared data conversions used by language bindings.

/// JavaScript-specific data transfer objects shared by Node.js and WebAssembly.
pub mod js {
    use serde::{Deserialize, Serialize};
    use serde_json::Value as Json;

    use crate::api::event::{CategoryProfile, EventCategory, PendingMarkSpec};

    /// JavaScript-facing pending mark DTO.
    ///
    /// JavaScript bindings use camelCase while canonical Relay JSON keeps
    /// snake_case field names.
    #[derive(Debug, Deserialize, Serialize)]
    #[serde(rename_all = "camelCase", deny_unknown_fields)]
    pub struct JsPendingMarkSpec {
        name: String,
        #[serde(default)]
        category: Option<EventCategory>,
        #[serde(default)]
        category_profile: Option<CategoryProfile>,
        #[serde(default)]
        data: Option<Json>,
        #[serde(default)]
        metadata: Option<Json>,
    }

    impl From<JsPendingMarkSpec> for PendingMarkSpec {
        fn from(mark: JsPendingMarkSpec) -> Self {
            Self {
                name: mark.name,
                category: mark.category,
                category_profile: mark.category_profile,
                data: mark.data,
                metadata: mark.metadata,
            }
        }
    }

    impl From<PendingMarkSpec> for JsPendingMarkSpec {
        fn from(mark: PendingMarkSpec) -> Self {
            Self {
                name: mark.name,
                category: mark.category,
                category_profile: mark.category_profile,
                data: mark.data,
                metadata: mark.metadata,
            }
        }
    }

    /// Convert canonical pending marks to JavaScript-facing DTOs.
    #[must_use]
    pub fn js_pending_marks(marks: Vec<PendingMarkSpec>) -> Vec<JsPendingMarkSpec> {
        marks.into_iter().map(Into::into).collect()
    }
}
