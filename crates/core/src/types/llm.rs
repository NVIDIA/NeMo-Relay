// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct LLMAttributes: u32 {
        const STATELESS = 0b01;
        const STREAMING = 0b10;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMHandle {
    pub uuid: Uuid,
    pub name: String,
    pub data: Option<crate::json::Json>,
    pub metadata: Option<crate::json::Json>,
    pub attributes: LLMAttributes,
    pub parent_uuid: Option<Uuid>,
    pub model_name: Option<String>,
}

impl LLMHandle {
    pub fn new(
        name: String,
        attributes: LLMAttributes,
        parent_uuid: Option<Uuid>,
        data: Option<crate::json::Json>,
        metadata: Option<crate::json::Json>,
    ) -> Self {
        Self {
            uuid: Uuid::now_v7(),
            name,
            data,
            metadata,
            attributes,
            parent_uuid,
            model_name: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMRequest {
    pub headers: serde_json::Map<String, crate::json::Json>,
    pub content: crate::json::Json,
}
