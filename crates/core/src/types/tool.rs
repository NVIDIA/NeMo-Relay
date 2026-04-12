// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct ToolAttributes: u32 {
        const LOCAL = 0b01;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolHandle {
    pub uuid: Uuid,
    pub name: String,
    pub data: Option<crate::json::Json>,
    pub metadata: Option<crate::json::Json>,
    pub attributes: ToolAttributes,
    pub parent_uuid: Option<Uuid>,
    pub tool_call_id: Option<String>,
}

impl ToolHandle {
    pub fn new(
        name: String,
        attributes: ToolAttributes,
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
            tool_call_id: None,
        }
    }
}
