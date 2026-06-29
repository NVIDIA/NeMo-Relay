// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared tool data types.

use bitflags::bitflags;
use serde::{Deserialize, Serialize};

bitflags! {
    /// Bitflags that modify tool-call behavior and observability.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct ToolAttributes: u32 {
        /// Marks the tool as executing out-of-process.
        const REMOTE = 0b01;
    }
}
