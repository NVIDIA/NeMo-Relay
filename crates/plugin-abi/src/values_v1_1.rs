// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Owned record transfer for the additive v1.1 tool lane.
use crate::{Json, transfer::record, v1_0 as a, v1_1::*};

record!(ToolInvocationWire, ToolInvocation {invocation:a::Invocation,execution_context:Option<Json>});
record!(
    ToolRegistrationWire,
    ToolRegistration {
        name: String,
        priority: i32,
        callback: ToolCallbackBox
    }
);
