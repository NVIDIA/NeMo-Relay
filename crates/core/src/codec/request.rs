// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Re-exported normalized LLM request data types.

pub use nemo_relay_types::codec::request::*;

#[cfg(test)]
#[path = "../../tests/unit/codec/request_tests.rs"]
mod tests;
