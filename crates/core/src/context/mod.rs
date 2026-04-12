// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Global context, scope stack, and middleware chain execution.

pub mod callbacks;
pub mod global;
pub mod registries;
pub mod scope_stack;
pub mod state;

#[cfg(test)]
#[path = "../../tests/unit/context_tests.rs"]
mod tests;
