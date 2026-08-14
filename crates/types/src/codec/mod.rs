// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared normalized LLM request and response data types.

/// Stable LLM codec identities shared across SDK boundaries.
pub mod identity;
/// Plugin-neutral LLM optimization evidence and summaries.
pub mod optimization;
/// Normalized LLM request data types.
pub mod request;
/// Normalized LLM response data types.
pub mod response;
