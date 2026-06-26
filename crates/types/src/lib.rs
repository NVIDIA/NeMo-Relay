// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![deny(rustdoc::broken_intra_doc_links, rustdoc::private_intra_doc_links)]

//! Shared serializable data model types for NeMo Relay.
//!
//! This crate contains DTOs and wire-compatible type definitions used by the
//! runtime and native plugin SDK. It intentionally does not contain runtime
//! registries, dynamic loading, codecs, exporters, or process-global state.

/// Public runtime API data types.
pub mod api;
/// Normalized LLM request and response data types.
pub mod codec;
/// Plugin configuration diagnostic data types.
pub mod plugin;

/// Type alias for [`serde_json::Value`], used as the universal JSON
/// representation throughout NeMo Relay APIs.
pub type Json = serde_json::Value;
