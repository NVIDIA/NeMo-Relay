// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![deny(rustdoc::broken_intra_doc_links, rustdoc::private_intra_doc_links)]

//! Versioned gRPC protocol for NeMo Relay out-of-process worker plugins.
//!
//! The protobuf schema owns transport control flow. Relay data transfer objects
//! are carried as JSON envelopes so `nemo-relay-types` remains the shared source
//! of truth for event, LLM, tool, scope, and plugin diagnostic data shapes.

/// Stable worker protocol identifier accepted by `compat.worker_protocol`.
pub const WORKER_PROTOCOL_GRPC_V1: &str = "grpc-v1";

/// Generated protobuf and gRPC service definitions.
#[allow(missing_docs)]
pub mod v1 {
    tonic::include_proto!("nemo.relay.worker.v1");
}

/// Creates a JSON envelope from a serializable DTO.
///
/// # Errors
/// Returns a serde error when the supplied value cannot be serialized as JSON.
pub fn json_envelope<T: serde::Serialize>(
    schema: impl Into<String>,
    value: &T,
) -> Result<v1::JsonEnvelope, serde_json::Error> {
    Ok(v1::JsonEnvelope {
        schema: schema.into(),
        json: serde_json::to_vec(value)?,
    })
}

/// Decodes a JSON envelope into the requested DTO type.
///
/// # Errors
/// Returns a serde error when the envelope bytes are not valid JSON for `T`.
pub fn decode_json_envelope<T: serde::de::DeserializeOwned>(
    envelope: &v1::JsonEnvelope,
) -> Result<T, serde_json::Error> {
    serde_json::from_slice(&envelope.json)
}
