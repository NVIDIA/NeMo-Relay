// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![deny(rustdoc::broken_intra_doc_links, rustdoc::private_intra_doc_links)]

//! Versioned gRPC protocol for NeMo Relay out-of-process worker plugins.
//!
//! The protobuf schema owns transport control flow and the tool-result wrapper
//! structure. Open application payloads remain lossless JSON values, while
//! other Relay data transfer objects are carried in JSON envelopes.

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

/// Creates a lossless protocol JSON value from a serializable value.
///
/// # Errors
/// Returns a serde error when the supplied value cannot be serialized as JSON.
pub fn json_value<T: serde::Serialize>(value: &T) -> Result<v1::JsonValue, serde_json::Error> {
    Ok(v1::JsonValue {
        json: serde_json::to_vec(value)?,
    })
}

/// Decodes a lossless protocol JSON value into the requested type.
///
/// # Errors
/// Returns a serde error when the bytes are not valid JSON for `T`.
pub fn decode_json_value<T: serde::de::DeserializeOwned>(
    value: &v1::JsonValue,
) -> Result<T, serde_json::Error> {
    serde_json::from_slice(&value.json)
}
