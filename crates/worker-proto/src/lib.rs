// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![deny(rustdoc::broken_intra_doc_links, rustdoc::private_intra_doc_links)]

//! Versioned gRPC protocol for NeMo Relay out-of-process worker plugins.
//!
//! The protobuf schema owns transport control flow and the tool-result wrapper
//! structure. Open application payloads remain lossless JSON values, while
//! other Relay data transfer objects are carried in JSON envelopes.

mod message_limits;
pub use message_limits::{
    CallbackMessageLimitService, WORKER_CALLBACK_METADATA, WORKER_MAX_MESSAGE_BYTES_BY_NAME_ENV,
    WorkerCallbackMessageLimits, WorkerCallbackMetadata, callback_name, callback_request,
    validate_callback_request,
};

/// Stable worker protocol identifier accepted by `compat.worker_protocol`.
pub const WORKER_PROTOCOL_GRPC_V1: &str = "grpc-v1";

/// Host-owned limit for a complete worker execution gRPC message, in bytes.
pub const WORKER_MAX_MESSAGE_BYTES_ENV: &str = "NEMO_RELAY_WORKER_MAX_MESSAGE_BYTES";

/// Preserve the gRPC libraries' default receive limit unless explicitly configured.
pub const DEFAULT_WORKER_MAX_MESSAGE_BYTES: usize = 4 * 1024 * 1024;

/// Reads the worker message limit once when creating a host activation or worker.
///
/// # Errors
/// Returns an error for a non-Unicode, empty, non-decimal, zero, or oversized value.
pub fn worker_max_message_bytes_from_env() -> Result<usize, String> {
    worker_max_message_bytes_from_env_with_default(DEFAULT_WORKER_MAX_MESSAGE_BYTES)
}

/// Reads the host override, falling back to a plugin's execution-message default.
///
/// The host resolves this once per activation and passes the effective value to
/// its worker. SDKs use [`worker_max_message_bytes_from_env`] to read that value.
///
/// # Errors
/// Returns an error for an invalid override or an out-of-range fallback.
pub fn worker_max_message_bytes_from_env_with_default(default: usize) -> Result<usize, String> {
    match std::env::var(WORKER_MAX_MESSAGE_BYTES_ENV) {
        Ok(value) => parse_worker_max_message_bytes(&value),
        Err(std::env::VarError::NotPresent) => parse_worker_max_message_bytes(&default.to_string()),
        Err(std::env::VarError::NotUnicode(_)) => Err(worker_message_limit_error()),
    }
}

/// Parses the positive decimal byte limit shared by Rust and Python workers.
///
/// The signed 32-bit ceiling matches Python gRPC's channel option representation.
///
/// # Errors
/// Returns an error unless the value is an ASCII decimal integer in `1..=i32::MAX`.
pub fn parse_worker_max_message_bytes(value: &str) -> Result<usize, String> {
    let limit = value.parse::<usize>().ok();
    if value.bytes().all(|byte| byte.is_ascii_digit())
        && let Some(limit) = limit
        && (1..=i32::MAX as usize).contains(&limit)
    {
        return Ok(limit);
    }
    Err(worker_message_limit_error())
}

/// Describes the supported byte range without echoing the supplied environment value.
fn worker_message_limit_error() -> String {
    format!("{WORKER_MAX_MESSAGE_BYTES_ENV} must be a decimal integer in 1..=2147483647")
}

/// Metadata set to `1` on a diagnostic admission rejection before execution.
/// This certifies only that attempt; an earlier lost reply can still be uncertain.
pub const DIAGNOSTIC_NOT_CONSUMED_METADATA: &str = "nemo-relay-diagnostic-not-consumed";

/// Continuation provenance for an error returned as gRPC status metadata.
/// Used for stream-opening failures, where no `WorkerError` envelope is returned.
pub const DOWNSTREAM_CONTINUATION_METADATA: &str = "nemo-relay-downstream-continuation";

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
