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

/// Error produced while mapping the protobuf tool-result contract to shared
/// Relay application types.
#[derive(Debug, thiserror::Error)]
pub enum ToolExecutionContractError {
    /// A semantically required protobuf message field was absent.
    #[error("{0} is missing")]
    MissingField(&'static str),
    /// A JSON-bearing field could not be encoded or decoded.
    #[error("invalid JSON in {field}: {source}")]
    InvalidJson {
        /// Fully qualified logical field name.
        field: &'static str,
        /// JSON codec failure, including invalid UTF-8 when decoding bytes.
        #[source]
        source: serde_json::Error,
    },
}

/// Encodes a shared tool execution result into its structural protobuf form.
///
/// JSON null annotations are normalized to protobuf absence.
///
/// # Errors
/// Returns a contextual JSON serialization error.
pub fn encode_tool_execution_result(
    value: &nemo_relay_types::api::tool::ToolExecutionResult,
) -> Result<v1::ToolExecutionResult, ToolExecutionContractError> {
    Ok(v1::ToolExecutionResult {
        result: Some(encode_json_field(
            "tool execution result.result",
            &value.result,
        )?),
        annotation: encode_optional_json_field(
            "tool execution result.annotation",
            value.annotation.as_ref().filter(|value| !value.is_null()),
        )?,
    })
}

/// Decodes a structural protobuf tool execution result into the shared type.
///
/// JSON null annotations are normalized to absence.
///
/// # Errors
/// Returns an error for a missing result or invalid JSON bytes.
pub fn decode_tool_execution_result(
    value: v1::ToolExecutionResult,
) -> Result<nemo_relay_types::api::tool::ToolExecutionResult, ToolExecutionContractError> {
    let result = decode_required_json_field("tool execution result.result", value.result.as_ref())?;
    let annotation: Option<nemo_relay_types::Json> = decode_optional_json_field(
        "tool execution result.annotation",
        value.annotation.as_ref(),
    )?;
    let annotation = annotation.filter(|value| !value.is_null());
    Ok(nemo_relay_types::api::tool::ToolExecutionResult { result, annotation })
}

/// Encodes a shared tool execution intercept outcome into protobuf.
///
/// JSON null annotations are normalized to protobuf absence.
///
/// # Errors
/// Returns a contextual JSON serialization error.
pub fn encode_tool_execution_intercept_outcome(
    value: &nemo_relay_types::api::tool::ToolExecutionInterceptOutcome,
) -> Result<v1::ToolExecutionInterceptOutcome, ToolExecutionContractError> {
    Ok(v1::ToolExecutionInterceptOutcome {
        result: Some(encode_json_field(
            "tool execution intercept outcome.result",
            &value.result,
        )?),
        annotation: encode_optional_json_field(
            "tool execution intercept outcome.annotation",
            value.annotation.as_ref().filter(|value| !value.is_null()),
        )?,
        pending_marks: value
            .pending_marks
            .iter()
            .map(encode_pending_mark)
            .collect::<Result<_, _>>()?,
    })
}

/// Decodes a structural protobuf tool execution intercept outcome into the
/// shared type.
///
/// JSON null annotations are normalized to absence.
///
/// # Errors
/// Returns an error for a missing result or invalid JSON bytes.
pub fn decode_tool_execution_intercept_outcome(
    value: v1::ToolExecutionInterceptOutcome,
) -> Result<nemo_relay_types::api::tool::ToolExecutionInterceptOutcome, ToolExecutionContractError>
{
    let result = decode_required_json_field(
        "tool execution intercept outcome.result",
        value.result.as_ref(),
    )?;
    let annotation: Option<nemo_relay_types::Json> = decode_optional_json_field(
        "tool execution intercept outcome.annotation",
        value.annotation.as_ref(),
    )?;
    let annotation = annotation.filter(|value| !value.is_null());
    let pending_marks = value
        .pending_marks
        .into_iter()
        .map(decode_pending_mark)
        .collect::<Result<_, _>>()?;
    Ok(nemo_relay_types::api::tool::ToolExecutionInterceptOutcome {
        result,
        annotation,
        pending_marks,
    })
}

fn encode_pending_mark(
    value: &nemo_relay_types::api::event::PendingMarkSpec,
) -> Result<v1::PendingMarkSpec, ToolExecutionContractError> {
    Ok(v1::PendingMarkSpec {
        name: value.name.clone(),
        category: value
            .category
            .as_ref()
            .map(|category| category.as_str().to_owned()),
        category_profile: encode_optional_json_field(
            "tool execution intercept outcome.pending_marks.category_profile",
            value.category_profile.as_ref(),
        )?,
        data: encode_optional_json_field(
            "tool execution intercept outcome.pending_marks.data",
            value.data.as_ref(),
        )?,
        metadata: encode_optional_json_field(
            "tool execution intercept outcome.pending_marks.metadata",
            value.metadata.as_ref(),
        )?,
    })
}

fn decode_pending_mark(
    value: v1::PendingMarkSpec,
) -> Result<nemo_relay_types::api::event::PendingMarkSpec, ToolExecutionContractError> {
    Ok(nemo_relay_types::api::event::PendingMarkSpec {
        name: value.name,
        category: value
            .category
            .map(nemo_relay_types::api::event::EventCategory::new),
        category_profile: decode_optional_json_field(
            "tool execution intercept outcome.pending_marks.category_profile",
            value.category_profile.as_ref(),
        )?,
        data: decode_optional_json_field(
            "tool execution intercept outcome.pending_marks.data",
            value.data.as_ref(),
        )?,
        metadata: decode_optional_json_field(
            "tool execution intercept outcome.pending_marks.metadata",
            value.metadata.as_ref(),
        )?,
    })
}

fn encode_json_field<T: serde::Serialize>(
    field: &'static str,
    value: &T,
) -> Result<v1::JsonValue, ToolExecutionContractError> {
    json_value(value).map_err(|source| ToolExecutionContractError::InvalidJson { field, source })
}

fn encode_optional_json_field<T: serde::Serialize>(
    field: &'static str,
    value: Option<&T>,
) -> Result<Option<v1::JsonValue>, ToolExecutionContractError> {
    value
        .map(|value| encode_json_field(field, value))
        .transpose()
}

fn decode_required_json_field<T: serde::de::DeserializeOwned>(
    field: &'static str,
    value: Option<&v1::JsonValue>,
) -> Result<T, ToolExecutionContractError> {
    let value = value.ok_or(ToolExecutionContractError::MissingField(field))?;
    decode_json_field(field, value)
}

fn decode_optional_json_field<T: serde::de::DeserializeOwned>(
    field: &'static str,
    value: Option<&v1::JsonValue>,
) -> Result<Option<T>, ToolExecutionContractError> {
    value
        .map(|value| decode_json_field(field, value))
        .transpose()
}

fn decode_json_field<T: serde::de::DeserializeOwned>(
    field: &'static str,
    value: &v1::JsonValue,
) -> Result<T, ToolExecutionContractError> {
    decode_json_value(value)
        .map_err(|source| ToolExecutionContractError::InvalidJson { field, source })
}
