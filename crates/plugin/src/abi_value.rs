// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed JSON conversion without encoding or parsing JSON text.

use crate::Json;
use crate::abi::{Field, Json as AbiJson};

/// Move an SDK value into an owned ABI-safe tree.
pub fn into_abi(value: Json) -> AbiJson {
    match value {
        Json::Null => AbiJson::Null,
        Json::Bool(v) => AbiJson::Bool(v),
        Json::Number(v) if v.is_i64() => AbiJson::Signed(v.as_i64().expect("signed integer")),
        Json::Number(v) if v.is_u64() => AbiJson::Unsigned(v.as_u64().expect("unsigned integer")),
        Json::Number(v) => AbiJson::Float(v.as_f64().expect("finite JSON number")),
        Json::String(v) => AbiJson::String(v),
        Json::Array(v) => AbiJson::Array(v.into_iter().map(into_abi).collect()),
        Json::Object(v) => AbiJson::Object(
            v.into_iter()
                .map(|(key, value)| Field {
                    key,
                    value: into_abi(value),
                })
                .collect(),
        ),
    }
}

/// Copy an SDK value into an ABI-safe tree without first cloning an SDK tree.
pub fn to_abi(value: &Json) -> AbiJson {
    match value {
        Json::Null => AbiJson::Null,
        Json::Bool(v) => AbiJson::Bool(*v),
        Json::Number(v) if v.is_i64() => AbiJson::Signed(v.as_i64().expect("signed integer")),
        Json::Number(v) if v.is_u64() => AbiJson::Unsigned(v.as_u64().expect("unsigned integer")),
        Json::Number(v) => AbiJson::Float(v.as_f64().expect("finite JSON number")),
        Json::String(v) => AbiJson::String(v.as_str().into()),
        Json::Array(v) => AbiJson::Array(v.iter().map(to_abi).collect()),
        Json::Object(v) => AbiJson::Object(
            v.iter()
                .map(|(key, value)| Field {
                    key: key.as_str().into(),
                    value: to_abi(value),
                })
                .collect(),
        ),
    }
}

/// Recover an SDK value, rejecting values outside JSON's representation.
pub fn from_abi(value: AbiJson) -> crate::Result<Json> {
    Ok(match value {
        AbiJson::Null => Json::Null,
        AbiJson::Bool(v) => Json::Bool(v),
        AbiJson::Signed(v) => Json::Number(v.into()),
        AbiJson::Unsigned(v) => Json::Number(v.into()),
        AbiJson::Float(v) => Json::Number(
            serde_json::Number::from_f64(v).ok_or_else(|| "non-finite JSON number".to_owned())?,
        ),
        AbiJson::String(v) => Json::String(v),
        AbiJson::Array(v) => {
            Json::Array(v.into_iter().map(from_abi).collect::<crate::Result<_>>()?)
        }
        AbiJson::Object(v) => {
            let mut fields = serde_json::Map::new();
            for field in v {
                if fields.insert(field.key, from_abi(field.value)?).is_some() {
                    return Err("duplicate JSON object key".into());
                }
            }
            Json::Object(fields)
        }
    })
}
