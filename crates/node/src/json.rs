// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use napi::bindgen_prelude::{FromNapiValue, ToNapiValue, TypeName, ValidateNapiValue};
use napi::{Env, JsObject, JsUnknown, NapiRaw, NapiValue, Status, ValueType, sys};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Number, Value};

pub(crate) const JAVASCRIPT_MAX_SAFE_INTEGER: u64 = (1 << 53) - 1;

const MAX_JSON_DEPTH: u32 = 128;

const MAX_JSON_NODES: usize = 1_000_000;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Json(pub Value);

impl From<Value> for Json {
    fn from(value: Value) -> Self {
        Self(value)
    }
}

impl From<Json> for Value {
    fn from(json: Json) -> Self {
        json.0
    }
}

impl TypeName for Json {
    fn type_name() -> &'static str {
        "Json"
    }

    fn value_type() -> ValueType {
        ValueType::Unknown
    }
}

impl ValidateNapiValue for Json {}

impl ToNapiValue for Json {
    unsafe fn to_napi_value(env: sys::napi_env, value: Self) -> napi::Result<sys::napi_value> {
        // SAFETY: `env` is live for this call.
        unsafe { Value::to_napi_value(env, value.0) }
    }
}

impl FromNapiValue for Json {
    unsafe fn from_napi_value(env: sys::napi_env, value: sys::napi_value) -> napi::Result<Self> {
        // SAFETY: `env` and `value` are live handles of this call.
        let value = unsafe { JsUnknown::from_raw_unchecked(env, value) };
        // SAFETY: `env` is live for this call.
        let env = unsafe { Env::from_raw(env) };
        let mut remaining_nodes = MAX_JSON_NODES;
        json_from_js(&env, value, MAX_JSON_DEPTH, &mut remaining_nodes).map(Self)
    }
}

pub(crate) fn json_number(value: f64) -> Option<Number> {
    if value.fract() == 0.0 && value.abs() <= JAVASCRIPT_MAX_SAFE_INTEGER as f64 {
        Some(Number::from(value as i64))
    } else {
        Number::from_f64(value)
    }
}

fn invalid_argument(reason: &str) -> napi::Error {
    napi::Error::new(Status::InvalidArg, reason.to_owned())
}

fn json_from_js(
    env: &Env,
    value: JsUnknown,
    remaining_depth: u32,
    remaining_nodes: &mut usize,
) -> napi::Result<Value> {
    *remaining_nodes = remaining_nodes
        .checked_sub(1)
        .ok_or_else(|| invalid_argument("exceeded maximum JSON value count"))?;
    match value.get_type()? {
        ValueType::Null => Ok(Value::Null),
        // SAFETY: `value` was just typed as a boolean in `env`.
        ValueType::Boolean => Ok(Value::Bool(unsafe {
            bool::from_napi_value(env.raw(), value.raw())?
        })),
        ValueType::Number => {
            // SAFETY: `value` was just typed as a number in `env`.
            let number = unsafe { f64::from_napi_value(env.raw(), value.raw())? };
            json_number(number).map(Value::Number).ok_or_else(|| {
                invalid_argument("JavaScript number must be finite to be represented as JSON")
            })
        }
        // SAFETY: `value` was just typed as a string in `env`.
        ValueType::String => Ok(Value::String(unsafe {
            String::from_napi_value(env.raw(), value.raw())?
        })),
        ValueType::Object => {
            let remaining_depth = remaining_depth
                .checked_sub(1)
                .ok_or_else(|| invalid_argument("exceeded maximum JSON nesting depth"))?;
            // SAFETY: `value` was just typed as an object in `env`.
            let object = unsafe { value.cast::<JsObject>() };
            json_from_object(env, object, remaining_depth, remaining_nodes)
        }
        ValueType::BigInt => {
            env.throw_type_error("BigInt cannot be represented as JSON", None)?;
            Err(napi::Error::new(
                Status::PendingException,
                "BigInt cannot be represented as JSON".to_owned(),
            ))
        }
        ValueType::Undefined => Err(invalid_argument("undefined cannot be represented as JSON")),
        ValueType::Function => Err(invalid_argument("functions cannot be represented as JSON")),
        ValueType::Symbol => Err(invalid_argument("symbols cannot be represented as JSON")),
        ValueType::External => Err(invalid_argument(
            "external values cannot be represented as JSON",
        )),
        ValueType::Unknown => Err(invalid_argument(
            "unknown values cannot be represented as JSON",
        )),
    }
}

fn json_from_object(
    env: &Env,
    object: JsObject,
    remaining_depth: u32,
    remaining_nodes: &mut usize,
) -> napi::Result<Value> {
    if object.is_array()? {
        let length = object.get_array_length()?;
        let mut values = Vec::with_capacity((length as usize).min(*remaining_nodes));
        for index in 0..length {
            values.push(json_from_js(
                env,
                object.get_element(index)?,
                remaining_depth,
                remaining_nodes,
            )?);
        }
        return Ok(Value::Array(values));
    }
    let keys = JsObject::keys(&object)?;
    let mut values = Map::with_capacity(keys.len());
    for key in keys {
        if let Some(value) = object.get::<_, JsUnknown>(&key)? {
            values.insert(
                key,
                json_from_js(env, value, remaining_depth, remaining_nodes)?,
            );
        }
    }
    Ok(Value::Object(values))
}

#[cfg(test)]
#[path = "../tests/rust/json_tests.rs"]
mod tests;
