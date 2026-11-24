// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use napi::bindgen_prelude::{
    FromNapiValue, JsObjectValue, JsValue, KeyCollectionMode, KeyConversion, KeyFilter, Object,
    Unknown,
};
use napi::{Env, Property, ValueType};
use nemo_relay_types::api::llm::ApplicationExecutionContext;
use serde_json::Value;

use crate::convert::{JsArg, detach, js_value};
use crate::error::{Error, Result};
use crate::json::Json;

const MAXIMUM_CONTEXT_CONTAINER_DEPTH: u32 = 64;
const MAXIMUM_CONTEXT_VALUE_COUNT: usize = 16_384;
const MAXIMUM_CONTEXT_STRING_BYTES: usize = 16_384;

/// Copy context from its own enumerable JSON properties.
/// Ignore inherited properties. Read getter values as JSON input.
/// This function does not use JavaScript globals.
pub(crate) fn execution_context_json(
    value: Option<Unknown>,
) -> Result<Option<ApplicationExecutionContext>> {
    let Some(value) = value else { return Ok(None) };
    if matches!(value.get_type()?, ValueType::Null | ValueType::Undefined) {
        return Ok(None);
    }
    let mut remaining_nodes = MAXIMUM_CONTEXT_VALUE_COUNT;
    let mut remaining_string_bytes = MAXIMUM_CONTEXT_STRING_BYTES;
    let value = own_context_json(
        value,
        MAXIMUM_CONTEXT_CONTAINER_DEPTH,
        &mut remaining_nodes,
        &mut remaining_string_bytes,
    )?;
    if !value.is_object() {
        return Err(Error::invalid_argument(
            "execution context must be a JSON object",
        ));
    }
    ApplicationExecutionContext::try_from(value)
        .map(Some)
        .map_err(|error| Error::invalid_argument(error.to_string()))
}

/// Convert a validated context using own data properties, including `__proto__`.
/// The context has already passed depth and size checks.
pub(crate) fn context_json_to_js(env: &Env, value: &Value) -> napi::Result<JsArg> {
    match value {
        Value::Object(fields) => {
            let mut object = Object::new(env)?;
            for (key, value) in fields {
                let property = Property::default()
                    .with_name(env, key.as_str())?
                    .with_napi_value(env, context_json_to_js(env, value)?)?;
                object.define_properties(&[property])?;
            }
            Ok(detach(env, &object))
        }
        Value::Array(values) => {
            let mut array = env.create_array(values.len() as u32)?;
            for (index, value) in values.iter().enumerate() {
                array.set(index as u32, context_json_to_js(env, value)?)?;
            }
            Ok(detach(env, &array))
        }
        _ => js_value(env, value.clone()),
    }
}

fn context_string(value: Unknown, remaining_string_bytes: &mut usize) -> Result<String> {
    let string = value.coerce_to_string()?;
    // UTF-16 length is a cheap lower bound on UTF-8 length, so huge strings
    // are refused before even scanning their UTF-8 representation.
    if string.utf16_len()? > *remaining_string_bytes {
        return Err(Error::invalid_argument(
            "LLM execution context exceeds 16384 serialized bytes",
        ));
    }
    // N-API's UTF-8 conversion replaces lone surrogates with U+FFFD. Decode
    // the bounded UTF-16 input strictly so distinct keys cannot collapse.
    let text = string.into_utf16()?.as_str().map_err(|_| {
        Error::invalid_argument("LLM execution context contains an unpaired UTF-16 surrogate")
    })?;
    *remaining_string_bytes = remaining_string_bytes
        .checked_sub(text.len())
        .ok_or_else(|| {
            Error::invalid_argument("LLM execution context exceeds 16384 serialized bytes")
        })?;
    Ok(text)
}

fn own_context_json(
    value: Unknown,
    remaining_depth: u32,
    remaining_nodes: &mut usize,
    remaining_string_bytes: &mut usize,
) -> Result<Value> {
    *remaining_nodes = remaining_nodes
        .checked_sub(1)
        .ok_or_else(|| Error::invalid_argument("Exceeded maximum JSON value count"))?;
    let value_type = value.get_type()?;
    if value_type == ValueType::String {
        return context_string(value, remaining_string_bytes).map(Value::String);
    }
    if value_type != ValueType::Object {
        return Ok(Json::from_unknown(value)?.0);
    }
    let remaining_depth = remaining_depth
        .checked_sub(1)
        .ok_or_else(|| Error::invalid_argument("Exceeded maximum JSON nesting depth"))?;
    let object = Object::from_unknown(value)?;
    if object.is_array()? {
        let length = object.get_array_length()?;
        if length as usize > *remaining_nodes {
            return Err(Error::invalid_argument("Exceeded maximum JSON value count"));
        }
        let mut values = Vec::new();
        for index in 0..length {
            // A sparse array must not import a value from its prototype.
            values.push(if object.has_own_property(&index.to_string())? {
                own_context_json(
                    object.get_element(index)?,
                    remaining_depth,
                    remaining_nodes,
                    remaining_string_bytes,
                )?
            } else {
                *remaining_nodes = remaining_nodes
                    .checked_sub(1)
                    .ok_or_else(|| Error::invalid_argument("Exceeded maximum JSON value count"))?;
                Value::Null
            });
        }
        return Ok(Value::Array(values));
    }
    let keys = object.get_all_property_names(
        KeyCollectionMode::OwnOnly,
        KeyFilter::Enumerable,
        KeyConversion::NumbersToStrings,
    )?;
    let mut values = serde_json::Map::new();
    for index in 0..keys.get_array_length()? {
        let key: Unknown = keys.get_element(index)?;
        if key.get_type()? != ValueType::String {
            continue;
        }
        // An earlier getter may have removed a key from the captured own-key list.
        if !object.has_own_property_js(key)? {
            continue;
        }
        let name = context_string(key, remaining_string_bytes)?;
        let value = object.get_property(key)?;
        values.insert(
            name,
            own_context_json(
                value,
                remaining_depth,
                remaining_nodes,
                remaining_string_bytes,
            )?,
        );
    }
    Ok(Value::Object(values))
}
