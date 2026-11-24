// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay_types::Json;
use pyo3::types::{
    PyByteArray, PyBytes, PyDict, PyFrozenSet, PyMapping, PySequence, PySet, PyString,
};
use pyo3::{intern, prelude::*};

/// Convert bounded execution context before recursive JSON conversion can overflow.
///
/// The shared Rust type validates its schema and serialized byte budget. This
/// conversion checks the exact bytes and depth of each copied value, including
/// cyclic containers, without reading application values a second time.
pub(crate) fn py_application_execution_context(
    obj: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<nemo_relay_types::api::llm::ApplicationExecutionContext>> {
    let Some(value) = obj.filter(|value| !value.is_none()) else {
        return Ok(None);
    };
    let json = context_json_snapshot(value, 0, &mut 16_384)?;
    nemo_relay_types::api::llm::ApplicationExecutionContext::try_from(json)
        .map(Some)
        .map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("invalid application execution context")
        })
}

fn context_json_snapshot(
    value: &Bound<'_, PyAny>,
    parent_depth: usize,
    remaining_bytes: &mut usize,
) -> PyResult<Json> {
    use pyo3::exceptions::PyValueError;
    use pyo3::types::{PyBool, PyFloat, PyInt};

    fn charge(remaining: &mut usize, amount: usize) -> PyResult<()> {
        *remaining = remaining.checked_sub(amount).ok_or_else(|| {
            PyValueError::new_err("LLM execution context exceeds 16384 serialized bytes")
        })?;
        Ok(())
    }
    if value.is_none() {
        charge(remaining_bytes, 4)?;
        return Ok(Json::Null);
    }
    if value.is_instance_of::<PyBool>() {
        let boolean: bool = value.extract()?;
        charge(remaining_bytes, if boolean { 4 } else { 5 })?;
        return Ok(Json::Bool(boolean));
    }
    if value.is_instance_of::<PyFloat>() {
        let number: f64 = value.extract()?;
        let number = serde_json::Number::from_f64(number)
            .ok_or_else(|| PyValueError::new_err("invalid application execution context"))?;
        charge(remaining_bytes, number.to_string().len())?;
        return Ok(Json::Number(number));
    }
    if value.is_instance_of::<PyInt>() {
        let number: Json = pythonize::depythonize(value)
            .map_err(|_| PyValueError::new_err("invalid application execution context"))?;
        charge(remaining_bytes, number.to_string().len())?;
        return Ok(number);
    }
    if let Ok(text) = value.cast::<PyString>() {
        // Check character length before creating a UTF-8 representation.
        validate_context_container_length(value, Some(*remaining_bytes))?;
        let text = text.to_cow()?;
        charge(remaining_bytes, 2)?;
        for character in text.chars() {
            let encoded_bytes = match character {
                '"' | '\\' | '\u{0008}' | '\u{000c}' | '\n' | '\r' | '\t' => 2,
                '\u{0000}'..='\u{001f}' => 6,
                _ => character.len_utf8(),
            };
            charge(remaining_bytes, encoded_bytes)?;
        }
        return Ok(Json::String(text.into_owned()));
    }
    if value.is_instance_of::<PyBytes>() || value.is_instance_of::<PyByteArray>() {
        validate_context_container_length(value, Some(*remaining_bytes))?;
        return Err(PyValueError::new_err(
            "invalid application execution context",
        ));
    }
    let depth = parent_depth + 1;
    if depth > 64 {
        return Err(PyValueError::new_err(
            "LLM execution context exceeds 64 container levels",
        ));
    }
    charge(remaining_bytes, 2)?;
    if let Ok(mapping) = value.cast::<PyMapping>() {
        validate_context_container_length(value, Some(*remaining_bytes))?;
        let mut fields = serde_json::Map::new();
        for (index, key) in mapping.keys()?.iter().enumerate() {
            charge(remaining_bytes, 1 + usize::from(index > 0))?;
            let key = context_json_snapshot(&key, depth, remaining_bytes)?;
            let Json::String(key) = key else {
                return Err(PyValueError::new_err(
                    "invalid application execution context",
                ));
            };
            let child = mapping.get_item(&key)?;
            let child = context_json_snapshot(&child, depth, remaining_bytes)?;
            fields.insert(key, child);
        }
        return Ok(Json::Object(fields));
    }
    if let Ok(sequence) = value.cast::<PySequence>() {
        let length = sequence.len()?;
        if length > *remaining_bytes {
            return Err(PyValueError::new_err(
                "LLM execution context exceeds 16384 serialized bytes",
            ));
        }
        let mut items = Vec::new();
        for index in 0..length {
            charge(remaining_bytes, usize::from(index > 0))?;
            let child = sequence.get_item(index)?;
            items.push(context_json_snapshot(&child, depth, remaining_bytes)?);
        }
        return Ok(Json::Array(items));
    }
    if let Ok(values) = value.cast::<PySet>() {
        validate_context_container_length(value, Some(*remaining_bytes))?;
        let mut items = Vec::new();
        for (index, child) in values.iter().enumerate() {
            charge(remaining_bytes, usize::from(index > 0))?;
            items.push(context_json_snapshot(&child, depth, remaining_bytes)?);
        }
        return Ok(Json::Array(items));
    }
    if let Ok(values) = value.cast::<PyFrozenSet>() {
        validate_context_container_length(value, Some(*remaining_bytes))?;
        let mut items = Vec::new();
        for (index, child) in values.iter().enumerate() {
            charge(remaining_bytes, usize::from(index > 0))?;
            items.push(context_json_snapshot(&child, depth, remaining_bytes)?);
        }
        return Ok(Json::Array(items));
    }
    if let Some(fields) = value
        .getattr_opt(intern!(value.py(), "__dataclass_fields__"))
        .ok()
        .flatten()
    {
        validate_context_container_length(&fields, Some(*remaining_bytes))?;
        let attributes = value.getattr(intern!(value.py(), "__dict__"))?;
        let attributes = attributes.cast::<PyDict>()?;
        let mut snapshot = serde_json::Map::new();
        for (index, key) in fields.cast::<PyDict>()?.keys().iter().enumerate() {
            charge(remaining_bytes, 1 + usize::from(index > 0))?;
            let Json::String(key) = context_json_snapshot(&key, depth, remaining_bytes)? else {
                return Err(PyValueError::new_err(
                    "invalid application execution context",
                ));
            };
            let child = attributes
                .get_item(&key)?
                .ok_or_else(|| PyValueError::new_err("invalid application execution context"))?;
            snapshot.insert(key, context_json_snapshot(&child, depth, remaining_bytes)?);
        }
        return Ok(Json::Object(snapshot));
    }
    Err(PyValueError::new_err(
        "invalid application execution context",
    ))
}

fn validate_context_container_length(
    value: &Bound<'_, PyAny>,
    remaining: Option<usize>,
) -> PyResult<()> {
    if let Some(limit) = remaining
        && value.len()? > limit
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "LLM execution context exceeds 16384 serialized bytes",
        ));
    }
    Ok(())
}
