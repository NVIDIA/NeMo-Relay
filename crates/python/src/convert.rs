// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bidirectional conversion between Python objects and `serde_json::Value`.
//!
//! Uses the [`pythonize`] crate under the hood.  The four public helpers cover
//! the required/optional × to-json/from-json matrix used throughout the PyO3
//! binding layer.

use std::collections::HashSet;

use chrono::{DateTime, Utc};
use nemo_relay_types::error::FlowError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::{
    PyByteArray, PyBytes, PyDict, PyFrozenSet, PyMapping, PySequence, PySet, PyString,
};
use pyo3::{intern, prelude::*};
use serde_json::Value as Json;

/// Convert a [`FlowError`] into Python exception the caller sees.
///
/// A callback exception is re-raised as the original object, all others as `RuntimeError`.
pub(crate) fn flow_error_to_py_err(error: FlowError) -> PyErr {
    Python::attach(move |py| {
        let original = match &error {
            FlowError::CallbackException {
                source: Some(source),
                ..
            } => source
                .downcast_ref::<PyErr>()
                .map(|original| original.clone_ref(py)),
            _ => None,
        };
        match original {
            Some(original) => {
                // drop runtime error while attached so shared reference
                // is released now instead of queued until next attach.
                drop(error);
                original
            }
            None => PyRuntimeError::new_err(error.to_string()),
        }
    })
}

fn validate_context_container_length(
    value: &Bound<'_, PyAny>,
    remaining_units: Option<usize>,
) -> PyResult<()> {
    if let Some(remaining) = remaining_units
        && value.len()? > remaining
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "tool execution context exceeds 16384 serialized bytes",
        ));
    }
    Ok(())
}

fn charge_context_value(
    value: &Bound<'_, PyAny>,
    remaining_context_units: &mut Option<usize>,
) -> PyResult<()> {
    if let Some(remaining_units) = remaining_context_units {
        let minimum_units = if value.is_instance_of::<PyString>()
            || value.is_instance_of::<PyBytes>()
            || value.is_instance_of::<PyByteArray>()
        {
            value.len()?.saturating_add(1)
        } else {
            1
        };
        *remaining_units = remaining_units.checked_sub(minimum_units).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "tool execution context exceeds 16384 serialized bytes",
            )
        })?;
    }
    Ok(())
}

fn validate_acyclic(
    value: &Bound<'_, PyAny>,
    active_containers: &mut HashSet<usize>,
    parent_depth: usize,
    depth_limit: Option<usize>,
    remaining_context_units: &mut Option<usize>,
) -> PyResult<()> {
    charge_context_value(value, remaining_context_units)?;
    if value.is_instance_of::<PyString>()
        || value.is_instance_of::<PyBytes>()
        || value.is_instance_of::<PyByteArray>()
    {
        return Ok(());
    }

    let dataclass_fields = value
        .getattr_opt(intern!(value.py(), "__dataclass_fields__"))
        .ok()
        .flatten();
    let is_container = value.cast::<PySet>().is_ok()
        || value.cast::<PyFrozenSet>().is_ok()
        || value.cast::<PySequence>().is_ok()
        || value.cast::<PyMapping>().is_ok();
    if !is_container && dataclass_fields.is_none() {
        return Ok(());
    }

    let depth = parent_depth + 1;
    if depth_limit.is_some_and(|limit| depth > limit) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "tool execution context exceeds 64 container levels",
        ));
    }
    if is_container {
        validate_context_container_length(value, *remaining_context_units)?;
    } else if let Some(fields) = &dataclass_fields {
        validate_context_container_length(fields, *remaining_context_units)?;
    }

    let identity = value.as_ptr() as usize;
    if !active_containers.insert(identity) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Failed to convert to JSON: circular reference detected",
        ));
    }

    let result = if let Ok(set) = value.cast::<PySet>() {
        set.iter().try_for_each(|child| {
            validate_acyclic(
                &child,
                active_containers,
                depth,
                depth_limit,
                remaining_context_units,
            )
        })
    } else if let Ok(set) = value.cast::<PyFrozenSet>() {
        set.iter().try_for_each(|child| {
            validate_acyclic(
                &child,
                active_containers,
                depth,
                depth_limit,
                remaining_context_units,
            )
        })
    } else if let Ok(sequence) = value.cast::<PySequence>() {
        (0..sequence.len()?).try_for_each(|index| {
            let child = sequence.get_item(index)?;
            validate_acyclic(
                &child,
                active_containers,
                depth,
                depth_limit,
                remaining_context_units,
            )
        })
    } else if let Ok(mapping) = value.cast::<PyMapping>() {
        let keys = mapping.keys()?;
        let values = mapping.values()?;
        keys.iter().chain(values.iter()).try_for_each(|child| {
            validate_acyclic(
                &child,
                active_containers,
                depth,
                depth_limit,
                remaining_context_units,
            )
        })
    } else if let Some(fields) = dataclass_fields {
        let fields = fields.cast::<PyDict>()?.keys();
        let attributes = value
            .getattr(intern!(value.py(), "__dict__"))?
            .cast_into::<PyDict>()?;
        fields.iter().try_for_each(|field| {
            charge_context_value(&field, remaining_context_units)?;
            if let Some(child) = attributes.get_item(&field)? {
                validate_acyclic(
                    &child,
                    active_containers,
                    depth,
                    depth_limit,
                    remaining_context_units,
                )?;
            }
            Ok(())
        })
    } else {
        Ok(())
    };

    active_containers.remove(&identity);
    result
}

/// Convert a Python object to serde_json::Value via pythonize.
pub fn py_to_json(obj: &Bound<'_, PyAny>) -> PyResult<Json> {
    validate_acyclic(obj, &mut HashSet::new(), 0, None, &mut None)?;
    pythonize::depythonize(obj).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to convert to JSON: {e}"))
    })
}

/// Convert bounded execution context before recursive JSON conversion can overflow.
///
/// The shared Rust type validates its schema and serialized byte budget. This
/// conversion checks the exact bytes and depth of each copied value, including
/// cyclic containers, without reading application values a second time.
pub(crate) fn py_application_execution_context(
    obj: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<nemo_relay_types::api::tool::ApplicationExecutionContext>> {
    let Some(value) = obj.filter(|value| !value.is_none()) else {
        return Ok(None);
    };
    let json = context_json_snapshot(value, 0, &mut 16_384)?;
    nemo_relay_types::api::tool::ApplicationExecutionContext::try_from(json)
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
            PyValueError::new_err("tool execution context exceeds 16384 serialized bytes")
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
            "tool execution context exceeds 64 container levels",
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
                "tool execution context exceeds 16384 serialized bytes",
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

/// Convert a serde_json::Value to a Python object via pythonize.
pub fn json_to_py(py: Python<'_>, value: &Json) -> PyResult<Py<PyAny>> {
    let obj: Bound<'_, PyAny> = pythonize::pythonize(py, value).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to convert from JSON: {e}"))
    })?;
    Ok(obj.unbind())
}

/// Convert an optional Python object to Option<Json>.
pub fn opt_py_to_json(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Json>> {
    match obj {
        Some(o) if !o.is_none() => Ok(Some(py_to_json(o)?)),
        _ => Ok(None),
    }
}

/// Convert an Option<Json> to a Python object (or None).
pub fn opt_json_to_py(py: Python<'_>, value: &Option<Json>) -> PyResult<Py<PyAny>> {
    match value {
        Some(v) => json_to_py(py, v),
        None => Ok(py.None()),
    }
}

/// Convert an optional timezone-aware Python datetime to a UTC timestamp.
pub fn opt_py_to_timestamp(value: Option<&Bound<'_, PyAny>>) -> PyResult<Option<DateTime<Utc>>> {
    let Some(timestamp) = value.filter(|timestamp| !timestamp.is_none()) else {
        return Ok(None);
    };

    let py = timestamp.py();
    let datetime_type = py.import("datetime")?.getattr("datetime")?;
    if !timestamp.is_instance(&datetime_type)? {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "timestamp must be a datetime.datetime object",
        ));
    }
    if timestamp.getattr("tzinfo")?.is_none() || timestamp.call_method0("utcoffset")?.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "timestamp datetime must be timezone-aware",
        ));
    }

    let iso_timestamp: String = timestamp.call_method0("isoformat")?.extract()?;
    DateTime::parse_from_rfc3339(&iso_timestamp)
        .map(|timestamp| Some(timestamp.with_timezone(&Utc)))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid timestamp: {e}")))
}

#[cfg(test)]
#[path = "../tests/unit/convert_tests.rs"]
mod tests;
