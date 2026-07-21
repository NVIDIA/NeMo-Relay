// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bidirectional conversion between Python objects and `serde_json::Value`.
//!
//! Uses the [`pythonize`] crate under the hood.  The four public helpers cover
//! the required/optional × to-json/from-json matrix used throughout the PyO3
//! binding layer.

use std::collections::HashSet;

use chrono::{DateTime, Utc};
use pyo3::types::{
    PyByteArray, PyBytes, PyDict, PyFrozenSet, PyMapping, PySequence, PySet, PyString,
};
use pyo3::{intern, prelude::*};
use serde_json::Value as Json;

fn validate_acyclic(
    value: &Bound<'_, PyAny>,
    active_containers: &mut HashSet<usize>,
) -> PyResult<()> {
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
        || value.cast::<PyMapping>().is_ok()
        || dataclass_fields.is_some();
    if !is_container {
        return Ok(());
    }

    let identity = value.as_ptr() as usize;
    if !active_containers.insert(identity) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Failed to convert to JSON: circular reference detected",
        ));
    }

    let result = if let Ok(set) = value.cast::<PySet>() {
        set.iter()
            .try_for_each(|child| validate_acyclic(&child, active_containers))
    } else if let Ok(set) = value.cast::<PyFrozenSet>() {
        set.iter()
            .try_for_each(|child| validate_acyclic(&child, active_containers))
    } else if let Ok(sequence) = value.cast::<PySequence>() {
        (0..sequence.len()?).try_for_each(|index| {
            let child = sequence.get_item(index)?;
            validate_acyclic(&child, active_containers)
        })
    } else if let Ok(mapping) = value.cast::<PyMapping>() {
        let keys = mapping.keys()?;
        let values = mapping.values()?;
        keys.iter()
            .chain(values.iter())
            .try_for_each(|child| validate_acyclic(&child, active_containers))
    } else if let Some(fields) = dataclass_fields {
        let fields = fields.cast::<PyDict>()?.keys();
        let attributes = value
            .getattr(intern!(value.py(), "__dict__"))?
            .cast_into::<PyDict>()?;
        fields.iter().try_for_each(|field| {
            if let Some(child) = attributes.get_item(&field)? {
                validate_acyclic(&child, active_containers)?;
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
    validate_acyclic(obj, &mut HashSet::new())?;
    pythonize::depythonize(obj).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to convert to JSON: {e}"))
    })
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
