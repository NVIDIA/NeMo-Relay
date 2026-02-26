use pyo3::prelude::*;
use serde_json::Value as Json;

/// Convert a Python object to serde_json::Value via pythonize.
pub fn py_to_json(obj: &Bound<'_, PyAny>) -> PyResult<Json> {
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
