// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use serde_json::json;

#[test]
fn optional_python_json_helpers_round_trip_values() {
    Python::initialize();
    Python::attach(|py| {
        let value = Some(json!({"nested": {"value": 7}}));

        let py_value = opt_json_to_py(py, &value).unwrap();
        let roundtrip = opt_py_to_json(Some(py_value.bind(py))).unwrap();
        assert_eq!(roundtrip, value);

        let none_value = opt_json_to_py(py, &None).unwrap();
        assert!(none_value.bind(py).is_none());
        assert_eq!(opt_py_to_json(Some(py.None().bind(py))).unwrap(), None);
        assert_eq!(opt_py_to_json(None).unwrap(), None);
    });
}

#[test]
fn json_helpers_report_invalid_python_objects() {
    Python::initialize();
    Python::attach(|py| {
        let builtins = py.import("builtins").unwrap();
        let object = builtins.getattr("object").unwrap().call0().unwrap();

        let err = py_to_json(&object).unwrap_err();
        assert!(err.to_string().contains("Failed to convert to JSON"));
    });
}
