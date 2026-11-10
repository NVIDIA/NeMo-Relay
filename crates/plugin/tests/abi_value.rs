// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lossless conversion of open application payloads at the native boundary.

use nemo_relay_plugin::{
    abi::{Field, Json as AbiJson},
    abi_value::{from_abi, into_abi, to_abi},
};
use serde_json::{Value, json};

#[test]
fn application_values_round_trip_without_a_text_codec() {
    let original = json!({"values":[null,true,false,"NUL\u{0}雪",[],{},i64::MIN,u64::MAX,9007199254740993_u64,1.25,-0.0]});
    assert_eq!(from_abi(to_abi(&original)).unwrap(), original);
    assert_eq!(from_abi(into_abi(original.clone())).unwrap(), original);
    let negative_zero = Value::from(-0.0);
    assert_eq!(
        from_abi(to_abi(&negative_zero))
            .unwrap()
            .as_f64()
            .unwrap()
            .to_bits(),
        (-0.0_f64).to_bits()
    );
}

#[test]
fn reject_invalid_numbers_and_duplicate_object_keys() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            from_abi(AbiJson::Float(value))
                .unwrap_err()
                .contains("non-finite")
        );
    }
    let fields = vec![
        Field {
            key: "same".into(),
            value: AbiJson::Null,
        },
        Field {
            key: "same".into(),
            value: AbiJson::Bool(true),
        },
    ];
    assert!(
        from_abi(AbiJson::Object(fields))
            .unwrap_err()
            .contains("duplicate")
    );
}
