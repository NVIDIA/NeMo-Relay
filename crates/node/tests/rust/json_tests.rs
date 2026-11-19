// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{JAVASCRIPT_MAX_SAFE_INTEGER, json_number};

const MAX_SAFE: f64 = JAVASCRIPT_MAX_SAFE_INTEGER as f64;

#[test]
fn whole_numbers_within_the_safe_integer_range_become_integers() {
    for value in [
        0.0,
        1.0,
        4_294_967_295.0,
        4_294_967_296.0,
        8_589_934_592.0,
        MAX_SAFE,
    ] {
        let number = json_number(value).expect("finite");
        assert_eq!(number.as_u64(), Some(value as u64), "{value}");
        assert_eq!(number.as_i64(), Some(value as i64), "{value}");
        assert!(!number.is_f64(), "{value}");
    }
    for value in [-1.0, -2_147_483_648.0, -2_147_483_649.0, -MAX_SAFE] {
        let number = json_number(value).expect("finite");
        assert_eq!(number.as_i64(), Some(value as i64), "{value}");
        assert_eq!(number.as_u64(), None, "{value}");
    }
}

#[test]
fn negative_zero_becomes_integer_zero() {
    let number = json_number(-0.0).expect("finite");
    assert_eq!(number.as_u64(), Some(0));
    assert!(!number.is_f64());
}

#[test]
fn whole_numbers_beyond_the_safe_integer_range_stay_floats() {
    for value in [MAX_SAFE + 1.0, -(MAX_SAFE + 1.0), 9e18, 1e300] {
        let number = json_number(value).expect("finite");
        assert!(number.is_f64(), "{value}");
        assert_eq!(number.as_f64(), Some(value), "{value}");
    }
}

#[test]
fn fractions_stay_floats() {
    for value in [1.5, -0.25, 4_294_967_296.5] {
        let number = json_number(value).expect("finite");
        assert!(number.is_f64(), "{value}");
        assert_eq!(number.as_f64(), Some(value), "{value}");
    }
}

#[test]
fn non_finite_numbers_are_rejected() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(json_number(value), None, "{value}");
    }
}
