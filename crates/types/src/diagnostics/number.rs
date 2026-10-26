// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lossless diagnostic integers and their schema.

use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use std::fmt;

/// Unsigned diagnostic integer serialized losslessly as a decimal string.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DiagnosticNumber(u64);

impl DiagnosticNumber {
    /// Construct a diagnostic integer without losing precision.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Return the complete Rust integer value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl From<u64> for DiagnosticNumber {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl From<DiagnosticNumber> for u64 {
    fn from(value: DiagnosticNumber) -> Self {
        value.0
    }
}

impl fmt::Display for DiagnosticNumber {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl Serialize for DiagnosticNumber {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for DiagnosticNumber {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        let canonical = value == "0"
            || (value.len() <= 20
                && value
                    .as_bytes()
                    .first()
                    .is_some_and(|byte| matches!(byte, b'1'..=b'9'))
                && value.bytes().all(|byte| byte.is_ascii_digit()));
        if !canonical {
            return Err(de::Error::custom(
                "expected a canonical unsigned decimal u64 string",
            ));
        }
        value.parse::<u64>().map(Self).map_err(de::Error::custom)
    }
}

#[cfg(feature = "schema")]
impl schemars::JsonSchema for DiagnosticNumber {
    fn schema_name() -> String {
        "DiagnosticNumber".into()
    }

    fn json_schema(_generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
        use schemars::schema::{InstanceType, Schema, SchemaObject, StringValidation};

        // Every 20-digit alternative fixes the maximum prefix and lowers one
        // digit; shorter canonical integers are always within u64's range.
        let maximum = u64::MAX.to_string();
        let mut alternatives = vec!["0".to_owned(), "[1-9][0-9]{0,18}".to_owned()];
        for (index, digit) in maximum.bytes().enumerate() {
            let first = if index == 0 { b'1' } else { b'0' };
            if digit > first {
                alternatives.push(format!(
                    "{}[{}-{}][0-9]{{{}}}",
                    &maximum[..index],
                    char::from(first),
                    char::from(digit - 1),
                    maximum.len() - index - 1,
                ));
            }
        }
        alternatives.push(maximum);

        Schema::Object(SchemaObject {
            instance_type: Some(InstanceType::String.into()),
            string: Some(Box::new(StringValidation {
                max_length: Some(20),
                min_length: Some(1),
                pattern: Some(format!("^({})$", alternatives.join("|"))),
            })),
            ..Default::default()
        })
    }
}
