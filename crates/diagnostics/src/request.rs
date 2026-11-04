// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared bounded canonical query decoding for HTTP and embedded readers.

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Value};

use super::{DIAGNOSTIC_SCHEMA_VERSION, DiagnosticQuery, DiagnosticReadReason};

/// Maximum canonical UTF-8 query size, before JSON parsing. HTTP applies this
/// ceiling to the encoded URI query as well. Native helpers also bound conversion.
pub const MAX_DIAGNOSTIC_QUERY_BYTES: usize = 4096;

/// Decode an optional schema selector and the shared query. Unknown selectors,
/// duplicate fields (including nested ones), noncanonical numbers, and excessive
/// nesting fail safely. No file lookup, service startup, or snapshot lease occurs.
pub fn decode_diagnostic_query(input: &str) -> Result<DiagnosticQuery, DiagnosticReadReason> {
    if input.len() > MAX_DIAGNOSTIC_QUERY_BYTES {
        return Err(DiagnosticReadReason::InvalidQuery);
    }
    let StrictValue(Value::Object(fields)) =
        serde_json::from_str(input).map_err(|_| DiagnosticReadReason::InvalidQuery)?
    else {
        return Err(DiagnosticReadReason::InvalidQuery);
    };
    if let Some(version) = fields.get("schema_version") {
        let version: Option<u32> = serde_json::from_value(version.clone())
            .map_err(|_| DiagnosticReadReason::InvalidQuery)?;
        if version.is_some_and(|v| v != DIAGNOSTIC_SCHEMA_VERSION) {
            return Err(DiagnosticReadReason::UnsupportedSchema);
        }
    }
    // These structs also appear in snapshots, where additive fields are allowed.
    // Query admission must reject unknown selectors without changing that decoder.
    for (key, allowed) in [
        (
            "ownership",
            &[
                "subscriber_instance_id",
                "plugin_instance_id",
                "configuration_id",
                "responsibility",
            ][..],
        ),
        (
            "continuation",
            &[
                "source_instance_id",
                "sequence",
                "component_offset",
                "event_offset",
                "scope",
            ][..],
        ),
    ] {
        if let Some(Value::Object(object)) = fields.get(key)
            && object.keys().any(|name| !allowed.contains(&name.as_str()))
        {
            return Err(DiagnosticReadReason::InvalidQuery);
        }
    }
    serde_json::from_value(Value::Object(fields)).map_err(|_| DiagnosticReadReason::InvalidQuery)
}

// Query values are scalar or nested objects. Arrays have no meaning in this
// contract. serde_json's recursion limit plus the input cap bound parse scratch.
struct StrictValue(Value);
impl<'de> Deserialize<'de> for StrictValue {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct StrictVisitor;
        impl<'de> Visitor<'de> for StrictVisitor {
            type Value = StrictValue;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a diagnostic query object or scalar")
            }
            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
                let mut values = Map::new();
                while let Some(key) = map.next_key::<String>()? {
                    if values.contains_key(&key) {
                        return Err(de::Error::custom("duplicate query field"));
                    }
                    values.insert(key, map.next_value::<StrictValue>()?.0);
                }
                Ok(StrictValue(Value::Object(values)))
            }
            fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
                Ok(StrictValue(Value::String(value.into())))
            }
            fn visit_bool<E: de::Error>(self, value: bool) -> Result<Self::Value, E> {
                Ok(StrictValue(value.into()))
            }
            fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
                Ok(StrictValue(value.into()))
            }
            fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
                Ok(StrictValue(value.into()))
            }
            fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(StrictValue(Value::Null))
            }
        }
        d.deserialize_any(StrictVisitor)
    }
}
