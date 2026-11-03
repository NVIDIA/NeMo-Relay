// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed conversions between shared domain values and protobuf values.
//!
//! The host and Rust worker use the same [`WireValue`] implementations so native
//! and worker callbacks retain the types from `nemo_relay_types`. Generated
//! protobuf types describe transport presence and alternatives separately.
//!
//! `dto.rs` declares struct fields, enum mappings, and collection wrappers.
//! Matching field names use `field: mode`; renames use `source => target: mode`.
//! Both directions construct every struct field, making omitted mappings a
//! compile-time error. `variants.rs` handles enums and structural differences
//! explicitly. Shared helpers here check required values, integer widths,
//! timestamps, and JSON syntax while preserving native null semantics.
//!
//! See the crate's `CONTRACT.md`, "Rust domain conversions", for field modes,
//! representation differences, and the maintenance workflow.

use crate::nvidia_worker_v1 as pb;
use chrono::{DateTime, Utc};
use nemo_relay_types::Json;
use std::collections::HashMap;
use uuid::Uuid;

/// A malformed or unsupported value at the worker protocol boundary.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct ConversionError(pub String);

/// Result of a domain/protobuf conversion.
pub type Result<T> = std::result::Result<T, ConversionError>;

/// Bidirectional conversion without exposing transport representation to callbacks.
pub trait WireValue<W>: Sized {
    /// Encode an owned domain value.
    fn to_wire(self) -> Result<W>;
    /// Decode a protobuf value into an owned domain value.
    fn from_wire(value: W) -> Result<Self>;
}

/// Encode a domain value into its protobuf representation.
pub fn encode<D: WireValue<W>, W>(value: D) -> Result<W> {
    value.to_wire()
}

/// Decode a protobuf value into its shared domain representation.
pub fn decode<D: WireValue<W>, W>(value: W) -> Result<D> {
    D::from_wire(value)
}

fn required<T>(value: Option<T>, field: &str) -> Result<T> {
    value.ok_or_else(|| ConversionError(format!("missing required {field}")))
}

macro_rules! identity {
    ($($ty:ty),* $(,)?) => {$(impl WireValue<$ty> for $ty {
        fn to_wire(self) -> Result<$ty> { Ok(self) }
        fn from_wire(value: $ty) -> Result<Self> { Ok(value) }
    })*};
}
identity!(String, bool, u32, u64, i32, i64);

impl WireValue<f64> for f64 {
    fn to_wire(self) -> Result<f64> {
        if self.is_finite() {
            Ok(self)
        } else {
            Err(ConversionError("non-finite number".into()))
        }
    }
    fn from_wire(value: f64) -> Result<Self> {
        value.to_wire()
    }
}

impl WireValue<u64> for usize {
    fn to_wire(self) -> Result<u64> {
        u64::try_from(self).map_err(|e| ConversionError(e.to_string()))
    }
    fn from_wire(value: u64) -> Result<Self> {
        usize::try_from(value).map_err(|e| ConversionError(e.to_string()))
    }
}

impl WireValue<String> for Uuid {
    fn to_wire(self) -> Result<String> {
        Ok(self.to_string())
    }
    fn from_wire(value: String) -> Result<Self> {
        value
            .parse()
            .map_err(|e: uuid::Error| ConversionError(format!("invalid UUID: {e}")))
    }
}

impl WireValue<prost_types::Timestamp> for DateTime<Utc> {
    fn to_wire(self) -> Result<prost_types::Timestamp> {
        let value = prost_types::Timestamp {
            seconds: self.timestamp(),
            nanos: self.timestamp_subsec_nanos() as i32,
        };
        if !(-62135596800..=253402300799).contains(&value.seconds) {
            return Err(ConversionError("timestamp outside protobuf range".into()));
        }
        Ok(value)
    }
    fn from_wire(value: prost_types::Timestamp) -> Result<Self> {
        if !(-62135596800..=253402300799).contains(&value.seconds)
            || !(0..1_000_000_000).contains(&value.nanos)
        {
            return Err(ConversionError("invalid protobuf timestamp".into()));
        }
        DateTime::from_timestamp(value.seconds, value.nanos as u32)
            .ok_or_else(|| ConversionError("invalid timestamp".into()))
    }
}

impl WireValue<pb::JsonValue> for Json {
    fn to_wire(self) -> Result<pb::JsonValue> {
        Ok(pb::JsonValue {
            json: serde_json::to_vec(&self).map_err(|e| ConversionError(e.to_string()))?,
        })
    }
    fn from_wire(value: pb::JsonValue) -> Result<Self> {
        serde_json::from_slice(&value.json)
            .map_err(|e| ConversionError(format!("invalid JSON: {e}")))
    }
}

impl WireValue<pb::JsonObject> for Json {
    fn to_wire(self) -> Result<pb::JsonObject> {
        let Json::Object(fields) = self else {
            return Err(ConversionError("expected JSON object".into()));
        };
        Ok(pb::JsonObject {
            fields: fields
                .into_iter()
                .map(|(k, v)| Ok((k, encode(v)?)))
                .collect::<Result<_>>()?,
        })
    }
    fn from_wire(value: pb::JsonObject) -> Result<Self> {
        Ok(Json::Object(
            value
                .fields
                .into_iter()
                .map(|(k, v)| Ok((k, decode(v)?)))
                .collect::<Result<_>>()?,
        ))
    }
}

impl WireValue<pb::JsonArray> for Json {
    fn to_wire(self) -> Result<pb::JsonArray> {
        let Json::Array(values) = self else {
            return Err(ConversionError("expected JSON array".into()));
        };
        Ok(pb::JsonArray {
            values: values.into_iter().map(encode).collect::<Result<_>>()?,
        })
    }
    fn from_wire(value: pb::JsonArray) -> Result<Self> {
        Ok(Json::Array(
            value
                .values
                .into_iter()
                .map(decode)
                .collect::<Result<_>>()?,
        ))
    }
}

fn encode_map<D: WireValue<W>, W>(
    value: impl IntoIterator<Item = (String, D)>,
) -> Result<HashMap<String, W>> {
    value
        .into_iter()
        .map(|(k, v)| Ok((k, encode(v)?)))
        .collect()
}
fn decode_map<D: WireValue<W>, W, C: FromIterator<(String, D)>>(
    value: HashMap<String, W>,
) -> Result<C> {
    value
        .into_iter()
        .map(|(k, v)| Ok((k, decode(v)?)))
        .collect()
}

fn decode_optional_codec_data<D: serde::de::DeserializeOwned>(
    value: Option<pb::JsonValue>,
) -> Result<Option<D>> {
    value
        .map(|value| {
            serde_json::from_slice::<Option<D>>(&value.json)
                .map_err(|error| ConversionError(error.to_string()))
        })
        .transpose()
        .map(Option::flatten)
}

macro_rules! field_encode {
    ($v:expr, codec_data) => {
        $v.map(encode).transpose()?
    };
    ($v:expr, annotation) => {
        $v.filter(|value| !value.is_null())
            .map(encode)
            .transpose()?
    };
    ($v:expr, fixed_slots) => {
        $v.into_iter().map(encode).collect::<Result<Vec<_>>>()?
    };
    ($v:expr, scalar) => {
        encode($v)?
    };
    ($v:expr, required) => {
        Some(encode($v)?)
    };
    ($v:expr, optional) => {
        $v.map(encode).transpose()?
    };
    ($v:expr, repeated) => {
        $v.into_iter().map(encode).collect::<Result<_>>()?
    };
    ($v:expr, map) => {
        encode_map($v)?
    };
}
macro_rules! field_decode {
    ($v:expr, codec_data) => {
        decode_optional_codec_data($v)?
    };
    ($v:expr, annotation) => {
        $v.map(decode::<Json, _>)
            .transpose()?
            .filter(|value| !value.is_null())
    };
    ($v:expr, fixed_slots) => {
        $v.into_iter()
            .map(decode)
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_: Vec<_>| ConversionError("expected eight limit slots".into()))?
    };
    ($v:expr, scalar) => {
        decode($v)?
    };
    ($v:expr, required) => {
        decode(required($v, stringify!($v))?)?
    };
    ($v:expr, optional) => {
        $v.map(decode).transpose()?
    };
    ($v:expr, repeated) => {
        $v.into_iter().map(decode).collect::<Result<_>>()?
    };
    ($v:expr, map) => {
        decode_map($v)?
    };
}
// A flattened `extra` map is not itself a modeled JSON property. Guard native
// field names before flattening; protobuf field names may intentionally differ.
fn check_extra_keys<'a>(
    keys: impl IntoIterator<Item = &'a String>,
    modeled: &[&str],
) -> Result<()> {
    for key in keys {
        if key != "extra" && modeled.contains(&key.as_str()) {
            return Err(ConversionError(format!(
                "extension field {key:?} shadows a modeled field"
            )));
        }
    }
    Ok(())
}

fn encode_extra(
    value: serde_json::Map<String, Json>,
    modeled: &[&str],
) -> Result<HashMap<String, pb::JsonValue>> {
    check_extra_keys(value.keys(), modeled)?;
    encode_map(value)
}

fn decode_extra(
    value: HashMap<String, pb::JsonValue>,
    modeled: &[&str],
) -> Result<serde_json::Map<String, Json>> {
    check_extra_keys(value.keys(), modeled)?;
    decode_map(value)
}

macro_rules! check_dto_extra {
    ($value:expr, [$($all:ident),*];) => {};
    ($value:expr, [$($all:ident),*]; extra $(, $rest:ident)*) => {
        check_extra_keys($value.extra.keys(), &[$(stringify!($all)),*])?;
    };
    ($value:expr, [$($all:ident),*]; $first:ident $(, $rest:ident)*) => {
        check_dto_extra!($value, [$($all),*]; $($rest),*);
    };
}

// Normalize matching field names before emitting complete struct initializers.
// Leaving out a domain or protobuf field must remain a compile-time error.
macro_rules! dto {
    (@fields [$domain:path => $wire:ident] [$($mapped:tt)*];) => {
        dto!(@mapped $domain => $wire { $($mapped)* });
    };
    (@fields [$domain:path => $wire:ident] [$($mapped:tt)*];
        $source:ident => $target:ident : $mode:ident, $($rest:tt)*) => {
        dto!(@fields [$domain => $wire] [$($mapped)* $source => $target : $mode,]; $($rest)*);
    };
    (@fields [$domain:path => $wire:ident] [$($mapped:tt)*];
        $field:ident : $mode:ident, $($rest:tt)*) => {
        dto!(@fields [$domain => $wire] [$($mapped)* $field => $field : $mode,]; $($rest)*);
    };
    (@mapped $domain:path => $wire:ident { $($source:ident => $target:ident : $mode:ident),* $(,)? }) => {
        impl WireValue<pb::$wire> for $domain {
            fn to_wire(self) -> Result<pb::$wire> {
                check_dto_extra!(self, [$($source),*]; $($source),*);
                Ok(pb::$wire { $($target: field_encode!(self.$source, $mode)),* })
            }
            fn from_wire(value: pb::$wire) -> Result<Self> {
                check_dto_extra!(value, [$($source),*]; $($source),*);
                Ok(Self { $($source: field_decode!(value.$target, $mode)),* })
            }
        }
        impl WireValue<Box<pb::$wire>> for $domain {
            fn to_wire(self) -> Result<Box<pb::$wire>> { Ok(Box::new(<Self as WireValue<pb::$wire>>::to_wire(self)?)) }
            fn from_wire(value: Box<pb::$wire>) -> Result<Self> { <Self as WireValue<pb::$wire>>::from_wire(*value) }
        }
    };
    ($domain:path => $wire:ident { $($source:ident $(=> $target:ident)? : $mode:ident),* $(,)? }) => {
        dto!(@fields [$domain => $wire] []; $($source $(=> $target)? : $mode,)*);
    };
}

// Collection wrappers preserve the presence of explicitly empty lists.
macro_rules! list_codec {
    ($domain:ty => $wire:ident) => {
        impl WireValue<pb::$wire> for Vec<$domain> {
            fn to_wire(self) -> Result<pb::$wire> {
                Ok(pb::$wire {
                    values: self.into_iter().map(encode).collect::<Result<_>>()?,
                })
            }
            fn from_wire(value: pb::$wire) -> Result<Self> {
                value.values.into_iter().map(decode).collect()
            }
        }
    };
}

macro_rules! enum_codec {
    ($domain:path => $wire:ident { $($variant:ident => $target:ident),* $(,)? }) => {
        impl WireValue<i32> for $domain {
            #[allow(unreachable_patterns)]
            fn to_wire(self) -> Result<i32> { Ok(match self { $(Self::$variant => pb::$wire::$target as i32,)* _ => return Err(ConversionError(concat!("unsupported ", stringify!($wire)).into())) }) }
            fn from_wire(value: i32) -> Result<Self> {
                match pb::$wire::try_from(value).ok() { $(Some(pb::$wire::$target) => Ok(Self::$variant),)* _ => Err(ConversionError(concat!("unknown ", stringify!($wire)).into())) }
            }
        }
    };
}

mod dto;
mod variants;

impl<D: Clone + WireValue<W>, W> WireValue<W> for std::sync::Arc<D> {
    fn to_wire(self) -> Result<W> {
        (*self).clone().to_wire()
    }
    fn from_wire(value: W) -> Result<Self> {
        Ok(std::sync::Arc::new(D::from_wire(value)?))
    }
}

impl WireValue<pb::JsonObject> for serde_json::Map<String, Json> {
    fn to_wire(self) -> Result<pb::JsonObject> {
        Ok(pb::JsonObject {
            fields: encode_map(self)?,
        })
    }
    fn from_wire(value: pb::JsonObject) -> Result<Self> {
        decode_map(value.fields)
    }
}

impl WireValue<u64> for nemo_relay_types::diagnostics::DiagnosticNumber {
    fn to_wire(self) -> Result<u64> {
        Ok(self.get())
    }
    fn from_wire(value: u64) -> Result<Self> {
        Ok(Self::new(value))
    }
}

impl WireValue<String> for nemo_relay_types::api::event::EventCategory {
    fn to_wire(self) -> Result<String> {
        Ok(self.as_str().to_owned())
    }
    fn from_wire(value: String) -> Result<Self> {
        Ok(Self::new(value))
    }
}

impl WireValue<String> for nemo_relay_types::codec::optimization::LlmOptimizationKind {
    fn to_wire(self) -> Result<String> {
        Ok(self.as_str().to_owned())
    }
    fn from_wire(value: String) -> Result<Self> {
        Ok(Self::new(value))
    }
}

impl WireValue<String> for nemo_relay_types::diagnostics::PluginExecutionName {
    fn to_wire(self) -> Result<String> {
        Ok(self.into_name().into_owned())
    }
    fn from_wire(value: String) -> Result<Self> {
        Ok(Self::from_opaque_name(value.into()))
    }
}
impl WireValue<String> for nemo_relay_types::diagnostics::PluginPhaseName {
    fn to_wire(self) -> Result<String> {
        Ok(self.into_name().into_owned())
    }
    fn from_wire(value: String) -> Result<Self> {
        Ok(Self::from_opaque_name(value.into()))
    }
}
impl WireValue<pb::MemoryLimitEventSlot>
    for Option<nemo_relay_types::diagnostics::MemoryLimitEvent>
{
    fn to_wire(self) -> Result<pb::MemoryLimitEventSlot> {
        Ok(pb::MemoryLimitEventSlot {
            state: Some(match self {
                None => pb::memory_limit_event_slot::State::Absent(()),
                Some(value) => pb::memory_limit_event_slot::State::Value(encode(value)?),
            }),
        })
    }
    fn from_wire(value: pb::MemoryLimitEventSlot) -> Result<Self> {
        match required(value.state, "memory slot state")? {
            pb::memory_limit_event_slot::State::Absent(_) => Ok(None),
            pb::memory_limit_event_slot::State::Value(value) => Ok(Some(decode(value)?)),
        }
    }
}

impl WireValue<pb::JsonObject> for std::collections::BTreeMap<String, Json> {
    fn to_wire(self) -> Result<pb::JsonObject> {
        Ok(pb::JsonObject {
            fields: encode_map(self)?,
        })
    }
    fn from_wire(value: pb::JsonObject) -> Result<Self> {
        decode_map(value.fields)
    }
}
