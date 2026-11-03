// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Validate shapes, semantic annotations, and the syntax of open JSON values.

use prost::{Message, Name};
use prost_reflect::{DescriptorPool, DynamicMessage, ReflectMessage, Value};
use std::sync::OnceLock;

fn pool() -> &'static DescriptorPool {
    static POOL: OnceLock<DescriptorPool> = OnceLock::new();
    POOL.get_or_init(|| {
        DescriptorPool::decode(crate::FILE_DESCRIPTOR_SET).expect("compiled worker descriptors")
    })
}

/// Validate the complete message before dispatch or state mutation.
///
/// This does not authenticate credentials or establish capability ownership.
pub fn validate<M: Message + Name>(message: &M) -> Result<(), String> {
    static VALIDATOR: OnceLock<prost_protovalidate::Validator> = OnceLock::new();
    let descriptor = pool()
        .get_message_by_name(&M::full_name())
        .ok_or_else(|| "unknown worker message".to_owned())?;
    let dynamic = DynamicMessage::decode(descriptor, message.encode_to_vec().as_slice())
        .map_err(|e| e.to_string())?;
    VALIDATOR
        .get_or_init(prost_protovalidate::Validator::new)
        .validate(&dynamic)
        .map_err(|e| e.to_string())?;
    validate_json(&dynamic)
}

fn validate_json(message: &DynamicMessage) -> Result<(), String> {
    let name = message.descriptor().full_name().to_owned();
    if name == "google.protobuf.Timestamp" || name == "google.protobuf.Duration" {
        let seconds = message
            .get_field_by_name("seconds")
            .map(|v| match v.as_ref() {
                Value::I64(v) => *v,
                _ => 0,
            })
            .unwrap_or(0);
        let nanos = message
            .get_field_by_name("nanos")
            .map(|v| match v.as_ref() {
                Value::I32(v) => *v,
                _ => 0,
            })
            .unwrap_or(0);
        let valid = if name.ends_with("Timestamp") {
            (-62135596800..=253402300799).contains(&seconds) && (0..1_000_000_000).contains(&nanos)
        } else {
            (-315576000000..=315576000000).contains(&seconds)
                && (-999999999..=999999999).contains(&nanos)
                && (seconds == 0 || nanos == 0 || seconds.signum() == i64::from(nanos).signum())
        };
        if !valid {
            return Err(format!("invalid {name}"));
        }
    }
    if message.descriptor().full_name() == "nvidia.nemo.relay.worker.v1.JsonValue" {
        let field = message
            .descriptor()
            .get_field_by_name("json")
            .expect("JSON descriptor");
        if let Value::Bytes(bytes) = message.get_field(&field).as_ref() {
            serde_json::from_slice::<Box<serde_json::value::RawValue>>(bytes)
                .map_err(|e| format!("invalid JSON value: {e}"))?;
        }
    }
    for (_, value) in message.fields() {
        validate_value(value)?;
    }
    Ok(())
}

fn validate_value(value: &Value) -> Result<(), String> {
    match value {
        Value::Message(message) => validate_json(message),
        Value::List(values) => {
            for value in values {
                validate_value(value)?;
            }
            Ok(())
        }
        Value::Map(values) => {
            for value in values.values() {
                validate_value(value)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}
