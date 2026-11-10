// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// Frozen JSON vocabulary for native API v1.0. New variants require a new ABI major.
#[repr(u8)]
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// JSON null.
    Null,
    /// Boolean value.
    Bool(bool),
    /// Signed 64-bit integer, preserved without a floating-point round trip.
    Signed(i64),
    /// Unsigned 64-bit integer, including values greater than i64::MAX.
    Unsigned(u64),
    /// Finite IEEE-754 value. Adapters reject NaN and infinities.
    Float(f64),
    /// Owned UTF-8 string; embedded NUL bytes are allowed.
    String(String),
    /// Ordered sequence of values.
    Array(Vec<Json>),
    /// Object fields. Adapters reject duplicate keys.
    Object(Vec<Field>),
}

/// An owned JSON object field.
#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    /// UTF-8 object key.
    pub key: String,
    /// Field value.
    pub value: Json,
}
