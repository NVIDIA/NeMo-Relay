// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Embedded diagnostics readers. No language object retains a core query lease.
use napi::bindgen_prelude::{BigInt, JsObjectValue, JsValue, Object, Unknown};
use napi::{Env, JsString};
use napi_derive::napi;

use crate::convert::js_value;
use crate::error::{Error, Result};
use nemo_relay::diagnostics::embedded::{self, EmbeddedDiagnosticError, EmbeddedDiagnosticsReader};
use serde_json::Value as Json;

fn error(error: EmbeddedDiagnosticError) -> Error {
    Error::from(error)
}
fn bounded_query(query: Option<JsString>) -> Result<String> {
    let Some(query) = query else {
        return Ok("{}".into());
    };
    if query.utf8_len()? > embedded::MAX_BINDING_QUERY_BYTES {
        return Err(Error::invalid_argument("invalid_query"));
    }
    Ok(query.into_utf8()?.as_str()?.to_owned())
}
fn camel_case(key: &str) -> String {
    let mut result = String::new();
    let mut uppercase = false;
    for ch in key.chars() {
        if ch == '_' {
            uppercase = true;
        } else if uppercase {
            result.extend(ch.to_uppercase());
            uppercase = false;
        } else {
            result.push(ch);
        }
    }
    result
}
fn native_value<'env>(env: &'env Env, value: &Json, field: &str) -> Result<Unknown<'env>> {
    match value {
        Json::String(text) if embedded::is_diagnostic_number_field(field) => {
            let number = text
                .parse::<u64>()
                .map_err(|_| Error::generic_failure("invalid_snapshot"))?;
            let value = js_value(env, BigInt::from(number))?;
            // SAFETY: `value` was created by the line above inside this call, in `env`.
            Ok(unsafe { value.into_unknown(env) })
        }
        Json::Object(fields) => {
            let mut object = Object::new(env)?;
            for (key, value) in fields {
                object.set_named_property(&camel_case(key), native_value(env, value, key)?)?;
            }
            Ok(object.to_unknown())
        }
        Json::Array(values) => {
            let mut array = env.create_array(values.len() as u32)?;
            for (index, value) in values.iter().enumerate() {
                array.set(index as u32, native_value(env, value, "")?)?;
            }
            Ok(array.to_unknown())
        }
        Json::Number(number) => {
            if let Some(integer) = number.as_u64()
                && integer > 9_007_199_254_740_991
            {
                return Err(Error::invalid_argument("numeric_out_of_range"));
            }
            let value = number
                .as_f64()
                .filter(|n| n.is_finite())
                .ok_or_else(|| Error::invalid_argument("numeric_out_of_range"))?;
            Ok(env.create_double(value)?.to_unknown())
        }
        _ => Ok(env.to_js_value(value)?),
    }
}

/// Authority to inspect restricted diagnostic evidence in the current process.
#[napi]
pub struct DiagnosticsReader {
    inner: EmbeddedDiagnosticsReader,
}
#[napi]
impl DiagnosticsReader {
    /// Return canonical snake_case JSON with decimal-string diagnostic integers.
    #[napi(catch_unwind)]
    pub fn snapshot_json(&self, query_json: Option<JsString>) -> Result<String> {
        let query = bounded_query(query_json)?;
        let page = self.inner.snapshot_json(&query).map_err(error)?;
        Ok(page.as_str().to_owned())
    }
    /// Return a caller-owned camelCase object. Diagnostic u64 fields are always bigint.
    /// Use snapshotJson for portable serialization; JSON.stringify cannot encode bigint.
    #[napi(ts_return_type = "object", catch_unwind)]
    pub fn snapshot<'env>(
        &self,
        env: &'env Env,
        query_json: Option<JsString>,
    ) -> Result<Unknown<'env>> {
        let query = bounded_query(query_json)?;
        let page = self.inner.snapshot_json(&query).map_err(error)?;
        let value: Json = serde_json::from_str(page.as_str())
            .map_err(|_| Error::generic_failure("invalid_snapshot"))?;
        native_value(env, &value, "")
    }
    /// Return canonical snake_case JSON with decimal-string diagnostic integers.
    #[napi(catch_unwind)]
    pub fn inspect_memory_json(&self, query_json: Option<JsString>) -> Result<String> {
        let query = bounded_query(query_json)?;
        let page = self.inner.inspect_memory_json(&query).map_err(error)?;
        Ok(page.as_str().to_owned())
    }
    /// Return a caller-owned camelCase object. Diagnostic u64 fields are always bigint.
    /// Use inspectMemoryJson for portable serialization; JSON.stringify cannot encode bigint.
    #[napi(ts_return_type = "object", catch_unwind)]
    pub fn inspect_memory<'env>(
        &self,
        env: &'env Env,
        query_json: Option<JsString>,
    ) -> Result<Unknown<'env>> {
        let query = bounded_query(query_json)?;
        let page = self.inner.inspect_memory_json(&query).map_err(error)?;
        let value: Json = serde_json::from_str(page.as_str())
            .map_err(|_| Error::generic_failure("invalid_snapshot"))?;
        native_value(env, &value, "")
    }
    /// Return independent core/adapter budgets, occupancy, and logging coverage.
    #[napi(ts_return_type = "object", catch_unwind)]
    pub fn info<'env>(&self, env: &'env Env) -> Result<Unknown<'env>> {
        let value = serde_json::to_value(self.inner.info().map_err(error)?)
            .map_err(|_| Error::generic_failure("invalid_metadata"))?;
        native_value(env, &value, "")
    }
}

/// Inspect the configured service without starting it. Errors use a stable reason message.
#[napi(catch_unwind)]
pub fn diagnostics_reader() -> Result<DiagnosticsReader> {
    embedded::reader()
        .map(|inner| DiagnosticsReader { inner })
        .map_err(error)
}

/// Internal descriptor consumed by the JavaScript query adapter.
#[napi(js_name = "_diagnosticsQueryContract", catch_unwind)]
pub fn diagnostics_query_contract() -> String {
    embedded::binding_query_contract_json()
}
