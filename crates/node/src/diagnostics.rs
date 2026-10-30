// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Embedded diagnostics readers. No language object retains a core query lease.
use napi::{Env, JsString, JsUnknown};
use napi_derive::napi;
use nemo_relay_runtime::diagnostics::embedded::{
    self, EmbeddedDiagnosticError, EmbeddedDiagnosticsReader,
};
use serde_json::Value as Json;

fn error(error: EmbeddedDiagnosticError) -> napi::Error {
    napi::Error::new(napi::Status::GenericFailure, error.reason())
}
fn bounded_query(query: Option<JsString>) -> napi::Result<String> {
    let Some(query) = query else {
        return Ok("{}".into());
    };
    if query.utf8_len()? > embedded::MAX_BINDING_QUERY_BYTES {
        return Err(napi::Error::from_reason("invalid_query"));
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
fn native_value(env: &Env, value: &Json, field: &str) -> napi::Result<JsUnknown> {
    match value {
        Json::String(text) if embedded::is_diagnostic_number_field(field) => {
            let number = text
                .parse::<u64>()
                .map_err(|_| napi::Error::from_reason("invalid_snapshot"))?;
            env.create_bigint_from_u64(number)?.into_unknown()
        }
        Json::Object(fields) => {
            let mut object = env.create_object()?;
            for (key, value) in fields {
                object.set_named_property(&camel_case(key), native_value(env, value, key)?)?;
            }
            Ok(object.into_unknown())
        }
        Json::Array(values) => {
            let mut array = env.create_array_with_length(values.len())?;
            for (index, value) in values.iter().enumerate() {
                array.set_element(index as u32, native_value(env, value, "")?)?;
            }
            Ok(array.into_unknown())
        }
        Json::Number(number) => {
            if let Some(integer) = number.as_u64()
                && integer > 9_007_199_254_740_991
            {
                return Err(napi::Error::from_reason("numeric_out_of_range"));
            }
            let value = number
                .as_f64()
                .filter(|n| n.is_finite())
                .ok_or_else(|| napi::Error::from_reason("numeric_out_of_range"))?;
            Ok(env.create_double(value)?.into_unknown())
        }
        _ => env.to_js_value(value),
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
    #[napi]
    pub fn snapshot_json(&self, query_json: Option<JsString>) -> napi::Result<String> {
        let query = bounded_query(query_json)?;
        let page = self.inner.snapshot_json(&query).map_err(error)?;
        Ok(page.as_str().to_owned())
    }
    /// Return a caller-owned camelCase object. Diagnostic u64 fields are always bigint.
    /// Use snapshotJson for portable serialization; JSON.stringify cannot encode bigint.
    #[napi(ts_return_type = "object")]
    pub fn snapshot(&self, env: Env, query_json: Option<JsString>) -> napi::Result<JsUnknown> {
        let query = bounded_query(query_json)?;
        let page = self.inner.snapshot_json(&query).map_err(error)?;
        let value: Json = serde_json::from_str(page.as_str())
            .map_err(|_| napi::Error::from_reason("invalid_snapshot"))?;
        native_value(&env, &value, "")
    }
    /// Return canonical snake_case JSON with decimal-string diagnostic integers.
    #[napi]
    pub fn inspect_memory_json(&self, query_json: Option<JsString>) -> napi::Result<String> {
        let query = bounded_query(query_json)?;
        let page = self.inner.inspect_memory_json(&query).map_err(error)?;
        Ok(page.as_str().to_owned())
    }
    /// Return a caller-owned camelCase object. Diagnostic u64 fields are always bigint.
    /// Use inspectMemoryJson for portable serialization; JSON.stringify cannot encode bigint.
    #[napi(ts_return_type = "object")]
    pub fn inspect_memory(
        &self,
        env: Env,
        query_json: Option<JsString>,
    ) -> napi::Result<JsUnknown> {
        let query = bounded_query(query_json)?;
        let page = self.inner.inspect_memory_json(&query).map_err(error)?;
        let value: Json = serde_json::from_str(page.as_str())
            .map_err(|_| napi::Error::from_reason("invalid_snapshot"))?;
        native_value(&env, &value, "")
    }
    /// Return independent core/adapter budgets, occupancy, and logging coverage.
    #[napi(ts_return_type = "object")]
    pub fn info(&self, env: Env) -> napi::Result<JsUnknown> {
        let value = serde_json::to_value(self.inner.info().map_err(error)?)
            .map_err(|_| napi::Error::from_reason("invalid_metadata"))?;
        native_value(&env, &value, "")
    }
}

/// Inspect the configured service without starting it. Errors use a stable reason message.
#[napi]
pub fn diagnostics_reader() -> napi::Result<DiagnosticsReader> {
    embedded::reader()
        .map(|inner| DiagnosticsReader { inner })
        .map_err(error)
}

/// Internal descriptor consumed by the JavaScript query adapter.
#[napi(js_name = "_diagnosticsQueryContract")]
pub fn diagnostics_query_contract() -> String {
    embedded::binding_query_contract_json()
}
