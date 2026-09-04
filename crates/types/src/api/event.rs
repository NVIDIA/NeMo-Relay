// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Event types for Agent Trajectory Observability Format (ATOF) runtime events.

use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;
use uuid::Uuid;

use crate::Json;
use crate::api::llm::LlmAttributes;
use crate::api::scope::{HandleAttributes, ScopeAttributes, ScopeType};
use crate::api::tool::ToolAttributes;
use crate::codec::request::AnnotatedLlmRequest;
use crate::codec::response::AnnotatedLlmResponse;

/// Agent Trajectory Observability Format (ATOF) protocol version emitted by this runtime.
pub const ATOF_VERSION: &str = "0.1";

/// Reserved metadata key carrying the canonical severity of a mark exported as a log.
pub const LOG_SEVERITY_METADATA_KEY: &str = "nemo_relay.log.severity";

/// Relay-owned schema name for mark payloads containing metric measurements.
pub const METRIC_DATA_SCHEMA_NAME: &str = "nemo.relay.metric_measurements";

/// Current Relay-owned metric measurement schema version.
pub const METRIC_DATA_SCHEMA_VERSION: &str = "1";

/// Severity attached to a mark that may be projected as an OpenTelemetry log.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogSeverity {
    /// Fine-grained tracing information.
    Trace,
    /// Diagnostic information useful during development.
    Debug,
    /// Normal informational event.
    #[default]
    Info,
    /// Potential problem that did not prevent the operation from continuing.
    Warn,
    /// Failure or error condition.
    Error,
}

impl LogSeverity {
    /// Return the canonical lowercase wire value.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Trace => "trace",
            Self::Debug => "debug",
            Self::Info => "info",
            Self::Warn => "warn",
            Self::Error => "error",
        }
    }
}

impl fmt::Display for LogSeverity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Error returned when parsing an unsupported log severity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseLogSeverityError {
    value: String,
}

impl fmt::Display for ParseLogSeverityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid log severity {:?}; expected trace, debug, info, warn, warning, or error",
            self.value
        )
    }
}

impl std::error::Error for ParseLogSeverityError {}

impl FromStr for LogSeverity {
    type Err = ParseLogSeverityError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "trace" => Ok(Self::Trace),
            "debug" => Ok(Self::Debug),
            "info" => Ok(Self::Info),
            "warn" | "warning" => Ok(Self::Warn),
            "error" => Ok(Self::Error),
            _ => Err(ParseLogSeverityError {
                value: value.to_string(),
            }),
        }
    }
}

impl Serialize for LogSeverity {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for LogSeverity {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        value.parse().map_err(serde::de::Error::custom)
    }
}

/// OpenTelemetry instrument kind represented by a metric measurement mark.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricKind {
    /// Monotonic additive value.
    Counter,
    /// Additive value that may increase or decrease.
    UpDownCounter,
    /// Current sampled value.
    Gauge,
    /// Distribution sample.
    Histogram,
}

impl MetricKind {
    /// Return the canonical lowercase wire value.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Counter => "counter",
            Self::UpDownCounter => "up_down_counter",
            Self::Gauge => "gauge",
            Self::Histogram => "histogram",
        }
    }
}

impl fmt::Display for MetricKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Numeric representation used by a metric measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MetricValueType {
    /// Unsigned 64-bit integer, restricted to values representable as `i64` on export.
    U64,
    /// Signed 64-bit integer.
    I64,
    /// Finite IEEE 754 double-precision value.
    F64,
}

impl MetricValueType {
    /// Return the canonical lowercase wire value.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::U64 => "u64",
            Self::I64 => "i64",
            Self::F64 => "f64",
        }
    }
}

impl fmt::Display for MetricValueType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// One recording operation in a Relay metric mark.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TypedBuilder)]
#[serde(deny_unknown_fields)]
#[builder(field_defaults(setter(into, strip_option(ignore_invalid, fallback_suffix = "_opt"))))]
pub struct MetricMeasurement {
    /// OpenTelemetry instrument name.
    pub name: String,
    /// Instrument kind used to record the value.
    pub kind: MetricKind,
    /// Explicit numeric representation of `value`.
    pub value_type: MetricValueType,
    /// Numeric recording value.
    pub value: Json,
    /// Optional OpenTelemetry instrument unit.
    #[builder(default)]
    pub unit: Option<String>,
    /// Optional OpenTelemetry instrument description.
    #[builder(default)]
    pub description: Option<String>,
    /// Optional low-cardinality OpenTelemetry attributes object.
    #[builder(default)]
    pub attributes: Option<Json>,
    /// Optional explicit histogram bucket boundaries.
    #[builder(default)]
    pub boundaries: Option<Vec<f64>>,
}

/// Payload stored in a mark using [`METRIC_DATA_SCHEMA_NAME`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TypedBuilder)]
#[serde(deny_unknown_fields)]
#[builder(field_defaults(setter(into)))]
pub struct MetricEnvelope {
    /// Metric recording operations that must be accepted or rejected atomically.
    pub measurements: Vec<MetricMeasurement>,
}

impl MetricEnvelope {
    /// Validate every measurement and their shared instrument descriptors.
    ///
    /// # Errors
    /// Returns a [`MetricValidationError`] when any field violates the Relay
    /// metric schema. The complete envelope must be rejected on error.
    pub fn validate(&self) -> Result<(), MetricValidationError> {
        self.validated_measurements().map(|_| ())
    }

    /// Parse this wire envelope into metric measurements safe for export.
    ///
    /// # Errors
    /// Returns a [`MetricValidationError`] when any field violates the Relay
    /// metric schema. The complete envelope is rejected on error.
    pub fn validated_measurements(
        &self,
    ) -> Result<Vec<ValidatedMetricMeasurement>, MetricValidationError> {
        validate_metric_measurements(&self.measurements)
    }
}

/// Validation error for a Relay metric envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetricValidationError {
    message: String,
}

impl MetricValidationError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Return the detailed validation message.
    pub fn message(&self) -> &str {
        self.message.as_str()
    }
}

impl fmt::Display for MetricValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for MetricValidationError {}

const MAX_HISTOGRAM_BOUNDARIES: usize = 64;

/// A finite IEEE 754 double-precision value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FiniteF64(f64);

impl FiniteF64 {
    /// Return the contained finite value.
    pub const fn get(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for FiniteF64 {
    type Error = MetricValidationError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        value
            .is_finite()
            .then_some(Self(value))
            .ok_or_else(|| MetricValidationError::new("must be finite"))
    }
}

/// An OpenTelemetry instrument name together with its canonical form.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InstrumentName(String);

impl InstrumentName {
    /// Return the original valid instrument name.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Return the case-insensitive canonical instrument name.
    pub fn canonical(&self) -> String {
        self.0.to_ascii_lowercase()
    }
}

impl FromStr for InstrumentName {
    type Err = MetricValidationError;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        let bytes = name.as_bytes();
        let valid = matches!(bytes.first(), Some(first) if first.is_ascii_alphabetic())
            && bytes.len() <= 255
            && bytes[1..].iter().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-' | b'/')
            });
        valid.then(|| Self(name.to_owned())).ok_or_else(|| {
            MetricValidationError::new(
                "name must be 1-255 ASCII bytes, start with a letter, and contain only letters, digits, '_', '.', '-', or '/'",
            )
        })
    }
}

/// One typed metric value accepted by Relay's metric schema.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetricValue {
    /// Unsigned integer value.
    U64(u64),
    /// Signed integer value.
    I64(i64),
    /// Finite floating-point value.
    F64(FiniteF64),
}

impl MetricValue {
    /// Return the matching metric wire value type.
    pub const fn value_type(self) -> MetricValueType {
        match self {
            Self::U64(_) => MetricValueType::U64,
            Self::I64(_) => MetricValueType::I64,
            Self::F64(_) => MetricValueType::F64,
        }
    }

    fn parse(value_type: MetricValueType, value: &Json) -> Result<Self, MetricValidationError> {
        match value_type {
            MetricValueType::U64 => value
                .as_u64()
                .filter(|value| *value <= i64::MAX as u64)
                .map(Self::U64)
                .ok_or_else(|| {
                    MetricValidationError::new(
                        "value must be an unsigned integer no greater than i64::MAX",
                    )
                }),
            MetricValueType::I64 => value
                .as_i64()
                .map(Self::I64)
                .ok_or_else(|| MetricValidationError::new("value must be a signed integer")),
            MetricValueType::F64 => value
                .as_f64()
                .ok_or_else(|| MetricValidationError::new("value must be a number"))
                .and_then(FiniteF64::try_from)
                .map(Self::F64),
        }
    }
}

/// Explicit histogram bucket boundaries that are finite and strictly increasing.
#[derive(Debug, Clone, PartialEq)]
pub struct HistogramBoundaries(Vec<FiniteF64>);

impl HistogramBoundaries {
    /// Return the validated boundaries as floating-point values.
    pub fn values(&self) -> Vec<f64> {
        self.0.iter().map(|boundary| boundary.get()).collect()
    }
}

impl TryFrom<Vec<f64>> for HistogramBoundaries {
    type Error = MetricValidationError;

    fn try_from(boundaries: Vec<f64>) -> Result<Self, Self::Error> {
        if boundaries.len() > MAX_HISTOGRAM_BOUNDARIES {
            return Err(MetricValidationError::new(
                "boundaries must contain at most 64 entries",
            ));
        }
        let boundaries = boundaries
            .into_iter()
            .map(FiniteF64::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        if boundaries
            .windows(2)
            .any(|pair| pair[0].get() >= pair[1].get())
        {
            return Err(MetricValidationError::new(
                "boundaries must be strictly increasing without duplicates",
            ));
        }
        Ok(Self(boundaries))
    }
}

/// Descriptor fields shared by all recordings of an instrument.
#[derive(Debug, Clone, PartialEq)]
pub struct InstrumentDescriptor {
    /// Valid instrument name.
    pub name: InstrumentName,
    /// OpenTelemetry instrument kind.
    pub kind: MetricKind,
    /// Optional OpenTelemetry instrument unit.
    pub unit: Option<String>,
    /// Optional non-identifying description.
    pub description: Option<String>,
    /// Optional non-identifying histogram bucket boundaries.
    pub boundaries: Option<HistogramBoundaries>,
}

impl InstrumentDescriptor {
    fn new(
        name: InstrumentName,
        kind: MetricKind,
        unit: Option<String>,
        description: Option<String>,
        boundaries: Option<HistogramBoundaries>,
    ) -> Result<Self, MetricValidationError> {
        if boundaries.is_some() && kind != MetricKind::Histogram {
            return Err(MetricValidationError::new(
                "boundaries are only valid for histogram measurements",
            ));
        }
        if unit
            .as_ref()
            .is_some_and(|unit| !unit.is_ascii() || unit.len() > 63)
        {
            return Err(MetricValidationError::new(
                "unit must be ASCII and at most 63 bytes",
            ));
        }
        Ok(Self {
            name,
            kind,
            unit,
            description,
            boundaries,
        })
    }

    /// Return the canonical name used to group an instrument within an envelope.
    pub fn descriptor_key(&self) -> String {
        self.name.canonical()
    }

    fn accepts(&self, value: MetricValue) -> Result<(), MetricValidationError> {
        if self.kind == MetricKind::Counter
            && matches!(value, MetricValue::F64(value) if value.get() < 0.0)
        {
            return Err(MetricValidationError::new(
                "counter values must be non-negative",
            ));
        }
        let accepted = matches!(
            (self.kind, value),
            (MetricKind::Counter, MetricValue::U64(_))
                | (MetricKind::UpDownCounter, MetricValue::I64(_))
                | (MetricKind::UpDownCounter, MetricValue::F64(_))
                | (MetricKind::Gauge, _)
                | (MetricKind::Histogram, MetricValue::U64(_))
                | (MetricKind::Histogram, MetricValue::F64(_))
                | (MetricKind::Counter, MetricValue::F64(_))
        );
        accepted.then_some(()).ok_or_else(|| {
            MetricValidationError::new(format!(
                "kind {} does not support value_type {}",
                self.kind,
                value.value_type()
            ))
        })
    }

    fn has_same_identity(&self, other: &Self) -> bool {
        self.kind == other.kind && self.unit == other.unit
    }
}

/// Typed OpenTelemetry attribute values, including homogeneous primitive arrays.
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeValue {
    /// String scalar.
    String(String),
    /// Boolean scalar.
    Bool(bool),
    /// Signed integer scalar.
    I64(i64),
    /// Finite floating-point scalar.
    F64(FiniteF64),
    /// String array.
    StringArray(Vec<String>),
    /// Boolean array.
    BoolArray(Vec<bool>),
    /// Signed integer array.
    I64Array(Vec<i64>),
    /// Finite floating-point array.
    F64Array(Vec<FiniteF64>),
}

/// A deterministic map of typed OpenTelemetry metric attributes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MetricAttributes(BTreeMap<String, AttributeValue>);

impl MetricAttributes {
    /// Insert or replace a typed metric attribute.
    pub fn insert(&mut self, key: String, value: AttributeValue) {
        self.0.insert(key, value);
    }

    /// Iterate over validated attribute name and value pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &AttributeValue)> {
        self.0.iter()
    }

    /// Return whether this attribute set is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl TryFrom<Option<&Json>> for MetricAttributes {
    type Error = MetricValidationError;

    fn try_from(attributes: Option<&Json>) -> Result<Self, Self::Error> {
        let Some(attributes) = attributes else {
            return Ok(Self::default());
        };
        let object = attributes
            .as_object()
            .ok_or_else(|| MetricValidationError::new("attributes must be a JSON object"))?;
        object
            .iter()
            .map(|(key, value)| {
                if key.trim().is_empty() {
                    return Err(MetricValidationError::new(
                        "attributes contains a blank attribute key",
                    ));
                }
                parse_attribute_value(value).map(|value| (key.clone(), value))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()
            .map(Self)
    }
}

fn parse_attribute_value(value: &Json) -> Result<AttributeValue, MetricValidationError> {
    match value {
        Json::String(value) => Ok(AttributeValue::String(value.clone())),
        Json::Bool(value) => Ok(AttributeValue::Bool(*value)),
        Json::Number(number) if number.as_i64().is_some() => Ok(AttributeValue::I64(
            number.as_i64().expect("checked signed integer"),
        )),
        Json::Number(number) if number.as_u64().is_some() => Err(MetricValidationError::new(
            "attribute values must not exceed the maximum signed 64-bit integer",
        )),
        Json::Number(number) => number
            .as_f64()
            .ok_or_else(|| MetricValidationError::new("attribute values must be finite numbers"))
            .and_then(FiniteF64::try_from)
            .map(AttributeValue::F64),
        Json::Array(values) => parse_attribute_array(values),
        Json::Null => Err(MetricValidationError::new(
            "attribute values must not be null",
        )),
        Json::Object(_) => Err(MetricValidationError::new(
            "attribute values must be primitive values or homogeneous primitive arrays",
        )),
    }
}

fn parse_attribute_array(values: &[Json]) -> Result<AttributeValue, MetricValidationError> {
    let Some(first) = values.first() else {
        return Err(MetricValidationError::new(
            "attribute arrays must not be empty and untyped",
        ));
    };
    match parse_attribute_value(first)? {
        AttributeValue::String(_) => values
            .iter()
            .map(|value| value.as_str().map(str::to_owned))
            .collect::<Option<Vec<_>>>()
            .map(AttributeValue::StringArray)
            .ok_or_else(|| {
                MetricValidationError::new("attribute arrays must contain one primitive type")
            }),
        AttributeValue::Bool(_) => values
            .iter()
            .map(Json::as_bool)
            .collect::<Option<Vec<_>>>()
            .map(AttributeValue::BoolArray)
            .ok_or_else(|| {
                MetricValidationError::new("attribute arrays must contain one primitive type")
            }),
        AttributeValue::I64(_) => values
            .iter()
            .map(Json::as_i64)
            .collect::<Option<Vec<_>>>()
            .map(AttributeValue::I64Array)
            .ok_or_else(|| {
                MetricValidationError::new("attribute arrays must contain one primitive type")
            }),
        AttributeValue::F64(_) => values
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .filter(|_| value.as_i64().is_none())
                    .and_then(|value| FiniteF64::try_from(value).ok())
            })
            .collect::<Option<Vec<_>>>()
            .map(AttributeValue::F64Array)
            .ok_or_else(|| {
                MetricValidationError::new("attribute arrays must contain one primitive type")
            }),
        AttributeValue::StringArray(_)
        | AttributeValue::BoolArray(_)
        | AttributeValue::I64Array(_)
        | AttributeValue::F64Array(_) => Err(MetricValidationError::new(
            "attribute arrays must contain primitive values",
        )),
    }
}

/// Parsed metric measurement that is safe for OTLP export.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedMetricMeasurement {
    /// Validated instrument descriptor.
    pub descriptor: InstrumentDescriptor,
    /// Validated typed metric value.
    pub value: MetricValue,
    /// Validated typed metric attributes.
    pub attributes: MetricAttributes,
}

impl TryFrom<&MetricMeasurement> for ValidatedMetricMeasurement {
    type Error = MetricValidationError;

    fn try_from(wire: &MetricMeasurement) -> Result<Self, Self::Error> {
        let descriptor = InstrumentDescriptor::new(
            wire.name.parse()?,
            wire.kind,
            wire.unit.clone(),
            wire.description.clone(),
            wire.boundaries
                .clone()
                .map(HistogramBoundaries::try_from)
                .transpose()?,
        )?;
        let value = MetricValue::parse(wire.value_type, &wire.value)?;
        descriptor.accepts(value)?;
        Ok(Self {
            descriptor,
            value,
            attributes: MetricAttributes::try_from(wire.attributes.as_ref())?,
        })
    }
}

/// Validate a complete list of metric measurements atomically.
///
/// # Errors
/// Returns a [`MetricValidationError`] for the first schema violation. Callers
/// must reject the entire list when validation fails.
pub fn validate_metric_measurements(
    measurements: &[MetricMeasurement],
) -> Result<Vec<ValidatedMetricMeasurement>, MetricValidationError> {
    if measurements.is_empty() {
        return Err(MetricValidationError::new(
            "measurements must contain at least one entry",
        ));
    }

    let parsed = measurements
        .iter()
        .enumerate()
        .map(|(index, measurement)| {
            ValidatedMetricMeasurement::try_from(measurement).map_err(|error| {
                MetricValidationError::new(format!("measurements[{index}] {error}"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut descriptors = BTreeMap::<String, usize>::new();
    for (index, measurement) in parsed.iter().enumerate() {
        let key = measurement.descriptor.descriptor_key();
        if let Some(previous_index) = descriptors.insert(key, index) {
            let previous = &parsed[previous_index];
            if !previous
                .descriptor
                .has_same_identity(&measurement.descriptor)
                || previous.value.value_type() != measurement.value.value_type()
            {
                return Err(MetricValidationError::new(format!(
                    "measurements[{index}] conflicts with the descriptor for measurements[{previous_index}]"
                )));
            }
        }
    }
    Ok(parsed)
}

/// Identifier for the schema that describes an event's opaque `data` payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, TypedBuilder)]
#[builder(field_defaults(setter(into)))]
pub struct DataSchema {
    /// Schema name.
    pub name: String,
    /// Schema version.
    pub version: String,
}

/// Semantic category carried by ATOF `category`.
///
/// This is intentionally string-backed so consumers can preserve category
/// values from newer producers without failing deserialization.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EventCategory(String);

impl EventCategory {
    /// Top-level agent or workflow scope.
    ///
    /// # Returns
    /// An [`EventCategory`] with the wire value `agent`.
    pub fn agent() -> Self {
        Self("agent".into())
    }

    /// Generic function or application step.
    ///
    /// # Returns
    /// An [`EventCategory`] with the wire value `function`.
    pub fn function() -> Self {
        Self("function".into())
    }

    /// LLM call.
    ///
    /// # Returns
    /// An [`EventCategory`] with the wire value `llm`.
    pub fn llm() -> Self {
        Self("llm".into())
    }

    /// Tool invocation.
    ///
    /// # Returns
    /// An [`EventCategory`] with the wire value `tool`.
    pub fn tool() -> Self {
        Self("tool".into())
    }

    /// Retrieval step.
    ///
    /// # Returns
    /// An [`EventCategory`] with the wire value `retriever`.
    pub fn retriever() -> Self {
        Self("retriever".into())
    }

    /// Embedding-generation step.
    ///
    /// # Returns
    /// An [`EventCategory`] with the wire value `embedder`.
    pub fn embedder() -> Self {
        Self("embedder".into())
    }

    /// Result reranking step.
    ///
    /// # Returns
    /// An [`EventCategory`] with the wire value `reranker`.
    pub fn reranker() -> Self {
        Self("reranker".into())
    }

    /// Guardrail or validation step.
    ///
    /// # Returns
    /// An [`EventCategory`] with the wire value `guardrail`.
    pub fn guardrail() -> Self {
        Self("guardrail".into())
    }

    /// Evaluation or scoring step.
    ///
    /// # Returns
    /// An [`EventCategory`] with the wire value `evaluator`.
    pub fn evaluator() -> Self {
        Self("evaluator".into())
    }

    /// Vendor-defined custom category.
    ///
    /// # Returns
    /// An [`EventCategory`] with the wire value `custom`.
    pub fn custom() -> Self {
        Self("custom".into())
    }

    /// Unknown or unclassified work.
    ///
    /// # Returns
    /// An [`EventCategory`] with the wire value `unknown`.
    pub fn unknown() -> Self {
        Self("unknown".into())
    }

    /// Create a category from an arbitrary producer-provided string.
    ///
    /// # Parameters
    /// - `value`: Wire category value to preserve.
    ///
    /// # Returns
    /// An [`EventCategory`] containing `value`.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Return the string form serialized on the wire.
    ///
    /// # Returns
    /// The category value as a string slice.
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    /// Convert this category to the closest legacy scope type for internal
    /// adapters that still need span-kind classification.
    ///
    /// # Returns
    /// The closest matching [`ScopeType`], or [`ScopeType::Unknown`] when the
    /// category has no legacy equivalent.
    pub fn to_scope_type(&self) -> ScopeType {
        match self.as_str() {
            "agent" => ScopeType::Agent,
            "function" => ScopeType::Function,
            "tool" => ScopeType::Tool,
            "llm" => ScopeType::Llm,
            "retriever" => ScopeType::Retriever,
            "embedder" => ScopeType::Embedder,
            "reranker" => ScopeType::Reranker,
            "guardrail" => ScopeType::Guardrail,
            "evaluator" => ScopeType::Evaluator,
            "custom" => ScopeType::Custom,
            _ => ScopeType::Unknown,
        }
    }
}

impl From<ScopeType> for EventCategory {
    fn from(value: ScopeType) -> Self {
        match value {
            ScopeType::Agent => Self::agent(),
            ScopeType::Function => Self::function(),
            ScopeType::Tool => Self::tool(),
            ScopeType::Llm => Self::llm(),
            ScopeType::Retriever => Self::retriever(),
            ScopeType::Embedder => Self::embedder(),
            ScopeType::Reranker => Self::reranker(),
            ScopeType::Guardrail => Self::guardrail(),
            ScopeType::Evaluator => Self::evaluator(),
            ScopeType::Custom => Self::custom(),
            ScopeType::Unknown => Self::unknown(),
        }
    }
}

impl From<&EventCategory> for ScopeType {
    fn from(value: &EventCategory) -> Self {
        value.to_scope_type()
    }
}

/// Agent Trajectory Observability Format (ATOF) lifecycle phase for a scope event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScopeCategory {
    /// Scope was entered.
    Start,
    /// Scope was exited.
    End,
}

/// Category-specific profile data.
///
/// Unknown wire keys are preserved in `extra`. LLM annotations are serialized
/// under `category_profile` when a codec captures them.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, TypedBuilder)]
#[builder(field_defaults(setter(into, strip_option(ignore_invalid, fallback_suffix = "_opt"))))]
pub struct CategoryProfile {
    /// Normalized model identifier for LLM events.
    #[builder(default)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,

    /// LLM-provider correlation ID for Tool events.
    #[builder(default)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,

    /// Vendor subtype required when `category == "custom"`.
    #[builder(default)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtype: Option<String>,

    /// Normalized tool result annotation for successful tool end events.
    #[builder(default)]
    #[serde(default, skip_serializing_if = "option_json_is_none_or_null")]
    pub tool_result_annotation: Option<Json>,

    /// Unknown category-profile keys preserved from newer producers.
    #[builder(default)]
    #[serde(flatten)]
    pub extra: BTreeMap<String, Json>,

    /// Normalized request annotation for LLM start events.
    #[builder(default)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotated_request: Option<Arc<AnnotatedLlmRequest>>,

    /// Normalized response annotation for LLM end events.
    #[builder(default)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotated_response: Option<Arc<AnnotatedLlmResponse>>,
}

impl CategoryProfile {
    /// Return true when the profile has no wire-serialized fields.
    ///
    /// # Returns
    /// `true` when no profile fields would be serialized on the wire.
    pub fn is_wire_empty(&self) -> bool {
        self.model_name.is_none()
            && self.tool_call_id.is_none()
            && self.subtype.is_none()
            && option_json_is_none_or_null(&self.tool_result_annotation)
            && self.annotated_request.is_none()
            && self.annotated_response.is_none()
            && self.extra.is_empty()
    }
}

fn option_json_is_none_or_null(value: &Option<Json>) -> bool {
    value.as_ref().is_none_or(Json::is_null)
}

/// Shared event metadata carried by every ATOF event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TypedBuilder)]
#[builder(field_defaults(setter(into, strip_option(ignore_invalid, fallback_suffix = "_opt"))))]
pub struct BaseEvent {
    /// ATOF protocol version.
    #[builder(default = ATOF_VERSION.to_string())]
    pub atof_version: String,
    /// UUID of the parent scope, if any.
    #[builder(default)]
    pub parent_uuid: Option<Uuid>,
    /// Stable causal root imported or resolved when the event was emitted.
    #[builder(default)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub propagation_root_uuid: Option<Uuid>,
    /// Unique identifier for the event or span.
    #[builder(default = Uuid::now_v7())]
    pub uuid: Uuid,
    /// Event timestamp in UTC.
    #[builder(default = Utc::now())]
    #[serde(with = "timestamp")]
    pub timestamp: DateTime<Utc>,
    /// Human-readable event name.
    pub name: String,
    /// Application-defined payload.
    #[builder(default)]
    pub data: Option<Json>,
    /// Optional schema identifier for `data`.
    #[builder(default)]
    pub data_schema: Option<DataSchema>,
    /// Optional tracing/correlation metadata.
    #[builder(default)]
    pub metadata: Option<Json>,
}

/// ATOF scope lifecycle event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TypedBuilder)]
#[builder(field_defaults(setter(into, strip_option(ignore_invalid, fallback_suffix = "_opt"))))]
pub struct ScopeEvent {
    /// Shared ATOF envelope.
    #[serde(flatten)]
    #[builder(setter(skip), default = BaseEvent::builder().name("").build())]
    pub base: BaseEvent,
    /// Scope lifecycle phase.
    pub scope_category: ScopeCategory,
    /// Canonical lowercase behavioral flags.
    #[builder(default)]
    pub attributes: Vec<String>,
    /// Semantic category of work.
    pub category: EventCategory,
    /// Category-specific typed fields.
    #[builder(default)]
    pub category_profile: Option<CategoryProfile>,
}

impl ScopeEvent {
    /// Construct a scope event from a base envelope and ATOF-specific fields.
    ///
    /// # Parameters
    /// - `base`: Shared ATOF event envelope.
    /// - `scope_category`: Lifecycle phase for the scope event.
    /// - `attributes`: Scope attributes to canonicalize and attach.
    /// - `category`: Semantic event category.
    /// - `category_profile`: Optional category-specific profile data.
    ///
    /// # Returns
    /// A [`ScopeEvent`] containing the provided fields.
    pub fn new(
        base: BaseEvent,
        scope_category: ScopeCategory,
        attributes: Vec<String>,
        category: EventCategory,
        category_profile: Option<CategoryProfile>,
    ) -> Self {
        Self {
            base,
            scope_category,
            attributes: canonicalize_attributes(attributes),
            category,
            category_profile,
        }
    }
}

/// ATOF point-in-time mark event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TypedBuilder)]
#[builder(field_defaults(setter(into, strip_option(ignore_invalid, fallback_suffix = "_opt"))))]
pub struct MarkEvent {
    /// Shared ATOF envelope.
    #[serde(flatten)]
    #[builder(setter(skip), default = BaseEvent::builder().name("").build())]
    pub base: BaseEvent,
    /// Optional semantic category for the checkpoint.
    #[builder(default)]
    pub category: Option<EventCategory>,
    /// Optional category-specific typed fields.
    #[builder(default)]
    pub category_profile: Option<CategoryProfile>,
}

/// Observability fields that event sanitizers may rewrite.
///
/// Event identity and lifecycle fields are intentionally excluded so sanitizer
/// callbacks cannot alter correlation, ordering, or category semantics.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, TypedBuilder)]
#[builder(field_defaults(setter(into, strip_option(ignore_invalid, fallback_suffix = "_opt"))))]
pub struct EventSanitizeFields {
    /// Application-defined event payload.
    #[builder(default)]
    pub data: Option<Json>,
    /// Category-specific typed fields.
    #[builder(default)]
    pub category_profile: Option<CategoryProfile>,
    /// Tracing and correlation metadata.
    #[builder(default)]
    pub metadata: Option<Json>,
}

/// Mark requested by middleware for materialization by a lifecycle owner.
///
/// The runtime assigns the parent UUID, event UUID, and timestamp when it
/// materializes the mark at the appropriate lifecycle boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TypedBuilder)]
#[builder(field_defaults(setter(into, strip_option(ignore_invalid, fallback_suffix = "_opt"))))]
pub struct PendingMarkSpec {
    /// Human-readable mark name.
    pub name: String,
    /// Optional semantic category for the mark.
    #[builder(default)]
    pub category: Option<EventCategory>,
    /// Optional category-specific typed fields.
    #[builder(default)]
    pub category_profile: Option<CategoryProfile>,
    /// Optional application payload attached to the mark.
    #[builder(default)]
    pub data: Option<Json>,
    /// Optional schema identifier for the mark data.
    #[builder(default)]
    pub data_schema: Option<DataSchema>,
    /// Optional metadata attached to the mark.
    #[builder(default)]
    pub metadata: Option<Json>,
    /// Optional typed log severity applied authoritatively to mark metadata.
    #[builder(default)]
    pub severity: Option<LogSeverity>,
}

impl MarkEvent {
    /// Construct a mark event from a base envelope and optional category data.
    ///
    /// # Parameters
    /// - `base`: Shared ATOF event envelope.
    /// - `category`: Optional semantic event category.
    /// - `category_profile`: Optional category-specific profile data.
    ///
    /// # Returns
    /// A [`MarkEvent`] containing the provided fields.
    pub fn new(
        base: BaseEvent,
        category: Option<EventCategory>,
        category_profile: Option<CategoryProfile>,
    ) -> Self {
        Self {
            base,
            category,
            category_profile,
        }
    }
}

/// Tagged union covering the two ATOF event kinds emitted by the runtime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Event {
    /// Scope lifecycle event.
    Scope(ScopeEvent),
    /// Point-in-time checkpoint event.
    Mark(MarkEvent),
}

impl Event {
    /// Return the ATOF event kind.
    ///
    /// # Returns
    /// `"scope"` for [`Event::Scope`] and `"mark"` for [`Event::Mark`].
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Scope(_) => "scope",
            Self::Mark(_) => "mark",
        }
    }

    /// Try to return this event as the canonical JSON object delivered by
    /// language bindings to subscriber callbacks and ATOF exporters.
    pub fn try_to_json_value(&self) -> serde_json::Result<Json> {
        serde_json::to_value(self)
    }

    /// Return this event as the canonical JSON object delivered by language
    /// bindings to subscriber callbacks.
    pub fn to_json_value(&self) -> Json {
        self.try_to_json_value()
            .expect("serializing an ATOF event to JSON should not fail")
    }

    /// Return this event as canonical JSON.
    pub fn to_json_string(&self) -> serde_json::Result<String> {
        serde_json::to_string(&self.try_to_json_value()?)
    }

    /// Return the lifecycle phase for scope events.
    ///
    /// # Returns
    /// `Some` lifecycle phase for scope events, otherwise `None`.
    pub fn scope_category(&self) -> Option<ScopeCategory> {
        match self {
            Self::Scope(event) => Some(event.scope_category),
            Self::Mark(_) => None,
        }
    }

    /// Return the semantic category if present.
    ///
    /// # Returns
    /// `Some` category for scope events and categorized mark events, otherwise
    /// `None`.
    pub fn category(&self) -> Option<&EventCategory> {
        match self {
            Self::Scope(event) => Some(&event.category),
            Self::Mark(event) => event.category.as_ref(),
        }
    }

    /// Return the category-specific profile if present.
    ///
    /// # Returns
    /// `Some` profile when category-specific fields are present.
    pub fn category_profile(&self) -> Option<&CategoryProfile> {
        match self {
            Self::Scope(event) => event.category_profile.as_ref(),
            Self::Mark(event) => event.category_profile.as_ref(),
        }
    }

    /// Return the mutable category-specific profile if present.
    ///
    /// # Returns
    /// `Some` mutable profile when category-specific fields are present.
    pub fn category_profile_mut(&mut self) -> Option<&mut CategoryProfile> {
        match self {
            Self::Scope(event) => event.category_profile.as_mut(),
            Self::Mark(event) => event.category_profile.as_mut(),
        }
    }

    /// Return an owned copy of the normalized tool result annotation.
    ///
    /// JSON null is normalized to absence.
    pub fn tool_result_annotation(&self) -> Option<Json> {
        self.category_profile()?
            .tool_result_annotation
            .as_ref()
            .filter(|value| !value.is_null())
            .cloned()
    }

    /// Return the parent scope UUID, if the event is nested under a scope.
    ///
    /// # Returns
    /// `Some` parent UUID when the event has a parent scope, otherwise `None`.
    pub fn parent_uuid(&self) -> Option<Uuid> {
        self.base().parent_uuid
    }

    /// Return the stable causal root captured when this event was emitted.
    pub fn propagation_root_uuid(&self) -> Option<Uuid> {
        self.base().propagation_root_uuid
    }

    /// Attach the causal root captured at event emission.
    pub fn set_propagation_root_uuid(&mut self, root_uuid: Option<Uuid>) {
        self.base_mut().propagation_root_uuid = root_uuid;
    }

    /// Return the unique event or span UUID.
    ///
    /// # Returns
    /// The event UUID.
    pub fn uuid(&self) -> Uuid {
        self.base().uuid
    }

    /// Return the event timestamp.
    ///
    /// # Returns
    /// The UTC event timestamp.
    pub fn timestamp(&self) -> &DateTime<Utc> {
        &self.base().timestamp
    }

    /// Return the human-readable event name.
    ///
    /// # Returns
    /// The event name.
    pub fn name(&self) -> &str {
        self.base().name.as_str()
    }

    /// Return the optional application payload attached to the event.
    ///
    /// # Returns
    /// `Some` payload when event data is present, otherwise `None`.
    pub fn data(&self) -> Option<&Json> {
        self.base().data.as_ref()
    }

    /// Snapshot the observability fields that sanitizers may rewrite.
    pub fn sanitize_fields(&self) -> EventSanitizeFields {
        EventSanitizeFields {
            data: self.base().data.clone(),
            category_profile: self.category_profile().cloned(),
            metadata: self.base().metadata.clone(),
        }
    }

    /// Replace the observability fields controlled by event sanitizers.
    pub fn apply_sanitize_fields(&mut self, fields: EventSanitizeFields) {
        self.base_mut().data = fields.data;
        self.base_mut().metadata = fields.metadata;
        match self {
            Self::Scope(event) => event.category_profile = fields.category_profile,
            Self::Mark(event) => event.category_profile = fields.category_profile,
        }
    }

    /// Return the optional data schema.
    ///
    /// # Returns
    /// `Some` schema when the event payload declares one, otherwise `None`.
    pub fn data_schema(&self) -> Option<&DataSchema> {
        self.base().data_schema.as_ref()
    }

    /// Return the optional metadata attached to the event.
    ///
    /// # Returns
    /// `Some` metadata when present, otherwise `None`.
    pub fn metadata(&self) -> Option<&Json> {
        self.base().metadata.as_ref()
    }

    /// Return attributes for scope events.
    ///
    /// # Returns
    /// `Some` attributes for scope events, otherwise `None`.
    pub fn attributes(&self) -> Option<&[String]> {
        match self {
            Self::Scope(event) => Some(event.attributes.as_slice()),
            Self::Mark(_) => None,
        }
    }

    /// Return the semantic scope category for scope events.
    ///
    /// # Returns
    /// `Some` legacy [`ScopeType`] when the event has a category.
    pub fn scope_type(&self) -> Option<ScopeType> {
        self.category().map(EventCategory::to_scope_type)
    }

    /// Return the semantic input payload for start events.
    ///
    /// # Returns
    /// `Some` payload for scope-start events with data, otherwise `None`.
    pub fn input(&self) -> Option<&Json> {
        match self {
            Self::Scope(event) if event.scope_category == ScopeCategory::Start => {
                event.base.data.as_ref()
            }
            _ => None,
        }
    }

    /// Return the semantic output payload for end events.
    ///
    /// # Returns
    /// `Some` payload for scope-end events with data, otherwise `None`.
    pub fn output(&self) -> Option<&Json> {
        match self {
            Self::Scope(event) if event.scope_category == ScopeCategory::End => {
                event.base.data.as_ref()
            }
            _ => None,
        }
    }

    /// Return the normalized model name for LLM events.
    ///
    /// # Returns
    /// `Some` model name when the event profile includes one.
    pub fn model_name(&self) -> Option<&str> {
        self.category_profile()
            .and_then(|profile| profile.model_name.as_deref())
    }

    /// Return the provider-specific tool-call correlation identifier.
    ///
    /// # Returns
    /// `Some` tool call identifier when the event profile includes one.
    pub fn tool_call_id(&self) -> Option<&str> {
        self.category_profile()
            .and_then(|profile| profile.tool_call_id.as_deref())
    }

    /// Return the runtime-only annotated LLM request.
    ///
    /// # Returns
    /// `Some` annotated request when the event profile includes one.
    pub fn annotated_request(&self) -> Option<&Arc<AnnotatedLlmRequest>> {
        self.category_profile()
            .and_then(|profile| profile.annotated_request.as_ref())
    }

    /// Return the runtime-only annotated LLM response.
    ///
    /// # Returns
    /// `Some` annotated response when the event profile includes one.
    pub fn annotated_response(&self) -> Option<&Arc<AnnotatedLlmResponse>> {
        self.category_profile()
            .and_then(|profile| profile.annotated_response.as_ref())
    }

    /// Return true for scope-start events.
    ///
    /// # Returns
    /// `true` when the event is a scope-start event.
    pub fn is_scope_start(&self) -> bool {
        matches!(
            self,
            Self::Scope(ScopeEvent {
                scope_category: ScopeCategory::Start,
                ..
            })
        )
    }

    /// Return true for scope-end events.
    ///
    /// # Returns
    /// `true` when the event is a scope-end event.
    pub fn is_scope_end(&self) -> bool {
        matches!(
            self,
            Self::Scope(ScopeEvent {
                scope_category: ScopeCategory::End,
                ..
            })
        )
    }

    fn base(&self) -> &BaseEvent {
        match self {
            Self::Scope(event) => &event.base,
            Self::Mark(event) => &event.base,
        }
    }

    fn base_mut(&mut self) -> &mut BaseEvent {
        match self {
            Self::Scope(event) => &mut event.base,
            Self::Mark(event) => &mut event.base,
        }
    }
}

/// Convert handle bitflags into ATOF attributes.
///
/// # Parameters
/// - `attributes`: Handle-specific attribute bitflags.
///
/// # Returns
/// Canonical lowercase ATOF attribute strings for the provided bitflags.
pub fn attributes_from_handle(attributes: HandleAttributes) -> Vec<String> {
    match attributes {
        HandleAttributes::Scope(attributes) => scope_attributes_to_strings(attributes),
        HandleAttributes::Tool(attributes) => tool_attributes_to_strings(attributes),
        HandleAttributes::Llm(attributes) => llm_attributes_to_strings(attributes),
    }
}

/// Convert scope bitflags into ATOF attributes.
///
/// # Parameters
/// - `attributes`: Scope attribute bitflags.
///
/// # Returns
/// Canonical lowercase ATOF attribute strings for the provided bitflags.
pub fn scope_attributes_to_strings(attributes: ScopeAttributes) -> Vec<String> {
    let mut values = Vec::new();
    if attributes.contains(ScopeAttributes::PARALLEL) {
        values.push("parallel".to_string());
    }
    if attributes.contains(ScopeAttributes::RELOCATABLE) {
        values.push("relocatable".to_string());
    }
    values
}

/// Convert tool bitflags into ATOF attributes.
///
/// # Parameters
/// - `attributes`: Tool attribute bitflags.
///
/// # Returns
/// Canonical lowercase ATOF attribute strings for the provided bitflags.
pub fn tool_attributes_to_strings(attributes: ToolAttributes) -> Vec<String> {
    let mut values = Vec::new();
    if attributes.contains(ToolAttributes::REMOTE) {
        values.push("remote".to_string());
    }
    values
}

/// Convert LLM bitflags into ATOF attributes.
///
/// # Parameters
/// - `attributes`: LLM attribute bitflags.
///
/// # Returns
/// Canonical lowercase ATOF attribute strings for the provided bitflags.
pub fn llm_attributes_to_strings(attributes: LlmAttributes) -> Vec<String> {
    let mut values = Vec::new();
    if attributes.contains(LlmAttributes::STATEFUL) {
        values.push("stateful".to_string());
    }
    if attributes.contains(LlmAttributes::STREAMING) {
        values.push("streaming".to_string());
    }
    values
}

fn canonicalize_attributes(mut attributes: Vec<String>) -> Vec<String> {
    attributes.sort();
    attributes.dedup();
    attributes
}

mod timestamp {
    use chrono::{DateTime, Utc};
    use serde::{
        Deserializer, Serializer,
        de::{self, Visitor},
    };
    use std::fmt;

    /// Serialize a UTC timestamp as RFC 3339.
    ///
    /// # Parameters
    /// - `value`: Timestamp to serialize.
    /// - `serializer`: Serde serializer receiving the string value.
    ///
    /// # Returns
    /// The serializer's success value.
    ///
    /// # Errors
    /// Returns any error produced by the serializer.
    pub fn serialize<S>(value: &DateTime<Utc>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&value.to_rfc3339())
    }

    /// Deserialize a UTC timestamp from an RFC 3339 string.
    ///
    /// # Parameters
    /// - `deserializer`: Serde deserializer providing the timestamp value.
    ///
    /// # Returns
    /// Parsed UTC timestamp.
    ///
    /// # Errors
    /// Returns a serde error when the input is not a valid RFC 3339 timestamp.
    pub fn deserialize<'de, D>(deserializer: D) -> Result<DateTime<Utc>, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(TimestampVisitor)
    }

    struct TimestampVisitor;

    impl<'de> Visitor<'de> for TimestampVisitor {
        type Value = DateTime<Utc>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("an RFC 3339 timestamp string or epoch microseconds integer")
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            DateTime::parse_from_rfc3339(value)
                .map(|timestamp| timestamp.with_timezone(&Utc))
                .map_err(E::custom)
        }

        fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            DateTime::<Utc>::from_timestamp_micros(value)
                .ok_or_else(|| E::custom("epoch microseconds value is out of range"))
        }

        fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            let value = i64::try_from(value)
                .map_err(|_| E::custom("epoch microseconds value is out of range"))?;
            self.visit_i64(value)
        }
    }
}
