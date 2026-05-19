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
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;
use uuid::Uuid;

use crate::api::llm::LlmAttributes;
use crate::api::scope::{HandleAttributes, ScopeAttributes, ScopeType};
use crate::api::tool::ToolAttributes;
use crate::codec::request::AnnotatedLlmRequest;
use crate::codec::response::AnnotatedLlmResponse;
use crate::json::Json;

/// Agent Trajectory Observability Format (ATOF) protocol version emitted by this runtime.
pub const ATOF_VERSION: &str = "0.1";

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
    pub fn agent() -> Self {
        Self("agent".into())
    }

    /// Generic function or application step.
    pub fn function() -> Self {
        Self("function".into())
    }

    /// LLM call.
    pub fn llm() -> Self {
        Self("llm".into())
    }

    /// Tool invocation.
    pub fn tool() -> Self {
        Self("tool".into())
    }

    /// Retrieval step.
    pub fn retriever() -> Self {
        Self("retriever".into())
    }

    /// Embedding-generation step.
    pub fn embedder() -> Self {
        Self("embedder".into())
    }

    /// Result reranking step.
    pub fn reranker() -> Self {
        Self("reranker".into())
    }

    /// Guardrail or validation step.
    pub fn guardrail() -> Self {
        Self("guardrail".into())
    }

    /// Evaluation or scoring step.
    pub fn evaluator() -> Self {
        Self("evaluator".into())
    }

    /// Vendor-defined custom category.
    pub fn custom() -> Self {
        Self("custom".into())
    }

    /// Unknown or unclassified work.
    pub fn unknown() -> Self {
        Self("unknown".into())
    }

    /// Create a category from an arbitrary producer-provided string.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Return the string form serialized on the wire.
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    /// Convert this category to the closest legacy scope type for internal
    /// adapters that still need span-kind classification.
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
/// Unknown wire keys are preserved in `extra`. LLM annotations are runtime-only
/// enrichment used by internal adaptive and Agent Trajectory Interchange Format
/// (ATIF) logic and are never serialized.
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
    pub fn is_wire_empty(&self) -> bool {
        self.model_name.is_none()
            && self.tool_call_id.is_none()
            && self.subtype.is_none()
            && self.extra.is_empty()
    }
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

impl MarkEvent {
    /// Construct a mark event from a base envelope and optional category data.
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
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Scope(_) => "scope",
            Self::Mark(_) => "mark",
        }
    }

    /// Return this event as the canonical JSON object delivered by language
    /// bindings to subscriber callbacks.
    pub fn to_json_value(&self) -> Json {
        let mut value = serde_json::to_value(self).unwrap_or(Json::Null);
        if let Json::Object(ref mut object) = value {
            if let Some(request) = self.annotated_request() {
                object.insert(
                    "annotated_request".to_string(),
                    serde_json::to_value(request.as_ref()).unwrap_or(Json::Null),
                );
            }
            if let Some(response) = self.annotated_response() {
                object.insert(
                    "annotated_response".to_string(),
                    serde_json::to_value(response.as_ref()).unwrap_or(Json::Null),
                );
            }
        }
        value
    }

    /// Return this event as canonical JSON.
    pub fn to_json_string(&self) -> serde_json::Result<String> {
        serde_json::to_string(&self.to_json_value())
    }

    /// Return the lifecycle phase for scope events.
    pub fn scope_category(&self) -> Option<ScopeCategory> {
        match self {
            Self::Scope(event) => Some(event.scope_category),
            Self::Mark(_) => None,
        }
    }

    /// Return the semantic category if present.
    pub fn category(&self) -> Option<&EventCategory> {
        match self {
            Self::Scope(event) => Some(&event.category),
            Self::Mark(event) => event.category.as_ref(),
        }
    }

    /// Return the category-specific profile if present.
    pub fn category_profile(&self) -> Option<&CategoryProfile> {
        match self {
            Self::Scope(event) => event.category_profile.as_ref(),
            Self::Mark(event) => event.category_profile.as_ref(),
        }
    }

    /// Return the mutable category-specific profile if present.
    pub fn category_profile_mut(&mut self) -> Option<&mut CategoryProfile> {
        match self {
            Self::Scope(event) => event.category_profile.as_mut(),
            Self::Mark(event) => event.category_profile.as_mut(),
        }
    }

    /// Return the parent scope UUID, if the event is nested under a scope.
    pub fn parent_uuid(&self) -> Option<Uuid> {
        self.base().parent_uuid
    }

    /// Return the unique event or span UUID.
    pub fn uuid(&self) -> Uuid {
        self.base().uuid
    }

    /// Return the event timestamp.
    pub fn timestamp(&self) -> &DateTime<Utc> {
        &self.base().timestamp
    }

    /// Return the human-readable event name.
    pub fn name(&self) -> &str {
        self.base().name.as_str()
    }

    /// Return the optional application payload attached to the event.
    pub fn data(&self) -> Option<&Json> {
        self.base().data.as_ref()
    }

    /// Return the optional data schema.
    pub fn data_schema(&self) -> Option<&DataSchema> {
        self.base().data_schema.as_ref()
    }

    /// Return the optional metadata attached to the event.
    pub fn metadata(&self) -> Option<&Json> {
        self.base().metadata.as_ref()
    }

    /// Return attributes for scope events.
    pub fn attributes(&self) -> Option<&[String]> {
        match self {
            Self::Scope(event) => Some(event.attributes.as_slice()),
            Self::Mark(_) => None,
        }
    }

    /// Return the semantic scope category for scope events.
    pub fn scope_type(&self) -> Option<ScopeType> {
        self.category().map(EventCategory::to_scope_type)
    }

    /// Return the semantic input payload for start events.
    pub fn input(&self) -> Option<&Json> {
        match self {
            Self::Scope(event) if event.scope_category == ScopeCategory::Start => {
                event.base.data.as_ref()
            }
            _ => None,
        }
    }

    /// Return the semantic output payload for end events.
    pub fn output(&self) -> Option<&Json> {
        match self {
            Self::Scope(event) if event.scope_category == ScopeCategory::End => {
                event.base.data.as_ref()
            }
            _ => None,
        }
    }

    /// Return the normalized model name for LLM events.
    pub fn model_name(&self) -> Option<&str> {
        self.category_profile()
            .and_then(|profile| profile.model_name.as_deref())
    }

    /// Return the provider-specific tool-call correlation identifier.
    pub fn tool_call_id(&self) -> Option<&str> {
        self.category_profile()
            .and_then(|profile| profile.tool_call_id.as_deref())
    }

    /// Return the runtime-only annotated LLM request.
    pub fn annotated_request(&self) -> Option<&Arc<AnnotatedLlmRequest>> {
        self.category_profile()
            .and_then(|profile| profile.annotated_request.as_ref())
    }

    /// Return the runtime-only annotated LLM response.
    pub fn annotated_response(&self) -> Option<&Arc<AnnotatedLlmResponse>> {
        self.category_profile()
            .and_then(|profile| profile.annotated_response.as_ref())
    }

    /// Return true for scope-start events.
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
}

/// Convert handle bitflags into ATOF attributes.
pub fn attributes_from_handle(attributes: HandleAttributes) -> Vec<String> {
    match attributes {
        HandleAttributes::Scope(attributes) => scope_attributes_to_strings(attributes),
        HandleAttributes::Tool(attributes) => tool_attributes_to_strings(attributes),
        HandleAttributes::Llm(attributes) => llm_attributes_to_strings(attributes),
    }
}

/// Convert scope bitflags into ATOF attributes.
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
pub fn tool_attributes_to_strings(attributes: ToolAttributes) -> Vec<String> {
    let mut values = Vec::new();
    if attributes.contains(ToolAttributes::REMOTE) {
        values.push("remote".to_string());
    }
    values
}

/// Convert LLM bitflags into ATOF attributes.
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

    pub fn serialize<S>(value: &DateTime<Utc>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&value.to_rfc3339())
    }

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
