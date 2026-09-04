// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed, fail-closed removal of conversational trajectory content.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde_json::{Map, Value as Json};

use nemo_relay::api::event::{
    CategoryProfile, Event, LOG_SEVERITY_METADATA_KEY, LogSeverity, METRIC_DATA_SCHEMA_NAME,
    METRIC_DATA_SCHEMA_VERSION, MetricEnvelope,
};
use nemo_relay::codec::optimization::LlmOptimizationSummary;
use nemo_relay::codec::request::{
    AnnotatedLlmRequest, ApiSpecificRequest, ContentPart, FunctionCall, FunctionDefinition,
    Message, MessageContent, OpenAiImageUrl, ProviderNativeComponent, ToolCall, ToolChoice,
    ToolChoiceFunction, ToolChoiceFunctionName, ToolDefinition,
};
use nemo_relay::codec::response::{
    AnnotatedLlmResponse, ApiSpecificResponse, CostEstimate, FinishReason, ResponseToolCall, Usage,
};

const PROVIDER_ATTRIBUTE: &str = "gen_ai.provider.name";
const KNOWN_MARK_SUBTYPES: &[&str] = &["llm.chunk", "nemo_relay.llm.optimization", "skill.load"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CustomMarkPayloadPolicy {
    Preserve,
    RedactAllLeaves,
}

impl CustomMarkPayloadPolicy {
    pub(super) fn parse(value: &str) -> Option<Self> {
        match value {
            "preserve" => Some(Self::Preserve),
            "redact_all_leaves" => Some(Self::RedactAllLeaves),
            _ => None,
        }
    }
}

#[derive(Clone)]
pub(super) struct TrajectorySanitizer {
    replacement: Arc<String>,
    custom_mark_payload_policy: CustomMarkPayloadPolicy,
    metric_string_attribute_allowlist: Arc<BTreeMap<String, BTreeSet<String>>>,
}

impl TrajectorySanitizer {
    pub(super) fn new(
        replacement: String,
        policy: CustomMarkPayloadPolicy,
        metric_string_attribute_allowlist: BTreeMap<String, Vec<String>>,
    ) -> Self {
        Self {
            replacement: Arc::new(replacement),
            custom_mark_payload_policy: policy,
            metric_string_attribute_allowlist: Arc::new(
                metric_string_attribute_allowlist
                    .into_iter()
                    .map(|(attribute, values)| (attribute, values.into_iter().collect()))
                    .collect(),
            ),
        }
    }

    pub(super) fn sanitize_tool_payload(&self, _value: Json) -> Json {
        empty_object()
    }

    pub(super) fn sanitize_provider_payload(&self, _value: Json) -> Json {
        empty_object()
    }

    pub(super) fn sanitize_annotated_request(
        &self,
        mut request: AnnotatedLlmRequest,
    ) -> Option<AnnotatedLlmRequest> {
        request.messages = request
            .messages
            .into_iter()
            .map(|message| sanitize_message(message, &self.replacement))
            .collect();
        request.instructions = request
            .instructions
            .map(|content| sanitize_message_content(content, &self.replacement));
        if let Some(params) = request.params.as_mut() {
            params.stop = params.stop.take().map(|values| {
                values
                    .into_iter()
                    .map(|_| (*self.replacement).clone())
                    .collect()
            });
        }
        request.tools = request.tools.map(|tools| {
            tools
                .into_iter()
                .map(|tool| sanitize_tool_definition(tool, &self.replacement))
                .collect()
        });
        request.tool_choice = request
            .tool_choice
            .map(|choice| sanitize_tool_choice(choice, &self.replacement));
        replace_optional_string(&mut request.previous_response_id, &self.replacement);
        replace_optional_json(&mut request.truncation);
        replace_optional_json(&mut request.reasoning);
        replace_optional_json(&mut request.include);
        replace_optional_string(&mut request.user, &self.replacement);
        replace_optional_json(&mut request.metadata);
        request.service_tier = None;
        request.api_specific = request
            .api_specific
            .map(|specific| sanitize_api_specific_request(specific, &self.replacement));
        request.extra.clear();
        Some(request)
    }

    pub(super) fn sanitize_annotated_response(
        &self,
        mut response: AnnotatedLlmResponse,
    ) -> Option<AnnotatedLlmResponse> {
        replace_optional_string(&mut response.id, &self.replacement);
        response.message = response
            .message
            .map(|content| sanitize_message_content(content, &self.replacement));
        response.tool_calls = response.tool_calls.map(|calls| {
            calls
                .into_iter()
                .map(|call| sanitize_response_tool_call(call, &self.replacement))
                .collect()
        });
        if let Some(FinishReason::Unknown(value)) = response.finish_reason.as_mut() {
            *value = (*self.replacement).clone();
        }
        response.usage = response
            .usage
            .map(|usage| sanitize_usage(usage, &self.replacement));
        response.optimization_summary = response
            .optimization_summary
            .map(|summary| sanitize_optimization_summary(summary, &self.replacement));
        response.api_specific = response
            .api_specific
            .map(|specific| sanitize_api_specific_response(specific, &self.replacement));
        response.extra.clear();
        Some(response)
    }

    pub(super) fn sanitize_event_fields(
        &self,
        event: &Event,
        mut fields: nemo_relay::api::event::EventSanitizeFields,
    ) -> nemo_relay::api::event::EventSanitizeFields {
        let log_severity = valid_mark_log_severity(event, fields.metadata.as_ref());
        let category = event.category().map(|category| category.as_str());
        let unknown_custom_mark = matches!(event, Event::Mark(_))
            && category == Some("custom")
            && !is_known_mark_subtype(event.name());

        if unknown_custom_mark
            && self.custom_mark_payload_policy == CustomMarkPayloadPolicy::Preserve
        {
            return fields;
        }

        if is_relay_metric_mark(event) {
            fields.data = fields
                .data
                .and_then(|data| self.sanitize_metric_envelope(data));
        } else {
            fields.data = fields.data.map(|_| empty_object());
        }
        fields.metadata = fields.metadata.map(|_| empty_object());

        let provider = (category == Some("llm"))
            .then(|| provider_name(event))
            .flatten();
        fields.category_profile = fields
            .category_profile
            .map(|profile| sanitize_category_profile(profile, self));
        if let Some(provider) = provider {
            let profile = fields
                .category_profile
                .get_or_insert_with(CategoryProfile::default);
            profile
                .extra
                .insert(PROVIDER_ATTRIBUTE.to_string(), Json::String(provider));
        }

        restore_log_severity(fields, log_severity)
    }

    /// Redact optional metric text and reject every attribute that is not explicitly allowed.
    fn sanitize_metric_envelope(&self, data: Json) -> Option<Json> {
        let mut envelope = match serde_json::from_value::<MetricEnvelope>(data) {
            Ok(envelope) => envelope,
            Err(_) => {
                return log_metric_envelope_omitted("metric envelope deserialization failure");
            }
        };
        if envelope.validate().is_err() {
            return log_metric_envelope_omitted(
                "metric envelope validation failure before redaction",
            );
        }
        for measurement in &mut envelope.measurements {
            measurement.description = measurement
                .description
                .take()
                .map(|_| (*self.replacement).clone());
            measurement.attributes = measurement.attributes.take().map(|attributes| {
                retain_allowed_metric_string_attributes(
                    attributes,
                    &self.metric_string_attribute_allowlist,
                )
            });
        }
        if envelope.validate().is_err() {
            return log_metric_envelope_omitted(
                "metric envelope validation failure after redaction",
            );
        }
        match serde_json::to_value(envelope) {
            Ok(data) => Some(data),
            Err(_) => log_metric_envelope_omitted("metric envelope serialization failure"),
        }
    }
}

fn log_metric_envelope_omitted(reason: &str) -> Option<Json> {
    log::warn!(
        target: "nemo_relay.plugin",
        event = "pii_metric_envelope_omitted",
        reason;
        "PII redaction omitted a metric envelope"
    );
    None
}

fn empty_object() -> Json {
    Json::Object(Map::new())
}

fn replace_optional_json(value: &mut Option<Json>) {
    if value.is_some() {
        *value = Some(empty_object());
    }
}

fn replace_optional_string(value: &mut Option<String>, replacement: &str) {
    if value.is_some() {
        *value = Some(replacement.to_string());
    }
}

fn sanitize_message(message: Message, replacement: &str) -> Message {
    match message {
        Message::System { content, name } => Message::System {
            content: sanitize_message_content(content, replacement),
            name: name.map(|_| replacement.to_string()),
        },
        Message::User { content, name } => Message::User {
            content: sanitize_message_content(content, replacement),
            name: name.map(|_| replacement.to_string()),
        },
        Message::Developer { content, name } => Message::Developer {
            content: sanitize_message_content(content, replacement),
            name: name.map(|_| replacement.to_string()),
        },
        Message::Assistant {
            content,
            tool_calls,
            name,
        } => Message::Assistant {
            content: content.map(|content| sanitize_message_content(content, replacement)),
            tool_calls: tool_calls.map(|calls| {
                calls
                    .into_iter()
                    .map(|call| sanitize_tool_call(call, replacement))
                    .collect()
            }),
            name: name.map(|_| replacement.to_string()),
        },
        Message::Tool { content, .. } => Message::Tool {
            content: sanitize_message_content(content, replacement),
            tool_call_id: replacement.to_string(),
        },
        Message::Function { content, name } => Message::Function {
            content: content.map(|_| replacement.to_string()),
            name,
        },
        Message::ToolCallItem { id, name, .. } => Message::ToolCallItem {
            id: id.map(|_| replacement.to_string()),
            call_id: replacement.to_string(),
            name,
            arguments: empty_object(),
            extra: Map::new(),
        },
        Message::ToolResultItem { id, .. } => Message::ToolResultItem {
            id: id.map(|_| replacement.to_string()),
            call_id: replacement.to_string(),
            output: empty_object(),
            extra: Map::new(),
        },
        Message::ProviderNative { provider, kind, .. } => Message::ProviderNative {
            provider,
            kind: sanitize_native_kind(kind, replacement),
            value: empty_object(),
        },
    }
}

fn sanitize_message_content(content: MessageContent, replacement: &str) -> MessageContent {
    match content {
        MessageContent::Text(_) => MessageContent::Text(replacement.to_string()),
        MessageContent::Parts(parts) => MessageContent::Parts(
            parts
                .into_iter()
                .map(|part| sanitize_content_part(part, replacement))
                .collect(),
        ),
    }
}

fn sanitize_content_part(part: ContentPart, replacement: &str) -> ContentPart {
    match part {
        ContentPart::Text { .. } => ContentPart::Text {
            text: replacement.to_string(),
            extra: Map::new(),
        },
        ContentPart::ImageUrl { image_url, .. } => ContentPart::ImageUrl {
            image_url: sanitize_image_url(image_url, replacement),
            extra: Map::new(),
        },
        ContentPart::Image { .. } => ContentPart::Image {
            image: empty_object(),
            extra: Map::new(),
        },
        ContentPart::Audio { .. } => ContentPart::Audio {
            audio: empty_object(),
            extra: Map::new(),
        },
        ContentPart::File { .. } => ContentPart::File {
            file: empty_object(),
            extra: Map::new(),
        },
        ContentPart::Refusal { .. } => ContentPart::Refusal {
            refusal: replacement.to_string(),
            extra: Map::new(),
        },
        ContentPart::ToolUse { name, .. } => ContentPart::ToolUse {
            id: replacement.to_string(),
            name,
            input: empty_object(),
            extra: Map::new(),
        },
        ContentPart::ToolResult { is_error, .. } => ContentPart::ToolResult {
            tool_use_id: replacement.to_string(),
            content: empty_object(),
            is_error: is_error.map(|_| false),
            extra: Map::new(),
        },
        ContentPart::ProviderNative { provider, kind, .. } => ContentPart::ProviderNative {
            provider,
            kind: sanitize_native_kind(kind, replacement),
            value: empty_object(),
        },
    }
}

fn sanitize_image_url(image_url: OpenAiImageUrl, replacement: &str) -> OpenAiImageUrl {
    OpenAiImageUrl {
        url: replacement.to_string(),
        detail: image_url
            .detail
            .map(|detail| preserve_known_string(detail, &["auto", "low", "high"], replacement)),
    }
}

fn sanitize_tool_call(call: ToolCall, replacement: &str) -> ToolCall {
    ToolCall {
        id: replacement.to_string(),
        call_type: preserve_known_string(call.call_type, &["function"], replacement),
        function: sanitize_function_call(call.function, replacement),
    }
}

fn sanitize_function_call(call: FunctionCall, _replacement: &str) -> FunctionCall {
    FunctionCall {
        name: call.name,
        arguments: empty_object().to_string(),
    }
}

fn sanitize_tool_definition(tool: ToolDefinition, replacement: &str) -> ToolDefinition {
    match tool {
        ToolDefinition::Function { function, .. } => ToolDefinition::Function {
            function: sanitize_function_definition(function, replacement),
            extra: Map::new(),
        },
        ToolDefinition::ProviderNative { provider, kind, .. } => ToolDefinition::ProviderNative {
            provider,
            kind: sanitize_native_kind(kind, replacement),
            value: empty_object(),
        },
    }
}

fn sanitize_function_definition(
    function: FunctionDefinition,
    replacement: &str,
) -> FunctionDefinition {
    FunctionDefinition {
        name: function.name,
        description: function.description.map(|_| replacement.to_string()),
        parameters: function.parameters.map(|_| empty_object()),
        strict: function.strict.map(|_| false),
        extra: Map::new(),
    }
}

fn sanitize_tool_choice(choice: ToolChoice, replacement: &str) -> ToolChoice {
    match choice {
        ToolChoice::Auto => ToolChoice::Auto,
        ToolChoice::None => ToolChoice::None,
        ToolChoice::Required => ToolChoice::Required,
        ToolChoice::Specific(ToolChoiceFunction {
            choice_type,
            function: ToolChoiceFunctionName { name },
        }) => ToolChoice::Specific(ToolChoiceFunction {
            choice_type: preserve_known_string(choice_type, &["function"], replacement),
            function: ToolChoiceFunctionName { name },
        }),
        ToolChoice::ProviderNative(ProviderNativeComponent { provider, kind, .. }) => {
            ToolChoice::ProviderNative(ProviderNativeComponent {
                provider,
                kind: sanitize_native_kind(kind, replacement),
                value: empty_object(),
            })
        }
    }
}

fn sanitize_api_specific_request(
    request: ApiSpecificRequest,
    replacement: &str,
) -> ApiSpecificRequest {
    match request {
        ApiSpecificRequest::AnthropicMessages {
            cache_control,
            container,
            inference_geo,
            output_config,
            thinking,
            top_k,
            user_profile_id,
        } => ApiSpecificRequest::AnthropicMessages {
            cache_control: opaque_option(cache_control),
            container: marked_option(container, replacement),
            inference_geo: marked_option(inference_geo, replacement),
            output_config: opaque_option(output_config),
            thinking: opaque_option(thinking),
            top_k,
            user_profile_id: marked_option(user_profile_id, replacement),
        },
        ApiSpecificRequest::OpenAIChat {
            audio,
            frequency_penalty,
            function_call,
            functions,
            logit_bias,
            logprobs,
            modalities,
            moderation,
            n,
            prediction,
            presence_penalty,
            prompt_cache_key,
            prompt_cache_options,
            prompt_cache_retention,
            reasoning_effort,
            response_format,
            safety_identifier,
            seed,
            stream_options,
            verbosity,
            web_search_options,
        } => ApiSpecificRequest::OpenAIChat {
            audio: opaque_option(audio),
            frequency_penalty,
            function_call: opaque_option(function_call),
            functions: functions.map(|values| values.into_iter().map(|_| empty_object()).collect()),
            logit_bias: opaque_option(logit_bias),
            logprobs,
            modalities: modalities.map(|values| {
                values
                    .into_iter()
                    .map(|value| preserve_known_string(value, &["text", "audio"], replacement))
                    .collect()
            }),
            moderation: opaque_option(moderation),
            n,
            prediction: opaque_option(prediction),
            presence_penalty,
            prompt_cache_key: marked_option(prompt_cache_key, replacement),
            prompt_cache_options: opaque_option(prompt_cache_options),
            prompt_cache_retention: marked_option(prompt_cache_retention, replacement),
            reasoning_effort: marked_option(reasoning_effort, replacement),
            response_format: opaque_option(response_format),
            safety_identifier: marked_option(safety_identifier, replacement),
            seed,
            stream_options: opaque_option(stream_options),
            verbosity: marked_option(verbosity, replacement),
            web_search_options: opaque_option(web_search_options),
        },
        ApiSpecificRequest::OpenAIResponses {
            background,
            context_management,
            conversation,
            moderation,
            prompt,
            prompt_cache_key,
            prompt_cache_options,
            prompt_cache_retention,
            safety_identifier,
            stream_options,
            text,
        } => ApiSpecificRequest::OpenAIResponses {
            background: background.map(|_| false),
            context_management: opaque_option(context_management),
            conversation: opaque_option(conversation),
            moderation: opaque_option(moderation),
            prompt: opaque_option(prompt),
            prompt_cache_key: marked_option(prompt_cache_key, replacement),
            prompt_cache_options: opaque_option(prompt_cache_options),
            prompt_cache_retention: marked_option(prompt_cache_retention, replacement),
            safety_identifier: marked_option(safety_identifier, replacement),
            stream_options: opaque_option(stream_options),
            text: opaque_option(text),
        },
        ApiSpecificRequest::OCIGenAI {
            compartment_id,
            serving_mode,
            api_format,
        } => ApiSpecificRequest::OCIGenAI {
            compartment_id: marked_option(compartment_id, replacement),
            serving_mode: opaque_option(serving_mode),
            api_format: api_format.map(|value| {
                preserve_known_string(value, &["GENERIC", "COHERE", "COHEREV2"], replacement)
            }),
        },
        ApiSpecificRequest::Custom { .. } => ApiSpecificRequest::Custom {
            api_name: replacement.to_string(),
            data: empty_object(),
        },
    }
}

fn sanitize_response_tool_call(call: ResponseToolCall, replacement: &str) -> ResponseToolCall {
    ResponseToolCall {
        id: replacement.to_string(),
        name: call.name,
        arguments: empty_object(),
    }
}

fn sanitize_usage(mut usage: Usage, replacement: &str) -> Usage {
    usage.cost = usage.cost.map(|cost| sanitize_cost(cost, replacement));
    usage
}

fn sanitize_cost(mut cost: CostEstimate, replacement: &str) -> CostEstimate {
    cost.currency = sanitize_currency(cost.currency, replacement);
    cost.pricing_provider = None;
    cost.pricing_model = None;
    cost.pricing_as_of = None;
    cost.pricing_source = None;
    cost
}

fn sanitize_currency(currency: String, replacement: &str) -> String {
    let currency = currency.trim();
    if currency.len() == 3 && currency.bytes().all(|byte| byte.is_ascii_alphabetic()) {
        currency.to_ascii_uppercase()
    } else {
        replacement.to_string()
    }
}

fn sanitize_optimization_summary(
    mut summary: LlmOptimizationSummary,
    replacement: &str,
) -> LlmOptimizationSummary {
    summary.schema_version = preserve_known_string(summary.schema_version, &["1"], replacement);
    summary.calculation_version =
        preserve_known_string(summary.calculation_version, &["1"], replacement);
    summary.limitations.clear();
    summary.baseline_model = None;
    summary.effective_model = None;
    summary.effective_usage = summary
        .effective_usage
        .map(|usage| sanitize_usage(usage, replacement));
    summary.baseline_usage = summary
        .baseline_usage
        .map(|usage| sanitize_usage(usage, replacement));
    summary.baseline_cost = summary
        .baseline_cost
        .map(|cost| sanitize_cost(cost, replacement));
    summary.actual_cost = summary
        .actual_cost
        .map(|cost| sanitize_cost(cost, replacement));
    summary.currency = summary
        .currency
        .map(|currency| sanitize_currency(currency, replacement));
    summary.contributions.clear();
    summary
}

fn sanitize_api_specific_response(
    response: ApiSpecificResponse,
    replacement: &str,
) -> ApiSpecificResponse {
    match response {
        ApiSpecificResponse::OpenAIChat {
            logprobs,
            system_fingerprint,
            ..
        } => ApiSpecificResponse::OpenAIChat {
            logprobs: opaque_option(logprobs),
            system_fingerprint: marked_option(system_fingerprint, replacement),
            service_tier: None,
        },
        ApiSpecificResponse::OpenAIResponses {
            output_items,
            status,
            incomplete_details,
            previous_response_id,
            store,
            truncation,
            reasoning,
            input_tokens_details,
            output_tokens_details,
            ..
        } => ApiSpecificResponse::OpenAIResponses {
            output_items: output_items
                .map(|values| values.into_iter().map(|_| empty_object()).collect()),
            status: status.map(|value| {
                preserve_known_string(
                    value,
                    &[
                        "queued",
                        "in_progress",
                        "completed",
                        "incomplete",
                        "failed",
                        "cancelled",
                    ],
                    replacement,
                )
            }),
            incomplete_details: opaque_option(incomplete_details),
            previous_response_id: marked_option(previous_response_id, replacement),
            store: store.map(|_| false),
            service_tier: None,
            truncation: opaque_option(truncation),
            reasoning: opaque_option(reasoning),
            input_tokens_details: opaque_option(input_tokens_details),
            output_tokens_details: opaque_option(output_tokens_details),
        },
        ApiSpecificResponse::AnthropicMessages {
            object_type,
            role,
            stop_reason,
            stop_sequence,
            container,
            content_blocks,
            ..
        } => ApiSpecificResponse::AnthropicMessages {
            object_type: object_type
                .map(|value| preserve_known_string(value, &["message"], replacement)),
            role: role.map(|value| preserve_known_string(value, &["assistant"], replacement)),
            stop_reason: stop_reason.map(|value| {
                preserve_known_string(
                    value,
                    &[
                        "end_turn",
                        "max_tokens",
                        "stop_sequence",
                        "tool_use",
                        "pause_turn",
                        "refusal",
                    ],
                    replacement,
                )
            }),
            stop_sequence: marked_option(stop_sequence, replacement),
            service_tier: None,
            container: opaque_option(container),
            content_blocks: content_blocks
                .map(|values| values.into_iter().map(|_| empty_object()).collect()),
        },
        ApiSpecificResponse::OCIGenAI {
            api_format,
            model_version,
        } => ApiSpecificResponse::OCIGenAI {
            api_format: api_format.map(|value| {
                preserve_known_string(value, &["GENERIC", "COHERE", "COHEREV2"], replacement)
            }),
            model_version,
        },
        ApiSpecificResponse::GeminiGenerateContent {
            thoughts_tokens,
            safety_ratings,
            grounding_metadata,
            citation_metadata,
            ..
        } => ApiSpecificResponse::GeminiGenerateContent {
            thoughts_tokens,
            safety_ratings: opaque_option(safety_ratings),
            grounding_metadata: opaque_option(grounding_metadata),
            citation_metadata: opaque_option(citation_metadata),
            extra: Map::new(),
        },
        ApiSpecificResponse::Custom { .. } => ApiSpecificResponse::Custom {
            api_name: replacement.to_string(),
            data: empty_object(),
        },
    }
}

fn opaque_option(value: Option<Json>) -> Option<Json> {
    value.map(|_| empty_object())
}

fn marked_option(value: Option<String>, replacement: &str) -> Option<String> {
    value.map(|_| replacement.to_string())
}

fn preserve_known_string(value: String, allowed: &[&str], replacement: &str) -> String {
    if allowed.contains(&value.as_str()) {
        value
    } else {
        replacement.to_string()
    }
}

fn sanitize_native_kind(kind: String, replacement: &str) -> String {
    preserve_known_string(
        kind,
        &[
            "message",
            "text",
            "input_text",
            "output_text",
            "refusal",
            "image_url",
            "input_image",
            "function_call",
            "function_call_output",
            "custom_tool_call",
            "custom_tool_call_output",
            "tool_use",
            "tool_result",
            "computer_call",
            "computer_call_output",
            "reasoning",
        ],
        replacement,
    )
}

fn valid_mark_log_severity(event: &Event, metadata: Option<&Json>) -> Option<LogSeverity> {
    if !matches!(event, Event::Mark(_)) {
        return None;
    }
    metadata
        .and_then(Json::as_object)
        .and_then(|metadata| metadata.get(LOG_SEVERITY_METADATA_KEY))
        .and_then(Json::as_str)
        .and_then(|value| value.parse::<LogSeverity>().ok())
}

fn restore_log_severity(
    mut fields: nemo_relay::api::event::EventSanitizeFields,
    severity: Option<LogSeverity>,
) -> nemo_relay::api::event::EventSanitizeFields {
    if let Some(severity) = severity {
        let metadata = fields.metadata.get_or_insert_with(empty_object);
        if let Json::Object(metadata) = metadata {
            metadata.insert(
                LOG_SEVERITY_METADATA_KEY.to_string(),
                Json::String(severity.as_str().to_string()),
            );
        }
    }
    fields
}

/// Return whether an event carries Relay's typed metric schema.
pub(crate) fn is_relay_metric_mark(event: &Event) -> bool {
    matches!(event, Event::Mark(_))
        && event.data_schema().is_some_and(|schema| {
            schema.name == METRIC_DATA_SCHEMA_NAME && schema.version == METRIC_DATA_SCHEMA_VERSION
        })
}

fn retain_allowed_metric_string_attributes(
    value: Json,
    allowlist: &BTreeMap<String, BTreeSet<String>>,
) -> Json {
    let Json::Object(values) = value else {
        return empty_object();
    };
    Json::Object(
        values
            .into_iter()
            .filter(|(attribute, value)| metric_attribute_is_allowed(allowlist, attribute, value))
            .collect(),
    )
}

fn metric_attribute_is_allowed(
    allowlist: &BTreeMap<String, BTreeSet<String>>,
    attribute: &str,
    value: &Json,
) -> bool {
    let Some(allowed) = allowlist.get(attribute) else {
        return false;
    };
    match value {
        Json::String(value) => allowed.contains(value),
        Json::Array(values) if values.iter().all(Json::is_string) => values
            .iter()
            .filter_map(Json::as_str)
            .all(|value| allowed.contains(value)),
        _ => false,
    }
}

fn is_known_mark_subtype(value: &str) -> bool {
    KNOWN_MARK_SUBTYPES.contains(&value)
}

fn sanitize_category_profile(
    mut profile: CategoryProfile,
    sanitizer: &TrajectorySanitizer,
) -> CategoryProfile {
    replace_optional_string(&mut profile.tool_call_id, &sanitizer.replacement);
    profile.subtype = profile.subtype.map(|subtype| {
        if is_known_mark_subtype(&subtype) {
            subtype
        } else {
            (*sanitizer.replacement).clone()
        }
    });
    replace_optional_json(&mut profile.tool_result_annotation);
    profile.extra.clear();
    profile.annotated_request = profile.annotated_request.as_ref().and_then(|request| {
        sanitizer
            .sanitize_annotated_request((**request).clone())
            .map(Arc::new)
    });
    profile.annotated_response = profile.annotated_response.as_ref().and_then(|response| {
        sanitizer
            .sanitize_annotated_response((**response).clone())
            .map(Arc::new)
    });
    profile
}

fn provider_name(event: &Event) -> Option<String> {
    let keys = [PROVIDER_ATTRIBUTE, "provider_name", "provider"];
    event
        .category_profile()
        .and_then(|profile| find_btree_string(&profile.extra, &keys))
        .or_else(|| {
            event
                .metadata()
                .and_then(Json::as_object)
                .and_then(|value| find_string(value, &keys))
        })
        .or_else(|| {
            event
                .data()
                .and_then(Json::as_object)
                .and_then(|value| find_string(value, &keys))
        })
        .or_else(|| provider_from_event_name(event.name()).map(str::to_string))
        .or_else(|| {
            event
                .category_profile()
                .and_then(|profile| profile.annotated_request.as_deref())
                .and_then(provider_from_normalized_request)
                .map(str::to_string)
        })
}

fn find_btree_string(values: &BTreeMap<String, Json>, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| values.get(*key).and_then(Json::as_str).map(str::to_string))
        .or_else(|| {
            ["usage", "request", "response"]
                .into_iter()
                .find_map(|container| {
                    values
                        .get(container)
                        .and_then(Json::as_object)
                        .and_then(|nested| find_string(nested, keys))
                })
        })
}

fn find_string(values: &Map<String, Json>, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| values.get(*key).and_then(Json::as_str).map(str::to_string))
        .or_else(|| {
            ["usage", "request", "response"]
                .into_iter()
                .find_map(|container| {
                    values
                        .get(container)
                        .and_then(Json::as_object)
                        .and_then(|nested| {
                            keys.iter().find_map(|key| {
                                nested.get(*key).and_then(Json::as_str).map(str::to_string)
                            })
                        })
                })
        })
}

fn provider_from_event_name(name: &str) -> Option<&'static str> {
    let name = name.to_ascii_lowercase();
    [
        ("azure_ai_inference", "azure.ai.inference"),
        ("azure ai inference", "azure.ai.inference"),
        ("azure_openai", "azure.ai.openai"),
        ("azure openai", "azure.ai.openai"),
        ("anthropic", "anthropic"),
        ("claude", "anthropic"),
        ("bedrock", "aws.bedrock"),
        ("cohere", "cohere"),
        ("deepseek", "deepseek"),
        ("gemini", "gcp.gemini"),
        ("vertex", "gcp.vertex_ai"),
        ("groq", "groq"),
        ("mistral", "mistral_ai"),
        ("openai", "openai"),
        ("gpt", "openai"),
        ("perplexity", "perplexity"),
    ]
    .into_iter()
    .find_map(|(needle, provider)| name.contains(needle).then_some(provider))
}

fn provider_from_normalized_request(request: &AnnotatedLlmRequest) -> Option<&'static str> {
    match request.api_specific.as_ref()? {
        ApiSpecificRequest::AnthropicMessages { .. } => Some("anthropic"),
        ApiSpecificRequest::OpenAIChat { .. } | ApiSpecificRequest::OpenAIResponses { .. } => {
            Some("openai")
        }
        ApiSpecificRequest::OCIGenAI { .. } => Some("oci.genai"),
        ApiSpecificRequest::Custom { .. } => None,
    }
}
