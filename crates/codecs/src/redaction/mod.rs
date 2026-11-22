// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-owned sanitized projection and source-aware editing.

mod overlay;
mod projection;

use crate::resolve::ProviderSurface;
use nemo_relay_types::{
    Json,
    api::llm::LlmRequest,
    codec::{
        redaction::{LlmRequestRedaction, LlmResponseRedaction, NormalizedEdits},
        request::AnnotatedLlmRequest,
        response::AnnotatedLlmResponse,
        traits::{LlmCodec, LlmResponseCodec},
    },
    error::Result,
};

macro_rules! redaction {
    ($codec:path, $surface:ident) => {
        impl LlmRequestRedaction for $codec {
            fn edit_request(
                &self,
                original: &LlmRequest,
                edits: NormalizedEdits<'_>,
            ) -> Result<Option<LlmRequest>> {
                Ok(edit_normalized_request(self, original, edits))
            }
            fn project_request(
                &self,
                sanitized: &AnnotatedLlmRequest,
            ) -> Result<Option<LlmRequest>> {
                Ok(projection::render_request(
                    ProviderSurface::$surface,
                    sanitized,
                ))
            }
        }
        impl LlmResponseRedaction for $codec {
            fn edit_response(
                &self,
                original: &Json,
                edits: NormalizedEdits<'_>,
            ) -> Result<Option<Json>> {
                Ok(edit_response(
                    self,
                    ProviderSurface::$surface,
                    original,
                    edits,
                ))
            }
            fn project_response(&self, sanitized: &AnnotatedLlmResponse) -> Result<Option<Json>> {
                Ok(projection::render_response(
                    ProviderSurface::$surface,
                    sanitized,
                ))
            }
        }
    };
}
redaction!(crate::openai_chat::OpenAIChatCodec, OpenAIChat);
redaction!(
    crate::openai_responses::OpenAIResponsesCodec,
    OpenAIResponses
);
redaction!(crate::anthropic::AnthropicMessagesCodec, AnthropicMessages);
redaction!(crate::oci_genai::OCIGenAIChatCodec, OCIGenAI);
redaction!(
    crate::gemini_generate_content::GeminiGenerateContentCodec,
    GeminiGenerateContent
);

/// Apply normalized request edits using an opted-in codec’s source-preserving encoder.
/// Reject ambiguous structural edits and verify the result by decoding the wire payload.
pub fn edit_normalized_request(
    codec: &dyn LlmCodec,
    original: &LlmRequest,
    edits: NormalizedEdits<'_>,
) -> Option<LlmRequest> {
    let baseline = codec.decode(original).ok()?;
    (serde_json::to_value(&baseline).ok()?.eq(edits.baseline)).then_some(())?;
    // Request encoders do not yet carry stable identities for arbitrary removed
    // history parts. Reject structural array edits rather than guess alignment.
    same_array_slots(edits.baseline, edits.sanitized).then_some(())?;
    let sanitized: AnnotatedLlmRequest = serde_json::from_value(edits.sanitized.clone()).ok()?;
    if let Ok(encoded) = codec.encode_with_baseline(&sanitized, original, &baseline)
        && verifies(
            edits.baseline,
            edits.sanitized,
            &serde_json::to_value(codec.decode(&encoded).ok()?).ok()?,
        )
    {
        return Some(encoded);
    }
    // Some request encoders accept changes to one normalized field at a time.
    // All positions still refer to the original arrays; no compaction occurs.
    let mut paths = Vec::new();
    changed_paths(edits.baseline, edits.sanitized, &mut Vec::new(), &mut paths);
    let mut encoded = original.clone();
    for path in paths {
        let current = codec.decode(&encoded).ok()?;
        let mut value = serde_json::to_value(&current).ok()?;
        set_path(&mut value, &path, at_path(edits.sanitized, &path).cloned())?;
        let updated = serde_json::from_value(value).ok()?;
        encoded = codec
            .encode_with_baseline(&updated, &encoded, &current)
            .ok()?;
    }
    verifies(
        edits.baseline,
        edits.sanitized,
        &serde_json::to_value(codec.decode(&encoded).ok()?).ok()?,
    )
    .then_some(encoded)
}

fn edit_response(
    codec: &dyn LlmResponseCodec,
    surface: ProviderSurface,
    original: &Json,
    edits: NormalizedEdits<'_>,
) -> Option<Json> {
    let baseline = codec.decode_response(original).ok()?;
    (serde_json::to_value(&baseline).ok()?.eq(edits.baseline)).then_some(())?;
    // These decoders expose only the first choice/candidate. Conservatively
    // reject normalized edits when further output has no sanitized counterpart.
    for pointer in ["/choices", "/candidates", "/chatResponse/choices"] {
        if edits.require_complete_output
            && original
                .pointer(pointer)
                .and_then(Json::as_array)
                .is_some_and(|items| items.len() > 1)
        {
            return None;
        }
    }
    if edits.baseline == edits.sanitized {
        return Some(original.clone());
    }
    if surface == ProviderSurface::OpenAIResponses
        && edits.baseline.get("message").is_some_and(Json::is_array)
        && edits.baseline.get("message") != edits.sanitized.get("message")
    {
        return edit_responses_parts(codec, original, edits, &baseline);
    }
    same_array_slots(edits.baseline, edits.sanitized).then_some(())?;
    let sanitized = serde_json::from_value(edits.sanitized.clone()).ok()?;
    let fields = overlay::ResponseFields::between(&baseline, &sanitized);
    let payload = overlay::BuiltinCodecName::from_provider_surface(surface)
        .overlay_response_fields(original.clone(), &sanitized, fields);
    verifies(
        edits.baseline,
        edits.sanitized,
        &serde_json::to_value(codec.decode_response(&payload).ok()?).ok()?,
    )
    .then_some(payload)
}

/// OpenAI Responses flattens message blocks into a normalized array when a
/// refusal is present. Retain each original wire path through replacements and
/// removals; never infer identity from the surviving part's kind or value.
fn edit_responses_parts(
    codec: &dyn LlmResponseCodec,
    original: &Json,
    edits: NormalizedEdits<'_>,
    baseline: &AnnotatedLlmResponse,
) -> Option<Json> {
    let before = edits.baseline.get("message")?.as_array()?;
    let after = edits.sanitized.get("message").and_then(Json::as_array);
    if let Some(after) = after {
        (before.len() == after.len()).then_some(())?;
    } else if edits
        .sanitized
        .get("message")
        .is_some_and(|value| !value.is_null())
    {
        return None;
    }
    let origins = responses_part_origins(original)?;
    (origins.len() == before.len()).then_some(())?;
    let mut expected = edits.sanitized.clone();
    // Validate non-message edits separately; tool removal may compact the output array.
    let mut padded = expected.clone();
    padded["message"] = edits.baseline["message"].clone();
    same_array_slots(edits.baseline, &padded).then_some(())?;
    let padded = serde_json::from_value(padded).ok()?;
    let fields = overlay::ResponseFields::between(baseline, &padded);
    let mut payload = original.clone();
    for (index, path) in origins.iter().enumerate().rev() {
        let part = after
            .and_then(|parts| parts.get(index))
            .filter(|part| !part.is_null());
        match part {
            None => {
                set_path(&mut payload, path, None)?;
            }
            Some(part) => {
                (before[index].get("type") == part.get("type")).then_some(())?;
                let field = if part.get("type").and_then(Json::as_str) == Some("text") {
                    "text"
                } else {
                    "refusal"
                };
                let text = part.get(field)?.as_str()?;
                let mut supported = before[index].clone();
                supported[field] = Json::String(text.into());
                (&supported == part).then_some(())?;
                let mut field_path = path.clone();
                field_path.push(field.into());
                set_path(&mut payload, &field_path, Some(Json::String(text.into())))?;
            }
        }
    }
    // Match the decoder's canonical form after compaction: all-text output is
    // one joined string; an empty output has no message.
    let surviving: Vec<Json> = after
        .into_iter()
        .flatten()
        .filter(|part| !part.is_null())
        .cloned()
        .collect();
    let text: Vec<&str> = surviving
        .iter()
        .filter_map(|part| part.get("text").and_then(Json::as_str))
        .collect();
    if surviving.is_empty() {
        expected.as_object_mut()?.remove("message");
    } else if text.len() == surviving.len() {
        expected["message"] = Json::String(text.join("\n"));
    } else {
        expected["message"] = Json::Array(surviving.clone());
    }
    if payload.get("output_text").is_some() {
        let text: Vec<&str> = surviving
            .iter()
            .filter_map(|part| part.get("text").and_then(Json::as_str))
            .collect();
        let value = (!text.is_empty()).then(|| Json::String(text.join("\n")));
        set_path(&mut payload, &["output_text".into()], value)?;
    }
    let payload = overlay::BuiltinCodecName::OpenAIResponses
        .overlay_response_fields(payload, &padded, fields);
    let projected = serde_json::to_value(codec.decode_response(&payload).ok()?).ok()?;
    verifies(edits.baseline, &expected, &projected).then_some(payload)
}

fn responses_part_origins(original: &Json) -> Option<Vec<Vec<String>>> {
    let mut origins = Vec::new();
    for (i, item) in original.get("output")?.as_array()?.iter().enumerate() {
        if item.get("type").and_then(Json::as_str) == Some("message") {
            if let Some(blocks) = item.get("content").and_then(Json::as_array) {
                for (j, block) in blocks.iter().enumerate() {
                    if output_part_field(block).is_some() {
                        origins.push(vec![
                            "output".into(),
                            i.to_string(),
                            "content".into(),
                            j.to_string(),
                        ]);
                    }
                }
            }
        } else if item.get("type").and_then(Json::as_str) == Some("output_text")
            && output_part_field(item).is_some()
        {
            origins.push(vec!["output".into(), i.to_string()]);
        }
    }
    Some(origins)
}

fn output_part_field(block: &Json) -> Option<&'static str> {
    let field = match block.get("type")?.as_str()? {
        "output_text" => "text",
        "refusal" => "refusal",
        _ => return None,
    };
    block.get(field)?.as_str()?;
    Some(field)
}

fn same_array_slots(before: &Json, after: &Json) -> bool {
    match (before, after) {
        (Json::Array(before), Json::Array(after)) => {
            before.len() == after.len()
                && before.iter().zip(after).all(|(before, after)| {
                    (!after.is_null() || before.is_null()) && same_array_slots(before, after)
                })
        }
        (Json::Object(before), Json::Object(after)) => before.iter().all(|(key, before)| {
            after
                .get(key)
                .is_none_or(|after| same_array_slots(before, after))
        }),
        _ => true,
    }
}

/// Compare edits only: provider snapshots and unrelated normalized fields may
/// legitimately change on decode. Missing and null remain distinct.
fn verifies(before: &Json, after: &Json, projected: &Json) -> bool {
    if before == after {
        return true;
    }
    match (before, after, projected) {
        (Json::Object(before), Json::Object(after), Json::Object(projected)) => {
            before.keys().chain(after.keys()).all(|key| {
                if before.get(key) == after.get(key) {
                    return true;
                }
                match (before.get(key), after.get(key), projected.get(key)) {
                    (Some(before), Some(after), Some(projected)) => {
                        verifies(before, after, projected)
                    }
                    (_, after, projected) => after == projected,
                }
            })
        }
        (Json::Array(before), Json::Array(after), Json::Array(projected))
            if before.len() == after.len() && after.len() == projected.len() =>
        {
            before
                .iter()
                .zip(after)
                .zip(projected)
                .all(|((before, after), projected)| verifies(before, after, projected))
        }
        _ => after == projected,
    }
}

fn changed_paths(
    before: &Json,
    after: &Json,
    path: &mut Vec<String>,
    paths: &mut Vec<Vec<String>>,
) {
    if before == after {
        return;
    }
    match (before, after) {
        (Json::Object(before), Json::Object(after)) => {
            for key in before
                .keys()
                .chain(after.keys().filter(|key| !before.contains_key(*key)))
            {
                path.push(key.clone());
                match (before.get(key), after.get(key)) {
                    (Some(before), Some(after)) => changed_paths(before, after, path, paths),
                    _ => paths.push(path.clone()),
                }
                path.pop();
            }
        }
        (Json::Array(before), Json::Array(after)) if before.len() == after.len() => {
            for (index, (before, after)) in before.iter().zip(after).enumerate() {
                path.push(index.to_string());
                changed_paths(before, after, path, paths);
                path.pop();
            }
        }
        _ => paths.push(path.clone()),
    }
}

fn at_path<'a>(mut value: &'a Json, path: &[String]) -> Option<&'a Json> {
    for key in path {
        value = if let Some(array) = value.as_array() {
            array.get(key.parse::<usize>().ok()?)?
        } else {
            value.get(key)?
        };
    }
    Some(value)
}

fn set_path(value: &mut Json, path: &[String], replacement: Option<Json>) -> Option<()> {
    let Some((key, parent)) = path.split_last() else {
        *value = replacement.unwrap_or(Json::Null);
        return Some(());
    };
    let mut target = value;
    for key in parent {
        target = match target {
            Json::Array(array) => array.get_mut(key.parse::<usize>().ok()?)?,
            Json::Object(object) => object.get_mut(key)?,
            _ => return None,
        };
    }
    if let Some(array) = target.as_array_mut() {
        let index = key.parse::<usize>().ok()?;
        if index >= array.len() {
            return None;
        }
        match replacement {
            Some(value) => array[index] = value,
            None => {
                array.remove(index);
            }
        }
    } else {
        let object = target.as_object_mut()?;
        match replacement {
            Some(value) => {
                object.insert(key.clone(), value);
            }
            None => {
                object.remove(key);
            }
        }
    }
    Some(())
}

#[cfg(test)]
#[path = "../../tests/unit/redaction_tests.rs"]
mod tests;
