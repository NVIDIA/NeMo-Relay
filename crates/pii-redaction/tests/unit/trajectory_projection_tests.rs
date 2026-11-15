// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use serde_json::{Map, json};

use std::collections::BTreeMap;

use nemo_relay_runtime::api::llm::LlmRequest;

use crate::trajectory::{CustomMarkPayloadPolicy, TrajectorySanitizer};

#[test]
fn oci_enveloped_requests_render_from_sanitized_annotations() {
    let cases = [
        (
            "GENERIC",
            json!({
                "messages": [{"role": "USER", "content": [{"type": "TEXT", "text": "SECRET"}]}],
            }),
        ),
        ("COHERE", json!({"message": "SECRET"})),
        (
            "COHEREV2",
            json!({
                "messages": [{"role": "USER", "content": [{"type": "TEXT", "text": "SECRET"}]}],
            }),
        ),
    ];
    let sanitizer = TrajectorySanitizer::new(
        "[REDACTED]".into(),
        CustomMarkPayloadPolicy::RedactAllLeaves,
        BTreeMap::new(),
    );

    for (api_format, chat_request) in cases {
        let request = LlmRequest {
            headers: Map::new(),
            content: json!({
                "compartmentId": "SECRET",
                "servingMode": {
                    "servingType": "ON_DEMAND",
                    "modelId": "trusted-model",
                    "future": "SECRET",
                },
                "chatRequest": {
                    "apiFormat": api_format,
                    "future": "SECRET",
                },
            }),
        };
        let mut request = request;
        request.content["chatRequest"]
            .as_object_mut()
            .unwrap()
            .extend(chat_request.as_object().unwrap().clone());
        let annotated = request_codec(ProviderSurface::OCIGenAI)
            .decode(&request)
            .unwrap();
        let sanitized = sanitizer.sanitize_annotated_request(annotated).unwrap();

        let rendered = render_request(ProviderSurface::OCIGenAI, &sanitized)
            .unwrap_or_else(|| panic!("{api_format} enveloped request projection failed"));
        let serialized = serde_json::to_string(&rendered).unwrap();
        assert!(!serialized.contains("SECRET"), "{api_format}: {serialized}");
        assert!(rendered.headers.is_empty());
        assert_eq!(rendered.content["compartmentId"], "[REDACTED]");
        assert_eq!(
            rendered.content["servingMode"],
            json!({"servingType": "ON_DEMAND", "modelId": "trusted-model"})
        );
        assert_eq!(rendered.content["chatRequest"]["apiFormat"], api_format);
        let decoded = request_codec(ProviderSurface::OCIGenAI)
            .decode(&rendered)
            .unwrap_or_else(|error| panic!("{api_format} projection was invalid: {error}"));
        assert_eq!(decoded.model.as_deref(), Some("trusted-model"));
        assert_eq!(decoded.messages.len(), 1);
    }
}
