// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Borrowed event projection with provider normalization supplied by the runtime.

use crate::host::ObservabilityHost;
use nemo_relay_types::api::event::Event;
use nemo_relay_types::api::llm::LlmRequest;
use nemo_relay_types::codec::request::AnnotatedLlmRequest;
use nemo_relay_types::codec::response::AnnotatedLlmResponse;
use std::borrow::Cow;

#[derive(Clone, Copy)]
pub struct EventView<'a> {
    pub event: &'a Event,
    pub host: &'a dyn ObservabilityHost,
}
impl<'a> EventView<'a> {
    pub fn new(event: &'a Event, host: &'a dyn ObservabilityHost) -> Self {
        Self { event, host }
    }
    pub fn as_event(&self) -> &'a Event {
        self.event
    }
    pub fn normalized_llm_request(&self) -> Option<Cow<'a, AnnotatedLlmRequest>> {
        if let Some(annotated) = self.event.annotated_request() {
            return Some(Cow::Borrowed(annotated.as_ref()));
        }
        let request: LlmRequest = serde_json::from_value(self.event.input()?.clone()).ok()?;
        self.host
            .normalize_request(&request, Some(self.event.name()))
            .map(Cow::Owned)
    }
    pub fn normalized_llm_response(&self) -> Option<Cow<'a, AnnotatedLlmResponse>> {
        if let Some(annotated) = self.event.annotated_response() {
            return Some(Cow::Borrowed(annotated.as_ref()));
        }
        self.host
            .normalize_response(self.event.output()?)
            .map(Cow::Owned)
    }
}
impl std::ops::Deref for EventView<'_> {
    type Target = Event;
    fn deref(&self) -> &Event {
        self.event
    }
}
