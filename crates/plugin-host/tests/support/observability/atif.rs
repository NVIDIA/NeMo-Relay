// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(unused_imports)]

//! Core integration test views for the extracted observability implementation.

use nemo_relay_observability::atif as implementation;

pub(crate) use implementation::*;

use std::collections::{HashMap, HashSet};

use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};

use serde::{Deserialize, Serialize};

use uuid::Uuid;

use nemo_relay_runtime::api::event::{Event, EventNormalizationExt};

use nemo_relay_runtime::api::runtime::EventSubscriberFn;

use nemo_relay_runtime::api::subscriber::flush_subscribers;

use nemo_relay_types::codec::request::{AnnotatedLlmRequest, ContentPart, Message, MessageContent};

use nemo_relay_types::codec::response::AnnotatedLlmResponse;

use nemo_relay_types::error::Result;

use nemo_relay_types::Json;

use super::{
    estimate_cost_for_response_or_model, input_tokens_including_cache, manual, merge_usage,
    model_name_for_llm_event,
};

#[cfg(test)]
#[path = "../../integration/observability/atif_codec_tests.rs"]
mod tests;
