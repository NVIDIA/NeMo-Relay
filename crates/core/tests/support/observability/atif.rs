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

use crate::api::event::{Event, EventNormalizationExt};

use crate::api::runtime::EventSubscriberFn;

use crate::api::subscriber::flush_subscribers;

use crate::codec::request::{AnnotatedLlmRequest, ContentPart, Message, MessageContent};

use crate::codec::response::AnnotatedLlmResponse;

use crate::error::Result;

use crate::json::Json;

use super::{
    estimate_cost_for_response_or_model, input_tokens_including_cache, manual, merge_usage,
    model_name_for_llm_event,
};

#[cfg(test)]
#[path = "../../integration/observability/atif_codec_tests.rs"]
mod tests;
