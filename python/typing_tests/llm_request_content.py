# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protect public request-content typing outside relaxed runtime-test rules."""

from typing import assert_type

from nemo_relay import Json, LLMRequest

object_request = LLMRequest({}, {"nested": ["payload", None]})
list_request = LLMRequest({}, ["payload"])
string_request = LLMRequest({}, "payload")
integer_request = LLMRequest({}, 42)
float_request = LLMRequest({}, 1.5)
boolean_request = LLMRequest({}, True)
null_request = LLMRequest({}, None)

assert_type(object_request.content, Json)
assert_type(list_request.content, Json)
assert_type(string_request.content, Json)
assert_type(integer_request.content, Json)
assert_type(float_request.content, Json)
assert_type(boolean_request.content, Json)
assert_type(null_request.content, Json)
