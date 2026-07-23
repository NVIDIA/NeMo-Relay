# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-specific request and response codecs for ``nemo_relay.llm``.

Modules in this package implement the ``nemo_relay.codecs.LlmCodec`` and
``nemo_relay.codecs.LlmResponseCodec`` protocols for LLM providers whose wire
formats are not covered by the built-in OpenAI and Anthropic codecs.
"""

from nemo_relay.providers.oci_genai import OCIGenAIChatCodec, OCIGenAIResponseCodec

__all__ = [
    "OCIGenAIChatCodec",
    "OCIGenAIResponseCodec",
]
