# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guardrail registration for tools and LLMs.

Guardrails run inside the middleware pipeline and can sanitize or gate requests
and responses. They are priority-ordered (ascending) and registered by name.

**Tool guardrails** — callback signatures:

    register_tool_sanitize_request(name, priority, fn)
        ``fn(tool_name: str, args: Any) -> Any`` — sanitize tool arguments.

    register_tool_sanitize_response(name, priority, fn)
        ``fn(tool_name: str, result: Any) -> Any`` — sanitize tool result.

    register_tool_conditional_execution(name, priority, fn)
        ``fn(tool_name: str, args: Any) -> Optional[str]`` — return ``None``
        to allow, or a rejection reason to block.

**LLM guardrails** — callback signatures:

    register_llm_sanitize_request(name, priority, fn)
        ``fn(request: LLMRequest) -> LLMRequest`` — sanitize the LLM request.

    register_llm_sanitize_response(name, priority, fn)
        ``fn(response: LLMResponse) -> LLMResponse`` — sanitize the LLM response.

    register_llm_conditional_execution(name, priority, fn)
        ``fn(request: LLMRequest) -> Optional[str]`` — return ``None``
        to allow, or a rejection reason to block.

Each ``register_*`` has a corresponding ``deregister_*`` that takes the name
and returns ``True`` if a guardrail was found and removed.

Example::

    import nvagentrt

    def redact_pii(tool_name, args):
        # Sanitize PII from tool arguments
        return {k: "***" if "ssn" in k else v for k, v in args.items()}

    nvagentrt.guardrails.register_tool_sanitize_request("pii-filter", 10, redact_pii)
"""

from nvagentrt._native import (
    nvagentrt_deregister_llm_conditional_execution_guardrail as deregister_llm_conditional_execution,
)
from nvagentrt._native import (
    nvagentrt_deregister_llm_sanitize_request_guardrail as deregister_llm_sanitize_request,
)
from nvagentrt._native import (
    nvagentrt_deregister_llm_sanitize_response_guardrail as deregister_llm_sanitize_response,
)
from nvagentrt._native import (
    nvagentrt_deregister_tool_conditional_execution_guardrail as deregister_tool_conditional_execution,
)
from nvagentrt._native import (
    nvagentrt_deregister_tool_sanitize_request_guardrail as deregister_tool_sanitize_request,
)
from nvagentrt._native import (
    nvagentrt_deregister_tool_sanitize_response_guardrail as deregister_tool_sanitize_response,
)
from nvagentrt._native import (
    nvagentrt_register_llm_conditional_execution_guardrail as register_llm_conditional_execution,
)
from nvagentrt._native import (
    # LLM guardrails
    nvagentrt_register_llm_sanitize_request_guardrail as register_llm_sanitize_request,
)
from nvagentrt._native import (
    nvagentrt_register_llm_sanitize_response_guardrail as register_llm_sanitize_response,
)
from nvagentrt._native import (
    nvagentrt_register_tool_conditional_execution_guardrail as register_tool_conditional_execution,
)
from nvagentrt._native import (
    # Tool guardrails
    nvagentrt_register_tool_sanitize_request_guardrail as register_tool_sanitize_request,
)
from nvagentrt._native import (
    nvagentrt_register_tool_sanitize_response_guardrail as register_tool_sanitize_response,
)

__all__ = [
    "register_tool_sanitize_request",
    "deregister_tool_sanitize_request",
    "register_tool_sanitize_response",
    "deregister_tool_sanitize_response",
    "register_tool_conditional_execution",
    "deregister_tool_conditional_execution",
    "register_llm_sanitize_request",
    "deregister_llm_sanitize_request",
    "register_llm_sanitize_response",
    "deregister_llm_sanitize_response",
    "register_llm_conditional_execution",
    "deregister_llm_conditional_execution",
]
