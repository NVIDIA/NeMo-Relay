// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const {
  registerToolConditionalExecutionGuardrail,
  registerToolExecutionIntercept,
  registerToolSanitizeRequestGuardrail,
} = require('nemo-relay-node');

function registerToolPolicy() {
  registerToolSanitizeRequestGuardrail('search.redact_api_key', 10, (_toolName, toolArguments) => {
    if (!toolArguments.api_key) {
      return toolArguments;
    }
    return { ...toolArguments, api_key: '<redacted>' };
  });

  registerToolConditionalExecutionGuardrail('search.require_query', 20, (toolName, toolArguments) => {
    if (toolName !== 'search') {
      return null;
    }
    return toolArguments.query ? null : 'search.query is required';
  });

  registerToolExecutionIntercept('search.measure', 30, async (context, nextCall) => {
    const started = performance.now();
    try {
      const executionResult = await nextCall(context.args);
      return {
        result: executionResult.result,
        ...(executionResult.annotation === undefined ? {} : { annotation: executionResult.annotation }),
      };
    } finally {
      console.log(`search completed in ${Math.round(performance.now() - started)} ms`);
    }
  });
}

module.exports = { registerToolPolicy };
