// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const { registerToolConditionalExecutionGuardrail } = require('nemo-relay-node');

function registerQueryPolicy() {
  registerToolConditionalExecutionGuardrail('search.require_query', 20, (toolName, toolArguments) => {
    if (toolName !== 'search') {
      return null;
    }
    return toolArguments.query ? null : 'search.query is required';
  });
}

module.exports = { registerQueryPolicy };
