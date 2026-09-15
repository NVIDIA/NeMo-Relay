// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { builtinConfig, type BuiltinConfig } from '../pii_redaction.js';

const trajectoryContext: BuiltinConfig = {
  preset: 'trajectory_context',
  custom_mark_payload_policy: 'preserve',
  metric_string_attribute_allowlist: { 'gen_ai.operation.name': ['chat'] },
};

builtinConfig(trajectoryContext);
