// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { builtinConfig, type BuiltinConfig } from '../pii_redaction.js';

const trajectoryContext: BuiltinConfig = {
  preset: 'trajectory_context',
  custom_mark_payload_policy: 'preserve',
  metric_string_attribute_allowlist: { 'gen_ai.operation.name': ['chat'] },
};

builtinConfig(trajectoryContext);

// @ts-expect-error PII redaction supports only the trajectory-context preset.
builtinConfig({ preset: 'unknown_preset' });

// @ts-expect-error PII redaction supports only the documented custom-mark policies.
builtinConfig({ custom_mark_payload_policy: 'unknown_policy' });
