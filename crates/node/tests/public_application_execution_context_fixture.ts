// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ApplicationExecutionContext, ToolExecutionContext } from '../dist/index.js';
import type { TypedToolExecuteOptions } from '../dist/typed.js';

const context: ApplicationExecutionContext = {
  data: { host_approval: { approval_id: 'review-123', decision: 'application-owned' }, extra: [1, null] },
};
const options: TypedToolExecuteOptions = { executionContext: context };
void options;
function inspect(context: ToolExecutionContext): unknown {
  return context.executionContext?.data?.host_approval;
}
void inspect;
// @ts-expect-error Application fields belong under data, not beside it.
const invalidOuter: ApplicationExecutionContext = { host_approval: {} };
// @ts-expect-error Data is a JSON object, not an array.
const invalidData: ApplicationExecutionContext = { data: [] };
// @ts-expect-error Omitted data is distinct from an invalid null data field.
const nullData: ApplicationExecutionContext = { data: null };
void [invalidOuter, invalidData, nullData];
