// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ApplicationExecutionContext, HostApproval, ToolExecutionContext } from '../dist/index.js';
import type { TypedToolExecuteOptions } from '../dist/typed.js';

const approval: HostApproval = {
  version: 1,
  source: 'application',
  approval_id: 'review-123',
  tool_name: 'write_file',
  approved_arguments: { path: 'notes.txt' },
  extension: { application_owned: true },
};
const context: ApplicationExecutionContext = { host_approval: approval, extra: [1, null] };
const options: TypedToolExecuteOptions = { executionContext: context };
void options;
function inspect(context: ToolExecutionContext): string | undefined {
  return context.executionContext?.host_approval?.approval_id;
}
void inspect;
// @ts-expect-error The standardized approval version is a numeric literal.
const invalidVersion: HostApproval = { ...approval, version: 2 };
// @ts-expect-error The obsolete decision field is reserved and forbidden.
const invalidDecision: HostApproval = { ...approval, decision: 'denied' };
// @ts-expect-error The obsolete approved value is also forbidden.
const legacyApproved: HostApproval = { ...approval, decision: 'approved' };
// @ts-expect-error Standard approval fields are required when host_approval is present.
const incomplete: ApplicationExecutionContext = { host_approval: { version: 1 } };
void [invalidVersion, invalidDecision, legacyApproved, incomplete];
