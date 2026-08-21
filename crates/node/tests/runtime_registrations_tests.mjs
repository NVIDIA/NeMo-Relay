// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const lib = require('../index.js');

describe('conditional middleware guardrails', () => {
  it('discovers and dynamically toggles an existing registration', async () => {
    const suffix = `${process.pid}-${Date.now()}`;
    const targetName = `node-runtime-target-${suffix}`;
    const gateName = `node-runtime-gate-${suffix}`;
    const seen = [];

    lib.registerToolRequestIntercept(targetName, 0, false, (_name, args) => ({
      ...args,
      intercepted: true,
    }));
    try {
      const registrations = lib.listRuntimeRegistrations(['tool_request_intercept']);
      const target = registrations.find((registration) => registration.localName === targetName);
      assert.ok(target);
      assert.equal(target.effectiveName, targetName);

      lib.registerConditionalMiddlewareGuardrail(
        gateName,
        ['tool_request_intercept'],
        target.effectiveName,
        (kinds, registrationName) => {
          seen.push([kinds, registrationName]);
          return 'timer active';
        },
      );
      try {
        assert.deepEqual(await lib.toolRequestIntercepts('tool', {}), {});
        assert.deepEqual(seen, [[['tool_request_intercept'], targetName]]);
      } finally {
        assert.equal(lib.deregisterConditionalMiddlewareGuardrail(gateName), true);
      }

      assert.deepEqual(await lib.toolRequestIntercepts('tool', {}), { intercepted: true });
    } finally {
      lib.deregisterToolRequestIntercept(targetName);
    }
  });

  it('fails open when the JavaScript gate throws', async () => {
    const suffix = `${process.pid}-${Date.now()}-fail-open`;
    const targetName = `node-runtime-target-${suffix}`;
    const gateName = `node-runtime-gate-${suffix}`;

    lib.registerToolRequestIntercept(targetName, 0, false, (_name, args) => ({
      ...args,
      intercepted: true,
    }));
    try {
      lib.registerConditionalMiddlewareGuardrail(
        gateName,
        ['tool_request_intercept'],
        targetName,
        () => {
          throw new Error('expected gate failure');
        },
      );
      try {
        assert.deepEqual(await lib.toolRequestIntercepts('tool', {}), { intercepted: true });
      } finally {
        lib.deregisterConditionalMiddlewareGuardrail(gateName);
      }
    } finally {
      lib.deregisterToolRequestIntercept(targetName);
    }
  });

  it('treats an implicit undefined result as enabled', async () => {
    const suffix = `${process.pid}-${Date.now()}-undefined`;
    const targetName = `node-runtime-target-${suffix}`;
    const gateName = `node-runtime-gate-${suffix}`;

    lib.registerToolRequestIntercept(targetName, 0, false, (_name, args) => ({
      ...args,
      intercepted: true,
    }));
    try {
      lib.registerConditionalMiddlewareGuardrail(
        gateName,
        ['tool_request_intercept'],
        targetName,
        () => {},
      );
      try {
        assert.deepEqual(await lib.toolRequestIntercepts('tool', {}), { intercepted: true });
      } finally {
        lib.deregisterConditionalMiddlewareGuardrail(gateName);
      }
    } finally {
      lib.deregisterToolRequestIntercept(targetName);
    }
  });
});
