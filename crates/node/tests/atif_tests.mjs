// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const lib = require('../index.js');

const {
  JsAtifExporter,
  ScopeType,
  pushScope,
  popScope,
  llmCall,
  llmCallEnd,
} = lib;

function makeNative() {
  return { headers: {}, content: { messages: [{ role: 'user', content: 'hello' }], model: 'atif-model' } };
}

describe('JsAtifExporter', () => {
  it('registers, exports, clears, and deregisters lifecycle events', () => {
    const exporter = new JsAtifExporter('session-node-types', 'node-agent', '1.0.0', 'atif-model');
    const subscriberName = `node_atif_${Date.now()}`;
    const scope = pushScope('atif_root', ScopeType.Agent, null, null);

    exporter.register(subscriberName);
    try {
      const handle = llmCall('atif_llm', makeNative(), scope, null, null, null, 'atif-model');
      llmCallEnd(handle, { content: 'world' }, null, null);

      const exportedAll = JSON.parse(exporter.exportJson());

      assert.equal(exportedAll.session_id, 'session-node-types');
      assert.equal(exportedAll.agent.name, 'node-agent');
      assert.ok(exportedAll.steps.length > 0);

      exporter.clear();
      const cleared = JSON.parse(exporter.exportJson());
      assert.deepEqual(cleared.steps, []);
    } finally {
      popScope(scope);
      assert.equal(exporter.deregister(subscriberName), true);
      assert.equal(exporter.deregister(subscriberName), false);
    }
  });

});
