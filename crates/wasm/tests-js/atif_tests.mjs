// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { resetScopeStack, unique, wasm } from './test_support.mjs';

test('WebAssembly AtifExporter accepts a null model name and exports schema metadata', () => {
  const exporter = new wasm.AtifExporter('session-js', 'wasm-js', '1.0.0', null);

  try {
    assert.equal(typeof JSON.parse(exporter.exportJson()).schema_version, 'string');
  } finally {
    exporter.free();
  }
});

test('WebAssembly AtifExporter registers, captures steps, and clears state', async () => {
  const stack = resetScopeStack();
  const exporter = new wasm.AtifExporter('session-js', 'wasm-js', '1.0.0', 'demo-model');
  const exporterName = unique('exporter');

  exporter.register(exporterName);
  try {
    const toolResult = await wasm.toolCallExecute(
      'atif_tool_exec',
      {
        value: 1,
      },
      async (args) => args,
    );
    assert.equal(toolResult.value, 1);

    const exported = JSON.parse(exporter.exportJson());
    assert.equal(typeof exported.schema_version, 'string');
    assert.ok(Array.isArray(exported.steps));
    assert.ok(exported.steps.length > 0);

    assert.equal(exporter.deregister(exporterName), true);
    exporter.clear();

    const afterClear = JSON.parse(exporter.exportJson());
    assert.deepEqual(afterClear.steps, []);
  } finally {
    exporter.free();
    stack.free();
  }
});
