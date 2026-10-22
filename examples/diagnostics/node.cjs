// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Run with node examples/diagnostics/node.cjs from the repository.
const assert = require('node:assert/strict');
const plugin = require('nemo-relay-node/plugin');
const diagnostics = require('nemo-relay-node/diagnostics');

async function main() {
  const config = { diagnostics: { enabled: true } };
  plugin.validate(config); // Validates startup settings without starting the service.
  const activation = await plugin.initialize(config);
  let reader;
  let page;
  try {
    reader = diagnostics.reader();
    page = reader.snapshot();
    assert.equal(typeof page.sequence, 'bigint');
    assert.equal(typeof JSON.parse(reader.snapshotJson()).sequence, 'string');
    assert.equal(reader.info().bindingCopyConcurrency, 2);

    // Process each page without accumulating a second retained history.
    const query = { maxComponents: 1, maxEvents: 1 };
    let complete = false;
    for (let remaining = 32; remaining > 0; --remaining) {
      page = reader.snapshot(query);
      // Consume the selected component/event rows within this trust boundary.
      if (page.continuation === null) {
        complete = true;
        break;
      }
      query.continuation = page.continuation;
    }
    if (!complete) throw new Error("The caller's page budget was exhausted");

    assert.throws(
      () => reader.snapshot({ unknown: true }),
      (error) => error instanceof diagnostics.DiagnosticsError && error.reason === 'invalid_query',
    );
  } finally {
    await activation.close();
  }
  // Plugin teardown releases its components; the process-owned reader stays valid.
  assert.deepEqual(reader.snapshot().source, page.source);
}

main().catch((error) => {
  // Report only the stable failure reason; never attach a diagnostic snapshot.
  console.error(error.reason ?? error.name);
  process.exitCode = 1;
});
