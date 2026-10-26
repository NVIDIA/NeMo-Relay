// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const plugin = require('nemo-relay-node/plugin');
const diagnostics = require('nemo-relay-node/diagnostics');

async function inspectRuntime() {
  const config = { diagnostics: { enabled: true } };
  plugin.validate(config);
  const activation = await plugin.initialize(config);
  let reader;
  try {
    reader = diagnostics.reader();
    const page = reader.snapshot({ componentKey: 'logging' });
    const failures = page.lifetime.totals.failuresTotal.value; // bigint
    const wire = JSON.parse(reader.snapshotJson());
    const sequence = wire.sequence; // Decimal string; portable across languages.
    const info = reader.info();
    const reserved = info.bindingReservedBytes;
    const loggingCoverage = info.loggingCoverage;
    // Inspect these values only inside the authorized application boundary.
  } finally {
    await activation.close();
  }
  const finalPage = reader.snapshot(); // The service outlives this activation.
}
