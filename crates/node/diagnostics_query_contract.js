// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use strict';

const lib = require('./index.js');
const core = JSON.parse(lib._diagnosticsQueryContract());

// These traversal caps bound the pre-serialization object walk in this adapter.
module.exports = Object.freeze({
  MAX_QUERY_BYTES: core.max_query_bytes,
  MAX_QUERY_NODES: 64,
  MAX_QUERY_DEPTH: 4,
  MAX_QUERY_OBJECT_FIELDS: 20,
  MAX_QUERY_FIELD_NAME_BYTES: 128,
  NUMERIC_QUERY_FIELDS: new Set(core.numeric_query_fields),
});
