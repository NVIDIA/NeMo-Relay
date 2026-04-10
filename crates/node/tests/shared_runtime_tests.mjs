// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const lib = require('../index.js');

describe('runtime surface', () => {
  it('does not expose manual runtime control helpers', () => {
    assert.equal('configureSharedRuntime' in lib, false);
    assert.equal('sharedRuntimeStatus' in lib, false);
    assert.equal('resetStaleSharedRuntime' in lib, false);
  });
});
