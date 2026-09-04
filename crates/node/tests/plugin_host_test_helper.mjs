// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const plugin = require('../plugin.js');

let activation;

export function validate(config) {
  return plugin.validate(config).config;
}

export async function initialize(config, additionalPluginsToml) {
  await close();
  const nextActivation = await plugin.initialize(config, additionalPluginsToml);
  activation = nextActivation;
  return nextActivation.report.config;
}

export function report() {
  return activation?.report.config ?? null;
}

export async function close() {
  const current = activation;
  activation = undefined;
  await current?.close();
}
