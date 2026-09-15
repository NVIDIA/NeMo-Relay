// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFileSync, writeFileSync } from 'node:fs';

const [declarationsPath, additionsPath] = process.argv.slice(2);

if (!declarationsPath || !additionsPath) {
  throw new Error('usage: append_root_types.mjs <index.d.ts> <root-types.d.ts>');
}

const declarations = readFileSync(declarationsPath, 'utf8').trimEnd();
const additions = readFileSync(additionsPath, 'utf8').trim();

writeFileSync(declarationsPath, `${declarations}\n\n${additions}\n`);
