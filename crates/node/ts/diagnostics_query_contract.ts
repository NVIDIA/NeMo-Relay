// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import * as lib from './index.js';

const contract = lib._diagnosticsQueryContract();

export const MAX_QUERY_BYTES: number = contract.maxQueryBytes;
export const MAX_QUERY_NODES: number = 64;
export const MAX_QUERY_DEPTH: number = 4;
export const MAX_QUERY_OBJECT_FIELDS: number = 20;
export const MAX_QUERY_FIELD_NAME_BYTES: number = 128;
export const NUMERIC_QUERY_FIELDS: ReadonlySet<string> = new Set(contract.numericQueryFields);
