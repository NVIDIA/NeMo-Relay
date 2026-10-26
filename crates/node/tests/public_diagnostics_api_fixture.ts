// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { reader, type DiagnosticQuery, type DiagnosticSnapshot } from '../diagnostics';

const query: DiagnosticQuery = {
  componentKey: 'logging',
  schemaVersion: 1,
  afterSequence: 0n,
  ownership: { responsibility: 'embedded' },
};
const page: DiagnosticSnapshot = reader().snapshot(query);
const failures: bigint = page.lifetime.totals.failuresTotal.value;
const source: string = page.source.instanceId;
const component: string | undefined = page.components[0]?.identity.instanceId;
const portable: string = reader().snapshotJson(query);
// @ts-expect-error Diagnostic u64 query values require bigint, including small values.
const lossy: DiagnosticQuery = { afterSequence: 1 };
export const checkedValues0 = [failures, source, component, portable, lossy];

const rowCap: bigint = reader().info().discovery.maxQueryEvents;
const coverage: boolean = reader().info().discovery.producerSupport[0].supported;
export const checkedValues1 = [rowCap, coverage];

const memory = reader().inspectMemory(query);
const reservation: bigint = memory.aggregate.reservedBytes;
const memoryJson: string = reader().inspectMemoryJson(query);
const sampleOrdinal: bigint | undefined = memory.resource.sample?.ordinal;
export const checkedValues2 = [reservation, memoryJson, sampleOrdinal];
