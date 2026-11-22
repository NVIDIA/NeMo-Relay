// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Test double for the CLI's private pi response mode. Rust tests verify credentials;
// this fixture keeps extension lifecycle tests independent of a compiled CLI.
import { createInterface } from 'node:readline';
import { setTimeout as delay } from 'node:timers/promises';

const args = process.argv.slice(2);
if (args[0] !== 'hook-forward' || args[1] !== 'pi' || !args.includes('--pi-extension-client')) {
  throw new Error('expected authenticated pi hook forwarding');
}
const streaming = args.includes('--pi-extension-stream');

async function forward(input) {
  const payload = JSON.parse(input);
  if (payload.hook_event_name === 'fixture_split_stderr') {
    const bytes = Buffer.from('é');
    process.stderr.write(bytes.subarray(0, 1));
    await delay(80);
    await new Promise(resolve => process.stderr.write(bytes.subarray(1), resolve));
    process.exit(1);
  }
  if (payload.hook_event_name === 'fixture_output_overflow') {
    process[payload.stream].write('x'.repeat(9 * 1024 * 1024));
    return;
  }
  if (payload.hook_event_name === 'fixture_process_id') {
    console.log(JSON.stringify({ status: 200, body: JSON.stringify({ pid: process.pid }) }));
    return;
  }
  const gateway = args[args.indexOf('--gateway-url') + 1];
  try {
    const response = await fetch(`${gateway}/hooks/pi`, {
      method: 'POST', redirect: 'error',
      headers: { 'content-type': 'application/json' }, body: input,
    });
    console.log(JSON.stringify({ status: response.status, body: await response.text() }));
  } catch (error) {
    if (streaming) console.log(JSON.stringify({ transportError: String(error) }));
    else {
      console.error(String(error));
      process.exitCode = 1;
    }
  }
}

if (streaming) {
  const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of lines) await forward(line);
} else {
  let input = '';
  for await (const chunk of process.stdin) input += chunk;
  await forward(input);
}
