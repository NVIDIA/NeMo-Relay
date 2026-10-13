// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Test double for the CLI's private pi response mode. Rust tests verify credentials;
// this fixture keeps extension lifecycle tests independent of a compiled CLI.
const args = process.argv.slice(2);
if (args[0] !== 'hook-forward' || args[1] !== 'pi' || !args.includes('--pi-extension-client')) {
  throw new Error('expected authenticated pi hook forwarding');
}
let input = '';
for await (const chunk of process.stdin) input += chunk;
const payload = JSON.parse(input);
if (payload.hook_event_name === 'fixture_output_overflow') {
  // Exercise execFile's real stdout/stderr limits without an HTTP server or CLI build.
  process[payload.stream].write('x'.repeat(9 * 1024 * 1024));
} else {
  const gateway = args[args.indexOf('--gateway-url') + 1];
  try {
    const response = await fetch(`${gateway}/hooks/pi`, {
      method: 'POST', redirect: 'error',
      headers: { 'content-type': 'application/json' }, body: input,
    });
    console.log(JSON.stringify({ status: response.status, body: await response.text() }));
  } catch (error) {
    console.error(String(error));
    process.exitCode = 1;
  }
}
