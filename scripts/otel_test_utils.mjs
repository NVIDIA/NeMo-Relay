// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { fileURLToPath } from 'node:url';

export async function startCollector() {
  const requests = [];
  let nextRequestIndex = 0;
  const pendingResolvers = [];
  let resolveReady;
  let rejectReady;
  const readyPromise = new Promise((resolve, reject) => {
    resolveReady = resolve;
    rejectReady = reject;
  });
  const collectorPath = fileURLToPath(new URL('./otel_collector.mjs', import.meta.url));
  const child = spawn(process.execPath, [collectorPath], {
    stdio: ['ignore', 'pipe', 'inherit'],
  });
  let stdout = '';

  child.stdout.setEncoding('utf8');
  child.stdout.on('data', (chunk) => {
    stdout += chunk;
    for (;;) {
      const newline = stdout.indexOf('\n');
      if (newline === -1) {
        break;
      }

      const line = stdout.slice(0, newline).trim();
      stdout = stdout.slice(newline + 1);
      if (!line) {
        continue;
      }

      let message;
      try {
        message = JSON.parse(line);
      } catch (error) {
        throw new Error(`Failed to parse collector output: ${line}`, { cause: error });
      }
      if (message.type === 'ready') {
        resolveReady(message.endpoint);
      } else if (message.type === 'request') {
        const request = {
          url: message.url,
          headers: message.headers,
          body: Buffer.from(message.body, 'base64'),
        };
        requests.push(request);
        const pendingResolver = pendingResolvers.shift();
        if (pendingResolver) {
          nextRequestIndex += 1;
          pendingResolver.resolve(request);
        }
      }
    }
  });
  child.on('exit', (code, signal) => {
    const error = new Error(`collector exited before responding (code=${code}, signal=${signal})`);
    rejectReady(error);
    for (const pendingResolver of pendingResolvers.splice(0)) {
      pendingResolver.reject(error);
    }
  });
  const endpoint = await Promise.race([
    readyPromise,
    new Promise((_, reject) => setTimeout(() => reject(new Error('timed out waiting for OTLP collector startup')), 5000)),
  ]);

  return {
    endpoint,
    requests,
    async nextRequest(timeoutMs = 5000) {
      if (nextRequestIndex < requests.length) {
        const request = requests[nextRequestIndex];
        nextRequestIndex += 1;
        return request;
      }

      const requestPromise = new Promise((resolve, reject) => {
        pendingResolvers.push({ resolve, reject });
      });
      return await Promise.race([
        requestPromise,
        new Promise((_, reject) => setTimeout(() => reject(new Error('timed out waiting for OTLP request')), timeoutMs)),
      ]);
    },
    async close() {
      if (child.exitCode !== null) {
        return;
      }
      child.kill('SIGTERM');
      await once(child, 'exit');
    },
  };
}
