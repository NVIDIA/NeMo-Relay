// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import http from 'node:http';

const server = http.createServer((req, res) => {
  const chunks = [];
  req.on('data', (chunk) => chunks.push(chunk));
  req.on('end', () => {
    process.stdout.write(
      `${JSON.stringify({
        type: 'request',
        url: req.url,
        headers: req.headers,
        body: Buffer.concat(chunks).toString('base64'),
      })}\n`,
    );
    res.statusCode = 200;
    res.end();
  });
});

server.listen(0, '127.0.0.1', () => {
  const address = server.address();
  process.stdout.write(
    `${JSON.stringify({
      type: 'ready',
      endpoint: `http://127.0.0.1:${address.port}/v1/traces`,
    })}\n`,
  );
});

function shutdown() {
  server.close(() => process.exit(0));
}

process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);
