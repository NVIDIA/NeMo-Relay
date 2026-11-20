// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, describe, it } from 'node:test';
import { pathToFileURL } from 'node:url';

const { CONFIG_SCHEMA, readCredential, readDeploymentConfig } = await import('../src/managed/deployment.ts');

const directories = [];
after(() => Promise.all(directories.map((directory) => rm(directory, { recursive: true, force: true }))));

async function configFile(contents) {
  const directory = await mkdtemp(join(tmpdir(), 'nemo-relay-pi-config-'));
  directories.push(directory);
  const path = join(directory, 'managed-config.json');
  await writeFile(path, typeof contents === 'string' ? contents : JSON.stringify(contents));
  return pathToFileURL(path);
}

const rendered = (extra = {}) => ({
  schema: CONFIG_SCHEMA,
  daemonAddress: 'https://relay.example.com:443',
  dispatcherCommand: '/opt/nvidia/bin/nemo-relay-dispatch',
  ...extra,
});

describe('deployment configuration', () => {
  it('accepts a rendered root address and drops its trailing slash', async () => {
    const config = readDeploymentConfig(
      await configFile(rendered({ daemonAddress: 'https://relay.example.com:8443/' })),
    );
    assert.equal(config.daemonAddress, 'https://relay.example.com:8443');
    assert.equal(config.dispatcherCommand, '/opt/nvidia/bin/nemo-relay-dispatch');
  });

  it('drops empty query and fragment delimiters so the hook path is appended to the origin', async () => {
    for (const daemonAddress of [
      'https://relay.example.com/?',
      'https://relay.example.com/#',
      'https://relay.example.com:443',
    ]) {
      const config = readDeploymentConfig(await configFile(rendered({ daemonAddress })));
      assert.equal(config.daemonAddress, 'https://relay.example.com', daemonAddress);
    }
  });

  it('accepts plain HTTP for loopback hosts only', async () => {
    for (const daemonAddress of ['http://127.0.0.1:4040', 'http://localhost:4040', 'http://[::1]:4040']) {
      const config = readDeploymentConfig(await configFile(rendered({ daemonAddress })));
      assert.equal(config.daemonAddress, daemonAddress);
    }
    await assert.rejects(
      async () => readDeploymentConfig(await configFile(rendered({ daemonAddress: 'http://relay.example.com' }))),
      /must use HTTPS/,
    );
  });

  it('rejects an address that is not a bare root', async () => {
    for (const daemonAddress of [
      'https://user:secret@relay.example.com',
      'https://relay.example.com/tenant',
      'https://relay.example.com/?team=a',
      'https://relay.example.com/#frag',
      'ftp://relay.example.com',
      'relay.example.com',
    ]) {
      await assert.rejects(
        async () => readDeploymentConfig(await configFile(rendered({ daemonAddress }))),
        /daemonAddress/,
        daemonAddress,
      );
    }
  });

  it('rejects an unrendered placeholder', async () => {
    await assert.rejects(
      async () =>
        readDeploymentConfig(await configFile(rendered({ dispatcherCommand: '__NEMO_RELAY_DISPATCHER_COMMAND__' }))),
      /was not rendered/,
    );
  });

  it('rejects a relative dispatcher command', async () => {
    await assert.rejects(
      async () => readDeploymentConfig(await configFile(rendered({ dispatcherCommand: 'nemo-relay' }))),
      /absolute/,
    );
  });

  it('rejects a schema it does not understand', async () => {
    await assert.rejects(
      async () => readDeploymentConfig(await configFile(rendered({ schema: 'nemo-relay-managed-pi-v0' }))),
      /schema/,
    );
  });

  it('rejects unreadable JSON', async () => {
    await assert.rejects(async () => readDeploymentConfig(await configFile('{ truncated')), /unreadable/);
  });
});

describe('client credential', () => {
  it('accepts an unpadded base64url encoding of exactly 32 bytes', () => {
    const token = Buffer.alloc(32, 7).toString('base64url');
    assert.equal(readCredential(token), token);
  });

  it('rejects absent, padded, short, and non-canonical values', () => {
    for (const value of [
      undefined,
      '',
      Buffer.alloc(32, 7).toString('base64'),
      Buffer.alloc(31, 7).toString('base64url'),
      Buffer.alloc(33, 7).toString('base64url'),
      'x'.repeat(43),
    ]) {
      assert.throws(() => readCredential(value), /base64url/, String(value));
    }
  });
});
