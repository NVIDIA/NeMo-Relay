// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFileSync } from 'node:fs';
import { isAbsolute } from 'node:path';

import { errorDetail, isRecord } from '../gateway.ts';

export const CLIENT_TOKEN_ENV = 'NEMO_RELAY_CLIENT_TOKEN';
export const CONFIG_SCHEMA = 'nemo-relay-managed-pi-v1';

const CONFIG_PLACEHOLDER_PREFIX = '__NEMO_RELAY_';
const CREDENTIAL_PATTERN = /^[A-Za-z0-9_-]{43}$/;
const CREDENTIAL_BYTES = 32;

export type DeploymentConfig = {
  schema: typeof CONFIG_SCHEMA;
  daemonAddress: string;
  dispatcherCommand: string;
};

export function readDeploymentConfig(location: URL): DeploymentConfig {
  let parsed: unknown;
  try {
    parsed = JSON.parse(readFileSync(location, 'utf8'));
  } catch (error) {
    throw new Error(`managed pi configuration is unreadable: ${errorDetail(error)}`);
  }
  if (!isRecord(parsed) || parsed.schema !== CONFIG_SCHEMA) {
    throw new Error(`managed pi configuration must use schema ${CONFIG_SCHEMA}`);
  }
  const daemonAddress = renderedString(parsed.daemonAddress, 'daemonAddress');
  const dispatcherCommand = renderedString(parsed.dispatcherCommand, 'dispatcherCommand');
  if (!isAbsolute(dispatcherCommand)) {
    throw new Error('managed pi dispatcherCommand must be an absolute administrator-owned path');
  }
  return {
    schema: CONFIG_SCHEMA,
    daemonAddress: rootDaemonAddress(daemonAddress),
    dispatcherCommand,
  };
}

function renderedString(value: unknown, field: string): string {
  if (
    typeof value !== 'string' ||
    value.length === 0 ||
    value.includes('\0') ||
    value.startsWith(CONFIG_PLACEHOLDER_PREFIX)
  ) {
    throw new Error(`managed pi ${field} was not rendered by the administrator`);
  }
  return value;
}

function rootDaemonAddress(value: string): string {
  let daemon: URL;
  try {
    daemon = new URL(value);
  } catch {
    throw new Error('managed pi daemonAddress must be an absolute HTTP(S) URL');
  }
  if (
    (daemon.protocol !== 'http:' && daemon.protocol !== 'https:') ||
    daemon.username !== '' ||
    daemon.password !== '' ||
    daemon.search !== '' ||
    daemon.hash !== '' ||
    (daemon.pathname !== '' && daemon.pathname !== '/')
  ) {
    throw new Error('managed pi daemonAddress must be a root HTTP(S) URL without credentials, query, or fragment');
  }
  const host = daemon.hostname.replace(/^\[|\]$/g, '');
  const loopback = host === 'localhost' || host === '::1' || /^127(?:\.\d{1,3}){3}$/.test(host);
  if (daemon.protocol === 'http:' && !loopback) {
    throw new Error('managed pi daemonAddress must use HTTPS unless it names a loopback host');
  }
  return daemon.origin;
}

export function readCredential(value: string | undefined): string {
  if (
    value === undefined ||
    !CREDENTIAL_PATTERN.test(value) ||
    Buffer.from(value, 'base64url').length !== CREDENTIAL_BYTES ||
    Buffer.from(value, 'base64url').toString('base64url') !== value
  ) {
    throw new Error(
      `${CLIENT_TOKEN_ENV} must be an unpadded base64url credential encoding exactly ${CREDENTIAL_BYTES} bytes`,
    );
  }
  return value;
}
