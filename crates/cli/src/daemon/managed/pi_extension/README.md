<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay Managed Pi Extension v1

This directory is the immutable version 1 managed Pi package. Pi loads
`index.ts` directly; the package has no runtime dependencies beyond Node.js.

The administrator-owned launcher must load only this managed extension:

```bash
pi --no-extensions -e /srv/nemo-relay/nemo-relay-managed-v1/pi/extension-v1/index.ts
```

`--no-extensions` is required. It preserves the explicitly loaded `-e`
extension while preventing user, project, and discovered extensions from
running alongside the managed policy boundary.

The managed-bundle renderer replaces the two complete JSON string values in
`managed-config.json`:

- `__NEMO_RELAY_DAEMON_ADDRESS__` becomes the fixed root daemon URL.
- `__NEMO_RELAY_DISPATCHER_COMMAND__` becomes the fixed administrator-owned
  dispatcher path.

The renderer must JSON-encode replacement values. It must not perform raw text
substitution. An extension installed with either placeholder still present
fails closed.

At runtime, the extension reads `NEMO_RELAY_CLIENT_TOKEN` from the process
environment. No credential, user identity, machine fingerprint, generation
identifier, or Relay binary version is written into this package.

The extension redirects the selected Pi provider when every known sibling model
uses an OpenAI Completions, OpenAI Responses, or Anthropic Messages API. This
includes custom provider names, but the administrator-managed worker upstream
remains authoritative: the extension never sends a per-user upstream URL or
routing header. Providers containing an unsupported API remain on their
original endpoint.
