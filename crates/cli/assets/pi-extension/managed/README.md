<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay Managed Pi Extension v1

This directory is the immutable version 1 managed Pi package. Pi loads
`index.ts` and the modules under `src/` directly. The package has no runtime
dependencies beyond Node.js.

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
uses an OpenAI Completions, OpenAI Responses, or Anthropic Messages API and
shares one endpoint. This includes custom provider names. The registration
carries `x-nemo-relay-client-token` and names the model's catalog endpoint in
`x-nemo-relay-upstream-base-url`, so the daemon forwards to the endpoint Pi
would have called. The extension adds nothing to that header beyond the
configured endpoint. An endpoint the daemon would refuse, such as a private
address or cleartext off loopback, is left on its original route. Providers
containing an unsupported API remain on their original endpoint.

The files in this package changed with this release. An earlier bundle fails
validation and `refresh` never rewrites it. Generate the bundle again into a new
directory, provision its new digest, and point the launcher at the new path.
