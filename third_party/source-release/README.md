<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Source Release Artifacts

This directory contains exact, as-received source/package artifacts for
third-party dependencies that require source availability review. The artifacts
are checked in without NVIDIA modifications and are accompanied by checksum
metadata.

Notices and license text for the Node.js dependency surface are maintained in
[`../../ATTRIBUTIONS-Node.md`](../../ATTRIBUTIONS-Node.md).

## npm Source Packages

| Package | Version | License | Dependency path | Source package | Attribution |
| --- | --- | --- | --- | --- | --- |
| `web-push` | `3.6.7` | `MPL-2.0` | `nemo-flow-openclaw -> openclaw -> web-push` | [`npm/web-push-3.6.7.tgz`](npm/web-push-3.6.7.tgz) | [`../../ATTRIBUTIONS-Node.md`](../../ATTRIBUTIONS-Node.md) |
| `tar` | `7.5.13` | `BlueOak-1.0.0` | `nemo-flow-openclaw -> openclaw -> tar` | [`npm/tar-7.5.13.tgz`](npm/tar-7.5.13.tgz) | [`../../ATTRIBUTIONS-Node.md`](../../ATTRIBUTIONS-Node.md) |
| `yallist` | `5.0.0` | `BlueOak-1.0.0` | `nemo-flow-openclaw -> openclaw -> tar -> yallist` | [`npm/yallist-5.0.0.tgz`](npm/yallist-5.0.0.tgz) | [`../../ATTRIBUTIONS-Node.md`](../../ATTRIBUTIONS-Node.md) |

The npm package archives are the lockfile-resolved package artifacts from the
public npm registry. They include the upstream license files and package
contents used by the Node.js package manager.

Checksums for the checked-in artifacts are recorded in
[`npm/SHA256SUMS`](npm/SHA256SUMS).

## npm Lockfile Provenance

| Package | Resolved package URL | npm integrity |
| --- | --- | --- |
| `web-push@3.6.7` | `https://registry.npmjs.org/web-push/-/web-push-3.6.7.tgz` | `sha512-OpiIUe8cuGjrj3mMBFWY+e4MMIkW3SVT+7vEIjvD9kejGUypv8GPDf84JdPWskK8zMRIJ6xYGm+Kxr8YkPyA0A==` |
| `tar@7.5.13` | `https://registry.npmjs.org/tar/-/tar-7.5.13.tgz` | `sha512-tOG/7GyXpFevhXVh8jOPJrmtRpOTsYqUIkVdVooZYJS/z8WhfQUX8RJILmeuJNinGAMSu1veBr4asSHFt5/hng==` |
| `yallist@5.0.0` | `https://registry.npmjs.org/yallist/-/yallist-5.0.0.tgz` | `sha512-YgvUTfwqyc7UXVMrB+SImsVYSmTS8X/tSrtdNZMImM+n7+QTriRXyXim0mBrTXNeqzVF0KWGgHPeiyViFFrNDw==` |
