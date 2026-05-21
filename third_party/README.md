<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Third-Party Integrations

NeMo Relay maintains some third-party integrations as patch sets applied to local
upstream checkouts under `third_party/`. The public wrapper commands stay at
the `scripts/` root, while their implementations live under
`scripts/third-party/`.

The tracked upstream sources live in [sources.lock](sources.lock). The NeMo Relay
patches live under [`../patches/`](../patches/).

Integration-specific setup, usage, and validation notes live next to this file:

- [Hermes Agent](README-hermes-agent.md)
- [LangChain](README-langchain.md)
- [LangChain NVIDIA](README-langchain-nvidia.md)
- [LangGraph](README-langgraph.md)
- [OpenClaw](README-openclaw.md)
- [opencode](README-opencode.md)

## Recommended Workflow

Bootstrap the tracked upstream checkouts from the manifest:

```bash
./scripts/bootstrap-third-party.sh
```

Apply the NeMo Relay integration patches:

```bash
./scripts/apply-patches.sh
```

Check whether the patches still apply cleanly without modifying your working
tree:

```bash
./scripts/apply-patches.sh --check
```

## What The Scripts Do

- `./scripts/bootstrap-third-party.sh` is the stable public wrapper for
  `scripts/third-party/bootstrap.sh`. It clones each upstream repository listed in
  `third_party/sources.lock` and checks out the pinned commit in a detached
  HEAD state.
- `./scripts/apply-patches.sh` is the stable public wrapper for
  `scripts/third-party/apply-patches.sh`. It applies the patch files from
  `patches/<name>/` to the
  corresponding local checkout.
- `./scripts/generate-patches.sh` is the stable public wrapper for
  `scripts/third-party/generate-patches.sh`. It regenerates the patch file for
  any third-party checkout with local changes.

`apply-patches.sh` refuses to apply patches on top of a dirty checkout. Commit,
stash, or discard local changes first, or use `--check` to validate patch
applicability against a clean detached checkout.

## Manual Clone And Apply

If you want to work on one integration manually instead of bootstrapping all of
them:

1. Find the upstream `url`, `path`, and pinned `commit` in
   [sources.lock](sources.lock).
2. Clone the upstream repository into the tracked path under `third_party/`.
3. Check out the pinned commit in detached HEAD mode.
4. Apply the patch files from the matching directory under `../patches/`.

Example for `langgraph`:

```bash
git clone https://github.com/langchain-ai/langgraph.git third_party/langgraph
git -C third_party/langgraph checkout --detach 5c9c1d598d65411317e0957a42cc3af681d395f8
git -C third_party/langgraph apply ../patches/langgraph/0001-add-nemo-relay-integration.patch
```

## Updating Patch Sets

After editing a local third-party checkout, regenerate the patch files with:

```bash
./scripts/generate-patches.sh
```

This writes updated patch files under `patches/<name>/` and verifies that each
generated patch still applies to a clean detached checkout of the local repo
HEAD.
