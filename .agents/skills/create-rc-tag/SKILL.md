---
name: create-rc-tag
description: Create and push a signed, annotated NeMo Relay release-candidate tag from its validated release branch. Use when cutting an RC tag; not for code freezes or stable tags.
license: Apache-2.0
---

# Create an RC Tag

Require an explicit stable base version in `<major>.<minor>[.<patch>]` form;
treat two components as patch `0`. Do not infer it from the branch or package
metadata, and do not edit version files while creating the tag. The matching
`release/<major>.<minor>` branch must exist on `upstream`.

Select the next unused positive RC number for that exact base version unless the
user supplies one. Relay uses raw-SemVer tags such as `0.9.0-rc.1`, without a
leading `v`.

Before the destructive tag push, require explicit approval. Preserve user
changes: do not stash, discard, or switch away from a dirty checkout. Fetch the
release branch and tags, require `HEAD` to equal the upstream release branch,
and require the workspace version to equal the stable base version. Verify that
both local and remote tags are absent, create a signed annotated tag, verify it
with `git tag -v`, and push only `refs/tags/<tag>`. Stop on any failed check;
never force-update, replace, or delete tags.

## Select and Verify the Source

Use `upstream` for `NVIDIA/NeMo-Relay`. The version and optional RC number are
user inputs. Validate a supplied number as a positive integer; otherwise derive
the next number after fetching the release branch and tags.

```bash
BASE_VERSION=<major.minor[.patch]>
[[ "$BASE_VERSION" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]] || { echo "Error: invalid base version" >&2; exit 1; }
if [[ "$BASE_VERSION" =~ ^[0-9]+\.[0-9]+$ ]]; then BASE_VERSION="$BASE_VERSION.0"; fi
RELEASE_BRANCH="release/$(printf '%s' "$BASE_VERSION" | cut -d. -f1,2)"
git ls-remote --exit-code --heads upstream "refs/heads/$RELEASE_BRANCH"
git fetch upstream "$RELEASE_BRANCH" --tags

LAST_RC="$(git tag --list "${BASE_VERSION}-rc.*" | sed -nE "s/^${BASE_VERSION//./\\.}-rc\.([1-9][0-9]*)$/\\1/p" | sort -n | tail -1)"
RC_NUMBER="${RC_NUMBER:-$(( ${LAST_RC:-0} + 1 ))}"
[[ "$RC_NUMBER" =~ ^[1-9][0-9]*$ ]]
TAG="${BASE_VERSION}-rc.${RC_NUMBER}"
```

From a clean checkout of `RELEASE_BRANCH`, require `HEAD` to match
`upstream/$RELEASE_BRANCH`, verify the workspace version matches `BASE_VERSION`,
and check that `TAG` is absent locally and on `upstream`. After the user
authorizes the push:

```bash
git tag -s -a -m "NVIDIA NeMo Relay ${TAG}" "$TAG" HEAD
git tag -v "$TAG"
git push upstream "refs/tags/$TAG"
```
