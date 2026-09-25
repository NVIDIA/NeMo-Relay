---
name: create-beta-tag
description: Create and push a signed, annotated NeMo Relay beta tag from its release branch or validated main. Use when cutting a beta tag; not for RC or stable tags.
license: Apache-2.0
---

# Create a Beta Tag

Require an explicit stable base version in `<major>.<minor>[.<patch>]` form;
treat two components as patch `0`. Do not infer it from the branch or package
metadata, and do not edit version files while creating the tag.

Prefer `release/<major>.<minor>`. If that remote branch does not exist, use
`main` only when its workspace version equals the requested base version. Select
the next unused positive beta number for that exact base version unless the user
provides one. Relay tags are raw SemVer: `0.9.0-beta.1`, never `v0.9.0-beta.1`.

Before the destructive tag push, require explicit approval. Preserve user
changes: do not stash, discard, or switch away from a dirty checkout. Fetch the
source branch and tags, require `HEAD` to equal `upstream/<source-branch>`, and
require the workspace version to equal the stable base version. Verify both
local and remote tags are absent, create a signed annotated tag, verify it with
`git tag -v`, and push only `refs/tags/<tag>`. Stop on any failed check; never
force-update, replace, or delete tags.

## Select and Verify the Source

Use `upstream` for `NVIDIA/NeMo-Relay`. The version and optional beta number
are user inputs. Validate a supplied number as a positive integer; otherwise
derive the next number after fetching tags for the selected source branch.

```bash
BASE_VERSION=<major.minor[.patch]>
[[ "$BASE_VERSION" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]] || { echo "Error: invalid base version" >&2; exit 1; }
if [[ "$BASE_VERSION" =~ ^[0-9]+\.[0-9]+$ ]]; then BASE_VERSION="$BASE_VERSION.0"; fi
RELEASE_BRANCH="release/$(printf '%s' "$BASE_VERSION" | cut -d. -f1,2)"
SOURCE_BRANCH="$RELEASE_BRANCH"
if git ls-remote --exit-code --heads upstream "refs/heads/$SOURCE_BRANCH" >/dev/null; then
  :
else
  lookup_status=$?
  case "$lookup_status" in
    2) SOURCE_BRANCH=main ;;
    *) echo "Error: release branch lookup failed" >&2; exit "$lookup_status" ;;
  esac
fi
git fetch upstream "$SOURCE_BRANCH" --tags

LAST_BETA="$(git tag --list "${BASE_VERSION}-beta.*" | sed -nE "s/^${BASE_VERSION//./\\.}-beta\.([1-9][0-9]*)$/\\1/p" | sort -n | tail -1)"
BETA_NUMBER="${BETA_NUMBER:-$(( ${LAST_BETA:-0} + 1 ))}"
[[ "$BETA_NUMBER" =~ ^[1-9][0-9]*$ ]]
TAG="${BASE_VERSION}-beta.${BETA_NUMBER}"
```

From a clean checkout of `SOURCE_BRANCH`, verify the branch is current, the
workspace version matches `BASE_VERSION`, and the tag is absent locally and on
`upstream`. When `SOURCE_BRANCH` is `main`, make the same version check against
`upstream/main` before switching. After the user authorizes the push:

```bash
git tag -s -a -m "NVIDIA NeMo Relay ${TAG}" "$TAG" HEAD
git tag -v "$TAG"
git push upstream "refs/tags/$TAG"
```
