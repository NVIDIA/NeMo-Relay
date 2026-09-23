---
name: create-release-tag
description: Create and push a signed, annotated NeMo Relay stable tag, then prepare an unpublished GitHub Release draft for review. Use when cutting a stable release; not for beta or release-candidate tags.
license: Apache-2.0
---

# Create a Release Tag

Require an explicit stable version in exact `<major>.<minor>.<patch>` form. Do
not infer it from metadata or a branch name, and do not edit package metadata
while tagging. The matching `release/<major>.<minor>` branch must exist on
`upstream`; Relay stable tags are raw SemVer, for example `0.9.0`, without `v`.

Before the destructive tag push, require explicit approval. Preserve user
changes: do not stash, discard, or switch away from a dirty checkout. Fetch the
release branch and tags, require `HEAD` to equal the upstream release branch,
and require the workspace version to equal the requested version. Verify both
local and remote tags are absent, create a signed annotated tag, verify it with
`git tag -v`, and push only `refs/tags/<tag>`. Stop on any failed check; never
force-update, replace, or delete tags.

After a successful push, create a GitHub Release only when the user explicitly
authorizes the draft. Use the prior published stable release returned by GitHub,
read the release-notes content at the new tag, and use `draft-release-notes`
evidence only to verify claims. Prepare `TITLE` and `NOTES_FILE`, then create
the draft with the command below. Present its URL for review; publishing is a
separate action.

## Create and Verify the Tag

Use `upstream` for `NVIDIA/NeMo-Relay`. Before requesting push approval, use
the following checks with the user-provided `VERSION`:

```bash
VERSION=<major.minor.patch>
RELEASE_BRANCH="release/$(printf '%s' "$VERSION" | cut -d. -f1,2)"
git ls-remote --exit-code --heads upstream "refs/heads/$RELEASE_BRANCH"
git fetch upstream "$RELEASE_BRANCH" --tags
test -z "$(git status --porcelain)"
git switch "$RELEASE_BRANCH"
git pull --ff-only --signoff upstream "$RELEASE_BRANCH"
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$(git rev-parse "upstream/$RELEASE_BRANCH")"
test "$(sed -nE 's/^version = "([^"]+)"$/\1/p' Cargo.toml | head -1)" = "$VERSION"
! git rev-parse --verify --quiet "refs/tags/$VERSION"
if git ls-remote --exit-code --tags upstream "refs/tags/$VERSION" >/dev/null; then
  echo "Error: tag already exists on upstream" >&2
  exit 1
else
  tag_lookup_status=$?
  case "$tag_lookup_status" in
    2) ;;
    *) echo "Error: remote tag lookup failed" >&2; exit "$tag_lookup_status" ;;
  esac
fi
```

After explicit approval, create and push only the verified tag:

```bash
git tag -s -a -m "NVIDIA NeMo Relay ${VERSION}" "$VERSION" HEAD
git tag -v "$VERSION"
git push upstream "refs/tags/$VERSION"
```

For an authorized GitHub Release draft, read the tagged
`docs/about-nemo-relay/release-notes/index.mdx`, find the previous published
stable release with `gh api repos/NVIDIA/NeMo-Relay/releases/latest`, and use
the tagged notes as the source for a concise body. Verify no release already
exists, then run:

```bash
gh release create "$VERSION" --repo NVIDIA/NeMo-Relay --title "$TITLE" --notes-file "$NOTES_FILE" --draft --verify-tag
```

Do not create a team announcement or publish the draft unless separately
requested.
