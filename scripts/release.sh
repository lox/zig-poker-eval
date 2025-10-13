#!/usr/bin/env bash
# Orchestrate the release workflow: enforce clean state, bump version, run tests, and push.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -n "$(git status --porcelain)" ]; then
  echo "Working tree must be clean before releasing." >&2
  exit 1
fi

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$CURRENT_BRANCH" != "main" ]; then
  echo "Releases must be performed from the main branch (current: $CURRENT_BRANCH)." >&2
  exit 1
fi

git fetch --tags --prune --force

echo "Running test suite before version bump..."
zig build test --summary all

NEW_VERSION="$(svu next --always)"
NEW_VERSION_NO_V="${NEW_VERSION#v}"
echo "Bumping version to ${NEW_VERSION}"

"$ROOT_DIR/scripts/update_version.sh" "$NEW_VERSION_NO_V"

echo "Running test suite after version bump..."
zig build test --summary all

git status --short

git add README.md build.zig.zon
git commit -m "chore: release ${NEW_VERSION}"
git push origin HEAD

echo "Release commit pushed. GitHub release workflow will publish tag ${NEW_VERSION}."
