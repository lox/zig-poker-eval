#!/usr/bin/env bash
# Orchestrate the release workflow: enforce clean state, bump version, run tests, and push.

set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

if [ -n "$(git status --porcelain)" ]; then
  echo "Working tree must be clean before releasing." >&2
  exit 1
fi

current_branch="$(git rev-parse --abbrev-ref HEAD)"
if [ "$current_branch" != "main" ]; then
  echo "Releases must be performed from the main branch (current: $current_branch)." >&2
  exit 1
fi

git fetch --tags --prune --force

echo "Running test suite before version bump..."
zig build test --summary all

new_version="$(svu next --always)"
new_version_no_v="${new_version#v}"
echo "Bumping version to ${new_version}"

"$root_dir/scripts/update_version.sh" "$new_version_no_v"

echo "Running test suite after version bump..."
zig build test --summary all

git status --short

git add README.md build.zig.zon
git commit -m "chore: release ${new_version}"
git push origin HEAD

echo "Release commit pushed. GitHub release workflow will publish tag ${new_version}."
