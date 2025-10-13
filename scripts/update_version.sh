#!/usr/bin/env bash
# Synchronize version references across project files without requiring Python.

set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 <version> [--readme PATH] [--zon PATH]

Updates README install instructions and build.zig.zon manifest to the provided semantic version.
Leading 'v' is tolerated in the version argument.
EOF
}

VERSION=""
README_PATH="README.md"
ZON_PATH="build.zig.zon"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --readme)
      README_PATH="$2"
      shift 2
      ;;
    --zon)
      ZON_PATH="$2"
      shift 2
      ;;
    *)
      if [[ -z "$VERSION" ]]; then
        VERSION="$1"
        shift
      else
        echo "Unexpected argument: $1" >&2
        exit 1
      fi
      ;;
  esac
done

if [[ -z "$VERSION" ]]; then
  echo "Missing required version argument." >&2
  usage
  exit 1
fi

VERSION="${VERSION#v}"
if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Invalid semantic version: $VERSION" >&2
  exit 1
fi

if [[ ! -f "$README_PATH" ]]; then
  echo "README file not found: $README_PATH" >&2
  exit 1
fi

if [[ ! -f "$ZON_PATH" ]]; then
  echo "Zig manifest not found: $ZON_PATH" >&2
  exit 1
fi

tmp_readme="$(mktemp)"
tmp_zon="$(mktemp)"

trap 'rm -f "$tmp_readme" "$tmp_zon"' EXIT

if ! awk -v ver="$VERSION" '
  BEGIN { count = 0 }
  {
    if ($0 ~ /(ref=v)([0-9]+\.[0-9]+\.[0-9]+)/) {
      if (count == 0) {
        sub(/(ref=v)([0-9]+\.[0-9]+\.[0-9]+)/, "ref=v" ver)
        count++
      } else {
        print "Multiple README matches for version reference" > "/dev/stderr"
        exit 1
      }
    }
    print
  }
  END {
    if (count != 1) {
      print "Expected exactly one README version reference, updated " count > "/dev/stderr"
      exit 1
    }
  }
' "$README_PATH" > "$tmp_readme"; then
  exit 1
fi

if ! awk -v ver="$VERSION" '
  BEGIN { count = 0 }
  {
    if ($0 ~ /(\.version[[:space:]]*=[[:space:]]*")[0-9]+\.[0-9]+\.[0-9]+(")/) {
      if (count == 0) {
        sub(/(\.version[[:space:]]*=[[:space:]]*")[0-9]+\.[0-9]+\.[0-9]+(")/, ".version = \"" ver "\"")
        count++
      } else {
        print "Multiple build.zig.zon version entries encountered" > "/dev/stderr"
        exit 1
      }
    }
    print
  }
  END {
    if (count != 1) {
      print "Expected exactly one build.zig.zon version entry, updated " count > "/dev/stderr"
      exit 1
    }
  }
' "$ZON_PATH" > "$tmp_zon"; then
  exit 1
fi

mv "$tmp_readme" "$README_PATH"
mv "$tmp_zon" "$ZON_PATH"
