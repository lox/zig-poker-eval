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

version=""
readme_path="README.md"
zon_path="build.zig.zon"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --readme)
      readme_path="$2"
      shift 2
      ;;
    --zon)
      zon_path="$2"
      shift 2
      ;;
    *)
      if [[ -z "$version" ]]; then
        version="$1"
        shift
      else
        echo "Unexpected argument: $1" >&2
        exit 1
      fi
      ;;
  esac
done

if [[ -z "$version" ]]; then
  echo "Missing required version argument." >&2
  usage
  exit 1
fi

version="${version#v}"
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Invalid semantic version: $version" >&2
  exit 1
fi

if [[ ! -f "$readme_path" ]]; then
  echo "README file not found: $readme_path" >&2
  exit 1
fi

if [[ ! -f "$zon_path" ]]; then
  echo "Zig manifest not found: $zon_path" >&2
  exit 1
fi

tmp_readme="$(mktemp)"
tmp_zon="$(mktemp)"

trap 'rm -f "$tmp_readme" "$tmp_zon"' EXIT

if ! awk -v ver="$version" '
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
' "$readme_path" > "$tmp_readme"; then
  exit 1
fi

if ! awk -v ver="$version" '
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
' "$zon_path" > "$tmp_zon"; then
  exit 1
fi

mv "$tmp_readme" "$readme_path"
mv "$tmp_zon" "$zon_path"
