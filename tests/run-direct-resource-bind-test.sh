#!/usr/bin/env bash
set -euo pipefail

# Usage: tests/run-direct-resource-bind-test.sh /path/to/mesa /path/to/mesa-build
MESA_SOURCE=${1:?Mesa source directory required}
MESA_BUILD=${2:?Mesa build directory required}
HERE=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

${CC:-clang} -std=c11 -Wno-deprecated-declarations \
  -I"$MESA_SOURCE/src/gallium/include" \
  -I"$MESA_SOURCE/src/gallium/auxiliary" \
  -I"$MESA_SOURCE/src" -I"$MESA_SOURCE/include" \
  -I"$MESA_BUILD/include" -I"$MESA_BUILD/src" \
  -I"$HERE/src/gallium/drivers/rdmetal" \
  "$HERE/tests/test_direct_resource_bind.c" \
  "$HERE/src/gallium/drivers/rdmetal/rdmetal_direct_resource.c" \
  -o "$OUT/test-direct-resource-bind"
"$OUT/test-direct-resource-bind"
