#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

LLAMA_LIB_DIR="$HOME/builds/llama.cpp/build/bin"

# bench and experiments are not part of the binary; check that they still compile
odin check src/bench -no-entry-point -collection:lib=./lib
odin check src/experiments -no-entry-point -collection:lib=./lib

odin build src -out:./inference-odin.out -collection:lib=./lib \
    -extra-linker-flags:"-L${LLAMA_LIB_DIR} -Wl,-rpath,${LLAMA_LIB_DIR}" \
    "$@"
