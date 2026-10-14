llama_dir := justfile_directory() / "vendor/llama.cpp"
llama_build := llama_dir / "build"
llama_bin := llama_build / "bin"

# check bench and experiments, then build the binary
build *args: check
    odin build src -out:./inference-odin.out -collection:lib=./lib \
        -extra-linker-flags:"-L{{llama_bin}} -Wl,-rpath,{{llama_bin}}" \
        {{args}}

# bench and experiments are not part of the binary; check that they still compile
check:
    odin check src/bench -no-entry-point -collection:lib=./lib
    odin check src/experiments -no-entry-point -collection:lib=./lib

# clone or update llama.cpp at master and build it (CUDA when an NVIDIA GPU is present)
llama-build:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -d "{{llama_dir}}/.git" ]]; then
        git -C "{{llama_dir}}" fetch --prune origin master
        git -C "{{llama_dir}}" reset --hard origin/master
    else
        git clone --filter=blob:none --branch master https://github.com/ggml-org/llama.cpp "{{llama_dir}}"
    fi
    cmake_args=(-DCMAKE_BUILD_TYPE=Release)
    if lspci 2>/dev/null | grep -qi nvidia; then
        cmake_args+=(-DGGML_CUDA=ON)
    fi
    cmake -S "{{llama_dir}}" -B "{{llama_build}}" "${cmake_args[@]}"
    cmake --build "{{llama_build}}" -j
    "{{llama_bin}}/llama-server" --version

# start llama-server with a model (default: Qwen3 1.7B Q8_0)
llama model="models/Qwen3-1.7B-Q8_0.gguf":
    {{llama_bin}}/llama-server \
        --model "{{model}}" \
        --ctx-size 4096 \
        --reasoning off \
        --no-cache-prompt \
        -lv 4

# build, then run the binary (example: just run worker 8081)
run *args: build
    ./inference-odin.out {{args}}
