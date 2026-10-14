llama_dir := justfile_directory() / "vendor/llama.cpp"
llama_build := llama_dir / "build"
llama_bin := llama_build / "bin"
# pinned llama.cpp commit; change it to upgrade
llama_rev := "1ab7e5ad2d4e7295c94c3b966a3e0b70fa365865"

# check bench and experiments, then build the binary
build *args: llama-build check
    mkdir -p builds
    odin build src -out:./builds/fala -collection:lib=./lib \
        -extra-linker-flags:"-L{{llama_bin}} -Wl,-rpath,{{llama_bin}}" \
        {{args}}

# bench and experiments are not part of the binary; check that they still compile
check:
    odin check src/bench -no-entry-point -collection:lib=./lib
    odin check src/experiments -no-entry-point -collection:lib=./lib

# clone llama.cpp at llama_rev and build it (CUDA when an NVIDIA GPU is present); skip when already built at llama_rev
llama-build:
    #!/usr/bin/env bash
    set -euo pipefail
    stamp="{{llama_build}}/.rev"
    if [[ -f "$stamp" && "$(cat "$stamp")" == "{{llama_rev}}" ]]; then
        exit 0
    fi
    if [[ ! -d "{{llama_dir}}/.git" ]]; then
        git clone --filter=blob:none https://github.com/ggml-org/llama.cpp "{{llama_dir}}"
    fi
    if ! git -C "{{llama_dir}}" cat-file -e "{{llama_rev}}^{commit}" 2>/dev/null; then
        git -C "{{llama_dir}}" fetch origin "{{llama_rev}}"
    fi
    git -C "{{llama_dir}}" checkout --detach "{{llama_rev}}"
    cmake_args=(-DCMAKE_BUILD_TYPE=Release)
    if lspci 2>/dev/null | grep -qi nvidia; then
        cmake_args+=(-DGGML_CUDA=ON)
    fi
    cmake -S "{{llama_dir}}" -B "{{llama_build}}" "${cmake_args[@]}"
    cmake --build "{{llama_build}}" -j
    "{{llama_bin}}/llama-server" --version
    echo "{{llama_rev}}" > "$stamp"

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
    ./builds/fala {{args}}
