llama_lib_dir := env("BUILD_DIR", env("HOME") / "builds") / "llama.cpp/build/bin"

# check bench and experiments, then build the binary
build *args: check
    odin build src -out:./inference-odin.out -collection:lib=./lib \
        -extra-linker-flags:"-L{{llama_lib_dir}} -Wl,-rpath,{{llama_lib_dir}}" \
        {{args}}

# bench and experiments are not part of the binary; check that they still compile
check:
    odin check src/bench -no-entry-point -collection:lib=./lib
    odin check src/experiments -no-entry-point -collection:lib=./lib

# start llama-server with a model (default: Qwen3 1.7B Q8_0)
llama model="models/Qwen3-1.7B-Q8_0.gguf":
    llama-server \
        --model "{{model}}" \
        --ctx-size 4096 \
        --reasoning off \
        --no-cache-prompt \
        -lv 4

# build, then run the binary (example: just run worker 8081)
run *args: build
    ./inference-odin.out {{args}}
