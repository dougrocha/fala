package main

import "lib:llama"

main :: proc() {
	llama.log_set(log_errors_only, nil)

	llama.backend_init()
	defer llama.backend_free()

	engine, ok := engine_init("models/Qwen3-1.7B-Q8_0.gguf", 4096, N_SEQ)
	if !ok {
		return
	}
	defer engine_free(&engine)

	run_continuous(&engine)
}
