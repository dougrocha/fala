package main

import "core:thread"

import "lib:llama"

import "inference"

Worker :: struct {
	tmpl:  cstring,
	port:  int,
	queue: inference.Queue,
}

run_worker :: proc(port: int) {
	llama.log_set(inference.log_errors_only, nil)

	llama.backend_init()
	defer llama.backend_free()

	engine, ok := inference.engine_init("models/Qwen3-1.7B-Q8_0.gguf", 4096, inference.N_SEQ)
	if !ok {
		return
	}
	defer inference.engine_free(&engine)

	worker := Worker {
		tmpl = engine.tmpl,
		port = port,
	}
	defer delete(worker.queue.items)

	thread.create_and_start_with_poly_data(&worker, accept_loop)

	inference.run_loop(&engine, &worker.queue, port)
}
