package main

import "core:fmt"
import "core:os"
import "core:strconv"

import "lib:llama"

main :: proc() {
	args := os.args

	if len(args) < 3 {
		fmt.println("usage: inference-odin.out worker <port> | router <port> <worker ports...>")
		return
	}

	port, port_ok := strconv.parse_int(args[2])
	if !port_ok {
		fmt.println("invalid port:", args[2])
		return
	}

	if args[1] == "worker" {
		run_worker(port)
		return
	}

	if args[1] == "router" {
		if len(args) < 4 {
			fmt.println("usage: inference-odin.out router <port> <worker ports...>")
			return
		}

		worker_ports := make([]int, len(args) - 3)
		defer delete(worker_ports)

		for worker_arg, i in args[3:] {
			worker_port, worker_ok := strconv.parse_int(worker_arg)
			if !worker_ok {
				fmt.println("invalid worker port:", worker_arg)
				return
			}
			worker_ports[i] = worker_port
		}

		run_router(port, worker_ports)
		return
	}

	fmt.println("unknown mode:", args[1])
}

run_worker :: proc(port: int) {
	llama.log_set(log_errors_only, nil)

	llama.backend_init()
	defer llama.backend_free()

	engine, ok := engine_init("models/Qwen3-1.7B-Q8_0.gguf", 4096, N_SEQ)
	if !ok {
		return
	}
	defer engine_free(&engine)

	run_server(&engine, port)
}
