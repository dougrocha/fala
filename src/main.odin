package main

import "core:fmt"
import "core:strings"

import "lib:llama"

main :: proc() {
	llama.backend_init()
	defer llama.backend_free()
	fmt.println("llama backend initialized")

	params := llama.model_default_params()

	fmt.println("gpu layers:", params.n_gpu_layers)
	fmt.println("main gpu:", params.main_gpu)
	fmt.println("vocab only:", params.vocab_only)

	model := llama.model_load_from_file("models/Qwen3-1.7B-Q8_0.gguf", params)
	defer llama.model_free(model)

	if model == nil {
		fmt.println("failed to load model")
		return
	}

	fmt.println("model loaded")

	ctx_params := llama.context_default_params()

	ctx := llama.init_from_model(model, ctx_params)
	if ctx == nil {
		fmt.println("failed to create context")
		return
	}
	defer llama.free(ctx)

	fmt.println("context created")

	vocab := llama.model_get_vocab(model)
	if vocab == nil {
		fmt.println("failed to get vocab")
		return
	}

	prompt := "Hello World! This is inferencing, stuff."
	c_prompt := strings.clone_to_cstring(prompt)

	count := -llama.tokenize(vocab, c_prompt, i32(len(prompt)), nil, 0, true, true)

	tokens := make([]llama.Token, count)
	defer delete(tokens)

	n := llama.tokenize(
		vocab,
		c_prompt,
		i32(len(prompt)),
		raw_data(tokens),
		i32(len(tokens)),
		true,
		true,
	)

	fmt.println("token count:", n)
	fmt.println("tokens:", tokens)


	buf: [128]u8
	for token in tokens {
		n := llama.token_to_piece(vocab, token, &buf[0], i32(len(buf)), 0, false)

		if n > 0 {
			piece := string(buf[:n])
			fmt.printfln("%v -> %q", token, piece)
		}
	}
}
