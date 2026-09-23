package main

import "core:c"
import "core:c/libc"
import "core:fmt"
import "core:os"
import "core:strings"

import "lib:llama"

main :: proc() {
	if len(os.args) < 2 {
		fmt.println("usage: inference-odin \"your message\"")
		return
	}

	llama.log_set(log_errors_only, nil)

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
	ctx_params.n_ctx = 4096

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

	c_user := strings.clone_to_cstring(os.args[1])
	defer delete(c_user)

	messages := [2]llama.Chat_Message {
		{role = "system", content = "Answer in under 60 words."},
		{role = "user", content = c_user},
	}

	tmpl := llama.model_chat_template(model, nil)

	prompt_buf: [4096]u8
	n_prompt := llama.chat_apply_template(tmpl, &messages[0], len(messages), true, &prompt_buf[0], i32(len(prompt_buf)))

	if n_prompt < 0 || n_prompt > i32(len(prompt_buf)) {
		fmt.println("chat template failed:", n_prompt)
		return
	}

	prompt := fmt.aprintf("%s<think>\n\n</think>\n\n", string(prompt_buf[:n_prompt]))
	defer delete(prompt)

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

	batch := llama.batch_get_one(raw_data(tokens), i32(len(tokens)))

	result := llama.decode(ctx, batch)
	if result != 0 {
		fmt.println("decode failed:", result)
		return
	}
	fmt.println("prefill complete")

	sampler := llama.sampler_chain_init(llama.Sampler_Chain_Params{})
	defer llama.sampler_free(sampler)

	llama.sampler_chain_add(sampler, llama.sampler_init_greedy())

	for _ in 0 ..< 100 {
		next_token := llama.sampler_sample(sampler, ctx, -1)

		if llama.vocab_is_eog(vocab, next_token) {
			break
		}

		print_token_piece(vocab, next_token)

		batch = llama.batch_get_one(&next_token, 1)

		result = llama.decode(ctx, batch)

		if result != 0 {
			fmt.println("decode failed:", result)
			return
		}
	}

	fmt.println()
}

print_token_piece :: proc(vocab: ^llama.Vocab, token: llama.Token) {
	buf: [128]u8

	n := llama.token_to_piece(vocab, token, &buf[0], i32(len(buf)), 0, false)

	if n > 0 {
		fmt.print(string(buf[:n]))
	}
}

log_errors_only :: proc "c" (level: c.int, text: cstring, user_data: rawptr) {
	if level == 4 {
		libc.fprintf(libc.stderr, "%s", text)
	}
}
