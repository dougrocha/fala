package main

import "core:c"
import "core:c/libc"
import "core:fmt"
import "core:os"
import "core:strings"
import "core:time"

import "lib:llama"

THINK_OFF :: "<think>\n\n</think>\n\n"

main :: proc() {
	llama.log_set(log_errors_only, nil)

	llama.backend_init()
	defer llama.backend_free()

	params := llama.model_default_params()

	model := llama.model_load_from_file("models/Qwen3-1.7B-Q8_0.gguf", params)
	defer llama.model_free(model)

	if model == nil {
		fmt.println("failed to load model")
		return
	}

	ctx_params := llama.context_default_params()
	ctx_params.n_ctx = 4096

	ctx := llama.init_from_model(model, ctx_params)
	if ctx == nil {
		fmt.println("failed to create context")
		return
	}
	defer llama.free(ctx)

	vocab := llama.model_get_vocab(model)
	if vocab == nil {
		fmt.println("failed to get vocab")
		return
	}

	sampler := llama.sampler_chain_init(llama.Sampler_Chain_Params{})
	defer llama.sampler_free(sampler)

	llama.sampler_chain_add(sampler, llama.sampler_init_top_k(40))
	llama.sampler_chain_add(sampler, llama.sampler_init_top_p(0.9, 1))
	llama.sampler_chain_add(sampler, llama.sampler_init_temp(0.7))
	llama.sampler_chain_add(sampler, llama.sampler_init_dist(llama.DEFAULT_SEED))

	tmpl := llama.model_chat_template(model, nil)
	mem := llama.get_memory(ctx)

	messages: [dynamic]llama.Chat_Message
	defer delete(messages)

	append(&messages, llama.Chat_Message{role = "system", content = "Answer in under 60 words."})

	formatted: [32768]u8
	prev_len: i32 = 0

	input_buf: [4096]u8

	answer: strings.Builder
	defer strings.builder_destroy(&answer)

	for {
		fmt.print("> ")

		n_input, err := os.read(os.stdin, input_buf[:])
		if err != nil || n_input <= 0 {
			break
		}

		line := strings.trim_space(string(input_buf[:n_input]))
		if line == "" {
			continue
		}

		append(&messages, llama.Chat_Message{role = "user", content = strings.clone_to_cstring(line)})

		new_len := llama.chat_apply_template(tmpl, raw_data(messages), c.size_t(len(messages)), true, &formatted[0], i32(len(formatted)))
		if new_len < 0 || new_len > i32(len(formatted)) {
			fmt.println("chat too long")
			break
		}

		prompt := fmt.aprintf("%s%s", string(formatted[prev_len:new_len]), THINK_OFF)
		c_prompt := strings.clone_to_cstring(prompt)

		count := -llama.tokenize(vocab, c_prompt, i32(len(prompt)), nil, 0, prev_len == 0, true)
		tokens := make([]llama.Token, count)
		llama.tokenize(vocab, c_prompt, i32(len(prompt)), raw_data(tokens), i32(len(tokens)), prev_len == 0, true)

		pp_start := time.tick_now()

		batch := llama.batch_get_one(raw_data(tokens), i32(len(tokens)))

		result := llama.decode(ctx, batch)
		if result != 0 {
			fmt.println("decode failed:", result)
			break
		}

		pp_ms := time.duration_milliseconds(time.tick_since(pp_start))

		strings.builder_reset(&answer)
		tg_start := time.tick_now()
		n_generated := 0

		for _ in 0 ..< 256 {
			next_token := llama.sampler_sample(sampler, ctx, -1)

			if llama.vocab_is_eog(vocab, next_token) {
				break
			}

			print_token_piece(vocab, next_token, &answer)
			n_generated += 1

			batch = llama.batch_get_one(&next_token, 1)

			result = llama.decode(ctx, batch)
			if result != 0 {
				fmt.println("decode failed:", result)
				break
			}
		}

		tg_ms := time.duration_milliseconds(time.tick_since(tg_start))

		fmt.println()
		fmt.printfln(
			"[prefill %d tok in %.1f ms | gen %d tok at %.0f tok/s | kv cells %d]",
			len(tokens),
			pp_ms,
			n_generated,
			f64(n_generated) / tg_ms * 1000,
			llama.memory_seq_pos_max(mem, 0) + 1,
		)

		content := fmt.aprintf("%s%s", THINK_OFF, strings.to_string(answer))
		append(&messages, llama.Chat_Message{role = "assistant", content = strings.clone_to_cstring(content)})

		prev_len = new_len + i32(len(content))

		delete(content)
		delete(tokens)
		delete(c_prompt)
		delete(prompt)
	}
}

print_token_piece :: proc(vocab: ^llama.Vocab, token: llama.Token, answer: ^strings.Builder) {
	buf: [128]u8

	n := llama.token_to_piece(vocab, token, &buf[0], i32(len(buf)), 0, false)

	if n > 0 {
		fmt.print(string(buf[:n]))
		strings.write_string(answer, string(buf[:n]))
	}
}

log_errors_only :: proc "c" (level: c.int, text: cstring, user_data: rawptr) {
	if level == 4 {
		libc.fprintf(libc.stderr, "%s", text)
	}
}
