package main

import "core:c"
import "core:fmt"
import "core:os"
import "core:strings"
import "core:time"

import "lib:llama"

run_chat :: proc(engine: ^Engine) {
	sampler := make_sampler(false)
	defer llama.sampler_free(sampler)

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

		append(
			&messages,
			llama.Chat_Message{role = "user", content = strings.clone_to_cstring(line)},
		)

		new_len := llama.chat_apply_template(
			engine.tmpl,
			raw_data(messages),
			c.size_t(len(messages)),
			true,
			&formatted[0],
			i32(len(formatted)),
		)
		if new_len < 0 || new_len > i32(len(formatted)) {
			fmt.println("chat too long")
			break
		}

		prompt := fmt.aprintf("%s%s", string(formatted[prev_len:new_len]), THINK_OFF)
		tokens := tokenize_text(engine.vocab, prompt, prev_len == 0)
		n_past := llama.memory_seq_pos_max(engine.mem, 0) + 1

		pp_start := time.tick_now()

		engine.batch.n_tokens = 0
		last := batch_add_tokens(&engine.batch, tokens, n_past, 0)
		n_past += i32(len(tokens))

		result := llama.decode(engine.ctx, engine.batch)
		if result != 0 {
			fmt.println("decode failed:", result)
			break
		}

		pp_ms := time.duration_milliseconds(time.tick_since(pp_start))

		strings.builder_reset(&answer)
		tg_start := time.tick_now()
		n_generated := 0

		for _ in 0 ..< 256 {
			next_token := llama.sampler_sample(sampler, engine.ctx, last)

			if llama.vocab_is_eog(engine.vocab, next_token) {
				break
			}

			fmt.print(append_token_piece(engine.vocab, next_token, &answer))
			n_generated += 1

			engine.batch.n_tokens = 0
			last = batch_add_tokens(&engine.batch, {next_token}, n_past, 0)
			n_past += 1

			result = llama.decode(engine.ctx, engine.batch)
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
			llama.memory_seq_pos_max(engine.mem, 0) + 1,
		)

		content := fmt.aprintf("%s%s", THINK_OFF, strings.to_string(answer))
		append(
			&messages,
			llama.Chat_Message{role = "assistant", content = strings.clone_to_cstring(content)},
		)

		prev_len = new_len + i32(len(content))

		delete(content)
		delete(tokens)
		delete(prompt)
	}
}
