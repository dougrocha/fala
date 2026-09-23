package main

import "core:fmt"
import "core:strings"
import "core:time"

import "lib:llama"

run_batch :: proc(engine: ^Engine) {
	prompts := [2]cstring{"What is the capital of France?", "Write one short sentence about the ocean."}

	sampler := make_sampler()
	defer llama.sampler_free(sampler)

	tmpl := llama.model_chat_template(engine.model, nil)

	batch := llama.batch_init(512, 0, 1)
	defer llama.batch_free(batch)

	last_index: [2]i32

	for user, seq in prompts {
		prompt := format_chat(tmpl, user)
		tokens := tokenize_text(engine.vocab, prompt, true)

		for token, i in tokens {
			batch_add(&batch, token, i32(i), i32(seq), i == len(tokens) - 1)
		}

		last_index[seq] = batch.n_tokens - 1
		fmt.printfln("seq %d: %d prompt tokens", seq, len(tokens))

		delete(tokens)
		delete(prompt)
	}

	result := llama.decode(engine.ctx, batch)
	if result != 0 {
		fmt.println("decode failed:", result)
		return
	}

	fmt.println("prefill complete:", batch.n_tokens, "tokens in 1 decode")

	answers: [2]strings.Builder
	defer for &answer in answers {
		strings.builder_destroy(&answer)
	}

	n_pos: [2]i32
	n_gen: [2]int
	done: [2]bool

	n_pos[0] = last_index[0] + 1
	n_pos[1] = last_index[1] - last_index[0]

	steps := 0
	gen_start := time.tick_now()

	for {
		batch.n_tokens = 0

		for seq in 0 ..< 2 {
			if done[seq] {
				continue
			}

			token := llama.sampler_sample(sampler, engine.ctx, last_index[seq])

			if llama.vocab_is_eog(engine.vocab, token) || n_gen[seq] == 256 {
				done[seq] = true
				continue
			}

			append_token_piece(engine.vocab, token, &answers[seq])

			last_index[seq] = batch.n_tokens
			batch_add(&batch, token, n_pos[seq], i32(seq), true)

			n_pos[seq] += 1
			n_gen[seq] += 1
		}

		if batch.n_tokens == 0 {
			break
		}

		result = llama.decode(engine.ctx, batch)
		if result != 0 {
			fmt.println("decode failed:", result)
			return
		}

		steps += 1
	}

	gen_ms := time.duration_milliseconds(time.tick_since(gen_start))
	total := n_gen[0] + n_gen[1]

	for seq in 0 ..< 2 {
		fmt.printfln("\nseq %d (%d tokens): %s", seq, n_gen[seq], strings.to_string(answers[seq]))
	}

	fmt.printfln("\n%d tokens in %d decode steps, %.1f ms (%.0f tok/s total)", total, steps, gen_ms, f64(total) / gen_ms * 1000)
}

format_chat :: proc(tmpl: cstring, user: cstring) -> string {
	messages := [2]llama.Chat_Message {
		{role = "system", content = "Answer in under 60 words."},
		{role = "user", content = user},
	}

	buf: [4096]u8
	n := llama.chat_apply_template(tmpl, &messages[0], len(messages), true, &buf[0], i32(len(buf)))

	if n < 0 || n > i32(len(buf)) {
		return ""
	}

	return fmt.aprintf("%s%s", string(buf[:n]), THINK_OFF)
}
