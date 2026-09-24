package main

import "core:fmt"
import "core:strings"
import "core:time"

import "lib:llama"

N_SEQ :: 4

PROMPTS := [8]cstring {
	"What is the capital of France?",
	"Write one short sentence about the ocean.",
	"Explain how a bicycle chain works.",
	"Describe a rainy day in a city.",
	"What does a CPU do?",
	"Give three tips for learning to cook.",
	"Why is the sky blue?",
	"Describe the taste of fresh bread.",
}

run_batch :: proc(engine: ^Engine) {
	sampler := make_sampler(false)
	defer llama.sampler_free(sampler)

	batch := &engine.batch
	batch.n_tokens = 0

	last_index: [N_SEQ]i32
	n_pos: [N_SEQ]i32

	for seq in 0 ..< N_SEQ {
		prompt := format_chat(engine, PROMPTS[seq % len(PROMPTS)])
		tokens := tokenize_text(engine.vocab, prompt, true)

		last_index[seq] = batch_add_tokens(batch, tokens, 0, i32(seq))
		n_pos[seq] = i32(len(tokens))
		fmt.printfln("seq %d: %d prompt tokens", seq, len(tokens))

		delete(tokens)
		delete(prompt)
	}

	result := llama.decode(engine.ctx, batch^)
	if result != 0 {
		fmt.println("decode failed:", result)
		return
	}

	fmt.println("prefill complete:", batch.n_tokens, "tokens in 1 decode")

	answers: [N_SEQ]strings.Builder
	defer for &answer in answers {
		strings.builder_destroy(&answer)
	}

	n_gen: [N_SEQ]int
	done: [N_SEQ]bool

	steps := 0
	gen_start := time.tick_now()

	for {
		batch.n_tokens = 0

		for seq in 0 ..< N_SEQ {
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
			batch_add(batch, token, n_pos[seq], i32(seq), true)

			n_pos[seq] += 1
			n_gen[seq] += 1
		}

		if batch.n_tokens == 0 {
			break
		}

		result = llama.decode(engine.ctx, batch^)
		if result != 0 {
			fmt.println("decode failed:", result)
			return
		}

		steps += 1
	}

	gen_ms := time.duration_milliseconds(time.tick_since(gen_start))
	total := 0
	for n in n_gen {
		total += n
	}

	for seq in 0 ..< N_SEQ {
		fmt.printfln("\nseq %d (%d tokens): %s", seq, n_gen[seq], strings.to_string(answers[seq]))
	}

	fmt.printfln(
		"\n%d tokens in %d decode steps, %.1f ms (%.0f tok/s total)",
		total,
		steps,
		gen_ms,
		f64(total) / gen_ms * 1000,
	)
}
