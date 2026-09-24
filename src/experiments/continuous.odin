package experiments

import "core:fmt"
import "core:strings"
import "core:time"

import "lib:llama"

import "../inference"

Gen_Request :: struct {
	prompt:     cstring,
	arrive_ms:  f64,
	max_tokens: int,
}

Slot :: struct {
	active:   bool,
	req:      int,
	sampler:  ^llama.Sampler,
	n_pos:    i32,
	n_gen:    int,
	i_batch:  i32,
	admit_ms: f64,
	answer:   strings.Builder,
}

run_continuous :: proc(engine: ^inference.Engine) {
	requests := [8]Gen_Request {
		{PROMPTS[0], 0, 128},
		{PROMPTS[1], 0, 128},
		{PROMPTS[2], 100, 128},
		{PROMPTS[3], 200, 128},
		{PROMPTS[4], 300, 128},
		{PROMPTS[5], 300, 128},
		{PROMPTS[6], 300, 128},
		{PROMPTS[7], 500, 128},
	}

	batch := &engine.batch

	slots: [inference.N_SEQ]Slot
	defer for &slot in slots {
		strings.builder_destroy(&slot.answer)
	}

	next_req := 0
	total_tokens := 0
	start := time.tick_now()

	for {
		now_ms := time.duration_milliseconds(time.tick_since(start))
		batch.n_tokens = 0

		// 1. sample active slots and add their next token
		for &slot, s in slots {
			if !slot.active {
				continue
			}

			token := llama.sampler_sample(slot.sampler, engine.ctx, slot.i_batch)

			if llama.vocab_is_eog(engine.vocab, token) ||
			   slot.n_gen == requests[slot.req].max_tokens {
				req := requests[slot.req]
				answer := strings.to_string(slot.answer)
				if len(answer) > 50 {
					answer = answer[:50]
				}

				fmt.printfln(
					"[%5.0f ms] req %d done  slot %d | wait %4.0f ms | %3d tokens | latency %4.0f ms | %q",
					now_ms,
					slot.req,
					s,
					slot.admit_ms - req.arrive_ms,
					slot.n_gen,
					now_ms - req.arrive_ms,
					answer,
				)

				llama.memory_seq_rm(engine.mem, i32(s), -1, -1)
				llama.sampler_free(slot.sampler)
				strings.builder_reset(&slot.answer)
				slot.active = false
				continue
			}

			inference.append_token_piece(engine.vocab, token, &slot.answer)

			slot.i_batch = batch.n_tokens
			inference.batch_add(batch, token, slot.n_pos, i32(s), true)

			slot.n_pos += 1
			slot.n_gen += 1
			total_tokens += 1
		}

		// 2. put arrived requests into free slots, prompt goes into the same batch
		for &slot, s in slots {
			if slot.active || next_req == len(requests) || requests[next_req].arrive_ms > now_ms {
				continue
			}

			prompt := format_chat(engine, requests[next_req].prompt)
			tokens := inference.tokenize_text(engine.vocab, prompt, true)

			slot.active = true
			slot.req = next_req
			slot.sampler = inference.make_sampler(false)
			slot.n_pos = i32(len(tokens))
			slot.n_gen = 0
			slot.i_batch = inference.batch_add_tokens(batch, tokens, 0, i32(s))
			slot.admit_ms = now_ms

			fmt.printfln(
				"[%5.0f ms] req %d start slot %d | %d prompt tokens",
				now_ms,
				next_req,
				s,
				len(tokens),
			)

			delete(tokens)
			delete(prompt)
			next_req += 1
		}

		if batch.n_tokens == 0 {
			if next_req == len(requests) {
				break
			}

			time.sleep(time.Millisecond)
			continue
		}

		result := llama.decode(engine.ctx, batch^)
		if result != 0 {
			fmt.println("decode failed:", result)
			return
		}
	}

	total_ms := time.duration_milliseconds(time.tick_since(start))
	fmt.printfln(
		"\n%d tokens in %.0f ms (%.0f tok/s total)",
		total_tokens,
		total_ms,
		f64(total_tokens) / total_ms * 1000,
	)
}
