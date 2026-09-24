package inference

import "core:fmt"
import "core:strings"
import "core:sync"
import "core:time"
import "core:unicode/utf8"

import "lib:llama"

Slot :: struct {
	pending:     ^Pending,
	chat_id:     string,
	tokens:      [dynamic]llama.Token,
	last_use:    time.Tick,
	sampler:     ^llama.Sampler,
	n_pos:       i32,
	n_gen:       int,
	reasoning:   bool,
	reasoning_n: int,
	i_batch:     i32,
	first_token: time.Tick,
	answer:      strings.Builder,
}

// run_loop owns the slots and the batch. It never returns unless decode fails.
// port is only for the log lines.
run_loop :: proc(engine: ^Engine, queue: ^Queue, port: int) {
	batch := &engine.batch
	n_ctx_seq := i32(llama.n_ctx_seq(engine.ctx))
	decode_start := time.tick_now()

	slots: [N_SEQ]Slot
	defer for &slot in slots {
		delete(slot.chat_id)
		delete(slot.tokens)
		strings.builder_destroy(&slot.answer)
	}

	for {
		batch.n_tokens = 0

		// 1. sample active slots and add their next token
		for &slot, s in slots {
			if slot.pending == nil {
				continue
			}

			if slot.n_gen >= slot.pending.max_tokens ||
			   slot.n_pos >= n_ctx_seq ||
			   sync.atomic_load(&slot.pending.cancelled) {
				finish_slot(engine, &slot, s, port)
				continue
			}

			if slot.reasoning && slot.reasoning_n >= slot.pending.reasoning_budget {
				end_tokens := tokenize_text(engine.vocab, THINK_END, false)
				if slot.n_pos + i32(len(end_tokens)) >= n_ctx_seq {
					delete(end_tokens)
					finish_slot(engine, &slot, s, port)
					continue
				}

				strings.write_string(&slot.answer, THINK_END)
				for end_token in end_tokens {
					slot.i_batch = batch.n_tokens
					batch_add(batch, end_token, slot.n_pos, i32(s), true)
					append(&slot.tokens, end_token)
					slot.n_pos += 1
					slot.n_gen += 1
				}
				delete(end_tokens)
				slot.reasoning = false
				continue
			}

			token := llama.sampler_sample(slot.sampler, engine.ctx, slot.i_batch)

			if slot.n_gen == 0 {
				slot.first_token = time.tick_now()
				slot.pending.prefill_ms = time.duration_milliseconds(time.tick_diff(decode_start, slot.first_token))
				slot.pending.ttft_ms = time.duration_milliseconds(time.tick_diff(slot.pending.arrive, slot.first_token))
			}

			if llama.vocab_is_eog(engine.vocab, token) {
				finish_slot(engine, &slot, s, port)
				continue
			}

			piece := append_token_piece(engine.vocab, token, &slot.answer)
			append(&slot.tokens, token)
			was_reasoning := slot.reasoning
			if slot.reasoning {
				slot.reasoning_n += 1
				if strings.contains(piece, "</think>") {
					slot.reasoning = false
				}
			}

			if slot.pending.stream && !was_reasoning {
				sync.mutex_lock(&slot.pending.mutex)
				strings.write_string(&slot.pending.stream_buf, piece)
				sync.mutex_unlock(&slot.pending.mutex)

				sync.sema_post(&slot.pending.signal)
			}

			slot.i_batch = batch.n_tokens
			batch_add(batch, token, slot.n_pos, i32(s), true)

			slot.n_pos += 1
			slot.n_gen += 1
		}

		// 2. put queued requests into their conversation slots.
		// A request leaves the queue only when it starts or fails. Only this
		// thread removes items, so index i stays valid while others append.
		i := 0
		for {
			p := queue_get(queue, i)
			if p == nil {
				break
			}

			s := find_slot(&slots, p.chat_id)
			if s == -1 {
				// its conversation is still generating: let later requests go first
				i += 1
				continue
			}
			if s == -2 {
				break
			}
			slot := &slots[s]

			tokens := tokenize_text(engine.vocab, p.prompt, true)

			if i32(len(tokens)) >= n_ctx_seq || len(tokens) > BATCH_SIZE {
				delete(tokens)
				queue_remove(queue, i)
				p.failed = true
				p.finished = true
				sync.sema_post(&p.signal)
				continue
			}

			cached := 0
			if p.chat_id != "" && slot.chat_id == p.chat_id && tokens_have_prefix(tokens, slot.tokens[:]) {
				cached = len(slot.tokens)
			}

			new_tokens := tokens[cached:]
			if len(new_tokens) == 0 {
				delete(tokens)
				queue_remove(queue, i)
				p.failed = true
				p.finished = true
				sync.sema_post(&p.signal)
				continue
			}

			if int(batch.n_tokens) + len(new_tokens) > BATCH_SIZE {
				delete(tokens)
				break
			}

			queue_remove(queue, i)

			if cached == 0 {
				clear_slot_cache(engine, slot, s)
				if p.chat_id != "" {
					slot.chat_id = strings.clone(p.chat_id)
				}
			}

			slot.pending = p
			slot.sampler = make_sampler(p.reasoning_effort != "low")
			slot.n_pos = i32(len(tokens))
			slot.n_gen = 0
			slot.reasoning = p.reasoning_effort != "low"
			slot.reasoning_n = 0
			slot.i_batch = batch_add_tokens(batch, new_tokens, i32(cached), i32(s))
			append(&slot.tokens, ..new_tokens)

			p.prompt_tokens = len(tokens)
			p.queue_ms = time.duration_milliseconds(time.tick_since(p.arrive))
			fmt.printfln(
				"worker %d: chat=%s slot=%d start: %d cached, %d new, max %d gen",
				port,
				p.chat_id if p.chat_id != "" else "untracked",
				s,
				cached,
				len(new_tokens),
				p.max_tokens,
			)
			delete(tokens)
		}

		if batch.n_tokens == 0 {
			sync.sema_wait(&queue.wake)
			continue
		}

		decode_start = time.tick_now()
		result := llama.decode(engine.ctx, batch^)
		if result != 0 {
			fmt.println("decode failed:", result)
			return
		}
	}
}

// finish_slot hands the answer to the waiting connection thread and frees the slot.
finish_slot :: proc(engine: ^Engine, slot: ^Slot, s: int, port: int) {
	p := slot.pending
	output := strings.to_string(slot.answer)
	answer := output
	if p.reasoning_effort != "low" {
		end := strings.index(output, "</think>")
		if end >= 0 {
			p.reasoning_content = strings.clone(strings.trim_space(output[:end]))
			answer = strings.trim_space(output[end + len("</think>"):])
		}
	}
	for len(answer) > 0 && !utf8.valid_string(answer) {
		answer = answer[:len(answer) - 1]
	}

	p.answer = strings.clone(answer)
	p.completion_tokens = slot.n_gen
	p.predicted_ms = time.duration_milliseconds(time.tick_since(slot.first_token))

	status := "cancelled" if sync.atomic_load(&p.cancelled) else "done"
	fmt.printfln(
		"worker %d: chat=%s slot=%d %s: %d prompt, %d gen | queue %.0f ms, prefill %.0f ms, ttft %.0f ms, decode %.0f ms",
		port,
		p.chat_id if p.chat_id != "" else "untracked",
		s,
		status,
		p.prompt_tokens,
		p.completion_tokens,
		p.queue_ms,
		p.prefill_ms,
		p.ttft_ms,
		p.predicted_ms,
	)

	if p.chat_id == "" {
		clear_slot_cache(engine, slot, s)
	} else {
		slot.last_use = time.tick_now()
	}
	llama.sampler_free(slot.sampler)
	strings.builder_reset(&slot.answer)
	slot.pending = nil

	sync.mutex_lock(&p.mutex)
	p.finished = true
	sync.mutex_unlock(&p.mutex)

	sync.sema_post(&p.signal)
}

find_slot :: proc(slots: ^[N_SEQ]Slot, chat_id: string) -> int {
	if chat_id != "" {
		for &slot, s in slots {
			if slot.chat_id == chat_id {
				return s if slot.pending == nil else -1
			}
		}
	}

	// prefer an empty slot, else evict the idle chat that was used least recently
	free := -2
	for &slot, s in slots {
		if slot.pending != nil {
			continue
		}
		if slot.chat_id == "" {
			return s
		}
		if free == -2 || time.tick_diff(slot.last_use, slots[free].last_use) > 0 {
			free = s
		}
	}

	return free
}

tokens_have_prefix :: proc(tokens: []llama.Token, prefix: []llama.Token) -> bool {
	if len(prefix) > len(tokens) {
		return false
	}

	for token, i in prefix {
		if tokens[i] != token {
			return false
		}
	}
	return true
}

clear_slot_cache :: proc(engine: ^Engine, slot: ^Slot, s: int) {
	llama.memory_seq_rm(engine.mem, i32(s), -1, -1)
	delete(slot.chat_id)
	slot.chat_id = ""
	clear(&slot.tokens)
}
