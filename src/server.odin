package main

import "core:encoding/json"
import "core:fmt"
import "core:net"
import "core:strings"
import "core:sync"
import "core:thread"
import "core:time"
import "core:unicode/utf8"

import "lib:llama"

Pending :: struct {
	chat_id:           string,
	reasoning_effort:  string,
	reasoning_budget:  int,
	prompt:            string,
	max_tokens:        int,
	arrive:            time.Tick,
	stream:            bool,
	signal:            sync.Sema,
	mutex:             sync.Mutex,
	stream_buf:        strings.Builder,
	finished:          bool,
	cancelled:         bool,
	answer:            string,
	reasoning_content: string,
	prompt_tokens:     int,
	completion_tokens: int,
	queue_ms:          f64, // arrive -> admitted to a slot
	prefill_ms:        f64, // start of the decode with the prompt -> first token sampled
	ttft_ms:           f64, // arrive -> first token sampled
	predicted_ms:      f64, // first token -> finished
	failed:            bool,
	overloaded:        bool,
}

Server :: struct {
	tmpl:  cstring,
	port:  int,
	mutex: sync.Mutex,
	queue: [dynamic]^Pending,
	wake:  sync.Sema,
}

Server_Slot :: struct {
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

run_server :: proc(engine: ^Engine, port: int) {
	server := Server {
		tmpl = engine.tmpl,
		port = port,
	}
	defer delete(server.queue)

	thread.create_and_start_with_poly_data(&server, accept_loop)

	batch := &engine.batch
	n_ctx_seq := i32(llama.n_ctx_seq(engine.ctx))
	decode_start := time.tick_now()

	slots: [N_SEQ]Server_Slot
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
				finish_slot(engine, &slot, s, server.port)
				continue
			}

			if slot.reasoning && slot.reasoning_n >= slot.pending.reasoning_budget {
				end_tokens := tokenize_text(engine.vocab, THINK_END, false)
				if slot.n_pos + i32(len(end_tokens)) >= n_ctx_seq {
					delete(end_tokens)
					finish_slot(engine, &slot, s, server.port)
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
				finish_slot(engine, &slot, s, server.port)
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

		// 2. put queued requests into their conversation slots
		for _ in 0 ..< N_SEQ {
			p := queue_pop(&server)
			if p == nil {
				break
			}

			s := find_slot(&slots, p.chat_id)
			if s == -2 {
				p.overloaded = true
				p.finished = true
				sync.sema_post(&p.signal)
				continue
			}
			if s == -1 {
				queue_push_front(&server, p)
				break
			}
			slot := &slots[s]

			tokens := tokenize_text(engine.vocab, p.prompt, true)

			if i32(len(tokens)) >= n_ctx_seq || len(tokens) > BATCH_SIZE {
				delete(tokens)
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
				p.failed = true
				p.finished = true
				sync.sema_post(&p.signal)
				continue
			}

			if int(batch.n_tokens) + len(new_tokens) > BATCH_SIZE {
				delete(tokens)
				queue_push_front(&server, p)
				break
			}

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
				server.port,
				p.chat_id if p.chat_id != "" else "untracked",
				s,
				cached,
				len(new_tokens),
				p.max_tokens,
			)
			delete(tokens)
		}

		if batch.n_tokens == 0 {
			sync.sema_wait(&server.wake)
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
finish_slot :: proc(engine: ^Engine, slot: ^Server_Slot, s: int, port: int) {
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

find_slot :: proc(slots: ^[N_SEQ]Server_Slot, chat_id: string) -> int {
	if chat_id != "" {
		for &slot, s in slots {
			if slot.chat_id == chat_id {
				return s if slot.pending == nil else -1
			}
		}
	}

	for &slot, s in slots {
		if slot.pending == nil && slot.chat_id == "" {
			return s
		}
	}

	return -2
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

clear_slot_cache :: proc(engine: ^Engine, slot: ^Server_Slot, s: int) {
	llama.memory_seq_rm(engine.mem, i32(s), -1, -1)
	delete(slot.chat_id)
	slot.chat_id = ""
	clear(&slot.tokens)
}

queue_pop :: proc(server: ^Server) -> ^Pending {
	sync.mutex_lock(&server.mutex)
	defer sync.mutex_unlock(&server.mutex)

	if len(server.queue) == 0 {
		return nil
	}

	p := server.queue[0]
	ordered_remove(&server.queue, 0)
	return p
}

queue_push :: proc(server: ^Server, p: ^Pending) {
	sync.mutex_lock(&server.mutex)
	append(&server.queue, p)
	sync.mutex_unlock(&server.mutex)

	sync.sema_post(&server.wake)
}

queue_push_front :: proc(server: ^Server, p: ^Pending) {
	sync.mutex_lock(&server.mutex)
	defer sync.mutex_unlock(&server.mutex)

	inject_at(&server.queue, 0, p)
}

accept_loop :: proc(server: ^Server) {
	address := fmt.tprintf("127.0.0.1:%d", server.port)
	endpoint, _ := net.parse_endpoint(address)

	listener, err := net.listen_tcp(endpoint)
	if err != nil {
		fmt.println("listen failed:", err)
		return
	}

	fmt.printfln("worker listening on http://%s", address)

	for {
		client, _, accept_err := net.accept_tcp(listener)
		if accept_err != nil {
			continue
		}

		thread.create_and_start_with_poly_data2(server, client, handle_connection, self_cleanup = true)
	}
}

handle_connection :: proc(server: ^Server, client: net.TCP_Socket) {
	defer net.close(client)
	defer free_all(context.temp_allocator)

	buf: [65536]u8
	head, body, ok := read_http_request(client, buf[:])
	if !ok {
		send_http(client, "413 Content Too Large", "{}")
		return
	}

	if strings.has_prefix(head, "GET /health ") {
		send_http(client, "200 OK", `{"status":"ok"}`)
		return
	}

	if !strings.has_prefix(head, "POST /v1/chat/completions ") {
		send_http(client, "404 Not Found", "{}")
		return
	}

	req: Request
	if json.unmarshal(body, &req, allocator = context.temp_allocator) != nil {
		send_http(client, "400 Bad Request", "{}")
		return
	}

	effort := req.reasoning_effort
	if effort == "" {
		effort = "low"
	}
	if effort != "low" && effort != "medium" && effort != "high" {
		send_http(client, "400 Bad Request", `{"error":"reasoning_effort must be low, medium, or high"}`)
		return
	}

	prompt, prompt_ok := format_messages(server.tmpl, req.messages, effort)
	if !prompt_ok {
		send_http(client, "400 Bad Request", "{}")
		return
	}

	pending := Pending {
		chat_id          = header_value(head, "x-conversation-id:"),
		reasoning_effort = effort,
		prompt            = prompt,
		max_tokens        = req.max_tokens if req.max_tokens > 0 else 256,
		arrive            = time.tick_now(),
		stream            = req.stream,
	}
	if effort == "medium" {
		pending.reasoning_budget = pending.max_tokens / 2
	} else if effort == "high" {
		pending.reasoning_budget = pending.max_tokens * 3 / 4
	}
	defer strings.builder_destroy(&pending.stream_buf)

	queue_push(server, &pending)
	sync.sema_wait(&pending.signal)
	defer delete(pending.answer)
	defer delete(pending.reasoning_content)

	if pending.overloaded {
		send_http(client, "503 Service Unavailable", `{"error":"worker conversation capacity reached"}`)
		return
	}

	if pending.stream && !pending.failed {
		stream_response(client, &pending)
		return
	}

	if pending.failed {
		send_http(client, "400 Bad Request", `{"error":"prompt too long"}`)
		return
	}

	json_body, _ := json.marshal(make_response(&pending), allocator = context.temp_allocator)
	send_http(client, "200 OK", string(json_body))
}

// stream_response sends new text as SSE events until the request is finished.
// The first signal was already received by the caller.
stream_response :: proc(client: net.TCP_Socket, p: ^Pending) {
	send_sse_headers(client)

	for {
		sync.mutex_lock(&p.mutex)
		text := strings.to_string(p.stream_buf)
		finished := p.finished

		// hold back an incomplete UTF-8 character until its next bytes arrive
		cut := len(text)
		for cut > 0 && !utf8.valid_string(text[:cut]) {
			cut -= 1
		}

		text = strings.clone(text[:cut], context.temp_allocator)
		remove_range(&p.stream_buf.buf, 0, cut)
		sync.mutex_unlock(&p.mutex)

		if len(text) > 0 {
			chunk: Stream_Chunk
			chunk.choices = make(type_of(chunk.choices), 1, context.temp_allocator)
			chunk.choices[0].delta.content = text

			json_body, _ := json.marshal(chunk, allocator = context.temp_allocator)
			if !send_sse_event(client, string(json_body)) {
				// client is gone: ask the loop to free the slot, then wait until
				// it stops using p, because p lives on this thread's stack
				sync.atomic_store(&p.cancelled, true)
				wait_finished(p)
				return
			}
		}

		if finished {
			send_sse_event(client, "[DONE]")
			return
		}

		sync.sema_wait(&p.signal)
	}
}

wait_finished :: proc(p: ^Pending) {
	for {
		sync.mutex_lock(&p.mutex)
		finished := p.finished
		sync.mutex_unlock(&p.mutex)

		if finished {
			return
		}

		sync.sema_wait(&p.signal)
	}
}

Stream_Chunk :: struct {
	choices: []struct {
		delta: struct {
			content: string,
		},
	},
}

make_response :: proc(p: ^Pending) -> Response {
	resp: Response
	resp.choices = make(type_of(resp.choices), 1, context.temp_allocator)
	resp.choices[0].message.role = "assistant"
	resp.choices[0].message.reasoning_content = p.reasoning_content
	resp.choices[0].message.content = p.answer

	resp.usage.prompt_tokens = p.prompt_tokens
	resp.usage.completion_tokens = p.completion_tokens
	resp.usage.total_tokens = p.prompt_tokens + p.completion_tokens

	resp.timings.prompt_n = p.prompt_tokens
	resp.timings.prompt_ms = f32(p.prefill_ms)
	resp.timings.prompt_per_second = f32(f64(p.prompt_tokens) / p.prefill_ms * 1000)
	resp.timings.predicted_n = p.completion_tokens
	resp.timings.predicted_ms = f32(p.predicted_ms)
	resp.timings.predicted_per_second = f32(f64(p.completion_tokens) / p.predicted_ms * 1000)
	resp.timings.queue_ms = f32(p.queue_ms)
	resp.timings.ttft_ms = f32(p.ttft_ms)

	return resp
}
