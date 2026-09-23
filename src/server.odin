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
	prompt_tokens:     int,
	completion_tokens: int,
	prompt_ms:         f64,
	predicted_ms:      f64,
	failed:            bool,
}

Server :: struct {
	tmpl:  cstring,
	mutex: sync.Mutex,
	queue: [dynamic]^Pending,
	wake:  sync.Sema,
}

Server_Slot :: struct {
	pending:     ^Pending,
	sampler:     ^llama.Sampler,
	n_pos:       i32,
	n_gen:       int,
	i_batch:     i32,
	first_token: time.Tick,
	answer:      strings.Builder,
}

run_server :: proc(engine: ^Engine) {
	server := Server {
		tmpl = engine.tmpl,
	}
	defer delete(server.queue)

	thread.create_and_start_with_poly_data(&server, accept_loop)

	batch := &engine.batch
	n_ctx_seq := i32(llama.n_ctx_seq(engine.ctx))

	slots: [N_SEQ]Server_Slot
	defer for &slot in slots {
		strings.builder_destroy(&slot.answer)
	}

	for {
		batch.n_tokens = 0

		// 1. sample active slots and add their next token
		for &slot, s in slots {
			if slot.pending == nil {
				continue
			}

			token := llama.sampler_sample(slot.sampler, engine.ctx, slot.i_batch)

			if slot.n_gen == 0 {
				slot.first_token = time.tick_now()
				slot.pending.prompt_ms = time.duration_milliseconds(time.tick_diff(slot.pending.arrive, slot.first_token))
			}

			if llama.vocab_is_eog(engine.vocab, token) ||
			   slot.n_gen == slot.pending.max_tokens ||
			   slot.n_pos >= n_ctx_seq ||
			   sync.atomic_load(&slot.pending.cancelled) {
				p := slot.pending
				answer := strings.to_string(slot.answer)
				for len(answer) > 0 && !utf8.valid_string(answer) {
					answer = answer[:len(answer) - 1]
				}

				p.answer = strings.clone(answer)
				p.completion_tokens = slot.n_gen
				p.predicted_ms = time.duration_milliseconds(time.tick_since(slot.first_token))

				status := "cancelled" if sync.atomic_load(&p.cancelled) else "done"
				fmt.printfln("slot %d %s: %d prompt, %d gen tokens, %.0f ms", s, status, p.prompt_tokens, p.completion_tokens, p.prompt_ms + p.predicted_ms)

				llama.memory_seq_rm(engine.mem, i32(s), -1, -1)
				llama.sampler_free(slot.sampler)
				strings.builder_reset(&slot.answer)
				slot.pending = nil

				sync.mutex_lock(&p.mutex)
				p.finished = true
				sync.mutex_unlock(&p.mutex)

				sync.sema_post(&p.signal)
				continue
			}

			piece := append_token_piece(engine.vocab, token, &slot.answer)

			if slot.pending.stream {
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

		// 2. put queued requests into free slots, prompt goes into the same batch
		for &slot, s in slots {
			if slot.pending != nil {
				continue
			}

			p := queue_pop(&server)
			if p == nil {
				break
			}

			tokens := tokenize_text(engine.vocab, p.prompt, true)

			if i32(len(tokens)) >= n_ctx_seq {
				delete(tokens)
				p.failed = true
				p.finished = true
				sync.sema_post(&p.signal)
				continue
			}

			if int(batch.n_tokens) + len(tokens) > BATCH_SIZE {
				delete(tokens)
				queue_push_front(&server, p)
				break
			}

			slot.pending = p
			slot.sampler = make_sampler()
			slot.n_pos = i32(len(tokens))
			slot.n_gen = 0
			slot.i_batch = batch_add_tokens(batch, tokens, 0, i32(s))

			p.prompt_tokens = len(tokens)
			delete(tokens)
		}

		if batch.n_tokens == 0 {
			sync.sema_wait(&server.wake)
			continue
		}

		result := llama.decode(engine.ctx, batch^)
		if result != 0 {
			fmt.println("decode failed:", result)
			return
		}
	}
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
	endpoint, _ := net.parse_endpoint("127.0.0.1:8080")

	listener, err := net.listen_tcp(endpoint)
	if err != nil {
		fmt.println("listen failed:", err)
		return
	}

	fmt.println("listening on http://127.0.0.1:8080")

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

	if !strings.has_prefix(head, "POST /v1/chat/completions ") {
		send_http(client, "404 Not Found", "{}")
		return
	}

	req: Request
	if json.unmarshal(body, &req, allocator = context.temp_allocator) != nil {
		send_http(client, "400 Bad Request", "{}")
		return
	}

	prompt, prompt_ok := format_messages(server.tmpl, req.messages)
	if !prompt_ok {
		send_http(client, "400 Bad Request", "{}")
		return
	}

	pending := Pending {
		prompt     = prompt,
		max_tokens = req.max_tokens if req.max_tokens > 0 else 256,
		arrive     = time.tick_now(),
		stream     = req.stream,
	}
	defer strings.builder_destroy(&pending.stream_buf)

	queue_push(server, &pending)
	sync.sema_wait(&pending.signal)
	defer delete(pending.answer)

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
	resp.choices[0].message.content = p.answer

	resp.usage.prompt_tokens = p.prompt_tokens
	resp.usage.completion_tokens = p.completion_tokens
	resp.usage.total_tokens = p.prompt_tokens + p.completion_tokens

	resp.timings.prompt_n = p.prompt_tokens
	resp.timings.prompt_ms = f32(p.prompt_ms)
	resp.timings.prompt_per_second = f32(f64(p.prompt_tokens) / p.prompt_ms * 1000)
	resp.timings.predicted_n = p.completion_tokens
	resp.timings.predicted_ms = f32(p.predicted_ms)
	resp.timings.predicted_per_second = f32(f64(p.completion_tokens) / p.predicted_ms * 1000)

	return resp
}
