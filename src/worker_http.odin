package main

import "core:encoding/json"
import "core:fmt"
import "core:net"
import "core:strings"
import "core:sync"
import "core:thread"
import "core:time"
import "core:unicode/utf8"

import "api"
import "inference"

accept_loop :: proc(worker: ^Worker) {
	address := fmt.tprintf("127.0.0.1:%d", worker.port)
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

		thread.create_and_start_with_poly_data2(worker, client, handle_connection, self_cleanup = true)
	}
}

handle_connection :: proc(worker: ^Worker, client: net.TCP_Socket) {
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

	req: api.Request
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

	prompt, prompt_ok := inference.format_messages(worker.tmpl, req.messages, effort)
	if !prompt_ok {
		send_http(client, "400 Bad Request", "{}")
		return
	}

	pending := inference.Pending {
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

	inference.queue_push(&worker.queue, &pending)
	sync.sema_wait(&pending.signal)
	defer delete(pending.answer)
	defer delete(pending.reasoning_content)

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
stream_response :: proc(client: net.TCP_Socket, p: ^inference.Pending) {
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
				inference.wait_finished(p)
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

wait_finished :: proc(p: ^inference.Pending) {
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

make_response :: proc(p: ^inference.Pending) -> api.Response {
	resp: api.Response
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
