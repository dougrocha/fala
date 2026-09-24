package main

import "core:fmt"
import "core:hash"
import "core:encoding/uuid"
import "core:net"
import "core:strings"
import "core:sync"
import "core:thread"

Router_Worker :: struct {
	port:   int,
	active: int,
}

Router :: struct {
	workers: []Router_Worker,
	mutex:   sync.Mutex,
}

run_router :: proc(port: int, worker_ports: []int) {
	workers := make([]Router_Worker, len(worker_ports))
	defer delete(workers)

	for worker_port, i in worker_ports {
		workers[i].port = worker_port
	}

	router := Router {
		workers = workers,
	}

	address := fmt.tprintf("127.0.0.1:%d", port)
	endpoint, _ := net.parse_endpoint(address)
	listener, err := net.listen_tcp(endpoint)
	if err != nil {
		fmt.println("router listen failed:", err)
		return
	}
	defer net.close(listener)

	fmt.printfln("router listening on http://%s", address)

	for {
		client, _, accept_err := net.accept_tcp(listener)
		if accept_err != nil {
			continue
		}

		thread.create_and_start_with_poly_data2(&router, client, route_connection, self_cleanup = true)
	}
}

route_connection :: proc(router: ^Router, client: net.TCP_Socket) {
	defer net.close(client)
	defer free_all(context.temp_allocator)

	request_buf: [65536]u8
	head, body, ok := read_http_request(client, request_buf[:])
	if !ok {
		send_http(client, "413 Content Too Large", "{}")
		return
	}

	if strings.has_prefix(head, "POST /v1/conversations ") {
		id_buf: [36]u8
		chat_id := uuid.to_string_buffer(uuid.generate_v4(), id_buf[:])
		body := fmt.tprintf("%c\"id\":\"%s\"%c", '{', chat_id, '}')
		send_http(client, "201 Created", body)
		fmt.printfln("router: created chat=%s", chat_id)
		return
	}

	chat_id := header_value(head, "x-conversation-id:")
	worker := acquire_worker(router, chat_id)
	defer release_worker(router, worker, chat_id)

	upstream, err := net.dial_tcp_from_hostname_with_port_override("127.0.0.1", worker.port)
	if err != nil {
		send_http(client, "502 Bad Gateway", `{"error":"worker unavailable"}`)
		return
	}
	defer net.close(upstream)

	request_len := len(head) + 4 + len(body)
	_, send_err := net.send_tcp(upstream, request_buf[:request_len])
	if send_err != nil {
		send_http(client, "502 Bad Gateway", `{"error":"worker send failed"}`)
		return
	}

	response_buf: [65536]u8
	for {
		n, recv_err := net.recv_tcp(upstream, response_buf[:])
		if recv_err != nil || n == 0 {
			return
		}

		_, client_err := net.send_tcp(client, response_buf[:n])
		if client_err != nil {
			return
		}
	}
}

acquire_worker :: proc(router: ^Router, chat_id: string) -> ^Router_Worker {
	sync.mutex_lock(&router.mutex)
	defer sync.mutex_unlock(&router.mutex)

	worker: ^Router_Worker
	if chat_id != "" {
		worker_index := int(hash.fnv64a(transmute([]u8)chat_id) % u64(len(router.workers)))
		worker = &router.workers[worker_index]
	} else {
		worker = &router.workers[0]
		for &candidate in router.workers[1:] {
			if candidate.active < worker.active {
				worker = &candidate
			}
		}
	}

	worker.active += 1
	if chat_id != "" {
		fmt.printfln(
			"router: sticky chat=%s -> worker=%d, active=%d",
			chat_id,
			worker.port,
			worker.active,
		)
	} else {
		fmt.printfln("router: least-active -> worker=%d, active=%d", worker.port, worker.active)
	}
	return worker
}

release_worker :: proc(router: ^Router, worker: ^Router_Worker, chat_id: string) {
	sync.mutex_lock(&router.mutex)
	worker.active -= 1
	if chat_id != "" {
		fmt.printfln(
			"router: complete chat=%s worker=%d active=%d",
			chat_id,
			worker.port,
			worker.active,
		)
	} else {
		fmt.printfln("router: complete chat=untracked worker=%d active=%d", worker.port, worker.active)
	}
	sync.mutex_unlock(&router.mutex)
}
