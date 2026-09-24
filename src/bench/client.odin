package bench

import "core:encoding/json"
import "core:fmt"
import "core:net"
import "core:strings"

import "../api"

IP :: "127.0.0.1"
PORT :: 8080

send_completion :: proc(message: string, max_tokens: int) -> (response: api.Response, ok: bool) {
	msg := api.Message {
		role    = "user",
		content = message,
	}

	api_request := api.Request {
		max_tokens = max_tokens,
		messages   = {msg},
	}

	resp_body, send_ok := post_json(IP, PORT, "/v1/chat/completions", api_request)
	defer delete(resp_body)
	if !send_ok {
		return
	}

	return parse_response(resp_body)
}

// post_json connects to ip:port, sends payload as a JSON POST to path, and
// returns the raw response body. Caller owns the returned bytes.
post_json :: proc(ip: string, port: int, path: string, payload: $T) -> (body: []u8, ok: bool) {
	addr, addr_ok := net.parse_ip4_address(ip)
	if !addr_ok {
		fmt.printfln("Failed to parse IP address %q", ip)
		return
	}

	sock, err := net.dial_tcp_from_address_and_port(addr, port)
	if err != nil {
		fmt.printfln("Failed to connect to %s:%d: %v", ip, port, err)
		return
	}
	defer net.close(sock)

	json_body, marshal_err := json.marshal(payload)
	if marshal_err != nil {
		fmt.printfln("Could not marshal body: %v", marshal_err)
		return
	}
	defer delete(json_body)

	request := build_http_request(ip, path, string(json_body))
	defer delete(request)

	_, err_send := net.send_tcp(sock, transmute([]u8)request)
	if err_send != nil {
		fmt.printfln("Failed to send data: %v", err_send)
		return
	}

	return read_http_response(sock)
}

build_http_request :: proc(host: string, path: string, body: string) -> string {
	return fmt.aprintf(
		"POST %s HTTP/1.1\r\n" +
		"Host: %s\r\n" +
		"Content-Type: application/json\r\n" +
		"Content-Length: %d\r\n" +
		"Connection: close\r\n\r\n" +
		"%s",
		path,
		host,
		len(body),
		body,
	)
}

read_http_response :: proc(sock: net.TCP_Socket) -> (body: []u8, ok: bool) {
	buffer: [4096]u8
	res_buf := make([dynamic]u8, 0)

	for {
		bytes_recv, err_recv := net.recv_tcp(sock, buffer[:])
		if err_recv != nil {
			fmt.printfln("Failed to receive data: %v", err_recv)
			delete(res_buf)
			return
		}

		if bytes_recv == 0 {
			break
		}

		received := buffer[:bytes_recv]
		append(&res_buf, ..received)
	}

	text := transmute(string)res_buf[:]
	idx := strings.index(text, "\r\n\r\n")
	if idx < 0 {
		fmt.println("Invalid HTTP response")
		delete(res_buf)
		return
	}

	body = make([]u8, len(res_buf) - idx - 4)
	copy(body, res_buf[idx + 4:])
	delete(res_buf)
	ok = true
	return
}

parse_response :: proc(body: []u8) -> (response: api.Response, ok: bool) {
	res: api.Response
	unmarshal_err := json.unmarshal(body, &res)
	if unmarshal_err != nil {
		fmt.printfln("Could not unmarshal body: %v", unmarshal_err)
		return
	}

	return res, true
}
