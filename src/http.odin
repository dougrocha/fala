package main

import "core:fmt"
import "core:net"
import "core:strconv"
import "core:strings"

// read_http_request reads until the headers and Content-Length body bytes are in buf.
// Returns false on a receive error or when the request does not fit in buf.
read_http_request :: proc(client: net.TCP_Socket, buf: []u8) -> (head: string, body: []u8, ok: bool) {
	total := 0

	for total < len(buf) {
		n, err := net.recv_tcp(client, buf[total:])
		if err != nil || n == 0 {
			return
		}
		total += n

		text := string(buf[:total])
		header_end := strings.index(text, "\r\n\r\n")
		if header_end < 0 {
			continue
		}

		body_start := header_end + 4
		body_end := body_start + content_length(text[:header_end])

		if total >= body_end {
			return text[:header_end], buf[body_start:body_end], true
		}
	}

	return
}

content_length :: proc(headers: string) -> int {
	lower := strings.to_lower(headers, context.temp_allocator)

	i := strings.index(lower, "content-length:")
	if i < 0 {
		return 0
	}

	rest := lower[i + len("content-length:"):]
	end := strings.index(rest, "\r\n")
	if end >= 0 {
		rest = rest[:end]
	}

	n, _ := strconv.parse_int(strings.trim_space(rest))
	return n
}

send_http :: proc(client: net.TCP_Socket, status: string, body: string) {
	response := fmt.tprintf(
		"HTTP/1.1 %s\r\n" +
		"Content-Type: application/json\r\n" +
		"Content-Length: %d\r\n" +
		"Connection: close\r\n\r\n" +
		"%s",
		status,
		len(body),
		body,
	)

	net.send_tcp(client, transmute([]u8)response)
}
