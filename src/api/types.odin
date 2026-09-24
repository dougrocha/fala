package api

import "../inference"

Request :: struct {
	messages:         []Message,
	max_tokens:       int,
	stream:           bool,
	reasoning_effort: string,
}

Message :: inference.Message

Response :: struct {
	choices: []struct {
		message: struct {
			role:              string,
			reasoning_content: string,
			content:           string,
		},
	},
	usage:   struct {
		prompt_tokens:     int,
		completion_tokens: int,
		total_tokens:      int,
	},
	timings: struct {
		prompt_n:             int,
		prompt_ms:            f32,
		prompt_per_second:    f32,
		predicted_n:          int,
		predicted_ms:         f32,
		predicted_per_second: f32,
		queue_ms:             f32,
		ttft_ms:              f32,
	},
}
