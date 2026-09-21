package main

import "core:fmt"

RUNS_PER_PROMPT :: 5
MAX_TOKENS :: 64
BASELINE_SPEED :: f32(263.0)

calc_kv_size :: proc(stage: string, tokens: int) -> int {
	n_layer := 28
	n_head_kv := 8
	n_embd_head_k := 128
	n_embd_head_v := 128

	bytes_per_element := 2 // if kv cache is FP16
	k_bytes_per_token := n_layer * n_head_kv * n_embd_head_k * bytes_per_element
	v_bytes_per_token := n_layer * n_head_kv * n_embd_head_v * bytes_per_element

	kv_bytes_per_token := k_bytes_per_token + v_bytes_per_token
	kv_bytes := kv_bytes_per_token * tokens

	// fmt.printfln(
	// 	"kv_cache[%s]: tokens=%d bytes_per_token=%d total_bytes=%d",
	// 	stage,
	// 	tokens,
	// 	kv_bytes_per_token,
	// 	kv_bytes,
	// )

	return kv_bytes
}

run_benchmark :: proc(index: int, message: string) {
	total_prompt_tokens := 0
	total_speed: f32 = 0
	successful_runs := 0

	prefill_size: int
	final_size: int
	log_kv := false

	for run in 0 ..< RUNS_PER_PROMPT {
		response, ok := send_completion(message, MAX_TOKENS)
		if !ok {
			continue
		}
		total_prompt_tokens += response.usage.prompt_tokens
		total_speed += response.timings.predicted_per_second
		successful_runs += 1

		if !log_kv {
			prefill_size = calc_kv_size("prefill", total_prompt_tokens)
			final_size = calc_kv_size(
				"final",
				total_prompt_tokens + response.usage.completion_tokens,
			)
			log_kv = true
		}
	}

	if successful_runs == 0 {
		fmt.printfln("prompt[%d]: all runs failed", index)
		return
	}

	avg_prompt_tokens := total_prompt_tokens / successful_runs
	avg_speed := total_speed / f32(successful_runs)
	slowdown := (BASELINE_SPEED - avg_speed) / BASELINE_SPEED * 100

	fmt.printfln(
		"prompt[%d]: prompt_tokens=%d avg_tok_s=%.2f avg_slowdown=%.1f%% runs=%d/%d prefill_size=%d final_size=%d",
		index,
		avg_prompt_tokens,
		avg_speed,
		slowdown,
		successful_runs,
		RUNS_PER_PROMPT,
		prefill_size,
		final_size,
	)
}
