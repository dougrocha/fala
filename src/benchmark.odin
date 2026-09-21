package main

import "core:fmt"
import "core:time"

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

run_scheduler_benchmark :: proc(
	work_queue: []Work_Item,
	slot_count: int,
	rounds: int,
) -> (
	avg_tok_s: f64,
	avg_latency_ms: f64,
) {
	tok_s_sum: f64
	latency_ms_sum: f64

	for round in 0 ..< rounds {
		fmt.printfln("\n=== %d-slot scheduler, round %d/%d ===", slot_count, round + 1, rounds)

		stopwatch: time.Stopwatch
		time.stopwatch_start(&stopwatch)
		tokens, completed, total_latency_ms := run_scheduler(work_queue, slot_count)
		time.stopwatch_stop(&stopwatch)

		wall_clock := time.duration_seconds(time.stopwatch_duration(stopwatch))
		tok_s := f64(tokens) / wall_clock
		round_avg_latency_ms := total_latency_ms / f64(completed)
		tok_s_sum += tok_s
		latency_ms_sum += round_avg_latency_ms

		fmt.printfln("Wall clock: %.2f seconds", wall_clock)
		fmt.printfln(
			"%d/%d requests, %.2f tok/s, avg latency %.0fms",
			completed,
			len(work_queue),
			tok_s,
			round_avg_latency_ms,
		)
	}

	return tok_s_sum / f64(rounds), latency_ms_sum / f64(rounds)
}

run_unbounded_benchmark :: proc(
	work_queue: []Work_Item,
	rounds: int,
) -> (
	avg_tok_s: f64,
	avg_latency_ms: f64,
) {
	tok_s_sum: f64
	latency_ms_sum: f64

	for round in 0 ..< rounds {
		fmt.printfln("\n=== Unbounded, round %d/%d ===", round + 1, rounds)

		stopwatch: time.Stopwatch
		time.stopwatch_start(&stopwatch)
		tokens, completed, total_latency_ms := run_unbounded(work_queue)
		time.stopwatch_stop(&stopwatch)

		wall_clock := time.duration_seconds(time.stopwatch_duration(stopwatch))
		tok_s := f64(tokens) / wall_clock
		round_avg_latency_ms := total_latency_ms / f64(completed)
		tok_s_sum += tok_s
		latency_ms_sum += round_avg_latency_ms

		fmt.printfln("Wall clock: %.2f seconds", wall_clock)
		fmt.printfln(
			"%d/%d requests, %.2f tok/s, avg latency %.0fms",
			completed,
			len(work_queue),
			tok_s,
			round_avg_latency_ms,
		)
	}

	return tok_s_sum / f64(rounds), latency_ms_sum / f64(rounds)
}

run_all_benchmarks :: proc(work_queue: []Work_Item, rounds: int) {
	tok_s_2, latency_ms_2 := run_scheduler_benchmark(work_queue, 2, rounds)
	tok_s_4, latency_ms_4 := run_scheduler_benchmark(work_queue, 4, rounds)
	tok_s_6, latency_ms_6 := run_scheduler_benchmark(work_queue, 6, rounds)
	tok_s_8, latency_ms_8 := run_scheduler_benchmark(work_queue, 8, rounds)
	tok_s_12, latency_ms_12 := run_scheduler_benchmark(work_queue, 12, rounds)
	tok_s_20, latency_ms_20 := run_scheduler_benchmark(work_queue, 20, rounds)
	unbounded_tok_s, unbounded_latency_ms := run_unbounded_benchmark(work_queue, rounds)

	fmt.printfln("\n=== Averages over %d rounds ===", rounds)
	fmt.printfln("2-slot scheduler avg tok/s: %.2f, avg latency: %.0fms", tok_s_2, latency_ms_2)
	fmt.printfln("4-slot scheduler avg tok/s: %.2f, avg latency: %.0fms", tok_s_4, latency_ms_4)
	fmt.printfln("6-slot scheduler avg tok/s: %.2f, avg latency: %.0fms", tok_s_6, latency_ms_6)
	fmt.printfln("8-slot scheduler avg tok/s: %.2f, avg latency: %.0fms", tok_s_8, latency_ms_8)
	fmt.printfln("12-slot scheduler avg tok/s: %.2f, avg latency: %.0fms", tok_s_12, latency_ms_12)
	fmt.printfln("20-slot scheduler avg tok/s: %.2f, avg latency: %.0fms", tok_s_20, latency_ms_20)
	fmt.printfln(
		"unbounded avg tok/s: %.2f, avg latency: %.0fms",
		unbounded_tok_s,
		unbounded_latency_ms,
	)
}
