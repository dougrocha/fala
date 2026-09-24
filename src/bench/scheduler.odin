package bench

import "core:fmt"
import "core:sync"
import "core:thread"
import "core:time"

import "../api"

Work_Item :: struct {
	prompt:     string,
	max_tokens: int,
}

Timed_Result :: struct {
	response:      api.Response,
	start_offset:  f64,
	finish_offset: f64,
}

Scheduler :: struct {
	slots: []Worker_Slot,
}

Worker_Slot :: struct {
	thread:        ^thread.Thread,
	index:         int,
	prompt:        string,
	max_tokens:    int,
	run_start:     time.Tick,
	start_offset:  f64,
	finish_offset: f64,
	response:      api.Response,
	ok:            bool,
	done:          bool,
	inactive:      bool,
}


worker :: proc(data: rawptr) {
	slot := cast(^Worker_Slot)data

	slot.start_offset = time.duration_seconds(time.tick_since(slot.run_start))
	response, ok := send_completion(slot.prompt, slot.max_tokens)
	slot.finish_offset = time.duration_seconds(time.tick_since(slot.run_start))

	slot.response = response
	slot.ok = ok

	sync.atomic_store(&slot.done, true)
}

print_result :: proc(
	index: int,
	dispatched_offset: f64,
	finish_offset: f64,
	response: api.Response,
) -> (
	latency_ms: f64,
) {
	queue_wait_ms := dispatched_offset * 1000
	latency_ms = finish_offset * 1000

	fmt.printfln(
		"prompt[%d]: dispatched=%.2fs finish=%.2fs queue_wait=%.0fms latency=%.0fms ttft=%.0fms gen=%.0fms tokens=%d tok/s=%.2f",
		index,
		dispatched_offset,
		finish_offset,
		queue_wait_ms,
		latency_ms,
		response.timings.prompt_ms,
		response.timings.predicted_ms,
		response.timings.predicted_n,
		response.timings.predicted_per_second,
	)

	return
}

run_scheduler :: proc(
	work_queue: []Work_Item,
	slot_count: int,
) -> (
	total_generated_tokens: int,
	completed_requests: int,
	total_latency_ms: f64,
) {
	scheduler: Scheduler
	scheduler.slots = make([]Worker_Slot, slot_count)
	defer delete(scheduler.slots)

	run_start := time.tick_now()
	next_worker_index := 0

	for &slot in scheduler.slots {
		if next_worker_index == len(work_queue) {
			sync.atomic_store(&slot.inactive, true)
			continue
		}

		work_item := work_queue[next_worker_index]
		slot.index = next_worker_index
		slot.prompt = work_item.prompt
		slot.max_tokens = work_item.max_tokens
		slot.run_start = run_start

		fmt.printfln("Staring [%d]", slot.index)
		slot.thread = thread.create_and_start_with_data(&slot, worker)

		next_worker_index += 1
	}

	for {
		all_inactive := true

		for &slot in scheduler.slots {
			if sync.atomic_load(&slot.inactive) {
				continue
			}
			all_inactive = false

			if sync.atomic_load(&slot.done) {
				thread.join(slot.thread)

				if slot.ok {
					total_latency_ms += print_result(
						slot.index,
						slot.start_offset,
						slot.finish_offset,
						slot.response,
					)
					total_generated_tokens += slot.response.timings.predicted_n
					completed_requests += 1
				} else {
					fmt.printfln("Skipping %d", slot.index)
				}

				if next_worker_index != len(work_queue) {
					work_item := work_queue[next_worker_index]
					slot.index = next_worker_index
					slot.prompt = work_item.prompt
					slot.max_tokens = work_item.max_tokens
					slot.run_start = run_start

					sync.atomic_store(&slot.done, false)

					fmt.printfln("Staring [%d]", slot.index)
					slot.thread = thread.create_and_start_with_data(&slot, worker)

					next_worker_index += 1
				} else {
					sync.atomic_store(&slot.inactive, true)
				}
			}
		}

		if all_inactive {
			break
		}

		time.sleep(time.Millisecond)
	}

	return
}

unbounded_run_start: time.Tick

run_unbounded :: proc(
	work_queue: []Work_Item,
) -> (
	total_generated_tokens: int,
	completed_requests: int,
	total_latency_ms: f64,
) {
	unbounded_run_start = time.tick_now()

	jobs := run_concurrent(work_queue, run_timed_work_item)
	defer delete(jobs)

	for job in jobs {
		if !job.ok {
			fmt.printfln("Skipping %d", job.index)
			continue
		}

		total_latency_ms += print_result(
			job.index,
			job.output.start_offset,
			job.output.finish_offset,
			job.output.response,
		)
		total_generated_tokens += job.output.response.timings.predicted_n
		completed_requests += 1
	}

	return
}

run_timed_work_item :: proc(item: Work_Item) -> (result: Timed_Result, ok: bool) {
	result.start_offset = time.duration_seconds(time.tick_since(unbounded_run_start))
	response, send_ok := send_completion(item.prompt, item.max_tokens)
	result.finish_offset = time.duration_seconds(time.tick_since(unbounded_run_start))
	result.response = response
	return result, send_ok
}
