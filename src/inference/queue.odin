package inference

import "core:strings"
import "core:sync"
import "core:time"

// Pending is one request. The connection thread owns it (on its stack).
// The loop fills the results and posts signal.
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
}

Queue :: struct {
	mutex: sync.Mutex,
	items: [dynamic]^Pending,
	wake:  sync.Sema,
}

// queue_get returns the request at index i, or nil when i is past the end.
queue_get :: proc(queue: ^Queue, i: int) -> ^Pending {
	sync.mutex_lock(&queue.mutex)
	defer sync.mutex_unlock(&queue.mutex)

	if i >= len(queue.items) {
		return nil
	}

	return queue.items[i]
}

queue_remove :: proc(queue: ^Queue, i: int) {
	sync.mutex_lock(&queue.mutex)
	defer sync.mutex_unlock(&queue.mutex)

	ordered_remove(&queue.items, i)
}

queue_push :: proc(queue: ^Queue, p: ^Pending) {
	sync.mutex_lock(&queue.mutex)
	append(&queue.items, p)
	sync.mutex_unlock(&queue.mutex)

	sync.sema_post(&queue.wake)
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
