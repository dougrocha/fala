package main

import "core:thread"

// Job pairs one input with the work proc that produces its output, plus a
// slot for the result. run_concurrent allocates and fills a []Job for you.
Job :: struct($In, $Out: typeid) {
	index:  int,
	input:  In,
	work:   proc(_: In) -> (Out, bool),
	output: Out,
	ok:     bool,
}

// run_concurrent runs work(inputs[i]) on its own thread for every input,
// waits for all of them to finish, and returns one Job per input in order.
// Caller owns the returned slice (delete it when done).
run_concurrent :: proc(inputs: []$In, work: proc(_: In) -> ($Out, bool)) -> []Job(In, Out) {
	jobs := make([]Job(In, Out), len(inputs))

	threads := make([dynamic]^thread.Thread, 0, len(inputs))
	defer delete(threads)

	for input, i in inputs {
		jobs[i].index = i
		jobs[i].input = input
		jobs[i].work = work

		t := thread.create(job_runner(In, Out))
		assert(t != nil)
		t.data = &jobs[i]

		thread.start(t)
		append(&threads, t)
	}

	thread.join_multiple(..threads[:])
	for t in threads {
		thread.destroy(t)
	}

	return jobs
}

job_runner :: proc($In, $Out: typeid) -> proc(_: ^thread.Thread) {
	return proc(t: ^thread.Thread) {
			job := (^Job(In, Out))(t.data)
			job.output, job.ok = job.work(job.input)
		}
}
