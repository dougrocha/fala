package main

import "core:c"
import "core:c/libc"
import "core:fmt"
import "core:strings"

import "lib:llama"

Engine :: struct {
	model: ^llama.Model,
	ctx:   ^llama.Context,
	vocab: ^llama.Vocab,
	mem:   ^llama.Memory,
}

engine_init :: proc(model_path: cstring, n_ctx: u32, n_seq_max: u32) -> (engine: Engine, ok: bool) {
	params := llama.model_default_params()

	engine.model = llama.model_load_from_file(model_path, params)
	if engine.model == nil {
		fmt.println("failed to load model")
		return
	}

	ctx_params := llama.context_default_params()
	ctx_params.n_ctx = n_ctx
	ctx_params.n_seq_max = n_seq_max

	engine.ctx = llama.init_from_model(engine.model, ctx_params)
	if engine.ctx == nil {
		fmt.println("failed to create context")
		llama.model_free(engine.model)
		return
	}

	engine.vocab = llama.model_get_vocab(engine.model)
	engine.mem = llama.get_memory(engine.ctx)

	return engine, true
}

engine_free :: proc(engine: ^Engine) {
	llama.free(engine.ctx)
	llama.model_free(engine.model)
}

make_sampler :: proc() -> ^llama.Sampler {
	sampler := llama.sampler_chain_init(llama.Sampler_Chain_Params{})

	llama.sampler_chain_add(sampler, llama.sampler_init_top_k(40))
	llama.sampler_chain_add(sampler, llama.sampler_init_top_p(0.9, 1))
	llama.sampler_chain_add(sampler, llama.sampler_init_temp(0.7))
	llama.sampler_chain_add(sampler, llama.sampler_init_dist(llama.DEFAULT_SEED))

	return sampler
}

tokenize_text :: proc(vocab: ^llama.Vocab, text: string, add_special: bool) -> []llama.Token {
	c_text := strings.clone_to_cstring(text)
	defer delete(c_text)

	count := -llama.tokenize(vocab, c_text, i32(len(text)), nil, 0, add_special, true)

	tokens := make([]llama.Token, count)
	llama.tokenize(vocab, c_text, i32(len(text)), raw_data(tokens), i32(len(tokens)), add_special, true)

	return tokens
}

batch_add :: proc(batch: ^llama.Batch, token: llama.Token, pos: i32, seq_id: i32, logits: bool) {
	n := batch.n_tokens

	batch.token[n] = token
	batch.pos[n] = pos
	batch.n_seq_id[n] = 1
	batch.seq_id[n][0] = seq_id
	batch.logits[n] = 1 if logits else 0

	batch.n_tokens += 1
}

append_token_piece :: proc(vocab: ^llama.Vocab, token: llama.Token, answer: ^strings.Builder) {
	buf: [128]u8

	n := llama.token_to_piece(vocab, token, &buf[0], i32(len(buf)), 0, false)

	if n > 0 {
		strings.write_string(answer, string(buf[:n]))
	}
}

print_token_piece :: proc(vocab: ^llama.Vocab, token: llama.Token, answer: ^strings.Builder) {
	buf: [128]u8

	n := llama.token_to_piece(vocab, token, &buf[0], i32(len(buf)), 0, false)

	if n > 0 {
		fmt.print(string(buf[:n]))
		if answer != nil {
			strings.write_string(answer, string(buf[:n]))
		}
	}
}

log_errors_only :: proc "c" (level: c.int, text: cstring, user_data: rawptr) {
	if level == 4 {
		libc.fprintf(libc.stderr, "%s", text)
	}
}
