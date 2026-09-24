package inference

import "core:c"
import "core:c/libc"
import "core:fmt"
import "core:strings"

import "lib:llama"

THINK_OFF :: "<think>\n\n</think>\n\n"
THINK_ON :: "<think>\n"
THINK_END :: "</think>\n\n"

BATCH_SIZE :: 2048

N_SEQ :: 4

Message :: struct {
	role:              string,
	reasoning_content: string,
	content:           string,
}

Engine :: struct {
	model: ^llama.Model,
	ctx:   ^llama.Context,
	vocab: ^llama.Vocab,
	mem:   ^llama.Memory,
	tmpl:  cstring,
	batch: llama.Batch,
}

engine_init :: proc(
	model_path: cstring,
	n_ctx: u32,
	n_seq_max: u32,
) -> (
	engine: Engine,
	ok: bool,
) {
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
	engine.tmpl = llama.model_chat_template(engine.model, nil)
	engine.batch = llama.batch_init(BATCH_SIZE, 0, 1)

	return engine, true
}

engine_free :: proc(engine: ^Engine) {
	llama.batch_free(engine.batch)
	llama.free(engine.ctx)
	llama.model_free(engine.model)
}

make_sampler :: proc(reasoning: bool) -> ^llama.Sampler {
	sampler := llama.sampler_chain_init(llama.Sampler_Chain_Params{})

	llama.sampler_chain_add(sampler, llama.sampler_init_top_k(20))
	llama.sampler_chain_add(sampler, llama.sampler_init_top_p(0.95 if reasoning else 0.8, 1))
	llama.sampler_chain_add(sampler, llama.sampler_init_temp(0.6 if reasoning else 0.7))
	llama.sampler_chain_add(sampler, llama.sampler_init_dist(llama.DEFAULT_SEED))

	return sampler
}

tokenize_text :: proc(vocab: ^llama.Vocab, text: string, add_special: bool) -> []llama.Token {
	c_text := strings.clone_to_cstring(text)
	defer delete(c_text)

	count := -llama.tokenize(vocab, c_text, i32(len(text)), nil, 0, add_special, true)

	tokens := make([]llama.Token, count)
	llama.tokenize(
		vocab,
		c_text,
		i32(len(text)),
		raw_data(tokens),
		i32(len(tokens)),
		add_special,
		true,
	)

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

batch_add_tokens :: proc(batch: ^llama.Batch, tokens: []llama.Token, start_pos: i32, seq_id: i32) -> i32 {
	for token, i in tokens {
		batch_add(batch, token, start_pos + i32(i), seq_id, i == len(tokens) - 1)
	}

	return batch.n_tokens - 1
}

format_messages :: proc(tmpl: cstring, messages: []Message, effort: string) -> (prompt: string, ok: bool) {
	reasoning := effort != "low"
	chat := make([]llama.Chat_Message, len(messages), context.temp_allocator)
	for m, i in messages {
		content := m.content
		if m.role == "assistant" {
			if reasoning {
				content = fmt.tprintf("%s%s%s%s", THINK_ON, m.reasoning_content, THINK_END, m.content)
			} else {
				content = fmt.tprintf("%s%s", THINK_OFF, m.content)
			}
		}
		chat[i] = {
			role    = strings.clone_to_cstring(m.role, context.temp_allocator),
			content = strings.clone_to_cstring(content, context.temp_allocator),
		}
	}

	buf: [32768]u8
	n := llama.chat_apply_template(tmpl, raw_data(chat), len(chat), true, &buf[0], i32(len(buf)))

	if n < 0 || n > i32(len(buf)) {
		return
	}

	mode := THINK_ON if reasoning else THINK_OFF
	return fmt.tprintf("%s%s", string(buf[:n]), mode), true
}

append_token_piece :: proc(vocab: ^llama.Vocab, token: llama.Token, answer: ^strings.Builder) -> string {
	buf: [128]u8

	n := llama.token_to_piece(vocab, token, &buf[0], i32(len(buf)), 0, false)

	if n <= 0 {
		return ""
	}

	start := strings.builder_len(answer^)
	strings.write_string(answer, string(buf[:n]))

	return strings.to_string(answer^)[start:]
}

log_errors_only :: proc "c" (level: c.int, text: cstring, user_data: rawptr) {
	if level == 4 {
		libc.fprintf(libc.stderr, "%s", text)
	}
}
