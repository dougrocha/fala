package llama

import "core:c"

foreign import lib "system:llama"

Model_Params :: struct {
	devices:                     rawptr,
	tensor_buft_overrides:       rawptr,
	n_gpu_layers:                i32,
	split_mode:                  i32,
	load_mode:                   i32,
	lazy_mode:                   i32,
	main_gpu:                    i32,
	tensor_split:                rawptr,
	progress_callback:           rawptr,
	progress_callback_user_data: rawptr,
	kv_overrides:                rawptr,
	vocab_only:                  c.bool,
	check_tensors:               c.bool,
	use_extra_bufts:             c.bool,
	no_host:                     c.bool,
	no_alloc:                    c.bool,
	load_mtp:                    c.bool,
}

Model :: struct {}

Context_Params :: struct {
	n_ctx:                 c.uint,
	n_batch:               c.uint,
	n_ubatch:              c.uint,
	n_seq_max:             c.uint,
	n_rs_seq:              c.uint,
	n_outputs_max:         c.uint,
	n_outputs_max_per_seq: c.uint,
	n_threads:             c.int,
	n_threads_batch:       c.int,
	ctx_type:              c.int,
	rope_scaling_type:     c.int,
	pooling_type:          c.int,
	attention_type:        c.int,
	flash_attn_type:       c.int,
	rope_freq_base:        f32,
	rope_freq_scale:       f32,
	yarn_ext_factor:       f32,
	yarn_attn_factor:      f32,
	yarn_beta_fast:        f32,
	yarn_beta_slow:        f32,
	yarn_orig_ctx:         c.uint,
	defrag_thold:          f32,
	cb_eval:               rawptr,
	cb_eval_user_data:     rawptr,
	type_k:                c.int,
	type_v:                c.int,
	abort_callback:        rawptr,
	abort_callback_data:   rawptr,
	embeddings:            c.bool,
	offload_kqv:           c.bool,
	no_perf:               c.bool,
	op_offload:            c.bool,
	swa_full:              c.bool,
	kv_unified:            c.bool,
	samplers:              rawptr,
	n_samplers:            c.size_t,
	ctx_other:             rawptr,
}

Context :: struct {}

Vocab :: struct {}

Token :: i32

DEFAULT_SEED :: 0xFFFFFFFF

Batch :: struct {
	n_tokens: i32,
	token:    [^]Token,
	embd:     [^]f32,
	pos:      [^]i32,
	n_seq_id: [^]i32,
	seq_id:   [^][^]i32,
	logits:   [^]i8,
}

Sampler :: struct {}

Memory :: struct {}

Chat_Message :: struct {
	role:    cstring,
	content: cstring,
}

Log_Callback :: proc "c" (level: c.int, text: cstring, user_data: rawptr)

Sampler_Chain_Params :: struct {
	no_perf: c.bool,
}

@(default_calling_convention = "c")
foreign lib {
	@(link_name = "llama_backend_init")
	backend_init :: proc() ---

	@(link_name = "llama_backend_free")
	backend_free :: proc() ---

	@(link_name = "llama_model_default_params")
	model_default_params :: proc() -> Model_Params ---

	@(link_name = "llama_model_load_from_file")
	model_load_from_file :: proc(path: cstring, params: Model_Params) -> ^Model ---

	@(link_name = "llama_model_free")
	model_free :: proc(model: ^Model) ---

	@(link_name = "llama_context_default_params")
	context_default_params :: proc() -> Context_Params ---

	@(link_name = "llama_init_from_model")
	init_from_model :: proc(model: ^Model, ctx_params: Context_Params) -> ^Context ---

	@(link_name = "llama_free")
	free :: proc(ctx: ^Context) ---

	@(link_name = "llama_model_get_vocab")
	model_get_vocab :: proc(mode: ^Model) -> ^Vocab ---

	@(link_name = "llama_tokenize")
	tokenize :: proc(vocab: ^Vocab, text: cstring, text_len: i32, tokens: [^]Token, n_tokens_max: i32, add_special: c.bool, parse_special: c.bool) -> i32 ---

	@(link_name = "llama_token_to_piece")
	token_to_piece :: proc(vocab: ^Vocab, token: Token, buf: [^]u8, length: i32, lstrip: i32, special: c.bool) -> i32 ---

	@(link_name = "llama_batch_get_one")
	batch_get_one :: proc(tokens: [^]Token, n_tokens: i32) -> Batch ---

	@(link_name = "llama_decode")
	decode :: proc(ctx: ^Context, batch: Batch) -> i32 ---

	@(link_name = "llama_get_logits_ith")
	get_logits_ith :: proc(ctx: ^Context, i: i32) -> [^]f32 ---

	@(link_name = "llama_vocab_n_tokens")
	vocab_n_tokens :: proc(vocab: ^Vocab) -> i32 ---

	@(link_name = "llama_vocab_is_eog")
	vocab_is_eog :: proc(vocab: ^Vocab, token: Token) -> c.bool ---

	@(link_name = "llama_log_set")
	log_set :: proc(callback: Log_Callback, user_data: rawptr) ---

	@(link_name = "llama_sampler_chain_init")
	sampler_chain_init :: proc(params: Sampler_Chain_Params) -> ^Sampler ---

	@(link_name = "llama_sampler_init_greedy")
	sampler_init_greedy :: proc() -> ^Sampler ---

	@(link_name = "llama_sampler_init_top_k")
	sampler_init_top_k :: proc(k: i32) -> ^Sampler ---

	@(link_name = "llama_sampler_init_top_p")
	sampler_init_top_p :: proc(p: f32, min_keep: c.size_t) -> ^Sampler ---

	@(link_name = "llama_sampler_init_min_p")
	sampler_init_min_p :: proc(p: f32, min_keep: c.size_t) -> ^Sampler ---

	@(link_name = "llama_sampler_init_temp")
	sampler_init_temp :: proc(t: f32) -> ^Sampler ---

	@(link_name = "llama_sampler_init_dist")
	sampler_init_dist :: proc(seed: u32) -> ^Sampler ---

	@(link_name = "llama_sampler_chain_add")
	sampler_chain_add :: proc(chain: ^Sampler, smpl: ^Sampler) ---

	@(link_name = "llama_sampler_sample")
	sampler_sample :: proc(smpl: ^Sampler, ctx: ^Context, idx: i32) -> Token ---

	@(link_name = "llama_sampler_free")
	sampler_free :: proc(smpl: ^Sampler) ---

	@(link_name = "llama_model_chat_template")
	model_chat_template :: proc(model: ^Model, name: cstring) -> cstring ---

	@(link_name = "llama_chat_apply_template")
	chat_apply_template :: proc(tmpl: cstring, chat: [^]Chat_Message, n_msg: c.size_t, add_ass: c.bool, buf: [^]u8, length: i32) -> i32 ---

	@(link_name = "llama_get_memory")
	get_memory :: proc(ctx: ^Context) -> ^Memory ---

	@(link_name = "llama_memory_clear")
	memory_clear :: proc(mem: ^Memory, data: c.bool) ---

	@(link_name = "llama_memory_seq_pos_max")
	memory_seq_pos_max :: proc(mem: ^Memory, seq_id: i32) -> i32 ---

	@(link_name = "llama_batch_init")
	batch_init :: proc(n_tokens: i32, embd: i32, n_seq_max: i32) -> Batch ---

	@(link_name = "llama_batch_free")
	batch_free :: proc(batch: Batch) ---
}
