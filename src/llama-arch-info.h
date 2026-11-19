#pragma once

//
// Per-architecture facts.
//
// Each fact about an architecture used to live in its own switch over llm_arch,
// spread over llama-arch.cpp and llama-model.cpp: the display name, whether it
// is recurrent / hybrid / diffusion, whether it supports recurrent-state
// rollback, mixed batches, SM tensors or saving, and which RoPE layout it uses.
// Adding an architecture meant editing nine switches and picking the correct
// case group in every one of them; missing one silently fell through to a
// default branch instead of failing.
//
// All of it now lives in one table (llama-arch-info.cpp), which is checked
// against enum llm_arch at compile time. The predicates below are flag tests,
// so the number of code paths no longer scales with the number of models.
//

#include "llama.h"
#include "llama-arch.h"

#include <cstdint>

struct llama_hparams;

enum llama_arch_flag : uint32_t {
    LLAMA_ARCH_FLAG_RECURRENT   = 1u << 0, // RWKV/Mamba-style recurrence
    LLAMA_ARCH_FLAG_HYBRID      = 1u << 1, // attention interleaved with recurrent layers
    LLAMA_ARCH_FLAG_DIFFUSION   = 1u << 2, // diffusion decoding
    LLAMA_ARCH_FLAG_RS_ROLLBACK = 1u << 3, // supports recurrent-state rollback
    LLAMA_ARCH_FLAG_MIXED_BATCH = 1u << 4, // token and embedding inputs in one batch
    LLAMA_ARCH_FLAG_SM_TENSOR   = 1u << 5, // uses the SM tensor family
    LLAMA_ARCH_FLAG_SAVE        = 1u << 6, // model can be re-saved
};

struct llama_arch_info {
    llm_arch        arch      = LLM_ARCH_UNKNOWN;
    const char *    name      = "unknown";
    llama_rope_type rope_type = LLAMA_ROPE_TYPE_NONE; // base layout, see llama_arch_rope_type()
    uint32_t        flags     = 0;
};

// table row for `arch`; out-of-range arches map to the LLM_ARCH_UNKNOWN row
const llama_arch_info & llama_arch_info_get(llm_arch arch);

inline bool llama_arch_has_flag(llm_arch arch, llama_arch_flag flag) {
    return (llama_arch_info_get(arch).flags & (uint32_t) flag) != 0;
}

// RoPE layout for `arch`, applying the runtime overrides (M-RoPE detection and
// the DFlash drafter variants) that cannot be expressed as a static table value
llama_rope_type llama_arch_rope_type(llm_arch arch, const llama_hparams & hparams);
