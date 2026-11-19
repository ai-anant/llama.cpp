// Tests the per-architecture table (llama-arch-info) for internal consistency.
//
// The table replaced nine switches over llm_arch, so the invariants it must
// uphold are: one correctly ordered row per enum value, unique GGUF names that
// round-trip through llm_arch_from_string(), and a usable rope layout for every
// architecture. These are cheap checks, no model required.

#include "../src/llama-arch-info.h"
#include "../src/llama-arch.h"
#include "../src/llama-hparams.h"

#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, ...)                             \
    do {                                             \
        if (!(cond)) {                               \
            printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            printf(__VA_ARGS__);                     \
            printf("\n");                            \
            ++g_failures;                            \
        }                                            \
    } while (0)

int main() {
    const int n_arch = (int) LLM_ARCH_UNKNOWN + 1;

    std::set<std::string> names;
    for (int i = 0; i < n_arch; ++i) {
        const llm_arch arch = (llm_arch) i;
        const llama_arch_info & info = llama_arch_info_get(arch);

        CHECK(info.arch == arch, "row %d describes arch %d", i, (int) info.arch);
        CHECK(info.name != nullptr && info.name[0] != '\0', "arch %d has no name", i);
        CHECK(names.insert(info.name).second, "duplicate arch name '%s'", info.name);

        // the GGUF architecture string must map back to the same enum value
        CHECK(llm_arch_from_string(info.name) == arch,
              "llm_arch_from_string(\"%s\") != %d", info.name, i);

        // every arch must resolve to one of the known rope layouts; NONE is valid
        switch (info.rope_type) {
            case LLAMA_ROPE_TYPE_NONE:
            case LLAMA_ROPE_TYPE_NORM:
            case LLAMA_ROPE_TYPE_NEOX:
            case LLAMA_ROPE_TYPE_MROPE:
            case LLAMA_ROPE_TYPE_IMROPE:
            case LLAMA_ROPE_TYPE_VISION:
                break;
            default:
                CHECK(false, "arch %s has unknown rope type %d", info.name, (int) info.rope_type);
        }
    }

    // out-of-range values fall back to the UNKNOWN row instead of reading past the table
    const llama_arch_info & fallback = llama_arch_info_get((llm_arch) n_arch);
    CHECK(fallback.arch == LLM_ARCH_UNKNOWN, "out-of-range arch did not fall back to UNKNOWN");

    CHECK(strcmp(llm_arch_name(LLM_ARCH_UNKNOWN), "(unknown)") == 0,
          "unexpected name for LLM_ARCH_UNKNOWN: %s", llm_arch_name(LLM_ARCH_UNKNOWN));
    CHECK(strcmp(llm_arch_name((llm_arch) n_arch), "unknown") == 0,
          "unexpected name for an invalid arch: %s", llm_arch_name((llm_arch) n_arch));

    // llm_arch_all() must cover every enum value exactly once, in enum order
    const std::vector<llm_arch> all = llm_arch_all();
    CHECK((int) all.size() == n_arch, "llm_arch_all() returned %zu of %d arches", all.size(), n_arch);
    for (size_t i = 0; i < all.size(); ++i) {
        CHECK((size_t) all[i] == i, "llm_arch_all()[%zu] = %d, expected %zu", i, (int) all[i], i);
    }

    // runtime rope overrides: the M-RoPE archs must actually follow use_mrope(),
    // and the static archs must not be disturbed by unrelated hparams fields
    llama_hparams hp;
    hp.rope_sections = {0, 0, 0, 0};
    CHECK(llama_arch_rope_type(LLM_ARCH_LLAMA, hp) == LLAMA_ROPE_TYPE_NORM,
          "llama should use NORM rope");
    CHECK(llama_arch_rope_type(LLM_ARCH_GLM4, hp) == LLAMA_ROPE_TYPE_NORM,
          "glm4 without mrope sections should use NORM rope");
    hp.rope_sections = {3, 8, 12, 16};
    CHECK(llama_arch_rope_type(LLM_ARCH_GLM4, hp) == LLAMA_ROPE_TYPE_MROPE,
          "glm4 with mrope sections should use MROPE");
    CHECK(llama_arch_rope_type(LLM_ARCH_GLM4_MOE, hp) == LLAMA_ROPE_TYPE_MROPE,
          "glm4-moe with mrope sections should use MROPE");

    // the q-gate flag replaced three inline copies of a six-arch list
    static const llm_arch q_gate_archs[] = {
        LLM_ARCH_QWEN3NEXT, LLM_ARCH_QWEN35, LLM_ARCH_QWEN35MOE,
        LLM_ARCH_CLEF, LLM_ARCH_CLOVER_MOE, LLM_ARCH_QWEN4EXP,
    };
    for (const llm_arch arch : q_gate_archs) {
        CHECK(llama_arch_has_flag(arch, LLAMA_ARCH_FLAG_ATTN_Q_GATE),
              "arch %s should carry the attn_q gate flag", llama_arch_info_get(arch).name);
    }
    int n_q_gate = 0;
    for (int i = 0; i < n_arch; ++i) {
        if (llama_arch_has_flag((llm_arch) i, LLAMA_ARCH_FLAG_ATTN_Q_GATE)) {
            ++n_q_gate;
        }
    }
    CHECK(n_q_gate == 6, "expected 6 q-gate arches, table has %d", n_q_gate);

    if (g_failures == 0) {
        printf("test-arch-info: %d architectures OK\n", n_arch);
        return 0;
    }

    printf("test-arch-info: %d failure(s)\n", g_failures);
    return 1;
}
