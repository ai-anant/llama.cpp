#include "llama-arch-info.h"

#include "llama-arch.h"
#include "llama-hparams.h"

#include "ggml.h"

#include <cstddef>
#include <cstdint>

//
// llama_arch_info_table - the single place where per-architecture facts live.
//
// Adding a new architecture means adding one row here, in enum order. Every
// predicate below is then a flag test, and llama_arch_rope_type() derives the
// RoPE layout from the same row. The static_asserts at the bottom turn a
// missing or misordered row into a compile error rather than a silent default.
//
// F_* abbreviations exist only to keep the table narrow; they are #undef'ed
// after the table on purpose.
//
#define F_RECURRENT   LLAMA_ARCH_FLAG_RECURRENT
#define F_HYBRID      LLAMA_ARCH_FLAG_HYBRID
#define F_DIFFUSION   LLAMA_ARCH_FLAG_DIFFUSION
#define F_RS_ROLLBACK LLAMA_ARCH_FLAG_RS_ROLLBACK
#define F_MIXED_BATCH LLAMA_ARCH_FLAG_MIXED_BATCH
#define F_SM_TENSOR   LLAMA_ARCH_FLAG_SM_TENSOR
#define F_SAVE        LLAMA_ARCH_FLAG_SAVE

static constexpr llama_arch_info LLAMA_ARCH_INFO[] = {
    { LLM_ARCH_CLIP,              "clip",             LLAMA_ROPE_TYPE_NONE,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_LLAMA,             "llama",            LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_LLAMA4,            "llama4",           LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_DECI,              "deci",             LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_FALCON,            "falcon",           LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_BAICHUAN,          "baichuan",         LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_GROK,              "grok",             LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_GPT2,              "gpt2",             LLAMA_ROPE_TYPE_NONE,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_GPTJ,              "gptj",             LLAMA_ROPE_TYPE_NONE,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_GPTNEOX,           "gptneox",          LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_MPT,               "mpt",              LLAMA_ROPE_TYPE_NONE,    F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_STARCODER,         "starcoder",        LLAMA_ROPE_TYPE_NONE,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_REFACT,            "refact",           LLAMA_ROPE_TYPE_NONE,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_BERT,              "bert",             LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_MODERN_BERT,       "modern-bert",      LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_NOMIC_BERT,        "nomic-bert",       LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_NOMIC_BERT_MOE,    "nomic-bert-moe",   LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_NEO_BERT,          "neo-bert",         LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_JINA_BERT_V2,      "jina-bert-v2",     LLAMA_ROPE_TYPE_NONE,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_JINA_BERT_V3,      "jina-bert-v3",     LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_EUROBERT,          "eurobert",         LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_BLOOM,             "bloom",            LLAMA_ROPE_TYPE_NONE,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_STABLELM,          "stablelm",         LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_QWEN,              "qwen",             LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_QWEN2,             "qwen2",            LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_QWEN2MOE,          "qwen2moe",         LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_QWEN2VL,           "qwen2vl",          LLAMA_ROPE_TYPE_MROPE,   F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_SENSENOVA_U1,      "sensenova_u1",     LLAMA_ROPE_TYPE_MROPE,   F_MIXED_BATCH | F_SM_TENSOR },
    { LLM_ARCH_QWEN3,             "qwen3",            LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_QWEN3MOE,          "qwen3moe",         LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_FLINT,             "flint",            LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_QWEN3NEXT,         "qwen3next",        LLAMA_ROPE_TYPE_NEOX,    F_HYBRID | F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_QWEN3VL,           "qwen3vl",          LLAMA_ROPE_TYPE_IMROPE,  F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_QWEN3VLMOE,        "qwen3vlmoe",       LLAMA_ROPE_TYPE_IMROPE,  F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_QWEN35,            "qwen35",           LLAMA_ROPE_TYPE_IMROPE,  F_HYBRID | F_RS_ROLLBACK | F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_QWEN35MOE,         "qwen35moe",        LLAMA_ROPE_TYPE_IMROPE,  F_HYBRID | F_RS_ROLLBACK | F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_CLEF,              "clef",             LLAMA_ROPE_TYPE_IMROPE,  F_MIXED_BATCH | F_SM_TENSOR },
    { LLM_ARCH_VLLM_DECISION2,    "vllm-decision2",   LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR },
    { LLM_ARCH_CLOVER_MOE,        "clover_moe",       LLAMA_ROPE_TYPE_NEOX,    F_HYBRID | F_RS_ROLLBACK | F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_QWEN4EXP,          "qwen4exp",         LLAMA_ROPE_TYPE_IMROPE,  F_HYBRID | F_RS_ROLLBACK | F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_PHI2,              "phi2",             LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_PHI3,              "phi3",             LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_PHIMOE,            "phimoe",           LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_PLAMO,             "plamo",            LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_PLAMO2,            "plamo2",           LLAMA_ROPE_TYPE_NEOX,    F_HYBRID | F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_PLAMO3,            "plamo3",           LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_CODESHELL,         "codeshell",        LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_ORION,             "orion",            LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_INTERNLM2,         "internlm2",        LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_MINICPM,           "minicpm",          LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_MINICPM3,          "minicpm3",         LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_GEMMA,             "gemma",            LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_GEMMA2,            "gemma2",           LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_GEMMA3,            "gemma3",           LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_GEMMA3N,           "gemma3n",          LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH },
    { LLM_ARCH_GEMMA4,            "gemma4",           LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_GEMMA4_ASSISTANT,  "gemma4-assistant", LLAMA_ROPE_TYPE_NEOX,    F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_GEMMA_EMBEDDING,   "gemma-embedding",  LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_GEMMA_EMBEDDING2,  "gemma-embedding2", LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_STARCODER2,        "starcoder2",       LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_MAMBA,             "mamba",            LLAMA_ROPE_TYPE_NONE,    F_RECURRENT | F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_MAMBA2,            "mamba2",           LLAMA_ROPE_TYPE_NONE,    F_RECURRENT | F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_MAPLE,             "maple",            LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_JAMBA,             "jamba",            LLAMA_ROPE_TYPE_NONE,    F_HYBRID | F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_FALCON_H1,         "falcon-h1",        LLAMA_ROPE_TYPE_NEOX,    F_HYBRID | F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_XVERSE,            "xverse",           LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_COMMAND_R,         "command-r",        LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_COHERE2,           "cohere2",          LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_COHERE2MOE,        "cohere2moe",       LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_DBRX,              "dbrx",             LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_OLMO,              "olmo",             LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_OLMO2,             "olmo2",            LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_OLMOE,             "olmoe",            LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_MUSE_GLIMMER,      "muse-glimmer",     LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_MUSE_SPARK,        "muse-spark",       LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_OPENELM,           "openelm",          LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_ARCTIC,            "arctic",           LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_DEEPSEEK,          "deepseek",         LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_DEEPSEEK2,         "deepseek2",        LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_DEEPSEEK2OCR,      "deepseek2-ocr",    LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_DEEPSEEK32,        "deepseek32",       LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_DEEPSEEK4,         "deepseek4",        LLAMA_ROPE_TYPE_NORM,    F_HYBRID | F_RS_ROLLBACK | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_DEEPSEEK41,        "deepseek41",       LLAMA_ROPE_TYPE_NORM,    F_HYBRID | F_RS_ROLLBACK | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_XING4_0,           "xing4_0",          LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_CHATGLM,           "chatglm",          LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_GLM4,              "glm4",             LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_GLM4_MOE,          "glm4moe",          LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_GLM_DSA,           "glm-dsa",          LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_BITNET,            "bitnet",           LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH },
    { LLM_ARCH_T5,                "t5",               LLAMA_ROPE_TYPE_NONE,    F_MIXED_BATCH },
    { LLM_ARCH_T5ENCODER,         "t5encoder",        LLAMA_ROPE_TYPE_NONE,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_JAIS,              "jais",             LLAMA_ROPE_TYPE_NONE,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_JAIS2,             "jais2",            LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_NEMOTRON,          "nemotron",         LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_NEMOTRON_H,        "nemotron_h",       LLAMA_ROPE_TYPE_NONE,    F_HYBRID | F_RS_ROLLBACK | F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_NEMOTRON_H_MOE,    "nemotron_h_moe",   LLAMA_ROPE_TYPE_NONE,    F_HYBRID | F_RS_ROLLBACK | F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_METIS16,           "metis16",          LLAMA_ROPE_TYPE_NONE,    F_HYBRID | F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_EXAONE,            "exaone",           LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_EXAONE4,           "exaone4",          LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_EXAONE_MOE,        "exaone-moe",       LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_RWKV6,             "rwkv6",            LLAMA_ROPE_TYPE_NONE,    F_RECURRENT | F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_RWKV6QWEN2,        "rwkv6qwen2",       LLAMA_ROPE_TYPE_NONE,    F_RECURRENT | F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_RWKV7,             "rwkv7",            LLAMA_ROPE_TYPE_NONE,    F_RECURRENT | F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_ARWKV7,            "arwkv7",           LLAMA_ROPE_TYPE_NONE,    F_RECURRENT | F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_GRANITE,           "granite",          LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_GRANITE_MOE,       "granitemoe",       LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_GRANITE_HYBRID,    "granitehybrid",    LLAMA_ROPE_TYPE_NORM,    F_HYBRID | F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_GRANITE_SWITCH,    "graniteswitch",    LLAMA_ROPE_TYPE_NORM,    F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_GRANITE_SWA,       "granite_swa",      LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_CHAMELEON,         "chameleon",        LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_WAVTOKENIZER_DEC,  "wavtokenizer-dec", LLAMA_ROPE_TYPE_NONE,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_PLM,               "plm",              LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_BAILINGMOE,        "bailingmoe",       LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_BAILINGMOE2,       "bailingmoe2",      LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_BAILINGMOE3,       "bailingmoe3",      LLAMA_ROPE_TYPE_NORM,    F_HYBRID | F_RS_ROLLBACK | F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_DOTS1,             "dots1",            LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_DOTS3NOTE,         "dots3note",        LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_ARCEE,             "arcee",            LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_AFMOE,             "afmoe",            LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_LAGUNA,            "laguna",           LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_ERNIE4_5,          "ernie4_5",         LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_ERNIE4_5_MOE,      "ernie4_5-moe",     LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_HUNYUAN_MOE,       "hunyuan-moe",      LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_HUNYUAN_DENSE,     "hunyuan-dense",    LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_HUNYUAN_VL,        "hunyuan_vl",       LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_HY_V3,             "hy_v3",            LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_HY_V4,             "hy_v4",            LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_SMOLLM3,           "smollm3",          LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_OPENAI_MOE,        "gpt-oss",          LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_LFM2,              "lfm2",             LLAMA_ROPE_TYPE_NEOX,    F_HYBRID | F_RS_ROLLBACK | F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_LFM2MOE,           "lfm2moe",          LLAMA_ROPE_TYPE_NEOX,    F_HYBRID | F_RS_ROLLBACK | F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_LFM3,              "lfm3",             LLAMA_ROPE_TYPE_NEOX,    F_HYBRID | F_RS_ROLLBACK | F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_ZAYA1,             "zaya1",            LLAMA_ROPE_TYPE_NEOX,    F_HYBRID | F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_DREAM,             "dream",            LLAMA_ROPE_TYPE_NEOX,    F_DIFFUSION | F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_SMALLTHINKER,      "smallthinker",     LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_LLADA,             "llada",            LLAMA_ROPE_TYPE_NORM,    F_DIFFUSION | F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_LLADA_MOE,         "llada-moe",        LLAMA_ROPE_TYPE_NEOX,    F_DIFFUSION | F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_SEED_OSS,          "seed_oss",         LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_GROVEMOE,          "grovemoe",         LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_APERTUS,           "apertus",          LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR },
    { LLM_ARCH_MINIMAX_M2,        "minimax-m2",       LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_COGVLM,            "cogvlm",           LLAMA_ROPE_TYPE_NEOX,    F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_RND1,              "rnd1",             LLAMA_ROPE_TYPE_NEOX,    F_DIFFUSION | F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_PANGU_EMBED,       "pangu-embedded",   LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_MISTRAL3,          "mistral3",         LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_MISTRAL4,          "mistral4",         LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_MISTRAL_LARGE4,    "mistral-large4",   LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_PADDLEOCR,         "paddleocr",        LLAMA_ROPE_TYPE_MROPE,   F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_MIMO2,             "mimo2",            LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_STEP35,            "step35",           LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR },
    { LLM_ARCH_SPARK2_5,          "spark2_5",         LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_LLAMA_EMBED,       "llama-embed",      LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_MAINCODER,         "maincoder",        LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_MAION_CODER,       "maion-coder",      LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_KIMI_LINEAR,       "kimi-linear",      LLAMA_ROPE_TYPE_NONE,    F_HYBRID | F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_KIMI_K3,           "kimi-k3",          LLAMA_ROPE_TYPE_NONE,    F_HYBRID | F_RS_ROLLBACK | F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_KIMI_K31,          "kimi-k31",         LLAMA_ROPE_TYPE_NONE,    F_HYBRID | F_RS_ROLLBACK | F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_GLM5_NEXT,         "glm5-next",        LLAMA_ROPE_TYPE_NONE,    F_HYBRID | F_RS_ROLLBACK | F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_TALKIE,            "talkie",           LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_MELLUM,            "mellum",           LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_EAGLE3,            "eagle3",           LLAMA_ROPE_TYPE_NORM,    F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_MINIMAX_M3,        "minimax-m3",       LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_DFLASH,            "dflash",           LLAMA_ROPE_TYPE_NEOX,    F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_NANBEIGE,          "nanbeige",         LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_QWEN3TTS,          "qwen3tts",         LLAMA_ROPE_TYPE_IMROPE,  F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_POCKETTTS,         "pockettts",        LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_ZONOS3,            "zonos3",           LLAMA_ROPE_TYPE_NORM,    F_HYBRID | F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_KOKORO,            "kokoro",           LLAMA_ROPE_TYPE_NONE,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_TINY_VOCA,         "tiny-voca",        LLAMA_ROPE_TYPE_NORM,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_MINIMAX_01,        "minimax-01",       LLAMA_ROPE_TYPE_NEOX,    F_HYBRID | F_MIXED_BATCH | F_SAVE },
    { LLM_ARCH_HRM_TEXT,          "hrm_text",         LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_K2_HORIZON,        "k2-horizon",       LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_BEAM,              "beam",             LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_KOLIBRI1,          "kolibri1",         LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_LIMITE,            "limite",           LLAMA_ROPE_TYPE_NORM,    F_SAVE },
    { LLM_ARCH_RICURSIVE,         "ricursive",        LLAMA_ROPE_TYPE_NEOX,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
    { LLM_ARCH_INKLING2,          "inkling2",         LLAMA_ROPE_TYPE_NONE,    F_HYBRID | F_RS_ROLLBACK | F_SAVE },
    { LLM_ARCH_UNKNOWN,           "(unknown)",        LLAMA_ROPE_TYPE_NONE,    F_MIXED_BATCH | F_SM_TENSOR | F_SAVE },
};

#undef F_RECURRENT
#undef F_HYBRID
#undef F_DIFFUSION
#undef F_RS_ROLLBACK
#undef F_MIXED_BATCH
#undef F_SM_TENSOR
#undef F_SAVE

static constexpr size_t LLAMA_ARCH_INFO_COUNT = sizeof(LLAMA_ARCH_INFO) / sizeof(LLAMA_ARCH_INFO[0]);

// LLM_ARCH_UNKNOWN is last in the enum, so it doubles as the arch count.
static_assert(LLAMA_ARCH_INFO_COUNT == LLM_ARCH_UNKNOWN + 1,
              "llama_arch_info table is out of sync with enum llm_arch: add the missing row");

constexpr bool llama_arch_info_is_ordered() {
    for (size_t i = 0; i < LLAMA_ARCH_INFO_COUNT; ++i) {
        if ((size_t) LLAMA_ARCH_INFO[i].arch != i) {
            return false;
        }
    }
    return true;
}

static_assert(llama_arch_info_is_ordered(),
              "llama_arch_info table must list architectures in enum order");

const llama_arch_info & llama_arch_info_get(llm_arch arch) {
    const size_t i = (size_t) arch;
    if (i >= LLAMA_ARCH_INFO_COUNT) {
        return LLAMA_ARCH_INFO[LLM_ARCH_UNKNOWN];
    }
    return LLAMA_ARCH_INFO[i];
}

llama_rope_type llama_arch_rope_type(llm_arch arch, const llama_hparams & hparams) {
    switch (arch) {
        // per-arch runtime overrides; everything else is the table value
        case LLM_ARCH_DFLASH:
            // M-RoPE targets carry rope sections; otherwise DSV4/ML4 drafters are
            // normal RoPE and legacy DFlash backbones are NeoX
            if (hparams.rope_sections[0] || hparams.rope_sections[1] ||
                hparams.rope_sections[2] || hparams.rope_sections[3]) {
                return LLAMA_ROPE_TYPE_MROPE;
            }
            return hparams.dsv4_hc_mult > 0 || hparams.n_lora_kv > 0 ? LLAMA_ROPE_TYPE_NORM
                                                                     : LLAMA_ROPE_TYPE_NEOX;
        case LLM_ARCH_BAILINGMOE3:
            // VL files carry mrope sections; text-only files keep NORM rope
            return hparams.use_mrope() ? LLAMA_ROPE_TYPE_MROPE : LLAMA_ROPE_TYPE_NORM;
        case LLM_ARCH_GLM4:
            return hparams.use_mrope() ? LLAMA_ROPE_TYPE_MROPE : LLAMA_ROPE_TYPE_NORM;
        case LLM_ARCH_GLM4_MOE:
        case LLM_ARCH_HUNYUAN_VL:
            return hparams.use_mrope() ? LLAMA_ROPE_TYPE_MROPE : LLAMA_ROPE_TYPE_NEOX;
        default:
            return llama_arch_info_get(arch).rope_type;
    }
}

