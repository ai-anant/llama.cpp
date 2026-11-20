# Simplification notes for this branch

This branch (`simplification-19-nov-2026`) works through the two structural
problems that make llama.cpp hard to change:

1. **The same fact repeated in many places.** A property of an architecture was
   encoded in up to nine separate switches; adding a model meant finding all of
   them and choosing a case group correctly, and a miss fell through to a silent
   default. Same for pre-tokenizer settings, which were a 320-line if/else chain.
2. **Long functions with deep nesting.** A single function that enumerates every
   category, format or tensor kind and applies a rule inline.

## Patterns used

- **One table, checked at compile time.** Per-architecture facts live in
  `llama_arch_info` (`src/llama-arch-info.{h,cpp}`), one row per `llm_arch`,
  indexed by the enum. `static_assert`s make a missing or misordered row a build
  error instead of a silent default. Adding an arch = add one row.
- **Tables over branch chains, with the runtime exceptions called out.** The
  pre-tokenizer selection is a 100-row table plus a lookup that fills a small
  params struct (`llama_vocab_pre_lookup`). Where a value genuinely depends on
  runtime state (M-RoPE detection, DFlash drafters), a short explicit `switch`
  applies the override on top of the table value.
- **One rule per function, verbatim bodies.** `llama_tensor_get_type_impl` was
  split into per-category helpers (`quant_ffn_down_type`, `quant_attn_v_type`,
  `quant_attn_output_type`, `quant_ffn_gate_up_type`) that the dispatcher calls.
  Bodies were moved without edits, so the only risk is transcription, which the
  byte-identical output check below covers.

## How these changes are verified

Nothing here is checked in on "it looks equivalent":

- `tests/test-arch-info` — table invariants, name round-trip, flagged arch sets.
- Differential harnesses (run locally, not committed) compile the *previous*
  implementation next to the new one and compare every input: 178 arches ×
  7 predicates and 2,124 rope/hparams combinations; 100 pre-tokenizer names plus
  `""` and an unknown name from two starting states (204 comparisons).
- `llama-quantize` output is compared byte for byte (SHA256) against the
  pre-refactor binary for two ftypes: `Q4_K_M` and `IQ3_XS`. Both binaries are
  deterministic run to run, so the hashes are meaningful.
- `llama-bench` with `llama-bench -m <model> -p 512 -n 128 -t 1` before and after,
  against an upstream worktree build, to prove no throughput regression. Always
  run the A/B **both ways round**: on a busy machine the second binary of each
  pair looks ~1% slower purely from drift, which is enough to fake a regression
  in the hottest kernel in the project.

## Performance

Two changes here are actually faster, and both are the same bug: the default
thread count ignored the CPU budget the process was given.

- `common_cpu_get_thread_limit()` = min(affinity mask, cgroup CPU quota). Defaults
  use it, so a container limited to one core no longer starts 4-8 threads that the
  scheduler has to timeshare. Measured on this box (Qwen2.5-0.5B Q4_K_M, defaults
  on both sides): pp128 12.98 -> 75.14 t/s, tg64 3.93 -> 25.29 t/s, i.e. 5.8x / 6.4x.
- `llama-quantize` had the same fallback through `hardware_concurrency()`:
  24.00 s -> 4.37 s for a 400 MB requantize, byte-identical output.

Everything else on this branch is structure, not speed. Measured on the same
model at 1 thread: the per-token graph build is 0.17 ms against 87 ms of compute
(0.2%), and 25 t/s corresponds to roughly 25 GFLOPS, which is about 80% of the
single-core AVX2 peak of the CPU in question - the kernels are not leaving much
on the table either. Config defaults are already optimal here (`-fa auto` beats
`-fa off` by 5-6%; `-ub` 64/128/256/512 and `-b` 256/512 are all within noise or
worse). Further gains need kernel work with a profiler on real hardware.

## The ratchet

```
scripts/check-complexity.py            # fail if a tracked unit grew
scripts/check-complexity.py --update   # accept the current tree
scripts/check-complexity.py --top 20   # worst offenders
```

`scripts/complexity-budget.json` snapshots decision points per file and per
function (files ≥ 300, functions ≥ 40). It only ratchets down. Do not run
`--update` to make a failing PR pass; run it after a change that genuinely
lowers the numbers, in the same commit.

## What is left

The remaining hot spots are code paths that cannot be exercised on a CPU-only
machine, so they need a GPU / multimodal setup before they are safe to touch:

| unit | decisions | why it is not done yet |
| --- | ---: | --- |
| `ggml-et::ggml_backend_et_device_supports_op` | 585 | backend not buildable here |
| `ggml-opencl::ggml_cl_mul_mat` | 584 | needs OpenCL device |
| `ggml-vulkan::ggml_vk_load_shaders` | 564 | needs Vulkan device |
| `mtmd::encode` (`tools/mtmd/mtmd-subcontext.cpp`) | 527 | needs a vision model |
| `ggml-vulkan::ggml_backend_vk_device_supports_op` | 448 | needs Vulkan device |
| `tools/server/server-context.cpp::init_routes` | 116 | route boilerplate; split by endpoint group |

The `*_supports_op` family is the same shape in every backend (a per-op decision
matrix). The eventual fix is one shared structure describing op support that
each backend fills in, instead of one hand-written chain per backend — worth
doing with a GPU runner available.
