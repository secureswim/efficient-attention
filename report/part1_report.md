# Efficient Attention Part 1: CPU batching and sequence isolation

## Status

**Complete for the required two-prompt experiment.** The CPU-only model loaded, both prompts generated separately and in one batch, and the final greedy comparison matched every generated token and decoded text in normal, reversed, and repeated batches. The installed `llama-cli` built and displayed help but stalled at its model-loading spinner on this Windows/MinGW setup, so the successful upstream single-prompt baseline was run with `llama-batched -np 1`. The custom executable provides the canonical separate-versus-batched token-ID comparison. This CLI substitution is a remaining deviation from the requested baseline method.

## Environment, repository, and model

- Host: Windows 10.0.26200, x64; PowerShell 7.6.5. Compiler: MinGW-w64 GCC 13.2.0; CMake 3.29.2; Ninja 1.12.0; Git 2.49.0.
- Isolated workspace: `efficient-attention/` inside the current checkout. The unrelated parent repository was clean before work began. Its files were not modified.
- Upstream: `https://github.com/ggml-org/llama.cpp.git`, commit `8bb3cf0d216a5ae48360179f2931daf284be4d01` (`0.6.0-dev`).
- Build: Release/Ninja, `GGML_CUDA=OFF`, `GGML_VULKAN=OFF`, `GGML_METAL=OFF`, CPU backend present. `CMAKE_CXX_FLAGS=-D_WIN32_WINNT=0x0A00`. See `results/build.log` and `build/CMakeCache.txt`.
- Model: Unsloth Llama 3.2 **1B Instruct** `Q4_K_M`, `Llama-3.2-1B-Instruct-Q4_K_M.gguf`, from `https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf`. The user authorized downloading this model under its Llama 3.2 license. Size: **807,694,368 bytes**. Local SHA-256: `3F5A22426976AB26CFE84DBA63C1D08391717ABB1AF893E10F1B2968D862DCC1`. Header begins `GGUF`, version 3; model load reports `1B`, `Q4_K - Medium`. The SHA-256 is a local integrity record, not a publisher checksum.
- CPU threads: 4 for prefill and generation. Max generated tokens: 24. Greedy argmax, no stochastic sampling. Context: 512 tokens, `n_seq_max=2`, `kv_unified=false`, flash attention **disabled** in the final custom runs. `n_gpu_layers=0` and CPU-only build.

## Build and changes

The first build failed because this MinGW header set lacks `THREAD_POWER_THROTTLING_STATE`; the bundled httplib then required `CreateFile2`, and its common code required a Windows 10 target macro. The source fixes were limited to guarding the optional thread-throttling call in `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c`, using `CreateFileW` in `llama.cpp/vendor/cpp-httplib/httplib.cpp`, and adding the custom target in `llama.cpp/examples/batched/CMakeLists.txt`. The original `batched.cpp` was not changed. A nonfatal UI asset download warning appeared during configuration. The final `llama-cli`, `llama-batched`, and `two-prompt` targets all built. Both upstream executables displayed `--help` successfully.

The custom `src/two_prompt.cpp` differs intentionally from upstream `examples/batched/batched.cpp`: it tokenizes **two distinct prompts**, assigns separate sequence IDs and positions, marks each prompt's final token for logits, samples greedily from the correct batch index, tracks each sequence's position/output/end state, and compares against two fresh-context single runs. It accepts an optional reversed order and optional prompt strings. Resources are freed on success and exceptions. The prompts tokenize to 12 and 13 tokens, so unequal lengths are exercised.

## Runs and comparison

Prompt A: `Explain in three sentences how a CPU cache works.`  
Prompt B: `Explain in three sentences how an operating system schedules processes.`

The strict criterion is equality of all 24 generated token IDs **and** decoded text, not semantic similarity. The custom program returns exit 1 on a mismatch and exit 2 for setup/runtime failures. `run.ps1` invokes normal, reversed, and repeated batches; `scripts/compare.py` independently parses logs, checks outputs, and returns nonzero on failure.

| Final setting: flash disabled | A vs single | B vs single | First mismatch |
|---|---|---|---|
| Normal batch | Match, 24/24 | Match, 24/24 | None |
| Reversed batch | Match, 24/24 | Match, 24/24 | None |
| Repeated batch | Match, 24/24 | Match, 24/24 | None |

The initial `auto` flash setting gave a reproducible B mismatch at generated token index 3, while A matched. B's batched IDs were identical in normal, reverse, and repeat order. At the divergent step, separate B ranked token 5829 at 19.6876 above token 37869 at 19.5991; batched B ranked 37869 at 19.7363 above 5829 at 19.6882. These close scores changed greedy selection. With flash disabled, separate B ranked 5829 at 19.6009 and batched B ranked it at 19.547; both selected it. This is evidence of kernel/batch numerical sensitivity, not evidence of cross-sequence contamination. Logs of the `auto` run are preserved as `results/flash_auto_*`.

The upstream `llama-batched -np 1` single baselines also generated 24 tokens successfully for both prompts; their full console logs are `results/single_prompt_A.log` and `results/single_prompt_B.log`. The custom single runs provide generated token IDs and wall-clock phase timings in `results/batched_run.log`. `llama-cli` stalled before writing inference output even for a one-token `Hi` prompt, so no CLI timing is claimed.

## CPU timings

`results/timings.csv` contains all nine final runs, model load, tokenization, prefill, generation, total context lifetime, prompt/generated token counts, and thread count. Representative normal-order measurements (milliseconds):

| Run | Prompt tokens | Generated tokens | Prefill | Generation | Total context lifetime |
|---|---:|---:|---:|---:|---:|
| Single A | 12 | 24 | 490.392 | 1316.31 | 1823.79 |
| Single B | 13 | 24 | 320.165 | 1216.43 | 1552.27 |
| Batched A+B | 25 | 48 | 519.158 | 2783.06 | 3320.9 |

The model was loaded once per process (1421.72 ms in the normal-order run), then fresh contexts were used for single A, single B, and batched A+B. The repeated measurements vary substantially, so these are observations on this shared host rather than a stable throughput benchmark. Batching can use matrix operations more efficiently and process requests together, but each request can wait for the shared batch and does not necessarily have lower individual latency. CPU and sampling settings are identical within each comparison; the upstream example logs use their own context defaults and are not compared as performance equivalents.

## Source and sequence isolation

See `report/source_trace.md` for functions and line ranges. The decisive path is `include/llama.h:248-273` → `src/llama-context.cpp:1817-1996` → `src/llama-batch.cpp:161-314` → `src/llama-kv-cache.cpp:1115-1176` → `src/llama-kv-cache.cpp:1569-1715` → `src/llama-graph.cpp:2927-2962` → `src/llama-context.cpp:4218-4229`. KV cells carry positions and memberships. The mask drops cells without matching sequence membership and separately drops future positions. Requested output rows are mapped back to batch indices before greedy sampling. The optimized flash operation receives the same mask tensor at `src/llama-graph.cpp:2883-2907`; it was disabled for the exact-match final test.

## Reproduce

From `efficient-attention/` in PowerShell:

```powershell
cmake -S llama.cpp -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_FLAGS=-D_WIN32_WINNT=0x0A00 -DCMAKE_CXX_FLAGS=-D_WIN32_WINNT=0x0A00 -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_METAL=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_CURL=OFF
cmake --build build --target llama-cli llama-batched two-prompt -j 4
.\run.ps1
```

The model path is `models/Llama-3.2-1B-Instruct-Q4_K_M.gguf`; it is excluded by `.gitignore`. The source also accepts explicit prompts:

```powershell
.\build\bin\two-prompt.exe .\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf 24 4 --reverse 'Explain in three sentences how a CPU cache works.' 'Explain in three sentences how an operating system schedules processes.'
```

## Remaining limitations

- The upstream CLI's Windows loading-spinner stall remains unresolved. The upstream batched example with `-np 1` and the custom direct API program provide verified single-prompt outputs instead.
- The `auto` flash path does not meet exact greedy equivalence for B on this host. The final configuration explicitly disables flash attention; the alternate logs remain available for investigation.
- The fixed 24-token limit is reached before end-of-generation for these prompts. The code tracks EOG per sequence, but a natural early EOG case was not observed in this run.
- Timings are noisy across three repetitions. No stable batching speedup is claimed.
