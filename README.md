# Project 0: Efficient attention on structured documents

**Individual submission:** Siddharth Kumar (2023522)  
**Repository:** https://github.com/secureswim/efficient-attention  
**Domains:** Markdown documents and Python source code  
**Compute:** CPU only, Llama 3.2 1B Instruct Q4_K_M through llama.cpp

## What this project tests

The assignment asks whether a document tree can control which tokens an answer attends to. A normal prompt places the whole document in one sequence. Here, the shared title or module context belongs to every sequence, while each substantial Markdown section or Python function belongs to its own sequence ID. The llama.cpp attention mask uses those IDs to keep an answer about subtree A from reading unrelated subtree B. The shared C++ driver asks about A, edits B, and evicts B by position range. It checks exact generated token IDs and text and measures B's live KV sequence state before and after eviction.

The preliminary two-prompt experiment traced `llama.h`, the upstream batched example, sequence-ID mask construction, and separate-versus-batched generation. Its source trace and CPU results are in [`report/source_trace.md`](report/source_trace.md) and [`report/part1_report.md`](report/part1_report.md). Flash attention was disabled for strict comparisons after an earlier numerical mismatch with the automatic setting. The mismatch is documented in the Part 1 report.

## Implementation

| Path | Purpose |
|---|---|
| `src/structured_attention_driver.cpp` | Shared CPU tokenizer, structured prefill, question answering, position-range eviction, and exact comparisons |
| `parsers/markdown_document.py` | CommonMark section tree from `markdown-it-py` |
| `parsers/python_source.py` | Python syntax tree from Tree-sitter |
| `parsers/structured_common.py` | Shared tokenization and schema checks |
| `schema/` | JSON interchange schemas, fields, parent links, and merge rules |
| `samples/markdown/`, `samples/source_code/` | Five original and five B-modified documents per domain, plus case definitions |
| `scripts/` | CPU setup and repeatable experiment runners |
| `tests/` | Parser and driver boundary tests |
| `results/` | Reference output, tree statistics, comparisons, and CPU timing CSVs |
| `report/` | Detailed domain reports, source trace, and presentation |

The serialized schema has `tokens`, `positions`, `seq_ids`, `node_type`, and `parent` for every node. It also records paths, merge targets, group ranges, and statistics. Root tokens are shared across all groups. Materialized group tokens start after the root at the same position and have one sequence ID each. A and B must have different IDs. The driver removes B with `llama_memory_seq_rm` over B's half-open position range. The JSON files in this repository encode the required schema and results; the project domains are Markdown and Python source code.

In Markdown, a leading H1 and text outside retained sections stay in the shared root. Peer sections become candidate groups; nested headings, lists, list items, and code blocks remain tree nodes within their section. In Python, functions and methods become candidate groups while imports, class declarations, and other surrounding text stay shared. Nested definitions stay with the enclosing function. In both parsers, a candidate with fewer than 16 raw model tokens merges into the root. If needed, the smallest candidates merge until no more than 256 groups remain. A merged candidate cannot be evicted independently. The five selected A/B pairs in each domain remain separate. Every tested document has fewer than 2,000 prefill tokens.

## Results: Markdown

| Document | Tree depth | Nodes | A/B section sizes, tokens | Prefill tokens | Pure syntax share | B KV bytes before to after | Answer |
|---|---:|---:|---|---:|---:|---:|---|
| Release | 4 | 9 | 60 / 43 | 135 | 26.4% | 2,458,916 to 1,049,376 | `BLUE` |
| Access | 3 | 7 | 49 / 45 | 126 | 24.4% | 2,524,476 to 1,049,376 | `VIEWER` |
| Incident | 3 | 8 | 53 / 40 | 126 | 26.8% | 2,393,356 to 1,082,156 | `P1` |
| Build | 3 | 7 | 50 / 37 | 119 | 28.0% | 2,262,236 to 1,049,376 | `whl` |
| Storage | 3 | 7 | 53 / 45 | 130 | 20.9% | 2,524,476 to 1,049,376 | `HOURLY` |

All five changed-B and five evicted-B runs produced exactly the same generated token IDs and text as their baseline. Reverse-order and repeat checks also matched: **20/20 exact comparisons**. All five evictions reduced B's live KV sequence state by **53.6%–58.4%**, while retaining A. The weighted pure syntax estimate is **105/416 original-source token pieces (25.2%)**. Median CPU baseline context lifetime was **1,795 ms**; document prefill **1,341 ms**; question prefill **376 ms**; generation **108 ms**. Median one-time model load was **1,379 ms**, and median parsing/tokenization subprocess time was **3,924 ms**. Eviction took **0.020–0.023 ms**.

## Results: Python source code

| Module | Tree depth | Nodes | A/B function sizes, tokens | Prefill tokens | Pure syntax share | B KV bytes before to after | Answer |
|---|---:|---:|---|---:|---:|---:|---|
| Pricing | 1 | 5 | 45 / 41 | 129 | 47.4% | 2,753,936 to 1,409,956 | `FREE` |
| Accounts | 2 | 7 | 46 / 44 | 142 | 56.1% | 3,147,296 to 1,704,976 | `GOLD` |
| Inventory | 1 | 4 | 56 / 40 | 131 | 53.8% | 2,458,916 to 1,147,716 | `OUT` |
| Alerts | 2 | 5 | 52 / 42 | 137 | 50.6% | 2,786,716 to 1,409,956 | `CRITICAL` |
| Exports | 1 | 5 | 44 / 38 | 129 | 48.1% | 2,786,716 to 1,541,076 | `csv` |

The same four comparisons matched for all five modules: **20/20 exact comparisons**. All five evictions reduced B's live KV sequence state by **44.7%–53.3%**, while retaining A. The weighted pure syntax estimate is **202/394 original-source token pieces (51.3%)**. Median CPU baseline context lifetime was **1,842 ms**; document prefill **1,312 ms**; question prefill **389 ms**; generation **52 ms**. Median one-time model load was **1,364 ms**, and median parsing/tokenization subprocess time was **3,540 ms**. Eviction took **0.018–0.023 ms**.

The syntax shares are conservative tokenizer-piece estimates over original documents, without added path labels or instructions. Markdown pieces count when entirely punctuation or whitespace. Python pieces also count when an exact keyword. Mixed pieces count as content. The KV byte figures are serialized live sequence-state sizes, not process RAM. CPU times are observed on one Windows machine; they do not establish a speedup.

## Reproduce the experiments

Install Python 3.11+, Git, CMake, a C/C++ toolchain, and preferably Ninja. Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Download the Llama 3.2 1B Instruct Q4_K_M GGUF to `models/Llama-3.2-1B-Instruct-Q4_K_M.gguf` after accepting Meta's model license. This run used [Unsloth's GGUF file](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf), 807,694,368 bytes, with local SHA-256 `3F5A22426976AB26CFE84DBA63C1D08391717ABB1AF893E10F1B2968D862DCC1`. The hash is a local integrity record, not a publisher checksum. Model weights are excluded from the repository and ZIP.

From the repository root:

```powershell
python scripts/setup_cpu.py
python scripts/run_markdown_domain.py --with-tests
python scripts/run_source_code_domain.py --with-tests
python scripts/run_part1.py
```

On Linux or macOS, use `python3` for the same scripts. CPU setup pins llama.cpp commit `8bb3cf0d216a5ae48360179f2931daf284be4d01`, applies the target patch and Windows compatibility patch when needed, and builds `llama-batched`, `two-prompt`, and `structured-attention-driver` with GPU backends disabled. The original upstream `batched.cpp` remains unchanged. Both domain runners accept `--model`, `--driver`, `--threads`, and `--output-dir`. Their default output directories are ignored local folders. Reference outputs were generated with `--output-dir results/markdown_domain` and `--output-dir results/source_code_domain`. The current results were verified on Windows; Linux and macOS builds need verification on those systems.

The driver runs five fresh contexts per case: baseline, changed B, evicted B, reversed group order, and repeated baseline. It uses four CPU threads, greedy decoding, a 2,048-token context, unified KV, and disabled flash attention. The default generation limit is 16 tokens. The Markdown storage case uses three tokens because a longer run sometimes differed only in a trailing newline after the complete `HOURLY` answer. The other nine cases use 16. The 12 parser and driver boundary tests pass. Full per-case outputs and timing rows are in `results/`; detailed explanations are in `report/`.

## Scope and attribution

The initial CPU batching and shared-driver work arose during the assignment's team collaboration. This is Siddharth Kumar's individual submission of the two domain experiments and report. The selected A/B pairs are independent; general cross-section references and Python call dependencies need explicit dependency grouping. Tiny subtrees merged into the root cannot be evicted alone. Results after the submission deadline should not be treated as part of the submitted snapshot; the repository link and ZIP should identify the same final files.
