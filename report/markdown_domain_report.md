# Week 1 Markdown domain

## Method

`markdown-it-py` parses five CommonMark documents into document, section, heading, list, list item, and code nodes. Each substantial peer section receives one sequence ID. A leading document title and any text outside retained sections stay in the shared prefix. Nested nodes keep parent links but share their section's sequence. Sections under 16 raw model tokens merge into the shared root; the parser also caps independent groups at 256. All five test pairs keep A and B distinct.

The shared C++ driver uses CPU-only Llama 3.2 1B Instruct Q4_K_M with four threads, greedy decoding, a 2,048-token context, and flash attention disabled. Each experiment compares the baseline answer about A with a changed B, an evicted B, reversed group order, and a repeated baseline. B is changed only within its own section. After eviction, the driver confirms that A remains present and B's live sequence state shrinks.

## Tree statistics and observed results

| Document | Max depth | Nodes | Section group sizes, tokens | Prefill tokens | Pure syntax estimate | B live KV state, bytes before to after | Answer |
|---|---:|---:|---|---:|---:|---:|---|
| Release | 4 | 9 | 60, 43 | 135 | 26.4% | 2,458,916 to 1,049,376 | `BLUE` |
| Access | 3 | 7 | 49, 45 | 126 | 24.4% | 2,524,476 to 1,049,376 | `VIEWER` |
| Incident | 3 | 8 | 53, 40 | 126 | 26.8% | 2,393,356 to 1,082,156 | `P1` |
| Build | 3 | 7 | 50, 37 | 119 | 28.0% | 2,262,236 to 1,049,376 | `whl` |
| Storage | 3 | 7 | 53, 45 | 130 | 20.9% | 2,524,476 to 1,049,376 | `HOURLY` |

All five B edits and five B evictions preserve the baseline generated token IDs and text. Reversed-order and repeat runs also match, for **20 of 20 exact comparisons**. Each eviction reduces B's measured live KV sequence state by **53.6% to 58.4%** and lowers its maximum retained position. These serialized sequence-state bytes include overhead and are not process RAM measurements.

The syntax percentage is a conservative tokenizer-piece estimate over the original Markdown, excluding added section labels and instructions. A piece counts only when it is entirely punctuation or whitespace; mixed pieces count as content. Across the five original documents, **105 of 416 source token pieces (25.2%)** meet this rule. The estimate is sensitive to tokenization and includes punctuation used in prose.

## CPU timing

The median baseline context lifetime was **1,795 ms**, including context setup, document prefill, question prefill, and generation. Median document prefill was **1,341 ms**, median question prefill **376 ms**, and median generation **108 ms**. Median one-time model load per file experiment was **1,379 ms**. Median parser and tokenizer subprocess time was **3,924 ms**. The five eviction calls took **0.020 to 0.023 ms**. These are observed times on one Windows CPU host, not a speedup claim.

## Validation and limits

All ten baseline and modified Markdown documents validate against the domain schema. Twelve parser and driver boundary tests pass across Markdown and Python source code. They include duplicate section titles, malformed section layout, parent links, tiny-section merging, unchanged A tokens, and rejection when A changes.

The first storage run produced the same answer text in every scenario but an inconsistent trailing newline token after `HOURLY`. That case uses a three-token generation limit, which captures the complete answer and yields exact comparisons. The other four cases use the runner's default 16-token limit. This is a narrow exact-match test and does not establish general output invariance for longer generations or dependent sections. Short sections merged into the root cannot be independently evicted.

Run `python scripts/run_markdown_domain.py --with-tests` to regenerate ignored local results. Pass `--output-dir results/markdown_domain` only when intentionally refreshing the reference results.
