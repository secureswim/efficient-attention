# Week 1 source code domain: Python modules

## Method

Tree-sitter parses five small Python modules into module, class, and function or method nodes. The parser keeps imports, class declarations, class attributes, and other code outside selected functions in a shared prefix. Each independently addressable function or method receives one sequence ID and contains its complete signature and body. Nested definitions remain tree nodes but share their enclosing function's materialized group. Fragments below 16 model tokens merge into the shared prefix; at most 256 groups are permitted.

The shared C++ driver uses CPU-only Llama 3.2 1B Instruct Q4_K_M: four threads, greedy decoding, a 2,048-token context, unified KV, and flash attention disabled. For each file, it compares a baseline answer with a changed unrelated function, eviction of that function, reversed prefill order, and a repeated baseline. A and B have distinct sequence IDs. The five questions refer only to A; none of the selected A functions calls B.

## Tree statistics and results

| Module | Max depth | Nodes | Group sizes, model tokens | Prefill tokens | Pure syntax estimate | B live KV state, bytes before to after | Answer |
|---|---:|---:|---|---:|---:|---:|---|
| Pricing | 1 | 5 | 45, 41 | 129 | 47.4% | 2,753,936 to 1,409,956 | `FREE` |
| Accounts | 2 | 7 | 46, 44 | 142 | 56.1% | 3,147,296 to 1,704,976 | `GOLD` |
| Inventory | 1 | 4 | 56, 40 | 131 | 53.8% | 2,458,916 to 1,147,716 | `OUT` |
| Alerts | 2 | 5 | 52, 42 | 137 | 50.6% | 2,786,716 to 1,409,956 | `CRITICAL` |
| Exports | 1 | 5 | 44, 38 | 129 | 48.1% | 2,786,716 to 1,541,076 | `csv` |

All five B edits and all five B evictions preserve the baseline generated token IDs and text. Reversing prefill order and repeating the baseline also match, for **20 of 20 exact comparisons**. Each eviction reduces B's measured live KV sequence state by **44.7% to 53.3%**, and its maximum retained position falls. A stays present. These serialized sequence-state bytes include overhead; they are not process RAM measurements.

The pure syntax share is a tokenizer-piece estimate over the original Python source, excluding added path labels and instructions. A piece counts when it contains only punctuation or whitespace, or a standalone Python keyword. Mixed pieces count as content. The weighted share is **202 of 394 source tokens, or 51.3%**. This definition is a conservative estimate, not a precise count of all syntactic information.

## CPU timing

The median baseline context lifetime across five files was **1,842 ms**, including context setup, document prefill, question prefill, and generation. Median document prefill was **1,312 ms**, median question prefill **389 ms**, and median generation **52 ms**. Median one-time model load per file experiment was **1,364 ms**. Median parser and tokenizer subprocess time was **3,540 ms**. The five eviction calls took **0.018 to 0.023 ms**. These are observed times on one Windows CPU host, not a speedup claim.

## Validation and limits

The five baseline and modified documents validate against the Python source schema. The shared test suite checks malformed Python, duplicate paths, class parent links, token metadata, and unauthorized changes outside B. The Markdown domain uses the same driver and has its own parser and boundary tests.

The selected test pairs contain independent functions or methods. A function that imports or calls another function may need that dependency included in its visible sequence; this first source-code experiment does not solve general dependency analysis. Short functions merged into the shared prefix cannot be independently evicted. The syntax percentage and KV bytes use the definitions above, and CPU timings should be rerun on each teammate's machine if reported for that machine.

Run `python scripts/run_source_code_domain.py --with-tests` to regenerate ignored local results. Pass `--output-dir results/source_code_domain` only when intentionally refreshing the committed reference results.
