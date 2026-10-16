# Structured document schemas v1

The Markdown and Python source domains use the same C++ driver contract: `schema_version`, `document_id`, `root_prefix`, `nodes`, `groups`, and `stats`. Each node carries `tokens`, `positions`, `seq_ids`, `node_type`, `parent`, and `merged_into`. The root has path `""` and null parent. Its tokens belong to every sequence. Each materialized group starts immediately after the shared root at the same position and owns one sequence ID. The driver asks about group A and evicts group B by B's half-open position interval using `llama_memory_seq_rm`.

Both domain schemas are JSON Schema files because the assignment specifies JSON as the interchange format. This repository does not include the JSON/API-response *domain*.

## Markdown documents

`markdown_document.schema.json` permits `document`, `section`, `heading`, `list`, `list_item`, and `code` nodes. `markdown-it-py` parses CommonMark. A leading H1 and text outside retained sections stay in the shared root. Peer sections at the first level below the H1 become independent candidates. Each retained section's whole Markdown text, including nested headings, lists, and code blocks, is tokenized as one group. Nested nodes preserve the tree shape, have empty token arrays, and set `merged_into` to their materialized section. Their text is already present in that section's token stream.

Sections with fewer than 16 raw model tokens merge into the shared root. If more than 256 groups remain, the smallest sections merge into the root by token count and source order until the limit is met. Such sections cannot be evicted independently. Two test targets must remain in distinct groups. Section titles must be unique because paths use escaped heading text.

The parser preserves every line of the original file either in the shared root or in one retained section. It rejects documents that use 1,900 or more prefill tokens, reserving space in the 2,048-token context for the question and generation. `stats.pure_syntax_share` is a conservative tokenizer-piece estimate over the original Markdown. A piece counts only when every character is punctuation or whitespace. Mixed pieces count as content. Added labels and instructions are excluded.

## Python source code

`source_code_document.schema.json` permits `module`, `class`, `function`, `method`, `import`, and `statement` nodes. Tree-sitter parses UTF-8 Python modules. A complete function or method, including signature, becomes an independent group. Imports, class declarations, class attributes, comments, and all source outside retained function spans form the shared prefix. Nested definitions are tree nodes merged into their enclosing function's group.

Functions with fewer than 16 raw model tokens merge into the root. The 256-group limit and 1,900-token context reserve apply here too. `stats.pure_syntax_share` counts original-source token pieces made entirely of Python punctuation or whitespace, plus standalone Python keywords. Both shares are estimates with the denominator recorded as `source_tokens`.
