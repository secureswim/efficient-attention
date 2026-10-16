"""Parse Markdown sections into the shared structured attention schema."""

import argparse
import json
import string
from pathlib import Path

from markdown_it import MarkdownIt

from structured_common import MAX_GROUPS, MIN_GROUP_TOKENS, call_tokenizer, validate_local

PREFIX = ('Answer using only the selected Markdown section and the shared document '
          'context. Reply with the requested value only.\n')
SYNTAX = set(string.punctuation + string.whitespace)


def build(source: str, document_id: str, driver: Path, model: Path):
    lines = source.splitlines(keepends=True)
    tokens = MarkdownIt('commonmark').parse(source)
    headings = [(index, token) for index, token in enumerate(tokens)
                if token.type == 'heading_open' and token.map]
    if not headings:
        raise ValueError('Markdown document needs section headings')
    first = headings[0][1]
    # A leading H1 is shared document context; its child headings are sections.
    section_level = min(int(token.tag[1:]) for _, token in headings
                        if token is not first or token.tag != 'h1') if len(headings) > 1 else 1
    if first.tag == 'h1' and section_level == 1:
        section_level = 2
    starts = [(index, token.map[0]) for index, token in headings
              if int(token.tag[1:]) == section_level]
    if len(starts) < 2:
        raise ValueError('Markdown document needs at least two peer sections')

    nodes = []

    def add(path, parent, node_type, text='', merged_into=None):
        item = {'id': f'n{len(nodes)}', 'path': path, 'parent': parent,
                'node_type': node_type, 'text': text, 'tokens': [],
                'positions': [], 'seq_ids': [], 'merged_into': merged_into}
        nodes.append(item)
        return item

    root = add('', None, 'document')
    sections = []
    names = set()
    for ordinal, (_, start) in enumerate(starts):
        end = starts[ordinal + 1][1] if ordinal + 1 < len(starts) else len(lines)
        title = lines[start].strip().lstrip('#').strip()
        if not title or title in names:
            raise ValueError('section headings must be nonempty and unique')
        names.add(title)
        path = f'/sections/{title.replace("~", "~0").replace("/", "~1")}'
        body = ''.join(lines[start:end])
        node = add(path, root['id'], 'section')
        sections.append((node, start, end, body, title))

    fragments = [{'text': f'\nSection: {title}\nMarkdown:\n{body}\n'}
                 for _, _, _, body, title in sections]
    raw_fragments = [{'text': body} for _, _, _, body, _ in sections]
    arrays = call_tokenizer(driver, model, fragments + raw_fragments + [{'text': source}])
    group_arrays = arrays['tokens'][:len(sections)]
    raw_arrays = arrays['tokens'][len(sections):2 * len(sections)]
    keep = [index for index, arr in enumerate(raw_arrays) if len(arr) >= MIN_GROUP_TOKENS]
    if len(keep) < 2:
        raise ValueError('fewer than two independently addressable sections after merging')
    while len(keep) > MAX_GROUPS:
        keep.remove(min(keep, key=lambda index: (len(raw_arrays[index]), index)))
    keep_set = set(keep)

    # Preserve every source line. Retained sections move to independent groups;
    # text outside them and tiny sections remain in the shared prefix.
    cursor = 0
    shared_parts = []
    for index, (_, start, end, _, _) in enumerate(sections):
        if index in keep_set:
            shared_parts.extend(lines[cursor:start])
            cursor = end
    shared_parts.extend(lines[cursor:])
    shared_source = ''.join(shared_parts)
    shared = PREFIX + f'\nDocument: {document_id}\nShared Markdown:\n{shared_source}\n'
    root_tokens = call_tokenizer(driver, model, [{'text': shared, 'add_special': True}])['tokens'][0]
    if not root_tokens:
        raise ValueError('empty shared prefix')
    root_count = len(root_tokens)
    groups = []
    for index, (section, start, end, body, _) in enumerate(sections):
        owner = section['id'] if index in keep_set else root['id']
        if index in keep_set:
            seq = len(groups)
            group_tokens = group_arrays[index]
            section.update({'text': fragments[index]['text'], 'tokens': group_tokens,
                            'positions': list(range(root_count, root_count + len(group_tokens))),
                            'seq_ids': [[seq] for _ in group_tokens]})
            groups.append({'seq_id': seq, 'path': section['path'], 'node_id': section['id'],
                           'start_pos': root_count, 'end_pos': root_count + len(group_tokens)})
        else:
            section['merged_into'] = root['id']

        # Keep the Markdown tree visible even though the section owns the tokens.
        # Block text is already present in the section fragment, so child nodes
        # have empty token arrays and point to their materialized ancestor.
        local = MarkdownIt('commonmark').parse(body)
        heading_stack = [(section_level, section)]
        list_stack = []
        block_index = 0
        for token in local:
            kind = token.type
            if kind == 'heading_open':
                level = int(token.tag[1:])
                if level <= section_level:
                    continue
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                parent = heading_stack[-1][1] if heading_stack else section
                block_index += 1
                item = add(f'{section["path"]}/blocks/{block_index}', parent['id'],
                           'heading', ''.join(body.splitlines(keepends=True)[token.map[0]:token.map[1]]), owner)
                heading_stack.append((level, item))
            elif kind in ('bullet_list_open', 'ordered_list_open'):
                block_index += 1
                parent = list_stack[-1] if list_stack else heading_stack[-1][1]
                item = add(f'{section["path"]}/blocks/{block_index}', parent['id'], 'list', '', owner)
                list_stack.append(item)
            elif kind == 'list_item_open':
                block_index += 1
                item = add(f'{section["path"]}/blocks/{block_index}', list_stack[-1]['id'],
                           'list_item', '', owner)
                list_stack.append(item)
            elif kind in ('list_item_close', 'bullet_list_close', 'ordered_list_close'):
                list_stack.pop()
            elif kind in ('fence', 'code_block'):
                block_index += 1
                parent = list_stack[-1] if list_stack else heading_stack[-1][1]
                add(f'{section["path"]}/blocks/{block_index}', parent['id'], 'code',
                    ''.join(body.splitlines(keepends=True)[token.map[0]:token.map[1]]), owner)

    root.update({'text': shared, 'tokens': root_tokens,
                 'positions': list(range(root_count)),
                 'seq_ids': [list(range(len(groups))) for _ in root_tokens]})
    model_tokens = root_count + sum(len(group_arrays[index]) for index in keep)
    if model_tokens >= 1900:
        raise ValueError(f'document uses {model_tokens} model tokens; keep under 1900 for question capacity')
    source_pieces = arrays['pieces'][-1]
    pure = sum(bool(piece) and all(char in SYNTAX for char in piece) for piece in source_pieces)
    by_id = {node['id']: node for node in nodes}

    def depth(node):
        return 0 if node['parent'] is None else 1 + depth(by_id[node['parent']])

    stats = {'node_count': len(nodes), 'max_depth': max(map(depth, nodes)),
             'group_count': len(groups),
             'group_token_sizes': [group['end_pos'] - group['start_pos'] for group in groups],
             'model_token_count': model_tokens, 'source_tokens': len(source_pieces),
             'pure_syntax_tokens': pure,
             'pure_syntax_share': round(pure / len(source_pieces), 4) if source_pieces else 0,
             'syntax_method': 'whole model token piece is punctuation or whitespace in original Markdown; excludes added prompts'}
    result = {'schema_version': 1, 'document_id': document_id, 'root_prefix': shared,
              'nodes': nodes, 'groups': groups, 'stats': stats}
    validate_local(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--driver', type=Path, required=True)
    parser.add_argument('--document-id')
    args = parser.parse_args()
    document_id = args.document_id or args.input.stem.removesuffix('_modified')
    result = build(args.input.read_text(encoding='utf-8'), document_id, args.driver, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'{document_id}: {result["stats"]["model_token_count"]} model tokens; '
          f'{result["stats"]["group_count"]} groups; {result["stats"]["node_count"]} nodes')


if __name__ == '__main__':
    main()
