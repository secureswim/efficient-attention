"""Parse one Python module into independently addressable function subtrees."""

import argparse
import json
import keyword
import string
from pathlib import Path

from structured_common import MAX_GROUPS, MIN_GROUP_TOKENS, call_tokenizer, validate_local

PREFIX = ('Answer using only the selected Python function or method and the shared '
          'module context. Reply with the requested value only.\n')
SYNTAX = set(string.punctuation + string.whitespace)


def escaped(name):
    return name.replace('~', '~0').replace('/', '~1')


def named_definition(node):
    if node.type == 'decorated_definition':
        return next((child for child in node.named_children
                     if child.type in ('class_definition', 'function_definition')), None)
    return node if node.type in ('class_definition', 'function_definition') else None


def source_text(source, node):
    return source[node.start_byte:node.end_byte].decode('utf-8')


def build(source: bytes, document_id: str, driver: Path, model: Path):
    try:
        import tree_sitter_python
        from tree_sitter import Language, Parser
    except ImportError as exc:
        raise RuntimeError('install tree-sitter and tree-sitter-python first') from exc
    source.decode('utf-8')
    tree = Parser(Language(tree_sitter_python.language())).parse(source)
    if tree.root_node.has_error:
        raise ValueError('Python source has a Tree-sitter syntax error')

    nodes = []
    candidates = []

    def add(path, parent, node_type, merged_into=None):
        if any(item['path'] == path for item in nodes):
            raise ValueError(f'duplicate definition path: {path}')
        item = {'id': f'n{len(nodes)}', 'path': path, 'parent': parent,
                'node_type': node_type, 'text': '', 'tokens': [],
                'positions': [], 'seq_ids': [], 'merged_into': merged_into}
        nodes.append(item)
        return item

    root = add('', None, 'module')

    def nested_definitions(node, parent, owner):
        for child in node.named_children:
            definition = named_definition(child)
            if definition and definition.type == 'function_definition':
                name = source_text(source, definition.child_by_field_name('name'))
                path = parent['path'] + '/functions/' + escaped(name)
                inner = add(path, parent['id'], 'function', owner['id'])
                nested_definitions(definition.child_by_field_name('body'), inner, owner)
            else:
                nested_definitions(child, parent, owner)

    for ordinal, child in enumerate(tree.root_node.named_children):
        definition = named_definition(child)
        if definition and definition.type == 'function_definition':
            name = source_text(source, definition.child_by_field_name('name'))
            item = add('/functions/' + escaped(name), root['id'], 'function')
            candidates.append((item, child.start_byte, child.end_byte))
            nested_definitions(definition.child_by_field_name('body'), item, item)
        elif definition and definition.type == 'class_definition':
            name = source_text(source, definition.child_by_field_name('name'))
            cls = add('/classes/' + escaped(name), root['id'], 'class', root['id'])
            body = definition.child_by_field_name('body')
            for index, member in enumerate(body.named_children):
                method = named_definition(member)
                if method and method.type == 'function_definition':
                    method_name = source_text(source, method.child_by_field_name('name'))
                    item = add(cls['path'] + '/methods/' + escaped(method_name), cls['id'], 'method')
                    candidates.append((item, member.start_byte, member.end_byte))
                    nested_definitions(method.child_by_field_name('body'), item, item)
                else:
                    add(cls['path'] + f'/context/{index}', cls['id'], 'statement', root['id'])
        else:
            kind = 'import' if child.type in ('import_statement', 'import_from_statement') else 'statement'
            add(f'/context/{ordinal}', root['id'], kind, root['id'])

    candidates.sort(key=lambda entry: entry[1])
    for left, right in zip(candidates, candidates[1:]):
        if left[2] > right[1]:
            raise ValueError('overlapping function spans')
    if len(candidates) < 2:
        raise ValueError('module needs at least two functions or methods')

    fragments = [{'text': f'\nPath: {item["path"]}\nPython:\n{source[start:end].decode("utf-8")}\n'}
                 for item, start, end in candidates]
    raw_fragments = [{'text': source[start:end].decode('utf-8')} for _, start, end in candidates]
    original = source.decode('utf-8')
    tokenized = call_tokenizer(driver, model, fragments + raw_fragments + [{'text': original}])
    arrays = tokenized['tokens']
    raw_arrays = arrays[len(candidates):2 * len(candidates)]
    keep = [index for index in range(len(candidates)) if len(raw_arrays[index]) >= MIN_GROUP_TOKENS]
    if len(keep) < 2:
        raise ValueError('fewer than two independently addressable functions after merging')
    while len(keep) > MAX_GROUPS:
        keep.remove(min(keep, key=lambda index: (len(raw_arrays[index]), index)))
    keep_set = set(keep)
    cursor = 0
    shared_parts = []
    for index, (_, start, end) in enumerate(candidates):
        if index in keep_set:
            shared_parts.append(source[cursor:start])
            cursor = end
    shared_parts.append(source[cursor:])
    shared_source = b''.join(shared_parts).decode('utf-8')
    shared = PREFIX + f'\nModule: {document_id}\nShared source:\n{shared_source}\n'
    root_tokens = call_tokenizer(driver, model, [{'text': shared, 'add_special': True}])['tokens'][0]
    if not root_tokens:
        raise ValueError('empty shared prefix')
    root_count = len(root_tokens)
    groups = []
    for index, (item, _, _) in enumerate(candidates):
        if index in keep_set:
            seq = len(groups)
            tokens = arrays[index]
            item.update({'text': fragments[index]['text'], 'tokens': tokens,
                         'positions': list(range(root_count, root_count + len(tokens))),
                         'seq_ids': [[seq] for _ in tokens], 'merged_into': None})
            groups.append({'seq_id': seq, 'path': item['path'], 'node_id': item['id'],
                           'start_pos': root_count, 'end_pos': root_count + len(tokens)})
        else:
            item['merged_into'] = root['id']
            for descendant in nodes:
                if descendant['merged_into'] == item['id']:
                    descendant['merged_into'] = root['id']
    root.update({'text': shared, 'tokens': root_tokens,
                 'positions': list(range(root_count)),
                 'seq_ids': [list(range(len(groups))) for _ in root_tokens]})
    model_tokens = root_count + sum(len(arrays[index]) for index in keep)
    if model_tokens >= 2000:
        raise ValueError(f'document has {model_tokens} model tokens; limit is under 2000')
    if model_tokens >= 1900:
        raise ValueError('document leaves insufficient context capacity for the question')
    source_pieces = tokenized['pieces'][-1]
    pure = sum(bool(piece) and (all(char in SYNTAX for char in piece)
                  or piece.strip() in keyword.kwlist) for piece in source_pieces)
    by_id = {item['id']: item for item in nodes}

    def depth(item):
        return 0 if item['parent'] is None else 1 + depth(by_id[item['parent']])

    stats = {'node_count': len(nodes), 'max_depth': max(map(depth, nodes)),
             'group_count': len(groups),
             'group_token_sizes': [group['end_pos'] - group['start_pos'] for group in groups],
             'model_token_count': model_tokens, 'source_tokens': len(source_pieces),
             'pure_syntax_tokens': pure,
             'pure_syntax_share': round(pure / len(source_pieces), 4) if source_pieces else 0,
             'syntax_method': 'whole model token piece is Python punctuation/whitespace or a standalone keyword; original source only'}
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
    result = build(args.input.read_bytes(), document_id, args.driver, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'{document_id}: {result["stats"]["model_token_count"]} model tokens; '
          f'{result["stats"]["group_count"]} groups; {result["stats"]["node_count"]} nodes')


if __name__ == '__main__':
    main()
